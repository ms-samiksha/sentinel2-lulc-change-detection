"""Live explainability for a loaded BT-SSSCNet (adapted from notebook Cells 36-38 and 46).

All methods describe what THIS model used. They are not causal explanations. Missing layers raise
XAIUnavailable (the UI shows a warning); nothing is approximated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from model_utils import output_tensors, predict
from preprocessing import PS

CAM_LAYERS = {"bottleneck 16x16": "tfrm4_residual", "decoder 64x64": "dec2_blk_b_relu", "decoder 128x128": "dec1_blk_b_relu"}
SA_LAYERS = [("128x128", "tfrm1_sa_conv"), ("64x64", "tfrm2_sa_conv"), ("32x32", "tfrm3_sa_conv"), ("16x16", "tfrm4_sa_conv")]
ENCODER_NAME, SE_LAYER_NAME = "shared_siamese_encoder", "spectral_band_se_fc2"
DEL_FRACS = (0.05, 0.10, 0.20, 0.40)

_SUBMODELS: dict = {}


class XAIUnavailable(RuntimeError):
    """A layer or capability needed by an XAI method is not available in the loaded model."""


def find_layer(container, name):
    try:
        return container.get_layer(name)
    except (ValueError, AttributeError):
        return None


def layer_report(model) -> dict:
    enc = find_layer(model, ENCODER_NAME)
    rep = {n: find_layer(model, n) is not None for _, n in SA_LAYERS}
    rep.update({n: find_layer(model, n) is not None for n in CAM_LAYERS.values()})
    rep[ENCODER_NAME] = enc is not None
    rep[SE_LAYER_NAME] = enc is not None and find_layer(enc, SE_LAYER_NAME) is not None
    return rep


def _head(model, name):
    outs = output_tensors(model)
    if name not in outs:
        raise XAIUnavailable(f"model has no output '{name}' (found {list(outs)})")
    return outs[name]


def _pick(o, head, model):
    if isinstance(o, dict):
        return o[head]
    return dict(zip(list(model.output_names), o))[head]


def _sub(model, key, builder):
    k = (id(model),) + key
    if k not in _SUBMODELS:
        _SUBMODELS[k] = builder()
    return _SUBMODELS[k]


# ------------------------------------------------------------------ attention / SE
def spatial_attention(model, z20, z25):
    """TFRM spatial-attention gates, bilinearly resized to 128x128. Returns (maps dict, missing layer names)."""
    present = [(r, n) for r, n in SA_LAYERS if find_layer(model, n) is not None]
    missing = [n for _, n in SA_LAYERS if find_layer(model, n) is None]
    if not present:
        avail = [l.name for l in model.layers if "tfrm" in l.name][:12]
        raise XAIUnavailable(f"None of {[n for _, n in SA_LAYERS]} exist. TFRM-related layers found: {avail or 'none'}.")
    sub = _sub(model, ("sa",), lambda: keras.Model(model.inputs, [model.get_layer(n).output for _, n in present]))
    outs = sub([z20[None], z25[None]], training=False)
    if not isinstance(outs, (list, tuple)):
        outs = [outs]
    maps = {r: tf.image.resize(o, (PS, PS), method="bilinear").numpy()[0, ..., 0] for (r, _), o in zip(present, outs)}
    return maps, missing


def spectral_se(model, z20, z25):
    """Sigmoid band weights of the spectral SE block (shared encoder), for each date. Returns (w20, w25), each (9,)."""
    enc = find_layer(model, ENCODER_NAME)
    if enc is None:
        raise XAIUnavailable(f"layer '{ENCODER_NAME}' not found; top-level layers: {[l.name for l in model.layers][:10]}")
    se = find_layer(enc, SE_LAYER_NAME)
    if se is None:
        raise XAIUnavailable(f"layer '{SE_LAYER_NAME}' not found inside the encoder.")
    sub = _sub(model, ("se",), lambda: keras.Model(enc.inputs, se.output))
    w = [np.asarray(sub(z[None], training=False)).reshape(-1) for z in (z20, z25)]
    return w[0], w[1]


# ------------------------------------------------------------------ Grad-CAM
def gradcam(model, z20, z25, layer, head, cls, roi):
    """Seg-Grad-CAM: target = summed class probability over `roi`; weights = spatially averaged gradients."""
    lyr = find_layer(model, layer)
    if lyr is None:
        raise XAIUnavailable(f"layer '{layer}' not found")
    sub = _sub(model, ("cam", layer, head), lambda: keras.Model(model.inputs, [lyr.output, _head(model, head)]))
    roi_t = tf.constant(roi[None].astype(np.float32))
    with tf.GradientTape() as tape:
        feat, out = sub([z20[None], z25[None]], training=False)
        score = tf.reduce_sum(out[..., cls] * roi_t)
    g = tape.gradient(score, feat)
    if g is None:
        raise XAIUnavailable(f"no gradient reached layer '{layer}'")
    cam = tf.nn.relu(tf.reduce_sum(tf.reduce_mean(g, axis=(1, 2), keepdims=True) * feat, axis=-1, keepdims=True))
    cam = tf.image.resize(cam, (PS, PS), method="bilinear")[0, ..., 0].numpy()
    return cam / (cam.max() + 1e-9)


def gradcam_all(model, z20, z25, head, cls, roi):
    cams, missing = {}, []
    for name, layer in CAM_LAYERS.items():
        if find_layer(model, layer) is None:
            missing.append(layer)
            continue
        cams[name] = gradcam(model, z20, z25, layer, head, cls, roi)
    if not cams:
        raise XAIUnavailable(f"none of the Grad-CAM layers {list(CAM_LAYERS.values())} exist in this model")
    return cams, missing


# ------------------------------------------------------------------ Integrated Gradients
def _F(model, a, b, head, cls, roi):
    o = _pick(model([a, b], training=False), head, model)
    return tf.reduce_sum(o[..., cls] * roi, axis=[1, 2])


def integrated_gradients(model, z20, z25, head, cls, roi, steps=24, chunk=8, progress=None):
    """Midpoint-rule IG. Baseline = dataset-mean image (z = 0) on both dates.
    Target F = sum over `roi` of P(class). Completeness: sum(attr) ~ F(x) - F(baseline)."""
    t20, t25 = tf.constant(z20[None]), tf.constant(z25[None])
    rt = tf.constant(roi[None].astype(np.float32))
    alphas = (np.arange(steps, dtype=np.float32) + 0.5) / steps
    g20, g25 = np.zeros(z20.shape, np.float64), np.zeros(z25.shape, np.float64)
    for s in range(0, steps, chunk):
        al = tf.constant(alphas[s:s + chunk])[:, None, None, None]
        x20, x25 = al * t20, al * t25
        with tf.GradientTape() as tape:
            tape.watch([x20, x25])
            f = tf.reduce_sum(_F(model, x20, x25, head, cls, rt))
        d20, d25 = tape.gradient(f, [x20, x25])
        if d20 is None or d25 is None:
            raise XAIUnavailable("no gradient with respect to the inputs")
        g20 += d20.numpy().sum(0)
        g25 += d25.numpy().sum(0)
        if progress:
            progress(min(1.0, (s + chunk) / steps))
    f1 = float(_F(model, t20, t25, head, cls, rt)[0])
    f0 = float(_F(model, tf.zeros_like(t20), tf.zeros_like(t25), head, cls, rt)[0])
    ig20, ig25 = (z20 * g20 / steps).astype(np.float32), (z25 * g25 / steps).astype(np.float32)
    a20, a25 = np.abs(ig20).sum((0, 1)), np.abs(ig25).sum((0, 1))
    tot = float(a20.sum() + a25.sum()) + 1e-12
    return dict(ig20=ig20, ig25=ig25, F_x=f1, F_baseline=f0, sum_attribution=float(ig20.sum() + ig25.sum()),
                completeness_rel_err=abs(float(ig20.sum() + ig25.sum()) - (f1 - f0)) / max(abs(f1 - f0), 1e-6),
                band_share=np.concatenate([a20, a25]) / tot, score=np.abs(ig20).sum(-1) + np.abs(ig25).sum(-1), steps=steps)


def deletion_test(model, z20, z25, score, head, cls, roi, fracs=DEL_FRACS, n_rand=3, seed=42):
    """Faithfulness diagnostic: output ratio after replacing the top-IG pixels (both dates) by the mean (z=0),
    versus replacing the same number of random pixels."""
    H = PS * PS
    order = np.argsort(-score.reshape(-1))
    rng = np.random.default_rng(seed)
    rt = tf.constant(roi[None].astype(np.float32))
    Fn = lambda a, b: float(_F(model, a[None], b[None], head, cls, rt)[0])
    f0 = Fn(z20, z25)

    def delete(sel):
        a, b = z20.copy(), z25.copy()
        mk = sel.reshape(PS, PS)
        a[mk] = 0
        b[mk] = 0
        return Fn(a, b) / max(f0, 1e-9)

    rows = []
    for fr in fracs:
        k = int(round(fr * H))
        s = np.zeros(H, bool)
        s[order[:k]] = True
        rr = []
        for _ in range(n_rand):
            s2 = np.zeros(H, bool)
            s2[rng.choice(H, k, replace=False)] = True
            rr.append(delete(s2))
        rows.append(dict(deleted_fraction=fr, output_ratio_IG_ranked=delete(s), output_ratio_random=float(np.mean(rr))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ date dependency
def date_dependency(model, z20, z25, change_id=1):
    """Counterfactual inputs from notebook Cell 36."""
    scen = {"baseline (earlier, later)": (z20, z25), "identical dates (earlier, earlier)": (z20, z20),
            "identical dates (later, later)": (z25, z25), "swapped dates (later, earlier)": (z25, z20)}
    rows, base = [], None
    for name, (a, b) in scen.items():
        _, pc = predict(model, a, b)
        pred = pc.argmax(-1) == change_id
        base = pred if base is None else base
        rows.append({"Scenario": name, "Predicted change (%)": 100 * float(pred.mean()),
                     "Mean P(change)": float(pc[..., change_id].mean()), "Binary agreement with baseline": float((pred == base).mean())})
    return pd.DataFrame(rows)