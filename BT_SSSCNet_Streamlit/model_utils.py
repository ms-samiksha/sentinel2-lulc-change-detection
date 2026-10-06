"""Model loading, inference, hashing, class mapping and sample-level metrics for BT-SSSCNet.

Custom layers are copied verbatim from the training notebook (Cell 9 / exported inference module).
This module does not import Streamlit.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf                      # noqa: E402
from tensorflow import keras                 # noqa: E402
from tensorflow.keras import layers          # noqa: E402

import preprocessing as P                    # noqa: E402

NT, NC, NL, IGN = 16, 2, 4, 255
MODEL_FILENAME = "bt_ssscnet_best_val_loss.keras"
# Hash quoted in the project brief. The authoritative record is sample_data/model_hash.json
PROJECT_REPORTED_SHA256 = "82DC887B34D5117CD98E60C44CD7AE0AFBBDE63C6241E7CDED92E0651DDA217A"
REQUIRED_SAMPLE_FILES = ("x2020.npy", "x2025.npy", "gt_transition.npy", "gt_change.npy", "frozen_predictions.npz")


@keras.utils.register_keras_serializable(package="BTSSSCNet")
class AbsDiff(layers.Layer):
    def call(self, inputs):
        a, b = inputs
        return tf.abs(a - b)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


@keras.utils.register_keras_serializable(package="BTSSSCNet")
class ChannelAvgMax(layers.Layer):
    def call(self, x):
        return tf.concat([tf.reduce_mean(x, axis=-1, keepdims=True), tf.reduce_max(x, axis=-1, keepdims=True)], axis=-1)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (2,)


CUSTOM_OBJECTS = {"AbsDiff": AbsDiff, "ChannelAvgMax": ChannelAvgMax}


class ModelLoadError(RuntimeError):
    """Raised with a user-readable message when the model cannot be loaded or validated."""


# ------------------------------------------------------------------ files / hashing
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    """Returns (data, error_message). Never raises."""
    p = Path(path)
    if not p.is_file():
        return None, f"{p.name} not found"
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as exc:
        return None, f"{p.name} could not be read: {exc}"


def find_model_candidates(app_dir) -> list:
    cands = []
    env = os.environ.get("BT_MODEL_PATH")
    if env:
        cands.append(Path(env))
    mdir = Path(app_dir) / "model"
    cands.append(mdir / MODEL_FILENAME)
    if mdir.is_dir():
        cands += sorted(mdir.glob("*.keras"))
    out, seen = [], set()
    for p in cands:
        if p.is_file() and p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out


def hash_status(model_path, sample_dir, sha: str | None = None) -> dict:
    """Compare the model file hash with sample_data/model_hash.json (authoritative) and with the brief's hash."""
    sha = (sha or sha256_file(model_path)).lower()
    rec, err = read_json(Path(sample_dir) / "model_hash.json")
    recorded = str(rec.get("sha256", "")).lower() if isinstance(rec, dict) else ""
    status = ("match" if recorded == sha else "mismatch") if recorded else "unrecorded"
    return dict(sha256=sha, recorded=recorded or None, recorded_file=(rec or {}).get("file") if isinstance(rec, dict) else None,
                status=status, record_error=err, reported=PROJECT_REPORTED_SHA256.lower(),
                reported_match=(sha == PROJECT_REPORTED_SHA256.lower()))


# ------------------------------------------------------------------ model
def output_tensors(model) -> dict:
    """Map output name -> symbolic tensor, whatever container Keras uses."""
    out = model.output
    if isinstance(out, dict):
        return dict(out)
    names = list(model.output_names)
    if isinstance(out, (list, tuple)):
        return dict(zip(names, out))
    return {names[0]: out}


def validate_model(model) -> list:
    problems = []
    shapes = [tuple(i.shape) for i in model.inputs]
    if shapes != [(None, P.PS, P.PS, P.C)] * 2:
        problems.append(f"inputs are {shapes}, expected two (None,128,128,9)")
    outs = output_tensors(model)
    for name, k in (("transition_output", NT), ("change_output", NC)):
        if name not in outs:
            problems.append(f"missing output '{name}' (found {list(outs)})")
        elif tuple(outs[name].shape)[1:] != (P.PS, P.PS, k):
            problems.append(f"{name} has shape {tuple(outs[name].shape)}, expected (None,128,128,{k})")
    return problems


def load_bt_ssscnet(path):
    """Same call as the notebook: compile=False plus the custom layers. Validates inputs/outputs."""
    p = Path(path)
    if not p.is_file():
        raise ModelLoadError(f"Model file not found: {p}")
    try:
        model = keras.models.load_model(str(p), custom_objects=CUSTOM_OBJECTS, compile=False)
    except Exception as exc:
        raise ModelLoadError(
            f"Could not load {p.name}: {type(exc).__name__}: {exc}\n"
            f"Installed TensorFlow {tf.__version__} / Keras {keras.__version__}. A .keras file written by another "
            f"Keras version may not load; install the TensorFlow version in sample_data/preprocessing_config.json ('tf_version').") from exc
    problems = validate_model(model)
    if problems:
        raise ModelLoadError("Loaded model has an unexpected structure: " + "; ".join(problems))
    return model


def _split_outputs(o, model=None):
    if isinstance(o, dict):
        return o["transition_output"], o["change_output"]
    names = list(model.output_names)
    d = dict(zip(names, o))
    return d["transition_output"], d["change_output"]


def predict(model, z20: np.ndarray, z25: np.ndarray):
    """z20, z25: NORMALISED (128,128,9). Returns (transition_probs (128,128,16), change_probs (128,128,2))."""
    o = model([z20[None].astype(np.float32), z25[None].astype(np.float32)], training=False)
    t, c = _split_outputs(o, model)
    return np.asarray(t)[0].astype(np.float32), np.asarray(c)[0].astype(np.float32)


def compare_with_frozen(pt, pc, frozen: dict, atol: float = 1e-3, min_agree: float = 0.999) -> dict:
    for k in ("transition_probs", "change_probs"):
        if k not in frozen:
            raise ValueError(f"frozen predictions are missing '{k}'")
    ft, fc = frozen["transition_probs"], frozen["change_probs"]
    if ft.shape != pt.shape or fc.shape != pc.shape:
        raise ValueError(f"frozen shapes {ft.shape}/{fc.shape} differ from predictions {pt.shape}/{pc.shape}")
    dp = max(float(np.abs(pt - ft).max()), float(np.abs(pc - fc).max()))
    at = float((pt.argmax(-1) == ft.argmax(-1)).mean())
    ac = float((pc.argmax(-1) == fc.argmax(-1)).mean())
    return dict(max_abs_prob_diff=dp, transition_argmax_agreement=at, change_argmax_agreement=ac,
                passed=bool(dp < atol and at >= min_agree and ac >= min_agree))


def check_all_samples(model, sample_dir, mean, std, clip=P.DEFAULT_CLIP) -> pd.DataFrame:
    """Run every bundled sample through the exact app pipeline and compare with the notebook's frozen predictions."""
    sample_dir = Path(sample_dir)
    idx, err = read_json(sample_dir / "samples_index.json")
    if err or not isinstance(idx, list) or not idx:
        raise FileNotFoundError(err or "samples_index.json is empty")
    rows = []
    for e in idx:
        sid = str(e.get("id", "?"))
        row = {"sample": sid, "group": e.get("group", "")}
        try:
            d = sample_dir / sid
            a20, a25 = np.load(d / "x2020.npy", allow_pickle=False), np.load(d / "x2025.npy", allow_pickle=False)
            with np.load(d / "frozen_predictions.npz") as z:
                frozen = {k: z[k] for k in z.files}
            v20, v25 = P.validate_input(a20, "earlier"), P.validate_input(a25, "later")
            if not (v20.ok and v25.ok):
                raise ValueError("; ".join(v20.errors + v25.errors))
            pt, pc = predict(model, P.normalize(v20.x_hwc, mean, std, clip), P.normalize(v25.x_hwc, mean, std, clip))
            cmp_ = compare_with_frozen(pt, pc, frozen)
            row.update(max_abs_dp=cmp_["max_abs_prob_diff"], transition_argmax_agreement=cmp_["transition_argmax_agreement"],
                       change_argmax_agreement=cmp_["change_argmax_agreement"], result="PASS" if cmp_["passed"] else "FAIL")
        except Exception as exc:
            row.update(max_abs_dp=np.nan, transition_argmax_agreement=np.nan, change_argmax_agreement=np.nan,
                       result=f"ERROR: {type(exc).__name__}: {exc}")
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ class mapping
def load_lulc_names(sample_dir):
    """Optional sample_data/lulc_names.json -> ({0: name, ...}, warning|None). Never guesses names."""
    p = Path(sample_dir) / "lulc_names.json"
    if not p.is_file():
        return {}, None
    data, err = read_json(p)
    if err:
        return {}, f"lulc_names.json ignored: {err}"
    if not isinstance(data, dict):
        return {}, "lulc_names.json ignored: expected an object like {\"0\": \"name\", ...}"
    names = {}
    for k, v in data.items():
        try:
            ik = int(k)
        except (TypeError, ValueError):
            return {}, f"lulc_names.json ignored: key {k!r} is not an integer"
        if not 0 <= ik < NL or not isinstance(v, str) or not v.strip():
            return {}, f"lulc_names.json ignored: invalid entry {k!r}: {v!r} (keys must be 0..{NL - 1}, values non-empty strings)"
        names[ik] = v.strip()
    return names, None


def lulc_name(k: int, names: dict | None = None) -> str:
    return (names or {}).get(int(k), f"LULC {int(k)}")


def transition_label(k: int, names: dict | None = None) -> str:
    f, t = divmod(int(k), NL)                    # id = 4*from + to
    return f"T{int(k)}: {lulc_name(f, names)} → {lulc_name(t, names)}"


def transition_table(names: dict | None = None) -> pd.DataFrame:
    rows = []
    for k in range(NT):
        f, t = divmod(k, NL)
        rows.append({"ID": f"T{k}", "From": lulc_name(f, names), "To": lulc_name(t, names), "Type": "Stay" if f == t else "Change"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ summaries / metrics
def summarize_prediction(pt, pc, names=None, change_id: int = 1) -> dict:
    pred, conf = pt.argmax(-1), pt.max(-1)
    n = pred.size
    binary_change = pc.argmax(-1) == change_id
    implied = (pred // NL) != (pred % NL)
    rows = []
    for k in range(NT):
        m = pred == k
        c = int(m.sum())
        rows.append({"ID": k, "Transition": transition_label(k, names), "Type": "Stay" if k // NL == k % NL else "Change",
                     "Pixels": c, "Percent": 100.0 * c / n,
                     "Mean confidence": float(conf[m].mean()) if c else float("nan"), "Mean P(class)": float(pt[..., k].mean())})
    df = pd.DataFrame(rows)
    chg = df[(df.Type == "Change") & (df.Pixels > 0)]
    dom_change = chg.sort_values("Pixels", ascending=False).iloc[0].to_dict() if len(chg) else None
    dom_all = df.sort_values("Pixels", ascending=False).iloc[0].to_dict()
    return dict(table=df, n_pixels=n, changed_pixels=int(binary_change.sum()), unchanged_pixels=int(n - binary_change.sum()),
                binary_changed_pct=100.0 * float(binary_change.mean()), mean_p_change=float(pc[..., change_id].mean()),
                mean_confidence=float(conf.mean()), implied_changed_pct=100.0 * float(implied.mean()),
                disagreement_pct=100.0 * float((binary_change != implied).mean()), dominant_change=dom_change, dominant_overall=dom_all)


def confusion_matrix(y, p, k: int) -> np.ndarray:
    v = y != IGN
    return np.bincount(y[v].astype(np.int64) * k + p[v].astype(np.int64), minlength=k * k).reshape(k, k)


def _prf(cm):
    tp, gt, pr = np.diag(cm).astype(float), cm.sum(1).astype(float), cm.sum(0).astype(float)
    iou = np.divide(tp, gt + pr - tp, out=np.full(len(tp), np.nan), where=(gt + pr - tp) > 0)
    f1 = np.divide(2 * tp, gt + pr, out=np.full(len(tp), np.nan), where=(gt + pr) > 0)
    return iou, f1, gt


def sample_metrics(gt_t, gt_c, pred_t, pred_c, change_id: int = 1) -> dict:
    """Sample-level metrics on valid (label != 255) pixels. Macro averages use classes present in the ground truth."""
    cmt, cmb = confusion_matrix(gt_t, pred_t, NT), confusion_matrix(gt_c, pred_c, NC)
    iou_t, f1_t, gt_n = _prf(cmt)
    iou_b, f1_b, _ = _prf(cmb)
    pres = gt_n > 0
    return dict(n_valid=int(cmt.sum()), transition_acc=float(np.trace(cmt) / max(cmt.sum(), 1)),
                transition_mIoU=float(np.nanmean(iou_t[pres])) if pres.any() else float("nan"),
                transition_macro_f1=float(np.nanmean(f1_t[pres])) if pres.any() else float("nan"),
                n_classes_present=int(pres.sum()), binary_acc=float(np.trace(cmb) / max(cmb.sum(), 1)),
                binary_iou=float(iou_b[change_id]), binary_f1=float(f1_b[change_id]), cm_transition=cmt, cm_binary=cmb)