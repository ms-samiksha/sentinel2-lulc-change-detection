#!/usr/bin/env python
"""Self-test for the demo's logic. Exit code 0 = all PASS, 1 = any FAIL. Model-dependent tests are SKIPPED if files are missing."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
import analysis as A   # noqa: E402
import preprocessing as P   # noqa: E402

RESULTS = []


def test(name):
    def deco(fn):
        try:
            out = fn()
            RESULTS.append((name, "SKIP" if out == "skip" else "PASS", ""))
        except Exception as exc:
            RESULTS.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
        return fn
    return deco


def synth(seed=0):
    r = np.random.default_rng(seed)
    x = np.zeros((128, 128, 9), np.float32)
    x[..., :6] = r.uniform(0.05, 0.5, (128, 128, 6))
    for k, (_, i, j) in P.INDEX_DEFINITIONS.items():
        x[..., k] = (x[..., i] - x[..., j]) / (x[..., i] + x[..., j])
    return x


@test("validate_input accepts both layouts and rejects bad shape, NaN, dtype, object")
def _():
    x = synth()
    assert P.validate_input(x).ok and P.validate_input(np.transpose(x, (2, 0, 1))).ok
    assert not P.validate_input(np.zeros((3, 64, 64), np.float32)).ok
    bad = x.copy(); bad[0, 0, 0] = np.nan
    assert not P.validate_input(bad).ok
    assert not P.validate_input(np.array([["a"]], dtype=object)).ok
    assert not P.validate_input(x.astype(np.complex64)).ok


@test("normalisation: matches formula, clipped, and applying it twice changes the data")
def _():
    x = synth(); mean = x.reshape(-1, 9).mean(0); std = x.reshape(-1, 9).std(0)
    z = P.normalize(x, mean, std, 10.0)
    assert np.allclose(z, np.clip((x - mean) / std, -10, 10), atol=1e-6) and np.abs(z).max() <= 10
    assert not np.allclose(P.normalize(z, mean, std, 10.0), z)
    assert A.looks_standardised(z) and not A.looks_standardised(x)


@test("class percentages use valid pixels and sum to 100")
def _():
    r = np.random.default_rng(1)
    pt = r.dirichlet(np.ones(16), (128, 128)).astype(np.float32); pc = r.dirichlet(np.ones(2), (128, 128)).astype(np.float32)
    gt = np.zeros((128, 128), np.uint8); gt[:32] = 255
    valid, _ = A.analysis_mask(gt)
    S = A.prediction_summary(pt, pc, valid, 1)
    assert S["n_valid"] == int(valid.sum()) == 128 * 96
    assert abs(S["table"]["Percent"].sum() - 100) < 1e-9 and int(S["table"]["Pixels"].sum()) == S["n_valid"]
    assert abs(S["binary_changed_pct"] - 100 * (pc.argmax(-1) == 1)[valid].mean()) < 1e-9
    assert abs(S["shares"].iloc[:, 2].sum() - 100) < 1e-9 and abs(S["shares"].iloc[:, 3].sum() - 100) < 1e-9
    v_all, _ = A.analysis_mask(None)
    assert A.prediction_summary(pt, pc, v_all, 1)["n_valid"] == 128 * 128


@test("index report equals a manual computation on RAW channels; provenance detects corruption")
def _():
    x20, x25 = synth(1), synth(2) + 0
    valid = np.ones((128, 128), bool)
    df = A.index_report(x20, x25, valid, None)
    for name, info in A.INDEX_INFO.items():
        k = info["ch"]
        row = df[df.Index == name].iloc[0]
        assert abs(row["Mean earlier"] - x20[..., k].mean()) < 1e-6 and abs(row["Δ mean"] - (x25[..., k].mean() - x20[..., k].mean())) < 1e-6
    assert A.index_provenance(x20, x25)["ok"]
    broken = x25.copy(); broken[..., 6] += 0.3
    assert not A.index_provenance(x20, broken)["ok"]


@test("exploratory threshold shares are sane")
def _():
    x20, x25 = synth(1), synth(2); valid = np.ones((128, 128), bool)
    t = A.threshold_shares(x20, x25, valid, {"NDVI": -2.0, "NDWI": 2.0, "NDBI": 0.0})
    assert t.iloc[0]["Share earlier (%)"] == 100.0 and t.iloc[1]["Share later (%)"] == 0.0


@test("model, hash, bundled samples reproduce the notebook, XAI smoke test")
def _():
    sd = APP_DIR / "sample_data"
    import model_utils as M
    import xai_utils as X
    cands = M.find_model_candidates(APP_DIR)
    if not cands or not (sd / "norm_stats.npz").is_file() or not (sd / "samples_index.json").is_file():
        return "skip"
    model = M.load_bt_ssscnet(cands[0])
    hs = M.hash_status(cands[0], sd)
    assert hs["status"] == "match", f"hash status {hs['status']}"
    mean, std = P.load_norm(sd / "norm_stats.npz")
    df = M.check_all_samples(model, sd, mean, std)
    assert (df["result"] == "PASS").all(), df.to_string()
    entry = M.read_json(sd / "samples_index.json")[0][0] if False else M.read_json(sd / "samples_index.json")[0][0]
    d = sd / entry["id"]
    x20, x25 = P.to_hwc(np.load(d / "x2020.npy")), P.to_hwc(np.load(d / "x2025.npy"))
    z20, z25 = P.normalize(x20, mean, std), P.normalize(x25, mean, std)
    pt, pc = M.predict(model, z20, z25)
    roi = pc.argmax(-1) == 1
    if roi.sum() < 50:
        roi = np.ones_like(roi)
    maps, _ = X.spatial_attention(model, z20, z25); assert len(maps) >= 1
    w20, w25 = X.spectral_se(model, z20, z25); assert w20.shape == (9,)
    cams, _ = X.gradcam_all(model, z20, z25, "change_output", 1, roi); assert len(cams) >= 1
    r = X.integrated_gradients(model, z20, z25, "change_output", 1, roi, steps=8)
    assert np.isfinite(r["completeness_rel_err"])
    print(f"  IG completeness relative error at 8 steps: {r['completeness_rel_err']:.3f}")
    dd = X.date_dependency(model, z20, z25, 1); assert len(dd) == 4


if __name__ == "__main__":
    print()
    for n, s, m in RESULTS:
        print(f"[{s}] {n}" + (f"  -> {m}" if m else ""))
    sys.exit(0 if all(s != "FAIL" for _, s, _ in RESULTS) else 1)