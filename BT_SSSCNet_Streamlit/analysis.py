"""Analysis helpers for the BT-SSSCNet demo: prediction statistics, spectral-index analysis,
plain-language report text and XAI explanation text. All numbers come from the actual arrays passed in."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

import model_utils as M
import preprocessing as P

NT, NL, IGN = 16, 4, 255
SATURATION = 0.999

INDEX_INFO = {
    "NDVI": dict(ch=6, formula="(B8 − B4) / (B8 + B4)", topic="vegetation-related", cmap="RdYlGn",
                 higher="a stronger vegetation-related signal", lower="a weaker vegetation-related signal"),
    "NDWI": dict(ch=7, formula="(B3 − B8) / (B3 + B8)", topic="water-related", cmap="BrBG",
                 higher="a stronger open-water / surface-moisture-related signal", lower="a weaker water-related signal"),
    "NDBI": dict(ch=8, formula="(B11 − B8) / (B11 + B8)", topic="built-up-related", cmap="RdBu_r",
                 higher="a stronger built-up / bare-surface-related signal", lower="a weaker built-up-related signal"),
}


# ------------------------------------------------------------------ keys / masks / guards
def input_key(x20, x25, model_sha, mean, std, clip) -> str:
    """Identity of one analysis: the exact raw arrays + model file + normalisation."""
    h = hashlib.sha256()
    for a in (x20, x25, mean, std):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(str(model_sha).encode())
    h.update(repr(clip).encode())
    return h.hexdigest()


def analysis_mask(gt_c, shape=(P.PS, P.PS)):
    """Pixels used as the denominator for every percentage."""
    if gt_c is None:
        return np.ones(shape, bool), "all pixels (no reference labels supplied)"
    g = np.asarray(gt_c)
    if g.shape != tuple(shape):
        raise ValueError(f"reference label shape {g.shape} != {shape}")
    return g != IGN, "pixels that carry a reference label (label 255 = ignore is excluded)"


def looks_standardised(x_hwc) -> bool:
    """Raw reflectance bands have a small spread (~0.03-0.1). A spread > 0.4 in all six bands suggests the file
    was already standardised, which would be normalised a second time."""
    return bool(np.all(x_hwc[..., :6].reshape(-1, 6).std(0) > 0.4))


# ------------------------------------------------------------------ prediction statistics
def prediction_summary(pt, pc, valid, change_id=1, names=None) -> dict:
    n = int(valid.sum())
    if n == 0:
        raise ValueError("no valid pixels to analyse")
    pred, conf = pt.argmax(-1), pt.max(-1)
    bin_chg = pc.argmax(-1) == change_id
    implied = (pred // NL) != (pred % NL)
    pv, cv = pred[valid], conf[valid]
    rows = []
    for k in range(NT):
        m = pv == k
        c = int(m.sum())
        f, t = divmod(k, NL)
        rows.append({"ID": k, "Transition": M.transition_label(k, names), "From": M.lulc_name(f, names), "To": M.lulc_name(t, names),
                     "Type": "Stay" if f == t else "Change", "Pixels": c, "Percent": 100.0 * c / n,
                     "Mean confidence": float(cv[m].mean()) if c else float("nan"),
                     "Mean P(class)": float(pt[..., k][valid].mean())})
    df = pd.DataFrame(rows)
    assert int(df["Pixels"].sum()) == n, "transition pixel counts must sum to the number of valid pixels"
    chg = df[(df.Type == "Change") & (df.Pixels > 0)].sort_values("Pixels", ascending=False)
    fc, tc = np.bincount(pv // NL, minlength=NL), np.bincount(pv % NL, minlength=NL)
    shares = pd.DataFrame({"LULC id": range(NL), "Name": [M.lulc_name(i, names) for i in range(NL)],
                           "Share as 'from' class, earlier date (%)": 100.0 * fc / n,
                           "Share as 'to' class, later date (%)": 100.0 * tc / n})
    shares["Δ (percentage points)"] = shares.iloc[:, 3] - shares.iloc[:, 2]
    changed = int(bin_chg[valid].sum())
    return dict(table=df, shares=shares, n_valid=n, changed_pixels=changed, unchanged_pixels=n - changed,
                binary_changed_pct=100.0 * changed / n, mean_p_change=float(pc[..., change_id][valid].mean()),
                binary_confidence=float(pc.max(-1)[valid].mean()), mean_confidence=float(cv.mean()),
                implied_changed_pct=100.0 * float(implied[valid].mean()),
                disagreement_pct=100.0 * float((bin_chg != implied)[valid].mean()),
                dominant_change=chg.iloc[0].to_dict() if len(chg) else None,
                dominant_overall=df.sort_values("Pixels", ascending=False).iloc[0].to_dict())


# ------------------------------------------------------------------ spectral indices
def index_provenance(x20, x25) -> dict:
    """Are the stored index channels equal to the formulas applied to bands 0-5? (display values come from raw data)"""
    e20, e25 = P.index_consistency(x20), P.index_consistency(x25)
    vals = [v for d in (e20, e25) for v in d.values() if v is not None]
    worst = max(vals) if vals else None
    return dict(err_earlier=e20, err_later=e25, worst=worst, ok=bool(vals) and worst <= P.INDEX_TOL)


def index_report(x20, x25, valid, chg_mask=None) -> pd.DataFrame:
    """Statistics of the RAW stored index channels (not the normalised network input) over `valid` pixels."""
    rows = []
    for name, info in INDEX_INFO.items():
        k = info["ch"]
        a, b = x20[..., k][valid].astype(np.float64), x25[..., k][valid].astype(np.float64)
        d = b - a
        r = {"Index": name, "Formula": info["formula"], "Signal": info["topic"],
             "Mean earlier": a.mean(), "Mean later": b.mean(), "Δ mean": b.mean() - a.mean(),
             "Median earlier": np.median(a), "Median later": np.median(b), "Δ median": np.median(b) - np.median(a),
             "P10 earlier": np.percentile(a, 10), "P90 earlier": np.percentile(a, 90),
             "P10 later": np.percentile(b, 10), "P90 later": np.percentile(b, 90),
             "Saturated earlier (%)": 100.0 * float((np.abs(a) >= SATURATION).mean()),
             "Saturated later (%)": 100.0 * float((np.abs(b) >= SATURATION).mean()),
             "Mean Δ in predicted-change pixels": np.nan, "Mean Δ in predicted-no-change pixels": np.nan}
        if chg_mask is not None:
            m = chg_mask[valid]
            if m.any():
                r["Mean Δ in predicted-change pixels"] = float(d[m].mean())
            if (~m).any():
                r["Mean Δ in predicted-no-change pixels"] = float(d[~m].mean())
        rows.append(r)
    return pd.DataFrame(rows)


def threshold_shares(x20, x25, valid, thresholds: dict) -> pd.DataFrame:
    """EXPLORATORY: share of valid pixels with index > threshold at each date. Not a validated land-cover class."""
    rows = []
    for name, t in thresholds.items():
        k = INDEX_INFO[name]["ch"]
        s20 = 100.0 * float((x20[..., k][valid] > t).mean())
        s25 = 100.0 * float((x25[..., k][valid] > t).mean())
        rows.append({"Index": name, "Threshold (>)": t, "Share earlier (%)": s20, "Share later (%)": s25, "Δ (percentage points)": s25 - s20})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plain-language report
def _dir(d):
    return "higher" if d > 0 else ("lower" if d < 0 else "identical")


def build_report(S, idx_df, idx_ok, valid_desc, gt=None) -> dict:
    out = {"Model-predicted": [], "Observed spectral-index change": [], "Reference labels": [],
           "Possible interpretation (not a conclusion)": []}
    mp = out["Model-predicted"]
    mp.append(f"The binary head assigns **{S['binary_changed_pct']:.1f}%** of the {S['n_valid']:,} analysed pixels to *change* and "
              f"**{100 - S['binary_changed_pct']:.1f}%** to *no change* (analysed pixels = {valid_desc}).")
    dom = S["dominant_change"]
    if dom:
        mp.append(f"The most frequent predicted change transition is **{dom['Transition']}**: {dom['Percent']:.1f}% of analysed pixels ({int(dom['Pixels']):,} px).")
    else:
        mp.append("No analysed pixel is predicted as a transition with different source and destination class.")
    do = S["dominant_overall"]
    mp.append(f"The most frequent predicted class overall is {do['Transition']} ({do['Percent']:.1f}%).")
    mp.append(f"The transition head implies {S['implied_changed_pct']:.1f}% change; the two heads disagree on {S['disagreement_pct']:.2f}% of pixels. "
              "They are predicted independently, so they need not agree.")
    mp.append(f"Mean P(change) is {S['mean_p_change']:.3f}; mean transition confidence (max softmax) is {S['mean_confidence']:.3f}. This is not a calibrated probability.")
    ob = out["Observed spectral-index change"]
    if not idx_ok:
        ob.append("⚠ The stored index channels could not be verified against bands 0-5, so the values below should be treated with caution.")
    for r in idx_df.to_dict("records"):
        info = INDEX_INFO[r["Index"]]
        d = r["Δ mean"]
        ob.append(f"Mean **{r['Index']}** ({info['topic']}) is {_dir(d)} in the later image: {r['Mean earlier']:.3f} → {r['Mean later']:.3f} (Δ = {d:+.3f}); "
                  f"median {r['Median earlier']:.3f} → {r['Median later']:.3f}.")
        a, b = r["Mean Δ in predicted-change pixels"], r["Mean Δ in predicted-no-change pixels"]
        if np.isfinite(a) and np.isfinite(b):
            ob.append(f"Mean Δ{r['Index']} is {a:+.3f} inside predicted-change pixels and {b:+.3f} inside predicted-no-change pixels.")
    rl = out["Reference labels"]
    if gt is None:
        rl.append("No reference labels were supplied for this input, so prediction accuracy cannot be measured.")
    else:
        rl.append(f"The reference labels mark **{gt['gt_changed_pct']:.1f}%** of the labelled pixels as changed.")
        rl.append(f"For this single patch the binary head has IoU {gt['binary_iou']:.3f}, F1 {gt['binary_f1']:.3f}, accuracy {gt['binary_acc']:.3f}; "
                  f"transition accuracy is {gt['transition_acc']:.3f}. These are sample-level values, not the model's overall performance.")
    pi = out["Possible interpretation (not a conclusion)"]
    pi.append("Index trends describe spectral signal. They are **not** a measured change in vegetation, water or built-up *area*, and this app does not convert them into area percentages.")
    pi.append("Predicted transitions and index changes are different kinds of evidence and can agree or disagree for any given patch.")
    pi.append("Class names are not available, so transitions are reported by class id. No real-world cause (e.g. construction, vegetation loss, flooding) can be inferred from this analysis.")
    return out


# ------------------------------------------------------------------ XAI explanation text (from the actual results)
def explain_attention(maps, chg, valid) -> list:
    lines = []
    for res, m in maps.items():
        a, b = chg & valid, (~chg) & valid
        if a.any() and b.any():
            ia, ib = float(m[a].mean()), float(m[b].mean())
            lines.append(f"Scale {res}: mean gate value is {ia:.3f} inside predicted-change pixels and {ib:.3f} outside ({'higher' if ia > ib else 'lower or equal'} inside).")
        else:
            lines.append(f"Scale {res}: the patch has no predicted-change or no predicted-no-change pixels, so inside/outside cannot be compared.")
    lines.append("Gate values are learned without supervision; a higher value means the gate lets more of that feature through, not that the location is the cause of the prediction.")
    return lines


def explain_se(w20, w25) -> list:
    t20, t25 = np.argsort(-w20)[:3], np.argsort(-w25)[:3]
    j = int(np.argmax(np.abs(w25 - w20)))
    return [f"Highest weights, earlier date: " + ", ".join(f"{P.BAND_SHORT[i]} ({w20[i]:.2f})" for i in t20) + ".",
            f"Highest weights, later date: " + ", ".join(f"{P.BAND_SHORT[i]} ({w25[i]:.2f})" for i in t25) + ".",
            f"The largest difference between the dates is in {P.BAND_SHORT[j]} ({w25[j] - w20[j]:+.3f}).",
            "These weights are applied before further layers that can rescale them, so they are not a ranking of how much each band matters."]


def cam_focus(cam, roi, top=0.2):
    if float(cam.max()) <= 0:
        return None
    k = max(1, int(top * cam.size))
    thr = np.partition(cam.ravel(), -k)[-k]
    sel = cam >= thr
    return float((sel & roi).sum() / max(sel.sum(), 1)), float(roi.mean())


def explain_cam(cams, roi) -> list:
    lines = []
    for nm, c in cams.items():
        f = cam_focus(c, roi)
        if f is None:
            lines.append(f"{nm}: the map is empty (no positive evidence for the selected output at this layer).")
        else:
            lines.append(f"{nm}: {100 * f[0]:.0f}% of the most highlighted 20% of pixels lie inside the explained region, which itself covers {100 * f[1]:.0f}% of the patch.")
    lines.append("Grad-CAM shows where this layer's features weigh most for the selected output; it is a model view, not a ground-truth change map.")
    return lines


def explain_ig(r, roi_pct) -> list:
    sh = r["band_share"]
    c = len(sh) // 2
    comb = sh[:c] + sh[c:]
    top = np.argsort(-comb)[:3]
    return [f"The explained region covers {roi_pct:.1f}% of the patch (predicted pixels of the chosen output).",
            f"{100 * sh[:c].sum():.0f}% of the total attribution magnitude comes from the earlier image and {100 * sh[c:].sum():.0f}% from the later image.",
            "Largest combined band shares: " + ", ".join(f"{P.BAND_SHORT[i]} ({100 * comb[i]:.1f}%)" for i in top) + ".",
            f"Completeness check: attributions sum to {r['sum_attribution']:.3f} versus F(x) − F(baseline) = {r['F_x'] - r['F_baseline']:.3f} (relative error {r['completeness_rel_err']:.3f}).",
            "Attributions are relative to a dataset-mean baseline and depend on the number of steps and the chosen region."]


def explain_date(df) -> list:
    d = df.set_index("Scenario")
    base = float(d.iloc[0]["Predicted change (%)"])
    lines = [f"With the real pair the binary head predicts {base:.1f}% change."]
    for k in d.index[1:]:
        lines.append(f"{k}: {float(d.loc[k, 'Predicted change (%)']):.1f}% predicted change; binary agreement with the baseline {100 * float(d.loc[k, 'Binary agreement with baseline']):.1f}%.")
    ident = [k for k in d.index if k.startswith("identical")]
    if ident:
        lo = max(float(d.loc[k, "Predicted change (%)"]) for k in ident)
        lines.append("If the model relies on differences between dates, identical inputs should predict much less change" +
                     (f" (here at most {lo:.1f}% versus {base:.1f}%)." if lo < base else " (here they do not, so single-date appearance also influences the output)."))
    lines.append("The model was not trained for date-order symmetry, so a swapped pair is informative, not an expected invariance.")
    return lines