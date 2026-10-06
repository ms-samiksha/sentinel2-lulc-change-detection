"""BT-SSSCNet Streamlit demo: inference, change analysis and explanations around the FROZEN trained model."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="BT-SSSCNet demo", page_icon="🛰️", layout="wide")

try:
    import matplotlib.pyplot as plt
    import analysis as A
    import model_utils as M
    import preprocessing as P
    import visualization as V
    import visualization_ext as VE
    import xai_utils as X
except Exception as exc:
    st.error(f"Could not import the project dependencies: {type(exc).__name__}: {exc}\n\n"
             "Activate your virtual environment and run `pip install -r requirements.txt`.")
    st.stop()

APP_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = APP_DIR / "sample_data"
MIN_ROI_PIXELS = 50

# Supplied by the project author from the notebook evaluation of the best_val_loss model. NOT computed by this app.
NOTEBOOK_METRICS = {"Transition (16 classes)": {"Overall accuracy": 82.73, "mIoU": 30.70, "Macro-F1": 40.39},
                    "Binary change": {"Overall accuracy": 87.19, "Change IoU": 70.10, "Change F1": 82.43}}
RAW_IMAGE_NOTE = ("The trained model expects the dataset's prepared 9-channel 128×128 representation. Raw image-to-patch "
                  "preprocessing is not part of the exported model, so JPG/PNG/GeoTIFF files, RGB images and other sizes are not accepted. "
                  "An RGB image does not contain the NIR/SWIR bands that the model and its indices need.")

st.markdown("""<style>
.block-container{padding-top:1.3rem;max-width:1400px}
.hero{padding:1.2rem 1.5rem;border-radius:12px;background:linear-gradient(100deg,#0b1f33,#14507a 70%,#1b7a8c);color:#fff;margin-bottom:.7rem}
.hero h1{margin:0;font-size:2.1rem;color:#fff;letter-spacing:.02em}
.hero h3{margin:.15rem 0 .35rem 0;font-weight:400;color:#cfe3f1;font-size:1.1rem}
.hero p{margin:0;color:#dbe9f4;font-size:.95rem}
.wf{display:flex;gap:.6rem;margin:.2rem 0 1rem 0;flex-wrap:wrap}
.wf div{flex:1;min-width:190px;padding:.6rem .8rem;border-radius:9px;background:rgba(20,80,122,.10);border-left:4px solid #1b7a8c}
.wf b{display:block;font-size:.9rem}.wf span{font-size:.8rem;opacity:.85}
.sec{font-size:.78rem;letter-spacing:.14em;color:#1b7a8c;font-weight:700;margin:1.3rem 0 -.4rem 0}
.banner{padding:.55rem .9rem;border-radius:8px;background:rgba(27,122,140,.12);border:1px solid rgba(27,122,140,.4);margin:.4rem 0 .8rem 0;font-size:.92rem}
</style>""", unsafe_allow_html=True)


def sec(txt):
    st.markdown(f'<div class="sec">{txt}</div>', unsafe_allow_html=True)


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def bullets(lines):
    for l in lines:
        st.markdown(f"- {l}")


def method_box(measures, colours, output, limits):
    with st.expander("What am I looking at?", expanded=True):
        st.markdown(f"**What it measures.** {measures}\n\n**Colours.** {colours}\n\n**Output explained.** {output}\n\n**Limitations.** {limits}")


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading BT-SSSCNet (once per session)...")
def cached_model(path: str, mtime: float):
    return M.load_bt_ssscnet(path)


@st.cache_data(show_spinner=False)
def cached_sha(path: str, mtime: float) -> str:
    return M.sha256_file(path)


@st.cache_data(show_spinner=False)
def cached_norm(path: str, mtime: float):
    return P.load_norm(path)


@st.cache_data(show_spinner=False)
def cached_sample(sample_dir: str, sid: str) -> dict:
    d = Path(sample_dir) / sid
    out = {"errors": [], "notes": [], "x20": None, "x25": None, "gt_t": None, "gt_c": None, "frozen": None}
    for key, fname, required in (("x20", "x2020.npy", True), ("x25", "x2025.npy", True), ("gt_t", "gt_transition.npy", False), ("gt_c", "gt_change.npy", False)):
        f = d / fname
        if not f.is_file():
            (out["errors"] if required else out["notes"]).append(f"{sid}/{fname} is missing")
            continue
        try:
            out[key] = np.load(f, allow_pickle=False)
        except Exception as exc:
            (out["errors"] if required else out["notes"]).append(f"{sid}/{fname} is corrupted: {exc}")
    if out["gt_t"] is not None and out["gt_c"] is not None and (out["gt_t"].shape != (P.PS, P.PS) or out["gt_c"].shape != (P.PS, P.PS)):
        out["notes"].append(f"{sid}: ground-truth arrays have unexpected shapes and are ignored")
        out["gt_t"] = out["gt_c"] = None
    f = d / "frozen_predictions.npz"
    if f.is_file():
        try:
            with np.load(f) as z:
                out["frozen"] = {k: z[k] for k in z.files}
        except Exception as exc:
            out["notes"].append(f"{sid}/frozen_predictions.npz is corrupted: {exc}")
    else:
        out["notes"].append(f"{sid}/frozen_predictions.npz is missing: no notebook comparison for this sample")
    return out


# ------------------------------------------------------------------ header
st.markdown("""<div class="hero"><h1>BT-SSSCNet</h1><h3>Bi-Temporal Land-Cover Transition &amp; Change Detection</h3>
<p>Dual-output Siamese network for bi-temporal land-cover transition and binary change detection. Inference only: the trained model is frozen.</p></div>
<div class="wf"><div><b>① Select inputs</b><span>Bundled test patch or prepared 9-channel .npy pair</span></div>
<div><b>② Run analysis</b><span>Saved normalisation → frozen model</span></div>
<div><b>③ Explore changes</b><span>Change map, transitions, spectral indices</span></div>
<div><b>④ Understand the prediction</b><span>Attention, Grad-CAM, Integrated Gradients</span></div></div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ startup
if not SAMPLE_DIR.is_dir():
    st.error(f"`sample_data/` not found at {SAMPLE_DIR}. Unzip BT_SSSCNet_Streamlit_samples.zip (notebook Cell 73) into it.")
    st.stop()
norm_path = SAMPLE_DIR / "norm_stats.npz"
if not norm_path.is_file():
    st.error("`sample_data/norm_stats.npz` is missing. It holds the training-split normalisation statistics and cannot be recreated by the app.")
    st.stop()
try:
    MEAN, STD = cached_norm(str(norm_path), norm_path.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not read norm_stats.npz: {exc}")
    st.stop()

cfg, cfg_err = M.read_json(SAMPLE_DIR / "preprocessing_config.json")
if cfg_err:
    st.warning(f"{cfg_err}; using the default clip ±{P.DEFAULT_CLIP:g}.")
cfg = cfg or {}
CLIP = float(cfg.get("norm_clip", P.DEFAULT_CLIP))
mapping, map_err = M.read_json(SAMPLE_DIR / "class_mappings.json")
if map_err:
    st.warning(f"{map_err}; using id = 4·from + to and binary change id = 1 (the notebook's verified encoding).")
CHANGE_ID = int(((mapping or {}).get("binary") or {}).get("change", 1))
NO_CHANGE_ID = int(((mapping or {}).get("binary") or {}).get("no_change", 0))
names, names_warn = M.load_lulc_names(SAMPLE_DIR)
if names_warn:
    st.warning(names_warn)
label = lambda k: M.transition_label(k, names)

index, idx_err = M.read_json(SAMPLE_DIR / "samples_index.json")
if idx_err or not isinstance(index, list) or not index:
    index = []

cands = M.find_model_candidates(APP_DIR)
model_path = cands[0] if cands else APP_DIR / "model" / M.MODEL_FILENAME
if not model_path.is_file():
    st.error(f"Model file not found: `{model_path}`. Place `{M.MODEL_FILENAME}` in the `model/` folder (see README).")
    st.stop()
try:
    model = cached_model(str(model_path), model_path.stat().st_mtime)
except M.ModelLoadError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Unexpected error while loading the model: {type(exc).__name__}: {exc}")
    st.stop()
sha = cached_sha(str(model_path), model_path.stat().st_mtime)
hs = M.hash_status(model_path, SAMPLE_DIR, sha)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Model")
    st.markdown(f"**BT-SSSCNet**  \n`{model_path.name}`  \n{model.count_params():,} parameters")
    st.markdown({"match": "Hash: **✓ verified**", "mismatch": "Hash: **⚠ MISMATCH**", "unrecorded": "Hash: **? not recorded**"}[hs["status"]])
    st.markdown("2020 → 2025 · 128×128 · 9 channels  \nOutputs: 16 transitions + binary change")
    with st.expander("Notebook evaluation metrics", expanded=True):
        st.caption("Model-wide results reported from the notebook evaluation. **Not recomputed by this app.**")
        for grp, vals in NOTEBOOK_METRICS.items():
            st.markdown(f"**{grp}**")
            for k, v in vals.items():
                st.write(f"{k}: **{v:.2f}%**")
    with st.expander("Technical details"):
        st.markdown("Shared Siamese encoder · spectral SE · per-scale temporal fusion · TFRM (SE + spatial attention) · multi-scale decoder · two heads.  \n"
                    "Channels: B2, B3, B4, B8, B11, B12, NDVI, NDWI, NDBI.  \nTransition id = 4·from + to; binary change id = " + str(CHANGE_ID) + ".")
        st.code(f"loaded file : {hs['sha256']}\nrecorded    : {hs['recorded']}", language="text")
        st.caption(f"TensorFlow {M.tf.__version__} · Keras {M.keras.__version__}")

# ------------------------------------------------------------------ model status (one line)
if hs["status"] == "match":
    st.success("✓ Verified model: the file's SHA-256 matches `sample_data/model_hash.json`, the file used to generate the frozen notebook predictions.")
elif hs["status"] == "mismatch":
    st.error("⚠ Model hash mismatch: this is NOT the exact model that produced the frozen predictions. Reproducibility is not confirmed by the hash.")
else:
    st.warning(f"No usable hash record ({hs['record_error']}); cannot confirm this is the model used for the frozen predictions.")

# ------------------------------------------------------------------ 1. inputs
sec("1 · SELECT INPUTS")
modes = (["Bundled test sample"] if index else []) + ["Upload prepared .npy pair"]
if not index:
    st.warning(f"No usable samples_index.json ({idx_err or 'empty'}); only uploads are available.")
mode = st.radio("Input source", modes, horizontal=True)
inp = None
if mode.startswith("Bundled"):
    entry = st.selectbox("Bundled sample", index, format_func=lambda e: f"{e['id']}  ·  {e.get('group', '')}  ·  tile {e.get('parent_tile', '?')}")
    b = cached_sample(str(SAMPLE_DIR), entry["id"])
    for e_ in b["errors"]:
        st.error(e_)
    for n_ in b["notes"]:
        st.info(n_)
    if not b["errors"]:
        v20, v25 = P.validate_input(b["x20"], "earlier (x2020.npy)"), P.validate_input(b["x25"], "later (x2025.npy)")
        for e_ in v20.errors + v25.errors:
            st.error(e_)
        for w_ in v20.warnings + v25.warnings:
            st.warning(w_)
        if v20.ok and v25.ok:
            inp = dict(name=entry["id"], source="bundled test sample", detail=f"{entry.get('group', '')}, tile {entry.get('parent_tile', '?')}",
                       x20=v20.x_hwc, x25=v25.x_hwc, gt_t=b["gt_t"], gt_c=b["gt_c"], frozen=b["frozen"])
else:
    st.info("Accepted: prepared `.npy` patches of shape (9,128,128) or (128,128,9), channels [B2, B3, B4, B8, B11, B12, NDVI, NDWI, NDBI] in the dataset's raw value ranges "
            "(the app applies the saved normalisation itself). " + RAW_IMAGE_NOTE)
    u1, u2 = st.columns(2)
    f20 = u1.file_uploader("Earlier observation (2020) · .npy", type=["npy"], key="u20")
    f25 = u2.file_uploader("Later observation (2025) · .npy", type=["npy"], key="u25")
    if f20 and f25:
        arrs, bad = [], False
        for f, nm in ((f20, "earlier"), (f25, "later")):
            try:
                arrs.append(np.load(io.BytesIO(f.getvalue()), allow_pickle=False))
            except Exception as exc:
                st.error(f"{nm}: '{f.name}' could not be read as a NumPy array ({exc}). " + RAW_IMAGE_NOTE)
                bad = True
        if not bad:
            v20, v25 = P.validate_input(arrs[0], f"earlier ({f20.name})"), P.validate_input(arrs[1], f"later ({f25.name})")
            for e_ in v20.errors + v25.errors:
                st.error(e_)
            for w_ in v20.warnings + v25.warnings:
                st.warning(w_)
            if v20.ok and v25.ok:
                for v, nm in ((v20, "earlier"), (v25, "later")):
                    if A.looks_standardised(v.x_hwc):
                        st.warning(f"The {nm} file looks already standardised (large spread in bands 0-5). The app normalises raw dataset values exactly once; "
                                   "feeding standardised data would normalise it twice and give invalid predictions.")
                if np.array_equal(v20.x_hwc, v25.x_hwc):
                    st.warning("The earlier and later arrays are identical; no change is expected.")
                inp = dict(name=f"{f20.name} + {f25.name}", source="uploaded pair", detail=f"{f20.name} / {f25.name}",
                           x20=v20.x_hwc, x25=v25.x_hwc, gt_t=None, gt_c=None, frozen=None)
    else:
        st.caption("Upload both files to continue.")

if inp is None:
    st.info("Select a valid input to continue.")
    st.stop()

cur_key = A.input_key(inp["x20"], inp["x25"], sha, MEAN, STD, CLIP)
auto = st.checkbox("Run automatically when the input changes", value=True,
                   help="Inference takes about a second on CPU. Explanations (XAI) always wait for a button.")
res = st.session_state.get("res")
need = res is None or res["key"] != cur_key
clicked = st.button("Run analysis", type="primary")


def run_analysis(inp, key):
    raw_hash = hashlib.md5(inp["x20"].tobytes() + inp["x25"].tobytes()).hexdigest()
    z20, z25 = P.normalize(inp["x20"], MEAN, STD, CLIP), P.normalize(inp["x25"], MEAN, STD, CLIP)   # applied exactly once
    if hashlib.md5(inp["x20"].tobytes() + inp["x25"].tobytes()).hexdigest() != raw_hash:
        raise RuntimeError("raw input arrays were modified during preprocessing")
    pt, pc = M.predict(model, z20, z25)
    valid, vdesc = A.analysis_mask(inp["gt_c"])
    pred_t, pred_c = pt.argmax(-1), pc.argmax(-1)
    chg = pred_c == CHANGE_ID
    S = A.prediction_summary(pt, pc, valid, CHANGE_ID, names)
    prov = A.index_provenance(inp["x20"], inp["x25"])
    idx_df = A.index_report(inp["x20"], inp["x25"], valid, chg)
    r = dict(key=key, name=inp["name"], source=inp["source"], detail=inp["detail"], z20=z20, z25=z25, pt=pt, pc=pc, pred_t=pred_t, pred_c=pred_c,
             chg=chg, valid=valid, vdesc=vdesc, S=S, prov=prov, idx_df=idx_df, gt=None, frozen_cmp=None, xai={})
    if inp["gt_t"] is not None and inp["gt_c"] is not None:
        gt_t, gt_c = inp["gt_t"].astype(np.int64), inp["gt_c"].astype(np.int64)
        sm = M.sample_metrics(gt_t, gt_c, pred_t, pred_c, CHANGE_ID)
        sm["gt_changed_pct"] = 100.0 * float(((gt_c == CHANGE_ID) & valid).sum() / max(valid.sum(), 1))
        r["gt"] = sm
    if inp["frozen"] is not None:
        try:
            r["frozen_cmp"] = M.compare_with_frozen(pt, pc, inp["frozen"])
        except Exception as exc:
            r["frozen_cmp"] = {"error": str(exc)}
    return r


if need and (clicked or auto):
    try:
        with st.spinner("Normalising with the saved training statistics and running BT-SSSCNet..."):
            res = run_analysis(inp, cur_key)
        st.session_state["res"] = res
    except Exception as exc:
        st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
        st.stop()
elif need:
    st.info("The selected input has changed. Click **Run analysis**; the previous results are hidden so they cannot be mistaken for this input.")
    st.stop()

res = st.session_state["res"]
assert res["key"] == cur_key, "stale result"
S, valid, pred_t, pred_c, chg, ptp, pcp = res["S"], res["valid"], res["pred_t"], res["pred_c"], res["chg"], res["pt"], res["pc"]
z20, z25, xai = res["z20"], res["z25"], res["xai"]
rgb20, rgb25 = P.rgb_pair(inp["x20"], inp["x25"])
has_gt = res["gt"] is not None
st.markdown(f'<div class="banner">Results below belong to: <b>{res["name"]}</b> ({res["source"]}; {res["detail"]}) · input fingerprint <code>{cur_key[:10]}</code></div>', unsafe_allow_html=True)

names_tabs = ["Summary", "Imagery", "Change & transitions", "Spectral indices"] + (["Ground truth"] if has_gt else []) + ["Explainability", "Reproducibility", "About & limitations"]
T = dict(zip(names_tabs, st.tabs(names_tabs)))

# ------------------------------------------------------------------ Summary
with T["Summary"]:
    sec("2 · OVERALL CHANGE SUMMARY")
    k = st.columns(6)
    k[0].metric("Predicted changed", f"{S['binary_changed_pct']:.1f}%", help="Share of analysed pixels the binary head assigns to the change class (argmax).")
    k[1].metric("Predicted unchanged", f"{100 - S['binary_changed_pct']:.1f}%")
    k[2].metric("Pixels analysed", f"{S['n_valid']:,}", help=f"Denominator of every percentage: {res['vdesc']}.")
    k[3].metric("Mean P(change)", f"{S['mean_p_change']:.3f}")
    dom = S["dominant_change"]
    k[4].metric("Dominant change transition", f"T{int(dom['ID'])}" if dom else "none", help=(f"{dom['Transition']} · {dom['Percent']:.1f}% of analysed pixels") if dom else "No from ≠ to transition predicted")
    k[5].metric("Transition confidence", f"{S['mean_confidence']:.3f}", help="Mean of the largest softmax probability of the transition head over analysed pixels. Not a calibrated probability.")
    st.caption(f"All percentages are **shares of pixels** ({res['vdesc']}), not geographic area. No pixel size or georeferencing is available, so no hectares or km² are reported.")
    if dom:
        st.markdown(f"Dominant predicted change transition: **{dom['Transition']}** ({dom['Percent']:.1f}% of analysed pixels).")
    if S["disagreement_pct"] > 0:
        st.warning(f"The binary head and the transition head disagree on {S['disagreement_pct']:.2f}% of pixels (binary: {S['binary_changed_pct']:.1f}% change; "
                   f"transition head implies {S['implied_changed_pct']:.1f}%). The heads are trained and predicted independently, so some disagreement is expected; nothing here forces them to agree.")
    else:
        st.info("The binary head and the transition-implied change agree on every analysed pixel of this input.")
    if not has_gt:
        st.info("No reference labels for this input: accuracy cannot be measured. Reference labels are required to compute accuracy, IoU or F1.")
    fc = res["frozen_cmp"]
    if fc and "error" not in fc:
        (st.success if fc["passed"] else st.error)(f"Notebook reproducibility for this sample: max |Δp| = {fc['max_abs_prob_diff']:.2e}, argmax agreement transition {fc['transition_argmax_agreement']:.5f} / binary {fc['change_argmax_agreement']:.5f} → {'PASS' if fc['passed'] else 'FAIL'}")
    show(V.fig_overview(rgb20, rgb25, chg, pred_t))
    st.caption("RGB composites (R=B4, G=B3, B=B2, shared 2-98 % stretch) are for display only and are not the model input.")
    sec("PLAIN-LANGUAGE CHANGE REPORT")
    rep = A.build_report(S, res["idx_df"], res["prov"]["ok"], res["vdesc"], res["gt"])
    c1, c2 = st.columns(2)
    for col, keys in ((c1, ["Model-predicted", "Observed spectral-index change"]), (c2, ["Reference labels", "Possible interpretation (not a conclusion)"])):
        with col:
            for kk in keys:
                st.markdown(f"**{kk}**")
                bullets(rep[kk])
    st.caption("These sentences are generated from the numbers in this run. Model predictions, index observations and reference labels are different kinds of evidence and are kept separate.")

# ------------------------------------------------------------------ Imagery
with T["Imagery"]:
    comp = st.radio("Composite", ["True colour (R=B4, G=B3, B=B2)", "False colour (R=B8, G=B4, B=B3)"], horizontal=True)
    a_, b_ = P.rgb_pair(inp["x20"], inp["x25"]) if comp.startswith("True") else P.false_colour_pair(inp["x20"], inp["x25"])
    show(V.fig_rgb_pair(a_, b_, "Earlier (2020)", "Later (2025)", comp))
    st.markdown("**Temporal difference view**")
    show(V.fig_temporal_difference(inp["x20"], inp["x25"]))
    st.caption("Absolute difference |later − earlier| of the raw bands. Bright = the two dates differ more. It is an input view, not a model output or a causal explanation.")

# ------------------------------------------------------------------ Change & transitions
with T["Change & transitions"]:
    st.markdown("#### Binary change")
    show(V.fig_binary_panel(rgb25, chg, pcp[..., CHANGE_ID]))
    st.caption("Left: predicted mask. Middle: probability of change from the binary head (colour bar). Right: predicted change over the later image.")
    with st.expander("Binary head versus transition-implied change"):
        show(V.fig_head_agreement(chg, (pred_t // M.NL) != (pred_t % M.NL)))
    st.markdown("#### Semantic transitions")
    show(V.fig_transition_map(pred_t, rgb25, label))
    tbl = S["table"]
    allc = st.checkbox("Include transitions with 0 predicted pixels", value=False)
    t = (tbl if allc else tbl[tbl.Pixels > 0]).sort_values("Pixels", ascending=False)
    st.markdown("**Transition breakdown** (percent of analysed pixels)")
    st.dataframe(t[["ID", "Transition", "Type", "Pixels", "Percent", "Mean confidence", "Mean P(class)"]].round({"Percent": 2, "Mean confidence": 3, "Mean P(class)": 4}), hide_index=True)
    show(VE.fig_transition_bar(tbl))
    st.caption("Type follows the verified encoding id = 4·from + to: Stay if from = to, Change otherwise. Mean confidence = mean largest softmax probability over the pixels assigned to that class.")
    st.markdown("**Model-implied class shares per date**")
    st.dataframe(S["shares"].round(2), hide_index=True)
    show(VE.fig_class_shares(S["shares"]))
    st.caption("Shares are derived from the transition head (from = id // 4 for the earlier date, to = id % 4 for the later date). Changes are in percentage points of pixels, not area. "
               "Class ids carry no verified names; supply `sample_data/lulc_names.json` to display names.")
    with st.expander("All 16 transition classes (legend)"):
        st.dataframe(V.style_legend(M.transition_table(names)), hide_index=True)

# ------------------------------------------------------------------ Spectral indices
with T["Spectral indices"]:
    st.markdown("#### Vegetation, water and built-up signals")
    prov, idx_df = res["prov"], res["idx_df"]
    if prov["ok"]:
        st.success(f"Index channels verified: the stored NDVI/NDWI/NDBI equal the formulas applied to bands 0-5 (worst median abs error {prov['worst']:.4f} ≤ {P.INDEX_TOL}). "
                   "The values below are the raw stored channels, **not** the normalised network input.")
    else:
        st.warning("The stored index channels could NOT be verified against bands 0-5 (or the check was undefined). Interpret the index values with caution.")
    with st.expander("Verification detail"):
        st.dataframe(pd.DataFrame({"Earlier": prov["err_earlier"], "Later": prov["err_later"]}).rename_axis("Index (median |stored − recomputed|)"))
    cols_main = ["Index", "Signal", "Formula", "Mean earlier", "Mean later", "Δ mean", "Median earlier", "Median later", "Δ median"]
    st.dataframe(idx_df[cols_main].round(4), hide_index=True)
    show(VE.fig_index_means(idx_df))
    st.dataframe(idx_df[["Index", "P10 earlier", "P90 earlier", "P10 later", "P90 later", "Mean Δ in predicted-change pixels", "Mean Δ in predicted-no-change pixels",
                         "Saturated earlier (%)", "Saturated later (%)"]].round(4), hide_index=True)
    st.caption("Saturated = |index| ≥ 0.999, often caused by a near-zero band sum; high values here make means less reliable.")
    show(VE.fig_index_distributions(inp["x20"], inp["x25"], valid))
    with st.expander("Spatial maps of each index and its change"):
        show(VE.fig_index_maps(inp["x20"], inp["x25"]))
    st.markdown("**Reading the numbers**")
    for r in idx_df.to_dict("records"):
        info = A.INDEX_INFO[r["Index"]]
        d = r["Δ mean"]
        st.markdown(f"- **{r['Index']}** ({info['topic']}, {info['formula']}): Δ mean = {d:+.3f}. A positive change means {info['higher']}; a negative change means {info['lower']}.")
    st.warning("These indices are spectral trends, not land-cover labels. A change in mean NDVI/NDWI/NDBI is **not** a measured change in vegetated, water or built-up *area*, and no percentage area change is claimed here.")
    with st.expander("Exploratory threshold analysis (optional, not validated)"):
        st.caption("No verified class mapping or threshold rule exists in this project. If you enable this, the thresholds are YOUR choice; the starting values are common "
                   "literature conventions and have not been validated for this dataset. Output = share of analysed pixels above the threshold at each date.")
        if st.checkbox("Enable exploratory threshold analysis", value=False, key="thr_on"):
            tc = st.columns(3)
            thr = {"NDVI": tc[0].number_input("NDVI >", -1.0, 1.0, 0.3, 0.05), "NDWI": tc[1].number_input("NDWI >", -1.0, 1.0, 0.0, 0.05), "NDBI": tc[2].number_input("NDBI >", -1.0, 1.0, 0.0, 0.05)}
            st.dataframe(A.threshold_shares(inp["x20"], inp["x25"], valid, thr).round(3), hide_index=True)
            st.caption("Exploratory only. Results change with the thresholds.")
    with st.expander("What is still needed for validated land-cover proportions"):
        st.markdown("- A verified mapping from the 4 class ids to real land-cover names (or a documented threshold rule per class).\n"
                    "- Documentation of how bands 0-5 were produced (processing level, scaling, cloud/shadow masking, resampling, handling of zero-reflectance pixels). "
                    "The values reach about 1.8, so they are not plain 0-1 reflectance.\n"
                    "- Pixel resolution and georeferencing, if any area figure is wanted.")

# ------------------------------------------------------------------ Ground truth
if has_gt:
    with T["Ground truth"]:
        sm = res["gt"]
        gt_t, gt_c = inp["gt_t"].astype(np.int64), inp["gt_c"].astype(np.int64)
        st.markdown("**Sample-level metrics for this bundled patch** (valid labelled pixels only)")
        g = st.columns(3)
        g[0].metric("Binary accuracy", f"{sm['binary_acc']:.4f}"); g[1].metric("Binary change IoU", f"{sm['binary_iou']:.4f}"); g[2].metric("Binary change F1", f"{sm['binary_f1']:.4f}")
        h = st.columns(3)
        h[0].metric("Transition accuracy", f"{sm['transition_acc']:.4f}"); h[1].metric("Transition mIoU", f"{sm['transition_mIoU']:.4f}"); h[2].metric("Transition macro-F1", f"{sm['transition_macro_f1']:.4f}")
        st.caption(f"Computed on {sm['n_valid']:,} valid pixels (label 255 ignored); mIoU/macro-F1 average the {sm['n_classes_present']} classes present in this patch. "
                   "These are **not** the model-wide notebook metrics shown in the sidebar and can be unstable for one patch.")
        show(V.fig_gt_comparison(gt_t, pred_t, gt_c, pred_c, CHANGE_ID))
        with st.expander("Transition confusion matrix (this patch)"):
            ids = sorted(set(np.unique(gt_t[gt_t != 255]).tolist()) | set(np.unique(pred_t).tolist()))
            show(V.fig_confusion_matrix(sm["cm_transition"], ids))
            st.caption("Sparse classes make single-patch confusion matrices unstable.")

# ------------------------------------------------------------------ Explainability
with T["Explainability"]:
    st.markdown("### Why did the model predict this?")
    st.info("These views show what the **model** used or emphasised. They are attention/attribution visualisations, not ground-truth change maps, not proof of the model's reasoning, "
            "and not evidence of what physically caused a change. Everything is computed live on the loaded model; nothing runs until you click a button.")
    with st.expander("Layer availability in the loaded model"):
        st.dataframe(pd.DataFrame([{"layer": k_, "available": v_} for k_, v_ in X.layer_report(model).items()]), hide_index=True)

    opts = [("Binary head · change (predicted change region)", "change_output", CHANGE_ID, chg & valid),
            ("Binary head · no change (predicted no-change region)", "change_output", NO_CHANGE_ID, (~chg) & valid)]
    for _, r in S["table"][S["table"].Pixels >= MIN_ROI_PIXELS].sort_values("Pixels", ascending=False).iterrows():
        kk = int(r["ID"])
        opts.append((f"Transition head · {label(kk)} ({int(r['Pixels']):,} px)", "transition_output", kk, (pred_t == kk) & valid))
    sel = st.selectbox("Output to explain (used by Grad-CAM and Integrated Gradients)", range(len(opts)), format_func=lambda j: opts[j][0])
    tname, head, cls, roi = opts[sel]
    st.caption(f"Explained output: **{tname}**. The explained region is the set of pixels predicted as this class ({100 * roi.mean():.1f}% of the patch).")
    xt = st.tabs(["Temporal difference", "Spatial attention", "Spectral SE", "Grad-CAM", "Integrated Gradients", "Date dependency"])

    with xt[0]:
        method_box("How much each spectral band differs between the two dates, computed directly from the inputs.",
                   "Brighter = larger absolute difference between the dates.", "None. This is an input-only view and does not use the model.",
                   "A large difference is not necessarily a land-cover change (clouds, haze, seasonal or sensor effects also change pixel values).")
        show(V.fig_temporal_difference(inp["x20"], inp["x25"]))

    with xt[1]:
        method_box("The spatial gates inside the model's Temporal Feature Refinement Modules: how strongly each location of the fused features is passed on, at four scales.",
                   "Bright = higher gate value (colour range is relative to each scale). Cyan outline = predicted binary change.",
                   "Not tied to one output; these are internal gates of the network.",
                   "Learned without supervision. A bright gate means the feature passes through, not that the location caused the prediction.")
        if st.button("Compute spatial attention", key="b_sa"):
            try:
                xai["sa"] = X.spatial_attention(model, z20, z25)
            except X.XAIUnavailable as exc:
                st.warning(f"Spatial attention unavailable: {exc}")
            except Exception as exc:
                st.warning(f"Spatial attention failed ({type(exc).__name__}: {exc}).")
        if "sa" in xai:
            maps, missing = xai["sa"]
            if missing:
                st.warning(f"Layers not found and skipped: {missing}")
            show(VE.fig_attention_explained(rgb25, maps, chg))
            st.markdown("**What this result shows**")
            bullets(A.explain_attention(maps, chg, valid))

    with xt[2]:
        method_box("The weights of the spectral squeeze-and-excitation block that rescales the 9 input channels before the encoder.",
                   "Bar height 0-1: low = channel suppressed, high = channel kept. One bar per channel and date.",
                   "Not tied to one output; the weights depend on this patch's spectral content.",
                   "Later layers can rescale these weights, so they are not a ranking of band importance and not causal.")
        if st.button("Compute spectral SE weights", key="b_se"):
            try:
                xai["se"] = X.spectral_se(model, z20, z25)
            except X.XAIUnavailable as exc:
                st.warning(f"Spectral SE unavailable: {exc}")
            except Exception as exc:
                st.warning(f"Spectral SE failed ({type(exc).__name__}: {exc}).")
        if "se" in xai:
            w20, w25 = xai["se"]
            show(VE.fig_se_explained(w20, w25))
            st.markdown("**What this result shows**")
            bullets(A.explain_se(w20, w25))

    with xt[3]:
        method_box("Seg-Grad-CAM: which locations in a chosen layer's feature maps most increase the selected output (gradient-weighted activations), shown at three depths.",
                   "Bright = more evidence for the selected output at that layer (scaled 0-1). Cyan outline = the explained region.",
                   f"{tname}.", "Coarse layers are upsampled, so maps are blurry. It shows where the layer's features weigh most for the model, not whether the prediction is right or what caused the change.")
        if int(roi.sum()) < MIN_ROI_PIXELS:
            st.warning(f"Fewer than {MIN_ROI_PIXELS} pixels in the selected region; choose another output.")
        else:
            ck = ("cam", head, cls)
            if st.button("Compute Grad-CAM", key="b_cam"):
                try:
                    with st.spinner("Computing Seg-Grad-CAM at three depths..."):
                        xai[ck] = X.gradcam_all(model, z20, z25, head, cls, roi)
                except X.XAIUnavailable as exc:
                    st.warning(f"Grad-CAM unavailable: {exc}")
                except Exception as exc:
                    st.warning(f"Grad-CAM failed ({type(exc).__name__}: {exc}). Layer access can differ between Keras versions.")
            if ck in xai:
                cams, missing = xai[ck]
                if missing:
                    st.warning(f"Layers not found and skipped: {missing}")
                show(VE.fig_cam_explained(rgb25, cams, roi, tname))
                st.markdown("**What this result shows**")
                bullets(A.explain_cam(cams, roi))

    with xt[4]:
        method_box("Integrated Gradients: how much each input value (pixel × band × date) contributed to the selected output, compared with a dataset-mean baseline image.",
                   "Signed maps: red = raises the output, blue = lowers it. Hot map = total magnitude (bright = larger). Bars = share of attribution per band and date.",
                   f"{tname}.", "Depends on the baseline, the number of steps and the region. It describes the model's sensitivity, not causation, and does not show the prediction is correct.")
        steps = st.slider("Integration steps", 8, 64, 24, 8, help="More steps lower the completeness error but are slower.")
        st.info("Integrated Gradients may take some time on CPU.")
        if int(roi.sum()) < MIN_ROI_PIXELS:
            st.warning(f"Fewer than {MIN_ROI_PIXELS} pixels in the selected region; choose another output.")
        else:
            ik = ("ig", head, cls, steps)
            if st.button("Compute Integrated Gradients", key="b_ig"):
                bar = st.progress(0.0)
                try:
                    xai[ik] = X.integrated_gradients(model, z20, z25, head, cls, roi, steps=steps, progress=lambda f_: bar.progress(float(min(f_, 1.0))))
                except X.XAIUnavailable as exc:
                    st.warning(f"Integrated Gradients unavailable: {exc}")
                except Exception as exc:
                    st.warning(f"Integrated Gradients failed ({type(exc).__name__}: {exc}).")
                bar.empty()
            if ik in xai:
                r = xai[ik]
                show(VE.fig_ig_explained(rgb20, rgb25, r, roi))
                (st.success if r["completeness_rel_err"] < 0.10 else st.warning)(
                    f"Completeness check: relative error {r['completeness_rel_err']:.3f} (attributions should sum to F(x) − F(baseline); below 0.10 is acceptable, otherwise raise the steps).")
                st.markdown("**What this result shows**")
                bullets(A.explain_ig(r, 100 * roi.mean()))
                dk = ("del",) + ik
                if st.button("Run deletion test (faithfulness diagnostic)", key="b_del"):
                    with st.spinner("Running deletion test..."):
                        xai[dk] = X.deletion_test(model, z20, z25, r["score"], head, cls, roi)
                if dk in xai:
                    d = xai[dk]
                    show(V.fig_deletion(d))
                    st.dataframe(d.round(4), hide_index=True)
                    drops = bool((d.output_ratio_IG_ranked <= d.output_ratio_random).all())
                    st.markdown(f"- Deleting the most-attributed pixels {'lowered the output at least as much as' if drops else 'did not always lower the output more than'} deleting random pixels at every tested fraction.")
                    st.caption("This tests whether the attribution is faithful to the model, not whether the prediction is correct.")

    with xt[5]:
        method_box("Whether the prediction depends on the two dates differing: the binary head is re-run with the dates replaced by copies of each other, and swapped.",
                   "Bars = percent of pixels predicted as change in each scenario.", "Binary change head.",
                   "The model was not trained for date-order symmetry, so a swapped pair is informative rather than an expected invariance.")
        if st.button("Run date-dependency test", key="b_dd"):
            try:
                xai["dd"] = X.date_dependency(model, z20, z25, CHANGE_ID)
            except Exception as exc:
                st.warning(f"Date-dependency test failed ({type(exc).__name__}: {exc}).")
        if "dd" in xai:
            st.dataframe(xai["dd"].round(4), hide_index=True)
            show(VE.fig_date_dependency(xai["dd"]))
            st.markdown("**What this result shows**")
            bullets(A.explain_date(xai["dd"]))

# ------------------------------------------------------------------ Reproducibility
with T["Reproducibility"]:
    st.markdown("Each bundled sample is converted to (128,128,9), normalised **once** with the saved `norm_stats.npz` "
                f"(clip ±{CLIP:g}), run through the loaded `.keras` and compared with `frozen_predictions.npz` from the notebook. PASS = max |Δp| < 1e-3 and ≥ 99.9% identical argmax pixels.")
    st.write(f"Model hash status: **{hs['status'].upper()}**")
    if res["frozen_cmp"] is None:
        st.info("The current input is an uploaded pair, so no frozen notebook prediction exists for it. The all-samples check below uses the bundled samples.")
    if st.button("Run reproducibility check on all bundled samples", key="b_rep"):
        try:
            with st.spinner("Running all bundled samples..."):
                st.session_state["repro"] = M.check_all_samples(model, SAMPLE_DIR, MEAN, STD, CLIP)
        except Exception as exc:
            st.error(f"Reproducibility check could not run: {exc}")
    df = st.session_state.get("repro")
    if df is not None:
        st.dataframe(df.round({"max_abs_dp": 8, "transition_argmax_agreement": 6, "change_argmax_agreement": 6}), hide_index=True)
        (st.success if (df["result"] == "PASS").all() else st.error)(f"{int((df['result'] == 'PASS').sum())} of {len(df)} samples PASS.")
        st.caption("This confirms the app reproduces the notebook's inference for these patches. It does not prove model accuracy.")

# ------------------------------------------------------------------ About
with T["About & limitations"]:
    st.markdown("#### Model-wide notebook evaluation (reported by the author; not recomputed here)")
    c = st.columns(6); i = 0
    for grp, vals in NOTEBOOK_METRICS.items():
        for k_, v_ in vals.items():
            c[i].metric(f"{'Transition' if grp.startswith('Trans') else 'Binary'} · {k_}", f"{v_:.2f}%"); i += 1
    st.caption("Keep these separate from the sample-level numbers on the Ground truth tab, which describe a single patch.")
    st.markdown("#### Limitations")
    st.markdown("""
- **Inputs:** only prepared 9-channel 128×128 `.npy` patches. Raw Sentinel-2 downloads, JPG/PNG/GeoTIFF/RGB images and other sizes are not supported.
- **Generalisation:** held-out parent tiles are not independent geographic regions; independent geographic generalisation was not established.
- **Rare classes:** some transitions have very few pixels (class 0→3 has almost none), so their per-class behaviour is unreliable.
- **Class names:** not part of the dataset metadata; ids are shown unless a documented `lulc_names.json` is supplied.
- **Spectral indices:** index trends are not land-cover area changes; thresholds, if used, are exploratory.
- **XAI:** attention and attribution maps are descriptive of the model, not causal, and are not a measure of correctness.
- **Band identity:** the index channels were verified numerically; band 0 being blue is inferred from channel order.
- **Pixels, not area:** all percentages are pixel shares; no georeferencing is available.
""")