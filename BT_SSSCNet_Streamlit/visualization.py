"""Matplotlib figures for the BT-SSSCNet demo. Every function returns a Figure; the caller closes it."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.patches import Patch   # noqa: E402
import numpy as np   # noqa: E402

from preprocessing import BAND_SHORT   # noqa: E402

NL = 4
GRAY, RED = np.array([0.88, 0.88, 0.88]), np.array([0.84, 0.15, 0.16])
BIN_COL = {"TN": (225, 225, 225), "TP": (46, 160, 67), "FP": (220, 50, 47), "FN": (38, 110, 200)}


def _palette():
    pal = np.full((256, 3), 255, np.uint8)
    pal[:16] = (plt.get_cmap("tab20")(np.arange(16))[:, :3] * 255).astype(np.uint8)
    return pal


PAL_T = _palette()


def palette_hex(k: int) -> str:
    r, g, b = PAL_T[k]
    return f"#{r:02x}{g:02x}{b:02x}"


def overlay(rgb, mask, color=RED, alpha=0.5):
    out = rgb.copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.asarray(color)
    return out


def _off(axes):
    for a in np.atleast_1d(axes).ravel():
        a.set_xticks([])
        a.set_yticks([])


def _cbar(fig, ax, im, label=None):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    if label:
        cb.set_label(label, fontsize=8)


def _binary_rgb(mask):
    return np.where(mask[..., None], RED, GRAY)


# ------------------------------------------------------------------ imagery
def fig_rgb_pair(a, b, title_a="Earlier (2020)", title_b="Later (2025)", suptitle=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.8))
    ax[0].imshow(a); ax[0].set_title(title_a)
    ax[1].imshow(b); ax[1].set_title(title_b)
    _off(ax)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    return fig


def fig_overview(rgb20, rgb25, change_mask, pt):
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
    ax[0].imshow(rgb20); ax[0].set_title("2020 RGB (B4-B3-B2)")
    ax[1].imshow(rgb25); ax[1].set_title("2025 RGB (B4-B3-B2)")
    ax[2].imshow(_binary_rgb(change_mask)); ax[2].set_title("Binary change map (red = change)")
    ax[3].imshow(PAL_T[pt]); ax[3].set_title("Predicted transition map")
    _off(ax)
    fig.tight_layout()
    return fig


def fig_temporal_difference(x20_hwc, x25_hwc):
    """|later - earlier| on the raw (un-normalised) spectral bands; display only."""
    d = np.abs(x25_hwc - x20_hwc)
    mag = np.sqrt((d[..., :6] ** 2).sum(-1))
    fig, ax = plt.subplots(2, 4, figsize=(16, 7.4))
    im = ax[0, 0].imshow(mag, cmap="magma"); ax[0, 0].set_title("Aggregate |Δ| (L2 over 6 bands)"); _cbar(fig, ax[0, 0], im)
    for a_, k in zip(list(ax.ravel())[1:7], range(6)):
        im = a_.imshow(d[..., k], cmap="magma", vmin=0, vmax=np.percentile(d[..., k], 99) + 1e-6)
        a_.set_title(f"|Δ| {BAND_SHORT[k]}"); _cbar(fig, a_, im)
    ax[1, 3].axis("off")
    _off(ax)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ binary change
def _draw_mask(ax, mask):
    ax.imshow(_binary_rgb(mask)); ax.set_title("Binary change mask (red = change)")


def _draw_prob(fig, ax, p):
    im = ax.imshow(p, vmin=0, vmax=1, cmap="viridis"); ax.set_title("P(change), binary head"); _cbar(fig, ax, im)


def _draw_overlay(ax, rgb, mask):
    ax.imshow(overlay(rgb, mask)); ax.set_title("Later RGB + predicted change")


def fig_binary_panel(rgb_later, change_mask, p_change):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    _draw_mask(ax[0], change_mask); _draw_prob(fig, ax[1], p_change); _draw_overlay(ax[2], rgb_later, change_mask)
    _off(ax)
    fig.legend(handles=[Patch(facecolor=GRAY, edgecolor="k", label="No change"), Patch(facecolor=RED, edgecolor="k", label="Change")],
               loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def fig_head_agreement(binary_mask, implied_mask):
    """Where the binary head and the transition head (from != to) agree or disagree."""
    img = np.full(binary_mask.shape + (3,), 225, np.uint8)
    img[binary_mask & implied_mask] = (46, 160, 67)
    img[binary_mask & ~implied_mask] = (255, 165, 0)
    img[~binary_mask & implied_mask] = (148, 103, 189)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.imshow(img); ax.set_title("Binary head vs transition-implied change"); _off(ax)
    fig.legend(handles=[Patch(facecolor=np.array(c) / 255, edgecolor="k", label=l) for c, l in (
        ((46, 160, 67), "both: change"), ((225, 225, 225), "both: no change"),
        ((255, 165, 0), "binary=change, transition=stay"), ((148, 103, 189), "binary=no change, transition=change"))],
        loc="lower center", fontsize=7, ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig


# ------------------------------------------------------------------ transitions
def fig_transition_map(pt, rgb_later, label_fn):
    chg = (pt // NL) != (pt % NL)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.8))
    ax[0].imshow(PAL_T[pt]); ax[0].set_title("Predicted semantic transition map (all pixels)")
    ov = rgb_later.copy()
    ov[chg] = 0.25 * ov[chg] + 0.75 * (PAL_T[pt][chg] / 255.0)
    ax[1].imshow(ov); ax[1].set_title("Later RGB with change transitions (from != to) coloured")
    _off(ax)
    ids = [int(k) for k in np.unique(pt)]
    fig.legend(handles=[Patch(facecolor=PAL_T[k] / 255, edgecolor="k", label=f"{label_fn(k)}  ({100 * (pt == k).mean():.1f}%)") for k in ids],
               loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04 + 0.032 * ((len(ids) + 1) // 2), 1, 1))
    return fig


def style_legend(df):
    """pandas Styler adding a colour swatch column to the 16-class legend table."""
    d = df.copy()
    d.insert(0, "Colour", [palette_hex(i) for i in range(len(d))])
    return d.style.apply(lambda s: [f"background-color: {v}; color: {v}" for v in s], subset=["Colour"], axis=0)


# ------------------------------------------------------------------ ground truth
def fig_gt_comparison(gt_t, pred_t, gt_c, pred_c, change_id=1):
    valid = gt_c != 255
    g, p = (gt_c == change_id) & valid, pred_c == change_id
    fig, ax = plt.subplots(2, 3, figsize=(14, 9))

    def binary_gt(mask):
        img = np.full(mask.shape + (3,), 255, np.uint8)
        img[valid & ~mask] = (225, 225, 225)
        img[mask] = (214, 38, 40)
        return img

    ax[0, 0].imshow(binary_gt(g)); ax[0, 0].set_title("Ground truth: binary change")
    ax[0, 1].imshow(binary_gt(p & valid)); ax[0, 1].set_title("Predicted: binary change")
    err = np.full(g.shape + (3,), 255, np.uint8)
    for m, c in (((~g) & (~p) & valid, "TN"), (g & p, "TP"), ((~g) & p & valid, "FP"), (g & ~p, "FN")):
        err[m] = BIN_COL[c]
    ax[0, 2].imshow(err); ax[0, 2].set_title("Binary error map")
    vt = gt_t != 255
    ax[1, 0].imshow(PAL_T[gt_t]); ax[1, 0].set_title("Ground truth: transition")
    ax[1, 1].imshow(PAL_T[pred_t]); ax[1, 1].set_title("Predicted: transition")
    ok = np.full(gt_t.shape + (3,), 255, np.uint8)
    ok[vt & (gt_t == pred_t)] = (46, 160, 67)
    ok[vt & (gt_t != pred_t)] = (220, 50, 47)
    ax[1, 2].imshow(ok); ax[1, 2].set_title("Transition correct (green) / wrong (red)")
    _off(ax)
    fig.legend(handles=[Patch(facecolor=np.array(BIN_COL[k]) / 255, edgecolor="k", label=l) for k, l in (
        ("TN", "TN: no change, correct"), ("TP", "TP: change, correct"), ("FP", "FP: false change"), ("FN", "FN: missed change"))],
        loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_confusion_matrix(cm, class_ids, label_fn=None):
    """Row-normalised confusion matrix over the classes present in truth or prediction (counts annotated)."""
    sub = cm[np.ix_(class_ids, class_ids)].astype(float)
    row = sub.sum(1, keepdims=True)
    norm = np.divide(sub, row, out=np.zeros_like(sub), where=row > 0)
    n = len(class_ids)
    fig, ax = plt.subplots(figsize=(max(4.5, 0.7 * n + 3), max(4, 0.7 * n + 2.5)))
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    for i in range(n):
        for j in range(n):
            if sub[i, j] > 0:
                ax.text(j, i, f"{int(sub[i, j])}", ha="center", va="center", fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
    labs = [f"T{k}" for k in class_ids]
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(labs, rotation=90); ax.set_yticklabels(labs)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Transition confusion (colour = row %, text = pixel count)")
    _cbar(fig, ax, im)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ XAI
def fig_attention_maps(rgb_later, maps, p_change):
    n = len(maps)
    fig, ax = plt.subplots(1, n + 2, figsize=(3.5 * (n + 2), 3.8))
    ax[0].imshow(rgb_later); ax[0].set_title("Later RGB")
    for a_, (res, m) in zip(ax[1:], maps.items()):
        im = a_.imshow(m, cmap="inferno", vmin=m.min(), vmax=m.max())
        a_.set_title(f"TFRM spatial attention {res}\nrange {m.min():.2f}-{m.max():.2f}", fontsize=8)
    ax[-1].imshow(p_change, vmin=0, vmax=1, cmap="viridis"); ax[-1].set_title("P(change)")
    _off(ax)
    fig.tight_layout()
    return fig


def fig_spectral_importance(w20, w25):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(w20))
    ax.bar(x - 0.2, w20, 0.4, label="earlier date"); ax.bar(x + 0.2, w25, 0.4, label="later date")
    ax.set_xticks(x); ax.set_xticklabels(BAND_SHORT); ax.set_ylim(0, 1); ax.set_ylabel("SE band weight (sigmoid)")
    ax.set_title("Spectral SE band weights for this patch"); ax.legend()
    fig.tight_layout()
    return fig


def fig_gradcam(rgb_later, cams, roi, title):
    n = len(cams)
    fig, ax = plt.subplots(1, n + 2, figsize=(3.7 * (n + 2), 4))
    ax[0].imshow(rgb_later); ax[0].set_title("Later RGB")
    ax[1].imshow(roi, cmap="gray"); ax[1].set_title(f"Explained region\n{title}", fontsize=8)
    for a_, (nm, c) in zip(ax[2:], cams.items()):
        a_.imshow(rgb_later); a_.imshow(c, cmap="inferno", alpha=0.6, vmin=0, vmax=1); a_.set_title(f"Grad-CAM: {nm}", fontsize=8)
    _off(ax)
    fig.tight_layout()
    return fig


def fig_integrated_gradients(rgb20, rgb25, r, roi):
    s20, s25 = r["ig20"].sum(-1), r["ig25"].sum(-1)
    lim = np.percentile(np.abs(np.stack([s20, s25])), 99.5) + 1e-9
    fig, ax = plt.subplots(2, 3, figsize=(14.5, 9))
    ax[0, 0].imshow(rgb20); ax[0, 0].set_title("Earlier RGB")
    ax[0, 1].imshow(rgb25); ax[0, 1].set_title("Later RGB")
    ax[0, 2].imshow(roi, cmap="gray"); ax[0, 2].set_title("Explained region (predicted)")
    for a_, mp, t in ((ax[1, 0], s20, "IG, earlier input (signed, summed over bands)"), (ax[1, 1], s25, "IG, later input (signed, summed over bands)")):
        im = a_.imshow(mp, cmap="RdBu_r", vmin=-lim, vmax=lim); a_.set_title(t, fontsize=9); _cbar(fig, a_, im)
    sh = r["band_share"]; c = len(sh) // 2
    ax[1, 2].bar(np.arange(c) - 0.2, sh[:c], 0.4, label="earlier"); ax[1, 2].bar(np.arange(c) + 0.2, sh[c:], 0.4, label="later")
    ax[1, 2].set_xticks(range(c)); ax[1, 2].set_xticklabels(BAND_SHORT, rotation=60, fontsize=8); ax[1, 2].legend(fontsize=8)
    ax[1, 2].set_title("Share of total |IG| per band")
    _off(ax[:, :2]); _off(ax[0, 2])
    fig.tight_layout()
    return fig


def fig_deletion(df):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(df.deleted_fraction * 100, df.output_ratio_IG_ranked, "o-", label="IG-ranked deletion")
    ax.plot(df.deleted_fraction * 100, df.output_ratio_random, "s--", label="random deletion")
    ax.set_xlabel("pixels deleted (%)"); ax.set_ylabel("output ratio vs original"); ax.set_title("Deletion test (faithfulness diagnostic)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig