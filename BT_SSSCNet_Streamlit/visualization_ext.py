"""Extra figures: transition bars, class shares, spectral-index charts, and XAI views with colour bars."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.patches import Patch   # noqa: E402
import numpy as np   # noqa: E402

import visualization as V   # noqa: E402
from analysis import INDEX_INFO   # noqa: E402
from preprocessing import BAND_SHORT   # noqa: E402


def _cb(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=8)
    return cb


def _off(axes):
    for a in np.atleast_1d(axes).ravel():
        a.set_xticks([]); a.set_yticks([])


def _contour(ax, mask, color="cyan"):
    if mask.any() and (~mask).any():
        ax.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=0.9)


# ------------------------------------------------------------------ transitions
def fig_transition_bar(df):
    d = df[df.Pixels > 0].sort_values("Pixels")
    fig, ax = plt.subplots(figsize=(8.5, max(2.6, 0.45 * len(d) + 1.2)))
    labels = [f"T{int(r.ID)}: {r.From} → {r.To}" for r in d.itertuples()]
    bars = ax.barh(labels, d.Percent, color=[V.PAL_T[int(i)] / 255 for i in d.ID], edgecolor="k")
    for b, typ, r in zip(bars, d.Type, d.itertuples()):
        if typ == "Stay":
            b.set_hatch("//")
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f"  {r.Percent:.1f}%  ({int(r.Pixels):,} px)", va="center", fontsize=8)
    ax.set_xlim(0, max(d.Percent.max() * 1.3, 1)); ax.set_xlabel("% of analysed pixels")
    ax.set_title("Predicted transitions (hatched = stay, solid = change)")
    ax.legend(handles=[Patch(facecolor="white", edgecolor="k", hatch="//", label="Stay (from = to)"), Patch(facecolor="white", edgecolor="k", label="Change (from ≠ to)")],
              loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_class_shares(shares):
    x = np.arange(len(shares))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    a, b = shares.iloc[:, 2], shares.iloc[:, 3]
    ax.bar(x - 0.2, a, 0.4, label="earlier date (implied by 'from')"); ax.bar(x + 0.2, b, 0.4, label="later date (implied by 'to')")
    for i, (u, v) in enumerate(zip(a, b)):
        ax.text(i - 0.2, u, f"{u:.1f}", ha="center", va="bottom", fontsize=8); ax.text(i + 0.2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(shares["Name"]); ax.set_ylabel("% of analysed pixels")
    ax.set_title("Model-implied class shares per date"); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ spectral indices
def fig_index_means(idx_df):
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for a, r in zip(ax, idx_df.to_dict("records")):
        m = [r["Mean earlier"], r["Mean later"]]
        lo = [max(m[0] - r["P10 earlier"], 0), max(m[1] - r["P10 later"], 0)]
        hi = [max(r["P90 earlier"] - m[0], 0), max(r["P90 later"] - m[1], 0)]
        bars = a.bar(["earlier", "later"], m, yerr=[lo, hi], capsize=4, color=["#7f8fa6", "#1b7a8c"])
        for b, v in zip(bars, m):
            a.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        a.axhline(0, color="k", lw=0.6)
        a.set_title(f"{r['Index']} ({r['Signal']})\nΔ mean = {r['Δ mean']:+.3f}", fontsize=9)
    fig.suptitle("Mean index per date (bars) with 10th-90th percentile range (whiskers)", fontsize=9)
    fig.tight_layout()
    return fig


def fig_index_distributions(x20, x25, valid):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, (name, info) in zip(ax, INDEX_INFO.items()):
        u, v = x20[..., info["ch"]][valid], x25[..., info["ch"]][valid]
        bins = np.linspace(min(u.min(), v.min()), max(u.max(), v.max()), 60)
        a.hist(u, bins=bins, density=True, alpha=0.5, label="earlier"); a.hist(v, bins=bins, density=True, alpha=0.5, label="later")
        a.axvline(u.mean(), color="C0", ls="--", lw=1); a.axvline(v.mean(), color="C1", ls="--", lw=1)
        a.set_title(f"{name} distribution (dashed = mean)", fontsize=9); a.set_xlabel("index value"); a.set_ylabel("density"); a.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_index_maps(x20, x25):
    fig, ax = plt.subplots(3, 3, figsize=(13, 11))
    for r, (name, info) in enumerate(INDEX_INFO.items()):
        k = info["ch"]; a, b = x20[..., k], x25[..., k]; d = b - a
        lim = float(np.percentile(np.abs(d), 99)) + 1e-6
        for c, (m, t) in enumerate(((a, f"{name} earlier"), (b, f"{name} later"))):
            im = ax[r, c].imshow(m, cmap=info["cmap"], vmin=-1, vmax=1); ax[r, c].set_title(t, fontsize=9); _cb(fig, ax[r, c], im, "index value")
        im = ax[r, 2].imshow(d, cmap="RdBu_r" if name != "NDVI" else "PiYG", vmin=-lim, vmax=lim)
        ax[r, 2].set_title(f"Δ{name} = later − earlier", fontsize=9); _cb(fig, ax[r, 2], im, "difference (centre = no change)")
    _off(ax)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ XAI
def fig_attention_explained(rgb, maps, chg):
    n = len(maps)
    fig, ax = plt.subplots(1, n, figsize=(3.9 * n, 4.3))
    for a, (res, m) in zip(np.atleast_1d(ax), maps.items()):
        a.imshow(rgb)
        im = a.imshow(m, cmap="inferno", alpha=0.65, vmin=float(m.min()), vmax=float(m.max()))
        _contour(a, chg)
        a.set_title(f"TFRM spatial gate, {res}", fontsize=9)
        _cb(fig, a, im, "gate value (bright = higher)")
    _off(ax)
    fig.suptitle("Learned spatial-attention gates on the later image; cyan outline = predicted change", fontsize=9)
    fig.tight_layout()
    return fig


def fig_se_explained(w20, w25):
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(w20))
    b1 = ax.bar(x - 0.2, w20, 0.4, label="earlier date"); b2 = ax.bar(x + 0.2, w25, 0.4, label="later date")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(BAND_SHORT); ax.set_ylim(0, 1.1); ax.set_ylabel("SE weight (0 = suppressed, 1 = kept)")
    ax.set_title("Spectral SE band weights for this patch"); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_cam_explained(rgb, cams, roi, title):
    n = len(cams)
    fig, ax = plt.subplots(1, n + 1, figsize=(3.9 * (n + 1), 4.3))
    ax[0].imshow(rgb); ax[0].imshow(np.where(roi, 1.0, np.nan), cmap="autumn", alpha=0.55, vmin=0, vmax=1)
    ax[0].set_title(f"Explained region (orange)\n{title}", fontsize=8)
    for a, (nm, c) in zip(ax[1:], cams.items()):
        a.imshow(rgb)
        im = a.imshow(c, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
        _contour(a, roi)
        a.set_title(f"Grad-CAM, {nm}", fontsize=9)
        _cb(fig, a, im, "relative evidence (1 = strongest)")
    _off(ax)
    fig.tight_layout()
    return fig


def fig_ig_explained(rgb20, rgb25, r, roi):
    s20, s25 = r["ig20"].sum(-1), r["ig25"].sum(-1)
    lim = float(np.percentile(np.abs(np.stack([s20, s25])), 99.5)) + 1e-9
    tot = r["score"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.4))
    ax[0, 0].imshow(rgb20); _contour(ax[0, 0], roi); ax[0, 0].set_title("Earlier RGB (cyan = explained region)", fontsize=9)
    ax[0, 1].imshow(rgb25); _contour(ax[0, 1], roi); ax[0, 1].set_title("Later RGB", fontsize=9)
    ax[0, 2].imshow(rgb25)
    im = ax[0, 2].imshow(tot, cmap="hot", alpha=0.65, vmin=0, vmax=float(np.percentile(tot, 99.5)) + 1e-9)
    ax[0, 2].set_title("Total attribution magnitude (both dates, all bands)", fontsize=9); _cb(fig, ax[0, 2], im, "|IG| (bright = larger)")
    for a, mp, t in ((ax[1, 0], s20, "Signed IG, earlier input (sum over bands)"), (ax[1, 1], s25, "Signed IG, later input (sum over bands)")):
        im = a.imshow(mp, cmap="RdBu_r", vmin=-lim, vmax=lim); a.set_title(t, fontsize=9)
        _cb(fig, a, im, "red = raises the output, blue = lowers it")
    sh = r["band_share"]; c = len(sh) // 2
    ax[1, 2].bar(np.arange(c) - 0.2, sh[:c], 0.4, label="earlier"); ax[1, 2].bar(np.arange(c) + 0.2, sh[c:], 0.4, label="later")
    ax[1, 2].set_xticks(range(c)); ax[1, 2].set_xticklabels(BAND_SHORT, rotation=60, fontsize=8); ax[1, 2].set_ylabel("share of total |IG|")
    ax[1, 2].legend(fontsize=8); ax[1, 2].set_title("Attribution share per band", fontsize=9)
    _off(ax[:, :2]); _off(ax[0, 2])
    fig.tight_layout()
    return fig


def fig_date_dependency(df):
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    names = [s.split(" (")[0] if s.startswith("baseline") else s for s in df["Scenario"]]
    bars = ax.barh(names[::-1], df["Predicted change (%)"][::-1], color="#1b7a8c")
    for b in bars:
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {b.get_width():.1f}%", va="center", fontsize=8)
    ax.set_xlabel("predicted change (% of pixels)"); ax.set_title("Date-dependency test"); ax.set_xlim(0, max(float(df["Predicted change (%)"].max()) * 1.2, 1))
    fig.tight_layout()
    return fig