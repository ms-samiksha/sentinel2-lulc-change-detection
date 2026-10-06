"""Preprocessing for BT-SSSCNet inference.

Mirrors the training notebook exactly:
  * layout  : (9,128,128) dataset patches -> (128,128,9)
  * scaling : z = (x - mean) / std using TRAIN-split statistics from sample_data/norm_stats.npz
  * clipping: z clipped to [-10, +10]
No statistics are ever computed from user data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PS = 128
C = 9
BAND_NAMES = ["B2 blue", "B3 green", "B4 red", "B8 NIR", "B11 SWIR1", "B12 SWIR2", "NDVI", "NDWI", "NDBI"]
BAND_SHORT = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI", "NDBI"]
RGB_IDX = (2, 1, 0)             # R = B4, G = B3, B = B2
FALSE_COLOUR_IDX = (3, 2, 1)    # R = B8, G = B4, B = B3
# band k = (b_i - b_j) / (b_i + b_j)
INDEX_DEFINITIONS = {6: ("NDVI", 3, 2),    # (B8 - B4) / (B8 + B4)
                     7: ("NDWI", 1, 3),    # (B3 - B8) / (B3 + B8)
                     8: ("NDBI", 4, 3)}    # (B11 - B8) / (B11 + B8)
DEFAULT_CLIP = 10.0
DENOM_EPS = 0.02      # safe-division threshold (same as the notebook's verification)
INDEX_TOL = 0.01      # median absolute error tolerated between stored and recomputed index


class InputValidationError(ValueError):
    """Raised for inputs that cannot be fed to the model."""


@dataclass
class ValidationResult:
    x_hwc: np.ndarray | None = None
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    index_error: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.x_hwc is not None and not self.errors


def load_norm(path):
    """Load the training-split per-band mean/std (shape (9,))."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Normalisation file not found: {p}")
    with np.load(p) as d:
        if "mean" not in d.files or "std" not in d.files:
            raise ValueError(f"{p.name} must contain 'mean' and 'std' (found {d.files}).")
        mean = np.asarray(d["mean"], np.float32).reshape(-1)
        std = np.asarray(d["std"], np.float32).reshape(-1)
    if mean.shape != (C,) or std.shape != (C,):
        raise ValueError(f"mean/std must have shape ({C},), got {mean.shape} and {std.shape}.")
    if not (np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()):
        raise ValueError("mean/std contain non-finite values or std <= 0.")
    return mean, std


def to_hwc(x) -> np.ndarray:
    """(9,128,128) -> (128,128,9); (128,128,9) is passed through. Output is contiguous float32."""
    x = np.asarray(x, np.float32)
    if x.shape == (C, PS, PS):
        x = np.transpose(x, (1, 2, 0))
    elif x.shape != (PS, PS, C):
        raise InputValidationError(f"Expected shape (9,128,128) or (128,128,9), got {x.shape}.")
    return np.ascontiguousarray(x)


def index_consistency(x_hwc: np.ndarray) -> dict:
    """Median |stored index - index recomputed from bands 0-5| for NDVI/NDWI/NDBI (None if undefined)."""
    out = {}
    for k, (name, i, j) in INDEX_DEFINITIONS.items():
        den = x_hwc[..., i] + x_hwc[..., j]
        m = den > DENOM_EPS
        if not m.any():
            out[name] = None
            continue
        rec = (x_hwc[..., i] - x_hwc[..., j])[m] / den[m]
        out[name] = float(np.median(np.abs(rec - x_hwc[..., k][m])))
    return out


def recompute_indices_if_needed(x_hwc: np.ndarray, tol: float = INDEX_TOL, apply: bool = False):
    """Verify the three index channels against bands 0-5.

    By default this is a REPORT ONLY: the input is returned unchanged (the app never alters model input).
    With apply=True, indices are rewritten where the safe-division mask holds (denominator > DENOM_EPS)
    and only for indices whose median error exceeds `tol`.
    Returns (x_out, report) with report = {"errors": {...}, "needs_recompute": bool, "applied": bool}.
    """
    errs = index_consistency(x_hwc)
    needs = any(v is not None and v > tol for v in errs.values())
    out, applied = x_hwc, False
    if apply and needs:
        out = x_hwc.copy()
        for k, (name, i, j) in INDEX_DEFINITIONS.items():
            if errs[name] is not None and errs[name] > tol:
                den = x_hwc[..., i] + x_hwc[..., j]
                m = den > DENOM_EPS
                rec = np.where(m, (x_hwc[..., i] - x_hwc[..., j]) / np.where(m, den, 1.0), x_hwc[..., k])
                out[..., k] = rec
        applied = True
    return out, {"errors": errs, "needs_recompute": needs, "applied": applied}


def validate_input(arr, name: str = "input") -> ValidationResult:
    """Validate a prepared patch. Never modifies the data."""
    res = ValidationResult()
    if not isinstance(arr, np.ndarray):
        res.errors.append(f"{name}: not a NumPy array.")
        return res
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
        res.errors.append(f"{name}: dtype {arr.dtype} is not a real numeric type.")
        return res
    if arr.ndim != 3 or arr.shape not in ((C, PS, PS), (PS, PS, C)):
        res.errors.append(
            f"{name}: shape {arr.shape} is not (9,128,128) or (128,128,9). The model has a fixed 128x128 input with "
            f"9 channels [B2, B3, B4, B8, B11, B12, NDVI, NDWI, NDBI]; other sizes or channel counts are rejected.")
        return res
    if not np.isfinite(arr).all():
        res.errors.append(f"{name}: contains {int((~np.isfinite(arr)).sum())} non-finite values (NaN/Inf).")
        return res
    x = to_hwc(arr)
    res.x_hwc = x
    lo, hi = float(x[..., :6].min()), float(x[..., :6].max())
    if hi > 5.0 or lo < -0.5:
        res.warnings.append(f"{name}: bands 0-5 range [{lo:.3f}, {hi:.3f}]. The training data are reflectance-like "
                            f"(roughly 0-1.8). Is this the dataset's prepared product?")
    if x[..., 6:].max() > 1.001 or x[..., 6:].min() < -1.001:
        res.warnings.append(f"{name}: bands 6-8 (NDVI/NDWI/NDBI) fall outside [-1, 1].")
    res.index_error = index_consistency(x)
    for nm, e in res.index_error.items():
        if e is not None and e > INDEX_TOL:
            res.warnings.append(f"{name}: stored {nm} differs from the value recomputed from bands 0-5 "
                                f"(median abs error {e:.3f}); the band order or index definition may not match training.")
    return res


def normalize(x_hwc: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float | None = DEFAULT_CLIP) -> np.ndarray:
    """(x - mean) / std, then clip to [-clip, +clip]. Same as the notebook's norm_np / exported preprocess."""
    if x_hwc.shape != (PS, PS, C):
        raise InputValidationError(f"normalize() expects (128,128,9), got {x_hwc.shape}.")
    z = (x_hwc - mean) / std
    if clip is not None:
        z = np.clip(z, -clip, clip)
    return z.astype(np.float32)


# ---------------------------------------------------------------- visualisation-only helpers
def composite(x_hwc: np.ndarray, idx=RGB_IDX) -> np.ndarray:
    return x_hwc[..., list(idx)]


def _stretch(a: np.ndarray, b: np.ndarray, pct=(2, 98)):
    lo, hi = np.percentile(np.concatenate([a.ravel(), b.ravel()]), pct)
    f = lambda v: np.clip((v - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return f(a), f(b)


def rgb_pair(x20_hwc, x25_hwc, pct=(2, 98)):
    """True-colour composites (R=B4, G=B3, B=B2) with ONE shared 2-98 % stretch. Display only."""
    return _stretch(composite(x20_hwc, RGB_IDX), composite(x25_hwc, RGB_IDX), pct)


def false_colour_pair(x20_hwc, x25_hwc, pct=(2, 98)):
    """False-colour composites (R=B8, G=B4, B=B3) with ONE shared 2-98 % stretch. Display only."""
    return _stretch(composite(x20_hwc, FALSE_COLOUR_IDX), composite(x25_hwc, FALSE_COLOUR_IDX), pct)