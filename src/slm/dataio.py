"""Data IO helpers for notebook use: load camera BMPs, colorize, and analyze
light-sheet uniformity. All side-effect free — return arrays/dicts and let
the caller plot/save."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
import matplotlib
from scipy.ndimage import label, median_filter
from scipy.optimize import curve_fit

# Allied Vision Alvium 1800 U-1240m (Sony IMX226), 1.85 µm pitch.
CAM_PITCH_UM_DEFAULT = 1.85

PathLike = Union[str, Path]
ArrayLike = Union[PathLike, np.ndarray]


def load_bmp(path: PathLike, *, dtype=np.float64) -> np.ndarray:
    """Load an 8-bit grayscale BMP as a 2D ndarray."""
    return np.asarray(Image.open(path).convert("L"), dtype=dtype)


def colorize(
    img: np.ndarray,
    cmap: str = "hot",
    vmax: int | None = None,
    bbox: tuple[int, int, int, int] | None = None,  # (y0, y1, x0, x1)
    box_color: tuple[int, int, int] = (0, 255, 255),  # 默认青色
    box_thickness: int = 4,
) -> np.ndarray:
    arr = np.asarray(img)
    vmax_use = int(vmax) if vmax is not None else int(arr.max())
    vmax_use = max(vmax_use, 1)
    cm = matplotlib.colormaps[cmap]
    norm = np.clip(arr.astype(np.float32) / vmax_use, 0.0, 1.0)
    rgb = (cm(norm) * 255).astype(np.uint8)[..., :3]

    if bbox is not None:
        y0, y1, x0, x1 = [int(v) for v in bbox]
        h, w = rgb.shape[:2]
        y0 = max(0, min(y0, h - 1))
        y1 = max(0, min(y1, h))
        x0 = max(0, min(x0, w - 1))
        x1 = max(0, min(x1, w))
        if y1 > y0 and x1 > x0:
            t = max(int(box_thickness), 1)
            c = np.array(box_color, dtype=np.uint8)
            rgb[y0:y0+t, x0:x1] = c
            rgb[y1-t:y1, x0:x1] = c
            rgb[y0:y1, x0:x0+t] = c
            rgb[y0:y1, x1-t:x1] = c

    return rgb

def detect_sheet_bbox(
    after: np.ndarray, threshold_frac: float = 0.30, pad: int = 6,
) -> tuple[tuple[int, int, int, int], bool]:
    sig = median_filter(after, size=3)
    bg = float(np.median(sig))
    sig_bs = sig - bg
    peak = float(sig_bs.max())
    if peak <= 0:
        raise RuntimeError("nothing brighter than background in after image")

    mask = sig_bs > threshold_frac * peak
    lbl, n = label(mask)
    if n == 0:
        raise RuntimeError("no bright blobs above threshold")

    H, W = after.shape
    best_label = 0
    best_extent = -1.0
    best_dy = best_dx = 0.0
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lbl == i)
        dy = float(ys.max() - ys.min() + 1)
        dx = float(xs.max() - xs.min() + 1)
        extent = max(dy, dx)
        if extent > best_extent:
            best_extent = extent
            best_label = i
            best_dy, best_dx = dy, dx

    ys, xs = np.nonzero(lbl == best_label)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, H)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, W)
    major_is_y = best_dy > best_dx
    return (y0, y1, x0, x1), major_is_y






def compute_reweight(
    v: np.ndarray,
    steepness: float,
    clip_lo: float = 0.85,
    clip_hi: float = 1.15,
) -> np.ndarray:
    """Amplitude-domain feedback weight (sqrt-inverse, clipped, mean-normalized).

    Used by closed-loop SLM uniformity feedback: given a measured 1D
    intensity profile *v* across the flat region, return a length-N
    multiplicative reweight that pushes the next iteration toward
    uniformity.  ``steepness=0`` disables correction (returns ones-like);
    ``steepness=1`` applies the full ``1/sqrt(v)`` correction.  ``clip_lo``
    and ``clip_hi`` bound the per-sample weight to keep CGM stable.
    """
    v = np.asarray(v, dtype=np.float64)
    v_norm = np.clip(v / v.mean(), 1e-6, None)
    inv_sqrt = 1.0 / np.sqrt(v_norm)
    inv_sqrt = inv_sqrt / inv_sqrt.mean()
    w = (1.0 - steepness) + steepness * inv_sqrt
    w = np.clip(w, clip_lo, clip_hi)
    return w / w.mean()
