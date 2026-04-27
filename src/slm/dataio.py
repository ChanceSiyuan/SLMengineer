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
) -> np.ndarray:
    """Return an RGB uint8 array of *img* mapped through *cmap*.

    Pure transform — no IO, no figure. Caller does
    ``plt.imshow(colorize(arr))`` or ``Image.fromarray(...).save(...)``.
    """
    arr = np.asarray(img)
    vmax_use = int(vmax) if vmax is not None else int(arr.max())
    vmax_use = max(vmax_use, 1)
    cm = matplotlib.colormaps[cmap]
    norm = np.clip(arr.astype(np.float32) / vmax_use, 0.0, 1.0)
    return (cm(norm) * 255).astype(np.uint8)[..., :3]


def _detect_sheet_bbox(
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


def _top_hat_edge(u, amp, center, half_width, edge_sigma, baseline):
    dist = np.maximum(0.0, np.abs(u - center) - half_width)
    return baseline + amp * np.exp(-(dist ** 2) / (2.0 * edge_sigma ** 2))


def _fit_flat_top(profile: np.ndarray) -> tuple[float, float]:
    u = np.arange(len(profile), dtype=np.float64)
    baseline0 = float(profile.min())
    amp0 = float(profile.max() - baseline0)
    c0 = float(np.argmax(profile))
    above = profile > (baseline0 + 0.5 * amp0)
    hw0 = max(float(above.sum()) / 2.0, 2.0)
    try:
        popt, _ = curve_fit(
            _top_hat_edge, u, profile,
            p0=[amp0, c0, hw0, 2.0, baseline0],
            bounds=(
                [0.0, 0.0, 0.5, 0.1, -np.inf],
                [np.inf, len(profile), len(profile), len(profile), np.inf],
            ),
            maxfev=10000,
        )
        return float(popt[1]), float(abs(popt[2]))
    except Exception:
        idx = np.where(above)[0]
        if idx.size:
            return 0.5 * float(idx[0] + idx[-1]), max(0.5 * float(idx[-1] - idx[0]), 1.0)
        return c0, max(hw0, 1.0)


def _coerce_bmp(x: ArrayLike) -> np.ndarray:
    if isinstance(x, (str, Path)):
        return load_bmp(x)
    return np.asarray(x, dtype=np.float64)


def analyze_sheet(
    after: ArrayLike,
    *,
    before: ArrayLike | None = None,
    cam_pitch_um: float = CAM_PITCH_UM_DEFAULT,
    flat_a: float | None = 50,
    flat_b: float | None = 200,
) -> dict:
    """Analyze light-sheet uniformity from an ``after`` camera capture.

    Parameters
    ----------
    after : path or 2D array
        Camera image (or path to BMP).
    before : path or 2D array, optional
        Optional dark/blank frame for column-FPN subtraction.
    cam_pitch_um : float
        Camera pixel pitch (default Alvium IMX226: 1.85 µm).
    flat_a, flat_b : float or None
        Manually selected flat-top region in *µm* along the sheet major axis.
        When None the fitted top-hat half-width is used instead.

    Returns
    -------
    dict
        Same keys as the legacy ``analysis_sheet.analyze`` result, plus
        ``profile`` / ``flat_profile`` arrays and ROI ``roi`` for the caller
        to plot inline. No figure is created and no files are written.
    """
    after_arr = _coerce_bmp(after)
    dark_corrected = False
    if before is not None:
        before_arr = _coerce_bmp(before)
        if before_arr.shape == after_arr.shape:
            after_arr = np.clip(after_arr - before_arr, 0.0, None)
            dark_corrected = True

    (y0, y1, x0, x1), major_is_y = _detect_sheet_bbox(after_arr)
    roi = after_arr[y0:y1, x0:x1]
    ny, nx = roi.shape

    if major_is_y:
        minor_proj = roi.sum(axis=0)
        peak_minor = int(np.argmax(minor_proj))
        lo = max(peak_minor - 1, 0); hi = min(peak_minor + 2, nx)
        profile = roi[:, lo:hi].mean(axis=1)
        axis_label = "y (µm)"
    else:
        minor_proj = roi.sum(axis=1)
        peak_minor = int(np.argmax(minor_proj))
        lo = max(peak_minor - 1, 0); hi = min(peak_minor + 2, ny)
        profile = roi[lo:hi, :].mean(axis=0)
        axis_label = "x (µm)"

    center_px, half_width_px = _fit_flat_top(profile)
    if flat_a is not None and flat_b is not None:
        a = round(max(flat_a / cam_pitch_um, 0))
        b = round(min(flat_b / cam_pitch_um, len(profile)))
    else:
        a = max(int(round(center_px - half_width_px)), 0)
        b = min(int(round(center_px + half_width_px)), len(profile))
    flat = profile[a:b] if b > a else profile

    mean_val = float(flat.mean()) if flat.size else 0.0
    rms_pct = 100.0 * float(flat.std()) / mean_val if mean_val > 0 else float("nan")
    ppk_pct = (100.0 * float(flat.max() - flat.min()) / mean_val
               if mean_val > 0 else float("nan"))

    if major_is_y:
        sel_region = roi[a:b, :] if b > a else roi
    else:
        sel_region = roi[:, a:b] if b > a else roi
    sel_sum = float(sel_region.sum())
    total_sum = float(after_arr.sum())
    eff_obs = sel_sum / total_sum if total_sum > 0 else float("nan")

    W_um = nx * cam_pitch_um
    H_um = ny * cam_pitch_um
    lo_um = lo * cam_pitch_um
    hi_um = hi * cam_pitch_um

    return {
        "dark_corrected": dark_corrected,
        "cam_pitch_um": cam_pitch_um,
        "roi": roi,
        "after_corrected": after_arr,
        "roi_bbox_y0y1x0x1": [int(y0), int(y1), int(x0), int(x1)],
        "roi_shape_yx": [int(ny), int(nx)],
        "roi_size_um_yx": [round(H_um, 3), round(W_um, 3)],
        "major_is_y": bool(major_is_y),
        "axis_label": axis_label,
        "profile": profile,
        "profile_length_px": int(len(profile)),
        "profile_slice_minor_px": [int(lo), int(hi)],
        "profile_slice_minor_um": [round(lo_um, 3), round(hi_um, 3)],
        "flat_top_center_px": float(center_px),
        "flat_top_half_width_px": float(half_width_px),
        "flat_top_bounds_px": [int(a), int(b)],
        "flat_top_width_um": round((b - a) * cam_pitch_um, 3),
        "flat_top_mean_intensity": mean_val,
        "flat_profile": flat,
        "rms_percent": rms_pct,
        "pk_pk_percent": ppk_pct,
        "selected_region_intensity": sel_sum,
        "total_camera_intensity": total_sum,
        "efficiency_observed": eff_obs,
    }


def plot_sheet_analysis(result: dict, figsize=(10, 8)):
    """Render the standard 2-panel ROI-heatmap + profile figure inline.

    Convenience for notebook cells: takes a result dict from
    :func:`analyze_sheet` and produces the same figure the legacy
    ``analysis_sheet.py`` saved to PNG, but returned as a Figure (no save).
    """
    import matplotlib.pyplot as plt

    roi = result["roi"]
    cam_pitch_um = result["cam_pitch_um"]
    profile = result["profile"]
    ny, nx = roi.shape
    W_um = nx * cam_pitch_um
    H_um = ny * cam_pitch_um
    lo, hi = result["profile_slice_minor_px"]
    lo_um, hi_um = result["profile_slice_minor_um"]
    a, b = result["flat_top_bounds_px"]
    mean_val = result["flat_top_mean_intensity"]
    rms_pct = result["rms_percent"]
    ppk_pct = result["pk_pk_percent"]
    eff_pct = 100.0 * result["efficiency_observed"]
    major_is_y = result["major_is_y"]

    fig, (ax_img, ax_prof) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1.4, 1.0]},
    )

    im = ax_img.imshow(
        roi, cmap="jet", aspect="auto",
        extent=(0.0, W_um, H_um, 0.0),
    )
    if major_is_y:
        ax_img.axvline(lo_um, color="cyan", lw=1.3, ls="--")
        ax_img.axvline(hi_um, color="cyan", lw=1.3, ls="--",
                       label=f"3-col strip x∈[{lo_um:.1f}, {hi_um:.1f}] µm")
    else:
        ax_img.axhline(lo_um, color="cyan", lw=1.3, ls="--")
        ax_img.axhline(hi_um, color="cyan", lw=1.3, ls="--",
                       label=f"3-row strip y∈[{lo_um:.1f}, {hi_um:.1f}] µm")
    ax_img.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax_img.set_xlabel("x (µm)")
    ax_img.set_ylabel("y (µm)")
    ax_img.set_title(
        f"RMS: {rms_pct:.2f}%, Pk-Pk: {ppk_pct:.2f}%, Eff: {eff_pct:.2f}%"
    )
    fig.colorbar(im, ax=ax_img, fraction=0.035, pad=0.02)

    u_um = np.arange(len(profile)) * cam_pitch_um
    a_um = a * cam_pitch_um
    b_um = (b - 1) * cam_pitch_um
    ax_prof.plot(u_um, profile, color="tab:blue", lw=2.0,
                 label="Profile (3-row mean)")
    ax_prof.hlines(mean_val, a_um, b_um, colors="red",
                   linestyles="--", lw=1.4, label="Mean")
    ax_prof.axvline(a_um, color="gray", ls="--", lw=1.0)
    ax_prof.axvline(b_um, color="gray", ls="--", lw=1.0,
                    label="Pk-Pk Range")
    ax_prof.text(
        0.45, 0.15,
        f"RMS: {rms_pct:.4f}%\nPk-Pk: {ppk_pct:.4f}%\nEff: {eff_pct:.4f}%",
        transform=ax_prof.transAxes,
        ha="left", va="top", color="red", fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="red", lw=1.2),
    )
    ax_prof.set_xlim(0.0, (len(profile) - 1) * cam_pitch_um)
    ax_prof.set_ylabel("Intensity")
    ax_prof.set_xlabel(result["axis_label"])
    ax_prof.grid(True, alpha=0.3)
    ax_prof.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig
