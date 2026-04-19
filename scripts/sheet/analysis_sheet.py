"""Benchmark a single light-sheet camera capture.

Loads before/after frames from the Windows runner, auto-detects the
sheet ROI (per issue #17: start from the brightest point of *before*
and walk along x/y until intensity drops), rebuilds the expected
light-sheet reference on that ROI, and computes efficiency +
shape-fidelity metrics.

Usage::

    uv run python scripts/sheet/analysis_sheet.py \\
        --after data/sheet/testfile_sheet_after.bmp \\
        --before data/sheet/testfile_sheet_before.bmp \\
        --params payload/sheet/testfile_sheet_params.json

Also importable as a library (for sweep aggregation):

    from scripts.sheet.analysis_sheet import analyze_capture
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import matplotlib


def _load_capture_bmp(path) -> np.ndarray:
    """Load an 8-bit grayscale BMP capture into a float64 array."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.optimize import curve_fit

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure `scripts/` is importable for camera_roi fallback.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from camera_roi import find_target_center  # noqa: E402
except ImportError:
    find_target_center = None  # fallback path below guards on None
from slm.targets import light_sheet  # noqa: E402


def _walk_until_drop(
    row_or_col: np.ndarray, start: int, direction: int, bg_std: float,
) -> int:
    """Walk along a 1D array from *start* in *direction* (+1/-1); return
    the number of pixels traversed before intensity drops below the
    threshold for 3 consecutive pixels.
    """
    smooth = uniform_filter1d(row_or_col.astype(np.float64), 3)
    n = len(smooth)
    peak_along_walk = smooth[start] if 0 <= start < n else 0.0
    threshold = max(0.1 * peak_along_walk, 3.0 * bg_std)

    below_run = 0
    i = start
    steps = 0
    while 0 <= i < n:
        val = smooth[i]
        if val > peak_along_walk:
            peak_along_walk = val
            threshold = max(0.1 * peak_along_walk, 3.0 * bg_std)
        if val < threshold:
            below_run += 1
            if below_run >= 3:
                return max(steps - 2, 0)
        else:
            below_run = 0
        i += direction
        steps += 1
    return steps


def detect_roi(
    before: np.ndarray, after: np.ndarray, max_walk: int = 400,
    peak: str = "before",
) -> dict[str, Any]:
    """ROI detection: pick a starting peak, then walk outward in +/-x/y
    through *after* until intensity drops.

    peak="before" (default, legacy): start at the brightest point of
    *before* (= zero-order, since SLM is blank).  Correct when the
    target overlaps the zero-order.

    peak="signal": start at the brightest point of median(after-before).
    Correct when the target is shifted away from the zero-order (issue
    #19 sweep uses target_shift_fpx=30 → sheet is ~475 um off-center).
    """
    if peak == "signal":
        sig_s = median_filter(
            np.clip(after.astype(np.float64) - before.astype(np.float64), 0.0, None),
            size=3,
        )
        cy, cx = np.unravel_index(int(np.argmax(sig_s)), sig_s.shape)
    else:
        before_s = median_filter(before.astype(np.float64), size=3)
        cy, cx = np.unravel_index(int(np.argmax(before_s)), before_s.shape)

    after_f = after.astype(np.float64)
    H, W = after_f.shape

    yy, xx = np.ogrid[:H, :W]
    far_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) > (max_walk ** 2)
    bg_vals = after_f[far_mask]
    bg_median = float(np.median(bg_vals)) if bg_vals.size else 0.0
    bg_std = float(np.std(bg_vals)) if bg_vals.size else 1.0

    after_sub = after_f - bg_median

    row = after_sub[cy, :]
    col = after_sub[:, cx]

    max_x_plus = min(max_walk, W - 1 - cx)
    max_x_minus = min(max_walk, cx)
    max_y_plus = min(max_walk, H - 1 - cy)
    max_y_minus = min(max_walk, cy)

    dx_plus = _walk_until_drop(row[cx : cx + max_x_plus + 1], 0, +1, bg_std)
    dx_minus = _walk_until_drop(row[cx - max_x_minus : cx + 1][::-1], 0, +1, bg_std)
    dy_plus = _walk_until_drop(col[cy : cy + max_y_plus + 1], 0, +1, bg_std)
    dy_minus = _walk_until_drop(col[cy - max_y_minus : cy + 1][::-1], 0, +1, bg_std)

    dx = max(dx_plus, dx_minus, 2)
    dy = max(dy_plus, dy_minus, 2)

    warning = None
    if dx < 2 and dy < 2:
        warning = (
            "walk from _before peak through _after stayed below "
            f"background (dx={dx}, dy={dy}); falling back to after-before cluster"
        )
        if find_target_center is None:
            warning += " | camera_roi unavailable; using initial (cy,cx)"
        else:
            cy_fb, cx_fb, _, _ = find_target_center(after, before)
            cy, cx = cy_fb, cx_fb
        row = after_sub[cy, :]
        col = after_sub[:, cx]
        max_x_plus = min(max_walk, W - 1 - cx)
        max_x_minus = min(max_walk, cx)
        max_y_plus = min(max_walk, H - 1 - cy)
        max_y_minus = min(max_walk, cy)
        dx_plus = _walk_until_drop(row[cx : cx + max_x_plus + 1], 0, +1, bg_std)
        dx_minus = _walk_until_drop(row[cx - max_x_minus : cx + 1][::-1], 0, +1, bg_std)
        dy_plus = _walk_until_drop(col[cy : cy + max_y_plus + 1], 0, +1, bg_std)
        dy_minus = _walk_until_drop(col[cy - max_y_minus : cy + 1][::-1], 0, +1, bg_std)
        dx = max(dx_plus, dx_minus, 5)
        dy = max(dy_plus, dy_minus, 5)

    y0 = max(cy - dy, 0)
    y1 = min(cy + dy, H)
    x0 = max(cx - dx, 0)
    x1 = min(cx + dx, W)

    return {
        "cy": int(cy), "cx": int(cx),
        "dx": int(dx), "dy": int(dy),
        "dx_plus": int(dx_plus), "dx_minus": int(dx_minus),
        "dy_plus": int(dy_plus), "dy_minus": int(dy_minus),
        "bbox": (int(y0), int(y1), int(x0), int(x1)),
        "bg_median": bg_median,
        "bg_std": bg_std,
        "warning": warning,
    }


def _top_hat_gaussian_edge(u: np.ndarray, amp: float, center: float,
                           half_width: float, edge_sigma: float,
                           baseline: float) -> np.ndarray:
    """Line-axis model: flat-top of width 2*half_width with Gaussian edges."""
    dist_outside = np.maximum(0.0, np.abs(u - center) - half_width)
    return baseline + amp * np.exp(-(dist_outside ** 2) / (2.0 * edge_sigma ** 2))


def _gaussian(v: np.ndarray, amp: float, center: float,
              sigma: float, baseline: float) -> np.ndarray:
    return baseline + amp * np.exp(-((v - center) ** 2) / (2.0 * sigma ** 2))


def _fit_profiles(roi: np.ndarray, angle_rad: float) -> dict[str, Any]:
    """Project *roi* onto the sheet's major/minor axes (determined by
    *angle_rad*; 0 == horizontal flat, pi/2 == vertical flat) and fit
    flat-width + edge sigma + perpendicular Gaussian sigma.
    """
    major_is_y = abs(np.sin(angle_rad)) > abs(np.cos(angle_rad))
    if major_is_y:
        major_proj = roi.sum(axis=1)
        minor_proj = roi.sum(axis=0)
    else:
        major_proj = roi.sum(axis=0)
        minor_proj = roi.sum(axis=1)

    u = np.arange(len(major_proj), dtype=np.float64)
    v = np.arange(len(minor_proj), dtype=np.float64)

    major_peak = float(major_proj.max())
    major_base = float(major_proj.min())
    major_c0 = float(np.argmax(major_proj))
    above_half = major_proj > (major_base + 0.5 * (major_peak - major_base))
    major_hw0 = max(float(above_half.sum()) / 2.0, 1.0)

    minor_peak = float(minor_proj.max())
    minor_base = float(minor_proj.min())
    minor_c0 = float(np.argmax(minor_proj))
    minor_above = minor_proj > (minor_base + 0.5 * (minor_peak - minor_base))
    minor_sigma0 = max(float(minor_above.sum()) / 2.355, 1.0)

    fit: dict[str, Any] = {
        "major_is_y": bool(major_is_y),
        "major_length": int(len(major_proj)),
        "minor_length": int(len(minor_proj)),
    }
    major_len = float(len(major_proj))
    minor_len = float(len(minor_proj))
    try:
        popt_major, _ = curve_fit(
            _top_hat_gaussian_edge, u, major_proj,
            p0=[max(major_peak - major_base, 1.0), major_c0, major_hw0, 2.0, major_base],
            bounds=(
                [0.0, 0.0, 0.5, 0.1, -np.inf],
                [np.inf, major_len, major_len, major_len, np.inf],
            ),
            maxfev=10000,
        )
        amp, center, half_width, edge_sigma, baseline = popt_major
        fit["measured_flat_width_px"] = float(2.0 * abs(half_width))
        fit["measured_edge_sigma_px"] = float(abs(edge_sigma))
        fit["major_fit_amp"] = float(amp)
        fit["major_fit_center"] = float(center)
        fit["major_fit_baseline"] = float(baseline)
    except Exception as e:
        fit["measured_flat_width_px"] = float("nan")
        fit["measured_edge_sigma_px"] = float("nan")
        fit["major_fit_error"] = str(e)

    try:
        popt_minor, _ = curve_fit(
            _gaussian, v, minor_proj,
            p0=[max(minor_peak - minor_base, 1.0), minor_c0, minor_sigma0, minor_base],
            bounds=(
                [0.0, 0.0, 0.3, -np.inf],
                [np.inf, minor_len, minor_len, np.inf],
            ),
            maxfev=10000,
        )
        amp, center, sigma, baseline = popt_minor
        fit["measured_gauss_sigma_px"] = float(abs(sigma))
        fit["minor_fit_amp"] = float(amp)
        fit["minor_fit_center"] = float(center)
        fit["minor_fit_baseline"] = float(baseline)
    except Exception as e:
        fit["measured_gauss_sigma_px"] = float("nan")
        fit["minor_fit_error"] = str(e)

    fit["major_proj"] = major_proj.tolist()
    fit["minor_proj"] = minor_proj.tolist()
    return fit


def _build_reference(
    roi_shape: tuple[int, int], params: dict, fit: dict,
) -> np.ndarray:
    """Build the expected |light_sheet|^2 reference on the ROI grid."""
    flat_width = float(params["sheet_flat_width_px"])
    gaussian_sigma = float(params["sheet_gaussian_sigma_px"])
    edge_sigma = float(params.get("sheet_edge_sigma_px", 0.0))
    angle = float(params.get("sheet_angle_rad", 0.0))
    ny, nx = roi_shape
    center = (
        float(fit.get("major_fit_center", (ny - 1) / 2.0)) if fit.get("major_is_y") else (ny - 1) / 2.0,
        float(fit.get("major_fit_center", (nx - 1) / 2.0)) if not fit.get("major_is_y") else (nx - 1) / 2.0,
    )
    field = light_sheet(
        (ny, nx),
        flat_width=flat_width,
        gaussian_sigma=gaussian_sigma,
        angle=angle,
        center=center,
        edge_sigma=edge_sigma,
    )
    intensity = np.abs(field) ** 2
    total = intensity.sum()
    if total > 0:
        intensity /= total
    return intensity


def _intensity_fidelity(measured: np.ndarray, reference: np.ndarray) -> dict:
    """Two shape-fidelity flavors on intensity arrays."""
    m = measured.astype(np.float64).clip(min=0)
    r = reference.astype(np.float64).clip(min=0)
    m_sum = m.sum()
    r_sum = r.sum()
    if m_sum <= 0 or r_sum <= 0:
        return {"fidelity_corr": 0.0, "fidelity_overlap": 0.0}

    mn = m / m_sum
    rn = r / r_sum

    mm = mn - mn.mean()
    rr = rn - rn.mean()
    denom = float(np.sqrt((mm * mm).sum()) * np.sqrt((rr * rr).sum()))
    corr = float((mm * rr).sum() / denom) if denom > 0 else 0.0

    overlap = float((np.sqrt(mn * rn).sum()) ** 2)

    return {"fidelity_corr": corr, "fidelity_overlap": overlap}


def analyze_capture(
    after_path: str | Path,
    before_path: str | Path,
    params_path: str | Path | None = None,
    preview_path: str | Path | None = None,
    peak: str = "before",
) -> dict[str, Any]:
    """Full analysis pipeline. Returns a result dict suitable for JSON."""
    after = _load_capture_bmp(after_path)
    before = _load_capture_bmp(before_path)
    if params_path is not None:
        with open(params_path) as f:
            params = json.load(f)
    else:
        params = {}

    roi_info = detect_roi(before, after, peak=peak)
    y0, y1, x0, x1 = roi_info["bbox"]
    signal = after - before
    roi = np.clip(signal[y0:y1, x0:x1], 0.0, None)

    angle_rad = float(params.get("sheet_angle_rad", 0.0))
    fit = _fit_profiles(roi, angle_rad)

    reference = _build_reference(roi.shape, params, fit) if params else None
    metrics: dict[str, Any] = {}
    if reference is not None:
        metrics.update(_intensity_fidelity(roi, reference))
    else:
        metrics["fidelity_corr"] = float("nan")
        metrics["fidelity_overlap"] = float("nan")

    signal_pos = np.clip(signal, 0.0, None)
    total_signal = float(signal_pos.sum())
    roi_sum = float(roi.sum())
    metrics["efficiency"] = float(roi_sum / total_signal) if total_signal > 0 else 0.0
    metrics["total_intensity"] = roi_sum
    metrics["peak_intensity"] = float(roi.max())

    if not np.isnan(fit.get("measured_flat_width_px", float("nan"))):
        half = fit["measured_flat_width_px"] / 2.0
        center = fit.get("major_fit_center", roi.shape[0 if fit["major_is_y"] else 1] / 2.0)
        if fit["major_is_y"]:
            a = max(int(round(center - half)), 0)
            b = min(int(round(center + half)), roi.shape[0])
            plateau = roi[a:b, :].sum(axis=1) if b > a else np.array([])
        else:
            a = max(int(round(center - half)), 0)
            b = min(int(round(center + half)), roi.shape[1])
            plateau = roi[:, a:b].sum(axis=0) if b > a else np.array([])
        if plateau.size and plateau.mean() > 0:
            metrics["flat_region_rms"] = float(plateau.std() / plateau.mean())
        else:
            metrics["flat_region_rms"] = float("nan")
    else:
        metrics["flat_region_rms"] = float("nan")

    result: dict[str, Any] = {
        "after_path": str(after_path),
        "before_path": str(before_path),
        "params_path": str(params_path) if params_path else None,
        "roi": {
            "cy": roi_info["cy"], "cx": roi_info["cx"],
            "dx": roi_info["dx"], "dy": roi_info["dy"],
            "bbox": list(roi_info["bbox"]),
            "warning": roi_info["warning"],
        },
        "fit": {k: v for k, v in fit.items() if k not in ("major_proj", "minor_proj")},
        "metrics": metrics,
        "input_params": params,
    }

    if preview_path is not None:
        _save_preview(
            Path(preview_path), signal, roi_info, roi, fit, reference, metrics,
        )
        result["preview_path"] = str(preview_path)

    return result


def _save_preview(path: Path, signal: np.ndarray, roi_info: dict,
                  roi: np.ndarray, fit: dict, reference, metrics: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    vmax = float(np.percentile(signal, 99.9))
    ax = axes[0]
    ax.imshow(signal, cmap="hot", vmin=0, vmax=max(vmax, 1e-6))
    y0, y1, x0, x1 = roi_info["bbox"]
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "c-", lw=1.2)
    ax.plot(roi_info["cx"], roi_info["cy"], "c+", ms=12, mew=2)
    ax.set_title(
        f"signal = after - before\nc=({roi_info['cy']}, {roi_info['cx']})  "
        f"ROI=({y1-y0}x{x1-x0})"
    )
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    u = np.arange(len(fit["major_proj"]))
    ax.plot(u, fit["major_proj"], "k.", ms=3, label="data")
    if not np.isnan(fit.get("measured_flat_width_px", float("nan"))):
        model = _top_hat_gaussian_edge(
            u, fit["major_fit_amp"], fit["major_fit_center"],
            fit["measured_flat_width_px"] / 2.0, fit["measured_edge_sigma_px"],
            fit["major_fit_baseline"],
        )
        ax.plot(u, model, "r-", lw=1.2, label="fit")
    ax.set_title(
        f"major-axis projection\nflat={fit.get('measured_flat_width_px', float('nan')):.1f}px  "
        f"edge_sigma={fit.get('measured_edge_sigma_px', float('nan')):.1f}px"
    )
    ax.set_xlabel("pixel along sheet")
    ax.legend(fontsize=8)

    ax = axes[2]
    v = np.arange(len(fit["minor_proj"]))
    ax.plot(v, fit["minor_proj"], "k.", ms=3, label="data")
    if not np.isnan(fit.get("measured_gauss_sigma_px", float("nan"))):
        model = _gaussian(
            v, fit["minor_fit_amp"], fit["minor_fit_center"],
            fit["measured_gauss_sigma_px"], fit["minor_fit_baseline"],
        )
        ax.plot(v, model, "r-", lw=1.2, label="fit")
    ax.set_title(
        f"minor-axis projection\ngauss_sigma={fit.get('measured_gauss_sigma_px', float('nan')):.1f}px\n"
        f"eff={metrics.get('efficiency', 0)*100:.1f}%  "
        f"fid_corr={metrics.get('fidelity_corr', 0):.3f}"
    )
    ax.set_xlabel("pixel perpendicular")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark one light-sheet camera capture."
    )
    ap.add_argument("--after", default="data/sheet/testfile_sheet_after.bmp")
    ap.add_argument("--before", default="data/sheet/testfile_sheet_before.bmp")
    ap.add_argument("--params", default="payload/sheet/testfile_sheet_params.json")
    ap.add_argument("--out", default="scripts/sheet/analysis_sheet_result.json")
    ap.add_argument("--preview", default="scripts/sheet/analysis_sheet_preview.png")
    ap.add_argument("--peak", choices=["before", "signal"], default="before",
                    help="ROI start: 'before' (zero-order, target overlaps) or "
                         "'signal' (after-before peak, target shifted away).")
    args = ap.parse_args()

    for p in (args.after, args.before):
        if not Path(p).is_file():
            print(f"ERROR: {p} not found", file=sys.stderr)
            sys.exit(1)

    params_path = args.params if Path(args.params).is_file() else None
    if params_path is None:
        warnings.warn(f"{args.params} not found; skipping shape fidelity")

    result = analyze_capture(
        args.after, args.before, params_path, args.preview, peak=args.peak,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    m = result["metrics"]
    print(f"[ROI]  c=({result['roi']['cy']},{result['roi']['cx']})  "
          f"dx={result['roi']['dx']}  dy={result['roi']['dy']}")
    if result["roi"]["warning"]:
        print(f"[WARN] {result['roi']['warning']}")
    print(f"[FIT]  flat={result['fit'].get('measured_flat_width_px', float('nan')):.2f}px  "
          f"gauss_sigma={result['fit'].get('measured_gauss_sigma_px', float('nan')):.2f}px  "
          f"edge_sigma={result['fit'].get('measured_edge_sigma_px', float('nan')):.2f}px")
    print(f"[METR] eff={m['efficiency']*100:.2f}%  "
          f"fid_corr={m['fidelity_corr']:.4f}  "
          f"fid_overlap={m['fidelity_overlap']:.4f}  "
          f"flat_rms={m['flat_region_rms']:.4f}")
    print(f"[SAVE] {args.out}")
    print(f"[SAVE] {args.preview}")


if __name__ == "__main__":
    main()
