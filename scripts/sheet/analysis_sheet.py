"""Plot light-sheet uniformity from a single ``after`` camera capture.

Input: the ``_after.bmp`` frame produced by running::

    ./push_run.sh payload/sheet/testfile_sheet_payload.npz

The script auto-detects the elongated sheet-like region inside the
frame (rejecting the compact zero-order blob and background) and emits
a 2-panel figure in the style of ``scripts/sheet/demo.png``:

    top:    cropped 2D heatmap of the detected sheet ROI (jet).
    bottom: 3-row-mean profile along the sheet axis with the flat-top
            mean (red dashed), flat-top boundaries (gray dashed), and
            an RMS / peak-to-peak stat box.

Usage::

    uv run python scripts/sheet/analysis_sheet.py
    uv run python scripts/sheet/analysis_sheet.py --after path/to/after.bmp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import label, median_filter
from scipy.optimize import curve_fit

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Allied Vision Alvium 1800 U-1240m (Sony IMX226), 4024 × 3036 px, 1.85 µm
# pixel pitch.  Override via ``--cam-pitch-um`` if the camera changes.
CAM_PITCH_UM_DEFAULT = 1.85


def _load_bmp(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def _detect_sheet_bbox(
    after: np.ndarray, threshold_frac: float = 0.30, pad: int = 6,
) -> tuple[tuple[int, int, int, int], bool]:
    """Find the elongated sheet-like region in *after*.

    - Median-filter + background-subtract.
    - Threshold at ``threshold_frac`` of the peak to get bright blobs.
    - Connected-component label; pick the blob with the largest
      major-axis extent (i.e. the most elongated / sheet-like).  The
      compact zero-order blob loses to a proper light sheet.
    - Pad the bbox.  Return (bbox, major_is_y).
    """
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
    """Fit top-hat + Gaussian-edge; return (center_px, half_width_px).

    Falls back to the half-max bracket if the fit fails.
    """
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


def analyze(after_path, plot_path=None, result_path=None,
            cam_pitch_um: float = CAM_PITCH_UM_DEFAULT,
            before_path=None,
            flat_a: int = 50, flat_b: int = 200) -> dict:
    after = _load_bmp(after_path)
    dark_corrected = False
    if before_path is not None and Path(before_path).is_file():
        # Dark/flat-field correction: SLM-blank "before" frame carries the
        # camera's column FPN + dark offset at the same exposure.  Subtracting
        # it from the "after" frame kills the 1-px-period sawtooth that the
        # IMX226's column amplifiers imprint on every row.
        before = _load_bmp(before_path)
        if before.shape == after.shape:
            after = np.clip(after - before, 0.0, None)
            dark_corrected = True
    (y0, y1, x0, x1), major_is_y = _detect_sheet_bbox(after)
    roi = after[y0:y1, x0:x1]
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
        a = round(max(flat_a/cam_pitch_um,0))
        b = round(min(flat_b/cam_pitch_um, len(profile)))    
    else:
        a = max(int(round(center_px - half_width_px)), 0)
        b = min(int(round(center_px + half_width_px)), len(profile))
    flat = profile[a:b] if b > a else profile

    mean_val = float(flat.mean()) if flat.size else 0.0
    rms_pct = 100.0 * float(flat.std()) / mean_val if mean_val > 0 else float("nan")
    ppk_pct = (100.0 * float(flat.max() - flat.min()) / mean_val
               if mean_val > 0 else float("nan"))

    W_um = nx * cam_pitch_um
    H_um = ny * cam_pitch_um
    lo_um = lo * cam_pitch_um
    hi_um = hi * cam_pitch_um

    fig, (ax_img, ax_prof) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1.4, 1.0]},
    )

    # Top: 2D heatmap on real camera µm axes (origin top-left, natural camera
    # orientation).  Cyan dashed lines mark the 3-row strip used for the
    # bottom profile — the colour contrasts strongly against ``jet``.
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
    ax_img.set_title(f"RMS: {rms_pct:.2f}%, Pk-Pk: {ppk_pct:.2f}%")
    fig.colorbar(im, ax=ax_img, fraction=0.035, pad=0.02)

    # Bottom: profile on the same µm scale as the top panel's along-sheet axis.
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
        f"RMS: {rms_pct:.4f}%\nPk-Pk: {ppk_pct:.4f}%",
        transform=ax_prof.transAxes,
        ha="left", va="top", color="red", fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="red", lw=1.2),
    )

    # Align bottom x-range with the corresponding top-panel axis (0 → W_um
    # along x, or 0 → H_um along y).  So the two panels share the same
    # along-sheet scale and the cyan markers in the top panel visually line
    # up with the intensity profile below.
    ax_prof.set_xlim(0.0, (len(profile) - 1) * cam_pitch_um)
    ax_prof.set_ylabel("Intensity")
    ax_prof.set_xlabel(axis_label)
    ax_prof.grid(True, alpha=0.3)
    ax_prof.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    if plot_path is None:
        plot_path = Path(after_path).with_name(Path(after_path).stem + "_analysis.png")
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    result = {
        "after_path": str(after_path),
        "before_path": str(before_path) if before_path is not None else None,
        "dark_corrected": dark_corrected,
        "cam_pitch_um": cam_pitch_um,
        "roi_bbox_y0y1x0x1": [int(y0), int(y1), int(x0), int(x1)],
        "roi_shape_yx": [int(ny), int(nx)],
        "roi_size_um_yx": [round(H_um, 3), round(W_um, 3)],
        "major_is_y": bool(major_is_y),
        "profile_length_px": int(len(profile)),
        "profile_values": [float(v) for v in profile],
        "profile_slice_minor_px": [int(lo), int(hi)],
        "profile_slice_minor_um": [round(lo_um, 3), round(hi_um, 3)],
        "flat_top_center_px": float(center_px),
        "flat_top_half_width_px": float(half_width_px),
        "flat_top_bounds_px": [int(a), int(b)],
        "flat_top_width_um": round((b - a) * cam_pitch_um, 3),
        "flat_top_mean_intensity": mean_val,
        "flat_profile": [float(v) for v in flat],
        "rms_percent": rms_pct,
        "pk_pk_percent": ppk_pct,
        "plot_path": str(plot_path),
    }
    if result_path is not None:
        result_path = Path(result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        result["result_path"] = str(result_path)

    print(f"[ROI]  bbox=({y0},{y1},{x0},{x1}) shape=({ny},{nx}) "
          f"major_is_y={major_is_y}  "
          f"({H_um:.1f} × {W_um:.1f} µm @ {cam_pitch_um:.2f} µm/px)"
          f"{'  [dark-corrected]' if dark_corrected else ''}")
    print(f"[FIT]  center={center_px:.2f}px  half_width={half_width_px:.2f}px")
    print(f"[FLAT] [{a},{b})  width={b-a}px "
          f"({(b-a)*cam_pitch_um:.1f} µm)  mean={mean_val:.1f}")
    print(f"[METR] RMS={rms_pct:.4f}%   Pk-Pk={ppk_pct:.4f}%")
    print(f"[SAVE] {plot_path}")
    if result_path is not None:
        print(f"[SAVE] {result_path}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--after", default="data/sheet/testfile_sheet_after.bmp")
    ap.add_argument("--plot", default="scripts/sheet/analysis_sheet.png")
    ap.add_argument("--result", default="scripts/sheet/analysis_sheet_result.json")
    ap.add_argument(
        "--cam-pitch-um", type=float, default=CAM_PITCH_UM_DEFAULT,
        help=f"Camera pixel pitch in µm (default {CAM_PITCH_UM_DEFAULT} — "
             "Alvium 1800 U-1240m, IMX226).",
    )
    ap.add_argument(
        "--before", default=None,
        help="Optional SLM-blank BMP for dark-frame / column-FPN subtraction. "
             "If set and same shape as --after, analysis runs on "
             "clip(after - before, 0) instead of after alone.",
    )

    ap.add_argument(
        "--flat-a", type=int, default=50,
        help="Manually select the start region index (mu m) for flat-top region counted for rms&pk-pk"
    )
    ap.add_argument(
        "--flat-b", type=int, default=200,
        help="Manually select the end region index (mu m) for flat-top region counted for rms&pk-pk"
    )
    args = ap.parse_args()

    if not Path(args.after).is_file():
        print(f"ERROR: {args.after} not found", file=sys.stderr)
        sys.exit(1)
    analyze(args.after, args.plot, args.result,
            cam_pitch_um=args.cam_pitch_um, before_path=args.before,
            flat_a=args.flat_a, flat_b=args.flat_b)


if __name__ == "__main__":
    main()
