"""Compare simulation vs camera intensity non-uniformity for the optimized top-hat.

Reads simulation data from the payload .npz and camera data from data/.
Produces a comparison figure and prints metrics.

Usage::

    uv run python scripts/analyze_tophat_uniformity.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
from PIL import Image
import matplotlib


def _load_capture_bmp(path) -> np.ndarray:
    """Load an 8-bit grayscale BMP capture into a float64 array."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PREFIX = "testfile_tophat_optimized"
PAYLOAD = f"scripts/{PREFIX}_payload.npz"
CAMERA_AFTER = f"data/{PREFIX}_after.bmp"
CAMERA_BEFORE = f"data/{PREFIX}_before.bmp"
RUN_JSON = f"data/{PREFIX}_run.json"
PARAMS_JSON = f"scripts/{PREFIX}_params.json"
OUTPUT_PDF = "scripts/tophat_uniformity_analysis.pdf"

# Camera pixel pitch in µm
CAMERA_PIXEL_UM = 1.85


def find_target_center(after, before):
    """Find the target spot center (not zero-order).

    Strategy: the target is the brightest cluster in 'after' that is NOT
    at the zero-order location.  Zero-order is the brightest spot in 'before'.
    """
    # Zero-order center (brightest in before)
    flat_b = before.flatten()
    top_b = np.argpartition(flat_b, -50)[-50:]
    rows_b, cols_b = np.unravel_index(top_b, before.shape)
    zo_cy, zo_cx = int(rows_b.mean()), int(cols_b.mean())

    # Target center: brightest pixels in after that are >100 cam-px from zero-order
    signal = after.astype(np.float64) - before.astype(np.float64)
    Y, X = np.ogrid[:signal.shape[0], :signal.shape[1]]
    R_zo = np.sqrt((X - zo_cx) ** 2 + (Y - zo_cy) ** 2)
    # Mask out zero-order vicinity
    signal_masked = np.where(R_zo > 100, signal, 0)
    flat = signal_masked.flatten()
    top_idx = np.argpartition(flat, -100)[-100:]
    rows, cols = np.unravel_index(top_idx, signal.shape)
    cy, cx = int(rows.mean()), int(cols.mean())
    return cy, cx, zo_cy, zo_cx


def radial_profile(img, center, max_r):
    """Azimuthally averaged radial intensity profile."""
    cy, cx = center
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r_int = R.astype(int)
    r_max = min(max_r, r_int.max())
    profile = np.zeros(r_max + 1)
    counts = np.zeros(r_max + 1)
    valid = r_int <= r_max
    np.add.at(profile, r_int[valid], img[valid])
    np.add.at(counts, r_int[valid], np.ones_like(img[valid]))
    counts[counts == 0] = 1
    return profile / counts


def main():
    # --- Load simulation data ---
    data = np.load(PAYLOAD)
    E_out = data["E_out"]          # complex128, (1024, 1024)
    targetAmp = data["targetAmp"]  # complex128, (1024, 1024)
    target_mask = data["target_mask"]  # bool/uint8, (1024, 1024)

    params = json.load(open(PARAMS_JSON))
    focal_pitch = params["focal_pitch_x_um_per_px"]
    tophat_radius_px = params["tophat_radius_px"]

    # Simulation intensity inside disc (where target = 1)
    I_sim = np.abs(E_out) ** 2
    disc_mask = target_mask > 0
    I_sim_disc = I_sim[disc_mask]

    unif_sim = float(np.std(I_sim_disc) / np.mean(I_sim_disc))
    print(f"[SIM] Pixels in disc: {disc_mask.sum()}")
    print(f"[SIM] I_disc: mean={np.mean(I_sim_disc):.6e}  std={np.std(I_sim_disc):.6e}")
    print(f"[SIM] Intensity uniformity (std/mean) = {unif_sim:.4f} ({unif_sim*100:.2f}%)")

    # --- Load camera data ---
    I_after = _load_capture_bmp(CAMERA_AFTER)
    I_before = _load_capture_bmp(CAMERA_BEFORE)
    I_signal = I_after - I_before

    run_meta = json.load(open(RUN_JSON))
    print(f"\n[CAM] Shape: {I_after.shape}  after_max={run_meta['after']['max']}")
    print(f"[CAM] Background: mean={run_meta['before']['mean']:.2f}  "
          f"std={run_meta['before']['std']:.4f}")

    # Find target center (away from zero-order)
    cy_cam, cx_cam, zo_cy, zo_cx = find_target_center(I_after, I_before)
    zo_dist = np.sqrt((cy_cam - zo_cy) ** 2 + (cx_cam - zo_cx) ** 2)
    print(f"[CAM] Zero-order center: ({zo_cy}, {zo_cx})")
    print(f"[CAM] Target center:     ({cy_cam}, {cx_cam})")
    print(f"[CAM] Separation: {zo_dist:.0f} cam-px = {zo_dist * CAMERA_PIXEL_UM:.0f} µm")

    # Expected disc radius on camera: tophat_radius is in focal-plane pixels,
    # and the optical relay maps ~1 focal pixel to ~1 camera pixel.
    disc_radius_cam = tophat_radius_px  # ~10 camera pixels
    disc_edge_um = disc_radius_cam * CAMERA_PIXEL_UM
    print(f"[CAM] Expected disc radius: {disc_radius_cam:.0f} camera pixels ({disc_edge_um:.0f} µm)")

    # --- Compute uniformity at multiple ROI radii ---
    Y_cam, X_cam = np.ogrid[:I_signal.shape[0], :I_signal.shape[1]]
    R_cam = np.sqrt((X_cam - cx_cam) ** 2 + (Y_cam - cy_cam) ** 2)

    print(f"\n[CAM] Uniformity vs ROI radius:")
    print(f"  {'r (cam-px)':>10}  {'r (µm)':>8}  {'r/r_disc':>8}  {'mean':>8}  "
          f"{'std':>8}  {'std/mean':>8}  {'npix':>6}")
    print(f"  {'-'*66}")
    roi_results = []
    for r in [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]:
        mask = R_cam <= r
        vals = I_signal[mask]
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 5:
            continue
        m = np.mean(vals_pos)
        s = np.std(vals_pos)
        u = s / m if m > 0 else float('inf')
        r_um = r * CAMERA_PIXEL_UM
        ratio = r / disc_radius_cam
        roi_results.append((r, r_um, ratio, m, s, u, len(vals_pos)))
        print(f"  {r:10d}  {r_um:8.1f}  {ratio:8.2f}  {m:8.2f}  "
              f"{s:8.2f}  {u:8.4f}  {len(vals_pos):6d}")

    # Use the expected disc radius for the "correct" comparison
    cam_disc_mask = R_cam <= disc_radius_cam
    I_cam_disc = I_signal[cam_disc_mask]
    I_cam_disc_pos = I_cam_disc[I_cam_disc > 0]
    unif_cam_disc = float(np.std(I_cam_disc_pos) / np.mean(I_cam_disc_pos))

    # Also compute uniformity using only the bright core (FWHM region)
    # Find the half-max radius from radial profile
    max_r_cam = int(disc_radius_cam * 4)
    rp_cam = radial_profile(I_signal, (cy_cam, cx_cam), max_r_cam)
    peak_val = rp_cam[:5].max()
    hm = peak_val / 2
    fwhm_r = 0
    for i, v in enumerate(rp_cam):
        if v < hm:
            fwhm_r = i
            break
    fwhm_um = fwhm_r * CAMERA_PIXEL_UM

    cam_core_mask = R_cam <= fwhm_r
    I_cam_core = I_signal[cam_core_mask]
    I_cam_core_pos = I_cam_core[I_cam_core > 0]
    unif_cam_core = float(np.std(I_cam_core_pos) / np.mean(I_cam_core_pos))

    print(f"\n[CAM] FWHM radius: {fwhm_r} cam-px = {fwhm_um:.1f} µm "
          f"(expected disc: {disc_radius_cam:.0f} cam-px = {disc_edge_um:.0f} µm)")
    print(f"[CAM] FWHM / expected_disc = {fwhm_r / disc_radius_cam:.2f}")

    # --- Simulation: compute uniformity within matching physical radii ---
    sim_center = (512, 512)
    Y_sim, X_sim = np.ogrid[:1024, :1024]
    R_sim = np.sqrt((X_sim - 512) ** 2 + (Y_sim - 512) ** 2)

    # Simulation at FWHM-equivalent radius (1 focal px ≈ 1 cam px)
    sim_fwhm_r = fwhm_r  # direct mapping: focal px ≈ camera px
    sim_fwhm_mask = R_sim <= sim_fwhm_r
    I_sim_fwhm = I_sim[sim_fwhm_mask]
    I_sim_fwhm_pos = I_sim_fwhm[I_sim_fwhm > 0]
    unif_sim_fwhm = float(np.std(I_sim_fwhm_pos) / np.mean(I_sim_fwhm_pos)) if len(I_sim_fwhm_pos) > 0 else 0

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Region':<35}  {'Sim σ/μ':>8}  {'Cam σ/μ':>8}  {'Ratio':>6}")
    print(f"  {'-'*65}")
    print(f"  {'Full expected disc (r=86 cam-px)':<35}  {unif_sim*100:7.2f}%  "
          f"{unif_cam_disc*100:7.2f}%  {unif_cam_disc/unif_sim:5.1f}x")
    print(f"  {'Bright core (FWHM, r=' + str(fwhm_r) + ' cam-px)':<35}  "
          f"{unif_sim_fwhm*100:7.2f}%  {unif_cam_core*100:7.2f}%  "
          f"{unif_cam_core/max(unif_sim_fwhm, 1e-10):5.1f}x")
    print(f"{'='*70}")
    print(f"\n  Camera spot FWHM = {fwhm_um:.0f} µm, expected disc = {disc_edge_um:.0f} µm")
    print(f"  Spot is {disc_edge_um / max(fwhm_um, 1):.1f}x smaller than expected flat-top disc")

    # --- Radial profiles ---
    max_r_sim = int(tophat_radius_px * 2)
    rp_sim = radial_profile(I_sim, sim_center, max_r_sim)
    rp_sim_norm = rp_sim / np.mean(I_sim_disc)
    rp_cam_norm = rp_cam / np.mean(I_cam_core_pos) if np.mean(I_cam_core_pos) > 0 else rp_cam

    # --- Generate comparison figure ---
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Panel 1: Simulation intensity (zoomed to disc)
    margin = int(tophat_radius_px * 2)
    s = slice(512 - margin, 512 + margin)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(I_sim[s, s], cmap="hot", origin="lower",
                   extent=[-margin, margin, -margin, margin])
    plt.colorbar(im, ax=ax, label="Intensity (a.u.)")
    circle_sim = plt.Circle((0, 0), tophat_radius_px, fill=False,
                             edgecolor="cyan", linewidth=1.5, linestyle="--",
                             label=f"Disc edge (r={tophat_radius_px:.0f} px)")
    ax.add_patch(circle_sim)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Simulation |E_out|²\nσ/μ = {unif_sim*100:.2f}% inside disc")
    ax.set_xlabel("focal pixels")
    ax.set_ylabel("focal pixels")

    # Panel 2: Camera intensity (zoomed to actual target spot)
    cam_margin = int(disc_radius_cam * 3)
    sy = slice(max(0, cy_cam - cam_margin), min(I_signal.shape[0], cy_cam + cam_margin))
    sx = slice(max(0, cx_cam - cam_margin), min(I_signal.shape[1], cx_cam + cam_margin))
    cam_crop = I_signal[sy, sx]
    extent_cam = [-(cx_cam - sx.start), (sx.stop - cx_cam),
                  -(cy_cam - sy.start), (sy.stop - cy_cam)]
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(cam_crop, cmap="hot", origin="lower", extent=extent_cam)
    plt.colorbar(im, ax=ax, label="ADU (bg-subtracted)")
    circle_disc = plt.Circle((0, 0), disc_radius_cam, fill=False,
                              edgecolor="cyan", linewidth=1.5, linestyle="--",
                              label=f"Expected disc (r={disc_radius_cam:.0f} px)")
    circle_fwhm = plt.Circle((0, 0), fwhm_r, fill=False,
                              edgecolor="lime", linewidth=1.5, linestyle="-",
                              label=f"FWHM (r={fwhm_r} px)")
    ax.add_patch(circle_disc)
    ax.add_patch(circle_fwhm)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Camera (bg-subtracted)\nσ/μ = {unif_cam_disc*100:.1f}% inside disc (r={disc_radius_cam:.0f} px)")
    ax.set_xlabel("camera pixels")
    ax.set_ylabel("camera pixels")

    # Panel 3: Radial profiles overlaid (in pixels — 1 focal px ≈ 1 cam px)
    ax = fig.add_subplot(gs[1, 0])
    r_sim_px = np.arange(len(rp_sim_norm))  # focal pixels
    r_cam_px = np.arange(len(rp_cam_norm))  # camera pixels
    ax.plot(r_sim_px, rp_sim_norm, "b-", linewidth=2, label="Simulation (focal px)")
    ax.plot(r_cam_px, rp_cam_norm, "r-", linewidth=1.5, alpha=0.8, label="Camera (cam px)")
    ax.axvline(disc_radius_cam, color="cyan", linestyle="--", alpha=0.6,
               label=f"Disc edge (r={disc_radius_cam:.0f} px)")
    ax.axvline(fwhm_r, color="lime", linestyle="-", alpha=0.6,
               label=f"Camera FWHM (r={fwhm_r} px)")
    ax.set_xlabel("Radius (pixels)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title("Radial profiles (1 focal px ≈ 1 cam px)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, disc_radius_cam * 3)
    ax.set_ylim(0, None)

    # Panel 4: Histogram of pixel intensities inside disc
    ax = fig.add_subplot(gs[1, 1])
    I_sim_disc_norm = I_sim_disc / np.mean(I_sim_disc)
    I_cam_disc_norm = I_cam_disc_pos / np.mean(I_cam_disc_pos)
    bins = np.linspace(0, 3, 60)
    ax.hist(I_sim_disc_norm, bins=bins, alpha=0.6, density=True,
            label=f"Sim (σ/μ={unif_sim*100:.1f}%)", color="blue")
    ax.hist(I_cam_disc_norm, bins=bins, alpha=0.6, density=True,
            label=f"Cam (σ/μ={unif_cam_disc*100:.1f}%)", color="red")
    ax.set_xlabel("Normalized intensity (mean=1)")
    ax.set_ylabel("Density")
    ax.set_title(f"Pixel intensity distributions (r={disc_radius_cam:.0f} disc)")
    ax.legend()

    # Panel 5: Camera uniformity vs ROI radius
    ax = fig.add_subplot(gs[2, 0])
    r_vals = [x[0] for x in roi_results]  # in camera pixels
    u_vals = [x[5] * 100 for x in roi_results]
    m_vals = [x[3] for x in roi_results]
    ax.plot(r_vals, u_vals, "ro-", linewidth=2, markersize=6)
    ax.axvline(disc_radius_cam, color="cyan", linestyle="--", alpha=0.6,
               label=f"Disc edge (r={disc_radius_cam:.0f} px)")
    ax.axvline(fwhm_r, color="lime", linestyle="-", alpha=0.6,
               label=f"FWHM (r={fwhm_r} px)")
    ax.axhline(unif_sim * 100, color="blue", linestyle=":", alpha=0.6,
               label=f"Sim uniformity ({unif_sim*100:.1f}%)")
    ax.set_xlabel("ROI radius (camera pixels)")
    ax.set_ylabel("σ/μ (%)")
    ax.set_title("Camera uniformity vs ROI radius")
    ax.legend(fontsize=8)

    # Panel 6: Mean intensity vs ROI radius
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(r_vals, m_vals, "gs-", linewidth=2, markersize=6)
    ax.axvline(disc_radius_cam, color="cyan", linestyle="--", alpha=0.6,
               label=f"Disc edge")
    ax.axvline(fwhm_r, color="lime", linestyle="-", alpha=0.6,
               label=f"FWHM")
    ax.set_xlabel("ROI radius (camera pixels)")
    ax.set_ylabel("Mean intensity (ADU)")
    ax.set_title("Camera mean signal vs ROI radius")
    ax.legend(fontsize=8)

    plt.suptitle(f"Top-Hat Uniformity: Simulation vs Camera\n"
                 f"beamwaist={params.get('beamwaist_um', '?')} µm, LUT={params['LUT']}, "
                 f"eta_min={params.get('eta_min', '?')}, theta={params['cgm_theta']:.2f}",
                 fontsize=13, y=1.01)
    plt.savefig(OUTPUT_PDF, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[SAVE] {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
