"""Local-only CGM compute: produce a ring-lattice-vortex phase payload
for the dedicated Windows hardware runner.

Parallel to ``scripts/testfile_lg.py`` but targeting a ring of Gaussian
peaks with a global vortex phase via ``SLM.ring_lattice_vortex_target``.
Shared CGM + optics config with the LG01 baseline so the per-shape diff
on the camera is attributable to the target choice alone.

Next step after running this script::

    ./push_run.sh payload/ring/testfile_ring_payload.npz
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slm.cgm import CGM_phase_generate, CGMConfig, _initial_phase
from slm.generation import SLM_class
from slm import imgpy
from slm.targets import mask_from_target

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = "payload/ring"
PAYLOAD_PATH = f"{OUTPUT_DIR}/testfile_ring_payload.npz"
PARAMS_PATH = f"{OUTPUT_DIR}/testfile_ring_params.json"
PREVIEW_PATH = f"{OUTPUT_DIR}/testfile_ring_preview.pdf"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # --- Capture parameters (used later by the Windows runner) ---
    etime_us = 4000        # 4 ms exposure
    n_avg = 10             # average 10 frames
    LUT = 207
    fresnel_sd = 1000      # um -- compensates camera-focal-plane offset

    # --- Ring-lattice-vortex target parameters (1024^2 grid, focal pitch 15.83 um/px) ---
    # ring_radius=20 px = 316 um on focal plane = 171 camera-px radius -> visible ring.
    ring_n_sites = 8            # 8 Gaussian peaks on the ring
    ring_radius = 20.0          # px (~316 um in focal plane)
    ring_peak_sigma = 2.0       # px per-peak 1-sigma (~32 um)
    ring_ell = 1                # topological charge of the global vortex

    # --- CGM analytical initial phase (issue #12 iteration #4: places target on-camera) ---
    # Paper's Table I values (R=4.5e-3, D=-pi/2, theta=pi/4) on our 4096^2 grid shift
    # the target by ~2865 um off zero-order, OFF the 5617x7444 um camera FOV.  Use the
    # reduced shift that hardware iteration #4 in issue #12 confirmed to land on-camera.
    cgm_R = 0.0            # no quadratic lensing (R=4.5e-3 caused crescent artifact per issue #12 #1)
    cgm_D = -np.pi / 6     # smaller linear phase offset (~950 um shift, on-camera)
    cgm_theta = 0.0        # horizontal offset only (keeps target within camera row extent)
    cgm_steepness = 9
    cgm_max_iterations = 200

    # --- 1. SLM_class setup (reads hamamatsu_test_config.json) ---
    # Override arraySizeBit from default [12,12] (=4096^2) to [10,10] (=1024^2) so the
    # CGM compute grid matches the SLM native short dimension (1024 rows).  This avoids
    # the lossy central 1024x1024 crop in phase_to_screen that previously dropped
    # fidelity from 0.98 -> 0.22.  See root-cause investigation.
    SLM = SLM_class()
    SLM.arraySizeBit = [10, 10]  # 1024 x 1024 compute grid
    SLM.image_init(
        initGaussianPhase_user_defined=np.zeros((1024, 1024)), Plot=False,
    )
    W, H = SLM.SLMRes  # (1272, 1024)
    cx, cy = W // 2, H // 2

    correct = lambda screen: imgpy.SLM_screen_Correct(  # noqa: E731
        screen, LUT=LUT, correctionImgPath="calibration/CAL_LSH0905549_1013nm.bmp"
    )

    print(f"Compute grid: ({SLM.ImgResY}, {SLM.ImgResX})  "
          f"focal pitch = {SLM.Focalpitchx:.3f} um/px")
    print(f"SLM native:   ({H}, {W})")

    # --- 2. Generate ring-lattice-vortex target on the 1024x1024 grid ---
    targetAmp = SLM.ring_lattice_vortex_target(
        n_sites=ring_n_sites,
        ring_radius=ring_radius,
        peak_sigma=ring_peak_sigma,
        ell=ring_ell,
    )
    print(
        f"\n[TARGET] ring lattice vortex: n_sites={ring_n_sites} "
        f"ring_radius={ring_radius:.0f}px peak_sigma={ring_peak_sigma:.0f}px "
        f"ell={ring_ell} dtype={targetAmp.dtype} shape={targetAmp.shape} "
        f"nonzero={np.count_nonzero(targetAmp)}"
    )

    # --- 3. Build the paper's analytical initial phase ---
    init_phi = _initial_phase(
        (int(SLM.ImgResY), int(SLM.ImgResX)),
        CGMConfig(R=cgm_R, D=cgm_D, theta=cgm_theta),
    )

    # --- 4. Run CGM on the compute grid via torch/CUDA ---
    # eta_min=0.05 forces CGM to find a higher-efficiency solution.  Default (0) lets
    # CGM converge to F~1 with eta~0.5%, where the D-grating-deflected zero-order
    # overwhelms the target shape on hardware.  eta_min=0.05 gives ~10x more light in
    # the target region at a small fidelity cost (F ~0.99 still).
    cgm_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[CGM] running {cgm_max_iterations} iterations on "
          f"{SLM.ImgResY}x{SLM.ImgResX} grid (device={cgm_device})...")
    t0 = time.perf_counter()
    SLM_Phase = CGM_phase_generate(
        torch.tensor(SLM.initGaussianAmp),
        torch.from_numpy(init_phi),
        torch.from_numpy(targetAmp),
        max_iterations=cgm_max_iterations,
        steepness=cgm_steepness,
        eta_min=0.05,
        Plot=False,
    )
    cgm_wall_time = time.perf_counter() - t0
    per_iter_ms = cgm_wall_time / max(cgm_max_iterations, 1) * 1000
    print(f"[CGM] done in {cgm_wall_time:.2f} s "
          f"({per_iter_ms:.1f} ms/iter)")

    # Wrap phase to [-pi, pi] before cropping.
    phase_np = SLM_Phase.cpu().clone().numpy()
    phase_wrapped = np.angle(np.exp(1j * phase_np))
    SLM_screen_raw = SLM.phase_to_screen(phase_wrapped)

    # --- 5. Post-hoc Fresnel lens ---
    fresnel = SLM.fresnel_lens_phase_generate(fresnel_sd, cx, cy)[0]
    SLM_screen_shift = (
        (SLM_screen_raw.astype(np.int32) + fresnel.astype(np.int32)) % 256
    ).astype(np.uint8)

    # --- 6. Calibration correction ---
    SLM_screen_final = correct(SLM_screen_shift)
    print(f"\n[SCREEN] ready-to-display uint8 {SLM_screen_final.shape} "
          f"range=[{SLM_screen_final.min()}, {SLM_screen_final.max()}]")

    # --- 7. Diagnostic metrics ---
    from slm.metrics import fidelity as _fidelity, efficiency as _efficiency
    from slm.propagation import fft_propagate
    from slm.targets import measure_region as _measure_region

    region_np = _measure_region(targetAmp.shape, targetAmp, margin=5)
    E_out_1024 = fft_propagate(SLM.initGaussianAmp * np.exp(1j * phase_wrapped))
    F = _fidelity(E_out_1024, targetAmp, region_np)
    eta = _efficiency(E_out_1024, region_np)
    print(f"[METRICS] fidelity F={F:.6f}  (1-F={1-F:.2e})  efficiency eta={eta*100:.2f}%")

    # --- 8. Save payload.npz ---
    np.savez_compressed(
        PAYLOAD_PATH,
        slm_screen=SLM_screen_final,
    )
    payload_size_kb = int(np.ceil(
        __import__("os").path.getsize(PAYLOAD_PATH) / 1024
    ))
    print(f"\n[SAVE] payload:  {PAYLOAD_PATH}  ({payload_size_kb} KB)")

    # --- 9. Save params.json ---
    params = {
        "algorithm": "CGM",
        "payload": PAYLOAD_PATH,
        "compute_grid": [int(SLM.ImgResY), int(SLM.ImgResX)],
        "slm_native": [int(H), int(W)],
        "focal_pitch_x_um_per_px": round(float(SLM.Focalpitchx), 4),
        "focal_pitch_y_um_per_px": round(float(SLM.Focalpitchy), 4),
        "runner_defaults": {
            "etime_us": etime_us,
            "n_avg": n_avg,
            "monitor": 1,
        },
        "fresnel_applied_on_linux": True,
        "fresnel_shift_distance_um": fresnel_sd,
        "fresnel_center_xy_px": [int(cx), int(cy)],
        "calibration_applied_on_linux": True,
        "calibration_bmp": "calibration/CAL_LSH0905549_1013nm.bmp",
        "LUT": LUT,
        "target": "ring_lattice_vortex",
        "ring_n_sites": ring_n_sites,
        "ring_radius_px": ring_radius,
        "ring_radius_um_focal": round(float(ring_radius * SLM.Focalpitchx), 2),
        "ring_peak_sigma_px": ring_peak_sigma,
        "ring_ell": ring_ell,
        "cgm_max_iterations": cgm_max_iterations,
        "cgm_steepness": cgm_steepness,
        "cgm_R": cgm_R,
        "cgm_D": cgm_D,
        "cgm_theta": cgm_theta,
        "cgm_wall_time_s": round(cgm_wall_time, 3),
        "cgm_per_iter_ms": round(per_iter_ms, 2),
        "cgm_device": cgm_device,
        "final_fidelity": round(float(F), 6),
        "final_one_minus_fidelity": float(f"{1 - F:.3e}"),
        "final_efficiency": round(float(eta), 6),
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] params:  {PARAMS_PATH}")

    # --- 10. Save preview.pdf ---
    _save_preview(
        SLM.initGaussianAmp, targetAmp, E_out_1024, region_np,
        SLM_screen_final, SLM_Phase.cpu().numpy(), F, eta,
    )
    print(f"[SAVE] preview: {PREVIEW_PATH}")

    # --- 11. Next-step hint ---
    print()
    print("=" * 72)
    print("Payload ready.  Next step (pushes to Windows and runs the experiment):")
    print()
    print(f"    ./push_run.sh {PAYLOAD_PATH}")
    print()
    print("=" * 72)


def _save_preview(input_amp, target, E_out, region, slm_screen_final, slm_phase_full, F, eta):
    """6-panel PDF: input / target intensity+phase / output intensity+phase / screen."""
    target_mask = mask_from_target(target)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(input_amp, cmap="viridis")
    axes[0, 0].set_title("Input amplitude |S|\n(Gaussian, 1024 grid)")

    axes[0, 1].imshow(np.abs(target) ** 2, cmap="hot")
    axes[0, 1].set_title("Target |tau|^2\n(ring lattice vortex)")

    target_phase_masked = np.where(target_mask > 0, np.angle(target), np.nan)
    axes[0, 2].imshow(target_phase_masked, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0, 2].set_title("Target phase arg(tau)\n(vortex)")

    out_int = (np.abs(E_out) ** 2) * region
    axes[1, 0].imshow(out_int, cmap="hot")
    axes[1, 0].set_title(
        f"Output |E_out|^2\nF={F:.4f}, eta={100*eta:.2f}%"
    )

    out_phase_masked = np.where(target_mask > 0, np.angle(E_out), np.nan)
    axes[1, 1].imshow(out_phase_masked, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title("Output phase arg(E_out)")

    axes[1, 2].imshow(slm_screen_final, cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title(
        f"SLM screen (Fresnel+calib applied)\n"
        f"{slm_screen_final.shape} uint8"
    )

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(PREVIEW_PATH, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
