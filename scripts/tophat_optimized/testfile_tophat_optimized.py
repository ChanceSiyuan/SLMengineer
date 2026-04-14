"""Optimized top-hat payload: uses best parameters from issue #15 sweeps.

Saves E_out, targetAmp, and region to the .npz for offline uniformity analysis.

Usage::

    uv run python scripts/tophat_optimized/testfile_tophat_optimized.py
    ./scripts/tophat_optimized/testfile_tophat_optimized.sh
"""
from __future__ import annotations

import json
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

PAYLOAD_PATH = "scripts/tophat_optimized/testfile_tophat_optimized_payload.npz"
PARAMS_PATH = "scripts/tophat_optimized/testfile_tophat_optimized_params.json"
PREVIEW_PATH = "scripts/tophat_optimized/testfile_tophat_optimized_preview.pdf"


def main():
    # --- Optimized parameters (issue #15 sweep results) ---
    etime_us = 4000
    n_avg = 10
    LUT = 200              # was 207, sweep winner
    fresnel_sd = 1000      # unchanged (already optimal)

    tophat_radius = 10.0

    cgm_R = 0
    cgm_D = -np.pi / 12   # unchanged
    cgm_theta = 0          # was -pi/4, sweep winner (pure horizontal offset)
    cgm_steepness = 9
    cgm_max_iterations = 2000

    # --- SLM setup with optimized beamwaist ---
    SLM = SLM_class()
    SLM.beamwaist = 4500   # was 5100, sweep winner
    SLM.arraySizeBit = [10, 10]
    SLM.image_init(
        initGaussianPhase_user_defined=np.zeros((1024, 1024)), Plot=False
    )
    W, H = SLM.SLMRes
    cx, cy = W // 2, H // 2

    correct = lambda screen: imgpy.SLM_screen_Correct(  # noqa: E731
        screen, LUT=LUT, correctionImgPath="calibration/CAL_LSH0905549_1013nm.bmp"
    )

    print(f"Compute grid: ({SLM.ImgResY}, {SLM.ImgResX})  "
          f"focal pitch = {SLM.Focalpitchx:.3f} um/px")
    print(f"SLM native:   ({H}, {W})")
    print(f"beamwaist = {SLM.beamwaist} um (optimized)")

    # --- Generate target ---
    targetAmp = SLM.top_hat_target(radius=tophat_radius)
    print(f"\n[TARGET] top-hat: radius={tophat_radius:.0f}px "
          f"nonzero={np.count_nonzero(targetAmp)}")

    # --- Initial phase ---
    init_phi = _initial_phase(
        (int(SLM.ImgResY), int(SLM.ImgResX)),
        CGMConfig(R=cgm_R, D=cgm_D, theta=cgm_theta),
    )

    # --- CGM ---
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
        eta_min=0.05,          # was 0.1, sweep winner
        Plot=False,
    )
    cgm_wall_time = time.perf_counter() - t0
    per_iter_ms = cgm_wall_time / max(cgm_max_iterations, 1) * 1000
    print(f"[CGM] done in {cgm_wall_time:.2f} s ({per_iter_ms:.1f} ms/iter)")

    phase_np = SLM_Phase.cpu().clone().numpy()
    phase_wrapped = np.angle(np.exp(1j * phase_np))
    SLM_screen_raw = SLM.phase_to_screen(phase_wrapped)

    # --- Fresnel lens ---
    fresnel = SLM.fresnel_lens_phase_generate(fresnel_sd, cx, cy)[0]
    SLM_screen_shift = (
        (SLM_screen_raw.astype(np.int32) + fresnel.astype(np.int32)) % 256
    ).astype(np.uint8)

    # --- Calibration correction ---
    SLM_screen_final = correct(SLM_screen_shift)
    print(f"\n[SCREEN] ready-to-display uint8 {SLM_screen_final.shape} "
          f"range=[{SLM_screen_final.min()}, {SLM_screen_final.max()}]")

    # --- Metrics ---
    from slm.metrics import fidelity as _fidelity, efficiency as _efficiency
    from slm.metrics import uniformity as _uniformity, non_uniformity_error as _nue
    from slm.propagation import fft_propagate
    from slm.targets import measure_region as _measure_region

    region_np = _measure_region(targetAmp.shape, targetAmp, margin=5)
    E_out_1024 = fft_propagate(SLM.initGaussianAmp * np.exp(1j * phase_wrapped))
    F = _fidelity(E_out_1024, targetAmp, region_np)
    eta = _efficiency(E_out_1024, region_np)

    # Intensity uniformity inside the disc
    target_mask = mask_from_target(targetAmp)
    I_sim = np.abs(E_out_1024) ** 2
    I_disc = I_sim[target_mask > 0]
    unif_sim = _uniformity(I_disc)
    nue = _nue(I_sim, np.abs(targetAmp) ** 2, target_mask)

    print(f"[METRICS] fidelity F={F:.6f}  efficiency eta={eta*100:.2f}%")
    print(f"[METRICS] intensity uniformity (std/mean) = {unif_sim:.4f} ({unif_sim*100:.2f}%)")
    print(f"[METRICS] non-uniformity error (Bowman)   = {nue:.6f}")

    # --- Save payload with simulation data for analysis ---
    np.savez_compressed(
        PAYLOAD_PATH,
        slm_screen=SLM_screen_final,
        E_out=E_out_1024,
        targetAmp=targetAmp,
        region=region_np,
        target_mask=target_mask,
        phase_wrapped=phase_wrapped,
        initGaussianAmp=SLM.initGaussianAmp,
    )
    payload_size_kb = int(np.ceil(
        __import__("os").path.getsize(PAYLOAD_PATH) / 1024
    ))
    print(f"\n[SAVE] payload:  {PAYLOAD_PATH}  ({payload_size_kb} KB)")

    # --- Save params ---
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
        "target": "top_hat",
        "tophat_radius_px": tophat_radius,
        "tophat_radius_um_focal": round(float(tophat_radius * SLM.Focalpitchx), 2),
        "beamwaist_um": SLM.beamwaist,
        "cgm_max_iterations": cgm_max_iterations,
        "cgm_steepness": cgm_steepness,
        "cgm_R": cgm_R,
        "cgm_D": cgm_D,
        "cgm_theta": cgm_theta,
        "eta_min": 0.05,
        "cgm_wall_time_s": round(cgm_wall_time, 3),
        "cgm_per_iter_ms": round(per_iter_ms, 2),
        "cgm_device": cgm_device,
        "final_fidelity": round(float(F), 6),
        "final_one_minus_fidelity": float(f"{1 - F:.3e}"),
        "final_efficiency": round(float(eta), 6),
        "intensity_uniformity_std_over_mean": round(float(unif_sim), 6),
        "non_uniformity_error_bowman": round(float(nue), 6),
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] params:  {PARAMS_PATH}")

    # --- Preview ---
    _save_preview(
        SLM.initGaussianAmp, targetAmp, E_out_1024, region_np,
        SLM_screen_final, SLM_Phase.cpu().numpy(), F, eta, unif_sim,
    )
    print(f"[SAVE] preview: {PREVIEW_PATH}")

    print()
    print("=" * 72)
    print("Payload ready.  Next step:")
    print()
    print("    ./scripts/tophat_optimized/testfile_tophat_optimized.sh")
    print()
    print("=" * 72)


def _save_preview(input_amp, target, E_out, region, slm_screen_final, slm_phase_full, F, eta, unif):
    target_mask = mask_from_target(target)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(input_amp, cmap="viridis")
    axes[0, 0].set_title("Input amplitude |S|\n(Gaussian, 1024 grid)")

    axes[0, 1].imshow(np.abs(target) ** 2, cmap="hot")
    axes[0, 1].set_title("Target |tau|^2\n(top-hat disc)")

    target_phase_masked = np.where(target_mask > 0, np.angle(target), np.nan)
    axes[0, 2].imshow(target_phase_masked, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0, 2].set_title("Target phase arg(tau)\n(flat, zero)")

    out_int = (np.abs(E_out) ** 2) * region
    axes[1, 0].imshow(out_int, cmap="hot")
    axes[1, 0].set_title(
        f"Output |E_out|^2\nF={F:.4f}, eta={100*eta:.2f}%, unif={unif*100:.1f}%"
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
