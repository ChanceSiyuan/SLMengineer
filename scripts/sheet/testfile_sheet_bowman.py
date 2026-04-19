"""Parallel to ``scripts/sheet/testfile_sheet.py`` but using the
*Bowman* structured guess-phase initialisation (Bowman et al., Opt. Express
2017; see ``references/top-hat.tex``) instead of the geometric-optics
``stationary_phase_light_sheet`` seed used in the companion script.

The Bowman init is the simple quadratic + linear combination::

    phi(p, q) = R * (p^2 + q^2) + D * (p * cos(theta) + q * sin(theta))

that the paper uses as a guess phase for conjugate-gradient minimisation.
The quadratic term controls the size of the output envelope (acts like a
weak lens); the linear term tilts the output to the target position (here,
the 30-focal-pixel diagonal offset off the zero-order).

Everything else — target shape, post-hoc Fresnel lens, LUT, calibration,
capture parameters, CGM polish iterations — is identical to the companion
script, so any on-camera difference is attributable to the *initialisation*.

Outputs
-------
    payload/sheet/testfile_sheet_bowman_payload.npz
    payload/sheet/testfile_sheet_bowman_params.json
    payload/sheet/testfile_sheet_bowman_preview.pdf

Usage::

    uv run python scripts/sheet/testfile_sheet_bowman.py
    ./push_run.sh payload/sheet/testfile_sheet_bowman_payload.npz --png
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

from slm.cgm import CGM_phase_generate
from slm.generation import SLM_class
from slm import imgpy
from slm.targets import mask_from_target

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = "payload/sheet"
BASE = "testfile_sheet_bowman"
PAYLOAD_PATH = f"{OUTPUT_DIR}/{BASE}_payload.npz"
PARAMS_PATH = f"{OUTPUT_DIR}/{BASE}_params.json"
PREVIEW_PATH = f"{OUTPUT_DIR}/{BASE}_preview.pdf"

# Beam-center offset in SLM-plane μm (same as companion script so the
# comparison is not confounded).
BEAM_CENTER_DX_UM = int(os.environ.get("SLM_BCM_DX_UM", 0))
BEAM_CENTER_DY_UM = int(os.environ.get("SLM_BCM_DY_UM", 0))


def bowman_initial_phase(
    shape: tuple[int, int],
    R_rad_px2: float,
    D_rad_px: float,
    theta_rad: float,
) -> np.ndarray:
    """phi(p, q) = R·(p²+q²) + D·(p·cos θ + q·sin θ).

    Same convention as ``slm.cgm._initial_phase``: pixel indices centred on
    ``((nx-1)/2, (ny-1)/2)`` so the quadratic + linear terms are symmetric
    around the FFT origin.
    """
    ny, nx = shape
    p = np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0
    q = np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0
    pp, qq = np.meshgrid(p, q, indexing="xy")
    return R_rad_px2 * (pp ** 2 + qq ** 2) + D_rad_px * (
        pp * np.cos(theta_rad) + qq * np.sin(theta_rad)
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Capture parameters (same as companion script) ---
    etime_us = 3500
    n_avg = 10
    LUT = 207
    fresnel_sd = 1000

    # --- Target parameters (same as companion script) ---
    sheet_flat_width = 10
    sheet_gaussian_sigma = 1.5
    sheet_angle = 0
    sheet_edge_sigma = 0
    target_shift_fpx = 20

    # --- CGM parameters (same as companion script) ---
    cgm_steepness = 9
    cgm_max_iterations = int(os.environ.get("SLM_CGM_MAX_ITER", 4000))
    setting_eta       = float(os.environ.get("SLM_SETTING_ETA", 0.1))
    cgm_eta_steepness = int(os.environ.get("SLM_CGM_ETA_STEEPNESS", 9))

    # --- SLM class setup (same as companion script) ---
    SLM = SLM_class()
    SLM.arraySizeBit = [10, 10]  # 1024 x 1024 compute grid
    SLM.image_init(
        initGaussianPhase_user_defined=np.zeros((1024, 1024)), Plot=False,
        beam_center_um=(BEAM_CENTER_DX_UM, BEAM_CENTER_DY_UM),
    )
    W, H = SLM.SLMRes
    cx, cy = W // 2, H // 2
    nx, ny = int(SLM.ImgResX), int(SLM.ImgResY)

    correct = lambda screen: imgpy.SLM_screen_Correct(  # noqa: E731
        screen, LUT=LUT, correctionImgPath="calibration/CAL_LSH0905549_1013nm.bmp"
    )

    print(f"Compute grid: ({ny}, {nx})  focal pitch = {SLM.Focalpitchx:.3f} um/px")
    print(f"SLM native:   ({H}, {W})")

    # --- Target (same as companion script) ---
    target_center = (
        (ny - 1) / 2.0 - target_shift_fpx,
        (nx - 1) / 2.0 - target_shift_fpx,
    )
    targetAmp = SLM.light_sheet_target(
        flat_width=sheet_flat_width,
        gaussian_sigma=sheet_gaussian_sigma,
        angle=sheet_angle,
        edge_sigma=sheet_edge_sigma,
        center=target_center,
    )

    # --- Bowman initial phase ---
    # R controls the size of the focal-plane envelope.  Setting R equal to
    # the paraxial thin-lens value puts the envelope at the 2F plane, which
    # matches where the rest of the pipeline (objective + post-hoc Fresnel)
    # focuses the output.  For our optics (λ=1.013 μm, pitch=12.5 μm,
    # f_eff=200 mm), R_natural ≈ π·pitch² / (λ·f) ≈ 2.42 mrad·px⁻².  The
    # Bowman paper used R = 2.3 mrad·px⁻² for the Gaussian Line at their
    # slightly-different optics, so we're in the same ballpark.
    #
    # D, θ place the focal-plane pattern at the commanded diagonal offset
    # of 30 focal pixels (same as the linear-ramp shift that
    # stationary_phase_light_sheet applies via center_um).
    #
    #     per-axis phase ramp = −2π · 30 / N    rad/px    (for an N-point FFT)
    #     D · cos(π/4) = D · sin(π/4) = −2π·30/N
    #     ⇒ D = −2π·30·√2 / N    with θ = π/4
    #
    lam_um = float(SLM.wavelength)
    pitch_um = float(SLM.pixelpitch)
    f_um = float(SLM.focallength) / float(SLM.magnification)
    R_rad_px2 = float(
        os.environ.get("BOWMAN_R_RAD_PX2",
                       np.pi * pitch_um ** 2 / (lam_um * f_um)),
    )

    per_axis_ramp = -2.0 * np.pi * target_shift_fpx / max(nx, ny)
    D_rad_px = per_axis_ramp * np.sqrt(2.0)
    theta_rad = np.pi / 4.0

    D_rad_px = float(os.environ.get("BOWMAN_D_RAD_PX", D_rad_px))
    theta_rad = float(os.environ.get("BOWMAN_THETA_RAD", theta_rad))

    print(
        f"\n[INIT] Bowman R={R_rad_px2*1e3:.3f} mrad·px⁻²  "
        f"D={D_rad_px:.4f} rad/px  θ={theta_rad:.4f} rad (= π/{np.pi/theta_rad:.1f})"
    )
    init_phi = bowman_initial_phase((ny, nx), R_rad_px2, D_rad_px, theta_rad)

    # --- CGM ---
    t0 = time.perf_counter()
    if cgm_max_iterations > 0:
        cgm_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[CGM] {cgm_max_iterations} iterations on {ny}x{nx}  "
              f"(device={cgm_device}, eta_min={setting_eta}, "
              f"eta_steepness={cgm_eta_steepness})")
        SLM_Phase = CGM_phase_generate(
            torch.tensor(SLM.initGaussianAmp),
            torch.from_numpy(init_phi),
            torch.from_numpy(targetAmp),
            max_iterations=cgm_max_iterations,
            steepness=cgm_steepness,
            eta_min=setting_eta,
            eta_steepness=cgm_eta_steepness,
            Plot=False,
        )
        phase_np = SLM_Phase.cpu().clone().numpy()
    else:
        print("[INIT-ONLY] using Bowman guess phase directly (no CGM)")
        phase_np = init_phi
    cgm_wall_time = time.perf_counter() - t0
    per_iter_ms = cgm_wall_time / max(cgm_max_iterations, 1) * 1000
    print(f"[PHASE] done in {cgm_wall_time:.2f} s  ({per_iter_ms:.2f} ms/iter)")

    # --- Post-hoc Fresnel + LUT (identical to companion) ---
    phase_wrapped = np.angle(np.exp(1j * phase_np))
    SLM_screen_raw = SLM.phase_to_screen(phase_wrapped)
    fresnel = SLM.fresnel_lens_phase_generate(fresnel_sd, cx, cy)[0]
    SLM_screen_shift = (
        (SLM_screen_raw.astype(np.int32) + fresnel.astype(np.int32)) % 256
    ).astype(np.uint8)
    SLM_screen_final = correct(SLM_screen_shift)
    print(f"[SCREEN] uint8 {SLM_screen_final.shape} "
          f"range=[{SLM_screen_final.min()}, {SLM_screen_final.max()}]")

    # --- Metrics (simulation) ---
    from slm.metrics import fidelity as _fidelity, efficiency as _efficiency
    from slm.propagation import fft_propagate
    from slm.targets import measure_region as _measure_region

    region_np = _measure_region(targetAmp.shape, targetAmp, margin=5)
    E_out_1024 = fft_propagate(SLM.initGaussianAmp * np.exp(1j * phase_wrapped))
    F = _fidelity(E_out_1024, targetAmp, region_np)
    eta = _efficiency(E_out_1024, region_np)
    print(f"[METRICS] simulated fidelity F={F:.6f} (1-F={1-F:.2e})  "
          f"efficiency eta={eta*100:.2f}%")

    # --- Save payload ---
    np.savez_compressed(PAYLOAD_PATH, slm_screen=SLM_screen_final)
    print(f"\n[SAVE] payload:  {PAYLOAD_PATH} "
          f"({int(np.ceil(os.path.getsize(PAYLOAD_PATH)/1024))} KB)")

    params = {
        "algorithm": "CGM+BowmanInit",
        "payload": PAYLOAD_PATH,
        "compute_grid": [ny, nx],
        "slm_native": [int(H), int(W)],
        "focal_pitch_x_um_per_px": round(float(SLM.Focalpitchx), 4),
        "focal_pitch_y_um_per_px": round(float(SLM.Focalpitchy), 4),
        "runner_defaults": {"etime_us": etime_us, "n_avg": n_avg, "monitor": 1},
        "fresnel_applied_on_linux": True,
        "fresnel_shift_distance_um": fresnel_sd,
        "fresnel_center_xy_px": [int(cx), int(cy)],
        "calibration_applied_on_linux": True,
        "calibration_bmp": "calibration/CAL_LSH0905549_1013nm.bmp",
        "LUT": LUT,
        "beam_center_um": [BEAM_CENTER_DX_UM, BEAM_CENTER_DY_UM],
        "target": "light_sheet",
        "sheet_flat_width_px": sheet_flat_width,
        "sheet_flat_width_um_focal": round(float(sheet_flat_width*SLM.Focalpitchx), 2),
        "sheet_gaussian_sigma_px": sheet_gaussian_sigma,
        "sheet_angle_rad": sheet_angle,
        "sheet_edge_sigma_px": sheet_edge_sigma,
        "cgm_max_iterations": cgm_max_iterations,
        "cgm_steepness": cgm_steepness,
        "cgm_eta_steepness": cgm_eta_steepness,
        "cgm_setting_eta": setting_eta,
        "init_phase_method": "bowman_quadratic_linear",
        "bowman_R_rad_px2": R_rad_px2,
        "bowman_D_rad_px": D_rad_px,
        "bowman_theta_rad": theta_rad,
        "cgm_wall_time_s": round(cgm_wall_time, 3),
        "cgm_per_iter_ms": round(per_iter_ms, 2),
        "cgm_device": "cuda" if torch.cuda.is_available() else "cpu",
        "final_fidelity": round(float(F), 6),
        "final_efficiency": round(float(eta), 6),
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] params:   {PARAMS_PATH}")

    _save_preview(
        SLM.initGaussianAmp, targetAmp, E_out_1024, region_np,
        SLM_screen_final, phase_wrapped, F, eta, init_phi,
    )
    print(f"[SAVE] preview:  {PREVIEW_PATH}")

    print()
    print("=" * 72)
    print("Bowman-init payload ready.  Push to hardware with:")
    print()
    print(f"    ./push_run.sh {PAYLOAD_PATH} --png")
    print()
    print("=" * 72)


def _save_preview(input_amp, target, E_out, region, slm_screen_final,
                  slm_phase_full, F, eta, init_phi):
    target_mask = mask_from_target(target)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(input_amp, cmap="viridis")
    axes[0, 0].set_title("Input |S| (Gaussian)")
    axes[0, 1].imshow(np.abs(target) ** 2, cmap="hot")
    axes[0, 1].set_title("Target |τ|²")
    axes[0, 2].imshow(init_phi, cmap="twilight")
    axes[0, 2].set_title("Bowman init φ (R·r² + D·(p cosθ+q sinθ))")

    out_int = (np.abs(E_out) ** 2) * region
    axes[1, 0].imshow(out_int, cmap="hot")
    axes[1, 0].set_title(f"Output |E|²  F={F:.4f}  η={100*eta:.2f}%")

    out_phase_masked = np.where(target_mask > 0, np.angle(E_out), np.nan)
    axes[1, 1].imshow(out_phase_masked, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title("Output phase (inside target)")

    axes[1, 2].imshow(slm_screen_final, cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title(f"SLM screen {slm_screen_final.shape}")

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(PREVIEW_PATH, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
