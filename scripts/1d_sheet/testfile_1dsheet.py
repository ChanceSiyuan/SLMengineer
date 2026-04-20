"""1D CGM light-sheet path (issue #21).

Dimension-decomposition: the input SLM beam is a 2D-separable Gaussian,
so if the SLM phase is set constant in y — ``phi(x, y) = phi(x)`` — the
focal-plane field factors as

    F_x{exp(i phi(x)) * exp(-x^2 / 2 sigma_x^2)}
        * F_y{exp(-y^2 / 2 sigma_y^2)}

The y factor is already a Gaussian of radius ``lambda f / pi w_0``, which
matches the perpendicular envelope of the 2D target for our optics, so we
only need to solve 1D CGM on the x-axis to produce a top-hat.  A linear
phase ramp on phi(x) shifts the pattern off the zero-order.

Pipeline is parallel to ``scripts/sheet/testfile_sheet.py`` — same env-var
knobs, same post-processing (phase_to_screen + Fresnel + calibration) —
but CGM runs on length-N instead of N x N, ~N-fold speedup.

Usage::

    uv run python scripts/1d_sheet/testfile_1dsheet.py
    ./push_run.sh payload/1d_sheet/testfile_1dsheet_payload.npz
"""
from __future__ import annotations

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from slm.cgm import CGM_phase_generate_1d
from slm.generation import SLM_class
from slm import imgpy
from slm.targets import light_sheet_1d, measure_region_1d

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = "payload/1d_sheet"
PAYLOAD_PATH = f"{OUTPUT_DIR}/testfile_1dsheet_payload.npz"
PARAMS_PATH  = f"{OUTPUT_DIR}/testfile_1dsheet_params.json"
PREVIEW_PATH = f"{OUTPUT_DIR}/testfile_1dsheet_preview.pdf"

BEAM_CENTER_DX_UM = int(os.environ.get("SLM_BCM_DX_UM", 0))
BEAM_CENTER_DY_UM = int(os.environ.get("SLM_BCM_DY_UM", 0))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    etime_us           = int(os.environ.get("SLM_ETIME_US", 1500))
    n_avg              = int(os.environ.get("SLM_N_AVG", 20))
    LUT                = 207
    fresnel_sd         = int(os.environ.get("SLM_FRESNEL_SD", 1200))

    sheet_flat_width     = int(os.environ.get("SLM_FLAT_WIDTH", 36))
    sheet_gaussian_sigma = float(os.environ.get("SLM_GAUSS_SIGMA", 4))
    sheet_edge_sigma     = float(os.environ.get("SLM_EDGE_SIGMA", 3))
    target_shift_fpx     = int(os.environ.get("SLM_TARGET_SHIFT_FPX", 80))

    cgm_steepness      = int(os.environ.get("SLM_CGM_STEEPNESS", 9))
    cgm_max_iterations = int(os.environ.get("SLM_CGM_MAX_ITER", 4000))
    setting_eta        = float(os.environ.get("SLM_SETTING_ETA", 0.1))
    cgm_eta_steepness  = int(os.environ.get("SLM_CGM_ETA_STEEPNESS", 8))

    array_bit = int(os.environ.get("SLM_ARRAY_BIT", 12))
    grid_px = 1 << array_bit

    SLM = SLM_class()
    SLM.arraySizeBit = [array_bit, array_bit]
    SLM.image_init(
        initGaussianPhase_user_defined=np.zeros((grid_px, grid_px)), Plot=False,
        beam_center_um=(BEAM_CENTER_DX_UM, BEAM_CENTER_DY_UM),
    )
    W, H = SLM.SLMRes
    cx, cy = W // 2, H // 2
    nx = int(SLM.ImgResX)
    ny = int(SLM.ImgResY)
    cy_compute = ny // 2

    correct = lambda screen: imgpy.SLM_screen_Correct(  # noqa: E731
        screen, LUT=LUT, correctionImgPath="calibration/CAL_LSH0905549_1013nm.bmp"
    )

    print(f"Compute grid: ({ny}, {nx})  focal pitch = {SLM.Focalpitchx:.3f} um/px")
    print(f"SLM native:   ({H}, {W})")

    # Sanity-check the y-axis assumption: natural focal Gaussian
    # lambda f / pi w0 vs the 2D target's target sigma.
    natural_perp_sigma_um = (
        float(SLM.wavelength) * float(SLM.focallength)
        / (np.pi * float(SLM.beamwaist) * float(SLM.magnification))
    )
    target_perp_sigma_um = sheet_gaussian_sigma * float(SLM.Focalpitchy)
    print(
        f"\n[Y-AXIS] natural focal sigma = lambda*f/(pi*w0) = "
        f"{natural_perp_sigma_um:.2f} um  |  2D target sigma = "
        f"{target_perp_sigma_um:.2f} um  "
        f"(ratio {natural_perp_sigma_um/target_perp_sigma_um:.2f}×)"
    )

    # -- 1D inputs --------------------------------------------------------
    input_1d = np.ascontiguousarray(SLM.initGaussianAmp[cy_compute, :])
    target_center_col = (nx - 1) / 2.0 - target_shift_fpx
    target_1d = light_sheet_1d(
        nx, flat_width=sheet_flat_width,
        center=target_center_col, edge_sigma=sheet_edge_sigma,
    )
    shift_um = target_shift_fpx * SLM.Focalpitchx
    print(
        f"[TARGET] 1D light sheet: flat_width={sheet_flat_width:.0f}px "
        f"edge_sigma={sheet_edge_sigma:.0f}px  center={target_center_col:.1f} "
        f"shift={shift_um:.0f}um from zero-order  "
        f"nonzero={int(np.count_nonzero(target_1d))}"
    )

    init_phi_1d = SLM.stationary_phase_sheet_1d(
        flat_width=sheet_flat_width, center=target_center_col,
    )

    # -- 1D CGM -----------------------------------------------------------
    t0 = time.perf_counter()
    cgm_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[CGM-1D] running {cgm_max_iterations} iterations on length-{nx} "
          f"(device={cgm_device})...")
    phi_1d_t = CGM_phase_generate_1d(
        torch.tensor(input_1d, dtype=torch.float32),
        torch.from_numpy(init_phi_1d.astype(np.float32)),
        torch.from_numpy(target_1d),
        max_iterations=cgm_max_iterations,
        steepness=cgm_steepness,
        eta_min=setting_eta,
        eta_steepness=cgm_eta_steepness,
        Plot=False,
    )
    cgm_wall_time = time.perf_counter() - t0
    per_iter_ms = cgm_wall_time / max(cgm_max_iterations, 1) * 1000
    print(f"[PHASE] done in {cgm_wall_time:.2f} s ({per_iter_ms:.3f} ms/iter)")

    phi_1d = phi_1d_t.cpu().numpy().astype(np.float64)

    # -- Broadcast 1D -> 2D, wrap, phase_to_screen, Fresnel, calibrate ----
    phase_2d = np.broadcast_to(phi_1d[None, :], (ny, nx)).copy()
    phase_wrapped = np.angle(np.exp(1j * phase_2d))
    SLM_screen_raw = SLM.phase_to_screen(phase_wrapped)

    fresnel = SLM.fresnel_lens_phase_generate(fresnel_sd, cx, cy)[0]
    SLM_screen_shift = (
        (SLM_screen_raw.astype(np.int32) + fresnel.astype(np.int32)) % 256
    ).astype(np.uint8)
    SLM_screen_final = correct(SLM_screen_shift)
    print(f"\n[SCREEN] ready-to-display uint8 {SLM_screen_final.shape} "
          f"range=[{SLM_screen_final.min()}, {SLM_screen_final.max()}]")

    # -- Diagnostics ------------------------------------------------------
    from slm.metrics import fidelity as _fidelity, efficiency as _efficiency
    from slm.propagation import fft_propagate

    region_1d = measure_region_1d(target_1d, margin=5)
    E_line = np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(input_1d * np.exp(1j * phi_1d)), norm="ortho")
    )
    F1 = float(_fidelity(E_line, target_1d, region_1d))
    eta1 = float(_efficiency(E_line, region_1d))
    print(f"[METRICS-1D] line fidelity F={F1:.6f} (1-F={1-F1:.2e})  "
          f"efficiency eta={eta1*100:.2f}%")

    # 2D propagation for the preview only
    E_out_2d = fft_propagate(SLM.initGaussianAmp * np.exp(1j * phase_wrapped))

    # -- Save -------------------------------------------------------------
    np.savez_compressed(PAYLOAD_PATH, slm_screen=SLM_screen_final)
    payload_kb = int(np.ceil(os.path.getsize(PAYLOAD_PATH) / 1024))
    print(f"\n[SAVE] payload:  {PAYLOAD_PATH}  ({payload_kb} KB)")

    params = {
        "algorithm": "CGM-1D (dimension-decomposition, issue #21)",
        "payload": PAYLOAD_PATH,
        "compute_grid": [int(ny), int(nx)],
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
        "target": "light_sheet_1d",
        "sheet_flat_width_px": sheet_flat_width,
        "sheet_flat_width_um_focal": round(float(sheet_flat_width * SLM.Focalpitchx), 2),
        "sheet_gaussian_sigma_px_logged_only": sheet_gaussian_sigma,
        "sheet_edge_sigma_px": sheet_edge_sigma,
        "target_shift_fpx": target_shift_fpx,
        "cgm_max_iterations": cgm_max_iterations,
        "cgm_steepness": cgm_steepness,
        "cgm_eta_steepness": cgm_eta_steepness,
        "cgm_setting_eta": setting_eta,
        "init_phase_method": "stationary_phase_sheet_1d",
        "cgm_wall_time_s": round(cgm_wall_time, 3),
        "cgm_per_iter_ms": round(per_iter_ms, 4),
        "cgm_device": cgm_device,
        "final_fidelity_1d": round(F1, 6),
        "final_efficiency_1d": round(eta1, 6),
        "natural_perp_sigma_um": round(natural_perp_sigma_um, 4),
        "target_perp_sigma_um": round(target_perp_sigma_um, 4),
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] params:  {PARAMS_PATH}")

    _save_preview(
        input_1d, target_1d, phi_1d, E_line, region_1d,
        E_out_2d, SLM_screen_final, F1, eta1,
    )
    print(f"[SAVE] preview: {PREVIEW_PATH}")

    print()
    print("=" * 72)
    print("1D-CGM payload ready.  Next step:")
    print()
    print(f"    ./push_run.sh {PAYLOAD_PATH}")
    print()
    print("=" * 72)


def _save_preview(input_1d, target_1d, phi_1d, E_line, region_1d,
                  E_out_2d, slm_screen, F1, eta1):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].plot(input_1d)
    axes[0, 0].set_title("Input 1D amplitude (center row)")
    axes[0, 0].grid(alpha=0.3)

    tgt_I = np.abs(target_1d) ** 2
    out_I = np.abs(E_line) ** 2
    axes[0, 1].plot(tgt_I / tgt_I.max() if tgt_I.max() > 0 else tgt_I, label="|target|²")
    axes[0, 1].plot(out_I / out_I.max() if out_I.max() > 0 else out_I, label="|output|²", alpha=0.8)
    axes[0, 1].plot(region_1d, "k--", alpha=0.3, label="region")
    axes[0, 1].set_title(f"1D line intensity  F={F1:.4f}, η={eta1*100:.2f}%")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(phi_1d)
    axes[0, 2].set_title("Optimised φ(x)")
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].imshow(np.abs(E_out_2d) ** 2, cmap="hot")
    axes[1, 0].set_title("Output |E_2D|²  (full 2D simulation)")
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])

    axes[1, 1].imshow(np.angle(E_out_2d), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title("Output phase (2D)")
    axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])

    axes[1, 2].imshow(slm_screen, cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title(f"SLM screen (Fresnel+calib)  {slm_screen.shape} uint8")
    axes[1, 2].set_xticks([]); axes[1, 2].set_yticks([])

    plt.tight_layout()
    plt.savefig(PREVIEW_PATH, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
