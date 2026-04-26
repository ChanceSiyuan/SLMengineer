"""Local-only CGM compute: produce a light-sheet phase payload for the
dedicated Windows hardware runner.

Parallel to ``scripts/testfile_lg.py`` but targeting a soft-edged 1D
top-hat via ``SLM.light_sheet_target`` (useful for Rydberg beam
shaping).  Shared CGM + optics config with the LG01 baseline so the
per-shape diff on the camera is attributable to the target choice alone.

uv run python scripts/sheet/testfile_sheet.py
This will generate the payload/sheet/testfile_sheet_payload.npz file.

Next step after running this script::

    ./push_run.sh payload/sheet/testfile_sheet_payload.npz

    uv run python processing/bmp_to_color.py data/sheet/testfile_sheet_after.bmp --out data/sheet/testfile_sheet_after.png
    uv run python processing/bmp_to_color.py data/sheet/testfile_sheet_before.bmp --out data/sheet/testfile_sheet_before.png
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

from slm.cgm import CGM_phase_generate, CGMConfig
from slm.generation import SLM_class
from slm import imgpy
from slm.targets import mask_from_target

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = "payload/sheet"
PAYLOAD_PATH = f"{OUTPUT_DIR}/testfile_sheet_payload.npz"
PARAMS_PATH = f"{OUTPUT_DIR}/testfile_sheet_params.json"
PREVIEW_PATH = f"{OUTPUT_DIR}/testfile_sheet_preview.pdf"

# Measured incident-beam center on the SLM plane (um, relative to the
# SLM compute-grid geometric center).  Used to model off-center
# illumination in CGM so the optimized phase matches the real input
# amplitude.  Fill in from calibration; (0, 0) == legacy behavior.
BEAM_CENTER_DX_UM = int(os.environ.get("SLM_BCM_DX_UM", 0))   # closed-loop overridable
BEAM_CENTER_DY_UM = int(os.environ.get("SLM_BCM_DY_UM", 0))      # closed-loop overridable


def main(reweight: np.ndarray | None = None) -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # All knobs exposed as env vars so param_sweep.sh / closed_loop_sheet.py
    # can perturb without editing this file.  Defaults target the 4096²
    # compute grid — ~2.7× better RMS on camera than the old 1024² path
    # at equivalent physical sheet size (focal pitch 3.957 µm/px, so
    # flat_width=36 px ≈ 142 µm, matching the original 9 px × 15.83 µm/px).
    etime_us           = int(os.environ.get("SLM_ETIME_US", 1000))
    n_avg              = int(os.environ.get("SLM_N_AVG", 20))
    LUT                = 207
    fresnel_sd         = int(os.environ.get("SLM_FRESNEL_SD", 1000))

    sheet_flat_width     = int(os.environ.get("SLM_FLAT_WIDTH", 35))
    sheet_gaussian_sigma = float(os.environ.get("SLM_GAUSS_SIGMA", 2))
    sheet_edge_sigma     = float(os.environ.get("SLM_EDGE_SIGMA", 5))
    sheet_angle          = 0
    # Target is shifted diagonally from the zero-order so the first-order
    # does not overlap with the undiffracted beam at grid centre.
    target_shift_fpx     = int(os.environ.get("SLM_TARGET_SHIFT_FPX", 50))

    cgm_steepness      = int(os.environ.get("SLM_CGM_STEEPNESS", 9))
    cgm_max_iterations = int(os.environ.get("SLM_CGM_MAX_ITER", 1000))
    setting_eta        = float(os.environ.get("SLM_SETTING_ETA", 0.1))
    cgm_eta_steepness  = int(os.environ.get("SLM_CGM_ETA_STEEPNESS", 8))

    # Compute-grid size.  SLM.phase_to_screen centre-crops to 1024×1024
    # and pads to 1024×1272 regardless of input, so higher bits = finer
    # focal pitch during CGM; only the central 1024 is then passed to
    # the SLM.  Default 12 (4096²) is the winning config from the sweep;
    # set 10 (1024²) for the historical baseline.
    array_bit = int(os.environ.get("SLM_ARRAY_BIT", 12))
    grid_px = 1 << array_bit
    SLM = SLM_class()
    SLM.arraySizeBit = [array_bit, array_bit]
    SLM.image_init(
        initGaussianPhase_user_defined=np.zeros((grid_px, grid_px)), Plot=False,
        beam_center_um=(BEAM_CENTER_DX_UM, BEAM_CENTER_DY_UM),
    )
    W, H = SLM.SLMRes  # (1272, 1024)
    cx, cy = W // 2, H // 2

    correct = lambda screen: imgpy.SLM_screen_Correct(  # noqa: E731
        screen, LUT=LUT, correctionImgPath="calibration/CAL_LSH0905549_1013nm.bmp"
    )

    print(f"Compute grid: ({SLM.ImgResY}, {SLM.ImgResX})  "
          f"focal pitch = {SLM.Focalpitchx:.3f} um/px")
    print(f"SLM native:   ({H}, {W})")

    # --- 2. Generate light-sheet target on the 1024x1024 computation grid ---
    # Place target at a shifted position so it doesn't overlap with the
    # zero-order beam (which is fixed at grid center).
    target_center = (
        (SLM.ImgResY - 1) / 2.0 - target_shift_fpx,
        (SLM.ImgResX - 1) / 2.0 - target_shift_fpx,
    )
    targetAmp = SLM.light_sheet_target(
        flat_width=sheet_flat_width,
        gaussian_sigma=sheet_gaussian_sigma,
        angle=sheet_angle,
        edge_sigma=sheet_edge_sigma,
        center=target_center,
        reweight=reweight,
    )
    shift_um = target_shift_fpx * SLM.Focalpitchx
    print(
        f"\n[TARGET] light sheet: flat_width={sheet_flat_width:.0f}px "
        f"gauss_sigma={sheet_gaussian_sigma:.0f}px edge_sigma={sheet_edge_sigma:.0f}px "
        f"center=({target_center[0]:.1f},{target_center[1]:.1f}) "
        f"shift={shift_um:.0f}um from zero-order "
        f"nonzero={np.count_nonzero(targetAmp)}"
    )

    # --- 3. Build the stationary-phase analytical initial phase ---
    # Geometric-optics seed: 1D top-hat along u + cylindrical Fresnel lens
    # along v.  The center shift adds a linear phase ramp that deflects
    # the first-order pattern away from the zero-order.
    init_phi = SLM.stationary_phase_sheet(
        flat_width=sheet_flat_width,
        gaussian_sigma=None,  # ← issue #20 fix: omit cylindrical-lens term
        angle=sheet_angle,
        center=target_center,
    )

    # --- 4. Seed + bounded CGM polish ---
    t0 = time.perf_counter()
    if cgm_max_iterations > 0:
        cgm_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[CGM] running {cgm_max_iterations} iterations on "
              f"{SLM.ImgResY}x{SLM.ImgResX} grid (device={cgm_device})...")
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
        print("\n[SEED] using stationary-phase seed directly (no CGM)")
        phase_np = init_phi
    cgm_wall_time = time.perf_counter() - t0
    per_iter_ms = cgm_wall_time / max(cgm_max_iterations, 1) * 1000
    print(f"[PHASE] done in {cgm_wall_time:.2f} s")

    # Wrap phase to [-pi, pi] before cropping.
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
        os.path.getsize(PAYLOAD_PATH) / 1024
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
        "beam_center_um": [BEAM_CENTER_DX_UM, BEAM_CENTER_DY_UM],
        "target": "light_sheet",
        "sheet_flat_width_px": sheet_flat_width,
        "sheet_flat_width_um_focal": round(float(sheet_flat_width * SLM.Focalpitchx), 2),
        "sheet_gaussian_sigma_px": sheet_gaussian_sigma,
        "sheet_angle_rad": sheet_angle,
        "sheet_edge_sigma_px": sheet_edge_sigma,
        "cgm_max_iterations": cgm_max_iterations,
        "cgm_steepness": cgm_steepness,
        "cgm_eta_steepness": cgm_eta_steepness,
        "cgm_setting_eta": setting_eta,
        "init_phase_method": "stationary_phase_2d",
        "cgm_wall_time_s": round(cgm_wall_time, 3),
        "cgm_per_iter_ms": round(per_iter_ms, 2),
        "cgm_device": "seed_only" if cgm_max_iterations == 0 else ("cuda" if torch.cuda.is_available() else "cpu"),
        "final_fidelity": round(float(F), 6),
        "final_one_minus_fidelity": float(f"{1 - F:.3e}"),
        "final_efficiency": round(float(eta), 6),
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] params:  {PARAMS_PATH}")

    # --- 10. Save preview.pdf ---
    # _save_preview(
    #     SLM.initGaussianAmp, targetAmp, E_out_1024, region_np,
    #     SLM_screen_final, F, eta,
    # )
    # print(f"[SAVE] preview: {PREVIEW_PATH}")

    # --- 11. Next-step hint ---
    print()
    print("=" * 72)
    print("Payload ready.  Next step (pushes to Windows and runs the experiment):")
    print()
    print(f"    ./push_run.sh {PAYLOAD_PATH}")
    print()
    print("=" * 72)

    return {
        "payload_path": PAYLOAD_PATH,
        "params_path": PARAMS_PATH,
        "fidelity": float(F),
        "efficiency": float(eta),
        "sheet_flat_width_px": int(sheet_flat_width),
        "reweight_applied": None if reweight is None else np.asarray(reweight).tolist(),
    }


def _save_preview(input_amp, target, E_out, region, slm_screen_final, F, eta):
    """6-panel PDF: input / target intensity+phase / output intensity+phase / screen."""
    target_mask = mask_from_target(target)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(input_amp, cmap="viridis")
    axes[0, 0].set_title("Input amplitude |S|\n(Gaussian, 1024 grid)")

    axes[0, 1].imshow(np.abs(target) ** 2, cmap="hot")
    axes[0, 1].set_title("Target |tau|^2\n(light sheet)")

    target_phase_masked = np.where(target_mask > 0, np.angle(target), np.nan)
    axes[0, 2].imshow(target_phase_masked, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0, 2].set_title("Target phase arg(tau)\n(flat)")

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
