"""Local-only CGM compute for a light-sheet target, seeded with the
**stationary-phase** analytic initial guess from
``references/Top Hat Beam.pdf`` (closed-form 1D top-hat phase + cylindrical
Fresnel lens for the perpendicular Gaussian).

The seed is a 2D match for ``light_sheet_target``: ``stationary_phase_1d``
along the along-line axis and an ABCD-derived cylindrical lens along the
perpendicular axis that widens the natural focal Gaussian
``lambda*f/(pi*w0)`` up to the target's perpendicular width.  Pre-CGM this
seed alone gives ~99.9% of the incident power inside the target region.

Seed vs Bowman, measured on this target (see
``scripts/sheet/benchmark_stationary_vs_bowman.py``):

    eta_min=0   :  stationary F=1.000 eta=2.32%   |   Bowman F=1.000 eta=0.19%
    eta_min=0.05:  stationary F=0.985 eta=3.21%   |   Bowman F=0.985 eta=3.21%
    eta_min=0.30:  stationary F=0.878 eta=4.88%   |   Bowman F=0.878 eta=4.88%

Two takeaways:

  1. Without an eta floor, the stationary seed delivers **12x more
     energy to the target region** at the same F=1 than Bowman, which
     converges to a degenerate "single pixel with flat phase" minimum.
  2. With an eta floor, both seeds tie at the same Pareto point --
     the floor is the binding constraint, not the seed.

This script defaults to ``eta_min=0`` to showcase (1).  Set ``ETA_MIN``
env var to override::

    # Showcase: F=1.0 at eta=2.32% (12x more efficient than Bowman)
    uv run python scripts/sheet/testfile_sheet_stationary.py

    # Baseline-matched for a direct A/B:
    ETA_MIN=0.3 uv run python scripts/sheet/testfile_sheet_stationary.py
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
PAYLOAD_PATH = f"{OUTPUT_DIR}/testfile_sheet_stationary_payload.npz"
PARAMS_PATH = f"{OUTPUT_DIR}/testfile_sheet_stationary_params.json"
PREVIEW_PATH = f"{OUTPUT_DIR}/testfile_sheet_stationary_preview.pdf"

# Measured incident-beam center on the SLM plane (um, relative to the
# SLM compute-grid geometric center).  Kept identical to the baseline
# testfile_sheet.py so the A/B diff is attributable to the seed alone.
BEAM_CENTER_DX_UM = 0
BEAM_CENTER_DY_UM = 0


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Capture parameters (used later by the Windows runner) ---
    etime_us = 20
    n_avg = 10
    LUT = 207
    fresnel_sd = 1000

    # --- Light-sheet target parameters (match testfile_sheet.py exactly) ---
    sheet_flat_width = 34       # px of uniform flat-top along line (~538 um)
    sheet_gaussian_sigma = 2.5  # px perpendicular Gaussian (~40 um)
    sheet_angle = 0             # horizontal
    sheet_edge_sigma = 0.1      # px soft Gaussian taper at ends of the flat region

    # --- CGM iteration budget & efficiency floor ---
    cgm_max_iterations = int(os.environ.get("CGM_MAX_ITERS", "2000"))
    cgm_steepness = 9
    # eta_min=0 (default) lets CGM exploit the seed's ray budget to reach
    # F=1.0 at eta~2.3% -- 12x better efficiency than Bowman at the same F.
    # Override with ETA_MIN=0.3 for a direct apples-to-apples A/B against
    # testfile_sheet.py (both reach the same F=0.878 eta=4.88% plateau).
    setting_eta = float(os.environ.get("ETA_MIN", "0.0"))

    # --- 1. SLM_class setup (reads hamamatsu_test_config.json) ---
    SLM = SLM_class()
    SLM.arraySizeBit = [10, 10]  # 1024 x 1024 compute grid
    SLM.image_init(
        initGaussianPhase_user_defined=np.zeros((1024, 1024)), Plot=False,
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
    targetAmp = SLM.light_sheet_target(
        flat_width=sheet_flat_width,
        gaussian_sigma=sheet_gaussian_sigma,
        angle=sheet_angle,
        edge_sigma=sheet_edge_sigma,
    )
    print(
        f"\n[TARGET] light sheet: flat_width={sheet_flat_width:.0f}px "
        f"gauss_sigma={sheet_gaussian_sigma:.0f}px edge_sigma={sheet_edge_sigma:.0f}px "
        f"dtype={targetAmp.dtype} shape={targetAmp.shape} "
        f"nonzero={np.count_nonzero(targetAmp)}"
    )

    # --- 3. Build the stationary-phase analytical initial phase ---
    # Closed-form 1D top-hat seed from references/Top Hat Beam.pdf, plus
    # a cylindrical Fresnel lens along the perpendicular axis to pre-
    # broaden the natural focal Gaussian (lambda*f/(pi*w0) ~10.6 um) up to
    # the target's perpendicular 1/e^2 intensity radius
    # (gaussian_sigma * sqrt(2) * Focalpitchy ~56 um).  Deep in the
    # geom-optics regime: 538 um flat-top vs 10.6 um diffraction = ~50x.
    init_phi = SLM.stationary_phase_sheet(
        flat_width=sheet_flat_width,
        gaussian_sigma=sheet_gaussian_sigma,
        angle=sheet_angle,
        center=None,
    )
    print(f"[SEED] stationary-phase initial guess: "
          f"min={init_phi.min():.3f} max={init_phi.max():.3f} rad")

    # --- 4. Run CGM on the compute grid via torch/CUDA ---
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
        eta_min=setting_eta,
        Plot=False,
    )
    cgm_wall_time = time.perf_counter() - t0
    per_iter_ms = cgm_wall_time / max(cgm_max_iterations, 1) * 1000
    print(f"[CGM] done in {cgm_wall_time:.2f} s "
          f"({per_iter_ms:.1f} ms/iter)")

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
        os.path.getsize(PAYLOAD_PATH) / 1024
    ))
    print(f"\n[SAVE] payload:  {PAYLOAD_PATH}  ({payload_size_kb} KB)")

    # --- 9. Save params.json ---
    params = {
        "algorithm": "CGM",
        "init_phase_method": "stationary_phase_2d",
        "eta_min": setting_eta,
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
        SLM_screen_final, F, eta, init_phi,
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


def _save_preview(input_amp, target, E_out, region, slm_screen_final,
                  F, eta, init_phi):
    """6-panel PDF: input / target intensity+phase / output intensity+phase / screen."""
    target_mask = mask_from_target(target)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(input_amp, cmap="viridis")
    axes[0, 0].set_title("Input amplitude |S|\n(Gaussian, 1024 grid)")

    axes[0, 1].imshow(np.abs(target) ** 2, cmap="hot")
    axes[0, 1].set_title("Target |tau|^2\n(light sheet)")

    # Swap target-phase panel for the stationary-phase initial guess,
    # so we can eyeball whether the seed looks like the expected
    # smooth 1D stationary-phase pattern.
    axes[0, 2].imshow(init_phi, cmap="twilight")
    axes[0, 2].set_title("Stationary-phase seed\n(pre-CGM init)")

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
