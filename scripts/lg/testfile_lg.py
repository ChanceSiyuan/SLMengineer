"""Local-only CGM compute: produce an LG^0_1 phase payload for the
dedicated Windows hardware runner.

This script runs ENTIRELY on the local (Linux+RTX 3090) box.  It does:

  1. SLM_class setup on the 4096x4096 computation grid.
  2. LG^0_1 donut target via ``SLM.lg_mode_target``.
  3. Bowman-analytical initial phase ``phi_0 = R*(p^2+q^2) + D*(p cos t + q sin t)``
     via ``slm.cgm._initial_phase``.
  4. CGM on torch/CUDA (``CGM_phase_generate``, ~100 s on 4096^2).
  5. ``SLM.phase_to_screen`` to crop the 4096^2 phase down to the SLM native
     resolution (1024, 1272) uint8.
  6. Post-hoc Fresnel lens (via ``SLM.fresnel_lens_phase_generate``) added to
     the screen modulo 256, matching the ``scripts/testfile.py`` WGS workflow.
  7. Calibration correction via ``slm.imgpy.SLM_screen_Correct`` with the
     LUT + calibration BMP -- matches the testfile.py pipeline byte-for-byte.
  8. Saves three artefacts under ``payload/lg/``:

       payload/lg/testfile_lg_payload.npz      (slm_screen uint8 + diagnostics)
       payload/lg/testfile_lg_params.json      (human-readable metadata)
       payload/lg/testfile_lg_preview.pdf      (6-panel visualisation)

The **Windows hardware runner** lives in a separate, lightweight repo at
``C:\\Users\\Galileo\\slm_runner\\`` and simply loads the payload .npz,
displays the pre-computed uint8 screen on the SLM, captures the camera
before/after, and returns the data to Linux.  All CGM and correction
logic stays on this (Linux) box.

Next step after running this script::

    ./push_run.sh payload/lg/testfile_lg_payload.npz
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

OUTPUT_DIR = "payload/lg"
PAYLOAD_PATH = f"{OUTPUT_DIR}/testfile_lg_payload.npz"
PARAMS_PATH = f"{OUTPUT_DIR}/testfile_lg_params.json"
PREVIEW_PATH = f"{OUTPUT_DIR}/testfile_lg_preview.pdf"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # --- Capture parameters (used later by the Windows runner) ---
    etime_us = 4000        # 4 ms exposure
    n_avg = 10             # average 10 frames
    LUT = 207
    fresnel_sd = 1000      # um -- compensates camera-focal-plane offset

    # --- LG mode parameters (Bowman et al. top-hat.tex Table I) ---
    lg_ell = 1             # topological charge (vortex index); LG^0_1 donut
    lg_p = 0               # radial index
    lg_w0 = 100.0          # beam waist in focal-plane pixels
                           # (= 396 um on the 4096-grid, focal pitch 3.96 um/px)

    # --- CGM analytical initial phase (paper's LG^0_1 optimum, Table I) ---
    cgm_R = 4.5e-3         # quadratic curvature (rad/px^2)
    cgm_D = -np.pi / 2     # linear phase offset magnitude
    cgm_theta = np.pi / 4  # linear phase angle (diagonal, shifts off zero-order)
    cgm_steepness = 9
    cgm_max_iterations = 200

    # --- 1. SLM_class setup (reads hamamatsu_test_config.json) ---
    SLM = SLM_class()
    SLM.image_init(
        initGaussianPhase_user_defined=np.zeros((4096, 4096)), Plot=False,
    )
    W, H = SLM.SLMRes  # (1272, 1024)
    cx, cy = W // 2, H // 2

    correct = lambda screen: imgpy.SLM_screen_Correct(  # noqa: E731
        screen, LUT=LUT, correctionImgPath="calibration/CAL_LSH0905549_1013nm.bmp"
    )

    print(f"Compute grid: ({SLM.ImgResY}, {SLM.ImgResX})  "
          f"focal pitch = {SLM.Focalpitchx:.3f} um/px")
    print(f"SLM native:   ({H}, {W})")

    # --- 2. Generate LG^0_1 donut target on the 4096x4096 computation grid ---
    targetAmp = SLM.lg_mode_target(ell=lg_ell, p=lg_p, w0=lg_w0)
    print(
        f"\n[TARGET] LG^{lg_p}_{lg_ell} donut: w0={lg_w0:.0f}px "
        f"dtype={targetAmp.dtype} shape={targetAmp.shape} "
        f"nonzero={np.count_nonzero(targetAmp)}"
    )

    # --- 3. Build the paper's analytical initial phase ---
    init_phi = _initial_phase(
        (int(SLM.ImgResY), int(SLM.ImgResX)),
        CGMConfig(R=cgm_R, D=cgm_D, theta=cgm_theta),
    )

    # --- 4. Run CGM on 4096^2 via torch/CUDA ---
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
        Plot=False,
    )
    cgm_wall_time = time.perf_counter() - t0
    per_iter_ms = cgm_wall_time / max(cgm_max_iterations, 1) * 1000
    print(f"[CGM] done in {cgm_wall_time:.2f} s "
          f"({per_iter_ms:.1f} ms/iter)")

    # Wrap phase to [-pi, pi] before cropping (CGM may accumulate phase
    # outside this range).
    phase_np = SLM_Phase.cpu().clone().numpy()
    phase_wrapped = np.angle(np.exp(1j * phase_np))
    SLM_screen_raw = SLM.phase_to_screen(phase_wrapped)

    # --- 5. Post-hoc Fresnel lens (matches testfile.py pipeline) ---
    fresnel = SLM.fresnel_lens_phase_generate(fresnel_sd, cx, cy)[0]
    SLM_screen_shift = (
        (SLM_screen_raw.astype(np.int32) + fresnel.astype(np.int32)) % 256
    ).astype(np.uint8)

    # --- 6. Calibration correction (LUT + calibration BMP, CPU-only) ---
    SLM_screen_final = correct(SLM_screen_shift)
    print(f"\n[SCREEN] ready-to-display uint8 {SLM_screen_final.shape} "
          f"range=[{SLM_screen_final.min()}, {SLM_screen_final.max()}]")

    # --- 7. Diagnostic metrics (final fidelity, efficiency on the 4096 grid) ---
    # Compute final metrics by re-evaluating E_out from the saved phase on
    # the 4096 grid.  Cheap compared to the CGM loop.
    from slm.metrics import fidelity as _fidelity, efficiency as _efficiency
    from slm.propagation import fft_propagate
    from slm.targets import measure_region as _measure_region

    region_np = _measure_region(targetAmp.shape, targetAmp, margin=5)
    E_out_4096 = fft_propagate(SLM.initGaussianAmp * np.exp(1j * phase_wrapped))
    F = _fidelity(E_out_4096, targetAmp, region_np)
    eta = _efficiency(E_out_4096, region_np)
    print(f"[METRICS] fidelity F={F:.6f}  (1-F={1-F:.2e})  efficiency eta={eta*100:.2f}%")

    # --- 8. Save payload.npz (MINIMAL: just the uint8 SLM screen that the
    #        Windows runner will upload -- kept small enough for fast scp).
    #        Diagnostic arrays (target/output fields) are NOT included; they
    #        stay local and only appear in the preview PDF below.
    np.savez_compressed(
        PAYLOAD_PATH,
        slm_screen=SLM_screen_final,  # uint8 (1024, 1272) ready to display
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
        # capture parameters for the Windows runner
        "runner_defaults": {
            "etime_us": etime_us,
            "n_avg": n_avg,
            "monitor": 1,
        },
        # what was already applied on the Linux side
        "fresnel_applied_on_linux": True,
        "fresnel_shift_distance_um": fresnel_sd,
        "fresnel_center_xy_px": [int(cx), int(cy)],
        "calibration_applied_on_linux": True,
        "calibration_bmp": "calibration/CAL_LSH0905549_1013nm.bmp",
        "LUT": LUT,
        # target
        "target": "LG mode",
        "lg_ell": lg_ell,
        "lg_p": lg_p,
        "lg_w0_px": lg_w0,
        "lg_w0_um_focal": round(float(lg_w0 * SLM.Focalpitchx), 2),
        # CGM parameters
        "cgm_max_iterations": cgm_max_iterations,
        "cgm_steepness": cgm_steepness,
        "cgm_R": cgm_R,
        "cgm_D": cgm_D,
        "cgm_theta": cgm_theta,
        # compute results
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

    # --- 10. Save preview.pdf (6-panel visual) ---
    _save_preview(
        SLM.initGaussianAmp, targetAmp, E_out_4096, region_np,
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
    """6-panel PDF: input / target intensity+phase / output intensity+phase / cost."""
    target_mask = mask_from_target(target)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(input_amp, cmap="viridis")
    axes[0, 0].set_title("Input amplitude |S|\n(Gaussian, 4096 grid)")

    axes[0, 1].imshow(np.abs(target) ** 2, cmap="hot")
    axes[0, 1].set_title("Target |tau|^2\n(LG^0_1 donut)")

    target_phase_masked = np.where(target_mask > 0, np.angle(target), np.nan)
    axes[0, 2].imshow(target_phase_masked, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0, 2].set_title("Target phase arg(tau)\n(vortex ell=1)")

    out_int = (np.abs(E_out) ** 2) * region
    axes[1, 0].imshow(out_int, cmap="hot")
    axes[1, 0].set_title(
        f"Output |E_out|^2\nF={F:.4f}, eta={100*eta:.2f}%"
    )

    out_phase_masked = np.where(target_mask > 0, np.angle(E_out), np.nan)
    axes[1, 1].imshow(out_phase_masked, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title("Output phase arg(E_out)")

    # Final uint8 screen that will be uploaded to the SLM
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
