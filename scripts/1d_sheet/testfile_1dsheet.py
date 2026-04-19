"""Local-only WGS compute: produce a 4x4 square trap-array phase payload
for the dedicated Windows hardware runner.

Parallel to ``scripts/sheet/testfile_sheet.py`` but uses WGS
(``slm.wgs.WGS_phase_generate``) on a rectangular 4x4 lattice target
(``SLM.target_generate("Rec", arraysize=[4, 4], translate=True)``)
instead of CGM on a continuous shape.  Same Fresnel + calibration
post-processing and same payload format as the sheet workflow, so the
unmodified Windows runner can display it.

Next step after running this script::

    ./push_run.sh payload/wgs_square/testfile_wgs_square_payload.npz
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

from slm.generation import SLM_class
from slm import imgpy
from slm import wgs

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = "payload/wgs_square"
PAYLOAD_PATH = f"{OUTPUT_DIR}/testfile_wgs_square_payload.npz"
PARAMS_PATH = f"{OUTPUT_DIR}/testfile_wgs_square_params.json"
PREVIEW_PATH = f"{OUTPUT_DIR}/testfile_wgs_square_preview.pdf"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Capture parameters (used later by the Windows runner) ---
    etime_us = 4000        # 4 ms exposure
    n_avg = 10             # average 10 frames
    LUT = 207
    fresnel_sd = 1000      # um -- best Fresnel shift distance from sweep

    # --- WGS parameters ---
    array_nx, array_ny = 8, 8
    wgs_loops = 100
    wgs_threshold = 0.005
    # --- 1. SLM_class setup (4096^2 compute grid, matches legacy testfile.py) ---
    SLM = SLM_class()
    SLM.image_init(
        initGaussianPhase_user_defined=np.zeros((4096, 4096)),
        Plot=False,
    )
    W, H = SLM.SLMRes  # (1272, 1024)
    cx, cy = W // 2, H // 2

    correct = lambda screen: imgpy.SLM_screen_Correct(  # noqa: E731
        screen, LUT=LUT, correctionImgPath="calibration/CAL_LSH0905549_1013nm.bmp"
    )

    print(f"Compute grid: ({SLM.ImgResY}, {SLM.ImgResX})  "
          f"focal pitch = {SLM.Focalpitchx:.3f} um/px")
    print(f"SLM native:   ({H}, {W})")
    # 新增：目标相对零级光偏移（单位：focal plane pixel）
    target_shift_fpx = 200
    shift_um_x = target_shift_fpx * SLM.Focalpitchx
    shift_um_y = target_shift_fpx * SLM.Focalpitchy

    # --- 2. Generate 4x4 rectangular trap-array target ---
    targetAmp = SLM.target_generate(
        "Rec",
        arraysize=[array_nx, array_ny],
        distance=[-shift_um_x, -shift_um_y],
        translate=False,
        Plot=False,
    )
    print(
        f"\n[TARGET] 4x4 square lattice  dtype={targetAmp.dtype} "
        f"shift={shift_um_x:.3f} um, {shift_um_y:.3f} um"
        f"shape={targetAmp.shape} nonzero={np.count_nonzero(targetAmp)}"
    )

    # --- 3. Run WGS (zero-phase init, deterministic) ---
    init_phase = torch.zeros((4096, 4096), dtype=torch.float32)
    print(f"\n[WGS] running {wgs_loops} loops on 4096x4096 grid...")
    t0 = time.perf_counter()
    SLM_Phase = wgs.WGS_phase_generate(
        torch.tensor(SLM.initGaussianAmp),
        init_phase,
        torch.from_numpy(targetAmp),
        Loop=wgs_loops,
        threshold=wgs_threshold,
        Plot=False,
    )
    wgs_wall_time = time.perf_counter() - t0
    per_iter_ms = wgs_wall_time / max(wgs_loops, 1) * 1000
    print(f"[WGS] done in {wgs_wall_time:.2f} s ({per_iter_ms:.1f} ms/iter)")

    phase_np = SLM_Phase.cpu().clone().numpy()
    SLM_screen_raw = SLM.phase_to_screen(phase_np)

    # --- 4. Post-hoc Fresnel lens ---
    fresnel = SLM.fresnel_lens_phase_generate(fresnel_sd, cx, cy)[0]
    SLM_screen_shift = (
        (SLM_screen_raw.astype(np.int32) + fresnel.astype(np.int32)) % 256
    ).astype(np.uint8)

    # --- 5. Calibration correction ---
    SLM_screen_final = correct(SLM_screen_shift)
    print(f"\n[SCREEN] ready-to-display uint8 {SLM_screen_final.shape} "
          f"range=[{SLM_screen_final.min()}, {SLM_screen_final.max()}]")

    # --- 6. Diagnostic metrics ---
    from slm.metrics import fidelity as _fidelity, efficiency as _efficiency
    from slm.propagation import fft_propagate
    from slm.targets import measure_region as _measure_region

    region_np = _measure_region(targetAmp.shape, targetAmp, margin=5)
    E_out = fft_propagate(SLM.initGaussianAmp * np.exp(1j * phase_np))
    F = _fidelity(E_out, targetAmp, region_np)
    eta = _efficiency(E_out, region_np)
    print(f"[METRICS] fidelity F={F:.6f}  efficiency eta={eta*100:.2f}%")

    # --- 7. Save payload.npz ---
    np.savez_compressed(PAYLOAD_PATH, slm_screen=SLM_screen_final)
    payload_size_kb = int(np.ceil(os.path.getsize(PAYLOAD_PATH) / 1024))
    print(f"\n[SAVE] payload:  {PAYLOAD_PATH}  ({payload_size_kb} KB)")

    # --- 8. Save params.json ---
    params = {
        "algorithm": "WGS",
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
        "target": "rectangular_lattice",
        "array_nx": array_nx,
        "array_ny": array_ny,
        "wgs_loops": wgs_loops,
        "wgs_threshold": wgs_threshold,
        "wgs_wall_time_s": round(wgs_wall_time, 3),
        "wgs_per_iter_ms": round(per_iter_ms, 2),
        "final_fidelity": round(float(F), 6),
        "final_efficiency": round(float(eta), 6),
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] params:  {PARAMS_PATH}")

    # --- 9. Save preview.pdf ---
    _save_preview(
        SLM.initGaussianAmp, targetAmp, E_out, region_np,
        SLM_screen_final, F, eta,
    )
    print(f"[SAVE] preview: {PREVIEW_PATH}")

    # --- 10. Next-step hint ---
    print()
    print("=" * 72)
    print("Payload ready.  Next step (pushes to Windows and runs the experiment):")
    print()
    print(f"    ./push_run.sh {PAYLOAD_PATH}")
    print()
    print("=" * 72)


def _save_preview(input_amp, target, E_out, region, slm_screen_final, F, eta):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(input_amp, cmap="viridis")
    axes[0, 0].set_title("Input amplitude |S|\n(Gaussian, 4096 grid)")

    axes[0, 1].imshow(np.abs(target) ** 2, cmap="hot")
    axes[0, 1].set_title("Target |tau|^2\n(4x4 square lattice)")

    axes[0, 2].imshow(np.angle(target), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0, 2].set_title("Target phase arg(tau)")

    out_int = (np.abs(E_out) ** 2) * region
    axes[1, 0].imshow(out_int, cmap="hot")
    axes[1, 0].set_title(f"Output |E_out|^2\nF={F:.4f}, eta={100*eta:.2f}%")

    axes[1, 1].imshow(np.angle(E_out), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title("Output phase arg(E_out)")

    axes[1, 2].imshow(slm_screen_final, cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title(
        f"SLM screen (Fresnel+calib applied)\n{slm_screen_final.shape} uint8"
    )

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(PREVIEW_PATH, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
