"""SLM before/after test: CGM 4x4 trap array via the unified SLM_class
target API (issue #13).

CGM counterpart of ``scripts/testfile.py``.  The two scripts differ only
in the algorithm they call -- everything upstream (SLM_class setup,
target generation) and downstream (phase_to_screen, hardware capture)
is identical.  Since issue #13 unified target dtypes, the same
``SLM.target_generate("Rec", ...)`` feeds both WGS and CGM:

    SLM = SLM_class()
    SLM.image_init(...)
    targetAmp = SLM.target_generate("Rec", arraysize=[4, 4], translate=True)

    # WGS version (testfile.py):
    SLM_Phase = WGS_phase_generate(amp, InitPhase, tgt, Loop=50, threshold=0.005)

    # CGM version (this file):
    SLM_Phase = CGM_phase_generate(amp, InitPhase, tgt, max_iterations=300, ...)

    SLM_screen = SLM.phase_to_screen(SLM_Phase.cpu().clone().numpy())

Fresnel lens and SLM corrections are applied post-hoc to the SLM screen
in the same way as ``scripts/testfile.py`` (WGS): the Fresnel lens is
added via modular-256 arithmetic to the uint8 screen, then the combined
screen is passed through ``imgpy.SLM_screen_Correct`` with the
calibration BMP and LUT.  This matches the WGS workflow byte-for-byte.

Output files:
  - testfile_cgm_before.npy / testfile_cgm_after.npy  raw camera arrays
  - testfile_cgm_diff.npy                             difference image
  - testfile_cgm_run.json                             metadata + CGM params
"""
import json
import time
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slm.cgm import CGM_phase_generate
from slm.generation import SLM_class
from slm import imgpy
from slm.display import SLMdisplay
from slm.camera import VimbaCamera

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def capture_stats(image, label):
    """Return a dict of basic image statistics."""
    return {
        "label": label,
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "min": round(float(np.min(image)), 2),
        "max": round(float(np.max(image)), 2),
        "mean": round(float(np.mean(image)), 4),
        "std": round(float(np.std(image)), 4),
    }


def multi_capture(camera, etime, n_frames=10):
    """Average multiple frames to improve SNR."""
    acc = None
    for _ in range(n_frames):
        frame = camera.capture(etime)
        if acc is None:
            acc = np.zeros(frame.shape, dtype=np.float64)
        acc += frame.astype(np.float64)
    return (acc / n_frames).astype(np.float32)


def main():
    etime_us = 4000        # 4 ms exposure
    n_avg = 10             # average 10 frames
    LUT = 207
    fresnel_sd = 1000      # um -- compensates camera-focal-plane offset
    arraysize = [4, 4]     # 4x4 trap array (16 spots)

    # --- 1. SLM_class setup (reads hamamatsu_test_config.json) ---
    slm = SLMdisplay(monitor=1, isImageLock=True)
    SLM = SLM_class()
    SLM.image_init(initGaussianPhase_user_defined=np.zeros((4096, 4096)), Plot=False)

    W, H = SLM.SLMRes  # (1272, 1024)
    cx, cy = W // 2, H // 2

    correct = lambda screen: imgpy.SLM_screen_Correct(
        screen, LUT=LUT, correctionImgPath="calibration/CAL_LSH0905549_1013nm.bmp"
    )

    # --- 2. Display blank, capture "before" ---
    blank_screen = np.zeros((H, W), dtype=np.uint8)
    slm.updateArray(blank_screen)
    time.sleep(0.5)

    with VimbaCamera() as camera:
        img_before = multi_capture(camera, etime_us, n_avg)
    print(f"[BEFORE] shape: {img_before.shape} max: {img_before.max():.1f} mean: {img_before.mean():.2f}")

    # --- 3. Generate 4x4 trap array target (complex128 after issue #13) ---
    targetAmp = SLM.target_generate(
        "Rec", arraysize=arraysize, translate=True, Plot=False,
    )
    print(
        f"[TARGET] 4x4 trap array: dtype={targetAmp.dtype} shape={targetAmp.shape} "
        f"nonzero={np.count_nonzero(targetAmp)}"
    )

    # --- 4. Run CGM via the unified torch API (parallel to WGS_phase_generate) ---
    input_shape = (4096, 4096)  # 输入振幅分布的尺寸，与 testfile.py 一致
    InitPhase = torch.zeros(input_shape, dtype=torch.float32)

    SLM_Phase = CGM_phase_generate(
        torch.tensor(SLM.initGaussianAmp),
        InitPhase,
        torch.from_numpy(targetAmp),
        max_iterations=300,
        steepness=9,
        eta_min=0.05,
        Plot=True,
    )

    # Wrap phase to [-pi, pi] before cropping (CGM may accumulate phase
    # outside this range).
    phase_np = SLM_Phase.cpu().clone().numpy()
    phase_wrapped = np.angle(np.exp(1j * phase_np))
    SLM_screen = SLM.phase_to_screen(phase_wrapped)

    # Add Fresnel lens
    fresnel = SLM.fresnel_lens_phase_generate(fresnel_sd, cx, cy)[0]
    SLM_screen_shift = (
        (SLM_screen.astype(np.int32) + fresnel.astype(np.int32)) % 256
    ).astype(np.uint8)
    SLM_screen_final = correct(SLM_screen_shift)

    slm.updateArray(SLM_screen_final)
    time.sleep(0.5)

    with VimbaCamera() as camera:
        img_after = multi_capture(camera, etime_us, n_avg)
    print(f"[AFTER]  shape: {img_after.shape} max: {img_after.max():.1f} mean: {img_after.mean():.2f}")

    # --- 6. Save results (npy + png for each capture) ---
    np.save("testfile_cgm_before.npy", img_before)
    np.save("testfile_cgm_after.npy", img_after)

    diff = img_after - img_before
    np.save("testfile_cgm_diff.npy", diff)

    for name, img in [
        ("testfile_cgm_before", img_before),
        ("testfile_cgm_after", img_after),
        ("testfile_cgm_diff", diff),
    ]:
        fig, ax = plt.subplots(figsize=(10, 8))
        if name == "testfile_cgm_diff":
            im = ax.imshow(img, cmap='RdBu_r', vmin=img.min(), vmax=img.max())
        else:
            im = ax.imshow(img, cmap='hot', vmin=img.min(), vmax=img.max())
        ax.set_title(
            f"{name}\nmin={img.min():.1f} max={img.max():.1f} mean={img.mean():.2f}"
        )
        plt.colorbar(im, ax=ax)
        fig.savefig(f"{name}.png", dpi=150)
        plt.close(fig)

    run_meta = {
        "exposure_us": etime_us,
        "n_avg_frames": n_avg,
        "LUT": LUT,
        "fresnel_shift_distance": fresnel_sd,
        "fresnel_applied_post_hoc": True,
        "arraysize": arraysize,
        "translate": True,
        "algorithm": "CGM",
        "cgm_max_iterations": 300,
        "cgm_steepness": 9,
        "cgm_eta_min": 0.05,
        "compute_grid": list(input_shape),
        "focal_pitch_x_um_per_px": round(float(SLM.Focalpitchx), 4),
        "focal_pitch_y_um_per_px": round(float(SLM.Focalpitchy), 4),
        "monitor": 1,
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "before": capture_stats(img_before, "SLM blank"),
        "after": capture_stats(
            img_after, f"SLM CGM 4x4 + Fresnel(sd={fresnel_sd})"
        ),
        "diff_max": round(float(diff.max()), 2),
        "diff_min": round(float(diff.min()), 2),
    }
    with open("testfile_cgm_run.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print("\n=== Run metadata ===")
    print(json.dumps(run_meta, indent=2))

    slm.close()


if __name__ == "__main__":
    main()
