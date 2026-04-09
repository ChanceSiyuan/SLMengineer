"""SLM before/after test: capture camera image with SLM blank, then with
WGS 4x4 trap array + Fresnel lens, and save both for comparison.

Fixes applied:
  - monitor=1 (the actual SLM, not the laptop display)
  - Must run in interactive Session 1 via ai_slm_loop.sh (schtasks bridge)
  - Fresnel shift_distance=1000 (best focus from parameter sweep)
  - Zero phase initialization (deterministic, good convergence)
  - WGS Loop=50 for adequate convergence
  - 10-frame averaging for better SNR

Output files:
  - testfile_before.npy / testfile_after.npy   raw camera arrays (float32, averaged)
  - testfile_run.json                           metadata for both captures
"""
import json
import time
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slm.generation import SLM_class
from slm import imgpy
from slm import wgs
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
    etime_us = 4000    # 4 ms exposure
    n_avg = 10          # average 10 frames
    fresnel_sd = 1000   # best Fresnel shift distance from sweep (beamwaist=5000)

    # --- 1. Open SLM on monitor 1 (the actual SLM hardware) ---
    slm = SLMdisplay(monitor=1, isImageLock=True)
    SLM = SLM_class()
    SLM.image_init(initGaussianPhase_user_defined=np.zeros((4096, 4096)), Plot=False)

    W, H = SLM.SLMRes
    cx, cy = W // 2, H // 2

    # Display blank and capture "before"
    blank_screen = np.zeros((H, W), dtype=np.uint8)
    slm.updateArray(blank_screen)
    time.sleep(0.5)

    with VimbaCamera() as camera:
        img_before = multi_capture(camera, etime_us, n_avg)
    print(f"[BEFORE] shape: {img_before.shape} max: {img_before.max():.1f} mean: {img_before.mean():.2f}")

# --- 2. Generate WGS 4x4 trap array + Fresnel lens ---
    targetAmp = SLM.target_generate("Rec", arraysize=[4, 4], translate=True, Plot=False)

    input_shape = (4096, 4096)  # 输入振幅分布的尺寸

    InitPhase = torch.zeros(input_shape, dtype=torch.float32)

    SLM_Phase=wgs.WGS_phase_generate(torch.tensor(SLM.initGaussianAmp), InitPhase, torch.from_numpy(targetAmp), Loop=50, threshold=0.005, Plot=True)
    SLM_screen=SLM.phase_to_screen(SLM_Phase.cpu().clone().numpy())

    LUT = 207
    correct = lambda screen: imgpy.SLM_screen_Correct(
        screen, LUT=LUT, correctionImgPath="calibration/CAL_LSH0905549_1013nm.bmp"
    )

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

    # --- 3. Save results (npy + png for each capture) ---
    np.save("testfile_before.npy", img_before)
    np.save("testfile_after.npy", img_after)

    diff = img_after - img_before
    np.save("testfile_diff.npy", diff)

    # Generate PNG previews
    for name, img in [("testfile_before", img_before), ("testfile_after", img_after), ("testfile_diff", diff)]:
        fig, ax = plt.subplots(figsize=(10, 8))
        if name == "testfile_diff":
            im = ax.imshow(img, cmap='RdBu_r', vmin=img.min(), vmax=img.max())
        else:
            im = ax.imshow(img, cmap='hot', vmin=img.min(), vmax=img.max())
        ax.set_title(f"{name}\nmin={img.min():.1f} max={img.max():.1f} mean={img.mean():.2f}")
        plt.colorbar(im, ax=ax)
        plt.savefig(f"{name}.png", dpi=150)
        plt.close()

    run_meta = {
        "exposure_us": etime_us,
        "n_avg_frames": n_avg,
        "fresnel_shift_distance": fresnel_sd,
        "monitor": 1,
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "before": capture_stats(img_before, "SLM blank"),
        "after": capture_stats(img_after, f"SLM WGS 4x4 + Fresnel(sd={fresnel_sd})"),
        "diff_max": round(float(diff.max()), 2),
        "diff_min": round(float(diff.min()), 2),
    }
    with open("testfile_run.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print(f"\n=== Run metadata ===")
    print(json.dumps(run_meta, indent=2))

    slm.close()


if __name__ == "__main__":
    main()
