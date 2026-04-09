"""SLM before/after test: CGM top-hat beam shaping on real hardware.

Generates a flat-top (top-hat) beam profile using CGM, displays on the SLM,
and captures with the camera. Follows the same pattern as testfile.py.

Fixes applied from issue #12 investigation:
  - Fresnel lens baked into CGM initial_phase (not added post-hoc)
  - No calibration BMP (destroys CGM phase structure) — LUT-only correction
  - Target shifted off-center to separate from zero-order

Output files:
  - tophat_before.npy / tophat_after.npy   raw camera arrays (float32, averaged)
  - tophat_diff.npy                         difference image
  - tophat_run.json                         metadata + CGM metrics
"""
import json
import time
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slm.beams import gaussian_beam
from slm.cgm import tophat_phase_generate
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
    etime_us = 4000    # 4 ms exposure
    n_avg = 10
    LUT = 207

    # --- Hardware parameters ---
    SLM_RES = (1272, 1024)   # (width, height)
    W, H = SLM_RES
    pixelpitch = 12.5        # um
    wavelength = 1.013       # um
    focallength = 200000     # um
    magnification = 1.0
    fresnel_sd = 1000        # um — compensates camera-focal-plane offset

    # --- 1. Open SLM on monitor 1, display blank, capture "before" ---
    slm = SLMdisplay(monitor=1, isImageLock=True)
    blank_screen = np.zeros((H, W), dtype=np.uint8)
    slm.updateArray(blank_screen)
    time.sleep(0.5)

    with VimbaCamera() as camera:
        img_before = multi_capture(camera, etime_us, n_avg)
    print(f"[BEFORE] shape: {img_before.shape} max: {img_before.max():.1f} mean: {img_before.mean():.2f}")

    # --- 2. Generate top-hat phase via CGM ---
    shape = (H, W)  # (1024, 1272) — SLM native resolution

    # Gaussian input beam: beamwaist=5000um / pixelpitch=12.5um = 400 px
    input_amp = gaussian_beam(shape, sigma=400.0, normalize=True)

    # Bake Fresnel lens into initial phase so CGM accounts for the defocus
    rows = np.arange(H) - H / 2.0
    cols = np.arange(W) - W / 2.0
    cc, rr = np.meshgrid(cols * pixelpitch, rows * pixelpitch)
    fresnel_phase = np.pi * (cc**2 + rr**2) * fresnel_sd / (wavelength * focallength**2) * magnification**2

    # Top-hat: radius=50px, shifted 100px right to separate from zero-order
    SLM_Phase = tophat_phase_generate(
        input_amp, radius=50.0,
        center=(H / 2.0, W / 2.0 + 100),
        max_iterations=300, steepness=9,
        initial_phase=fresnel_phase, Plot=True,
    )

    # --- 3. Convert phase to SLM screen ---
    SLM_Phase_wrapped = np.angle(np.exp(1j * SLM_Phase))
    SLM_screen = np.around((SLM_Phase_wrapped + np.pi) / (2 * np.pi) * 256).astype(np.uint8)

    # LUT scaling only (no calibration BMP — it destroys CGM phase, see issue #12)
    SLM_screen_final = (SLM_screen.astype(np.float64) / 256 * LUT).astype(np.uint8)

    slm.updateArray(SLM_screen_final)
    time.sleep(0.5)

    with VimbaCamera() as camera:
        img_after = multi_capture(camera, etime_us, n_avg)
    print(f"[AFTER]  shape: {img_after.shape} max: {img_after.max():.1f} mean: {img_after.mean():.2f}")

    # --- 4. Save results ---
    np.save("tophat_before.npy", img_before)
    np.save("tophat_after.npy", img_after)

    diff = img_after - img_before
    np.save("tophat_diff.npy", diff)

    for name, img in [("tophat_before", img_before), ("tophat_after", img_after), ("tophat_diff", diff)]:
        fig, ax = plt.subplots(figsize=(10, 8))
        if name == "tophat_diff":
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
        "LUT": LUT,
        "fresnel_sd_baked": fresnel_sd,
        "tophat_radius_px": 50.0,
        "tophat_shift_px": 100,
        "monitor": 1,
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "before": capture_stats(img_before, "SLM blank"),
        "after": capture_stats(img_after, "SLM CGM top-hat"),
        "diff_max": round(float(diff.max()), 2),
        "diff_min": round(float(diff.min()), 2),
    }
    with open("tophat_run.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print(f"\n=== Run metadata ===")
    print(json.dumps(run_meta, indent=2))

    slm.close()


if __name__ == "__main__":
    main()
