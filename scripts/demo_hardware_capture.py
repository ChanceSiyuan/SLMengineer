"""Single-shot SLM display + camera capture using the library API.

Equivalent of ~/slm-code/testfile.py but using the unified slm package.
Run this on the Windows lab machine with SLM and camera connected.

Usage:
    python scripts/demo_hardware_capture.py [config.json]
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from slm import (
    WGSConfig,
    initial_slm_field,
    mask_from_target,
    rectangular_grid,
    wgs,
)
from slm.camera import HardwareCamera
from slm.hardware import (
    HardwareConfig,
    combine_screens,
    fresnel_lens_phase,
    load_calibration_bmp,
)
from slm.hardware.slm_display import SLMDisplay
from slm.hardware.vimba_camera import VimbaCamera


def main(config_path: str | None = None) -> None:
    # Load hardware config
    if config_path:
        config = HardwareConfig.from_json(config_path)
    else:
        config = HardwareConfig()

    device = config.to_slm_device()
    n = min(device.n_pixels)

    # Generate input beam and 4x4 target
    field = initial_slm_field((n, n), sigma=config.beam_waist_um / config.pixel_pitch_um)
    target, positions = rectangular_grid((n, n), rows=4, cols=4, spacing=15)
    mask = mask_from_target(target)

    # Run WGS optimisation (pure simulation — no hardware needed)
    wgs_config = WGSConfig(
        n_iterations=config.loop,
        uniformity_threshold=config.threshold,
    )
    result = wgs(field, target, mask, wgs_config)
    print(f"WGS converged in {result.n_iterations} iterations")
    print(f"  uniformity: {result.uniformity_history[-1]:.4f}")

    # Optional: Fresnel lens for focal-plane shift
    fresnel_screen, _ = fresnel_lens_phase(
        slm_resolution=config.slm_resolution,
        pixel_pitch_um=config.pixel_pitch_um,
        focal_length_um=config.focal_length_um,
        wavelength_um=config.wavelength_um,
        shift_distance_um=20000,
    )

    # Load calibration if available
    calibration = None
    if config.calibration_bmp_path:
        calibration = load_calibration_bmp(config.calibration_bmp_path)

    # Open hardware and capture
    with SLMDisplay(monitor=config.monitor_index, is_image_lock=True) as display, \
         VimbaCamera() as cam:

        hw_camera = HardwareCamera(
            slm_display=display,
            camera=cam,
            slm_resolution=config.slm_resolution,
            exposure_time_us=config.exposure_time_us,
            settle_time_s=config.display_settle_time_s,
            lut_value=config.lut_value,
            calibration=calibration,
            fresnel_screen=fresnel_screen,
        )

        # This single call does: phase_to_screen -> LUT -> display -> capture
        image = hw_camera.capture_intensity(result.slm_phase)

    print(f"Captured: shape={image.shape}, max={image.max():.1f}, mean={image.mean():.1f}")

    # Save results
    stem = "hardware_capture"
    np.save(f"{stem}.npy", image)
    np.save(f"{stem}_phase.npy", result.slm_phase)
    meta = {
        "shape": list(image.shape),
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "mean": float(np.mean(image)),
        "exposure_us": config.exposure_time_us,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(f"{stem}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved to {stem}.npy / .json")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg)
