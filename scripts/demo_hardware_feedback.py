"""Closed-loop hardware feedback using the same code as simulation.

Demonstrates that experimental_feedback_loop works identically with
HardwareCamera (real device) or SimulatedCamera (simulation). The
algorithms are unchanged — only the camera backend differs.

Run on the Windows lab machine with SLM and camera connected.

Usage:
    python scripts/demo_hardware_feedback.py [config.json]
"""

from __future__ import annotations

import sys

import numpy as np

from slm import (
    cgm,
    CGMConfig,
    initial_slm_field,
    measure_region,
    rectangular_grid,
    mask_from_target,
    fidelity,
    efficiency,
)
from slm.camera import HardwareCamera
from slm.feedback import experimental_feedback_loop
from slm.hardware import (
    HardwareConfig,
    fresnel_lens_phase,
    load_calibration_bmp,
)
from slm.hardware.slm_display import SLMDisplay
from slm.hardware.vimba_camera import VimbaCamera


def main(config_path: str | None = None) -> None:
    # Load config
    config = HardwareConfig.from_json(config_path) if config_path else HardwareConfig()
    device = config.to_slm_device()
    n = min(device.n_pixels)

    # Setup target
    field = initial_slm_field((n, n), sigma=config.beam_waist_um / config.pixel_pitch_um)
    input_amp = np.abs(field)
    target, positions = rectangular_grid((n, n), rows=4, cols=4, spacing=15)
    mask = mask_from_target(target)
    mregion = measure_region((n, n), target)

    # Fresnel lens + calibration
    fresnel_screen, _ = fresnel_lens_phase(
        slm_resolution=config.slm_resolution,
        pixel_pitch_um=config.pixel_pitch_um,
        focal_length_um=config.focal_length_um,
        wavelength_um=config.wavelength_um,
        shift_distance_um=20000,
    )
    calibration = None
    if config.calibration_bmp_path:
        calibration = load_calibration_bmp(config.calibration_bmp_path)

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

        # Run the SAME feedback loop used in simulation — just with a real camera
        results = experimental_feedback_loop(
            input_amplitude=input_amp,
            target_field=target,
            measure_region=mregion,
            camera=hw_camera,
            n_steps=5,
            max_iter=100,
        )

    # Report results
    for i, r in enumerate(results):
        print(
            f"Step {i}: cost={r.cost_history[-1]:.6f}  "
            f"fidelity={r.final_fidelity:.4f}  "
            f"efficiency={r.final_efficiency:.4f}"
        )

    # Save final hologram
    np.save("feedback_final_phase.npy", results[-1].slm_phase)
    print("Done. Final phase saved to feedback_final_phase.npy")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg)
