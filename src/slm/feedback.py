"""Simulated adaptive camera feedback for hologram correction (Kim et al.)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from slm.propagation import fft_propagate
from slm.wgs import WGSConfig, WGSResult, wgs


@dataclass
class FeedbackConfig:
    """Configuration for simulated feedback loop."""

    n_correction_steps: int = 5
    inner_iterations: int = 200
    phase_fix_iteration: int = 12
    noise_level: float = 0.02  # simulated camera noise (fraction of signal)


def simulate_camera_measurement(
    focal_field: np.ndarray,
    spot_positions: np.ndarray,
    noise_level: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate a noisy camera measurement of spot intensities.

    Returns measured intensities at each spot position.
    """
    if rng is None:
        rng = np.random.default_rng()
    spot_positions = np.asarray(spot_positions)
    intensities = np.abs(focal_field[spot_positions[:, 0], spot_positions[:, 1]]) ** 2
    noise = rng.normal(0, noise_level * np.mean(intensities), size=len(intensities))
    return np.maximum(intensities + noise, 0)


def adjust_target_weights(
    target: np.ndarray,
    measured_intensities: np.ndarray,
    spot_positions: np.ndarray,
) -> np.ndarray:
    """Adjust target amplitudes based on measured intensity non-uniformity.

    T^(j)(u_m) = sqrt(mean(I) / I(u_m)) * T^(j-1)(u_m)
    """
    new_target = target.copy()
    mean_intensity = np.mean(measured_intensities)
    spot_positions = np.asarray(spot_positions)
    rows, cols = spot_positions[:, 0], spot_positions[:, 1]
    valid = measured_intensities > 0
    correction = np.where(valid, np.sqrt(mean_intensity / np.maximum(measured_intensities, 1e-30)), 1.0)
    new_target[rows, cols] *= correction
    return new_target


def adaptive_feedback_loop(
    initial_field: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    spot_positions: np.ndarray,
    config: FeedbackConfig = FeedbackConfig(),
    aberration_phase: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    callback: Callable[[int, WGSResult], None] | None = None,
) -> list[WGSResult]:
    """Run the full adaptive feedback loop.

    For each correction step j:
        1. Run phase-fixed WGS with current target
        2. Simulate camera measurement (with optional aberration)
        3. Adjust target weights based on measured non-uniformity
        4. Use previous hologram phase as starting point for next step

    Parameters
    ----------
    initial_field : L_0, complex (ny, nx).
    target : initial target field.
    mask : binary mask of target positions.
    spot_positions : (N, 2) array of (row, col) positions.
    config : feedback parameters.
    aberration_phase : optional phase aberration to simulate optical imperfections.
    rng : random number generator for noise.
    callback : called after each correction step with (step, result).
    """
    if rng is None:
        rng = np.random.default_rng()

    results = []
    current_target = target.copy()
    current_field = initial_field.copy()
    slm_amp = np.abs(initial_field)

    for j in range(config.n_correction_steps):
        # Run phase-fixed WGS
        wgs_config = WGSConfig(
            n_iterations=config.inner_iterations,
            phase_fix_iteration=config.phase_fix_iteration,
        )
        result = wgs(current_field, current_target, mask, wgs_config)
        results.append(result)

        if callback is not None:
            callback(j, result)

        # Simulate what the camera would see
        # Apply aberration if present
        slm_field = slm_amp * np.exp(1j * result.slm_phase)
        if aberration_phase is not None:
            # Aberration distorts the focal plane
            focal_with_aberration = fft_propagate(
                slm_field * np.exp(1j * aberration_phase)
            )
        else:
            focal_with_aberration = result.focal_field

        measured = simulate_camera_measurement(
            focal_with_aberration, spot_positions, config.noise_level, rng
        )

        # Adjust target weights
        current_target = adjust_target_weights(
            current_target, measured, spot_positions
        )

        # Use previous result as starting phase for next iteration
        current_field = slm_amp * np.exp(1j * result.slm_phase)

    return results
