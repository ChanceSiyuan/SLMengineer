"""Adaptive camera feedback for hologram correction.

Includes both discrete-spot feedback (Kim et al.) and continuous-pattern
feedback for top-hat / light-sheet targets.
"""

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
    correction = np.where(
        valid, np.sqrt(mean_intensity / np.maximum(measured_intensities, 1e-30)), 1.0
    )
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
        current_target = adjust_target_weights(current_target, measured, spot_positions)

        # Use previous result as starting phase for next iteration
        current_field = slm_amp * np.exp(1j * result.slm_phase)

    return results


# ---------------------------------------------------------------------------
# Continuous-pattern feedback (top-hat, light sheet, etc.)
# ---------------------------------------------------------------------------


def simulate_continuous_measurement(
    focal_field: np.ndarray,
    region_mask: np.ndarray,
    noise_level: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate a noisy camera measurement of a continuous focal-plane field.

    Returns the full (ny, nx) measured intensity with Gaussian noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    intensity = np.abs(focal_field) ** 2
    mean_signal = np.mean(intensity[region_mask > 0]) if np.any(region_mask > 0) else 1.0
    noise = rng.normal(0, noise_level * mean_signal, size=intensity.shape)
    return np.maximum(intensity + noise, 0.0)


def adjust_target_continuous(
    target: np.ndarray,
    measured_intensity: np.ndarray,
    region_mask: np.ndarray,
) -> np.ndarray:
    """Adjust target amplitude using measured continuous intensity field.

    Scales the target amplitude so under-illuminated regions get more
    weight and over-illuminated regions get less, within the measure region.
    """
    new_target = target.copy()
    mask = region_mask > 0
    if not np.any(mask):
        return new_target

    I_meas = np.maximum(measured_intensity[mask], 1e-30)
    I_mean = np.mean(I_meas)
    correction = np.sqrt(np.clip(I_mean / I_meas, 0.1, 10.0))
    new_target[mask] *= correction
    return new_target


def adaptive_feedback_continuous(
    input_amplitude: np.ndarray,
    target_field: np.ndarray,
    measure_region: np.ndarray,
    n_steps: int = 5,
    max_iter: int = 100,
    noise_level: float = 0.02,
    rng: np.random.Generator | None = None,
) -> list:
    """Closed-loop feedback for continuous targets using a simulated camera.

    Convenience wrapper around :func:`experimental_feedback_loop` that
    constructs a :class:`~slm.camera.SimulatedCamera` internally.
    """
    from slm.camera import SimulatedCamera

    camera = SimulatedCamera(
        input_amplitude, noise_level=noise_level, rng=rng,
    )
    return experimental_feedback_loop(
        input_amplitude, target_field, measure_region, camera,
        n_steps=n_steps, max_iter=max_iter,
    )


def experimental_feedback_loop(
    input_amplitude: np.ndarray,
    target_field: np.ndarray,
    measure_region: np.ndarray,
    camera,
    n_steps: int = 5,
    max_iter: int = 100,
    measure_phase: bool = False,
    cgm_config: "CGMConfig | None" = None,
) -> list:
    """Closed-loop feedback using a camera (real or simulated).

    Each step:
      1. Run CGM to produce a hologram
      2. Display hologram -> camera captures intensity (and optionally fringes)
      3. Adjust target based on measured intensity
      4. Optionally apply phase correction from fringe analysis
      5. Warm-start next CGM from previous hologram

    Parameters
    ----------
    camera : object with ``capture_intensity(slm_phase)`` and optionally
        ``capture_fringes(slm_phase)`` methods (see CameraInterface protocol).
    measure_phase : if True, also capture fringes and extract phase via Takeda.
    cgm_config : base CGMConfig for each step (max_iterations is overridden
        by *max_iter*).  If None, a default config with eta_min=0.05 is used.
    """
    from slm.cgm import CGMConfig, cgm

    results = []
    current_target = target_field.copy()
    prev_phase = None

    for _step in range(n_steps):
        if cgm_config is not None:
            config = CGMConfig(**{
                k: v for k, v in cgm_config.__dict__.items()
                if k != "initial_phase"
            })
        else:
            config = CGMConfig(eta_min=0.05)
        config.max_iterations = max_iter
        if prev_phase is not None:
            config.initial_phase = prev_phase

        result = cgm(input_amplitude, current_target, measure_region, config)
        results.append(result)

        measured_I = camera.capture_intensity(result.slm_phase)
        current_target = adjust_target_continuous(
            current_target, measured_I, measure_region,
        )

        if measure_phase and hasattr(camera, "capture_fringes"):
            from slm.camera import takeda_phase_retrieval

            fringes = camera.capture_fringes(result.slm_phase)
            measured_phase = takeda_phase_retrieval(fringes)
            target_phase = np.angle(target_field)
            mask = measure_region > 0
            phase_correction = np.zeros_like(result.slm_phase)
            phase_correction[mask] = target_phase[mask] - measured_phase[mask]
            prev_phase = result.slm_phase + phase_correction
        else:
            prev_phase = result.slm_phase

    return results
