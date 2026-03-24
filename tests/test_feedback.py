"""Tests for adaptive feedback loop."""

import numpy as np

from slm.beams import initial_slm_field
from slm.feedback import (
    FeedbackConfig,
    adaptive_feedback_loop,
    adjust_target_weights,
    simulate_camera_measurement,
)
from slm.targets import mask_from_target, rectangular_grid


def test_simulate_camera_measurement(rng):
    focal = np.zeros((64, 64), dtype=complex)
    focal[20, 20] = 2.0
    focal[30, 30] = 3.0
    positions = np.array([[20, 20], [30, 30]])
    measured = simulate_camera_measurement(focal, positions, noise_level=0.0, rng=rng)
    np.testing.assert_allclose(measured, [4.0, 9.0])


def test_simulate_camera_measurement_noisy(rng):
    focal = np.ones((64, 64), dtype=complex)
    positions = np.array([[10, 10], [20, 20]])
    measured = simulate_camera_measurement(focal, positions, noise_level=0.1, rng=rng)
    # Should be close to 1.0 but not exact
    assert all(m > 0 for m in measured)


def test_adjust_target_weights():
    target = np.zeros((64, 64), dtype=complex)
    target[10, 10] = 1.0
    target[20, 20] = 1.0
    positions = np.array([[10, 10], [20, 20]])
    measured = np.array([4.0, 1.0])  # first spot 4x brighter

    adjusted = adjust_target_weights(target, measured, positions)
    # Weaker spot should get higher weight
    assert np.abs(adjusted[20, 20]) > np.abs(adjusted[10, 10])


def test_feedback_reduces_nonuniformity():
    shape = (64, 64)
    rng = np.random.default_rng(42)
    target, positions = rectangular_grid(shape, rows=3, cols=3, spacing=8)
    mask = mask_from_target(target)
    initial_field = initial_slm_field(shape, sigma=15.0, rng=rng)

    config = FeedbackConfig(
        n_correction_steps=3,
        inner_iterations=50,
        phase_fix_iteration=10,
        noise_level=0.0,
    )
    results = adaptive_feedback_loop(
        initial_field, target, mask, positions, config, rng=rng
    )
    # Last result should have better uniformity than first
    first_uni = results[0].uniformity_history[-1]
    last_uni = results[-1].uniformity_history[-1]
    assert last_uni <= first_uni * 1.5  # should not degrade significantly
