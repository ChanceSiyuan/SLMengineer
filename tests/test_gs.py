"""Tests for Gerchberg-Saxton algorithm."""

import numpy as np

from slm.gs import gs


def test_gs_converges(gaussian_field_64, four_spot_target_64):
    target, positions = four_spot_target_64
    result = gs(gaussian_field_64, target, n_iterations=50)
    # Uniformity should improve (decrease) over iterations
    assert result.uniformity_history[-1] < result.uniformity_history[0]


def test_gs_preserves_slm_amplitude(gaussian_field_64, four_spot_target_64):
    target, _ = four_spot_target_64
    slm_amp = np.abs(gaussian_field_64)
    result = gs(gaussian_field_64, target, n_iterations=20)
    # Reconstruct final SLM field
    final_field = slm_amp * np.exp(1j * result.slm_phase)
    np.testing.assert_allclose(np.abs(final_field), slm_amp, atol=1e-10)


def test_gs_energy_conservation(gaussian_field_64, four_spot_target_64):
    target, _ = four_spot_target_64
    result = gs(gaussian_field_64, target, n_iterations=50)
    # Total power should be conserved (Parseval)
    input_power = np.sum(np.abs(gaussian_field_64) ** 2)
    output_power = np.sum(np.abs(result.focal_field) ** 2)
    np.testing.assert_allclose(output_power, input_power, rtol=1e-6)


def test_gs_callback(gaussian_field_64, four_spot_target_64):
    target, _ = four_spot_target_64
    iterations_seen = []

    def callback(i, slm_field, focal_field):
        iterations_seen.append(i)

    gs(gaussian_field_64, target, n_iterations=10, callback=callback)
    assert iterations_seen == list(range(10))
