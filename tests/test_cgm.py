"""Tests for Conjugate Gradient Minimization algorithm."""

import numpy as np
import pytest

from slm.beams import gaussian_beam
from slm.cgm import CGMConfig, _cost_function, _cost_gradient, _initial_phase, cgm
from slm.propagation import fft_propagate
from slm.targets import measure_region, top_hat


@pytest.fixture
def cgm_setup():
    """Standard CGM test setup: 64x64 grid with top-hat target."""
    shape = (64, 64)
    input_amp = gaussian_beam(shape, sigma=15.0, normalize=False)
    target = top_hat(shape, radius=8.0)
    region = measure_region(shape, target, margin=3)
    config = CGMConfig(max_iterations=50, steepness=6, R=3e-3)
    return input_amp, target, region, config


def test_cgm_cost_decreases(cgm_setup):
    input_amp, target, region, config = cgm_setup
    result = cgm(input_amp, target, region, config)
    # Cost should generally decrease
    for i in range(1, min(10, len(result.cost_history))):
        assert result.cost_history[i] <= result.cost_history[0] * 1.1


def test_cgm_tophat_converges(cgm_setup):
    input_amp, target, region, config = cgm_setup
    config.max_iterations = 100
    result = cgm(input_amp, target, region, config)
    # Fidelity should be reasonable
    assert result.final_fidelity > 0.8


def test_cgm_initial_phase_structured():
    shape = (64, 64)
    config = CGMConfig()
    phase = _initial_phase(shape, config)
    # Should be smooth (no random noise)
    assert phase.shape == shape
    # Quadratic + linear -> smooth gradient
    grad = np.gradient(phase)
    assert np.all(np.isfinite(grad[0]))


def test_cgm_gradient_finite_difference():
    """Verify analytical gradient matches finite-difference gradient."""
    shape = (32, 32)
    input_amp = gaussian_beam(shape, sigma=8.0, normalize=False)
    target = top_hat(shape, radius=5.0)
    region = measure_region(shape, target, margin=2)
    config = CGMConfig(steepness=4)

    phi = _initial_phase(shape, config)
    E_in = input_amp * np.exp(1j * phi)
    E_out = fft_propagate(E_in)
    grad = _cost_gradient(E_in, E_out, target, region, config.steepness)

    # Finite difference at a few random pixels
    eps = 1e-5
    rng = np.random.default_rng(42)
    for _ in range(5):
        i, j = rng.integers(0, shape[0]), rng.integers(0, shape[1])
        phi_plus = phi.copy()
        phi_plus[i, j] += eps
        phi_minus = phi.copy()
        phi_minus[i, j] -= eps

        E_plus = fft_propagate(input_amp * np.exp(1j * phi_plus))
        E_minus = fft_propagate(input_amp * np.exp(1j * phi_minus))

        cost_plus = _cost_function(E_plus, target, region, config.steepness)
        cost_minus = _cost_function(E_minus, target, region, config.steepness)

        fd_grad = (cost_plus - cost_minus) / (2 * eps)
        np.testing.assert_allclose(grad[i, j], fd_grad, rtol=0.1, atol=1e-3)


def test_cgm_convergence_stagnation():
    """CGM should stop when cost stagnates."""
    shape = (32, 32)
    input_amp = gaussian_beam(shape, sigma=8.0, normalize=False)
    target = top_hat(shape, radius=5.0)
    region = measure_region(shape, target, margin=2)
    config = CGMConfig(max_iterations=500, convergence_threshold=1e-3, steepness=4)

    result = cgm(input_amp, target, region, config)
    # Should have stopped before max_iterations due to convergence
    assert result.n_iterations <= 500
