"""Tests for the JAX-based CGM optimizer."""

import numpy as np
import pytest

from slm.beams import gaussian_beam  # noqa: E402
from slm.cgm import CGMConfig, _cost_function, _initial_phase  # noqa: E402
from slm.propagation import fft_propagate  # noqa: E402
from slm.targets import measure_region, top_hat  # noqa: E402

jax = pytest.importorskip("jax")

from slm.cgm_jax import _jax_cost, cgm_jax  # noqa: E402


def test_cgm_jax_cost_matches_numpy():
    """JAX cost function should match the numpy version on the same phi."""
    import jax.numpy as jnp

    shape = (32, 32)
    input_amp = gaussian_beam(shape, sigma=8.0, normalize=False)
    target = top_hat(shape, radius=5.0)
    region = measure_region(shape, target, margin=2)

    phi = _initial_phase(shape, CGMConfig(steepness=6))

    np_cost = _cost_function(
        fft_propagate(input_amp * np.exp(1j * phi)),
        target,
        region,
        6,
        0.0,
        0.0,
    )
    jax_cost = float(
        _jax_cost(
            jnp.array(phi.ravel()),
            jnp.array(input_amp),
            jnp.array(target),
            jnp.array(region),
            6,
            0.0,
            0.0,
        )
    )

    np.testing.assert_allclose(jax_cost, np_cost, rtol=1e-6)


def test_cgm_jax_gradient_vs_finite_diff():
    """JAX autograd gradient should match finite differences."""
    import jax.numpy as jnp

    shape = (32, 32)
    input_amp = gaussian_beam(shape, sigma=8.0, normalize=False)
    target = top_hat(shape, radius=5.0)
    region = measure_region(shape, target, margin=2)

    phi = _initial_phase(shape, CGMConfig(steepness=4))

    j_amp = jnp.array(input_amp)
    j_tgt = jnp.array(target)
    j_reg = jnp.array(region)
    grad_fn = jax.grad(_jax_cost)
    jax_grad = np.array(
        grad_fn(jnp.array(phi.ravel()), j_amp, j_tgt, j_reg, 4, 0.0, 0.0)
    )
    jax_grad_2d = jax_grad.reshape(shape)

    eps = 1e-5
    rng = np.random.default_rng(42)
    for _ in range(5):
        i, j = rng.integers(0, shape[0]), rng.integers(0, shape[1])
        phi_p, phi_m = phi.copy(), phi.copy()
        phi_p[i, j] += eps
        phi_m[i, j] -= eps
        c_p = float(
            _jax_cost(jnp.array(phi_p.ravel()), j_amp, j_tgt, j_reg, 4, 0.0, 0.0)
        )
        c_m = float(
            _jax_cost(jnp.array(phi_m.ravel()), j_amp, j_tgt, j_reg, 4, 0.0, 0.0)
        )
        fd = (c_p - c_m) / (2 * eps)
        np.testing.assert_allclose(jax_grad_2d[i, j], fd, rtol=0.1, atol=1e-3)


def test_cgm_jax_converges():
    """cgm_jax should reduce cost and produce reasonable fidelity."""
    shape = (64, 64)
    input_amp = gaussian_beam(shape, sigma=15.0, normalize=False)
    target = top_hat(shape, radius=8.0)
    region = measure_region(shape, target, margin=3)
    config = CGMConfig(max_iterations=50, steepness=6, R=3e-3)

    result = cgm_jax(input_amp, target, region, config)

    assert len(result.cost_history) > 1
    assert result.cost_history[-1] < result.cost_history[0]
    assert result.final_fidelity > 0.5
