"""JAX-based CGM — autograd through FFT, scipy CG optimizer.

Replicates the paper's Theano + fmin_cg stack using JAX autograd
and scipy.optimize.minimize(method='CG').

Requires: ``pip install jax jaxlib`` (or ``pip install slm[jax]``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from slm.cgm import (
    CGMConfig,
    CGMResult,
    _align_initial_phase,
    _build_result,
    _initial_phase,
)
from slm.propagation import fft_propagate, realistic_propagate, sinc_envelope

try:
    import jax
    import jax.numpy as jnp
    from jax.numpy.fft import fft2, fftshift, ifftshift

    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def _jax_fft_propagate(field: jnp.ndarray) -> jnp.ndarray:
    """FFT propagation in JAX (intentional duplicate — JAX autograd
    requires JAX ops in the computation graph)."""
    return fftshift(fft2(ifftshift(field), norm="ortho"))


def _jax_cost(
    phi_flat: jnp.ndarray,
    input_amplitude: jnp.ndarray,
    target_field: jnp.ndarray,
    measure_region: jnp.ndarray,
    steepness: int,
    efficiency_weight: float,
    eta_min: float,
    sinc_env: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Pure JAX cost function (forward only — jax.grad handles backward)."""
    shape = input_amplitude.shape
    phi = phi_flat.reshape(shape)
    E_in = input_amplitude * jnp.exp(1j * phi)
    E_out = _jax_fft_propagate(E_in)
    if sinc_env is not None:
        E_out = E_out * sinc_env

    out_m = E_out * measure_region
    tgt_m = target_field * measure_region

    out_norm = jnp.sqrt(jnp.sum(jnp.abs(out_m) ** 2))
    tgt_norm = jnp.sqrt(jnp.sum(jnp.abs(tgt_m) ** 2))

    eps = 1e-30
    overlap = jnp.sum(jnp.conj(tgt_m) * out_m) / (tgt_norm * out_norm + eps)
    cost = 10.0**steepness * (1.0 - jnp.real(overlap)) ** 2

    if efficiency_weight > 0 or eta_min > 0:
        intensity = jnp.abs(E_out) ** 2
        P_total = jnp.sum(intensity)
        eta = jnp.sum(intensity * measure_region) / (P_total + eps)
        if efficiency_weight > 0:
            cost = cost + efficiency_weight * 10.0**steepness * (1.0 - eta) ** 2
        if eta_min > 0:
            shortfall = jnp.maximum(0.0, eta_min - eta)
            cost = cost + 10.0**steepness * shortfall**2

    return cost


def cgm_jax(
    input_amplitude: np.ndarray,
    target_field: np.ndarray,
    measure_region: np.ndarray,
    config: CGMConfig = CGMConfig(),
    callback: Callable[[int, float], None] | None = None,
) -> CGMResult:
    """CGM using JAX autograd + scipy CG (replicates Theano + fmin_cg).

    Same interface as :func:`slm.cgm.cgm`.
    """
    if not HAS_JAX:
        raise ImportError(
            "JAX is required for cgm_jax.  Install with: pip install jax jaxlib"
        )

    shape = input_amplitude.shape
    if config.initial_phase is not None:
        phi_init = config.initial_phase.copy()
    else:
        phi_init = _initial_phase(shape, config)
    # Precompute sinc envelope when fill_factor < 1
    sinc_env_np = (
        sinc_envelope(target_field.shape, config.fill_factor)
        if config.fill_factor < 1.0
        else None
    )

    phi_init = _align_initial_phase(
        phi_init,
        input_amplitude,
        target_field,
        measure_region,
        sinc_env_np,
    )

    j_amp = jnp.array(input_amplitude)
    j_tgt = jnp.array(target_field)
    j_reg = jnp.array(measure_region)
    j_sinc = jnp.array(sinc_env_np) if sinc_env_np is not None else None
    d = int(config.steepness)
    ew = float(config.efficiency_weight)
    em = float(config.eta_min)

    @jax.jit
    def cost_and_grad(phi_flat):
        return jax.value_and_grad(_jax_cost)(
            phi_flat,
            j_amp,
            j_tgt,
            j_reg,
            d,
            ew,
            em,
            j_sinc,
        )

    cost_history: list[float] = []

    def scipy_obj_grad(phi_flat: np.ndarray) -> tuple[float, np.ndarray]:
        c, g = cost_and_grad(jnp.array(phi_flat))
        c = float(c)
        cost_history.append(c)
        return c, np.asarray(g, dtype=np.float64)

    iter_count = [0]

    def _callback(xk: np.ndarray) -> None:
        iter_count[0] += 1
        if callback is not None and cost_history:
            callback(iter_count[0], cost_history[-1])

    opt = scipy_minimize(
        scipy_obj_grad,
        phi_init.ravel(),
        method="CG",
        jac=True,
        callback=_callback,
        options={"maxiter": config.max_iterations, "gtol": 1e-12},
    )

    phi_final = opt.x.reshape(shape)
    E_in_final = input_amplitude * np.exp(1j * phi_final)
    E_out = (
        realistic_propagate(E_in_final, sinc_env_np)
        if sinc_env_np is not None
        else fft_propagate(E_in_final)
    )

    return _build_result(
        phi_final,
        E_out,
        target_field,
        measure_region,
        cost_history,
        opt.nit,
    )
