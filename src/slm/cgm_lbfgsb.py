"""L-BFGS-B optimizer for CGM beam shaping.

Wraps scipy's L-BFGS-B with the CGM cost function and analytical gradient.
L-BFGS-B approximates the Hessian from gradient history and converges
faster than Fletcher-Reeves CG for the top-hat cost landscape.
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
    _cost_function,
    _cost_gradient,
    _initial_phase,
)
from slm.propagation import fft_propagate


def cgm_lbfgsb(
    input_amplitude: np.ndarray,
    target_field: np.ndarray,
    measure_region: np.ndarray,
    config: CGMConfig = CGMConfig(),
    callback: Callable[[int, float], None] | None = None,
) -> CGMResult:
    """CGM using L-BFGS-B optimizer (quasi-Newton with Hessian approximation).

    Drop-in replacement for :func:`slm.cgm.cgm` that uses scipy's L-BFGS-B
    instead of Fletcher-Reeves conjugate gradient.

    Parameters
    ----------
    input_amplitude : real (ny, nx) -- incident beam amplitude.
    target_field : complex (ny, nx) -- desired focal-plane field.
    measure_region : binary (ny, nx) -- region of interest Omega.
    config : algorithm parameters.
    callback : optional function called each iteration with (i, cost).
    """
    shape = input_amplitude.shape

    if config.initial_phase is not None:
        phi = config.initial_phase.copy()
    else:
        phi = _initial_phase(shape, config)
    phi = _align_initial_phase(phi, input_amplitude, target_field, measure_region)

    cost_history: list[float] = []
    ew = config.efficiency_weight
    em = config.eta_min

    def objective(phi_flat: np.ndarray) -> tuple[float, np.ndarray]:
        phi_2d = phi_flat.reshape(shape)
        E_in = input_amplitude * np.exp(1j * phi_2d)
        E_out = fft_propagate(E_in)
        cost = _cost_function(E_out, target_field, measure_region,
                              config.steepness, ew, em)
        grad = _cost_gradient(E_in, E_out, target_field, measure_region,
                              config.steepness, ew, em)
        cost_history.append(cost)
        return float(cost), grad.ravel().astype(np.float64)

    iter_count = [0]

    def _cb(xk):
        iter_count[0] += 1
        if callback and cost_history:
            callback(iter_count[0], cost_history[-1])

    res = scipy_minimize(
        fun=objective, x0=phi.ravel(), method="L-BFGS-B", jac=True,
        callback=_cb,
        options={"maxiter": config.max_iterations, "ftol": 1e-15, "gtol": 1e-12},
    )

    phi_final = res.x.reshape(shape)
    E_out = fft_propagate(input_amplitude * np.exp(1j * phi_final))
    return _build_result(phi_final, E_out, target_field, measure_region,
                         cost_history, res.nit)
