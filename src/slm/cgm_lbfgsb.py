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
    _initial_phase,
)
from slm.propagation import (
    fft_propagate,
    ifft_propagate,
    realistic_propagate,
    sinc_envelope,
)


def _discrete_laplacian(phi: np.ndarray) -> np.ndarray:
    """Discrete Laplacian via 4-connected finite differences."""
    return (
        np.roll(phi, 1, 0)
        + np.roll(phi, -1, 0)
        + np.roll(phi, 1, 1)
        + np.roll(phi, -1, 1)
        - 4.0 * phi
    )


def cgm_lbfgsb(
    input_amplitude: np.ndarray,
    target_field: np.ndarray,
    measure_region: np.ndarray,
    config: CGMConfig = CGMConfig(),
    callback: Callable[[int, float], None] | None = None,
    cost_mode: str = "standard",
    smooth_weight: float = 0.0,
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
    cost_mode : ``"standard"`` (region-normalized overlap),
        ``"coupled"`` (efficiency-coupled: (1 - sqrt(eta)*overlap)^2),
        or ``"coupled+smooth"`` (coupled + Laplacian regularization).
    smooth_weight : weight for Laplacian phase smoothness penalty.
    """
    shape = input_amplitude.shape

    if config.initial_phase is not None:
        phi = config.initial_phase.copy()
    else:
        phi = _initial_phase(shape, config)

    # Precompute sinc envelope when fill_factor < 1
    sinc_env = (
        sinc_envelope(target_field.shape, config.fill_factor)
        if config.fill_factor < 1.0
        else None
    )

    phi = _align_initial_phase(
        phi, input_amplitude, target_field, measure_region, sinc_env
    )

    active = input_amplitude.ravel() > 1e-12
    phi_full = phi.ravel().copy()

    # Precompute constants: A and back_A are fixed across all iterations,
    # saving one full IFFT per function evaluation.
    A = target_field * measure_region
    norm_A = np.sqrt(np.sum(np.abs(A) ** 2))
    back_A = ifft_propagate(sinc_env * A) if sinc_env is not None else ifft_propagate(A)
    scale = 10**config.steepness

    cost_history: list[float] = []
    ew = config.efficiency_weight
    em = config.eta_min
    coupled = cost_mode in ("coupled", "coupled+smooth")
    use_smooth = smooth_weight > 0 or cost_mode == "coupled+smooth"
    sw = (
        smooth_weight if smooth_weight > 0 else (1e-3 if "smooth" in cost_mode else 0.0)
    )

    def objective(phi_active: np.ndarray) -> tuple[float, np.ndarray]:
        phi_full[active] = phi_active
        phi_2d = phi_full.reshape(shape)
        E_in = input_amplitude * np.exp(1j * phi_2d)
        E_out = (
            realistic_propagate(E_in, sinc_env)
            if sinc_env is not None
            else fft_propagate(E_in)
        )

        B = E_out * measure_region
        norm_B = np.sqrt(np.sum(np.abs(B) ** 2))

        if norm_A == 0 or norm_B == 0:
            cost_history.append(0.0)
            return 0.0, np.zeros(int(active.sum()), dtype=np.float64)

        r = np.sum(np.conj(A) * B)
        overlap_real = np.real(r) / (norm_A * norm_B)

        # Gradient building blocks (precomputed back_A saves one IFFT)
        back_B = (
            ifft_propagate(sinc_env * B) if sinc_env is not None else ifft_propagate(B)
        )
        d_Re_r = np.real(1j * E_in * np.conj(back_A))
        raw_B = np.real(1j * E_in * np.conj(back_B))
        d_norm_B = raw_B / norm_B
        d_overlap = d_Re_r / (norm_A * norm_B) - overlap_real * d_norm_B / norm_B

        P_total = np.sum(np.abs(E_out) ** 2)
        eta = float(norm_B**2 / P_total) if P_total > 0 else 0.0
        d_eta = 2.0 * raw_B / P_total if P_total > 0 else np.zeros_like(raw_B)

        if coupled:
            # Efficiency-coupled: C = scale * (1 - sqrt(eta) * overlap)^2
            sqrt_eta = np.sqrt(max(eta, 1e-30))
            S = sqrt_eta * overlap_real
            cost = float(scale * (1.0 - S) ** 2)
            dS = sqrt_eta * d_overlap + overlap_real / (2.0 * sqrt_eta) * d_eta
            grad = -2.0 * scale * (1.0 - S) * dS
        else:
            # Standard: C = scale * (1 - overlap)^2
            cost = float(scale * (1.0 - overlap_real) ** 2)
            grad = -2.0 * scale * (1.0 - overlap_real) * d_overlap

        # Efficiency penalties (eta_min floor, efficiency_weight)
        if em > 0 and eta < em:
            cost += float(scale * (em - eta) ** 2)
            grad += -2.0 * scale * (em - eta) * d_eta
        if ew > 0:
            cost += float(ew * scale * (1.0 - eta) ** 2)
            grad += -2.0 * ew * scale * (1.0 - eta) * d_eta

        # Phase smoothness regularization
        if use_smooth and sw > 0:
            lap = _discrete_laplacian(phi_2d)
            cost += float(sw * np.sum(lap**2))
            grad += 2.0 * sw * _discrete_laplacian(lap)

        cost_history.append(cost)
        return cost, grad.ravel()[active].astype(np.float64)

    iter_count = [0]

    def _cb(xk):
        iter_count[0] += 1
        if callback and cost_history:
            callback(iter_count[0], cost_history[-1])

    res = scipy_minimize(
        fun=objective,
        x0=phi.ravel()[active],
        method="L-BFGS-B",
        jac=True,
        callback=_cb,
        options={"maxiter": config.max_iterations, "ftol": 1e-15, "gtol": 1e-12},
    )

    phi_final = phi.ravel().copy()
    phi_final[active] = res.x
    phi_final = phi_final.reshape(shape)
    E_in_final = input_amplitude * np.exp(1j * phi_final)
    E_out = (
        realistic_propagate(E_in_final, sinc_env)
        if sinc_env is not None
        else fft_propagate(E_in_final)
    )
    return _build_result(
        phi_final, E_out, target_field, measure_region, cost_history, res.nit
    )
