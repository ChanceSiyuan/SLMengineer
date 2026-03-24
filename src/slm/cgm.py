"""Conjugate Gradient Minimization for continuous beam shaping (Bowman et al.)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

from slm.metrics import efficiency, fidelity, non_uniformity_error, phase_error
from slm.propagation import fft_propagate, ifft_propagate


@dataclass
class CGMConfig:
    """Configuration for CGM algorithm."""

    max_iterations: int = 200
    steepness: int = 9
    convergence_threshold: float = 1e-5
    R: float = 4.5e-3  # quadratic phase curvature (rad/px^2)
    D: float = -np.pi / 2  # linear phase offset magnitude
    theta: float = np.pi / 4  # linear phase angle (diagonal offset)
    track_fidelity: bool = False  # record fidelity each iteration (slower)


@dataclass
class CGMResult:
    """Result container for CGM algorithm."""

    slm_phase: np.ndarray = field(repr=False)
    output_field: np.ndarray = field(repr=False)
    cost_history: list[float] = field(default_factory=list)
    final_fidelity: float = 0.0
    final_efficiency: float = 0.0
    final_phase_error: float = 0.0
    final_non_uniformity: float = 0.0
    n_iterations: int = 0
    fidelity_history: list[float] = field(default_factory=list)


def _initial_phase(shape: tuple[int, int], config: CGMConfig) -> np.ndarray:
    """Generate structured initial guess phase (Bowman et al.).

    phi = R*(p^2 + q^2) + D*(p*cos(theta) + q*sin(theta))

    Quadratic term controls envelope size; linear term offsets pattern
    diagonally to avoid zero-order spot. Suppresses optical vortices.
    """
    ny, nx = shape
    p = np.arange(nx) - (nx - 1) / 2.0
    q = np.arange(ny) - (ny - 1) / 2.0
    pp, qq = np.meshgrid(p, q, indexing="xy")
    phase = config.R * (pp**2 + qq**2) + config.D * (
        pp * np.cos(config.theta) + qq * np.sin(config.theta)
    )
    return phase


def _compute_overlap(
    output_field: np.ndarray,
    target_field: np.ndarray,
    measure_region: np.ndarray,
) -> complex:
    """Compute normalized inner product over measure region."""
    out_masked = output_field * measure_region
    tgt_masked = target_field * measure_region

    out_norm = np.sqrt(np.sum(np.abs(out_masked) ** 2))
    tgt_norm = np.sqrt(np.sum(np.abs(tgt_masked) ** 2))

    if out_norm == 0 or tgt_norm == 0:
        return 0.0 + 0.0j

    return np.sum(np.conj(tgt_masked / tgt_norm) * (out_masked / out_norm))


def _cost_function(
    output_field: np.ndarray,
    target_field: np.ndarray,
    measure_region: np.ndarray,
    steepness: int,
) -> float:
    """Compute C = 10^d * (1 - Re{overlap})^2."""
    overlap = _compute_overlap(output_field, target_field, measure_region)
    return float(10**steepness * (1.0 - np.real(overlap)) ** 2)


def _cost_gradient(
    E_in: np.ndarray,
    E_out: np.ndarray,
    target_field: np.ndarray,
    measure_region: np.ndarray,
    steepness: int,
) -> np.ndarray:
    """Compute dC/d(phi_{p,q}) analytically.

    Accepts pre-computed E_in and E_out to avoid redundant FFT.
    """
    A = target_field * measure_region
    B = E_out * measure_region

    norm_A = np.sqrt(np.sum(np.abs(A) ** 2))
    norm_B = np.sqrt(np.sum(np.abs(B) ** 2))

    if norm_A == 0 or norm_B == 0:
        return np.zeros_like(slm_phase)

    # Unnormalized inner product and real overlap
    r = np.sum(np.conj(A) * B)
    overlap_real = np.real(r) / (norm_A * norm_B)

    # Back-propagate masked fields to SLM plane
    back_A = ifft_propagate(A)
    back_B = ifft_propagate(B)

    # d Re{r} / dphi (gradient of unnormalized inner product)
    d_Re_r = np.real(1j * E_in * np.conj(back_A))

    # d ||B|| / dphi (gradient of output norm)
    d_norm_B = np.real(1j * E_in * np.conj(back_B)) / norm_B

    # d overlap_real / dphi (quotient rule)
    d_overlap = d_Re_r / (norm_A * norm_B) - overlap_real * d_norm_B / norm_B

    # d C / dphi
    grad = -2.0 * 10**steepness * (1.0 - overlap_real) * d_overlap

    return grad


def cgm(
    input_amplitude: np.ndarray,
    target_field: np.ndarray,
    measure_region: np.ndarray,
    config: CGMConfig = CGMConfig(),
    callback: Callable[[int, float], None] | None = None,
) -> CGMResult:
    """Conjugate Gradient Minimization for continuous beam shaping.

    Parameters
    ----------
    input_amplitude : S_{p,q}, real (ny, nx) -- incident beam amplitude.
    target_field : tau = sqrt(T)*exp(i*Phi), complex (ny, nx) -- target field.
    measure_region : binary (ny, nx) -- region of interest Omega.
    config : algorithm parameters.
    callback : optional function called each iteration with (i, cost).
    """
    shape = input_amplitude.shape
    phi = _initial_phase(shape, config)
    cost_history = []
    fidelity_history = []

    prev_grad = None
    prev_direction = None
    restart_interval = max(shape[0] * shape[1] // 10, 50)

    for i in range(config.max_iterations):
        # Compute output field
        E_in = input_amplitude * np.exp(1j * phi)
        E_out = fft_propagate(E_in)

        # Compute cost
        cost = _cost_function(E_out, target_field, measure_region, config.steepness)
        cost_history.append(cost)

        if config.track_fidelity:
            fidelity_history.append(fidelity(E_out, target_field, measure_region))

        if callback is not None:
            callback(i, cost)

        # Check convergence
        if len(cost_history) > 1 and abs(cost_history[-1] - cost_history[-2]) < config.convergence_threshold:
            break

        grad = _cost_gradient(E_in, E_out, target_field, measure_region, config.steepness)

        # Conjugate direction (Polak-Ribiere-Polyak with periodic restart)
        if prev_grad is None or i % restart_interval == 0:
            direction = -grad
        else:
            # PR+ formula: more robust than Fletcher-Reeves
            diff = grad - prev_grad
            prev_dot = np.sum(prev_grad * prev_grad)
            if prev_dot > 0:
                beta = max(0.0, np.sum(grad * diff) / prev_dot)
            else:
                beta = 0.0
            direction = -grad + beta * prev_direction

        prev_grad = grad.copy()
        prev_direction = direction.copy()

        # Line search: find initial bracket then minimize
        def line_cost(alpha):
            phi_trial = phi + alpha * direction
            E_trial = fft_propagate(input_amplitude * np.exp(1j * phi_trial))
            return _cost_function(E_trial, target_field, measure_region, config.steepness)

        # Adaptive bracket: start small and expand
        c0 = cost
        alpha_test = 1e-4
        for _ in range(20):
            c_test = line_cost(alpha_test)
            if c_test < c0:
                break
            alpha_test *= 0.5
        else:
            alpha_test = 1e-6

        result = minimize_scalar(
            line_cost, bounds=(0, alpha_test * 10), method="bounded",
            options={"xatol": alpha_test * 1e-3},
        )
        alpha_opt = result.x

        if result.fun < cost:
            phi = phi + alpha_opt * direction
        else:
            # Reset to steepest descent with small step
            grad_norm = np.sqrt(np.sum(grad**2))
            if grad_norm > 0:
                phi = phi - (alpha_test * 0.1) * grad / grad_norm
            prev_grad = None
            prev_direction = None

    # Final output
    E_in = input_amplitude * np.exp(1j * phi)
    E_out = fft_propagate(E_in)

    # Compute final metrics
    target_mask = np.abs(target_field) > 0
    final_fid = fidelity(E_out, target_field, measure_region)
    final_eff = efficiency(E_out, measure_region)
    final_pe = phase_error(
        np.angle(E_out), np.angle(target_field), target_mask.astype(np.float64)
    )
    final_nu = non_uniformity_error(
        np.abs(E_out) ** 2,
        np.abs(target_field) ** 2,
        target_mask.astype(np.float64),
    )

    return CGMResult(
        slm_phase=phi,
        output_field=E_out,
        cost_history=cost_history,
        final_fidelity=final_fid,
        final_efficiency=final_eff,
        final_phase_error=final_pe,
        final_non_uniformity=final_nu,
        n_iterations=len(cost_history),
        fidelity_history=fidelity_history,
    )
