"""Conjugate Gradient Minimization for continuous beam shaping (Bowman et al.)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

from slm.metrics import efficiency, fidelity, non_uniformity_error, phase_error
from slm.propagation import (
    fft_propagate,
    ifft_propagate,
    realistic_propagate,
    sinc_envelope,
)
from slm.targets import mask_from_target


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
    efficiency_weight: float = 0.0  # weight for (1-η)^2 efficiency penalty
    eta_min: float = 0.0  # minimum efficiency floor; penalty when η < eta_min
    initial_phase: np.ndarray | None = (
        None  # measured/custom phase; overrides analytical
    )
    fill_factor: float = 1.0  # SLM pixel fill factor (1.0 = ideal, no sinc)


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


def _forward(
    E_in: np.ndarray,
    sinc_env: np.ndarray | None,
) -> np.ndarray:
    """Forward propagation, optionally with sinc envelope."""
    if sinc_env is not None:
        return realistic_propagate(E_in, sinc_env)
    return fft_propagate(E_in)


def _back(
    X: np.ndarray,
    sinc_env: np.ndarray | None,
) -> np.ndarray:
    """Adjoint back-propagation: IFFT(sinc * X) when sinc active."""
    if sinc_env is not None:
        return ifft_propagate(sinc_env * X)
    return ifft_propagate(X)


def _cost_function(
    output_field: np.ndarray,
    target_field: np.ndarray,
    measure_region: np.ndarray,
    steepness: int,
    efficiency_weight: float = 0.0,
    eta_min: float = 0.0,
) -> float:
    """Compute C = 10^d * ((1 - Re{overlap})^2 + penalty).

    The penalty term is either a continuous (1-η)^2 weighted by
    *efficiency_weight*, a threshold max(0, eta_min - η)^2, or both.
    """
    overlap = _compute_overlap(output_field, target_field, measure_region)
    cost = 10**steepness * (1.0 - np.real(overlap)) ** 2
    if efficiency_weight > 0 or eta_min > 0:
        intensity = np.abs(output_field) ** 2
        P_total = np.sum(intensity)
        if P_total > 0:
            eta = np.sum(intensity * measure_region) / P_total
            if efficiency_weight > 0:
                cost += efficiency_weight * 10**steepness * (1.0 - eta) ** 2
            if eta_min > 0 and eta < eta_min:
                cost += 10**steepness * (eta_min - eta) ** 2
    return float(cost)


def _cost_gradient(
    E_in: np.ndarray,
    E_out: np.ndarray,
    target_field: np.ndarray,
    measure_region: np.ndarray,
    steepness: int,
    efficiency_weight: float = 0.0,
    eta_min: float = 0.0,
    sinc_env: np.ndarray | None = None,
) -> np.ndarray:
    """Compute dC/d(phi_{p,q}) analytically.

    Accepts pre-computed E_in and E_out to avoid redundant FFT.
    When *sinc_env* is provided, uses the adjoint sinc·IFFT for
    back-propagation (since forward is sinc·FFT).
    """
    A = target_field * measure_region
    B = E_out * measure_region

    norm_A = np.sqrt(np.sum(np.abs(A) ** 2))
    norm_B = np.sqrt(np.sum(np.abs(B) ** 2))

    if norm_A == 0 or norm_B == 0:
        return np.zeros_like(E_in, dtype=np.float64)

    # Unnormalized inner product and real overlap
    r = np.sum(np.conj(A) * B)
    overlap_real = np.real(r) / (norm_A * norm_B)

    # Back-propagate masked fields to SLM plane (adjoint includes sinc)
    back_A = _back(A, sinc_env)
    back_B = _back(B, sinc_env)

    d_Re_r = np.real(1j * E_in * np.conj(back_A))
    raw_B = np.real(1j * E_in * np.conj(back_B))  # shared by d_norm_B and d_eta
    d_norm_B = raw_B / norm_B

    d_overlap = d_Re_r / (norm_A * norm_B) - overlap_real * d_norm_B / norm_B
    grad = -2.0 * 10**steepness * (1.0 - overlap_real) * d_overlap

    if efficiency_weight > 0 or eta_min > 0:
        P_total = np.sum(np.abs(E_out) ** 2)
        if P_total > 0:
            eta = float(norm_B**2 / P_total)
            d_eta = 2.0 * raw_B / P_total
            if efficiency_weight > 0:
                grad += -2.0 * efficiency_weight * 10**steepness * (1.0 - eta) * d_eta
            if eta_min > 0 and eta < eta_min:
                grad += -2.0 * 10**steepness * (eta_min - eta) * d_eta

    return grad


def _eval_cost_and_grad(
    phi: np.ndarray,
    input_amplitude: np.ndarray,
    target_field: np.ndarray,
    measure_region: np.ndarray,
    steepness: int,
    efficiency_weight: float = 0.0,
    eta_min: float = 0.0,
    sinc_env: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate cost and gradient. Returns (cost, gradient, E_out)."""
    E_in = input_amplitude * np.exp(1j * phi)
    E_out = _forward(E_in, sinc_env)
    cost = _cost_function(
        E_out,
        target_field,
        measure_region,
        steepness,
        efficiency_weight,
        eta_min,
    )
    grad = _cost_gradient(
        E_in,
        E_out,
        target_field,
        measure_region,
        steepness,
        efficiency_weight,
        eta_min,
        sinc_env,
    )
    return cost, grad, E_out


def _line_search(
    phi: np.ndarray,
    direction: np.ndarray,
    cost0: float,
    input_amplitude: np.ndarray,
    target_field: np.ndarray,
    measure_region: np.ndarray,
    steepness: int,
    efficiency_weight: float = 0.0,
    eta_min: float = 0.0,
    sinc_env: np.ndarray | None = None,
) -> float:
    """Geometric probing + bounded refinement line search."""

    def cost_at(alpha: float) -> float:
        trial_phi = phi + alpha * direction
        E_in = input_amplitude * np.exp(1j * trial_phi)
        E_out = _forward(E_in, sinc_env)
        return _cost_function(
            E_out,
            target_field,
            measure_region,
            steepness,
            efficiency_weight,
            eta_min,
        )

    # Probe at geometrically spaced alphas to cover many orders of magnitude.
    # Cap alpha so max pixel phase change is bounded (prevents large power
    # redistribution that the paper's CG avoids via conservative line search).
    dir_max = np.max(np.abs(direction))
    if dir_max == 0:
        return 0.0
    # Limit max phase change per pixel to ~pi/2
    base = (np.pi / 2.0) / dir_max
    probes = [base * f for f in [1e-3, 1e-2, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]]
    probe_costs = [(a, cost_at(a)) for a in probes]
    # Find best probe
    best_alpha, best_cost = min(probe_costs, key=lambda x: x[1])

    if best_cost >= cost0:
        # No improvement at any probe; try tiny steps
        tiny_probes = [base * f for f in [1e-5, 1e-4]]
        for a in tiny_probes:
            c = cost_at(a)
            if c < cost0:
                best_alpha, best_cost = a, c
                probe_costs.append((a, c))
                break
        if best_cost >= cost0:
            return 0.0

    # Refine: bracket around best probe and its neighbors
    all_probes = sorted([(0.0, cost0)] + probe_costs, key=lambda x: x[0])
    idx = next(i for i, (a, _) in enumerate(all_probes) if a == best_alpha)
    lo = all_probes[max(0, idx - 1)][0]
    hi = all_probes[min(len(all_probes) - 1, idx + 1)][0]
    if lo == hi:
        return best_alpha

    res = minimize_scalar(
        cost_at, bounds=(lo, hi), method="bounded", options={"xatol": (hi - lo) * 1e-4}
    )
    return float(res.x)


def _align_initial_phase(
    phi: np.ndarray,
    input_amplitude: np.ndarray,
    target_field: np.ndarray,
    measure_region: np.ndarray,
    sinc_env: np.ndarray | None = None,
) -> np.ndarray:
    """Rotate initial phase so Re{overlap} = |overlap| (positive real)."""
    E_tmp = _forward(input_amplitude * np.exp(1j * phi), sinc_env)
    init_overlap = _compute_overlap(E_tmp, target_field, measure_region)
    if abs(init_overlap) > 1e-10:
        phi = phi - np.angle(init_overlap)
    return phi


def _build_result(
    phi: np.ndarray,
    E_out: np.ndarray,
    target_field: np.ndarray,
    measure_region: np.ndarray,
    cost_history: list[float],
    n_iterations: int,
    fidelity_history: list[float] | None = None,
) -> CGMResult:
    """Compute final metrics and return CGMResult."""
    target_mask = mask_from_target(target_field)
    target_intensity = np.abs(target_field) ** 2
    return CGMResult(
        slm_phase=phi,
        output_field=E_out,
        cost_history=cost_history,
        final_fidelity=fidelity(E_out, target_field, measure_region),
        final_efficiency=efficiency(E_out, measure_region),
        final_phase_error=phase_error(
            np.angle(E_out),
            np.angle(target_field),
            target_mask,
            weights=target_intensity,
        ),
        final_non_uniformity=non_uniformity_error(
            np.abs(E_out) ** 2,
            target_intensity,
            target_mask,
        ),
        n_iterations=n_iterations,
        fidelity_history=fidelity_history or [],
    )


def cgm(
    input_amplitude: np.ndarray,
    target_field: np.ndarray,
    measure_region: np.ndarray,
    config: CGMConfig = CGMConfig(),
    callback: Callable[[int, float], None] | None = None,
) -> CGMResult:
    """Conjugate Gradient Minimization for continuous beam shaping.

    Uses the Fletcher-Reeves conjugate gradient method with an adaptive
    line search, following the algorithm in Bowman et al.

    Parameters
    ----------
    input_amplitude : S_{p,q}, real (ny, nx) -- incident beam amplitude.
    target_field : tau = sqrt(T)*exp(i*Phi), complex (ny, nx) -- target field.
    measure_region : binary (ny, nx) -- region of interest Omega.
    config : algorithm parameters.
    callback : optional function called each iteration with (i, cost).
    """
    shape = input_amplitude.shape
    if config.initial_phase is not None:
        phi = config.initial_phase.copy()
    else:
        phi = _initial_phase(shape, config)

    # Precompute sinc envelope when fill_factor < 1
    sinc_env: np.ndarray | None = None
    if config.fill_factor < 1.0:
        sinc_env = sinc_envelope(target_field.shape, config.fill_factor)

    phi = _align_initial_phase(
        phi, input_amplitude, target_field, measure_region, sinc_env
    )

    cost_history: list[float] = []
    fidelity_history: list[float] = []

    ew = config.efficiency_weight
    em = config.eta_min

    # Initial forward pass
    cost, grad, E_out = _eval_cost_and_grad(
        phi,
        input_amplitude,
        target_field,
        measure_region,
        config.steepness,
        ew,
        em,
        sinc_env,
    )
    cost_history.append(cost)
    if config.track_fidelity:
        fidelity_history.append(fidelity(E_out, target_field, measure_region))

    # First descent direction is steepest descent
    direction = -grad
    grad_norm_sq = np.sum(grad**2)
    n_iters = 0

    for i in range(config.max_iterations):
        # Line search along conjugate direction
        alpha = _line_search(
            phi,
            direction,
            cost,
            input_amplitude,
            target_field,
            measure_region,
            config.steepness,
            ew,
            em,
            sinc_env,
        )
        if alpha == 0.0:
            # Line search failed; reset to steepest descent
            direction = -grad
            alpha = _line_search(
                phi,
                direction,
                cost,
                input_amplitude,
                target_field,
                measure_region,
                config.steepness,
                ew,
                em,
                sinc_env,
            )
            if alpha == 0.0:
                break

        phi = phi + alpha * direction

        # Recompute cost and gradient at new point
        new_cost, new_grad, E_out = _eval_cost_and_grad(
            phi,
            input_amplitude,
            target_field,
            measure_region,
            config.steepness,
            ew,
            em,
            sinc_env,
        )
        cost_history.append(new_cost)
        if config.track_fidelity:
            fidelity_history.append(
                fidelity(E_out, target_field, measure_region),
            )
        n_iters = i + 1

        if callback is not None:
            callback(i + 1, new_cost)

        # Check convergence (relative criterion)
        if abs(cost - new_cost) < config.convergence_threshold * max(
            abs(new_cost), 1.0
        ):
            break

        # Fletcher-Reeves conjugate direction update with periodic restart
        new_grad_norm_sq = np.sum(new_grad**2)
        if grad_norm_sq > 0 and (i + 1) % 50 != 0:
            beta = new_grad_norm_sq / grad_norm_sq
        else:
            beta = 0.0  # restart: steepest descent
        direction = -new_grad + beta * direction

        cost = new_cost
        grad = new_grad
        grad_norm_sq = new_grad_norm_sq

    return _build_result(
        phi,
        E_out,
        target_field,
        measure_region,
        cost_history,
        n_iters,
        fidelity_history,
    )


def tophat_phase_generate(
    initSLMAmp: np.ndarray,
    shape: tuple[int, int] | None = None,
    radius: float = 50.0,
    center: tuple[float, float] | None = None,
    max_iterations: int = 300,
    steepness: int = 9,
    initial_phase: np.ndarray | None = None,
    Plot: bool = False,
) -> np.ndarray:
    """One-line top-hat phase generation via CGM.

    Parameters
    ----------
    initSLMAmp : 2D array — incident beam amplitude on SLM plane.
    shape : grid size (ny, nx). Defaults to initSLMAmp.shape.
    radius : top-hat radius in focal-plane pixels.
    center : (row, col) of top-hat center; defaults to grid center.
    max_iterations : CGM iterations (300 typical).
    steepness : cost exponent (10^steepness).
    initial_phase : optional starting phase (e.g. with Fresnel lens baked in).
    Plot : if True, print convergence summary.

    Returns
    -------
    SLM phase array (float64, same shape as initSLMAmp).
    """
    from slm.targets import measure_region as make_region
    from slm.targets import top_hat

    if shape is None:
        shape = initSLMAmp.shape

    target = top_hat(shape, radius=radius, center=center)
    region = make_region(shape, target, margin=5)

    config = CGMConfig(
        max_iterations=max_iterations,
        steepness=steepness,
        R=0.0,
        D=0.0,
        theta=0.0,
        eta_min=0.05,
        initial_phase=initial_phase,
    )

    result = cgm(initSLMAmp, target, region, config)

    if Plot:
        print(f"CGM top-hat: {result.n_iterations} iter, "
              f"F={result.final_fidelity:.4f}, "
              f"η={result.final_efficiency:.4f}, "
              f"ε_ν={result.final_non_uniformity:.4f}")

    return result.slm_phase
