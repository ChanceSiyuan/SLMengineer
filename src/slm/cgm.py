"""Conjugate Gradient Minimization for continuous beam shaping (Bowman et al.).

Torch-first implementation with CUDA acceleration.  The hot optimisation
loop runs on ``torch.complex64`` tensors on the best available device
(CUDA if present, else CPU), with invariant caching to keep FFT+memory
work at a minimum.

Public entry points
-------------------

- :func:`cgm` — numpy-in/numpy-out API (kept for future use).
- :func:`CGM_phase_generate` — torch-in/torch-out API mirroring
  ``slm.wgs.WGS_phase_generate`` (used by the hardware scripts under
  ``scripts/``).
- :func:`tophat_phase_generate` — convenience wrapper for top-hat targets.
- :class:`CGMConfig`, :class:`CGMResult` — dataclasses.

The internal helpers ``_initial_phase``, ``_forward``, ``_back``,
``_compute_overlap``, ``_cost_function``, ``_cost_gradient``,
``_align_initial_phase`` and ``_build_result`` stay numpy-native for
backward compatibility with ``tests/test_cgm.py``, which imports them
directly for finite-difference gradient checks.  They are *not* used
by the torch hot path.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dependency of the fast path
    torch = None  # type: ignore[assignment]

from slm.metrics import efficiency, fidelity, non_uniformity_error, phase_error
from slm.propagation import (
    fft_propagate,
    ifft_propagate,
    realistic_propagate,
    sinc_envelope,
)
from slm.targets import mask_from_target


# ---------------------------------------------------------------------------
# Config and result
# ---------------------------------------------------------------------------


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
    eta_steepness: float | None = None  # separate 10^s scale for eta penalty; None → uses steepness
    initial_phase: np.ndarray | None = (
        None  # measured/custom phase; overrides analytical
    )
    fill_factor: float = 1.0  # SLM pixel fill factor (1.0 = ideal, no sinc)
    device: str | None = None  # "cuda" | "cpu" | None (auto-pick)


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


# ---------------------------------------------------------------------------
# Device / dtype helpers
# ---------------------------------------------------------------------------


def _require_torch():
    if torch is None:
        raise ImportError(
            "PyTorch is required for CGM. Install with: uv pip install torch"
        )


def _resolve_device(device_spec):
    """Pick a torch device; default to CUDA if available."""
    _require_torch()
    if device_spec is not None:
        return torch.device(device_spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _hot_dtypes():
    """Hot-loop dtypes: complex64 + float32 for maximum throughput."""
    return torch.complex64, torch.float32


# ---------------------------------------------------------------------------
# Torch-native FFT propagation (hot path)
# ---------------------------------------------------------------------------


def _fft_propagate_t(field):
    """Forward propagation: SLM -> focal plane, norm='ortho'."""
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(field), norm="ortho")
    )


def _ifft_propagate_t(field):
    """Backward propagation: focal -> SLM plane, norm='ortho'."""
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(field), norm="ortho")
    )


def _forward_t(E_in, sinc_env_t):
    """Forward propagation with optional sinc envelope."""
    out = _fft_propagate_t(E_in)
    if sinc_env_t is not None:
        out = out * sinc_env_t
    return out


def _back_t(X, sinc_env_t):
    """Adjoint back-propagation: IFFT(sinc * X) when sinc active."""
    if sinc_env_t is not None:
        return _ifft_propagate_t(sinc_env_t * X)
    return _ifft_propagate_t(X)


def _sinc_envelope_t(shape, fill_factor, device, rdtype):
    """Torch sinc envelope matching :func:`slm.propagation.sinc_envelope`."""
    ny, nx = shape
    iy = torch.arange(ny, device=device, dtype=rdtype) - (ny // 2)
    ix = torch.arange(nx, device=device, dtype=rdtype) - (nx // 2)
    sy = torch.sinc(iy * fill_factor / ny)
    sx = torch.sinc(ix * fill_factor / nx)
    return sy[:, None] * sx[None, :]


# ---------------------------------------------------------------------------
# Core torch CGM loop (private)
# ---------------------------------------------------------------------------


def _run_cgm_torch(
    input_amp_t,
    target_t,
    region_t,
    config,
    phi_init_t=None,
    callback=None,
):
    """Run the CGM optimisation on torch tensors.

    Returns
    -------
    phi_t, E_out_t, cost_history, fidelity_history, n_iters
    """
    _require_torch()
    device = input_amp_t.device
    rdtype = input_amp_t.dtype  # float32 in the hot path
    cdtype = target_t.dtype     # complex64 in the hot path
    shape = tuple(int(s) for s in input_amp_t.shape)

    # -------- initial phase --------
    if phi_init_t is not None:
        phi_t = phi_init_t.to(device=device, dtype=rdtype).clone()
    else:
        phi_np = _initial_phase(shape, config)
        phi_t = torch.as_tensor(phi_np, dtype=rdtype, device=device)

    # -------- sinc envelope --------
    sinc_env_t = None
    if config.fill_factor < 1.0:
        sinc_env_t = _sinc_envelope_t(shape, config.fill_factor, device, rdtype).to(
            cdtype
        )

    # -------- invariants (computed ONCE per call) --------
    input_amp_c_t = input_amp_t.to(cdtype)
    target_masked_t = target_t * region_t  # complex * real -> complex (broadcast)
    norm_A_t = torch.linalg.vector_norm(target_masked_t)
    norm_A_val = float(norm_A_t.item())
    back_A_t = _back_t(target_masked_t, sinc_env_t)

    scale = 10.0 ** config.steepness
    eta_scale = 10.0 ** (config.eta_steepness if config.eta_steepness is not None else config.steepness)
    ew = float(config.efficiency_weight)
    em = float(config.eta_min)

    # -------- closures over invariants --------

    def cost_value(phi):
        E_in = input_amp_c_t * torch.exp(1j * phi)
        E_out = _forward_t(E_in, sinc_env_t)
        B = E_out * region_t
        norm_B = torch.linalg.vector_norm(B)
        if norm_A_val == 0.0 or norm_B.item() == 0.0:
            return torch.zeros((), device=device, dtype=rdtype)
        r = (torch.conj(target_masked_t) * B).sum()
        overlap_real = r.real / (norm_A_t * norm_B)
        cost = scale * (1.0 - overlap_real) ** 2
        if ew > 0.0 or em > 0.0:
            P = (E_out.abs() ** 2).sum()
            if P.item() > 0.0:
                eta = norm_B ** 2 / P
                if ew > 0.0:
                    cost = cost + ew * eta_scale * (1.0 - eta) ** 2
                if em > 0.0:
                    active = (eta < em).to(rdtype)
                    cost = cost + active * eta_scale * (em - eta) ** 2
        return cost

    def cost_and_grad(phi):
        E_in = input_amp_c_t * torch.exp(1j * phi)
        E_out = _forward_t(E_in, sinc_env_t)
        B = E_out * region_t
        norm_B = torch.linalg.vector_norm(B)
        if norm_A_val == 0.0 or norm_B.item() == 0.0:
            zero_cost = torch.zeros((), device=device, dtype=rdtype)
            zero_grad = torch.zeros_like(phi)
            return zero_cost, zero_grad, E_out
        r = (torch.conj(target_masked_t) * B).sum()
        overlap_real = r.real / (norm_A_t * norm_B)

        back_B = _back_t(B, sinc_env_t)
        d_Re_r = (1j * E_in * torch.conj(back_A_t)).real
        raw_B = (1j * E_in * torch.conj(back_B)).real
        d_norm_B = raw_B / norm_B

        d_overlap = d_Re_r / (norm_A_t * norm_B) - overlap_real * d_norm_B / norm_B
        cost = scale * (1.0 - overlap_real) ** 2
        grad = -2.0 * scale * (1.0 - overlap_real) * d_overlap

        if ew > 0.0 or em > 0.0:
            P = (E_out.abs() ** 2).sum()
            if P.item() > 0.0:
                eta = norm_B ** 2 / P
                d_eta = 2.0 * raw_B / P
                if ew > 0.0:
                    cost = cost + ew * eta_scale * (1.0 - eta) ** 2
                    grad = grad - 2.0 * ew * eta_scale * (1.0 - eta) * d_eta
                if em > 0.0:
                    active = (eta < em).to(rdtype)
                    cost = cost + active * eta_scale * (em - eta) ** 2
                    grad = grad - 2.0 * active * eta_scale * (em - eta) * d_eta
        return cost, grad, E_out

    def align_initial_phase(phi):
        E_tmp = _forward_t(input_amp_c_t * torch.exp(1j * phi), sinc_env_t)
        B = E_tmp * region_t
        norm_B = torch.linalg.vector_norm(B)
        if norm_A_val == 0.0 or norm_B.item() == 0.0:
            return phi
        r = (torch.conj(target_masked_t) * B).sum() / (norm_A_t * norm_B)
        if r.abs().item() < 1e-10:
            return phi
        return phi - torch.angle(r).to(rdtype)

    def line_search(phi, direction, cost0):
        dir_max = direction.abs().max().item()
        if dir_max == 0.0:
            return 0.0
        base = (math.pi / 2.0) / dir_max
        # 9 geometric probes spanning ~6 orders of magnitude
        probe_factors = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.6, 1.0, 1.5)
        probes = [base * f for f in probe_factors]
        probe_costs = [cost_value(phi + a * direction).item() for a in probes]
        best_k = int(np.argmin(probe_costs))
        best_cost = probe_costs[best_k]
        if best_cost >= cost0:
            # No improvement from main probes; try a wider fallback range
            # before giving up.  Original used only 2 steps; the wider range
            # prevents premature termination on flat cost landscapes.
            for a in (1e-6 * base, 1e-5 * base, 1e-4 * base,
                      1e-3 * base, 3e-3 * base, 1e-2 * base):
                c = cost_value(phi + a * direction).item()
                if c < cost0:
                    return a
            return 0.0
        # Golden-section refinement on the bracket around the best probe
        lo = probes[max(0, best_k - 1)]
        hi = probes[min(len(probes) - 1, best_k + 1)]
        if lo == hi:
            return probes[best_k]
        inv_phi_gs = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c = b - (b - a) * inv_phi_gs
        d = a + (b - a) * inv_phi_gs
        fc = cost_value(phi + c * direction).item()
        fd = cost_value(phi + d * direction).item()
        for _ in range(20):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - (b - a) * inv_phi_gs
                fc = cost_value(phi + c * direction).item()
            else:
                a, c, fc = c, d, fd
                d = a + (b - a) * inv_phi_gs
                fd = cost_value(phi + d * direction).item()
        return 0.5 * (a + b)

    # -------- algorithm --------

    phi_t = align_initial_phase(phi_t)

    cost_t, grad_t, E_out_t = cost_and_grad(phi_t)
    cost_history = [float(cost_t.item())]
    fidelity_history: list[float] = []
    if config.track_fidelity:
        fidelity_history.append(
            float(fidelity(
                E_out_t.detach().cpu().numpy().astype(np.complex128),
                target_t.detach().cpu().numpy().astype(np.complex128),
                region_t.detach().cpu().numpy().astype(np.float64),
            ))
        )

    direction_t = -grad_t
    grad_norm_sq_t = (grad_t * grad_t).sum()
    n_iters = 0

    for i in range(config.max_iterations):
        alpha = line_search(phi_t, direction_t, cost_history[-1])
        if alpha == 0.0:
            # Line search failed; reset to steepest descent
            direction_t = -grad_t
            alpha = line_search(phi_t, direction_t, cost_history[-1])
            if alpha == 0.0:
                break

        phi_t = phi_t + alpha * direction_t

        new_cost_t, new_grad_t, E_out_t = cost_and_grad(phi_t)
        new_cost = float(new_cost_t.item())
        cost_history.append(new_cost)
        if config.track_fidelity:
            fidelity_history.append(
                float(fidelity(
                    E_out_t.detach().cpu().numpy().astype(np.complex128),
                    target_t.detach().cpu().numpy().astype(np.complex128),
                    region_t.detach().cpu().numpy().astype(np.float64),
                ))
            )
        n_iters = i + 1

        if callback is not None:
            callback(n_iters, new_cost)

        # Relative convergence criterion
        prev_cost = cost_history[-2]
        if abs(prev_cost - new_cost) < config.convergence_threshold * max(
            abs(new_cost), 1.0
        ):
            break

        # Polak-Ribière+ direction update with periodic restart.
        # PR+ is more robust than Fletcher-Reeves on non-convex landscapes:
        # max(0, β) clips negative β, auto-resetting to steepest descent
        # when the gradient reverses direction.
        new_grad_norm_sq_t = (new_grad_t * new_grad_t).sum()
        if grad_norm_sq_t.item() > 0.0 and (i + 1) % 50 != 0:
            pr_num = ((new_grad_t - grad_t) * new_grad_t).sum()
            beta = max(0.0, float((pr_num / grad_norm_sq_t).item()))
        else:
            beta = 0.0  # restart: steepest descent
        direction_t = -new_grad_t + beta * direction_t

        grad_t = new_grad_t
        grad_norm_sq_t = new_grad_norm_sq_t

    return phi_t, E_out_t, cost_history, fidelity_history, n_iters


# ---------------------------------------------------------------------------
# Numpy-native shim helpers (for tests/test_cgm.py backward compat)
# ---------------------------------------------------------------------------
#
# These functions are the SAME math as the torch core, but implemented in
# numpy/float64 so `tests/test_cgm.py` can call them directly for its
# finite-difference gradient checks without running through a torch
# conversion round trip.  They are NOT called by the main torch hot path.


def _initial_phase(shape, config):
    """Structured initial guess phase (Bowman et al.).

    phi = R*(p^2 + q^2) + D*(p*cos(theta) + q*sin(theta))
    """
    ny, nx = shape
    p = np.arange(nx) - (nx - 1) / 2.0
    q = np.arange(ny) - (ny - 1) / 2.0
    pp, qq = np.meshgrid(p, q, indexing="xy")
    phase = config.R * (pp ** 2 + qq ** 2) + config.D * (
        pp * np.cos(config.theta) + qq * np.sin(config.theta)
    )
    return phase


def _compute_overlap(output_field, target_field, measure_region):
    """Compute normalised inner product over measure region (numpy shim)."""
    out_masked = output_field * measure_region
    tgt_masked = target_field * measure_region
    out_norm = np.sqrt(np.sum(np.abs(out_masked) ** 2))
    tgt_norm = np.sqrt(np.sum(np.abs(tgt_masked) ** 2))
    if out_norm == 0 or tgt_norm == 0:
        return 0.0 + 0.0j
    return np.sum(np.conj(tgt_masked / tgt_norm) * (out_masked / out_norm))


def _forward(E_in, sinc_env):
    """Forward propagation with optional sinc envelope (numpy shim)."""
    if sinc_env is not None:
        return realistic_propagate(E_in, sinc_env)
    return fft_propagate(E_in)


def _back(X, sinc_env):
    """Adjoint back-propagation (numpy shim)."""
    if sinc_env is not None:
        return ifft_propagate(sinc_env * X)
    return ifft_propagate(X)


def _cost_function(
    output_field,
    target_field,
    measure_region,
    steepness,
    efficiency_weight=0.0,
    eta_min=0.0,
    eta_steepness=None,
):
    """Cost function (numpy shim for tests)."""
    overlap = _compute_overlap(output_field, target_field, measure_region)
    cost = 10 ** steepness * (1.0 - np.real(overlap)) ** 2
    e_scale = 10 ** (eta_steepness if eta_steepness is not None else steepness)
    if efficiency_weight > 0 or eta_min > 0:
        intensity = np.abs(output_field) ** 2
        P_total = np.sum(intensity)
        if P_total > 0:
            eta = np.sum(intensity * measure_region) / P_total
            if efficiency_weight > 0:
                cost += efficiency_weight * e_scale * (1.0 - eta) ** 2
            if eta_min > 0 and eta < eta_min:
                cost += e_scale * (eta_min - eta) ** 2
    return float(cost)


def _cost_gradient(
    E_in,
    E_out,
    target_field,
    measure_region,
    steepness,
    efficiency_weight=0.0,
    eta_min=0.0,
    sinc_env=None,
    eta_steepness=None,
):
    """Analytical cost gradient dC/d(phi) (numpy shim for tests)."""
    A = target_field * measure_region
    B = E_out * measure_region
    norm_A = np.sqrt(np.sum(np.abs(A) ** 2))
    norm_B = np.sqrt(np.sum(np.abs(B) ** 2))
    if norm_A == 0 or norm_B == 0:
        return np.zeros_like(E_in, dtype=np.float64)

    r = np.sum(np.conj(A) * B)
    overlap_real = np.real(r) / (norm_A * norm_B)

    back_A = _back(A, sinc_env)
    back_B = _back(B, sinc_env)
    d_Re_r = np.real(1j * E_in * np.conj(back_A))
    raw_B = np.real(1j * E_in * np.conj(back_B))
    d_norm_B = raw_B / norm_B

    d_overlap = d_Re_r / (norm_A * norm_B) - overlap_real * d_norm_B / norm_B
    grad = -2.0 * 10 ** steepness * (1.0 - overlap_real) * d_overlap

    e_scale = 10 ** (eta_steepness if eta_steepness is not None else steepness)
    if efficiency_weight > 0 or eta_min > 0:
        P_total = np.sum(np.abs(E_out) ** 2)
        if P_total > 0:
            eta = float(norm_B ** 2 / P_total)
            d_eta = 2.0 * raw_B / P_total
            if efficiency_weight > 0:
                grad += -2.0 * efficiency_weight * e_scale * (1.0 - eta) * d_eta
            if eta_min > 0 and eta < eta_min:
                grad += -2.0 * e_scale * (eta_min - eta) * d_eta

    return grad


def _align_initial_phase(
    phi, input_amplitude, target_field, measure_region, sinc_env=None,
):
    """Rotate initial phase so Re{overlap} = |overlap| (numpy shim)."""
    E_tmp = _forward(input_amplitude * np.exp(1j * phi), sinc_env)
    init_overlap = _compute_overlap(E_tmp, target_field, measure_region)
    if abs(init_overlap) > 1e-10:
        phi = phi - np.angle(init_overlap)
    return phi


def _build_result(
    phi,
    E_out,
    target_field,
    measure_region,
    cost_history,
    n_iterations,
    fidelity_history=None,
):
    """Compute final metrics and return CGMResult (numpy)."""
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


# ---------------------------------------------------------------------------
# Public API: cgm (numpy entry point)
# ---------------------------------------------------------------------------


def cgm(
    input_amplitude: np.ndarray,
    target_field: np.ndarray,
    measure_region: np.ndarray,
    config: CGMConfig = CGMConfig(),
    callback: Callable[[int, float], None] | None = None,
) -> CGMResult:
    """Conjugate Gradient Minimisation for continuous beam shaping.

    Torch-first implementation: the optimisation loop runs on
    ``torch.complex64`` tensors on the best available device (CUDA when
    present, else CPU).  Input/output remain numpy arrays for callers
    that prefer the numpy-native API.

    Parameters
    ----------
    input_amplitude : S_{p,q}, real (ny, nx) -- incident beam amplitude.
    target_field : tau = sqrt(T)*exp(i*Phi), complex (ny, nx) -- target field.
    measure_region : binary (ny, nx) -- region of interest Omega.
    config : algorithm parameters.  ``config.device`` selects the torch
        device ("cuda", "cpu", or ``None`` for auto).
    callback : optional function called each iteration with (i, cost).
    """
    _require_torch()
    device = _resolve_device(config.device)
    cdtype, rdtype = _hot_dtypes()

    input_amp_t = torch.as_tensor(
        np.ascontiguousarray(np.asarray(input_amplitude).real), dtype=rdtype, device=device
    )
    target_t = torch.as_tensor(
        np.ascontiguousarray(np.asarray(target_field)), dtype=cdtype, device=device
    )
    region_t = torch.as_tensor(
        np.ascontiguousarray(np.asarray(measure_region)), dtype=rdtype, device=device
    )

    phi_init_t = None
    if config.initial_phase is not None:
        phi_init_t = torch.as_tensor(
            np.ascontiguousarray(config.initial_phase), dtype=rdtype, device=device
        )

    phi_t, E_out_t, cost_history, fid_history, n_iters = _run_cgm_torch(
        input_amp_t, target_t, region_t, config, phi_init_t, callback,
    )

    phi_np = phi_t.detach().cpu().numpy().astype(np.float64)
    E_out_np = E_out_t.detach().cpu().numpy().astype(np.complex128)

    return _build_result(
        phi_np, E_out_np, target_field, measure_region,
        cost_history, n_iters, fid_history,
    )


# ---------------------------------------------------------------------------
# Public API: CGM_phase_generate (torch entry point)
# ---------------------------------------------------------------------------


def CGM_phase_generate(
    initSLMAmp,
    initSLMPhase,
    targetAmp,
    max_iterations: int = 200,
    steepness: int = 9,
    R: float = 0.0,
    D: float = 0.0,
    theta: float = 0.0,
    efficiency_weight: float = 0.0,
    eta_min: float = 0.0,
    eta_steepness: float | None = None,
    fill_factor: float = 1.0,
    margin: int = 5,
    convergence_threshold: float = 1e-5,
    Plot: bool = False,
):
    """Torch-tensor entry point for CGM, mirroring ``WGS_phase_generate``.

    Pass torch tensors in, receive a real-valued torch phase tensor out.
    The measure region is auto-derived from the target via
    :func:`slm.targets.measure_region`.

    Parameters
    ----------
    initSLMAmp : 2D torch tensor, real
        Incident SLM-plane amplitude.
    initSLMPhase : 2D torch tensor, real
        Initial guess phase (e.g. with Fresnel lens baked in).  Overrides
        the analytical R/D/theta default.
    targetAmp : 2D torch tensor, real or complex
        Focal-plane target field.  Real targets are promoted to complex
        with zero phase; complex targets are used as-is.
    max_iterations, steepness, R, D, theta, efficiency_weight, eta_min,
    fill_factor, convergence_threshold :
        Forwarded to :class:`CGMConfig`.
    margin : int
        Dilation margin (pixels) around the target used to build the CGM
        measure region.
    Plot : bool
        If True, plot the cost history and print a one-line summary of
        fidelity, efficiency, and iteration count.

    Returns
    -------
    torch.Tensor
        Real-valued (float32) SLM phase in radians, same shape and device
        as ``initSLMAmp``.
    """
    _require_torch()
    from slm.targets import measure_region as _measure_region

    caller_device = initSLMAmp.device

    # Prefer running on the caller's device if it matches the auto-selected
    # (CUDA if available, else CPU); otherwise use auto.
    auto_device = _resolve_device(None)
    if caller_device.type == auto_device.type:
        run_device = caller_device
    else:
        run_device = auto_device

    cdtype, rdtype = _hot_dtypes()

    input_amp_t = initSLMAmp.detach().to(device=run_device, dtype=rdtype)
    phi_init_t = initSLMPhase.detach().to(device=run_device, dtype=rdtype)

    if torch.is_complex(targetAmp):
        target_t = targetAmp.detach().to(device=run_device, dtype=cdtype)
    else:
        # Real target -> complex with phase = 0
        target_t = targetAmp.detach().to(device=run_device).to(cdtype)

    # Build the measure region using the numpy utility (runs on CPU)
    target_np_for_region = target_t.detach().cpu().numpy()
    region_np = _measure_region(
        target_np_for_region.shape, target_np_for_region, margin=margin
    )
    region_t = torch.as_tensor(region_np, dtype=rdtype, device=run_device)

    config = CGMConfig(
        max_iterations=max_iterations,
        steepness=steepness,
        convergence_threshold=convergence_threshold,
        R=R,
        D=D,
        theta=theta,
        efficiency_weight=efficiency_weight,
        eta_min=eta_min,
        eta_steepness=eta_steepness,
        fill_factor=fill_factor,
        device=str(run_device),
    )

    phi_t, E_out_t, cost_history, _, n_iters = _run_cgm_torch(
        input_amp_t, target_t, region_t, config, phi_init_t,
    )

    if Plot:
        import matplotlib.pyplot as plt

        # Compute summary metrics on the CPU via the numpy helpers
        E_out_np = E_out_t.detach().cpu().numpy().astype(np.complex128)
        target_np = target_t.detach().cpu().numpy().astype(np.complex128)
        region_np2 = region_t.detach().cpu().numpy().astype(np.float64)
        f = fidelity(E_out_np, target_np, region_np2)
        e = efficiency(E_out_np, region_np2)

        plt.figure()
        plt.plot(cost_history)
        plt.yscale("log")
        plt.grid()
        plt.xlabel("Iteration")
        plt.ylabel("CGM cost")
        plt.title("CGM convergence")
        plt.show()
        print(
            f"CGM: {n_iters} iter, F={f:.4f}, eta={e:.4f}"
        )

    return phi_t.to(device=caller_device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Public API: tophat_phase_generate (convenience wrapper, preserved)
# ---------------------------------------------------------------------------


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
    initSLMAmp : 2D array -- incident beam amplitude on SLM plane.
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
        print(
            f"CGM top-hat: {result.n_iterations} iter, "
            f"F={result.final_fidelity:.4f}, "
            f"eta={result.final_efficiency:.4f}, "
            f"eps_nu={result.final_non_uniformity:.4f}"
        )

    return result.slm_phase
