"""Conjugate Gradient Minimisation for continuous beam shaping (Bowman et al.).

Torch-first implementation.  The only public entry point is
:func:`CGM_phase_generate` — a torch-tensor API that mirrors
``slm.wgs.WGS_phase_generate`` and is what every hardware script under
``scripts/`` calls.  :class:`CGMConfig` + :func:`_initial_phase` are kept
public because `testfile_{tophat,lg,ring,gline}.py` build a Bowman-style
quadratic+tilt guess via ``_initial_phase(shape, CGMConfig(R=..., D=..., theta=...))``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dependency
    torch = None  # type: ignore[assignment]

from slm.metrics import efficiency, fidelity


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
# Initial phase + public torch entry point
# ---------------------------------------------------------------------------


def _initial_phase(shape, config):
    """Structured initial guess phase (Bowman et al.).

    phi = R*(p^2 + q^2) + D*(p*cos(theta) + q*sin(theta))

    Public despite the leading underscore: several hardware scripts
    (``testfile_tophat``, ``testfile_lg``, ``testfile_ring``,
    ``testfile_gline``) import this to build their own init phase via
    ``CGMConfig(R=..., D=..., theta=...)``.
    """
    ny, nx = shape
    p = np.arange(nx) - (nx - 1) / 2.0
    q = np.arange(ny) - (ny - 1) / 2.0
    pp, qq = np.meshgrid(p, q, indexing="xy")
    return config.R * (pp ** 2 + qq ** 2) + config.D * (
        pp * np.cos(config.theta) + qq * np.sin(config.theta)
    )


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
        plt.savefig("cgm_convergence.png", dpi=300)
        plt.close()
        print(
            f"CGM: {n_iters} iter, F={f:.4f}, eta={e:.4f}"
        )

    return phi_t.to(device=caller_device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 1D CGM path (issue #21: dimension-decomposed light-sheet)
# ---------------------------------------------------------------------------
#
# When the input beam is 2D-separable Gaussian and the SLM phase is set
# constant in y, the focal-plane field factors and we only need to shape
# the x-axis.  The helpers below mirror the 2D torch path with 1D FFTs
# and a 1D sinc envelope; the optimiser body is otherwise the same.


def _fft_propagate_1d_t(field):
    return torch.fft.fftshift(
        torch.fft.fft(torch.fft.ifftshift(field), norm="ortho")
    )


def _ifft_propagate_1d_t(field):
    return torch.fft.fftshift(
        torch.fft.ifft(torch.fft.ifftshift(field), norm="ortho")
    )


def _forward_1d_t(E_in, sinc_env_t):
    out = _fft_propagate_1d_t(E_in)
    if sinc_env_t is not None:
        out = out * sinc_env_t
    return out


def _back_1d_t(X, sinc_env_t):
    if sinc_env_t is not None:
        return _ifft_propagate_1d_t(sinc_env_t * X)
    return _ifft_propagate_1d_t(X)


def _sinc_envelope_1d_t(n, fill_factor, device, rdtype):
    ix = torch.arange(n, device=device, dtype=rdtype) - (n // 2)
    return torch.sinc(ix * fill_factor / n)


def _initial_phase_1d(n, config):
    """1D Bowman-style guess phase: phi(p) = R*p^2 + D*p.

    ``config.theta`` is intentionally ignored — in 1D there is no second
    axis to tilt toward, so the linear term along p absorbs any shift.
    """
    p = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    return config.R * p ** 2 + config.D * p


def _run_cgm_torch_1d(
    input_amp_t,
    target_t,
    region_t,
    config,
    phi_init_t=None,
    callback=None,
):
    """1D port of :func:`_run_cgm_torch`; algorithmically identical, only
    the FFT / sinc helpers differ."""
    _require_torch()
    device = input_amp_t.device
    rdtype = input_amp_t.dtype
    cdtype = target_t.dtype
    n = int(input_amp_t.shape[0])

    if phi_init_t is not None:
        phi_t = phi_init_t.to(device=device, dtype=rdtype).clone()
    else:
        phi_t = torch.as_tensor(
            _initial_phase_1d(n, config), dtype=rdtype, device=device,
        )

    sinc_env_t = None
    if config.fill_factor < 1.0:
        sinc_env_t = _sinc_envelope_1d_t(n, config.fill_factor, device, rdtype).to(cdtype)

    input_amp_c_t = input_amp_t.to(cdtype)
    target_masked_t = target_t * region_t
    norm_A_t = torch.linalg.vector_norm(target_masked_t)
    norm_A_val = float(norm_A_t.item())
    back_A_t = _back_1d_t(target_masked_t, sinc_env_t)

    scale = 10.0 ** config.steepness
    eta_scale = 10.0 ** (config.eta_steepness if config.eta_steepness is not None else config.steepness)
    ew = float(config.efficiency_weight)
    em = float(config.eta_min)

    def cost_value(phi):
        E_in = input_amp_c_t * torch.exp(1j * phi)
        E_out = _forward_1d_t(E_in, sinc_env_t)
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
        E_out = _forward_1d_t(E_in, sinc_env_t)
        B = E_out * region_t
        norm_B = torch.linalg.vector_norm(B)
        if norm_A_val == 0.0 or norm_B.item() == 0.0:
            return (
                torch.zeros((), device=device, dtype=rdtype),
                torch.zeros_like(phi),
                E_out,
            )
        r = (torch.conj(target_masked_t) * B).sum()
        overlap_real = r.real / (norm_A_t * norm_B)

        back_B = _back_1d_t(B, sinc_env_t)
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
        E_tmp = _forward_1d_t(input_amp_c_t * torch.exp(1j * phi), sinc_env_t)
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
        probe_factors = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.6, 1.0, 1.5)
        probes = [base * f for f in probe_factors]
        probe_costs = [cost_value(phi + a * direction).item() for a in probes]
        best_k = int(np.argmin(probe_costs))
        best_cost = probe_costs[best_k]
        if best_cost >= cost0:
            for a in (1e-6 * base, 1e-5 * base, 1e-4 * base,
                      1e-3 * base, 3e-3 * base, 1e-2 * base):
                c = cost_value(phi + a * direction).item()
                if c < cost0:
                    return a
            return 0.0
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

        prev_cost = cost_history[-2]
        if abs(prev_cost - new_cost) < config.convergence_threshold * max(
            abs(new_cost), 1.0
        ):
            break

        new_grad_norm_sq_t = (new_grad_t * new_grad_t).sum()
        if grad_norm_sq_t.item() > 0.0 and (i + 1) % 50 != 0:
            pr_num = ((new_grad_t - grad_t) * new_grad_t).sum()
            beta = max(0.0, float((pr_num / grad_norm_sq_t).item()))
        else:
            beta = 0.0
        direction_t = -new_grad_t + beta * direction_t

        grad_t = new_grad_t
        grad_norm_sq_t = new_grad_norm_sq_t

    return phi_t, E_out_t, cost_history, fidelity_history, n_iters


def CGM_phase_generate_1d(
    initSLMAmp,
    initSLMPhase,
    targetAmp,
    max_iterations: int = 200,
    steepness: int = 9,
    R: float = 0.0,
    D: float = 0.0,
    efficiency_weight: float = 0.0,
    eta_min: float = 0.0,
    eta_steepness: float | None = None,
    fill_factor: float = 1.0,
    margin: int = 5,
    convergence_threshold: float = 1e-5,
    Plot: bool = False,
):
    """1D companion to :func:`CGM_phase_generate` (issue #21).

    All inputs are length-N torch tensors instead of 2D.  Builds the 1D
    measure region via :func:`slm.targets.measure_region_1d` and runs
    :func:`_run_cgm_torch_1d`.  Returns a length-N float32 phase tensor
    on the caller's device.
    """
    _require_torch()
    from slm.targets import measure_region_1d

    assert initSLMAmp.ndim == 1, "initSLMAmp must be 1D"
    assert initSLMPhase.ndim == 1, "initSLMPhase must be 1D"
    assert targetAmp.ndim == 1, "targetAmp must be 1D"

    caller_device = initSLMAmp.device
    auto_device = _resolve_device(None)
    run_device = caller_device if caller_device.type == auto_device.type else auto_device
    cdtype, rdtype = _hot_dtypes()

    input_amp_t = initSLMAmp.detach().to(device=run_device, dtype=rdtype)
    phi_init_t = initSLMPhase.detach().to(device=run_device, dtype=rdtype)
    if torch.is_complex(targetAmp):
        target_t = targetAmp.detach().to(device=run_device, dtype=cdtype)
    else:
        target_t = targetAmp.detach().to(device=run_device).to(cdtype)

    region_np = measure_region_1d(target_t.detach().cpu().numpy(), margin=margin)
    region_t = torch.as_tensor(region_np, dtype=rdtype, device=run_device)

    config = CGMConfig(
        max_iterations=max_iterations,
        steepness=steepness,
        convergence_threshold=convergence_threshold,
        R=R,
        D=D,
        theta=0.0,
        efficiency_weight=efficiency_weight,
        eta_min=eta_min,
        eta_steepness=eta_steepness,
        fill_factor=fill_factor,
        device=str(run_device),
    )

    phi_t, E_out_t, cost_history, _, n_iters = _run_cgm_torch_1d(
        input_amp_t, target_t, region_t, config, phi_init_t,
    )

    if Plot:
        import matplotlib.pyplot as plt

        E_out_np = E_out_t.detach().cpu().numpy().astype(np.complex128)
        target_np = target_t.detach().cpu().numpy().astype(np.complex128)
        region_np2 = region_t.detach().cpu().numpy().astype(np.float64)
        f = fidelity(E_out_np, target_np, region_np2)
        e = efficiency(E_out_np, region_np2)
        plt.figure()
        plt.plot(cost_history)
        plt.yscale("log"); plt.grid()
        plt.xlabel("Iteration"); plt.ylabel("CGM cost")
        plt.title("CGM 1D convergence")
        plt.show()
        print(f"CGM-1D: {n_iters} iter, F={f:.4f}, eta={e:.4f}")

    return phi_t.to(device=caller_device, dtype=torch.float32)
