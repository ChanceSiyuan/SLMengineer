"""Gerchberg-Saxton iterative phase retrieval algorithm."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from slm.metrics import uniformity
from slm.propagation import (
    fft_propagate,
    ifft_propagate,
    realistic_ifft_propagate,
    realistic_propagate,
)


@dataclass
class GSResult:
    """Result container for GS-family algorithms."""

    slm_phase: np.ndarray
    focal_field: np.ndarray
    uniformity_history: list[float] = field(default_factory=list)
    efficiency_history: list[float] = field(default_factory=list)
    n_iterations: int = 0


def gs(
    initial_field: np.ndarray,
    target: np.ndarray,
    n_iterations: int = 100,
    callback: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
    sinc_env: np.ndarray | None = None,
) -> GSResult:
    """Basic Gerchberg-Saxton iterative phase retrieval.

    Parameters
    ----------
    initial_field : complex (ny, nx) -- L_0 (Gaussian amplitude, random phase).
    target : complex (ny, nx) -- desired focal plane field.
    n_iterations : number of iterations.
    callback : optional function called each iteration with (i, slm_field, focal_field).
    sinc_env : optional precomputed sinc envelope for pixelation modeling.

    Algorithm per iteration:
        1. R = FFT(L)           [with optional sinc envelope]
        2. R' = |target| * exp(i * angle(R))   [replace amplitude, keep phase]
        3. L' = IFFT(R')        [with optional sinc pre-compensation]
        4. L = |L_0| * exp(i * angle(L'))      [restore SLM amplitude, keep phase]
    """
    target_amp = np.abs(target)
    slm_amp = np.abs(initial_field)
    L = initial_field.copy()

    uniformity_hist = []
    efficiency_hist = []

    spot_mask = target_amp > 0
    total_power = float(np.sum(slm_amp**2))  # constant under ortho FFT

    _fwd = (
        (lambda f: realistic_propagate(f, sinc_env))
        if sinc_env is not None
        else fft_propagate
    )
    _inv = (
        (lambda f: realistic_ifft_propagate(f, sinc_env))
        if sinc_env is not None
        else ifft_propagate
    )

    for i in range(n_iterations):
        R = _fwd(L)

        spot_intensities = np.abs(R[spot_mask]) ** 2
        if len(spot_intensities) > 0:
            uniformity_hist.append(uniformity(spot_intensities))
            if total_power > 0:
                efficiency_hist.append(float(np.sum(spot_intensities) / total_power))
            else:
                efficiency_hist.append(0.0)

        if callback is not None:
            callback(i, L, R)

        # Replace amplitude with target, keep phase
        R_prime = target_amp * np.exp(1j * np.angle(R))

        # Backward propagate (sinc pre-compensation if active)
        L_prime = _inv(R_prime)

        # Restore SLM amplitude, keep updated phase
        L = slm_amp * np.exp(1j * np.angle(L_prime))

    # Final forward propagation
    focal_field = _fwd(L)

    return GSResult(
        slm_phase=np.angle(L),
        focal_field=focal_field,
        uniformity_history=uniformity_hist,
        efficiency_history=efficiency_hist,
        n_iterations=n_iterations,
    )


def gs_phase_correction(
    slm_phase: np.ndarray,
    focal_amp_known: np.ndarray,
    slm_amp: np.ndarray | None = None,
    n_iterations: int = 4,
    tol: float | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Closed-loop GS phase correction (camera-feedback variant of `gs`).

    Differs from :func:`gs` in three ways:
      1. ``focal_amp_known`` comes from a camera measurement (with the
         measured 2D ROI embedded inside an otherwise-ideal focal-plane
         amplitude), not the design target.
      2. The starting SLM-plane field is ``slm_amp * exp(i*slm_phase)``
         using the *current* SLM phase actually being displayed, instead
         of an arbitrary initial guess.
      3. Returns a phase **increment** ``delta_phase`` so that the next
         SLM update is ``slm_phase + delta_phase``, plus the GS error
         history.

    The SLM-plane amplitude IS still constrained to ``slm_amp`` between
    iterations (matching :func:`gs` and the reference algorithm
    ``references/Top-hat code_2/IFT_ITER.py:gs_vortex_correction`` which
    uses ``profile_s = L`` as the global amplitude). Without this
    constraint the FFT/IFFT round-trip is degenerate and iterations
    contribute nothing.

    Parameters
    ----------
    slm_phase : real (Ny, Nx) — current SLM phase actually displayed.
    focal_amp_known : real (Ny, Nx) — focal amplitude with measured ROI
        embedded; outside the ROI it should be the ideal focal amplitude
        produced by ``slm_phase``.
    slm_amp : real (Ny, Nx) input Gaussian amplitude on the SLM plane.
        Pass ``SLM.initGaussianAmp``. If ``None``, uniform unit
        amplitude is used (less physical, but simpler for synthetic tests).
    n_iterations : small (default 4); intentionally undercooked to avoid
        over-fitting to camera noise.
    tol : optional convergence tolerance on relative amplitude error.

    Returns
    -------
    delta_phase : real (Ny, Nx) wrapped to [-pi, pi]; the SLM phase
        update is ``slm_phase + delta_phase``.
    err_history : list of relative amplitude errors per iteration.
    """
    if slm_amp is None:
        slm_amp_arr = np.ones_like(slm_phase, dtype=np.float64)
    else:
        slm_amp_arr = np.asarray(slm_amp, dtype=np.float64)
    L = (slm_amp_arr * np.exp(1j * slm_phase)).astype(np.complex128)
    R = fft_propagate(L)
    err_history: list[float] = []
    for i in range(n_iterations):
        R_known = focal_amp_known * np.exp(1j * np.angle(R))
        L_new = ifft_propagate(R_known)
        # SLM-plane constraint: restore input Gaussian amplitude, keep updated phase.
        L = slm_amp_arr * np.exp(1j * np.angle(L_new))
        R = fft_propagate(L)
        err = float(
            np.linalg.norm(np.abs(R) - focal_amp_known)
            / max(np.linalg.norm(focal_amp_known), 1e-12)
        )
        err_history.append(err)
        if tol is not None and i > 0 and abs(err_history[-2] - err) < tol:
            break
    delta_phase = np.angle(L) - slm_phase
    delta_phase = np.angle(np.exp(1j * delta_phase))
    return delta_phase, err_history
