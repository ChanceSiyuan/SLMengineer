"""Gerchberg-Saxton iterative phase retrieval algorithm."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from slm.metrics import uniformity
from slm.propagation import fft_propagate, ifft_propagate


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
) -> GSResult:
    """Basic Gerchberg-Saxton iterative phase retrieval.

    Parameters
    ----------
    initial_field : complex (ny, nx) -- L_0 (Gaussian amplitude, random phase).
    target : complex (ny, nx) -- desired focal plane field.
    n_iterations : number of iterations.
    callback : optional function called each iteration with (i, slm_field, focal_field).

    Algorithm per iteration:
        1. R = FFT(L)
        2. R' = |target| * exp(i * angle(R))   [replace amplitude, keep phase]
        3. L' = IFFT(R')
        4. L = |L_0| * exp(i * angle(L'))      [restore SLM amplitude, keep phase]
    """
    target_amp = np.abs(target)
    slm_amp = np.abs(initial_field)
    L = initial_field.copy()

    uniformity_hist = []
    efficiency_hist = []

    spot_mask = target_amp > 0
    total_power = float(np.sum(slm_amp**2))  # constant under ortho FFT

    for i in range(n_iterations):
        R = fft_propagate(L)

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

        # Backward propagate
        L_prime = ifft_propagate(R_prime)

        # Restore SLM amplitude, keep updated phase
        L = slm_amp * np.exp(1j * np.angle(L_prime))

    # Final forward propagation
    focal_field = fft_propagate(L)

    return GSResult(
        slm_phase=np.angle(L),
        focal_field=focal_field,
        uniformity_history=uniformity_hist,
        efficiency_history=efficiency_hist,
        n_iterations=n_iterations,
    )
