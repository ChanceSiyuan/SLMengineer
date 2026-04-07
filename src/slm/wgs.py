"""Weighted Gerchberg-Saxton and Phase-Fixed WGS algorithms (Kim et al.)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from slm.gs import GSResult
from slm.metrics import uniformity
from slm.propagation import (
    fft_propagate,
    ifft_propagate,
    realistic_ifft_propagate,
    realistic_propagate,
)


@dataclass
class WGSConfig:
    """Configuration for WGS algorithm."""

    n_iterations: int = 200
    uniformity_threshold: float = 0.005
    phase_fix_iteration: int | None = (
        None  # N for phase-fixed variant; None = never fix
    )


@dataclass
class WGSResult(GSResult):
    """Extended result with WGS-specific data."""

    weight_history: list[float] = field(default_factory=list)
    phase_fixed_at: int | None = None
    spot_phase_history: list[np.ndarray] = field(default_factory=list)
    spot_amplitude_history: list[np.ndarray] = field(default_factory=list)


def wgs(
    initial_field: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    config: WGSConfig = WGSConfig(),
    callback: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
    sinc_env: np.ndarray | None = None,
) -> WGSResult:
    """Weighted Gerchberg-Saxton with optional phase fixing.

    Parameters
    ----------
    initial_field : L_0, complex (ny, nx) -- Gaussian amplitude, random phase.
    target : E, complex (ny, nx) -- desired focal plane amplitudes.
    mask : E_mask, binary (ny, nx) -- 1 at target spot positions, 0 elsewhere.
    config : algorithm parameters.
    callback : optional per-iteration callback(i, slm_field, focal_field).

    The algorithm follows Kim et al.:
        1. R_i = FFT(L_i)
        2. R_mask_i = mask * R_i
        3. g_i update: cumulative weight correction
        4. Phase decision (update or freeze based on uniformity threshold)
        5. Optional phase fix at iteration N (Kim's key innovation)
        6. R_i' = target * g_i * exp(i * phase_i)
        7. L' = IFFT(R_i')
        8. L_{i+1} = |L_0| * exp(i * angle(L'))
    """
    target_amp = np.abs(target)
    slm_amp = np.abs(initial_field)
    mask_bool = mask > 0

    L = initial_field.copy()
    g = np.ones_like(target_amp)  # cumulative weight
    fixed_phase = None
    phase_fixed_at = None
    current_phase = None

    uniformity_hist = []
    efficiency_hist = []
    weight_hist = []
    spot_phase_hist = []
    spot_amp_hist = []

    # Parseval: total power is constant under ortho FFT
    total_power = float(np.sum(slm_amp**2))

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

    for i in range(config.n_iterations):
        R = _fwd(L)

        spot_amps = np.abs(R[mask_bool])
        if len(spot_amps) == 0:
            break
        mean_amp = np.mean(spot_amps)

        spot_intensities = spot_amps**2
        uniformity_hist.append(uniformity(spot_intensities))
        if total_power > 0:
            efficiency_hist.append(float(np.sum(spot_intensities) / total_power))
        else:
            efficiency_hist.append(0.0)
        weight_hist.append(float(np.std(g[mask_bool])))
        spot_phase_hist.append(np.angle(R[mask_bool]))
        spot_amp_hist.append(spot_amps)

        if callback is not None:
            callback(i, L, R)

        # Cumulative weight update (only at spot positions)
        if mean_amp > 0:
            g[mask_bool] *= mean_amp / np.maximum(spot_amps, 1e-30)

        # Phase decision
        if config.phase_fix_iteration is not None and i >= config.phase_fix_iteration:
            if fixed_phase is None:
                fixed_phase = np.angle(R)
                phase_fixed_at = i
            current_phase = fixed_phase
        elif i == 0 or (
            mean_amp > 0
            and np.max(spot_amps) / mean_amp - 1.0 > config.uniformity_threshold
        ):
            current_phase = np.angle(R)
        # else: keep current_phase from previous iteration

        R_prime = target_amp * g * np.exp(1j * current_phase)
        L_prime = _inv(R_prime)
        L = slm_amp * np.exp(1j * np.angle(L_prime))

    # Final propagation
    focal_field = _fwd(L)

    return WGSResult(
        slm_phase=np.angle(L),
        focal_field=focal_field,
        uniformity_history=uniformity_hist,
        efficiency_history=efficiency_hist,
        n_iterations=config.n_iterations,
        weight_history=weight_hist,
        phase_fixed_at=phase_fixed_at,
        spot_phase_history=spot_phase_hist,
        spot_amplitude_history=spot_amp_hist,
    )


def phase_fixed_wgs(
    initial_field: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    phase_fix_iteration: int = 12,
    n_iterations: int = 200,
    uniformity_threshold: float = 0.005,
    callback: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
    sinc_env: np.ndarray | None = None,
) -> WGSResult:
    """Convenience wrapper: WGS with phase fixing at iteration N.

    Kim et al. fix phase at iteration 12 with ~91.2% modulation efficiency,
    then continue for ~200 total iterations to reach <0.5% non-uniformity.
    """
    config = WGSConfig(
        n_iterations=n_iterations,
        uniformity_threshold=uniformity_threshold,
        phase_fix_iteration=phase_fix_iteration,
    )
    return wgs(initial_field, target, mask, config, callback, sinc_env)
