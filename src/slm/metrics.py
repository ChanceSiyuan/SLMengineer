"""Quality metrics for hologram evaluation."""

from __future__ import annotations

import numpy as np


def uniformity(intensities: np.ndarray) -> float:
    """Intensity non-uniformity: std(I) / mean(I).

    For discrete spot arrays (Kim et al.), target is < 0.5%.
    Input should be a 1D array of spot intensities.
    """
    intensities = np.asarray(intensities, dtype=np.float64)
    mean_val = np.mean(intensities)
    if mean_val == 0:
        return 0.0
    return float(np.std(intensities) / mean_val)


def efficiency(
    output_field: np.ndarray,
    region_mask: np.ndarray,
) -> float:
    """Light-usage efficiency: fraction of total output power within target region.

    eta = sum(|E|^2 in Omega) / sum(|E|^2 total)
    """
    intensity = np.abs(output_field) ** 2
    total_power = np.sum(intensity)
    if total_power == 0:
        return 0.0
    region_power = np.sum(intensity * region_mask)
    return float(region_power / total_power)


def modulation_efficiency(
    output_field: np.ndarray,
    spot_positions: np.ndarray,
) -> float:
    """Fraction of total power at discrete spot positions.

    Kim et al. target: > 90%.
    """
    total_power = np.sum(np.abs(output_field) ** 2)
    if total_power == 0:
        return 0.0
    spot_positions = np.asarray(spot_positions)
    spot_power = np.sum(
        np.abs(output_field[spot_positions[:, 0], spot_positions[:, 1]]) ** 2
    )
    return float(spot_power / total_power)


def fidelity(
    output_field: np.ndarray,
    target_field: np.ndarray,
    region: np.ndarray | None = None,
) -> float:
    """Complex-field fidelity F = |sum(tau* . E_out)|^2 (Bowman et al.).

    Both fields are normalized over the region before computing.
    """
    if region is not None:
        out = output_field * region
        tgt = target_field * region
    else:
        out = output_field
        tgt = target_field

    out_norm = np.sqrt(np.sum(np.abs(out) ** 2))
    tgt_norm = np.sqrt(np.sum(np.abs(tgt) ** 2))
    if out_norm == 0 or tgt_norm == 0:
        return 0.0

    overlap = np.sum(np.conj(tgt) * out)
    return float(np.abs(overlap / (out_norm * tgt_norm)) ** 2)


def phase_error(
    output_phase: np.ndarray,
    target_phase: np.ndarray,
    region: np.ndarray,
) -> float:
    """Relative phase error within measure region (Bowman et al.).

    Includes cyclic correction P to find the best global phase offset.
    """
    mask = region > 0
    if not np.any(mask):
        return 0.0

    phi_out = output_phase[mask]
    phi_tgt = target_phase[mask]
    diff = phi_tgt - phi_out

    # Find optimal global offset P to minimize error
    P = np.angle(np.sum(np.exp(1j * diff)))
    residual = np.angle(np.exp(1j * (diff - P)))

    denom = np.sum(phi_tgt**2)
    if denom == 0:
        return float(np.sum(residual**2))
    return float(np.sum(residual**2) / denom)


def non_uniformity_error(
    output_intensity: np.ndarray,
    target_intensity: np.ndarray,
    uniform_mask: np.ndarray,
) -> float:
    """Non-uniformity error for flat-intensity regions (Bowman et al.).

    epsilon_nu = sum(|M*(I_tilde - I_a)|^2) / sum(|M*T_tilde|^2)
    """
    mask = uniform_mask > 0
    if not np.any(mask):
        return 0.0

    # Normalize intensities over the mask region
    I_sum = np.sum(output_intensity[mask])
    T_sum = np.sum(target_intensity[mask])

    if I_sum == 0 or T_sum == 0:
        return 0.0

    I_tilde = output_intensity[mask] / I_sum
    T_tilde = target_intensity[mask] / T_sum
    I_a = np.mean(I_tilde)

    numerator = np.sum((I_tilde - I_a) ** 2)
    denominator = np.sum(T_tilde**2)
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
