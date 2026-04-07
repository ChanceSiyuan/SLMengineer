"""Phase-to-screen conversion for driving a physical SLM.

Converts continuous phase arrays (float64, [-pi, pi]) produced by algorithms
into uint8 screens (0-255) suitable for SLM display hardware.

Ported from ~/slm-code/SLMGeneration.py phase_to_screen().
"""

from __future__ import annotations

import numpy as np


def phase_to_uint8(phase: np.ndarray) -> np.ndarray:
    """Convert phase in [-pi, pi] to uint8 grey levels [0, 255].

    Linear mapping: -pi -> 0, +pi -> 255 (with 2*pi wrapping).
    """
    return np.around((phase + np.pi) / (2 * np.pi) * 256).astype(np.uint8)


def crop_to_slm(
    phase: np.ndarray,
    slm_resolution: tuple[int, int],
) -> np.ndarray:
    """Center-crop a computation-grid phase to the physical SLM active area.

    When the computation grid (e.g. 4096x4096) is larger than the SLM
    active area, this extracts the largest centered square that fits
    the smaller SLM dimension.

    Parameters
    ----------
    phase : (rows, cols) phase array from the algorithm.
    slm_resolution : (width, height) of the physical SLM in pixels,
        matching the convention in hamamatsu_test_config.json.

    Returns
    -------
    Cropped phase array fitting within the SLM active area.
    """
    rows, cols = phase.shape
    slm_w, slm_h = slm_resolution
    crop_size = min(slm_w, slm_h)

    cy, cx = rows // 2, cols // 2
    half = crop_size // 2

    r0 = cy - half
    r1 = r0 + crop_size
    c0 = cx - half
    c1 = c0 + crop_size

    return phase[r0:r1, c0:c1]


def phase_to_screen(
    phase: np.ndarray,
    slm_resolution: tuple[int, int],
) -> np.ndarray:
    """Full pipeline: crop, quantise, and embed phase onto an SLM screen.

    1. Center-crop the phase to the SLM active area.
    2. Convert to uint8 (0-255).
    3. Embed centered within the full SLM screen resolution.

    Parameters
    ----------
    phase : (rows, cols) float phase in [-pi, pi] from an algorithm.
    slm_resolution : (width, height) of the physical SLM.

    Returns
    -------
    uint8 array of shape (height, width) ready for SLMDisplay.update_array().
    """
    slm_w, slm_h = slm_resolution

    cropped = crop_to_slm(phase, slm_resolution)
    quantised = phase_to_uint8(cropped)

    screen = np.zeros((slm_h, slm_w), dtype=np.uint8)
    qh, qw = quantised.shape
    r0 = (slm_h - qh) // 2
    c0 = (slm_w - qw) // 2
    screen[r0 : r0 + qh, c0 : c0 + qw] = quantised

    return screen
