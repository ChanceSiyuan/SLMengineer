"""LUT correction and manufacturer calibration for SLM phase screens.

Ported from ~/slm-code/IMGpy.py SLM_screen_Correct().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_calibration_bmp(path: str | Path) -> np.ndarray:
    """Load a manufacturer per-pixel calibration BMP as a uint8 array.

    These are phase-correction maps supplied by the SLM manufacturer
    (e.g. ``CAL_LSH0905549_1013nm.bmp`` from Hamamatsu).
    """
    from PIL import Image

    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def apply_lut_correction(
    screen: np.ndarray,
    lut_value: int = 224,
    calibration: np.ndarray | None = None,
) -> np.ndarray:
    """Apply LUT scaling and per-pixel calibration to a phase screen.

    Pipeline:
        1. Add calibration offset (uint8 wrapping addition).
        2. Scale by ``lut_value / 256`` to map into the SLM's active
           phase range (accounts for the voltage-to-phase LUT).

    Parameters
    ----------
    screen : uint8 phase screen from :func:`phase_to_screen`.
    lut_value : LUT scaling factor (0-255). Typical Hamamatsu values:
        207 (for 1013 nm) or 224 (for 850 nm).
    calibration : per-pixel correction array from :func:`load_calibration_bmp`.
        Must match the shape of *screen*. If None, no calibration offset
        is applied.
    """
    corrected = screen.astype(np.uint16)
    if calibration is not None:
        corrected = corrected + calibration.astype(np.uint16)
    scaled = (corrected / 256 * lut_value).astype(np.uint8)
    return scaled
