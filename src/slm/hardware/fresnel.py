"""Fresnel lens phase generation for focal-plane shifting.

A Fresnel lens added to the hologram shifts the focal plane axially,
allowing optical traps at different depths. The phase is computed from
the thin-lens quadratic phase profile.

Ported from ~/slm-code/SLMGeneration.py fresnel_lens_phase_generate().
"""

from __future__ import annotations

import numpy as np


def fresnel_lens_phase(
    slm_resolution: tuple[int, int],
    pixel_pitch_um: float,
    focal_length_um: float,
    wavelength_um: float,
    shift_distance_um: float,
    center: tuple[int, int] | None = None,
    magnification: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a Fresnel lens phase pattern for axial focal-plane shifting.

    Parameters
    ----------
    slm_resolution : (width, height) of the SLM in pixels.
    pixel_pitch_um : SLM pixel pitch in micrometres.
    focal_length_um : objective focal length in micrometres.
    wavelength_um : laser wavelength in micrometres.
    shift_distance_um : axial shift distance in micrometres.
    center : (x0, y0) lens centre in pixels. Defaults to SLM centre.
    magnification : optical system magnification.

    Returns
    -------
    screen : uint8 (height, width) phase screen ready for combine_screens().
    phase : float (height, width) continuous phase in [0, 2*pi).
    """
    slm_w, slm_h = slm_resolution
    if center is None:
        x0, y0 = slm_w // 2, slm_h // 2
    else:
        x0, y0 = center

    xs = np.arange(slm_w)
    ys = np.arange(slm_h)
    X, Y = np.meshgrid(xs, ys)

    dx = (X - x0) * pixel_pitch_um
    dy = (Y - y0) * pixel_pitch_um

    phase = np.mod(
        np.pi * (dx**2 + dy**2) * shift_distance_um
        / (wavelength_um * focal_length_um**2)
        * magnification**2,
        2 * np.pi,
    )

    screen = np.around(phase / (2 * np.pi) * 256).astype(np.uint8)
    return screen, phase


def combine_screens(*screens: np.ndarray) -> np.ndarray:
    """Combine multiple uint8 phase screens via modular addition.

    ``(screen1 + screen2 + ...) % 256``, preserving 2*pi phase wrapping.
    """
    result = np.zeros_like(screens[0], dtype=np.int32)
    for s in screens:
        result += s.astype(np.int32)
    return (result % 256).astype(np.uint8)
