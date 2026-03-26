"""Input beam profile generators."""

from __future__ import annotations

import numpy as np


def gaussian_beam(
    shape: tuple[int, int],
    sigma: float,
    center: tuple[float, float] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """2D Gaussian amplitude profile.

    Parameters
    ----------
    shape : (ny, nx) pixel grid dimensions.
    sigma : 1/e^2 radius in pixels.
    center : beam center in pixels (row, col); defaults to grid center.
    normalize : if True, normalize so sum(|amp|^2) = 1.

    Returns
    -------
    Real-valued amplitude array of shape (ny, nx).
    """
    ny, nx = shape
    if center is None:
        center = ((ny - 1) / 2.0, (nx - 1) / 2.0)
    y = np.arange(ny) - center[0]
    x = np.arange(nx) - center[1]
    yy, xx = np.meshgrid(y, x, indexing="ij")
    amp = np.exp(-(xx**2 + yy**2) / (sigma**2))
    if normalize:
        power = np.sum(amp**2)
        if power > 0:
            amp /= np.sqrt(power)
    return amp


def uniform_beam(shape: tuple[int, int]) -> np.ndarray:
    """Uniform amplitude profile (all ones, normalized)."""
    amp = np.ones(shape, dtype=np.float64)
    amp /= np.sqrt(np.sum(amp**2))
    return amp


def random_phase(
    shape: tuple[int, int],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Uniform random phase in [-pi, pi).

    Returns complex phasor exp(i * phase).
    """
    if rng is None:
        rng = np.random.default_rng()
    phase = rng.uniform(-np.pi, np.pi, size=shape)
    return np.exp(1j * phase)


def initial_slm_field(
    shape: tuple[int, int],
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Gaussian amplitude with random phase -- standard WGS initial field L_0.

    Returns complex field: gaussian_beam * exp(i * random_phase).
    """
    amp = gaussian_beam(shape, sigma, normalize=True)
    phasor = random_phase(shape, rng=rng)
    return amp * phasor
