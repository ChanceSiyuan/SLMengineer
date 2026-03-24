"""Zernike polynomials and hologram geometric transformations."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter
from scipy.special import factorial

from slm.propagation import fft_propagate, ifft_propagate


def _noll_to_nm(j: int) -> tuple[int, int]:
    """Convert Noll index j to (n, m) Zernike indices."""
    if j < 1:
        raise ValueError(f"Noll index must be >= 1, got {j}")
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    m_options = list(range(-n, n + 1, 2))
    k = j - n * (n + 1) // 2 - 1
    # Noll ordering: even j -> positive m, odd j -> negative m
    m_sorted = sorted(m_options, key=lambda m: (abs(m), -1 if m < 0 else 1))
    m = m_sorted[k]
    return n, m


def _zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Compute Zernike radial polynomial R_n^|m|(rho)."""
    m_abs = abs(m)
    result = np.zeros_like(rho)
    for s in range((n - m_abs) // 2 + 1):
        coef = (
            (-1) ** s
            * factorial(n - s, exact=True)
            / (
                factorial(s, exact=True)
                * factorial((n + m_abs) // 2 - s, exact=True)
                * factorial((n - m_abs) // 2 - s, exact=True)
            )
        )
        result += coef * rho ** (n - 2 * s)
    return result


def zernike(
    n: int,
    m: int,
    shape: tuple[int, int],
    radius: float | None = None,
) -> np.ndarray:
    """Compute Zernike polynomial Z_n^m over a unit disk.

    Returns zero outside the unit disk (rho > 1).
    """
    ny, nx = shape
    if radius is None:
        radius = min(ny, nx) / 2.0
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    y = (np.arange(ny) - cy) / radius
    x = (np.arange(nx) - cx) / radius
    yy, xx = np.meshgrid(y, x, indexing="ij")
    rho = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)

    inside = rho <= 1.0
    R = np.zeros(shape)
    R[inside] = _zernike_radial(n, abs(m), rho[inside])

    if m > 0:
        Z = R * np.cos(m * theta)
    elif m < 0:
        Z = R * np.sin(abs(m) * theta)
    else:
        Z = R

    Z[~inside] = 0.0
    return Z


def zernike_from_noll(
    j: int,
    shape: tuple[int, int],
    radius: float | None = None,
) -> np.ndarray:
    """Compute Zernike polynomial by Noll index j.

    j=1: piston, j=2: x-tilt, j=3: y-tilt, j=4: defocus, j=5,6: astigmatism.
    """
    n, m = _noll_to_nm(j)
    return zernike(n, m, shape, radius)


def apply_zernike_correction(
    hologram_phase: np.ndarray,
    coefficients: dict[int, float],
    radius: float | None = None,
) -> np.ndarray:
    """Apply Zernike phase correction to a hologram.

    phi_aligned = phi_WGS + sum(a_j * Z_j)
    """
    return hologram_phase + generate_aberration(
        hologram_phase.shape, coefficients, radius
    )


def anti_aliased_affine_transform(
    hologram_phase: np.ndarray,
    rotation_angle: float = 0.0,
    stretch: tuple[float, float] = (1.0, 1.0),
    gaussian_sigma: float = 2.0,
) -> np.ndarray:
    """Anti-aliased affine transformation of a hologram.

    Algorithm from Manovitz et al.:
        1. E_img = FFT(exp(i * phi))
        2. E_img_smooth = convolve(E_img, Gaussian(sigma))
        3. E_img_transformed = affine_transform(E_img_smooth, rotation, stretch)
        4. phi' = angle(IFFT(E_img_transformed))
    """
    # Step 1: FFT to image plane
    E_slm = np.exp(1j * hologram_phase)
    E_img = fft_propagate(E_slm)

    # Step 2: Gaussian convolution (smooth sharp spots)
    E_img_smooth_real = gaussian_filter(np.real(E_img), sigma=gaussian_sigma)
    E_img_smooth_imag = gaussian_filter(np.imag(E_img), sigma=gaussian_sigma)
    E_img_smooth = E_img_smooth_real + 1j * E_img_smooth_imag

    # Step 3: Affine transformation
    cos_a = np.cos(rotation_angle)
    sin_a = np.sin(rotation_angle)
    sy, sx = stretch
    # Rotation then stretch matrix (inverse for scipy affine_transform)
    matrix = np.array(
        [
            [cos_a / sy, sin_a / sy],
            [-sin_a / sx, cos_a / sx],
        ]
    )
    center = np.array(hologram_phase.shape) / 2.0
    offset = center - matrix @ center

    E_transformed_real = affine_transform(
        np.real(E_img_smooth), matrix, offset=offset, order=3
    )
    E_transformed_imag = affine_transform(
        np.imag(E_img_smooth), matrix, offset=offset, order=3
    )
    E_transformed = E_transformed_real + 1j * E_transformed_imag

    # Step 4: IFFT back to SLM plane
    E_slm_new = ifft_propagate(E_transformed)
    return np.angle(E_slm_new)


def generate_aberration(
    shape: tuple[int, int],
    coefficients: dict[int, float],
    radius: float | None = None,
) -> np.ndarray:
    """Generate an aberration phase from Zernike coefficients.

    Useful for simulating optical imperfections in feedback loops.
    """
    aberration = np.zeros(shape, dtype=np.float64)
    for noll_j, coeff in coefficients.items():
        aberration += coeff * zernike_from_noll(noll_j, shape, radius)
    return aberration
