"""FFT/IFFT propagation between SLM plane and focal plane."""

from __future__ import annotations

import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift


def fft_propagate(field: np.ndarray) -> np.ndarray:
    """Forward propagation: SLM plane -> focal plane via FFT.

    Uses ortho normalization so that Parseval's theorem holds exactly:
    sum(|field|^2) == sum(|FFT(field)|^2).
    Applies fftshift so the zero-frequency component is centered.
    """
    return fftshift(fft2(ifftshift(field), norm="ortho"))


def ifft_propagate(field: np.ndarray) -> np.ndarray:
    """Backward propagation: focal plane -> SLM plane via IFFT."""
    return fftshift(ifft2(ifftshift(field), norm="ortho"))


# --- Pixelation effect modeling ---


def sinc_envelope(shape: tuple[int, int], fill_factor: float = 1.0) -> np.ndarray:
    """Compute 2D sinc envelope from finite pixel aperture.

    Each SLM pixel is a square aperture of width (fill_factor * pitch).
    In the focal plane this multiplies the field by sinc(u*ff) * sinc(v*ff),
    where u, v are normalised focal-plane coordinates.

    Parameters
    ----------
    shape : (ny, nx) of the focal-plane grid.
    fill_factor : active pixel width / pixel pitch (0-1). 1.0 = no dead space.

    Returns
    -------
    Real 2D array (field-level envelope; intensity envelope is sinc²).
    """
    ny, nx = shape
    iy = np.arange(ny) - ny // 2
    ix = np.arange(nx) - nx // 2
    # np.sinc(x) computes sin(pi*x) / (pi*x)
    return np.sinc(iy[:, None] * fill_factor / ny) * np.sinc(
        ix[None, :] * fill_factor / nx
    )


def zero_order_field(
    shape: tuple[int, int],
    fill_factor: float,
    input_power: float = 1.0,
) -> np.ndarray:
    """Compute the zero-order (DC) contribution from pixel dead space.

    Light hitting the inter-pixel gap is unmodulated and focuses to a
    bright spot at the optical axis.  Fraction of unmodulated power =
    1 - fill_factor².

    Returns a complex array with a single nonzero pixel at center.
    """
    field = np.zeros(shape, dtype=np.complex128)
    unmodulated_frac = max(0.0, 1.0 - fill_factor**2)
    if unmodulated_frac > 0:
        amp = np.sqrt(unmodulated_frac * input_power)
        cy, cx = shape[0] // 2, shape[1] // 2
        field[cy, cx] = amp
    return field


def realistic_propagate(
    field: np.ndarray,
    sinc_env: np.ndarray,
    zero_order_amp: float = 0.0,
) -> np.ndarray:
    """Forward propagation with pixelation: FFT -> sinc envelope -> zero-order.

    Parameters
    ----------
    field : complex SLM-plane field.
    sinc_env : precomputed sinc envelope (from :func:`sinc_envelope`).
    zero_order_amp : amplitude of the zero-order DC spike at centre.
    """
    E_out = fft_propagate(field) * sinc_env
    if zero_order_amp != 0.0:
        cy, cx = field.shape[0] // 2, field.shape[1] // 2
        E_out[cy, cx] += zero_order_amp
    return E_out


def realistic_ifft_propagate(
    field: np.ndarray,
    sinc_env: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """Inverse propagation with sinc pre-compensation.

    Divides out the sinc envelope (clamped to *eps*) before IFFT,
    so that GS/WGS iterations can compensate for the roll-off.
    """
    safe_env = np.where(np.abs(sinc_env) > eps, sinc_env, eps)
    return ifft_propagate(field / safe_env)


def pad_field(field: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Zero-pad a field to target_shape for increased focal-plane resolution.

    Padding is symmetric (centered). target_shape must be >= field.shape.
    """
    ny, nx = field.shape
    ty, tx = target_shape
    if ty < ny or tx < nx:
        raise ValueError(
            f"target_shape {target_shape} must be >= field shape {field.shape}"
        )
    if ty == ny and tx == nx:
        return field.copy()
    padded = np.zeros(target_shape, dtype=field.dtype)
    y0 = (ty - ny) // 2
    x0 = (tx - nx) // 2
    padded[y0 : y0 + ny, x0 : x0 + nx] = field
    return padded
