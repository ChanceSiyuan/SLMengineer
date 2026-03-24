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
