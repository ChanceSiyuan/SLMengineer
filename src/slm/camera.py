"""Camera interfaces and phase retrieval for experimental feedback.

Provides a camera abstraction (simulated or real) and Takeda
Fourier-transform fringe analysis for extracting phase from
interference patterns.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift

from slm.propagation import fft_propagate, realistic_propagate


class CameraInterface(Protocol):
    """Abstract camera — implement for real hardware or simulation."""

    def capture_intensity(self, slm_phase: np.ndarray) -> np.ndarray:
        """Display *slm_phase* on the SLM and return focal-plane intensity."""
        ...

    def capture_fringes(self, slm_phase: np.ndarray) -> np.ndarray:
        """Display *slm_phase* and return interference fringe image."""
        ...


class SimulatedCamera:
    """Simulates focal-plane capture via FFT propagation.

    Parameters
    ----------
    input_amplitude : beam amplitude on the SLM (real, ny x nx).
    aberration : optional phase aberration added before propagation.
    noise_level : Gaussian noise as a fraction of mean signal.
    reference_amplitude : amplitude of the reference beam for fringe mode.
    carrier_freq : (fy, fx) carrier spatial frequency for fringes (cycles/px).
    """

    def __init__(
        self,
        input_amplitude: np.ndarray,
        aberration: np.ndarray | None = None,
        noise_level: float = 0.02,
        reference_amplitude: float = 1.0,
        carrier_freq: tuple[float, float] = (0.0, 0.05),
        rng: np.random.Generator | None = None,
        sinc_env: np.ndarray | None = None,
    ):
        self.input_amplitude = input_amplitude
        self.aberration = aberration
        self.noise_level = noise_level
        self.reference_amplitude = reference_amplitude
        self.carrier_freq = carrier_freq
        self.rng = rng or np.random.default_rng()
        self.sinc_env = sinc_env

    def _propagate(self, slm_phase: np.ndarray) -> np.ndarray:
        """Propagate SLM field to focal plane, with optional aberration."""
        total_phase = slm_phase
        if self.aberration is not None:
            total_phase = slm_phase + self.aberration
        E_in = self.input_amplitude * np.exp(1j * total_phase)
        if self.sinc_env is not None:
            return realistic_propagate(E_in, self.sinc_env)
        return fft_propagate(E_in)

    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        mean_signal = np.mean(image[image > 0]) if np.any(image > 0) else 1.0
        noise = self.rng.normal(0, self.noise_level * mean_signal, image.shape)
        return np.maximum(image + noise, 0.0)

    def capture_intensity(self, slm_phase: np.ndarray) -> np.ndarray:
        E_out = self._propagate(slm_phase)
        intensity = np.abs(E_out) ** 2
        return self._add_noise(intensity)

    def capture_fringes(self, slm_phase: np.ndarray) -> np.ndarray:
        """Simulate interference fringes: |E_signal + E_reference|^2.

        The reference beam is a tilted plane wave creating carrier fringes.
        """
        E_signal = self._propagate(slm_phase)
        ny, nx = E_signal.shape
        y = np.arange(ny).reshape(-1, 1)
        x = np.arange(nx).reshape(1, -1)
        fy, fx = self.carrier_freq
        E_ref = self.reference_amplitude * np.exp(
            2j * np.pi * (fy * y + fx * x),
        )
        fringe = np.abs(E_signal + E_ref) ** 2
        return self._add_noise(fringe)


def takeda_phase_retrieval(
    fringe_image: np.ndarray,
    carrier_freq: tuple[float, float] | None = None,
) -> np.ndarray:
    """Extract phase from an interference fringe pattern (Takeda et al. 1982).

    The fringe pattern is ``|E_signal + E_reference|^2`` where the reference
    beam creates carrier fringes at a known spatial frequency.

    Algorithm:
      1. FFT the fringe image
      2. Isolate the sideband at the carrier frequency
      3. Demodulate by multiplying with ``exp(-2i*pi*(fy*y + fx*x))``
      4. Return ``angle()`` of the demodulated field

    Parameters
    ----------
    fringe_image : (ny, nx) real-valued fringe pattern.
    carrier_freq : (fy, fx) carrier frequency in cycles/pixel.
        If None, the strongest off-centre peak in the FFT is auto-detected.

    Returns
    -------
    (ny, nx) wrapped phase in [-pi, pi].
    """
    ny, nx = fringe_image.shape
    F = fftshift(fft2(fringe_image))

    if carrier_freq is None:
        mag = np.abs(F)
        cy, cx = ny // 2, nx // 2
        r_suppress = max(3, min(ny, nx) // 20)
        mag[
            cy - r_suppress : cy + r_suppress + 1, cx - r_suppress : cx + r_suppress + 1
        ] = 0
        mag[:cy, :] = 0
        peak_idx = np.unravel_index(np.argmax(mag), mag.shape)
        fy_det = (peak_idx[0] - cy) / ny
        fx_det = (peak_idx[1] - cx) / nx
    else:
        fy_det, fx_det = carrier_freq

    # In the shifted FFT, the cross-term conj(E_sig)*E_ref sits at
    # pixel (cy + fy*ny, cx + fx*nx).  Isolate it with a Gaussian window.
    cy, cx = ny // 2, nx // 2
    peak_y = cy + fy_det * ny
    peak_x = cx + fx_det * nx
    yy, xx = np.mgrid[:ny, :nx]
    separation = max(abs(fy_det * ny), abs(fx_det * nx), 1.0)
    sigma_win = separation * 0.6
    window = np.exp(-((yy - peak_y) ** 2 + (xx - peak_x) ** 2) / (2 * sigma_win**2))
    F_filtered = F * window

    # IFFT back to spatial domain, then demodulate the carrier
    field = ifft2(ifftshift(F_filtered))
    y = np.arange(ny).reshape(-1, 1)
    x = np.arange(nx).reshape(1, -1)
    demod = np.exp(-2j * np.pi * (fy_det * y + fx_det * x))
    return np.angle(field * demod)
