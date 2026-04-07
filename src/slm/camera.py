"""Camera interfaces and phase retrieval for experimental feedback.

Provides a camera abstraction (simulated or real) and Takeda
Fourier-transform fringe analysis for extracting phase from
interference patterns.

Camera implementations:
  - SimulatedCamera: FFT-based simulation for testing.
  - HardwareCamera: Real SLM display + physical camera capture.
"""

from __future__ import annotations

import time
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


class HardwareCamera:
    """Real-hardware camera: displays phase on SLM, captures via physical camera.

    Wraps the full display-and-capture pipeline behind :class:`CameraInterface`
    so that :func:`~slm.feedback.experimental_feedback_loop` and any other
    code expecting a camera object works transparently with real hardware.

    The ``capture_intensity(slm_phase)`` method:
      1. Converts ``slm_phase`` (float64, [-pi, pi]) to a uint8 screen.
      2. Optionally composites a Fresnel lens via modular addition.
      3. Applies LUT / manufacturer calibration correction.
      4. Displays on the SLM and waits for the panel to settle.
      5. Captures a frame from the physical camera.
      6. Returns the image as a float64 intensity array.

    Parameters
    ----------
    slm_display : object with ``update_array(array)`` method
        (e.g. :class:`~slm.hardware.slm_display.SLMDisplay`).
    camera : object with ``capture(exposure_time_us)`` method
        (e.g. :class:`~slm.hardware.vimba_camera.VimbaCamera`).
    slm_resolution : (width, height) of the physical SLM in pixels.
    exposure_time_us : camera exposure time in microseconds.
    settle_time_s : seconds to wait after displaying before capturing.
    lut_value : LUT scaling factor for phase correction (0-255).
    calibration : per-pixel correction array (uint8) from manufacturer BMP.
    fresnel_screen : optional precomputed Fresnel lens screen (uint8).
    """

    def __init__(
        self,
        slm_display,
        camera,
        slm_resolution: tuple[int, int] = (1272, 1024),
        exposure_time_us: float = 30.0,
        settle_time_s: float = 0.35,
        lut_value: int = 207,
        calibration: np.ndarray | None = None,
        fresnel_screen: np.ndarray | None = None,
    ) -> None:
        self._display = slm_display
        self._camera = camera
        self._slm_resolution = slm_resolution
        self._exposure_time_us = exposure_time_us
        self._settle_time_s = settle_time_s
        self._lut_value = lut_value
        self._calibration = calibration
        self._fresnel_screen = fresnel_screen

    @classmethod
    def from_config(cls, slm_display, camera, config) -> HardwareCamera:
        """Construct from a :class:`~slm.hardware.config.HardwareConfig`.

        Automatically loads the calibration BMP if ``config.calibration_bmp_path``
        is set.
        """
        from slm.hardware.lut import load_calibration_bmp

        calibration = None
        if config.calibration_bmp_path:
            calibration = load_calibration_bmp(config.calibration_bmp_path)

        return cls(
            slm_display=slm_display,
            camera=camera,
            slm_resolution=config.slm_resolution,
            exposure_time_us=config.exposure_time_us,
            settle_time_s=config.display_settle_time_s,
            lut_value=config.lut_value,
            calibration=calibration,
        )

    def _prepare_screen(self, slm_phase: np.ndarray) -> np.ndarray:
        """Convert algorithm phase to a corrected uint8 screen."""
        from slm.hardware.fresnel import combine_screens
        from slm.hardware.lut import apply_lut_correction
        from slm.hardware.phase_convert import phase_to_screen

        screen = phase_to_screen(slm_phase, self._slm_resolution)

        if self._fresnel_screen is not None:
            screen = combine_screens(screen, self._fresnel_screen)

        screen = apply_lut_correction(
            screen, self._lut_value, self._calibration
        )
        return screen

    def capture_intensity(self, slm_phase: np.ndarray) -> np.ndarray:
        """Display *slm_phase* on the SLM and return focal-plane intensity."""
        screen = self._prepare_screen(slm_phase)
        self._display.update_array(screen)
        time.sleep(self._settle_time_s)
        raw = self._camera.capture(self._exposure_time_us)
        return raw.astype(np.float64)

    def capture_fringes(self, slm_phase: np.ndarray) -> np.ndarray:
        """Display *slm_phase* and return interference fringe image.

        Requires a physical reference beam. Without one, this returns
        the same as ``capture_intensity`` (direct intensity image).
        """
        return self.capture_intensity(slm_phase)


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
