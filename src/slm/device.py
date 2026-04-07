"""Physical SLM device specification and unit conversions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SLMDevice:
    """Physical parameters of an SLM + optical system.

    Parameters
    ----------
    pixel_pitch_um : SLM pixel pitch in micrometres (e.g. 24.0 for BNS P1920).
    n_pixels : (rows, cols) of the active SLM area.
    wavelength_nm : laser wavelength in nanometres.
    focal_length_mm : focal length of the Fourier lens in millimetres.
    diffraction_efficiency : first-order diffraction efficiency (0-1).
    """

    pixel_pitch_um: float = 24.0
    n_pixels: tuple[int, int] = (256, 256)
    wavelength_nm: float = 1070.0
    focal_length_mm: float = 150.0
    diffraction_efficiency: float = 1.0
    fill_factor: float = 1.0  # active pixel width / pixel pitch (0-1)

    @property
    def pixel_pitch_mm(self) -> float:
        return self.pixel_pitch_um / 1e3

    def mm_to_slm_px(self, mm: float) -> float:
        """Convert a physical dimension (mm) on the SLM to pixels."""
        return mm / self.pixel_pitch_mm

    def focal_plane_pitch_um(self, n_pad: int | None = None) -> float:
        """Pixel pitch in the focal (output) plane in micrometres.

        pitch_focal = (wavelength * focal_length) / (N_pad * pixel_pitch)
        """
        n = n_pad if n_pad is not None else max(self.n_pixels)
        return (self.wavelength_nm * 1e-3 * self.focal_length_mm) / (
            n * self.pixel_pitch_um
        )

    def um_to_focal_px(self, um: float, n_pad: int | None = None) -> float:
        """Convert micrometres in the focal plane to focal-plane pixels."""
        pitch = self.focal_plane_pitch_um(n_pad)
        return um / pitch if pitch > 0 else 0.0

    def padded_shape(self, pad_factor: int = 2) -> tuple[int, int]:
        """Zero-padded output grid shape."""
        return (self.n_pixels[0] * pad_factor, self.n_pixels[1] * pad_factor)
