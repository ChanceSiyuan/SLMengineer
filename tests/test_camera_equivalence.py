"""Verify phase quantization roundtrip accuracy.

phase_to_uint8 maps [-pi, pi] → 256 levels, giving a maximum phase
error of 2*pi/256 ≈ 0.0245 rad per pixel.
"""

from __future__ import annotations

import numpy as np

from slm.hardware.phase_convert import phase_to_uint8


class TestPhaseQuantizationError:
    """Verify the uint8 quantization introduces only small errors."""

    def test_uniform_phase_roundtrip(self):
        """Constant phase should survive quantization exactly."""
        phase = np.full((64, 64), 0.5)
        uint8_val = phase_to_uint8(phase)
        recovered = uint8_val.astype(np.float64) / 256.0 * (2 * np.pi) - np.pi
        np.testing.assert_allclose(recovered, phase, atol=2 * np.pi / 256)

    def test_random_phase_roundtrip_error(self):
        """Random phase roundtrip error bounded by 2*pi/256."""
        rng = np.random.default_rng(42)
        phase = rng.uniform(-np.pi, np.pi, (128, 128))
        uint8_val = phase_to_uint8(phase)
        recovered = uint8_val.astype(np.float64) / 256.0 * (2 * np.pi) - np.pi

        # Account for wrapping near ±pi
        error = np.angle(np.exp(1j * (recovered - phase)))
        max_err = 2 * np.pi / 256
        assert np.max(np.abs(error)) <= max_err + 1e-10
        # RMS should be well below max
        assert np.sqrt(np.mean(error**2)) < max_err
