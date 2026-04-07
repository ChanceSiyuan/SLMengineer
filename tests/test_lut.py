"""Tests for slm.hardware.lut."""

import numpy as np
import pytest

from slm.hardware.lut import apply_lut_correction


class TestApplyLutCorrection:
    def test_no_calibration(self):
        screen = np.full((10, 10), 128, dtype=np.uint8)
        result = apply_lut_correction(screen, lut_value=256)
        # 128/256 * 256 = 128
        assert result.dtype == np.uint8
        assert np.all(result == 128)

    def test_lut_scaling(self):
        screen = np.full((10, 10), 255, dtype=np.uint8)
        result = apply_lut_correction(screen, lut_value=128)
        # 255/256 * 128 ≈ 127
        expected = int(255 / 256 * 128)
        assert np.all(result == expected)

    def test_with_calibration(self):
        screen = np.full((4, 4), 100, dtype=np.uint8)
        calibration = np.full((4, 4), 50, dtype=np.uint8)
        result = apply_lut_correction(screen, lut_value=256, calibration=calibration)
        # (100 + 50) / 256 * 256 = 150
        expected = int(150 / 256 * 256)
        assert np.all(result == expected)

    def test_output_dtype(self):
        screen = np.zeros((5, 5), dtype=np.uint8)
        result = apply_lut_correction(screen, lut_value=207)
        assert result.dtype == np.uint8

    def test_wraparound_with_calibration(self):
        # uint16 addition: 200 + 200 = 400 (no uint8 overflow)
        screen = np.full((4, 4), 200, dtype=np.uint8)
        calibration = np.full((4, 4), 200, dtype=np.uint8)
        result = apply_lut_correction(screen, lut_value=207, calibration=calibration)
        # (400) / 256 * 207 ≈ 323 -> truncated to uint8 = 67
        expected = int(400 / 256 * 207)
        assert np.all(result == expected & 0xFF)
