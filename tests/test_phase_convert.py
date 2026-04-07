"""Tests for slm.hardware.phase_convert."""

import numpy as np
import pytest

from slm.hardware.phase_convert import crop_to_slm, phase_to_screen, phase_to_uint8


class TestPhaseToUint8:
    def test_minus_pi_maps_to_zero(self):
        phase = np.full((4, 4), -np.pi)
        result = phase_to_uint8(phase)
        assert result.dtype == np.uint8
        assert np.all(result == 0)

    def test_pi_maps_to_zero_wrapped(self):
        # +pi wraps around: (pi+pi)/(2pi)*256 = 256 -> wraps to 0 in uint8
        phase = np.full((4, 4), np.pi)
        result = phase_to_uint8(phase)
        assert np.all(result == 0)

    def test_zero_maps_to_128(self):
        phase = np.zeros((4, 4))
        result = phase_to_uint8(phase)
        assert np.all(result == 128)

    def test_monotonic_mapping(self):
        # Exclude endpoints near wrapping boundary
        phases = np.linspace(-np.pi + 0.1, np.pi - 0.1, 200)
        result = phase_to_uint8(phases)
        # Should be monotonically non-decreasing (no wrap in this range)
        assert np.all(np.diff(result.astype(np.int16)) >= 0)


class TestCropToSlm:
    def test_square_crop(self):
        phase = np.random.default_rng(42).standard_normal((512, 512))
        result = crop_to_slm(phase, slm_resolution=(256, 256))
        assert result.shape == (256, 256)

    def test_rectangular_slm(self):
        phase = np.random.default_rng(42).standard_normal((1024, 1024))
        # SLM is 1272x1024 -> min dimension is 1024
        result = crop_to_slm(phase, slm_resolution=(1272, 1024))
        assert result.shape == (1024, 1024)

    def test_center_preserved(self):
        phase = np.zeros((100, 100))
        phase[50, 50] = 1.0
        result = crop_to_slm(phase, slm_resolution=(20, 20))
        assert result.shape == (20, 20)
        assert result[10, 10] == 1.0

    def test_no_crop_when_same_size(self):
        phase = np.ones((256, 256))
        result = crop_to_slm(phase, slm_resolution=(256, 256))
        assert result.shape == (256, 256)
        np.testing.assert_array_equal(result, phase)


class TestPhaseToScreen:
    def test_output_shape_matches_slm(self):
        phase = np.zeros((512, 512))
        screen = phase_to_screen(phase, slm_resolution=(1272, 1024))
        assert screen.shape == (1024, 1272)
        assert screen.dtype == np.uint8

    def test_zero_phase_centered(self):
        # Zero phase -> 128 in the active area, 0 in the border
        phase = np.zeros((256, 256))
        screen = phase_to_screen(phase, slm_resolution=(300, 300))
        # Borders should be zero
        assert screen[0, 0] == 0
        # Centre should be 128
        assert screen[150, 150] == 128

    def test_small_phase_embedded(self):
        phase = np.zeros((64, 64))
        screen = phase_to_screen(phase, slm_resolution=(100, 80))
        assert screen.shape == (80, 100)
        assert screen.dtype == np.uint8
