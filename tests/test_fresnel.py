"""Tests for slm.hardware.fresnel."""

import numpy as np
import pytest

from slm.hardware.fresnel import combine_screens, fresnel_lens_phase


class TestFresnelLensPhase:
    def test_output_shapes(self):
        screen, phase = fresnel_lens_phase(
            slm_resolution=(100, 80),
            pixel_pitch_um=12.5,
            focal_length_um=200_000.0,
            wavelength_um=1.013,
            shift_distance_um=20_000,
        )
        assert screen.shape == (80, 100)
        assert phase.shape == (80, 100)
        assert screen.dtype == np.uint8
        assert phase.dtype == np.float64

    def test_phase_in_range(self):
        _, phase = fresnel_lens_phase(
            slm_resolution=(64, 64),
            pixel_pitch_um=12.5,
            focal_length_um=200_000.0,
            wavelength_um=1.013,
            shift_distance_um=10_000,
        )
        assert np.all(phase >= 0)
        assert np.all(phase < 2 * np.pi)

    def test_zero_shift_is_flat(self):
        screen, phase = fresnel_lens_phase(
            slm_resolution=(64, 64),
            pixel_pitch_um=12.5,
            focal_length_um=200_000.0,
            wavelength_um=1.013,
            shift_distance_um=0,
        )
        assert np.all(phase == 0)
        assert np.all(screen == 0)

    def test_radial_symmetry(self):
        # Use odd-sized grid so centre pixel is exactly at the integer centre
        screen, phase = fresnel_lens_phase(
            slm_resolution=(65, 65),
            pixel_pitch_um=12.5,
            focal_length_um=200_000.0,
            wavelength_um=1.013,
            shift_distance_um=50_000,
        )
        # Phase should be symmetric about the centre
        np.testing.assert_array_almost_equal(phase, np.fliplr(phase), decimal=10)
        np.testing.assert_array_almost_equal(phase, np.flipud(phase), decimal=10)

    def test_custom_centre(self):
        screen, _ = fresnel_lens_phase(
            slm_resolution=(100, 80),
            pixel_pitch_um=12.5,
            focal_length_um=200_000.0,
            wavelength_um=1.013,
            shift_distance_um=20_000,
            center=(25, 30),
        )
        assert screen.shape == (80, 100)


class TestCombineScreens:
    def test_identity(self):
        a = np.full((10, 10), 100, dtype=np.uint8)
        b = np.zeros((10, 10), dtype=np.uint8)
        result = combine_screens(a, b)
        np.testing.assert_array_equal(result, a)

    def test_modular_wrap(self):
        a = np.full((4, 4), 200, dtype=np.uint8)
        b = np.full((4, 4), 100, dtype=np.uint8)
        result = combine_screens(a, b)
        # (200 + 100) % 256 = 44
        assert np.all(result == 44)

    def test_three_screens(self):
        a = np.full((4, 4), 100, dtype=np.uint8)
        b = np.full((4, 4), 100, dtype=np.uint8)
        c = np.full((4, 4), 100, dtype=np.uint8)
        result = combine_screens(a, b, c)
        # (300) % 256 = 44
        assert np.all(result == 44)

    def test_output_dtype(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        result = combine_screens(a, a)
        assert result.dtype == np.uint8
