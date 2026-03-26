"""Tests for SLMDevice class."""

import numpy as np

from slm.device import SLMDevice


def test_pixel_pitch_conversion():
    dev = SLMDevice(pixel_pitch_um=24.0)
    assert dev.pixel_pitch_mm == 0.024


def test_mm_to_slm_px():
    dev = SLMDevice(pixel_pitch_um=24.0)
    assert dev.mm_to_slm_px(1.0) == 1.0 / 0.024


def test_focal_plane_pitch():
    dev = SLMDevice(pixel_pitch_um=24.0, wavelength_nm=1070.0, focal_length_mm=150.0)
    pitch = dev.focal_plane_pitch_um(n_pad=512)
    expected = (1070e-3 * 150.0) / (512 * 24.0)
    np.testing.assert_allclose(pitch, expected)


def test_um_to_focal_px():
    dev = SLMDevice(pixel_pitch_um=24.0, wavelength_nm=1070.0, focal_length_mm=150.0)
    px = dev.um_to_focal_px(105.0, n_pad=512)
    assert px > 0


def test_padded_shape():
    dev = SLMDevice(n_pixels=(256, 256))
    assert dev.padded_shape(2) == (512, 512)
    assert dev.padded_shape(4) == (1024, 1024)
