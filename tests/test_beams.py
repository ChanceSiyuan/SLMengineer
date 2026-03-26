"""Tests for beam profile generators."""

import numpy as np

from slm.beams import gaussian_beam, initial_slm_field, random_phase, uniform_beam


def test_gaussian_normalization(small_grid):
    amp = gaussian_beam(small_grid, sigma=10.0, normalize=True)
    np.testing.assert_allclose(np.sum(amp**2), 1.0, atol=1e-10)


def test_gaussian_peak_at_center(small_grid):
    amp = gaussian_beam(small_grid, sigma=10.0, normalize=False)
    peak = np.unravel_index(np.argmax(amp), amp.shape)
    assert peak == (31, 31) or peak == (32, 32)  # center pixel


def test_gaussian_symmetry(small_grid):
    amp = gaussian_beam(small_grid, sigma=10.0, normalize=False)
    # Should be symmetric about center
    np.testing.assert_allclose(amp, amp[::-1, :], atol=1e-10)
    np.testing.assert_allclose(amp, amp[:, ::-1], atol=1e-10)


def test_gaussian_custom_center():
    amp = gaussian_beam((64, 64), sigma=5.0, center=(10.0, 20.0), normalize=False)
    peak = np.unravel_index(np.argmax(amp), amp.shape)
    assert peak == (10, 20)


def test_uniform_beam_normalization(small_grid):
    amp = uniform_beam(small_grid)
    np.testing.assert_allclose(np.sum(amp**2), 1.0, atol=1e-10)


def test_random_phase_unit_magnitude(small_grid, rng):
    phasor = random_phase(small_grid, rng=rng)
    np.testing.assert_allclose(np.abs(phasor), 1.0, atol=1e-10)


def test_initial_slm_field_amplitude(small_grid, rng):
    field = initial_slm_field(small_grid, sigma=10.0, rng=rng)
    amp = np.abs(field)
    expected = gaussian_beam(small_grid, sigma=10.0, normalize=True)
    np.testing.assert_allclose(amp, expected, atol=1e-10)


def test_initial_slm_field_random_phase(rng):
    f1 = initial_slm_field((32, 32), sigma=5.0, rng=np.random.default_rng(1))
    f2 = initial_slm_field((32, 32), sigma=5.0, rng=np.random.default_rng(2))
    # Different seeds -> different phases
    assert not np.allclose(np.angle(f1), np.angle(f2))


def test_from_camera_intensity_normalization():
    from slm.beams import from_camera_intensity

    image = np.random.default_rng(0).uniform(0, 255, (32, 32))
    amp = from_camera_intensity(image, normalize=True)
    np.testing.assert_allclose(np.sum(amp**2), 1.0, atol=1e-10)


def test_from_camera_intensity_sqrt():
    from slm.beams import from_camera_intensity

    # Float input is treated as-is (not rescaled)
    image = np.array([[0.04, 0.09], [0.16, 0.25]])
    amp = from_camera_intensity(image, normalize=False)
    np.testing.assert_allclose(amp, np.sqrt(image))

    # Integer input is rescaled by dtype max
    image_u8 = np.array([[0, 128, 255]], dtype=np.uint8)
    amp_u8 = from_camera_intensity(image_u8, normalize=False)
    np.testing.assert_allclose(amp_u8, np.sqrt(image_u8.astype(float) / 255.0))
