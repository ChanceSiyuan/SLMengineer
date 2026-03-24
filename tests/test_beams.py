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
