"""Tests for Bowman paper target patterns."""

import numpy as np

from slm.targets import (
    chicken_egg_pattern,
    gaussian_lattice,
    gaussian_line,
    graphene_lattice,
    lg_mode,
    ring_lattice_vortex,
    square_lattice_vortex,
)


class TestGaussianLattice:
    def test_shape_and_dtype(self):
        positions = np.array([[0, 0], [10, 0], [0, 10]])
        field = gaussian_lattice((64, 64), positions, peak_sigma=3.0)
        assert field.shape == (64, 64)
        assert field.dtype == np.complex128

    def test_peaks_at_positions(self):
        positions = np.array([[0, 0], [10, 0], [0, 10]])
        field = gaussian_lattice((64, 64), positions, peak_sigma=2.0)
        # Peak near center (0,0 offset)
        assert np.abs(field[32, 32]) > np.abs(field[0, 0])

    def test_normalization(self):
        positions = np.array([[0, 0], [5, 5]])
        field = gaussian_lattice((64, 64), positions, peak_sigma=3.0)
        np.testing.assert_allclose(np.sum(np.abs(field) ** 2), 1.0, atol=1e-10)

    def test_per_site_phases(self):
        positions = np.array([[0, 0], [15, 0]])
        phases = np.array([0.0, np.pi])
        field = gaussian_lattice((64, 64), positions, peak_sigma=2.0, phases=phases)
        # Two peaks should have opposite phase
        phase_a = np.angle(field[32, 32])
        phase_b = np.angle(field[46, 32])  # 31.5 + 15 ≈ 46
        diff = np.abs(np.angle(np.exp(1j * (phase_b - phase_a))))
        assert diff > 2.5  # close to pi


class TestSquareLatticeVortex:
    def test_shape(self):
        field = square_lattice_vortex(
            (128, 128), rows=4, cols=4, spacing=10, peak_sigma=2.0
        )
        assert field.shape == (128, 128)

    def test_normalization(self):
        field = square_lattice_vortex(
            (128, 128), rows=4, cols=4, spacing=10, peak_sigma=2.0
        )
        np.testing.assert_allclose(np.sum(np.abs(field) ** 2), 1.0, atol=1e-10)

    def test_intensity_symmetry(self):
        field = square_lattice_vortex(
            (128, 128), rows=4, cols=4, spacing=10, peak_sigma=2.0
        )
        intensity = np.abs(field) ** 2
        # Approximate 4-fold symmetry about center
        np.testing.assert_allclose(
            intensity, np.rot90(intensity), atol=np.max(intensity) * 0.2
        )


class TestRingLatticeVortex:
    def test_peaks_on_ring(self):
        field = ring_lattice_vortex(
            (128, 128), n_sites=8, ring_radius=20.0, peak_sigma=2.0
        )
        intensity = np.abs(field) ** 2
        # Maximum should be near the ring, not at center
        center_val = intensity[64, 64]
        ring_val = intensity[83, 64]  # 63.5 + 20 ≈ 83
        assert ring_val > center_val

    def test_normalization(self):
        field = ring_lattice_vortex(
            (128, 128), n_sites=8, ring_radius=20.0, peak_sigma=2.0
        )
        np.testing.assert_allclose(np.sum(np.abs(field) ** 2), 1.0, atol=1e-10)


class TestGrapheneLattice:
    def test_shape(self):
        field = graphene_lattice(
            (128, 128), rows=3, cols=3, spacing=8.0, peak_sigma=2.0
        )
        assert field.shape == (128, 128)

    def test_normalization(self):
        field = graphene_lattice(
            (128, 128), rows=3, cols=3, spacing=8.0, peak_sigma=2.0
        )
        np.testing.assert_allclose(np.sum(np.abs(field) ** 2), 1.0, atol=1e-10)

    def test_has_nonzero_content(self):
        field = graphene_lattice(
            (128, 128), rows=3, cols=3, spacing=8.0, peak_sigma=2.0
        )
        assert np.max(np.abs(field)) > 0


class TestGaussianLinePhaseGradient:
    def test_backward_compatible(self):
        """Default phase_gradient=0 should give real-valued target."""
        target = gaussian_line((64, 64), length=20.0, width_sigma=3.0)
        # With zero phase gradient, imaginary part should be negligible
        inside = np.abs(target) > np.max(np.abs(target)) * 0.1
        phases = np.angle(target[inside])
        assert np.std(phases) < 0.1

    def test_phase_ramp(self):
        """Non-zero phase_gradient should create a phase ramp along line."""
        target = gaussian_line(
            (64, 64), length=30.0, width_sigma=3.0, phase_gradient=0.5
        )
        # Sample phase along the center row (line direction)
        center = 32
        inside = np.abs(target[center, :]) > np.max(np.abs(target)) * 0.1
        phases = np.angle(target[center, inside])
        # Phase should generally increase
        if len(phases) > 5:
            diffs = np.diff(np.unwrap(phases))
            assert np.mean(diffs) > 0


class TestLgModeCenter:
    def test_default_center(self):
        """Without center param, should be centered on grid."""
        field = lg_mode((64, 64), ell=1, p=0, w0=10.0)
        intensity = np.abs(field) ** 2
        # Ring should be centered
        assert intensity[0, 0] < np.max(intensity) * 0.01

    def test_custom_center(self):
        """With center param, pattern should shift."""
        field_default = lg_mode((128, 128), ell=1, p=0, w0=10.0)
        field_shifted = lg_mode((128, 128), ell=1, p=0, w0=10.0, center=(80.0, 80.0))
        # Peak should be at different location
        peak_default = np.unravel_index(
            np.argmax(np.abs(field_default) ** 2), (128, 128)
        )
        peak_shifted = np.unravel_index(
            np.argmax(np.abs(field_shifted) ** 2), (128, 128)
        )
        assert peak_default != peak_shifted


class TestChickenEggPattern:
    def test_deterministic(self):
        f1 = chicken_egg_pattern((64, 64), rng=np.random.default_rng(42))
        f2 = chicken_egg_pattern((64, 64), rng=np.random.default_rng(42))
        np.testing.assert_array_equal(f1, f2)

    def test_normalization(self):
        field = chicken_egg_pattern((64, 64))
        np.testing.assert_allclose(np.sum(np.abs(field) ** 2), 1.0, atol=1e-10)

    def test_circular_boundary(self):
        field = chicken_egg_pattern((64, 64), radius=15.0)
        # Corners should be zero
        assert np.abs(field[0, 0]) == 0.0
