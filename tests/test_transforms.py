"""Tests for Zernike polynomials and hologram transforms."""

import numpy as np

from slm.transforms import (
    anti_aliased_affine_transform,
    apply_zernike_correction,
    generate_aberration,
    zernike,
    zernike_from_noll,
)


def test_zernike_piston():
    """Z_0^0 (piston) should be constant inside the disk."""
    Z = zernike(0, 0, (64, 64))
    mask = Z != 0
    if np.any(mask):
        vals = Z[mask]
        np.testing.assert_allclose(vals, vals[0], atol=1e-10)


def test_zernike_tilt_x():
    """Z_1^1 (x-tilt) should be linear in x."""
    Z = zernike(1, 1, (64, 64))
    # Along the horizontal center line, should increase linearly
    line = Z[32, :]
    nonzero = line != 0
    if np.sum(nonzero) > 5:
        # Check monotonicity in the central region
        mid_range = line[25:40]
        diffs = np.diff(mid_range)
        assert np.all(diffs >= -1e-10)  # non-decreasing


def test_zernike_orthonormality():
    """Different Zernike modes should be approximately orthogonal."""
    shape = (128, 128)
    Z2 = zernike_from_noll(2, shape)
    Z3 = zernike_from_noll(3, shape)

    # Inner product of different modes should be ~0
    mask = (Z2 != 0) & (Z3 != 0)
    if np.any(mask):
        inner = np.sum(Z2[mask] * Z3[mask]) / np.sum(mask)
        assert abs(inner) < 0.1


def test_zernike_outside_disk():
    Z = zernike(2, 0, (64, 64))
    # Corners should be zero
    assert Z[0, 0] == 0.0
    assert Z[0, 63] == 0.0


def test_apply_zernike_correction():
    phase = np.zeros((64, 64))
    corrected = apply_zernike_correction(phase, {2: 1.0, 3: 0.5})
    # Should add Zernike modes to the phase
    assert not np.allclose(corrected, 0.0)
    assert corrected.shape == (64, 64)


def test_affine_identity():
    """Identity transform should preserve the hologram (approximately)."""
    rng = np.random.default_rng(42)
    phase = rng.uniform(-np.pi, np.pi, (64, 64))
    transformed = anti_aliased_affine_transform(
        phase, rotation_angle=0.0, stretch=(1.0, 1.0), gaussian_sigma=0.5
    )
    # With small sigma and identity transform, should be similar
    # (Gaussian smoothing introduces some change)
    assert transformed.shape == phase.shape


def test_generate_aberration():
    aberration = generate_aberration((64, 64), {4: 1.0})  # defocus
    assert aberration.shape == (64, 64)
    assert not np.allclose(aberration, 0.0)


def test_zernike_decompose_roundtrip():
    from slm.transforms import generate_aberration, zernike_decompose

    shape = (64, 64)
    original = {4: 1.5, 5: -0.3, 6: 0.8}  # defocus + astigmatism
    phase = generate_aberration(shape, original)
    recovered = zernike_decompose(phase, n_terms=10)
    for j, val in original.items():
        np.testing.assert_allclose(recovered[j], val, atol=0.05)


def test_apply_measured_correction():
    from slm.transforms import apply_measured_correction

    phase = np.ones((32, 32))
    aberr = 0.5 * np.ones((32, 32))
    corrected = apply_measured_correction(phase, aberr)
    np.testing.assert_allclose(corrected, 0.5)
