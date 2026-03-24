"""Tests for Kim paper disordered array target."""

import numpy as np

from slm.targets import disordered_array


def test_disordered_array_count():
    target, positions = disordered_array(
        (256, 256),
        n_spots=50,
        extent=80,
        min_distance=3.0,
        rng=np.random.default_rng(42),
    )
    assert len(positions) == 50
    assert np.count_nonzero(target) == 50


def test_disordered_array_min_distance():
    _, positions = disordered_array(
        (256, 256),
        n_spots=30,
        extent=80,
        min_distance=5.0,
        rng=np.random.default_rng(42),
    )
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
            assert dist >= 5.0


def test_disordered_array_within_extent():
    center = (128, 128)
    _, positions = disordered_array(
        (256, 256),
        n_spots=30,
        extent=50,
        center=center,
        rng=np.random.default_rng(42),
    )
    for r, c in positions:
        dist = np.sqrt((r - center[0]) ** 2 + (c - center[1]) ** 2)
        assert dist <= 51  # +1 for integer rounding


def test_disordered_array_deterministic():
    _, pos1 = disordered_array(
        (256, 256),
        n_spots=20,
        extent=60,
        rng=np.random.default_rng(99),
    )
    _, pos2 = disordered_array(
        (256, 256),
        n_spots=20,
        extent=60,
        rng=np.random.default_rng(99),
    )
    np.testing.assert_array_equal(pos1, pos2)
