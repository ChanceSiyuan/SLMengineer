"""Tests for quality metrics."""

import numpy as np

from slm.metrics import (
    efficiency,
    fidelity,
    modulation_efficiency,
    non_uniformity_error,
    phase_error,
    uniformity,
)


def test_uniformity_perfect():
    intensities = np.array([1.0, 1.0, 1.0, 1.0])
    assert uniformity(intensities) == 0.0


def test_uniformity_known_value():
    intensities = np.array([1.0, 2.0, 3.0, 4.0])
    expected = np.std(intensities) / np.mean(intensities)
    np.testing.assert_allclose(uniformity(intensities), expected)


def test_efficiency_all_inside():
    field = np.ones((8, 8), dtype=complex)
    mask = np.ones((8, 8))
    assert efficiency(field, mask) == pytest.approx(1.0)


def test_efficiency_partial():
    field = np.ones((8, 8), dtype=complex)
    mask = np.zeros((8, 8))
    mask[:4, :4] = 1.0
    eff = efficiency(field, mask)
    assert eff == pytest.approx(0.25, rel=1e-10)


def test_modulation_efficiency():
    field = np.zeros((8, 8), dtype=complex)
    field[2, 2] = 1.0
    field[5, 5] = 1.0
    positions = np.array([[2, 2], [5, 5]])
    assert modulation_efficiency(field, positions) == pytest.approx(1.0)


def test_fidelity_identical_fields():
    field = np.random.default_rng(42).standard_normal((16, 16)) + 0j
    field += 1j * np.random.default_rng(43).standard_normal((16, 16))
    f = fidelity(field, field)
    assert f == pytest.approx(1.0, abs=1e-10)


def test_fidelity_orthogonal_fields():
    f1 = np.zeros((8, 8), dtype=complex)
    f2 = np.zeros((8, 8), dtype=complex)
    f1[0, 0] = 1.0
    f2[4, 4] = 1.0
    f = fidelity(f1, f2)
    assert f == pytest.approx(0.0, abs=1e-10)


def test_phase_error_zero():
    phase = np.ones((8, 8)) * 0.5
    region = np.ones((8, 8))
    pe = phase_error(phase, phase, region)
    assert pe == pytest.approx(0.0, abs=1e-10)


def test_non_uniformity_error_perfect():
    I_out = np.ones((8, 8))
    T = np.ones((8, 8))
    mask = np.ones((8, 8))
    nu = non_uniformity_error(I_out, T, mask)
    assert nu == pytest.approx(0.0, abs=1e-10)


import pytest
