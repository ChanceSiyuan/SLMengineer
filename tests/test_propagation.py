"""Tests for FFT/IFFT propagation."""

import numpy as np
import pytest

from slm.propagation import fft_propagate, ifft_propagate, pad_field


def test_fft_ifft_roundtrip(rng):
    field = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
    recovered = ifft_propagate(fft_propagate(field))
    np.testing.assert_allclose(recovered, field, atol=1e-10)


def test_parseval_theorem(rng):
    field = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
    ft = fft_propagate(field)
    power_in = np.sum(np.abs(field) ** 2)
    power_out = np.sum(np.abs(ft) ** 2)
    np.testing.assert_allclose(power_out, power_in, rtol=1e-10)


def test_delta_function_fft():
    field = np.zeros((64, 64), dtype=complex)
    field[32, 32] = 1.0
    ft = fft_propagate(field)
    # Delta function -> uniform Fourier amplitude
    amps = np.abs(ft)
    np.testing.assert_allclose(amps, amps[0, 0], atol=1e-10)


def test_pad_field_shape():
    field = np.ones((32, 32), dtype=complex)
    padded = pad_field(field, (64, 64))
    assert padded.shape == (64, 64)
    # Original content centered
    assert padded[16, 16] == 1.0
    assert padded[0, 0] == 0.0


def test_pad_field_identity():
    field = np.ones((32, 32), dtype=complex)
    padded = pad_field(field, (32, 32))
    np.testing.assert_array_equal(padded, field)


def test_pad_field_invalid():
    field = np.ones((32, 32), dtype=complex)
    with pytest.raises(ValueError):
        pad_field(field, (16, 16))
