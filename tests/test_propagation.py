"""Tests for FFT/IFFT propagation."""

import numpy as np
import pytest

from slm.propagation import (
    fft_propagate,
    ifft_propagate,
    pad_field,
    realistic_ifft_propagate,
    realistic_propagate,
    sinc_envelope,
    zero_order_field,
)


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


# --- Pixelation effect tests ---


class TestSincEnvelope:
    def test_shape(self):
        env = sinc_envelope((64, 128), fill_factor=0.9)
        assert env.shape == (64, 128)

    def test_center_is_unity(self):
        env = sinc_envelope((64, 64), fill_factor=0.9)
        assert env[32, 32] == pytest.approx(1.0, abs=1e-12)

    def test_symmetry(self):
        # Use odd size for exact flip symmetry
        env = sinc_envelope((65, 65), fill_factor=0.8)
        np.testing.assert_allclose(env, env[::-1, :], atol=1e-12)
        np.testing.assert_allclose(env, env[:, ::-1], atol=1e-12)

    def test_fill_factor_one_still_has_rolloff(self):
        env = sinc_envelope((64, 64), fill_factor=1.0)
        assert env[32, 32] == pytest.approx(1.0, abs=1e-12)
        # Edge should be attenuated (sinc(31/64) < 1)
        assert env[1, 32] < 1.0

    def test_lower_fill_factor_wider_envelope(self):
        env_high = sinc_envelope((64, 64), fill_factor=1.0)
        env_low = sinc_envelope((64, 64), fill_factor=0.5)
        # Lower fill factor => wider sinc main lobe => less attenuation at edges
        assert env_low[1, 32] > env_high[1, 32]


class TestZeroOrderField:
    def test_shape_and_center(self):
        field = zero_order_field((64, 64), fill_factor=0.9, input_power=100.0)
        assert field.shape == (64, 64)
        # Only center pixel is nonzero
        assert field[32, 32] != 0.0
        field[32, 32] = 0.0
        np.testing.assert_array_equal(field, 0.0)

    def test_amplitude(self):
        ff = 0.9
        power = 100.0
        field = zero_order_field((64, 64), fill_factor=ff, input_power=power)
        expected_amp = np.sqrt((1 - ff**2) * power)
        assert abs(field[32, 32]) == pytest.approx(expected_amp, rel=1e-10)

    def test_full_fill_factor_zero(self):
        field = zero_order_field((64, 64), fill_factor=1.0)
        np.testing.assert_array_equal(field, 0.0)


class TestRealisticPropagate:
    def test_reduces_to_fft_with_ones_envelope(self, rng):
        field = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
        env = np.ones((64, 64))
        result = realistic_propagate(field, env)
        expected = fft_propagate(field)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_sinc_attenuates_edges(self, rng):
        field = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
        env = sinc_envelope((64, 64), fill_factor=0.9)
        ideal = fft_propagate(field)
        real = realistic_propagate(field, env)
        # Center should be approximately same (sinc~1 near center)
        assert abs(real[32, 32]) == pytest.approx(abs(ideal[32, 32]), rel=1e-10)
        # Corner intensity should be reduced
        corner_ideal = np.abs(ideal[0, 0]) ** 2
        corner_real = np.abs(real[0, 0]) ** 2
        assert corner_real < corner_ideal

    def test_parseval_violated_with_sinc(self, rng):
        field = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
        env = sinc_envelope((64, 64), fill_factor=0.8)
        result = realistic_propagate(field, env)
        power_in = np.sum(np.abs(field) ** 2)
        power_out = np.sum(np.abs(result) ** 2)
        # Sinc attenuates, so output power < input (physically correct)
        assert power_out < power_in

    def test_roundtrip_approximate(self, rng):
        field = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
        env = sinc_envelope((64, 64), fill_factor=0.9)
        fwd = realistic_propagate(field, env)
        recovered = realistic_ifft_propagate(fwd, env)
        # Not exact due to epsilon clamping at sinc nulls, but close
        np.testing.assert_allclose(recovered, field, atol=0.1)
