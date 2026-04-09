import numpy as np
import pytest

torch = pytest.importorskip("torch")

from slm.wgs import (
    WGS_phase_generate,
    fresnel_lens_phase_generate,
    nonUniformity_adapt,
    phase_to_screen,
)


def test_wgs_phase_shape(small_gaussian_amp, small_target_4spots):
    """Output phase shape matches input."""
    amp = torch.tensor(small_gaussian_amp, dtype=torch.float32)
    phase = torch.zeros_like(amp)
    target = torch.tensor(small_target_4spots, dtype=torch.float32)
    result = WGS_phase_generate(amp, phase, target, Loop=3, Plot=False)
    assert result.shape == amp.shape


def test_wgs_phase_range(small_gaussian_amp, small_target_4spots):
    """Output phase is in [-pi, pi]."""
    amp = torch.tensor(small_gaussian_amp, dtype=torch.float32)
    phase = torch.zeros_like(amp)
    target = torch.tensor(small_target_4spots, dtype=torch.float32)
    result = WGS_phase_generate(amp, phase, target, Loop=5, Plot=False)
    assert result.min() >= -np.pi - 1e-6
    assert result.max() <= np.pi + 1e-6


def test_wgs_deterministic_with_zero_init(small_gaussian_amp, small_target_4spots):
    """Same zero-phase input gives same output."""
    amp = torch.tensor(small_gaussian_amp, dtype=torch.float32)
    target = torch.tensor(small_target_4spots, dtype=torch.float32)
    r1 = WGS_phase_generate(amp, torch.zeros_like(amp), target, Loop=5, Plot=False)
    r2 = WGS_phase_generate(amp, torch.zeros_like(amp), target, Loop=5, Plot=False)
    assert torch.allclose(r1.cpu(), r2.cpu(), atol=1e-5)


def test_phase_to_screen_output():
    """phase_to_screen produces uint8 in [0, 255] with correct shape."""
    phase = np.random.uniform(-np.pi, np.pi, (4096, 4096)).astype(np.float32)
    screen = phase_to_screen(phase)
    assert screen.dtype == np.uint8
    assert screen.shape == (1024, 1024)
    assert screen.min() >= 0
    assert screen.max() <= 255


def test_fresnel_lens_shape():
    """Fresnel lens output has correct shape."""
    result = fresnel_lens_phase_generate(1000, SLMRes=(64, 64), x0=32, y0=32)
    assert result.shape == (64, 64)


def test_fresnel_lens_zero_shift():
    """Zero shift distance produces zero phase."""
    result = fresnel_lens_phase_generate(0, SLMRes=(64, 64), x0=32, y0=32)
    assert torch.allclose(result.cpu(), torch.zeros(64, 64), atol=1e-6)


def test_nonuniformity_perfect():
    """Equal amplitudes at all foci gives zero non-uniformity."""
    target = torch.zeros(8, 8)
    target[2, 2] = 1.0
    target[2, 6] = 1.0
    target[6, 2] = 1.0
    target[6, 6] = 1.0
    target = target / torch.sqrt(torch.sum(target**2))
    totalsites = torch.tensor(4)
    # If foci amplitudes exactly match target, non-uniformity should be ~0
    result = nonUniformity_adapt(target, target, totalsites)
    assert result.item() < 1e-5
