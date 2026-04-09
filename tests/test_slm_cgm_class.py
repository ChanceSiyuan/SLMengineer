"""Tests for SLM_cgm_class (deprecated alias of SLM_class) and
CGM_phase_generate (torch wrapper mirroring WGS_phase_generate)."""

import numpy as np
import pytest

from slm.generation import SLM_class
from slm.targets import SLM_cgm_class

# SLM_cgm_class is deprecated but kept for back-compat; suppress the
# expected warning for every test in this file.
pytestmark = pytest.mark.filterwarnings(
    "ignore:SLM_cgm_class is deprecated:DeprecationWarning"
)


@pytest.fixture
def slm_cgm(slm_config, monkeypatch):
    """Create SLM_cgm_class instance with 64x64 temp config."""
    tmp_path, _ = slm_config
    monkeypatch.chdir(tmp_path)
    slm = SLM_cgm_class()
    slm.image_init(initGaussianPhase_user_defined=np.zeros((64, 64)), Plot=False)
    return slm


def test_slm_cgm_class_is_subclass_of_slm_class():
    """SLM_cgm_class inherits every method of SLM_class."""
    assert issubclass(SLM_cgm_class, SLM_class)


def test_slm_cgm_class_inherits_image_init(slm_cgm):
    """image_init sets initGaussianAmp correctly (inherited, real-valued)."""
    assert slm_cgm.initGaussianAmp.shape == (64, 64)
    assert slm_cgm.initGaussianAmp.dtype in (np.float32, np.float64)
    assert np.isclose(np.sum(slm_cgm.initGaussianAmp**2), 1.0)


def test_rec_lattice_returns_complex(slm_cgm):
    """SLM_cgm_class.RecLattice returns np.complex128 with 4 nonzero spots."""
    target = slm_cgm.RecLattice(distance=[0, 0], spacing=[150, 150], arraysize=[2, 2])
    assert target.dtype == np.complex128
    assert target.shape == (64, 64)
    assert np.count_nonzero(target) == 4


def test_rec_lattice_spot_amplitudes_are_real_positive(slm_cgm):
    """Spot-array targets have zero imaginary part (phase = 0)."""
    target = slm_cgm.RecLattice(distance=[0, 0], spacing=[150, 150], arraysize=[2, 2])
    nonzero_vals = target[target != 0]
    assert np.all(nonzero_vals.imag == 0)
    assert np.all(nonzero_vals.real > 0)


def test_rec_lattice_magnitude_matches_parent(slm_cgm, slm_config, monkeypatch):
    """|SLM_cgm_class.RecLattice| matches |SLM_class.RecLattice| bit-for-bit."""
    tmp_path, _ = slm_config
    monkeypatch.chdir(tmp_path)
    parent = SLM_class()
    parent.image_init(initGaussianPhase_user_defined=np.zeros((64, 64)), Plot=False)
    parent_target = parent.RecLattice(
        distance=[0, 0], spacing=[150, 150], arraysize=[2, 2]
    )
    cgm_target = slm_cgm.RecLattice(
        distance=[0, 0], spacing=[150, 150], arraysize=[2, 2]
    )
    np.testing.assert_array_equal(np.abs(cgm_target), np.abs(parent_target))


def test_translate_preserves_complex_dtype(slm_cgm):
    """translate_targetAmp returns complex when fed a complex target."""
    target = slm_cgm.RecLattice(distance=[0, 0], spacing=[150, 150], arraysize=[2, 2])
    translated = slm_cgm.translate_targetAmp(target)
    assert translated.dtype == np.complex128
    assert translated.shape == (64, 64)


def test_rotate_preserves_complex_dtype(slm_cgm):
    """rotate_targetAmp returns complex when fed a complex target."""
    target = slm_cgm.RecLattice(distance=[0, 0], spacing=[150, 150], arraysize=[2, 2])
    rotated = slm_cgm.rotate_targetAmp(target, angle=0.0)
    assert rotated.dtype == np.complex128


def test_phase_to_screen_inherited(slm_cgm):
    """phase_to_screen is inherited unchanged and produces uint8."""
    phase = np.random.uniform(-np.pi, np.pi, (64, 64))
    screen = slm_cgm.phase_to_screen(phase)
    assert screen.dtype == np.uint8


# ---------------------------------------------------------------------------
# Torch-based tests (WGS / CGM_phase_generate with complex targets)
# ---------------------------------------------------------------------------

try:
    import torch

    from slm.cgm import CGM_phase_generate
    from slm.wgs import WGS_phase_generate

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="PyTorch is required for these tests"
)


@requires_torch
def test_complex_target_accepted_by_wgs(slm_cgm):
    """WGS_phase_generate accepts a complex target (collapses to magnitude)."""
    target = slm_cgm.RecLattice(distance=[0, 0], spacing=[150, 150], arraysize=[2, 2])
    amp = torch.tensor(slm_cgm.initGaussianAmp, dtype=torch.float32)
    phase = torch.zeros_like(amp)
    tgt = torch.from_numpy(target)  # complex128 tensor
    assert torch.is_complex(tgt)
    out = WGS_phase_generate(amp, phase, tgt, Loop=5, Plot=False)
    assert out.shape == amp.shape
    assert not torch.is_complex(out)


@requires_torch
def test_wgs_complex_equivalent_to_real(slm_cgm, slm_config, monkeypatch):
    """WGS with complex target = WGS with |target| (phase is discarded)."""
    tmp_path, _ = slm_config
    monkeypatch.chdir(tmp_path)
    target_complex = slm_cgm.RecLattice(
        distance=[0, 0], spacing=[150, 150], arraysize=[2, 2]
    )
    target_real = np.abs(target_complex).astype(np.float64)
    amp = torch.tensor(slm_cgm.initGaussianAmp, dtype=torch.float64)
    phase = torch.zeros_like(amp)
    out_c = WGS_phase_generate(
        amp, phase, torch.from_numpy(target_complex), Loop=5, Plot=False
    )
    out_r = WGS_phase_generate(
        amp, phase, torch.from_numpy(target_real), Loop=5, Plot=False
    )
    torch.testing.assert_close(out_c.cpu(), out_r.cpu(), rtol=1e-5, atol=1e-5)


@requires_torch
def test_cgm_phase_generate_signature_mirrors_wgs(slm_cgm):
    """CGM_phase_generate accepts the same (amp, phase, target) pattern as WGS."""
    target = slm_cgm.RecLattice(distance=[0, 0], spacing=[150, 150], arraysize=[2, 2])
    amp = torch.tensor(slm_cgm.initGaussianAmp, dtype=torch.float64)
    phase = torch.zeros_like(amp)
    tgt = torch.from_numpy(target)
    out = CGM_phase_generate(amp, phase, tgt, max_iterations=10, steepness=4, Plot=False)
    assert out.shape == amp.shape
    assert not torch.is_complex(out)
    # Output is in [-pi, pi] after wrapping (CGM does not wrap, but magnitude bounded)
    assert torch.isfinite(out).all()


@requires_torch
def test_cgm_phase_generate_roundtrip_to_screen(slm_cgm):
    """CGM_phase_generate → phase_to_screen produces uint8 of correct shape."""
    target = slm_cgm.RecLattice(distance=[0, 0], spacing=[150, 150], arraysize=[2, 2])
    amp = torch.tensor(slm_cgm.initGaussianAmp, dtype=torch.float64)
    phase = torch.zeros_like(amp)
    tgt = torch.from_numpy(target)
    cgm_phase = CGM_phase_generate(
        amp, phase, tgt, max_iterations=10, steepness=4, Plot=False
    )
    phase_wrapped = torch.angle(torch.exp(1j * cgm_phase)).cpu().numpy()
    screen = slm_cgm.phase_to_screen(phase_wrapped)
    assert screen.dtype == np.uint8


@requires_torch
def test_cgm_phase_generate_accepts_real_target(slm_cgm):
    """CGM_phase_generate also accepts a raw real-valued numpy target
    (the function promotes it to complex internally)."""
    target = np.zeros((64, 64), dtype=np.float64)
    target[20, 20] = 0.5
    target[20, 44] = 0.5
    target[44, 20] = 0.5
    target[44, 44] = 0.5
    amp = torch.tensor(slm_cgm.initGaussianAmp, dtype=torch.float64)
    phase = torch.zeros_like(amp)
    tgt_tensor = torch.from_numpy(target)
    assert not torch.is_complex(tgt_tensor)
    out = CGM_phase_generate(
        amp, phase, tgt_tensor, max_iterations=10, steepness=4, Plot=False
    )
    assert out.shape == amp.shape
