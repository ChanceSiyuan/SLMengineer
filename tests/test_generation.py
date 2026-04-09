import numpy as np
import pytest

from slm.generation import SLM_class


@pytest.fixture
def slm_instance(slm_config, monkeypatch):
    """Create SLM_class with temp config."""
    tmp_path, _ = slm_config
    monkeypatch.chdir(tmp_path)
    slm = SLM_class()
    slm.image_init(initGaussianPhase_user_defined=np.zeros((64, 64)), Plot=False)
    return slm


def test_slm_class_init(slm_config, monkeypatch):
    """SLM_class reads config and sets attributes."""
    tmp_path, config = slm_config
    monkeypatch.chdir(tmp_path)
    slm = SLM_class()
    assert slm.pixelpitch == config["pixelpitch"]
    assert slm.SLMRes == config["SLMRes"]
    assert slm.beamwaist == config["beamwaist"]


def test_image_init_shapes(slm_instance):
    """image_init creates Gaussian amp and phase with correct shapes."""
    assert slm_instance.initGaussianAmp.shape == (64, 64)
    assert slm_instance.initGaussianPhase.shape == (64, 64)


def test_target_generate_rec_spot_count(slm_instance):
    """2x2 rectangular target has 4 nonzero spots."""
    target = slm_instance.target_generate("Rec", arraysize=[2, 2], translate=False, Plot=False)
    assert target.shape == (64, 64)
    assert np.count_nonzero(target) == 4


def test_phase_to_screen_uint8(slm_instance):
    """phase_to_screen produces uint8 output."""
    phase = np.random.uniform(-np.pi, np.pi, (64, 64))
    screen = slm_instance.phase_to_screen(phase)
    assert screen.dtype == np.uint8


def test_fresnel_lens_shape(slm_instance):
    """fresnel_lens_phase_generate returns correct shape."""
    result = slm_instance.fresnel_lens_phase_generate(1000, 32, 32)
    assert isinstance(result, tuple)
    assert result[0].shape == (64, 64)
