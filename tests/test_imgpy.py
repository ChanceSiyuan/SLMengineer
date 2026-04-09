import numpy as np

from slm.imgpy import IMG, WGS, SLM_screen_Correct


def test_wgs_fftloop_output_shapes(small_gaussian_amp, small_target_4spots):
    """fftLoop returns arrays with correct shapes."""
    phase = np.random.uniform(-np.pi, np.pi, (64, 64))
    wgs_obj = WGS(small_gaussian_amp, phase, small_target_4spots)
    slm_amp, slm_phase, fft_amp, non_uniform = wgs_obj.fftLoop(5, 0.01, Plot=False)
    assert slm_amp.shape == (64, 64)
    assert slm_phase.shape == (64, 64)
    assert fft_amp.shape == (64, 64)
    assert len(non_uniform) == 4  # Loop-1 because first is deleted


def test_wgs_fftloop_phase_range(small_gaussian_amp, small_target_4spots):
    """fftLoop phase output is in [-pi, pi]."""
    phase = np.zeros((64, 64))
    wgs_obj = WGS(small_gaussian_amp, phase, small_target_4spots)
    _, slm_phase, _, _ = wgs_obj.fftLoop(5, 0.01, Plot=False)
    assert slm_phase.min() >= -np.pi - 1e-6
    assert slm_phase.max() <= np.pi + 1e-6


def test_slm_screen_correct_output():
    """SLM_screen_Correct returns uint8 array."""
    screen = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    # No correction image -- function handles missing file gracefully
    result = SLM_screen_Correct(screen, LUT=207, correctionImgPath="nonexistent.bmp")
    assert result.dtype == np.uint8
    assert result.shape == (64, 64)


def test_img_init_shapes():
    """IMG.initSLMImage produces correct shape arrays."""
    img = IMG(
        pixelpitch=12.5,
        arraySizeBit=[6, 6],
        beamwaist=500,
        focallength=200000,
        magnification=1,
        wavelength=1.013,
        maskradius=5000,
        SLMRes=[64, 64],
    )
    amp, phase = img.initSLMImage(mask=False, Plot=False)
    assert amp.shape == (64, 64)
    assert phase.shape == (64, 64)


def test_img_init_amp_normalized():
    """initSLMImage amplitude is normalized (sum of squares ~1)."""
    img = IMG(
        pixelpitch=12.5,
        arraySizeBit=[6, 6],
        beamwaist=500,
        focallength=200000,
        magnification=1,
        wavelength=1.013,
        maskradius=5000,
        SLMRes=[64, 64],
    )
    amp, _ = img.initSLMImage(mask=False, Plot=False)
    assert abs(np.sum(amp**2) - 1.0) < 1e-6
