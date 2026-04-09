import numpy as np

from slm.aberration import Zernike


def test_zernike_phase_shape():
    """phase_Zernike output matches (SLMResY, SLMResX) shape."""
    z = Zernike(
        SLMResX=64, SLMResY=64,
        pixelpitch=12.5, aperture_radius=400,
        ind_Zernike=4, percent=0.5,
    )
    screen, m, n = z.phase_Zernike(Plot=False)
    assert screen.shape == (64, 64)
    assert isinstance(m, (int, np.integer))
    assert isinstance(n, (int, np.integer))


def test_zernike_phase_finite():
    """Zernike phase values are finite."""
    z = Zernike(
        SLMResX=64, SLMResY=64,
        pixelpitch=12.5, aperture_radius=400,
        ind_Zernike=4, percent=0.5,
    )
    screen, _, _ = z.phase_Zernike(Plot=False)
    assert np.all(np.isfinite(screen))


def test_zernike_defocus_symmetry():
    """Defocus (ind=4) is rotationally symmetric -- check vertical/horizontal symmetry."""
    z = Zernike(
        SLMResX=64, SLMResY=64,
        pixelpitch=12.5, aperture_radius=400,
        ind_Zernike=4, percent=0.5,
    )
    screen, _, _ = z.phase_Zernike(Plot=False)
    center = 32
    # Defocus should be roughly symmetric along both axes
    top = screen[center - 5, center]
    bottom = screen[center + 5, center]
    left = screen[center, center - 5]
    right = screen[center, center + 5]
    assert abs(float(top) - float(bottom)) < 0.1
    assert abs(float(left) - float(right)) < 0.1


def test_zernike_zero_percent():
    """Zero percent Zernike gives zero phase everywhere."""
    z = Zernike(
        SLMResX=64, SLMResY=64,
        pixelpitch=12.5, aperture_radius=400,
        ind_Zernike=4, percent=0.0,
    )
    screen, _, _ = z.phase_Zernike(Plot=False)
    assert np.allclose(screen, 0.0)
