"""Shared fixtures for SLM tests."""

import json

import numpy as np
import pytest

from slm.beams import initial_slm_field
from slm.targets import mask_from_target, rectangular_grid, top_hat


@pytest.fixture
def small_grid():
    return (64, 64)


@pytest.fixture
def medium_grid():
    return (128, 128)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def gaussian_field_64(small_grid, rng):
    return initial_slm_field(small_grid, sigma=10.0, rng=rng)


@pytest.fixture
def gaussian_field_128(medium_grid, rng):
    return initial_slm_field(medium_grid, sigma=20.0, rng=rng)


@pytest.fixture
def four_spot_target_64(small_grid):
    positions = np.array([[24, 24], [24, 40], [40, 24], [40, 40]])
    target = np.zeros(small_grid, dtype=np.complex128)
    for r, c in positions:
        target[r, c] = 1.0
    return target, positions


@pytest.fixture
def grid_target_128(medium_grid):
    target, positions = rectangular_grid(medium_grid, rows=4, cols=4, spacing=10)
    mask = mask_from_target(target)
    return target, positions, mask


@pytest.fixture
def tophat_target_64(small_grid):
    return top_hat(small_grid, radius=10.0)


# ---------------------------------------------------------------------------
# Fixtures from ~/slm-code (for PyTorch WGS and generation tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def small_gaussian_amp():
    """64x64 Gaussian amplitude for fast tests."""
    size = 64
    x = np.linspace(-size // 2, size // 2, size) * 12.5  # pixelpitch=12.5
    X, Y = np.meshgrid(x, x)
    beamwaist = 500
    amp = np.sqrt(2 / np.pi) / beamwaist * np.exp(-(X**2 + Y**2) / beamwaist**2)
    amp = amp / np.sqrt(np.sum(amp**2))
    return amp


@pytest.fixture
def small_target_4spots():
    """64x64 target with 4 spots in a 2x2 grid."""
    target = np.zeros((64, 64))
    target[20, 20] = 1.0
    target[20, 44] = 1.0
    target[44, 20] = 1.0
    target[44, 44] = 1.0
    target = target / np.sqrt(np.sum(target**2))
    return target


@pytest.fixture
def slm_config(tmp_path):
    """Create a temporary config file for SLM_class tests."""
    config = {
        "pixelpitch": 12.5,
        "SLMRes": [64, 64],
        "arraySizeBit": [6, 6],
        "Loop": 5,
        "threshold": 0.01,
        "beamwaist": 500,
        "focallength": 200000,
        "magnification": 1,
        "wavelength": 1.013,
        "mask": False,
        "maskradius": 5000,
        "distance": [0, 0],
        "spacing": [150, 150],
        "arraysize": [2, 2],
        "translate": False,
        "rotate": False,
        "angle": 0,
        "modify": False,
        "AddZernike": False,
        "ind_Zernike_list": [3],
        "percent_list": [0.1],
        "zernike_aperture_radius": 6000,
        "isZernikePhaseContinous": False,
    }
    config_file = tmp_path / "hamamatsu_test_config.json"
    config_file.write_text(json.dumps(config))
    return tmp_path, config
