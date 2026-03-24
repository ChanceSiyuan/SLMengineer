"""Shared fixtures for SLM tests."""

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
