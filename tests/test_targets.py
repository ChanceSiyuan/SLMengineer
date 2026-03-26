"""Tests for target pattern generators."""

import numpy as np

from slm.targets import (
    gaussian_line,
    hexagonal_grid,
    lg_mode,
    light_sheet,
    mask_from_target,
    measure_region,
    rectangular_grid,
    spot_array,
    top_hat,
)


def test_spot_array_positions(small_grid):
    positions = np.array([[10, 10], [20, 20], [30, 30]])
    target = spot_array(small_grid, positions)
    for r, c in positions:
        assert np.abs(target[r, c]) > 0
    # Total non-zero count
    assert np.count_nonzero(target) == 3


def test_rectangular_grid_count(small_grid):
    target, positions = rectangular_grid(small_grid, rows=3, cols=4, spacing=8)
    assert len(positions) == 12
    assert np.count_nonzero(target) == 12


def test_hexagonal_grid_geometry(small_grid):
    target, positions = hexagonal_grid(small_grid, rows=3, cols=4, spacing=10)
    assert len(positions) > 0
    # Check that odd rows are offset
    row_0 = positions[positions[:, 0] == positions[0, 0]]
    row_1_mask = positions[:, 0] != positions[0, 0]
    if np.any(row_1_mask):
        row_1 = positions[row_1_mask]
        row_1_first = row_1[row_1[:, 0] == row_1[0, 0]]
        if len(row_0) > 0 and len(row_1_first) > 0:
            # Columns should differ between rows (hex offset)
            assert row_0[0, 1] != row_1_first[0, 1]


def test_top_hat_flatness(small_grid):
    target = top_hat(small_grid, radius=10.0)
    inside = np.abs(target) > 0
    # All inside values should be equal
    vals = np.abs(target[inside])
    np.testing.assert_allclose(vals, vals[0], atol=1e-10)


def test_top_hat_outside_zero(small_grid):
    target = top_hat(small_grid, radius=10.0)
    # Far corner should be zero
    assert np.abs(target[0, 0]) == 0.0


def test_gaussian_line_shape(small_grid):
    target = gaussian_line(small_grid, length=20.0, width_sigma=3.0)
    assert target.shape == small_grid
    assert np.max(np.abs(target)) > 0


def test_lg_mode_phase_winding():
    shape = (64, 64)
    field = lg_mode(shape, ell=1, p=0, w0=10.0)
    # Phase should wind by 2*pi around the center
    cy, cx = 31.5, 31.5
    r = 10  # radius for phase sampling
    angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    phases = []
    for a in angles:
        row = int(cy + r * np.sin(a))
        col = int(cx + r * np.cos(a))
        if 0 <= row < 64 and 0 <= col < 64 and np.abs(field[row, col]) > 1e-10:
            phases.append(np.angle(field[row, col]))
    # Phase should span approximately 2*pi range
    if len(phases) > 10:
        phase_range = np.max(phases) - np.min(phases)
        assert phase_range > np.pi  # at least half-winding visible


def test_lg_mode_normalization():
    field = lg_mode((64, 64), ell=1, p=0, w0=10.0)
    np.testing.assert_allclose(np.sum(np.abs(field) ** 2), 1.0, atol=1e-10)


def test_light_sheet_normalization():
    field = light_sheet((64, 64), flat_width=20.0, gaussian_sigma=5.0)
    np.testing.assert_allclose(np.sum(np.abs(field) ** 2), 1.0, atol=1e-10)


def test_light_sheet_flat_region():
    shape = (128, 128)
    field = light_sheet(shape, flat_width=40.0, gaussian_sigma=8.0)
    amp = np.abs(field)
    cy, cx = 63.5, 63.5
    # Sample amplitude along the center row (the flat direction)
    center_row = int(cy)
    flat_half = 20  # flat_width/2
    col_center = int(cx)
    flat_vals = amp[center_row, col_center - flat_half + 2 : col_center + flat_half - 2]
    # Amplitude should be uniform in the flat region
    np.testing.assert_allclose(flat_vals, flat_vals[0], rtol=1e-10)


def test_light_sheet_phase_flat():
    field = light_sheet((64, 64), flat_width=20.0, gaussian_sigma=5.0)
    mask = np.abs(field) > 1e-10
    phases = np.angle(field[mask])
    # All phases should be zero (flat phase)
    np.testing.assert_allclose(phases, 0.0, atol=1e-10)


def test_light_sheet_soft_edge():
    shape = (128, 128)
    hard = light_sheet(shape, flat_width=40.0, gaussian_sigma=8.0, edge_sigma=0.0)
    soft = light_sheet(shape, flat_width=40.0, gaussian_sigma=8.0, edge_sigma=5.0)
    # Soft version should have nonzero amplitude beyond the hard cutoff
    hard_amp = np.abs(hard)
    soft_amp = np.abs(soft)
    # At a point beyond the flat region, soft should be larger
    cy = shape[0] // 2
    edge_col = shape[1] // 2 + 25  # beyond flat_width/2 = 20
    assert soft_amp[cy, edge_col] > hard_amp[cy, edge_col]


def test_mask_from_target_binary(small_grid):
    target = top_hat(small_grid, radius=10.0)
    mask = mask_from_target(target)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert np.sum(mask) > 0


def test_measure_region_includes_target(small_grid):
    target = top_hat(small_grid, radius=5.0)
    region = measure_region(small_grid, target, margin=3)
    mask = mask_from_target(target)
    # Region should include all target positions
    assert np.all(region[mask > 0] > 0)
    # Region should be larger than target mask
    assert np.sum(region) > np.sum(mask)
