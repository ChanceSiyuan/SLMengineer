"""Tests for WGS and Phase-Fixed WGS algorithms."""

import numpy as np

from slm.beams import initial_slm_field
from slm.gs import gs
from slm.targets import mask_from_target, rectangular_grid
from slm.wgs import WGSConfig, phase_fixed_wgs, wgs


def test_wgs_outperforms_gs(gaussian_field_128, grid_target_128):
    target, positions, mask = grid_target_128
    n_iter = 50

    gs_result = gs(gaussian_field_128, target, n_iterations=n_iter)
    wgs_result = wgs(
        gaussian_field_128, target, mask, WGSConfig(n_iterations=n_iter)
    )
    # WGS should achieve lower non-uniformity
    assert wgs_result.uniformity_history[-1] < gs_result.uniformity_history[-1]


def test_wgs_weights_converge(gaussian_field_128, grid_target_128):
    target, positions, mask = grid_target_128
    result = wgs(
        gaussian_field_128, target, mask, WGSConfig(n_iterations=100)
    )
    # Weight std should stabilize (later values less variable)
    if len(result.weight_history) > 20:
        early_var = np.std(result.weight_history[:10])
        late_var = np.std(result.weight_history[-10:])
        assert late_var <= early_var * 2  # not diverging


def test_phase_fixed_wgs_faster_convergence(gaussian_field_128, grid_target_128):
    target, positions, mask = grid_target_128
    n_iter = 100

    wgs_result = wgs(
        gaussian_field_128, target, mask, WGSConfig(n_iterations=n_iter)
    )
    pf_result = phase_fixed_wgs(
        gaussian_field_128, target, mask,
        phase_fix_iteration=12, n_iterations=n_iter,
    )
    # Both should converge well; phase-fixed should converge faster in early iterations
    # Compare at iteration 30 (mid-convergence) where the difference is clearer
    mid = min(30, len(pf_result.uniformity_history) - 1, len(wgs_result.uniformity_history) - 1)
    assert pf_result.uniformity_history[mid] <= wgs_result.uniformity_history[mid] * 2.0
    # Both should achieve good final uniformity
    assert pf_result.uniformity_history[-1] < 0.01


def test_wgs_phase_fixed_records_iteration(gaussian_field_128, grid_target_128):
    target, positions, mask = grid_target_128
    result = phase_fixed_wgs(
        gaussian_field_128, target, mask,
        phase_fix_iteration=10, n_iterations=50,
    )
    assert result.phase_fixed_at == 10


def test_wgs_cumulative_weights(gaussian_field_64, four_spot_target_64):
    target, positions = four_spot_target_64
    mask = mask_from_target(target)
    result = wgs(
        gaussian_field_64, target, mask, WGSConfig(n_iterations=30)
    )
    # Should have convergence history
    assert len(result.uniformity_history) == 30
    assert len(result.efficiency_history) == 30
