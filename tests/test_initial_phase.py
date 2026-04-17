"""Tests for slm.initial_phase (stationary-phase CGM seeds).

These lock the 1D closed-form algebra, the FFT shift-sign convention,
and the ``SLM_class.stationary_phase_sheet`` wrapper.  No CGM calls; no
GPU; all tests run in under 0.1 s total.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erf

from slm.generation import SLM_class
from slm.initial_phase import (
    _diffraction_scale_um,
    cylindrical_lens_for_gaussian_width,
    stationary_phase_1d,
    stationary_phase_light_sheet,
)
from slm.propagation import fft_propagate


# ---------------------------------------------------------------------------
# Test A -- 1D symmetry, zero at origin, finite, non-negative
# ---------------------------------------------------------------------------


def test_stationary_phase_1d_symmetry_sign_finite():
    x = np.linspace(-5000.0, 5000.0, 1001)  # um, symmetric about 0
    phi = stationary_phase_1d(
        x, b_um=1000.0, w0_um=2000.0, wavelength_um=1.013, focal_length_um=200000.0
    )
    assert phi.shape == x.shape
    assert np.all(np.isfinite(phi))
    # phi(0) == 0 exactly (both terms vanish)
    assert np.isclose(phi[500], 0.0, atol=1e-12)
    # even symmetry
    assert np.allclose(phi, phi[::-1], atol=1e-12)
    # non-negative (minimum at x=0, grows outward)
    assert np.all(phi >= -1e-12)


# ---------------------------------------------------------------------------
# Test B -- 1D derivative matches nu(x) = (b/(2*lambda*f)) * erf(sqrt(2)*x/w0)
#
# This is the strongest correctness test: it locks the prefactor *and*
# the erf+exp algebra through the known analytic cancellation
# (d/dx of the exp term exactly cancels the x*erf' piece, leaving
# only the erf term).
# ---------------------------------------------------------------------------


def test_stationary_phase_1d_gradient_matches_nu():
    x = np.linspace(-5000.0, 5000.0, 10001)
    dx = x[1] - x[0]
    b, w0, lam, f = 1000.0, 2000.0, 1.013, 200000.0
    phi = stationary_phase_1d(x, b, w0, lam, f)
    grad_num = np.gradient(phi, dx)
    grad_ana = (np.pi * b / (lam * f)) * erf(np.sqrt(2.0) * x / w0)
    # Skip the first/last few samples to avoid np.gradient edge error.
    sl = slice(10, -10)
    assert np.allclose(grad_num[sl], grad_ana[sl], rtol=1e-4, atol=1e-10)


# ---------------------------------------------------------------------------
# Test C -- 2D: stationary phase alone on a Gaussian input produces a
# recognisable top-hat along the flat-top axis and a natural Gaussian
# perpendicular.  This is an end-to-end check of the physics
# (geom-optics-limit parameters; no CGM involved).
# ---------------------------------------------------------------------------


def test_stationary_phase_light_sheet_yields_tophat_along_line():
    ny = nx = 512
    pixel_pitch_um = 12.5
    w0_um = 5100.0
    flat_width_um = 800.0       # deep in geom-optics limit:
                                # ratio = 800 / (1.013*200000/(pi*5100)) ~ 63x
    wavelength_um = 1.013
    focal_length_um = 200000.0

    # Input Gaussian, unit power, origin at (n-1)/2.
    ix = np.arange(nx) - (nx - 1) / 2.0
    iy = np.arange(ny) - (ny - 1) / 2.0
    X, Y = np.meshgrid(ix * pixel_pitch_um, iy * pixel_pitch_um, indexing="xy")
    gauss = np.exp(-(X ** 2 + Y ** 2) / w0_um ** 2)
    gauss /= np.sqrt(np.sum(gauss ** 2))

    phi = stationary_phase_light_sheet(
        (ny, nx),
        flat_width_um=flat_width_um,
        w0_um=w0_um,
        wavelength_um=wavelength_um,
        focal_length_um=focal_length_um,
        pixel_pitch_um=pixel_pitch_um,
        angle=0.0,
    )

    E_out = fft_propagate(gauss * np.exp(1j * phi))
    I = np.abs(E_out) ** 2

    focal_pitch_um = wavelength_um * focal_length_um / (nx * pixel_pitch_um)
    half_px = int(flat_width_um / (2.0 * focal_pitch_um))

    # Horizontal slice through the centre.
    center_row = I[ny // 2]
    # Guard: we can actually resolve the flat region on this grid.
    assert half_px >= 3, "flat-top too narrow to benchmark on 512 grid"

    inner = center_row[nx // 2 - half_px + 2 : nx // 2 + half_px - 2]
    outside = center_row[: nx // 2 - 2 * half_px]

    # The flat-top region should carry nearly all the power and be
    # roughly uniform.  Thresholds intentionally loose -- the geom-
    # optics solution has visible edge ringing at this FFT resolution.
    assert inner.max() > 0.0
    assert inner.mean() > 5.0 * outside.max()        # big in-vs-out contrast
    assert inner.min() > 0.15 * inner.max()          # flatness (loose)


# ---------------------------------------------------------------------------
# Test D -- Linear-shift sign convention.  Pins the + or - 2*pi choice
# against the project's fft_propagate (fftshift . fft2 . ifftshift,
# norm='ortho').  If this fails the whole light-sheet benchmark lands
# at -center instead of +center -- flip the sign in stationary_phase_
# light_sheet's ramp branch.
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:Stationary phase:UserWarning")
def test_stationary_phase_linear_shift_sign_x_axis():
    ny = nx = 128
    pixel_pitch_um = 12.5
    wavelength_um = 1.013
    focal_length_um = 100000.0
    w0_um = 500.0

    focal_pitch_um = wavelength_um * focal_length_um / (nx * pixel_pitch_um)
    # Shift 5 px to the +x direction at the focal plane.
    target_shift_um = 5.0 * focal_pitch_um

    # Small flat_width (50 um) intentionally near the diffraction limit
    # (~64.5 um); for this test we only care about the peak location, not
    # the flatness.  Ignore the near-diffraction warning.
    phi = stationary_phase_light_sheet(
        (ny, nx),
        flat_width_um=50.0,
        w0_um=w0_um,
        wavelength_um=wavelength_um,
        focal_length_um=focal_length_um,
        pixel_pitch_um=pixel_pitch_um,
        angle=0.0,
        center_um=(target_shift_um, 0.0),
    )

    ix = np.arange(nx) - (nx - 1) / 2.0
    iy = np.arange(ny) - (ny - 1) / 2.0
    X, Y = np.meshgrid(ix * pixel_pitch_um, iy * pixel_pitch_um, indexing="xy")
    gauss = np.exp(-(X ** 2 + Y ** 2) / w0_um ** 2)
    gauss /= np.sqrt(np.sum(gauss ** 2))

    E_out = fft_propagate(gauss * np.exp(1j * phi))
    col_max = int(np.argmax(np.abs(E_out).sum(axis=0)))
    # Expect the peak at column (nx // 2 + 5); fftshift-centered origin is nx // 2.
    assert abs(col_max - (nx // 2 + 5)) <= 1


@pytest.mark.filterwarnings("ignore:Stationary phase:UserWarning")
def test_stationary_phase_linear_shift_sign_y_axis():
    # Mirror of the x-axis test for the y-axis shift term.
    ny = nx = 128
    pixel_pitch_um = 12.5
    wavelength_um = 1.013
    focal_length_um = 100000.0
    w0_um = 500.0

    focal_pitch_um = wavelength_um * focal_length_um / (nx * pixel_pitch_um)
    target_shift_um = 5.0 * focal_pitch_um

    phi = stationary_phase_light_sheet(
        (ny, nx),
        flat_width_um=50.0,
        w0_um=w0_um,
        wavelength_um=wavelength_um,
        focal_length_um=focal_length_um,
        pixel_pitch_um=pixel_pitch_um,
        angle=0.0,
        center_um=(0.0, target_shift_um),
    )

    ix = np.arange(nx) - (nx - 1) / 2.0
    iy = np.arange(ny) - (ny - 1) / 2.0
    X, Y = np.meshgrid(ix * pixel_pitch_um, iy * pixel_pitch_um, indexing="xy")
    gauss = np.exp(-(X ** 2 + Y ** 2) / w0_um ** 2)
    gauss /= np.sqrt(np.sum(gauss ** 2))

    E_out = fft_propagate(gauss * np.exp(1j * phi))
    row_max = int(np.argmax(np.abs(E_out).sum(axis=1)))
    assert abs(row_max - (ny // 2 + 5)) <= 1


# ---------------------------------------------------------------------------
# Test E -- SLM_class.stationary_phase_sheet ergonomic wrapper
# ---------------------------------------------------------------------------


def test_slm_class_stationary_phase_sheet_shape_dtype_symmetry(slm_config, monkeypatch):
    tmp_path, _ = slm_config
    monkeypatch.chdir(tmp_path)
    slm = SLM_class()
    slm.image_init(initGaussianPhase_user_defined=np.zeros((64, 64)), Plot=False)

    # A horizontal line should give a phase that's symmetric in x
    # (flat-top direction) when angle=0.
    phi = slm.stationary_phase_sheet(flat_width=20, angle=0.0, center=None)

    assert phi.shape == (slm.ImgResY, slm.ImgResX)
    assert phi.dtype == np.float64
    assert np.all(np.isfinite(phi))

    # Along-line symmetry: each row is even in x.
    center_row = phi[phi.shape[0] // 2]
    assert np.allclose(center_row, center_row[::-1], atol=1e-10)
    # And the same for a non-centre row, since no perpendicular modulation.
    other_row = phi[phi.shape[0] // 4]
    assert np.allclose(other_row, other_row[::-1], atol=1e-10)


def test_slm_class_stationary_phase_sheet_matches_free_function(
    slm_config, monkeypatch
):
    tmp_path, _ = slm_config
    monkeypatch.chdir(tmp_path)
    slm = SLM_class()
    slm.image_init(initGaussianPhase_user_defined=np.zeros((64, 64)), Plot=False)

    phi_method = slm.stationary_phase_sheet(flat_width=20, angle=0.0, center=None)

    phi_free = stationary_phase_light_sheet(
        (slm.ImgResY, slm.ImgResX),
        flat_width_um=20.0 * slm.Focalpitchx,
        w0_um=slm.beamwaist,
        wavelength_um=slm.wavelength,
        focal_length_um=slm.focallength / slm.magnification,
        pixel_pitch_um=slm.pixelpitch,
        angle=0.0,
    )
    assert np.allclose(phi_method, phi_free, atol=1e-12)


# ---------------------------------------------------------------------------
# Test F -- Diffraction-limit warning fires for pixel-vs-um mixup
# ---------------------------------------------------------------------------


def test_warn_near_diffraction_limit():
    # 10 um flat-top for hardware params (diffraction 1/e^2 ~ 12.64 um)
    # -- ratio 0.79 << 3.0, warning must fire.
    x = np.array([0.0, 1.0, 2.0])
    with pytest.warns(UserWarning, match="geometric-optics"):
        stationary_phase_1d(
            x, b_um=10.0, w0_um=5100.0, wavelength_um=1.013, focal_length_um=200000.0
        )


def test_no_warn_in_geom_optics_regime():
    # Benchmark params: flat_width ~ 538 um, diffraction ~ 12.64 um,
    # ratio ~ 42.5 -- no warning.
    import warnings as _warnings

    x = np.array([0.0, 1.0, 2.0])
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        stationary_phase_1d(
            x, b_um=538.0, w0_um=5100.0, wavelength_um=1.013, focal_length_um=200000.0
        )


def test_diffraction_scale_um_matches_formula():
    d = _diffraction_scale_um(w0_um=5100.0, wavelength_um=1.013, focal_length_um=200000.0)
    assert np.isclose(d, 1.013 * 200000.0 / (np.pi * 5100.0))


# ---------------------------------------------------------------------------
# Test G -- Cylindrical Fresnel lens for perpendicular Gaussian widening
# ---------------------------------------------------------------------------


def test_cylindrical_lens_infinite_for_target_at_diffraction_limit():
    """Cannot shrink the focal Gaussian below diffraction with a thin lens."""
    w_nat = 1.013 * 200000.0 / (np.pi * 6100.0)
    f_cyl = cylindrical_lens_for_gaussian_width(
        target_w_um=w_nat, w0_um=6100.0, wavelength_um=1.013, focal_length_um=200000.0,
    )
    assert f_cyl == float("inf")


def test_cylindrical_lens_recovers_natural_width_at_infinity():
    """As f_cyl -> infinity, the focal-plane Gaussian approaches w_nat."""
    # Going the other way: compute f_cyl for a tiny widening, verify it is huge.
    w_nat = 1.013 * 200000.0 / (np.pi * 6100.0)
    f_cyl = cylindrical_lens_for_gaussian_width(
        target_w_um=1.001 * w_nat, w0_um=6100.0,
        wavelength_um=1.013, focal_length_um=200000.0,
    )
    assert f_cyl > 1e9  # very large (um) -- essentially no lens


def test_cylindrical_lens_widens_focal_gaussian_end_to_end():
    """Apply the computed f_cyl to a Gaussian SLM field and verify the FT
    Gaussian has the requested width."""
    ny = nx = 512
    pixel_pitch_um = 12.5
    # Small w0 so the natural focal Gaussian spans several pixels
    # (at w0=6100 and pitch=12.5, natural is sub-pixel and unmeasurable).
    w0_um = 500.0
    wavelength_um = 1.013
    focal_length_um = 200000.0

    # Ask for a 3x-widened focal Gaussian.
    w_nat = wavelength_um * focal_length_um / (np.pi * w0_um)   # ~128.9 um
    target_w_um = 3.0 * w_nat

    f_cyl = cylindrical_lens_for_gaussian_width(
        target_w_um, w0_um, wavelength_um, focal_length_um,
    )
    assert np.isfinite(f_cyl) and f_cyl > 0

    ix = np.arange(nx) - (nx - 1) / 2.0
    iy = np.arange(ny) - (ny - 1) / 2.0
    X, Y = np.meshgrid(ix * pixel_pitch_um, iy * pixel_pitch_um, indexing="xy")
    gauss = np.exp(-(X ** 2 + Y ** 2) / w0_um ** 2)
    gauss /= np.sqrt(np.sum(gauss ** 2))

    phi_cyl = -np.pi * Y ** 2 / (wavelength_um * f_cyl)
    E_out = fft_propagate(gauss * np.exp(1j * phi_cyl))
    I_out = np.abs(E_out) ** 2

    # Measure the 1/e^2 intensity half-width of the output Gaussian along y
    # at the column of peak brightness.  Use the central column.
    col = I_out[:, nx // 2]
    col_norm = col / col.max()
    above = np.where(col_norm >= np.exp(-2.0))[0]
    measured_half_px = (above[-1] - above[0]) / 2.0
    focal_pitch_um = wavelength_um * focal_length_um / (nx * pixel_pitch_um)
    measured_w_um = measured_half_px * focal_pitch_um

    # Allow 20% slack -- the FT is on a finite grid and peak-finding is
    # noisy.  The key is that we widened well beyond natural and close to
    # the target.
    assert measured_w_um > 2.0 * w_nat       # definitely wider than natural
    assert abs(measured_w_um - target_w_um) / target_w_um < 0.2


def test_stationary_phase_light_sheet_with_perp_lens_matches_target_width():
    """SLM_class.stationary_phase_sheet(gaussian_sigma=...) produces a seed
    whose output Gaussian perpendicular to the line has the requested
    width, matching what light_sheet_target(gaussian_sigma=...) expects."""
    ny = nx = 512
    pixel_pitch_um = 12.5
    # Small w0 to keep the natural focal Gaussian resolvable on 512^2.
    w0_um = 500.0
    wavelength_um = 1.013
    focal_length_um = 200000.0
    flat_width_um = 5000.0     # geom-optics regime (target >> natural 128.9 um)
    perp_target_w_um = 500.0   # ~4x natural

    phi = stationary_phase_light_sheet(
        (ny, nx),
        flat_width_um=flat_width_um,
        w0_um=w0_um,
        wavelength_um=wavelength_um,
        focal_length_um=focal_length_um,
        pixel_pitch_um=pixel_pitch_um,
        angle=0.0,
        perp_target_w_um=perp_target_w_um,
    )

    ix = np.arange(nx) - (nx - 1) / 2.0
    iy = np.arange(ny) - (ny - 1) / 2.0
    X, Y = np.meshgrid(ix * pixel_pitch_um, iy * pixel_pitch_um, indexing="xy")
    gauss = np.exp(-(X ** 2 + Y ** 2) / w0_um ** 2)
    gauss /= np.sqrt(np.sum(gauss ** 2))

    E_out = fft_propagate(gauss * np.exp(1j * phi))
    I_out = np.abs(E_out) ** 2

    # Perpendicular slice through the centre.
    col = I_out[:, nx // 2]
    col_norm = col / col.max()
    above = np.where(col_norm >= np.exp(-2.0))[0]
    measured_half_px = (above[-1] - above[0]) / 2.0
    focal_pitch_um = wavelength_um * focal_length_um / (nx * pixel_pitch_um)
    measured_w_um = measured_half_px * focal_pitch_um

    assert abs(measured_w_um - perp_target_w_um) / perp_target_w_um < 0.2
