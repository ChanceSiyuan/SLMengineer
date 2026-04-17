"""Analytic SLM-plane initial-phase constructors for CGM warm starts.

This module implements the stationary-phase / geometric-optics method for
mapping a Gaussian input beam to a flat-top at the Fourier plane of a 2F
system, as derived in ``references/Top Hat Beam.pdf``.  The closed-form
1D phase is

    phi_1D(x) = (pi*b/(lambda*f)) * [ x*erf(sqrt(2)*x/w0)
                                     + (w0/sqrt(2*pi)) * (exp(-2*x**2/w0**2) - 1) ]

which satisfies d(phi_1D)/dx = (pi*b/(lambda*f)) * erf(sqrt(2)*x/w0)
= 2*pi*nu(x), giving the exact ray-density redistribution that maps a
Gaussian of 1/e^2 radius w0 to a flat-top of full width b at the focal
plane.

The method is *exact* in the geometric-optics limit (target size
>> lambda*f/(pi*w0)).  Closer to the diffraction limit it is still a
much higher-quality CGM warm start than Bowman's quadratic + linear
seed, even though standalone it produces a ringy output.

Results are returned as ``np.ndarray`` of shape ``(ny, nx)``, dtype
``float64``, ready to pass via ``CGMConfig.initial_phase`` or as
``initSLMPhase`` to ``CGM_phase_generate``.

All length-valued arguments use the ``_um`` suffix to flag physical
micrometre units, matching the ``beam_center_um`` convention in
``src/slm/generation.py::image_init``.  The ``SLM_class`` wrapper
methods (e.g. ``SLM.stationary_phase_sheet``) take pixel-indexed
arguments for ergonomic parity with ``light_sheet_target`` and perform
the conversion internally.

Coordinate convention
---------------------
The SLM-plane origin is placed at fractional pixel ``((nx-1)/2,
(ny-1)/2)`` to match ``slm.cgm._initial_phase``.  The sign of the
focal-plane shift ramp is ``+2*pi*(xc*X + yc*Y)/(lambda*f)``; this is
pinned by ``tests/test_initial_phase.py::test_stationary_phase_linear_shift_sign``
against the project's ``fftshift(fft2(ifftshift(.), norm='ortho')))``
propagator.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.special import erf


__all__ = [
    "stationary_phase_1d",
    "stationary_phase_light_sheet",
    "cylindrical_lens_for_gaussian_width",
]


def _diffraction_scale_um(w0_um: float, wavelength_um: float, focal_length_um: float) -> float:
    """Focal-plane Gaussian 1/e^2 radius for an input beam of waist w0.

    lambda*f/(pi*w0) -- the natural diffraction scale of the 2F system.
    Targets much larger than this are in the geometric-optics limit
    where stationary phase is near-exact.
    """
    return wavelength_um * focal_length_um / (np.pi * w0_um)


def _warn_if_near_diffraction_limit(
    size_um: float,
    w0_um: float,
    wavelength_um: float,
    focal_length_um: float,
    safety_factor: float = 3.0,
) -> None:
    """Emit a UserWarning if ``size_um`` is within ``safety_factor`` of the
    diffraction 1/e^2 radius.  Does not raise.

    The stationary-phase derivation assumes ray-density redistribution, which
    fails as the target size approaches the diffraction limit.  The PDF's own
    worked examples show clear ringing when the ratio drops below ~3.  The
    warning also acts as a pixel-vs-um mixup guard: passing a pixel count
    (e.g. ``flat_width_um=34``) for our hardware trips it immediately.
    """
    diff = _diffraction_scale_um(w0_um, wavelength_um, focal_length_um)
    if size_um < safety_factor * diff:
        warnings.warn(
            f"Stationary phase: target size {size_um:.1f} um is "
            f"{size_um / diff:.2f}x the diffraction scale {diff:.1f} um. "
            f"The geometric-optics approximation will produce a ringy "
            f"standalone output; use only as a CGM warm start.",
            UserWarning,
            stacklevel=3,
        )


def cylindrical_lens_for_gaussian_width(
    target_w_um: float,
    w0_um: float,
    wavelength_um: float,
    focal_length_um: float,
) -> float:
    """Focal length of a cylindrical Fresnel lens that widens the focal-plane
    Gaussian of an input 1/e^2 amplitude waist ``w0_um`` from its natural
    2F size ``lambda*f/(pi*w0)`` to the requested ``target_w_um`` (also 1/e^2
    amplitude waist).

    Derivation (ABCD for a 2F system with thin cylindrical lens at the SLM):

        w_focal^2 = w_nat^2 * (1 + (pi*w0^2 / (lambda*f_cyl))^2)

    where w_nat = lambda*f/(pi*w0).  Solving for f_cyl:

        f_cyl = pi*w0^2 / (lambda * sqrt((target_w/w_nat)^2 - 1))

    Returns ``inf`` if ``target_w_um <= w_nat``; you can't shrink the focal
    Gaussian below the diffraction limit by adding a thin lens.  Caller may
    interpret this as "skip the cylindrical term".

    The sign is irrelevant: both +f_cyl and -f_cyl widen by the same factor
    (the formula depends on |f_cyl|).  We return positive by convention.
    """
    w_nat = wavelength_um * focal_length_um / (np.pi * w0_um)
    if target_w_um <= w_nat:
        return float("inf")
    ratio_sq_minus_1 = (target_w_um / w_nat) ** 2 - 1.0
    return np.pi * w0_um ** 2 / (wavelength_um * np.sqrt(ratio_sq_minus_1))


def stationary_phase_1d(
    x_um: np.ndarray,
    b_um: float,
    w0_um: float,
    wavelength_um: float,
    focal_length_um: float,
) -> np.ndarray:
    """Closed-form 1D stationary-phase SLM phase for a Gaussian -> top-hat.

    Implements the formula from ``references/Top Hat Beam.pdf`` p.2:

        phi(x) = (pi*b/(lambda*f)) * [ x*erf(sqrt(2)*x/w0)
                                      + (w0/sqrt(2*pi)) * (exp(-2*x**2/w0**2) - 1) ]

    Parameters
    ----------
    x_um : SLM-plane coordinate array (micrometres).
    b_um : full width of the target top-hat at the focal plane (um).
    w0_um : 1/e^2 radius of the Gaussian input beam at the SLM plane (um).
    wavelength_um : optical wavelength (um).
    focal_length_um : *effective* 2F focal length (f/magnification for a
        ``SLM_class`` instance).  Determines the focal pitch via
        ``focal_pitch = lambda * f / (N * pixel_pitch)``.

    Returns
    -------
    Phase (radians), same shape/dtype as ``x_um``.  Even in x, zero at x=0.

    Notes
    -----
    Emits a ``UserWarning`` if ``b_um`` is within 3x of the diffraction
    1/e^2 radius ``lambda*f/(pi*w0)``, since the geometric-optics
    approximation breaks down there.
    """
    _warn_if_near_diffraction_limit(b_um, w0_um, wavelength_um, focal_length_um)
    prefactor = np.pi * b_um / (wavelength_um * focal_length_um)     # rad / um
    sqrt2_over_w0 = np.sqrt(2.0) / w0_um
    term1 = x_um * erf(sqrt2_over_w0 * x_um)
    term2 = (w0_um / np.sqrt(2.0 * np.pi)) * (np.exp(-2.0 * x_um ** 2 / w0_um ** 2) - 1.0)
    return prefactor * (term1 + term2)


def stationary_phase_light_sheet(
    shape: tuple[int, int],
    flat_width_um: float,
    w0_um: float,
    wavelength_um: float,
    focal_length_um: float,
    pixel_pitch_um: float,
    angle: float = 0.0,
    center_um: tuple[float, float] = (0.0, 0.0),
    beam_center_um: tuple[float, float] = (0.0, 0.0),
    perp_target_w_um: float | None = None,
) -> np.ndarray:
    """Stationary-phase SLM seed for a light-sheet target (1D top-hat).

    Applies the 1D closed-form phase along the rotated along-line axis
    ``u = (x - beam_cx) * cos(angle) + (y - beam_cy) * sin(angle)`` at
    every point of the SLM grid, with no explicit modulation
    perpendicular to the line.  The natural 2F Fourier transform of the
    input Gaussian along the perpendicular axis gives a focal-plane
    Gaussian of 1/e^2 radius ``lambda*f/(pi*w0)``; any mismatch with the
    target ``gaussian_sigma`` is resolved by subsequent CGM iterations.
    The payoff is along the flat-top direction, where Bowman's
    quadratic + linear seed is worst.

    Parameters
    ----------
    shape : (ny, nx) of the SLM compute grid.
    flat_width_um : full width of the target top-hat at the focal plane (um).
    w0_um : input Gaussian 1/e^2 radius at the SLM plane (um).
    wavelength_um : wavelength (um).
    focal_length_um : effective 2F focal length (f/magnification) (um).
    pixel_pitch_um : SLM pixel pitch (um).
    angle : rotation of the along-line axis (radians, 0 = horizontal).
        Matches ``slm.targets.light_sheet`` convention.
    center_um : (x, y) focal-plane shift of the top-hat from the
        zero-order, in um.  Applied via a linear phase ramp.
    beam_center_um : (dx, dy) offset of the incident-beam centre from the
        SLM geometric centre, in um.  Shifts the stationary-phase origin
        so it stays on the beam.  Set to ``SLM.beam_center_um`` to match
        ``image_init(beam_center_um=...)``.
    perp_target_w_um : if given, the *1/e^2 intensity radius* of the target
        perpendicular Gaussian (um).  Adds a cylindrical Fresnel lens along
        v to pre-broaden the natural focal-plane Gaussian
        ``lambda*f/(pi*w0)`` up to this width.  Requires
        ``perp_target_w_um > lambda*f/(pi*w0)``; otherwise ignored with a
        warning (you can't shrink below diffraction with a thin lens).
        Leave ``None`` (default) for pure 1D shaping with no perpendicular
        modulation.

    Returns
    -------
    Phase array of shape ``(ny, nx)``, dtype float64.  Suitable as
    ``CGMConfig.initial_phase`` or ``CGM_phase_generate(initSLMPhase=...)``.

    Notes
    -----
    Emits a ``UserWarning`` near the diffraction limit.  Pixel-space users
    should prefer ``SLM_class.stationary_phase_sheet`` which handles the
    pixel -> um conversion and reads physical parameters from the
    instance's config.

    For a ``light_sheet`` target with pixel-indexed ``center=(row, col)``,
    convert via
    ``center_um = ((col - (nx-1)/2) * Focalpitchx, (row - (ny-1)/2) * Focalpitchy)``.
    """
    _warn_if_near_diffraction_limit(flat_width_um, w0_um, wavelength_um, focal_length_um)

    ny, nx = shape
    ix = np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0
    iy = np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0
    X, Y = np.meshgrid(ix, iy, indexing="xy")
    XX = X * pixel_pitch_um
    YY = Y * pixel_pitch_um

    # Shift origin so the along-line axis is measured from the beam centre.
    bx_um, by_um = beam_center_um
    Xb = XX - bx_um
    Yb = YY - by_um
    u_um = Xb * np.cos(angle) + Yb * np.sin(angle)
    v_um = -Xb * np.sin(angle) + Yb * np.cos(angle)

    phi = stationary_phase_1d(u_um, flat_width_um, w0_um, wavelength_um, focal_length_um)

    # Optional cylindrical Fresnel lens along v to widen the natural focal
    # Gaussian (lambda*f/(pi*w0)) to ``perp_target_w_um``.  Sign verified by
    # test_stationary_phase_perp_lens_widens_gaussian.
    if perp_target_w_um is not None:
        f_cyl_um = cylindrical_lens_for_gaussian_width(
            perp_target_w_um, w0_um, wavelength_um, focal_length_um
        )
        if np.isfinite(f_cyl_um):
            phi = phi - np.pi * v_um ** 2 / (wavelength_um * f_cyl_um)
        else:
            warnings.warn(
                f"stationary_phase_light_sheet: perp_target_w_um="
                f"{perp_target_w_um:.1f} is at or below the diffraction scale "
                f"lambda*f/(pi*w0)={wavelength_um*focal_length_um/(np.pi*w0_um):.2f} "
                f"um; skipping cylindrical lens (would require negative f_cyl).",
                UserWarning,
                stacklevel=2,
            )

    # Linear shift ramp: moves the top-hat to ``center_um`` at the focal plane.
    # Sign verified by test_stationary_phase_linear_shift_sign against the
    # project's fft_propagate (fftshift . fft2 . ifftshift, norm='ortho').
    xc_um, yc_um = center_um
    if xc_um != 0.0 or yc_um != 0.0:
        phi = phi + 2.0 * np.pi * (xc_um * XX + yc_um * YY) / (
            wavelength_um * focal_length_um
        )

    return phi
