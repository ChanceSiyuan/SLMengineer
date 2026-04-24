"""Target pattern generators for focal plane."""

from __future__ import annotations

import numpy as np
from scipy.special import assoc_laguerre

from slm.generation import SLM_class


class SLM_cgm_class(SLM_class):
    """Deprecated alias for :class:`slm.generation.SLM_class`.

    Since the unification in issue #13, ``SLM_class`` itself produces
    complex-valued targets and exposes the CGM-only factory methods
    (``top_hat_target``, ``lg_mode_target``, etc.).  ``SLM_cgm_class``
    is kept as a thin alias for backwards compatibility and emits a
    :class:`DeprecationWarning` on instantiation.
    """

    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "SLM_cgm_class is deprecated; use SLM_class directly — its "
            "target methods now return np.complex128 arrays.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def spot_array(
    shape: tuple[int, int],
    positions: np.ndarray,
    amplitudes: np.ndarray | None = None,
) -> np.ndarray:
    """Spot array target at discrete positions.

    Parameters
    ----------
    shape : output plane grid size (ny, nx).
    positions : (N, 2) array of (row, col) spot positions in pixels.
    amplitudes : (N,) per-spot amplitudes; defaults to uniform 1.0.

    Returns
    -------
    Complex target field with specified amplitudes at spot positions.
    """
    positions = np.asarray(positions, dtype=int)
    target = np.zeros(shape, dtype=np.complex128)
    if amplitudes is None:
        amplitudes = np.ones(len(positions))
    else:
        amplitudes = np.asarray(amplitudes, dtype=np.float64)
    for i, (r, c) in enumerate(positions):
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            target[r, c] = amplitudes[i]
    return target


def rectangular_grid(
    shape: tuple[int, int],
    rows: int,
    cols: int,
    spacing: float,
    center: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate spot_array for an M x N rectangular grid.

    Returns (target_field, positions) where positions is (M*N, 2).
    """
    ny, nx = shape
    if center is None:
        center = (ny // 2, nx // 2)
    positions = []
    for i in range(rows):
        for j in range(cols):
            r = int(center[0] + (i - (rows - 1) / 2.0) * spacing)
            c = int(center[1] + (j - (cols - 1) / 2.0) * spacing)
            if 0 <= r < ny and 0 <= c < nx:
                positions.append([r, c])
    positions = np.array(positions, dtype=int)
    target = spot_array(shape, positions)
    return target, positions


def hexagonal_grid(
    shape: tuple[int, int],
    rows: int,
    cols: int,
    spacing: float,
    center: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate spot_array for a hexagonal lattice.

    Returns (target_field, positions).
    """
    ny, nx = shape
    if center is None:
        center = (ny // 2, nx // 2)
    row_spacing = spacing * np.sqrt(3) / 2.0
    positions = []
    for i in range(rows):
        offset = spacing / 2.0 if i % 2 else 0.0
        for j in range(cols):
            r = int(center[0] + (i - (rows - 1) / 2.0) * row_spacing)
            c = int(center[1] + (j - (cols - 1) / 2.0) * spacing + offset)
            if 0 <= r < ny and 0 <= c < nx:
                positions.append([r, c])
    positions = np.array(positions, dtype=int)
    target = spot_array(shape, positions)
    return target, positions


def top_hat(
    shape: tuple[int, int],
    radius: float,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Circular flat-top intensity target with flat phase."""
    ny, nx = shape
    if center is None:
        center = ((ny - 1) / 2.0, (nx - 1) / 2.0)
    y = np.arange(ny) - center[0]
    x = np.arange(nx) - center[1]
    yy, xx = np.meshgrid(y, x, indexing="ij")
    rr = np.sqrt(xx**2 + yy**2)
    target = np.zeros(shape, dtype=np.complex128)
    target[rr <= radius] = 1.0
    return target


def gaussian_line(
    shape: tuple[int, int],
    length: float,
    width_sigma: float,
    angle: float = 0.0,
    center: tuple[float, float] | None = None,
    phase_gradient: float = 0.0,
) -> np.ndarray:
    """1D Gaussian line: flat along length, Gaussian cross-section.

    Parameters
    ----------
    phase_gradient : linear phase ramp along line direction (rad/px).
    """
    ny, nx = shape
    if center is None:
        center = ((ny - 1) / 2.0, (nx - 1) / 2.0)
    y = np.arange(ny) - center[0]
    x = np.arange(nx) - center[1]
    yy, xx = np.meshgrid(y, x, indexing="ij")
    # Rotate coordinates
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    u = xx * cos_a + yy * sin_a  # along line
    v = -xx * sin_a + yy * cos_a  # perpendicular
    # Flat along u within half-length, Gaussian in v
    amplitude = np.zeros(shape, dtype=np.float64)
    mask = np.abs(u) <= length / 2.0
    amplitude[mask] = np.exp(-(v[mask] ** 2) / (2.0 * width_sigma**2))
    target = amplitude.astype(np.complex128) * np.exp(1j * phase_gradient * u)
    return target


def light_sheet(
    shape: tuple[int, int],
    flat_width: float,
    gaussian_sigma: float,
    angle: float = 0.0,
    center: tuple[float, float] | None = None,
    edge_sigma: float = 0.0,
    reweight: np.ndarray | None = None,
) -> np.ndarray:
    """1D top-hat (light sheet): flat along one axis, Gaussian perpendicular.

    Produces the target field for Rydberg beam shaping: uniform amplitude
    for *flat_width* pixels along the line direction, Gaussian roll-off
    with *gaussian_sigma* perpendicular.  Phase is flat (zero).
    Normalized to unit power: sum(|field|^2) = 1.

    Parameters
    ----------
    shape : (ny, nx) output grid.
    flat_width : full width of the uniform region (pixels).
    gaussian_sigma : 1/e^2 Gaussian width perpendicular to line (pixels).
    angle : rotation angle in radians (0 = horizontal).
    center : pattern center (row, col); defaults to grid center.
    edge_sigma : if > 0, apply Gaussian taper at ends of the flat region
        instead of a hard cutoff.  The taper has this 1/e^2 width.
    reweight : optional 1D vector modulating the flat region amplitude
        (issue #23).  Length can differ from the flat region's pixel
        count; linear interpolation maps vector index to along-line
        position.  ``None`` or all-ones leaves the sheet uniform.  Only
        the nominal flat region (|u| <= flat_width/2) is modulated;
        any edge_sigma taper tails are left untouched.
    """
    ny, nx = shape
    if center is None:
        center = ((ny - 1) / 2.0, (nx - 1) / 2.0)
    y = np.arange(ny) - center[0]
    x = np.arange(nx) - center[1]
    yy, xx = np.meshgrid(y, x, indexing="ij")
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    u = xx * cos_a + yy * sin_a  # along line
    v = -xx * sin_a + yy * cos_a  # perpendicular

    # Perpendicular Gaussian envelope
    perp = np.exp(-(v**2) / (2.0 * gaussian_sigma**2))

    # Along-line profile: flat within flat_width, optional soft taper
    half = flat_width / 2.0
    if edge_sigma > 0:
        # Smooth taper: 1 inside flat region, Gaussian roll-off outside
        dist_outside = np.maximum(0.0, np.abs(u) - half)
        along = np.exp(-(dist_outside**2) / (2.0 * edge_sigma**2))
    else:
        along = (np.abs(u) <= half).astype(np.float64)

    if reweight is not None:
        rw = np.asarray(reweight, dtype=np.float64).ravel()
        if rw.size >= 2:
            idx_f = np.clip((u + half) / (2.0 * half), 0.0, 1.0) * (rw.size - 1)
            idx_lo = np.clip(np.floor(idx_f).astype(int), 0, rw.size - 1)
            idx_hi = np.minimum(idx_lo + 1, rw.size - 1)
            frac = idx_f - idx_lo
            weight_map = (1.0 - frac) * rw[idx_lo] + frac * rw[idx_hi]
            in_flat = np.abs(u) <= half
            along = np.where(in_flat, along * weight_map, along)

    field = (along * perp).astype(np.complex128)
    power = np.sum(np.abs(field) ** 2)
    if power > 0:
        field /= np.sqrt(power)
    return field


def light_sheet_1d(
    length: int,
    flat_width: float,
    center: float | None = None,
    edge_sigma: float = 0.0,
) -> np.ndarray:
    """Pure 1D top-hat along the line axis — companion to :func:`light_sheet`.

    For the dimension-decomposed light-sheet CGM (issue #21): when the SLM
    phase is set constant in ``y`` the focal-plane y-envelope is given by
    the natural 2F transform of the input Gaussian, so the target only
    needs to specify the along-line shape.  ``light_sheet_1d`` returns
    exactly that 1D profile.

    Parameters mirror :func:`light_sheet`'s along-line arguments.
    """
    if center is None:
        center = (length - 1) / 2.0
    u = np.arange(length, dtype=np.float64) - center
    half = flat_width / 2.0
    if edge_sigma > 0:
        dist_outside = np.maximum(0.0, np.abs(u) - half)
        profile = np.exp(-(dist_outside ** 2) / (2.0 * edge_sigma ** 2))
    else:
        profile = (np.abs(u) <= half).astype(np.float64)
    field = profile.astype(np.complex128)
    power = float(np.sum(np.abs(field) ** 2))
    if power > 0:
        field /= np.sqrt(power)
    return field


def lg_mode(
    shape: tuple[int, int],
    ell: int,
    p: int,
    w0: float,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Laguerre-Gaussian mode LG^p_ell.

    Returns complex field with both amplitude and vortex phase.
    """
    ny, nx = shape
    if center is None:
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    else:
        cy, cx = center
    y = np.arange(ny) - cy
    x = np.arange(nx) - cx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    rr = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)

    rho = rr / w0
    amp = (
        (np.sqrt(2) * rho) ** abs(ell)
        * np.exp(-(rho**2))
        * assoc_laguerre(2 * rho**2, p, abs(ell))
    )
    phase = ell * theta
    field = amp * np.exp(1j * phase)
    # Normalize
    power = np.sum(np.abs(field) ** 2)
    if power > 0:
        field /= np.sqrt(power)
    return field


def gaussian_lattice(
    shape: tuple[int, int],
    positions: np.ndarray,
    peak_sigma: float,
    phases: np.ndarray | None = None,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Continuous lattice: sum of Gaussian peaks at given positions.

    Parameters
    ----------
    shape : (ny, nx) output grid.
    positions : (N, 2) array of (row, col) offsets relative to center.
    peak_sigma : width of each Gaussian peak in pixels.
    phases : (N,) per-site phases; defaults to 0 for all sites.
    center : center of pattern; defaults to grid center.
    """
    ny, nx = shape
    if center is None:
        center = ((ny - 1) / 2.0, (nx - 1) / 2.0)
    positions = np.asarray(positions, dtype=np.float64)
    if phases is None:
        phases = np.zeros(len(positions))
    else:
        phases = np.asarray(phases, dtype=np.float64)

    y = np.arange(ny) - center[0]
    x = np.arange(nx) - center[1]
    yy, xx = np.meshgrid(y, x, indexing="ij")

    field = np.zeros(shape, dtype=np.complex128)
    for i, (dr, dc) in enumerate(positions):
        gauss = np.exp(-((yy - dr) ** 2 + (xx - dc) ** 2) / (2.0 * peak_sigma**2))
        field += gauss * np.exp(1j * phases[i])

    power = np.sum(np.abs(field) ** 2)
    if power > 0:
        field /= np.sqrt(power)
    return field


def _apply_vortex_phase(
    field: np.ndarray,
    ell: int,
    center: tuple[float, float],
) -> np.ndarray:
    """Replace field phase with global vortex exp(i*ell*theta), preserve amplitude."""
    ny, nx = field.shape
    y = np.arange(ny) - center[0]
    x = np.arange(nx) - center[1]
    yy, xx = np.meshgrid(y, x, indexing="ij")
    theta = np.arctan2(yy, xx)
    return np.abs(field) * np.exp(1j * ell * theta)


def square_lattice_vortex(
    shape: tuple[int, int],
    rows: int,
    cols: int,
    spacing: float,
    peak_sigma: float,
    ell: int = 1,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Square grid of Gaussian peaks with global vortex phase exp(i*l*theta)."""
    ny, nx = shape
    if center is None:
        center = ((ny - 1) / 2.0, (nx - 1) / 2.0)

    ii, jj = np.mgrid[:rows, :cols]
    dr = (ii - (rows - 1) / 2.0) * spacing
    dc = (jj - (cols - 1) / 2.0) * spacing
    positions = np.column_stack([dr.ravel(), dc.ravel()])

    field = gaussian_lattice(shape, positions, peak_sigma, center=center)
    field = _apply_vortex_phase(field, ell, center)
    power = np.sum(np.abs(field) ** 2)
    if power > 0:
        field /= np.sqrt(power)
    return field


def ring_lattice_vortex(
    shape: tuple[int, int],
    n_sites: int,
    ring_radius: float,
    peak_sigma: float,
    ell: int = 1,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Ring of Gaussian peaks with global vortex phase."""
    ny, nx = shape
    if center is None:
        center = ((ny - 1) / 2.0, (nx - 1) / 2.0)

    angles = np.linspace(0, 2 * np.pi, n_sites, endpoint=False)
    positions = np.column_stack(
        [
            ring_radius * np.sin(angles),
            ring_radius * np.cos(angles),
        ]
    )

    field = gaussian_lattice(shape, positions, peak_sigma, center=center)
    field = _apply_vortex_phase(field, ell, center)
    power = np.sum(np.abs(field) ** 2)
    if power > 0:
        field /= np.sqrt(power)
    return field


def graphene_lattice(
    shape: tuple[int, int],
    rows: int,
    cols: int,
    spacing: float,
    peak_sigma: float,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Honeycomb lattice with alternating phase between sublattices.

    Sublattice A sites have phase 0, sublattice B sites have phase pi.
    """
    ny, nx = shape
    if center is None:
        center = ((ny - 1) / 2.0, (nx - 1) / 2.0)

    # Honeycomb lattice vectors
    a1 = np.array([spacing * np.sqrt(3), 0.0])
    a2 = np.array([spacing * np.sqrt(3) / 2.0, spacing * 3.0 / 2.0])
    # Two-atom basis: A at origin, B offset
    b_offset = np.array([spacing * np.sqrt(3) / 2.0, spacing / 2.0])

    positions = []
    sublattice = []  # 0 for A, 1 for B
    for i in range(-(rows // 2), rows // 2 + 1):
        for j in range(-(cols // 2), cols // 2 + 1):
            pos_a = i * a1 + j * a2
            pos_b = pos_a + b_offset
            # (row, col) = (y, x)
            positions.append([pos_a[1], pos_a[0]])
            sublattice.append(0)
            positions.append([pos_b[1], pos_b[0]])
            sublattice.append(1)

    positions = np.array(positions)
    phases = np.array([0.0 if s == 0 else np.pi for s in sublattice])

    return gaussian_lattice(shape, positions, peak_sigma, phases, center)


def chicken_egg_pattern(
    shape: tuple[int, int],
    radius: float = 50.0,
    center: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Synthetic pattern with uncorrelated intensity and phase.

    Stand-in for the 'Chicken & Egg' arbitrary image test case.
    Uses smooth filtered noise for both amplitude and phase.
    """
    from scipy.ndimage import gaussian_filter

    ny, nx = shape
    if center is None:
        center = ((ny - 1) / 2.0, (nx - 1) / 2.0)
    if rng is None:
        rng = np.random.default_rng(12345)

    # Circular boundary
    y = np.arange(ny) - center[0]
    x = np.arange(nx) - center[1]
    yy, xx = np.meshgrid(y, x, indexing="ij")
    rr = np.sqrt(xx**2 + yy**2)
    disk = (rr <= radius).astype(np.float64)

    # Smooth random intensity
    noise_amp = rng.standard_normal(shape)
    smooth_amp = gaussian_filter(noise_amp, sigma=8.0)
    smooth_amp = (smooth_amp - smooth_amp.min()) / (
        smooth_amp.max() - smooth_amp.min() + 1e-30
    )
    amplitude = smooth_amp * disk

    # Independent smooth random phase
    noise_phase = rng.standard_normal(shape)
    smooth_phase = gaussian_filter(noise_phase, sigma=8.0)
    smooth_phase = (
        smooth_phase / (smooth_phase.max() - smooth_phase.min() + 1e-30) * 2 * np.pi
    )

    field = amplitude * np.exp(1j * smooth_phase)
    power = np.sum(np.abs(field) ** 2)
    if power > 0:
        field /= np.sqrt(power)
    return field


def disordered_array(
    shape: tuple[int, int],
    n_spots: int,
    extent: float,
    min_distance: float = 2.0,
    center: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Random non-overlapping spot positions within a circular region.

    Uses rejection sampling to enforce minimum inter-spot distance.
    Returns (target_field, positions) like rectangular_grid.
    """
    ny, nx = shape
    if center is None:
        center = (ny // 2, nx // 2)
    if rng is None:
        rng = np.random.default_rng()

    positions = []
    max_attempts = n_spots * 100
    attempts = 0
    while len(positions) < n_spots and attempts < max_attempts:
        # Random position within circular extent
        angle = rng.uniform(0, 2 * np.pi)
        r = extent * np.sqrt(rng.uniform(0, 1))  # uniform in disk
        dr, dc = r * np.sin(angle), r * np.cos(angle)
        row = int(center[0] + dr)
        col = int(center[1] + dc)

        if not (0 <= row < ny and 0 <= col < nx):
            attempts += 1
            continue

        # Check minimum distance
        too_close = False
        for pr, pc in positions:
            if (row - pr) ** 2 + (col - pc) ** 2 < min_distance**2:
                too_close = True
                break
        if not too_close:
            positions.append([row, col])
        attempts += 1

    positions = np.array(positions, dtype=int)
    target = spot_array(shape, positions)
    return target, positions


def mask_from_target(
    target: np.ndarray,
    threshold: float | None = None,
) -> np.ndarray:
    """Binary mask: 1 where |target| > threshold, 0 elsewhere.

    Default threshold is 0.1% of peak amplitude.
    """
    if threshold is None:
        max_amp = np.max(np.abs(target))
        threshold = 1e-3 * max_amp if max_amp > 0 else 0.0
    return (np.abs(target) > threshold).astype(np.float64)


def measure_region(
    shape: tuple[int, int],
    target: np.ndarray,
    margin: int = 5,
) -> np.ndarray:
    """Measure region Omega for CGM: non-zero target + surrounding margin.

    The region includes the target pattern plus a border of zero-intensity pixels.
    """
    mask = mask_from_target(target)
    from scipy.ndimage import binary_dilation

    struct = np.ones((2 * margin + 1, 2 * margin + 1))
    dilated = binary_dilation(mask.astype(bool), structure=struct)
    return dilated.astype(np.float64)


def measure_region_1d(
    target_1d: np.ndarray,
    margin: int = 5,
) -> np.ndarray:
    """1D companion to :func:`measure_region` for the 1D CGM path."""
    mask = np.abs(target_1d) > 0
    from scipy.ndimage import binary_dilation

    struct = np.ones(2 * margin + 1, dtype=bool)
    dilated = binary_dilation(mask.astype(bool), structure=struct)
    return dilated.astype(np.float64)
