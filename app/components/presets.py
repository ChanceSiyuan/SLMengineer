"""Hardware and pattern preset definitions."""

from slm.device import SLMDevice

DEVICE_PRESETS = {
    "BNS P1920 (256x256, 24um)": SLMDevice(
        pixel_pitch_um=24.0, n_pixels=(256, 256),
        wavelength_nm=1070.0, focal_length_mm=150.0,
    ),
    "BNS P2560 (1024x1024, 12.5um)": SLMDevice(
        pixel_pitch_um=12.5, n_pixels=(1024, 1024),
        wavelength_nm=420.0, focal_length_mm=150.0,
    ),
    "Hamamatsu X13138 (1272x1024, 12.5um)": SLMDevice(
        pixel_pitch_um=12.5, n_pixels=(1272, 1024),
        wavelength_nm=532.0, focal_length_mm=200.0,
    ),
    "Custom": None,
}

# Pattern definitions: name -> {description, builder_key, default_params}
PATTERN_CATALOG = {
    "LG01 (vortex beam)": {
        "key": "lg_mode",
        "params": {"ell": 1, "p": 0, "w0": 10.0},
        "desc": "Laguerre-Gaussian with orbital angular momentum l=1",
    },
    "Square Lattice (8x8)": {
        "key": "square_lattice_vortex",
        "params": {"rows": 8, "cols": 8, "spacing": 14.0, "peak_sigma": 3.0, "ell": 1},
        "desc": "Square grid of spots with global vortex phase",
    },
    "Ring Lattice (12 sites)": {
        "key": "ring_lattice_vortex",
        "params": {"n_sites": 12, "ring_radius": 25.0, "peak_sigma": 3.0, "ell": 1},
        "desc": "Ring of spots with vortex phase",
    },
    "Graphene (honeycomb)": {
        "key": "graphene_lattice",
        "params": {"rows": 4, "cols": 4, "spacing": 8.0, "peak_sigma": 2.5},
        "desc": "Honeycomb lattice with alternating sublattice phases",
    },
    "Flat Top (circular)": {
        "key": "top_hat",
        "params": {"radius": 25.0},
        "desc": "Circular uniform intensity with flat phase",
    },
    "Gaussian Line": {
        "key": "gaussian_line",
        "params": {"length": 30.0, "width_sigma": 5.0, "phase_gradient": 0.1},
        "desc": "1D line with Gaussian cross-section and phase gradient",
    },
    "Light Sheet (1D top-hat)": {
        "key": "light_sheet",
        "params": {"flat_width": 50.0, "gaussian_sigma": 10.0},
        "desc": "Rydberg beam: uniform along line, Gaussian perpendicular",
    },
    "Rectangular Grid (4x4)": {
        "key": "rectangular_grid",
        "params": {"rows": 4, "cols": 4, "spacing": 15},
        "desc": "Simple rectangular spot array",
    },
}

# Spatial params: default values are in focal-plane pixels; display in µm.
SPATIAL_PARAMS: set[str] = {
    "w0", "spacing", "peak_sigma", "radius", "ring_radius",
    "length", "width_sigma", "flat_width", "gaussian_sigma",
}

# Frequency params: rad/px in builders; display as mrad/µm.
GRADIENT_PARAMS: set[str] = {"phase_gradient"}
