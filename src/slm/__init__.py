"""SLM iterative optimization algorithms for holographic beam shaping."""

__version__ = "0.1.0"

from slm.propagation import fft_propagate, ifft_propagate, pad_field
from slm.beams import gaussian_beam, initial_slm_field, random_phase, uniform_beam
from slm.targets import (
    chicken_egg_pattern,
    disordered_array,
    gaussian_lattice,
    gaussian_line,
    graphene_lattice,
    hexagonal_grid,
    lg_mode,
    mask_from_target,
    measure_region,
    rectangular_grid,
    ring_lattice_vortex,
    spot_array,
    square_lattice_vortex,
    top_hat,
)
from slm.visualization import plot_target_field
from slm.metrics import (
    efficiency,
    fidelity,
    non_uniformity_error,
    phase_error,
    uniformity,
)
from slm.gs import gs, GSResult
from slm.wgs import wgs, phase_fixed_wgs, WGSConfig, WGSResult
from slm.cgm import cgm, CGMConfig, CGMResult
from slm.feedback import adaptive_feedback_loop, FeedbackConfig
from slm.transforms import (
    apply_zernike_correction,
    anti_aliased_affine_transform,
    generate_aberration,
    zernike,
    zernike_from_noll,
)

__all__ = [
    # propagation
    "fft_propagate",
    "ifft_propagate",
    "pad_field",
    # beams
    "gaussian_beam",
    "initial_slm_field",
    "random_phase",
    "uniform_beam",
    # targets
    "chicken_egg_pattern",
    "disordered_array",
    "gaussian_lattice",
    "gaussian_line",
    "graphene_lattice",
    "hexagonal_grid",
    "lg_mode",
    "mask_from_target",
    "measure_region",
    "plot_target_field",
    "rectangular_grid",
    "ring_lattice_vortex",
    "spot_array",
    "square_lattice_vortex",
    "top_hat",
    # metrics
    "efficiency",
    "fidelity",
    "non_uniformity_error",
    "phase_error",
    "uniformity",
    # algorithms
    "gs",
    "GSResult",
    "wgs",
    "phase_fixed_wgs",
    "WGSConfig",
    "WGSResult",
    "cgm",
    "CGMConfig",
    "CGMResult",
    # feedback
    "adaptive_feedback_loop",
    "FeedbackConfig",
    # transforms
    "apply_zernike_correction",
    "anti_aliased_affine_transform",
    "generate_aberration",
    "zernike",
    "zernike_from_noll",
]
