"""SLM iterative optimization algorithms for holographic beam shaping."""

__version__ = "0.1.0"

from slm.propagation import (
    fft_propagate,
    ifft_propagate,
    pad_field,
    realistic_ifft_propagate,
    realistic_propagate,
    sinc_envelope,
    zero_order_field,
)
from slm.beams import (
    from_camera_intensity,
    gaussian_beam,
    initial_slm_field,
    random_phase,
    uniform_beam,
)
from slm.targets import (
    chicken_egg_pattern,
    disordered_array,
    gaussian_lattice,
    gaussian_line,
    graphene_lattice,
    hexagonal_grid,
    lg_mode,
    light_sheet,
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
from slm.wgs import phase_fixed_wgs, WGSConfig, WGSResult  # noqa: F401
# Note: the `wgs` function is NOT re-exported here to avoid shadowing the
# `slm.wgs` module.  Import it directly: `from slm.wgs import wgs`

try:
    from slm.wgs import (
        WGS_phase_generate,
        WGS3D_phase_generate,
        fresnel_lens_phase_generate as wgs_fresnel_lens_phase_generate,
        nonUniformity_adapt,
    )
except ImportError:
    pass
from slm.cgm import cgm, CGMConfig, CGMResult, tophat_phase_generate
from slm.cgm_lbfgsb import cgm_lbfgsb
from slm.hybrid import gs_seed_phase
from slm.device import SLMDevice

try:
    from slm.cgm_jax import cgm_jax
except ImportError:
    pass

from slm.camera import CameraInterface, SimulatedCamera, takeda_phase_retrieval

try:
    from slm.camera import VimbaCamera
except ImportError:
    pass

try:
    from slm.generation import SLM_class
    from slm.aberration import Zernike
    from slm.imgpy import IMG, Tweezer, SLM_screen_Correct
except ImportError:
    pass

try:
    from slm.hardware import (
        HardwareConfig,
        apply_lut_correction,
        combine_screens,
        crop_to_slm,
        fresnel_lens_phase,
        load_calibration_bmp,
        phase_to_screen,
        phase_to_uint8,
    )
except ImportError:
    pass
from slm.feedback import (
    adaptive_feedback_continuous,
    adaptive_feedback_loop,
    experimental_feedback_loop,
    FeedbackConfig,
)
from slm.transforms import (
    apply_measured_correction,
    apply_zernike_correction,
    anti_aliased_affine_transform,
    generate_aberration,
    zernike,
    zernike_decompose,
    zernike_from_noll,
)

__all__ = [
    # propagation
    "fft_propagate",
    "ifft_propagate",
    "pad_field",
    "realistic_propagate",
    "realistic_ifft_propagate",
    "sinc_envelope",
    "zero_order_field",
    # beams
    "from_camera_intensity",
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
    "light_sheet",
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
    "phase_fixed_wgs",
    "WGSConfig",
    "WGSResult",
    "cgm",
    "CGMConfig",
    "CGMResult",
    "cgm_lbfgsb",
    "tophat_phase_generate",
    "cgm_jax",
    "gs_seed_phase",
    "SLMDevice",
    # camera
    "CameraInterface",
    "SimulatedCamera",
    "takeda_phase_retrieval",
    # hardware utilities
    "HardwareConfig",
    "apply_lut_correction",
    "combine_screens",
    "crop_to_slm",
    "fresnel_lens_phase",
    "load_calibration_bmp",
    "phase_to_screen",
    "phase_to_uint8",
    # feedback
    "adaptive_feedback_continuous",
    "adaptive_feedback_loop",
    "experimental_feedback_loop",
    "FeedbackConfig",
    # transforms
    "apply_measured_correction",
    "apply_zernike_correction",
    "anti_aliased_affine_transform",
    "generate_aberration",
    "zernike",
    "zernike_decompose",
    "zernike_from_noll",
    # PyTorch WGS (from slm-code)
    "WGS_phase_generate",
    "WGS3D_phase_generate",
    "wgs_fresnel_lens_phase_generate",
    "nonUniformity_adapt",
    # slm-code hardware modules
    "VimbaCamera",
    "SLM_class",
    "Zernike",
    "IMG",
    "Tweezer",
    "SLM_screen_Correct",
]
