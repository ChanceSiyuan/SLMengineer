"""Hardware control and utilities for real SLM devices and cameras.

Pure utility functions (phase_convert, lut, fresnel, config) work everywhere.
Hardware drivers (slm_display, vimba_camera) require optional dependencies
(wxPython, vmbpy) and only run on the lab machine.
"""

from slm.hardware.phase_convert import crop_to_slm, phase_to_screen, phase_to_uint8
from slm.hardware.lut import apply_lut_correction, load_calibration_bmp
from slm.hardware.fresnel import combine_screens, fresnel_lens_phase
from slm.hardware.config import HardwareConfig

__all__ = [
    "crop_to_slm",
    "phase_to_screen",
    "phase_to_uint8",
    "apply_lut_correction",
    "load_calibration_bmp",
    "combine_screens",
    "fresnel_lens_phase",
    "HardwareConfig",
]

# Hardware drivers are imported explicitly by the user when needed:
#   from slm.hardware.slm_display import SLMDisplay
#   from slm.hardware.vimba_camera import VimbaCamera
