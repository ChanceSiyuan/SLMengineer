"""Hardware configuration dataclass and JSON loader.

Matches the format of hamamatsu_test_config.json used in ~/slm-code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HardwareConfig:
    """Configuration for a real SLM + camera hardware setup.

    Parameters mirror the fields in ``hamamatsu_test_config.json``.
    """

    # SLM physical parameters
    pixel_pitch_um: float = 12.5
    slm_resolution: tuple[int, int] = (1272, 1024)  # (width, height)

    # Optical system
    wavelength_um: float = 1.013
    focal_length_um: float = 200_000.0
    beam_waist_um: float = 1480.0
    magnification: float = 1.0

    # WGS defaults
    loop: int = 25
    threshold: float = 0.01

    # LUT / calibration
    lut_value: int = 207
    calibration_bmp_path: str | None = None

    # Camera / display timing
    monitor_index: int = 0
    exposure_time_us: float = 30.0
    display_settle_time_s: float = 0.35

    @classmethod
    def from_json(cls, path: str | Path) -> HardwareConfig:
        """Load configuration from a JSON file.

        Supports the ``hamamatsu_test_config.json`` key naming convention.
        """
        with open(path) as f:
            data = json.load(f)

        field_map = {
            "pixelpitch": "pixel_pitch_um",
            "SLMRes": "slm_resolution",
            "wavelength": "wavelength_um",
            "focallength": "focal_length_um",
            "beamwaist": "beam_waist_um",
            "magnification": "magnification",
            "Loop": "loop",
            "threshold": "threshold",
        }

        kwargs: dict = {}
        for json_key, attr in field_map.items():
            if json_key in data:
                val = data[json_key]
                if attr == "slm_resolution":
                    val = tuple(val)
                kwargs[attr] = val

        return cls(**kwargs)

    def to_slm_device(self):
        """Convert to an :class:`~slm.device.SLMDevice` for use with algorithms."""
        from slm.device import SLMDevice

        slm_w, slm_h = self.slm_resolution
        return SLMDevice(
            pixel_pitch_um=self.pixel_pitch_um,
            n_pixels=(slm_h, slm_w),
            wavelength_nm=self.wavelength_um * 1000,
            focal_length_mm=self.focal_length_um / 1000,
        )
