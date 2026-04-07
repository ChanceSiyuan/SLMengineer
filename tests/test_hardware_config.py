"""Tests for slm.hardware.config."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from slm.hardware.config import HardwareConfig


class TestHardwareConfig:
    def test_defaults(self):
        config = HardwareConfig()
        assert config.pixel_pitch_um == 12.5
        assert config.slm_resolution == (1272, 1024)
        assert config.wavelength_um == 1.013

    def test_from_json(self, tmp_path):
        data = {
            "pixelpitch": 12.5,
            "SLMRes": [1272, 1024],
            "wavelength": 1.013,
            "focallength": 200000,
            "beamwaist": 1480,
            "magnification": 1,
            "Loop": 25,
            "threshold": 0.01,
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(data))

        config = HardwareConfig.from_json(path)
        assert config.pixel_pitch_um == 12.5
        assert config.slm_resolution == (1272, 1024)
        assert config.wavelength_um == 1.013
        assert config.focal_length_um == 200000
        assert config.beam_waist_um == 1480
        assert config.loop == 25
        assert config.threshold == 0.01

    def test_from_json_partial(self, tmp_path):
        data = {"pixelpitch": 24.0, "wavelength": 0.85}
        path = tmp_path / "partial.json"
        path.write_text(json.dumps(data))

        config = HardwareConfig.from_json(path)
        assert config.pixel_pitch_um == 24.0
        assert config.wavelength_um == 0.85
        # Defaults preserved
        assert config.slm_resolution == (1272, 1024)

    def test_to_slm_device(self):
        config = HardwareConfig(
            pixel_pitch_um=12.5,
            slm_resolution=(1272, 1024),
            wavelength_um=1.013,
            focal_length_um=200000,
        )
        device = config.to_slm_device()
        assert device.pixel_pitch_um == 12.5
        assert device.n_pixels == (1024, 1272)
        assert device.wavelength_nm == pytest.approx(1013.0)
        assert device.focal_length_mm == pytest.approx(200.0)
