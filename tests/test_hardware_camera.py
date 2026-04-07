"""Tests for HardwareCamera with mocked SLM display and camera."""

from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from slm.camera import HardwareCamera


class MockDisplay:
    """Mock SLM display that records calls."""

    def __init__(self):
        self.last_array = None
        self.call_count = 0

    def update_array(self, array: np.ndarray) -> None:
        self.last_array = array.copy()
        self.call_count += 1


class MockCamera:
    """Mock camera that returns a synthetic image."""

    def __init__(self, shape: tuple[int, int] = (480, 640)):
        self.shape = shape
        self.last_exposure = None

    def capture(self, exposure_time_us: float, timeout: float = 2.0) -> np.ndarray:
        self.last_exposure = exposure_time_us
        return np.ones(self.shape, dtype=np.uint8) * 42


class TestHardwareCamera:
    def test_capture_intensity_pipeline(self):
        display = MockDisplay()
        camera = MockCamera()

        hw = HardwareCamera(
            slm_display=display,
            camera=camera,
            slm_resolution=(64, 64),
            exposure_time_us=30.0,
            settle_time_s=0.0,  # no sleep in tests
            lut_value=256,
        )

        phase = np.zeros((64, 64))
        result = hw.capture_intensity(phase)

        # Display should have been called
        assert display.call_count == 1
        assert display.last_array is not None
        assert display.last_array.dtype == np.uint8
        assert display.last_array.shape == (64, 64)

        # Camera should have been called
        assert camera.last_exposure == 30.0

        # Result should be float64
        assert result.dtype == np.float64
        assert result.shape == (480, 640)
        np.testing.assert_array_equal(result, 42.0)

    def test_with_fresnel_screen(self):
        display = MockDisplay()
        camera = MockCamera()

        fresnel = np.full((64, 64), 100, dtype=np.uint8)

        hw = HardwareCamera(
            slm_display=display,
            camera=camera,
            slm_resolution=(64, 64),
            exposure_time_us=30.0,
            settle_time_s=0.0,
            lut_value=256,
            fresnel_screen=fresnel,
        )

        phase = np.zeros((64, 64))
        hw.capture_intensity(phase)

        # The displayed array should include the Fresnel lens
        # Zero phase -> 128, combined with 100 -> (128+100)%256 = 228
        # Then LUT 256/256 = no scaling
        screen = display.last_array
        centre = screen[32, 32]
        assert centre != 128  # Should be modified by Fresnel

    def test_with_calibration(self):
        display = MockDisplay()
        camera = MockCamera()

        cal = np.full((64, 64), 10, dtype=np.uint8)

        hw = HardwareCamera(
            slm_display=display,
            camera=camera,
            slm_resolution=(64, 64),
            exposure_time_us=30.0,
            settle_time_s=0.0,
            lut_value=207,
            calibration=cal,
        )

        phase = np.zeros((64, 64))
        hw.capture_intensity(phase)

        # Should have applied LUT correction
        assert display.call_count == 1

    def test_from_config(self):
        from slm.hardware.config import HardwareConfig

        display = MockDisplay()
        camera = MockCamera()
        config = HardwareConfig(
            slm_resolution=(64, 64),
            exposure_time_us=50.0,
            display_settle_time_s=0.0,
            lut_value=224,
        )

        hw = HardwareCamera.from_config(display, camera, config)
        assert hw._slm_resolution == (64, 64)
        assert hw._exposure_time_us == 50.0
        assert hw._lut_value == 224

    def test_capture_fringes_fallback(self):
        display = MockDisplay()
        camera = MockCamera()

        hw = HardwareCamera(
            slm_display=display,
            camera=camera,
            slm_resolution=(64, 64),
            settle_time_s=0.0,
        )

        phase = np.zeros((64, 64))
        result = hw.capture_fringes(phase)
        # Falls back to capture_intensity
        assert result.dtype == np.float64

    def test_larger_computation_grid(self):
        """Phase from a larger computation grid gets cropped to SLM size."""
        display = MockDisplay()
        camera = MockCamera()

        hw = HardwareCamera(
            slm_display=display,
            camera=camera,
            slm_resolution=(64, 64),
            settle_time_s=0.0,
            lut_value=256,
        )

        # Algorithm runs on 256x256 but SLM is 64x64
        phase = np.zeros((256, 256))
        hw.capture_intensity(phase)

        assert display.last_array.shape == (64, 64)
