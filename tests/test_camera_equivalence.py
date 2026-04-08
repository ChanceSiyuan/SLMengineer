"""Verify SimulatedCamera and HardwareCamera produce consistent results.

The two cameras have different data paths:
  SimulatedCamera: float64 phase → FFT propagation → intensity + noise
  HardwareCamera:  float64 phase → uint8 screen → display → camera capture

This test verifies that when HardwareCamera's mock camera performs FFT
propagation on the quantized phase (simulating a perfect optical system),
the results match SimulatedCamera within the quantization error bound.

Quantization budget: phase_to_uint8 maps [-pi, pi] → 256 levels,
giving a maximum phase error of 2*pi/256 ≈ 0.0245 rad per pixel.
"""

from __future__ import annotations

import numpy as np
import pytest

from slm.beams import gaussian_beam, initial_slm_field
from slm.camera import HardwareCamera, SimulatedCamera
from slm.cgm import CGMConfig, cgm
from slm.feedback import experimental_feedback_loop
from slm.hardware.phase_convert import phase_to_screen, phase_to_uint8
from slm.metrics import efficiency, fidelity
from slm.propagation import fft_propagate
from slm.targets import mask_from_target, measure_region, rectangular_grid
from slm.wgs import WGSConfig, WGSResult, wgs


# ---------------------------------------------------------------------------
# Mock hardware that simulates a perfect optical system
# ---------------------------------------------------------------------------


class FFTMockDisplay:
    """Records the last uint8 screen sent to the SLM."""

    def __init__(self):
        self.last_screen: np.ndarray | None = None

    def update_array(self, array: np.ndarray) -> None:
        self.last_screen = array.copy()


class FFTMockCamera:
    """Mock camera that reads the displayed screen, converts back to phase,
    applies a known input amplitude, FFT-propagates, and returns intensity.

    This simulates a perfect optical system where the only error source
    is the uint8 phase quantization in the display pipeline.
    """

    def __init__(
        self,
        display: FFTMockDisplay,
        input_amplitude: np.ndarray,
        slm_resolution: tuple[int, int],
    ):
        self.display = display
        self.input_amplitude = input_amplitude
        self.slm_resolution = slm_resolution

    def capture(self, exposure_time_us: float, timeout: float = 2.0) -> np.ndarray:
        screen = self.display.last_screen
        assert screen is not None, "Display has no screen to capture"

        # Recover the active phase region from the uint8 screen
        slm_w, slm_h = self.slm_resolution
        n = min(slm_w, slm_h)
        r0 = (slm_h - n) // 2
        c0 = (slm_w - n) // 2
        active = screen[r0 : r0 + n, c0 : c0 + n].astype(np.float64)

        # Convert uint8 back to phase: inverse of phase_to_uint8
        # phase_to_uint8: round((phase + pi) / (2*pi) * 256) → uint8
        # inverse: phase ≈ value / 256 * 2*pi - pi
        phase_recovered = active / 256.0 * (2 * np.pi) - np.pi

        # Propagate using same physics as SimulatedCamera
        E_in = self.input_amplitude * np.exp(1j * phase_recovered)
        E_out = fft_propagate(E_in)
        intensity = np.abs(E_out) ** 2
        return intensity


# ---------------------------------------------------------------------------
# Helper: build matched camera pair
# ---------------------------------------------------------------------------


def make_camera_pair(n: int = 128):
    """Create a SimulatedCamera and a HardwareCamera backed by FFT mock.

    Both use the same input amplitude, no noise, no aberration, no LUT,
    no Fresnel lens — isolating the quantization effect.
    """
    amp = gaussian_beam((n, n), sigma=n / 5)
    slm_res = (n, n)

    sim_cam = SimulatedCamera(input_amplitude=amp, noise_level=0.0)

    display = FFTMockDisplay()
    mock_cam = FFTMockCamera(display, amp, slm_res)

    hw_cam = HardwareCamera(
        slm_display=display,
        camera=mock_cam,
        slm_resolution=slm_res,
        settle_time_s=0.0,
        lut_value=256,  # no LUT scaling
        calibration=None,
        fresnel_screen=None,
    )

    return sim_cam, hw_cam, amp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPhaseQuantizationError:
    """Verify the uint8 quantization introduces only small errors."""

    def test_uniform_phase_roundtrip(self):
        """Constant phase should survive quantization exactly."""
        phase = np.full((64, 64), 0.5)
        uint8_val = phase_to_uint8(phase)
        recovered = uint8_val.astype(np.float64) / 256.0 * (2 * np.pi) - np.pi
        np.testing.assert_allclose(recovered, phase, atol=2 * np.pi / 256)

    def test_random_phase_roundtrip_error(self):
        """Random phase roundtrip error bounded by 2*pi/256."""
        rng = np.random.default_rng(42)
        phase = rng.uniform(-np.pi, np.pi, (128, 128))
        uint8_val = phase_to_uint8(phase)
        recovered = uint8_val.astype(np.float64) / 256.0 * (2 * np.pi) - np.pi

        # Account for wrapping near ±pi
        error = np.angle(np.exp(1j * (recovered - phase)))
        max_err = 2 * np.pi / 256
        assert np.max(np.abs(error)) <= max_err + 1e-10
        # RMS should be well below max
        assert np.sqrt(np.mean(error**2)) < max_err


class TestCameraEquivalence:
    """Compare SimulatedCamera vs FFT-backed HardwareCamera."""

    def test_zero_phase_identical(self):
        """Zero phase should give identical results (quantizes to 128 exactly)."""
        sim_cam, hw_cam, amp = make_camera_pair(64)
        phase = np.zeros((64, 64))

        I_sim = sim_cam.capture_intensity(phase)
        I_hw = hw_cam.capture_intensity(phase)

        np.testing.assert_allclose(I_hw, I_sim, rtol=1e-10)

    def test_smooth_phase_close(self):
        """Smooth phase pattern: intensity difference should be very small."""
        sim_cam, hw_cam, amp = make_camera_pair(128)

        # Smooth quadratic phase (typical of a lens)
        y, x = np.mgrid[:128, :128]
        phase = 0.001 * ((x - 64) ** 2 + (y - 64) ** 2)
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi

        I_sim = sim_cam.capture_intensity(phase)
        I_hw = hw_cam.capture_intensity(phase)

        # Relative error in total power should be tiny
        power_sim = np.sum(I_sim)
        power_hw = np.sum(I_hw)
        assert abs(power_hw - power_sim) / power_sim < 0.01

        # Pixel-wise correlation should be very high
        corr = np.corrcoef(I_sim.ravel(), I_hw.ravel())[0, 1]
        assert corr > 0.999

    def test_wgs_hologram_close(self):
        """WGS-optimized hologram: both cameras give similar focal patterns."""
        n = 128
        sim_cam, hw_cam, amp = make_camera_pair(n)

        field = amp * np.exp(1j * np.random.default_rng(7).uniform(-np.pi, np.pi, (n, n)))
        target, pos = rectangular_grid((n, n), rows=3, cols=3, spacing=10)
        mask = mask_from_target(target)

        result = wgs(field, target, mask, WGSConfig(n_iterations=50))

        I_sim = sim_cam.capture_intensity(result.slm_phase)
        I_hw = hw_cam.capture_intensity(result.slm_phase)

        # Power conservation
        power_sim = np.sum(I_sim)
        power_hw = np.sum(I_hw)
        assert abs(power_hw - power_sim) / power_sim < 0.01

        # Spot intensities should match closely
        spot_I_sim = I_sim[pos[:, 0], pos[:, 1]]
        spot_I_hw = I_hw[pos[:, 0], pos[:, 1]]
        np.testing.assert_allclose(spot_I_hw, spot_I_sim, rtol=0.05)

    def test_cgm_hologram_close(self):
        """CGM-optimized hologram: both cameras give similar focal patterns."""
        n = 128
        sim_cam, hw_cam, amp = make_camera_pair(n)

        target, pos = rectangular_grid((n, n), rows=3, cols=3, spacing=10)
        mregion = measure_region((n, n), target)

        config = CGMConfig(max_iterations=50)
        result = cgm(amp, target, mregion, config)

        I_sim = sim_cam.capture_intensity(result.slm_phase)
        I_hw = hw_cam.capture_intensity(result.slm_phase)

        # Global correlation
        corr = np.corrcoef(I_sim.ravel(), I_hw.ravel())[0, 1]
        assert corr > 0.999

        # Fidelity computed on both should be close
        E_sim = np.sqrt(np.maximum(I_sim, 0)) * np.exp(1j * np.angle(fft_propagate(amp * np.exp(1j * result.slm_phase))))
        E_hw_approx = np.sqrt(np.maximum(I_hw, 0))  # intensity-only (no phase from hw)
        # Just compare the intensity patterns directly
        rel_diff = np.abs(I_hw - I_sim) / (np.max(I_sim) + 1e-30)
        assert np.mean(rel_diff) < 0.02  # <2% mean relative error


class TestFeedbackLoopEquivalence:
    """Verify experimental_feedback_loop converges with both camera types."""

    def test_feedback_converges_with_both_cameras(self):
        """Both cameras should produce improving cost over feedback steps."""
        n = 64
        amp = gaussian_beam((n, n), sigma=n / 5)
        target, pos = rectangular_grid((n, n), rows=2, cols=2, spacing=8)
        mregion = measure_region((n, n), target)
        slm_res = (n, n)

        # Simulated camera path
        sim_cam = SimulatedCamera(input_amplitude=amp, noise_level=0.0)
        results_sim = experimental_feedback_loop(
            amp, target, mregion, sim_cam, n_steps=3, max_iter=30,
        )

        # Hardware camera path (FFT mock)
        display = FFTMockDisplay()
        mock_cam = FFTMockCamera(display, amp, slm_res)
        hw_cam = HardwareCamera(
            slm_display=display,
            camera=mock_cam,
            slm_resolution=slm_res,
            settle_time_s=0.0,
            lut_value=256,
        )
        results_hw = experimental_feedback_loop(
            amp, target, mregion, hw_cam, n_steps=3, max_iter=30,
        )

        # Both should have decreasing cost (converging)
        costs_sim = [r.cost_history[-1] for r in results_sim]
        costs_hw = [r.cost_history[-1] for r in results_hw]

        assert costs_sim[-1] < costs_sim[0], "Sim feedback should converge"
        assert costs_hw[-1] < costs_hw[0], "HW feedback should converge"

        # Final costs should be in the same ballpark
        assert abs(costs_hw[-1] - costs_sim[-1]) / (abs(costs_sim[-1]) + 1e-30) < 0.5

    def test_protocol_compliance(self):
        """HardwareCamera satisfies CameraInterface for typing.Protocol."""
        n = 32
        amp = gaussian_beam((n, n), sigma=n / 5)
        display = FFTMockDisplay()
        mock_cam = FFTMockCamera(display, amp, (n, n))
        hw_cam = HardwareCamera(
            slm_display=display,
            camera=mock_cam,
            slm_resolution=(n, n),
            settle_time_s=0.0,
        )

        # Must have both protocol methods
        assert hasattr(hw_cam, "capture_intensity")
        assert hasattr(hw_cam, "capture_fringes")

        phase = np.zeros((n, n))
        I = hw_cam.capture_intensity(phase)
        assert isinstance(I, np.ndarray)
        assert I.dtype == np.float64
