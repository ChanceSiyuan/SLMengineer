"""Tests for camera interfaces and Takeda phase retrieval."""

import numpy as np

from slm.beams import gaussian_beam
from slm.camera import SimulatedCamera, takeda_phase_retrieval


def test_simulated_camera_intensity():
    shape = (64, 64)
    amp = gaussian_beam(shape, sigma=15.0, normalize=False)
    cam = SimulatedCamera(amp, noise_level=0.0)
    phase = np.zeros(shape)
    img = cam.capture_intensity(phase)
    assert img.shape == shape
    assert np.all(img >= 0)
    assert np.max(img) > 0


def test_simulated_camera_fringes():
    shape = (64, 64)
    amp = gaussian_beam(shape, sigma=15.0, normalize=False)
    cam = SimulatedCamera(amp, noise_level=0.0, carrier_freq=(0.0, 0.1))
    phase = np.zeros(shape)
    fringes = cam.capture_fringes(phase)
    assert fringes.shape == shape
    assert np.max(fringes) > 0


def test_takeda_recovers_flat_phase():
    """Takeda on a flat-phase signal should return near-constant phase."""
    shape = (128, 128)
    # Use exact integer cycles to avoid sub-pixel shift artifacts
    carrier = (0.0, 13.0 / 128.0)
    ny, nx = shape
    y = np.arange(ny).reshape(-1, 1)
    x = np.arange(nx).reshape(1, -1)

    # Signal: uniform amplitude, zero phase
    E_sig = np.ones(shape, dtype=complex)
    E_ref = np.exp(2j * np.pi * (carrier[0] * y + carrier[1] * x))
    fringes = np.abs(E_sig + E_ref) ** 2

    recovered = takeda_phase_retrieval(fringes, carrier_freq=carrier)
    # Central region should have near-constant phase
    cy, cx, r = ny // 2, nx // 2, 15
    roi = recovered[cy - r : cy + r, cx - r : cx + r]
    roi -= np.mean(roi)
    assert np.std(roi) < 0.5  # ~lambda/12 accuracy for basic Takeda


def test_experimental_feedback_loop():
    from slm.feedback import experimental_feedback_loop
    from slm.targets import measure_region, top_hat

    shape = (64, 64)
    amp = gaussian_beam(shape, sigma=15.0, normalize=False)
    target = top_hat(shape, radius=8.0)
    region = measure_region(shape, target, margin=3)
    cam = SimulatedCamera(amp, noise_level=0.0)

    results = experimental_feedback_loop(
        amp, target, region, cam, n_steps=2, max_iter=20,
    )
    assert len(results) == 2
    assert results[-1].final_fidelity > 0
