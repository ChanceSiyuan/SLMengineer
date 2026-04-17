"""Probe whether the CGM F=0.878 plateau at eta_min=0.3 is a real optimum
or an early-stop artifact.

Tests:
  1. Default convergence_threshold=1e-5 vs tightened 1e-9.
  2. Final cost value -- is the line search actually hitting zero?
  3. Stationary vs Bowman at the tight threshold.

If a tighter threshold pushes F past the plateau, the stationary seed
may unlock a regime Bowman cannot reach with its pathological local
minimum.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from slm.cgm import CGM_phase_generate, CGMConfig, _initial_phase
from slm.generation import SLM_class
from slm.metrics import efficiency as _efficiency
from slm.metrics import fidelity as _fidelity
from slm.propagation import fft_propagate
from slm.targets import measure_region as _measure_region


SHEET_FLAT_WIDTH = 34
SHEET_GAUSSIAN_SIGMA = 2.5


def run(SLM, targetAmp, init_phi, label, max_iters, eta_min, threshold):
    t0 = time.perf_counter()
    SLM_Phase = CGM_phase_generate(
        torch.tensor(SLM.initGaussianAmp),
        torch.from_numpy(init_phi),
        torch.from_numpy(targetAmp),
        max_iterations=max_iters,
        steepness=9,
        eta_min=eta_min,
        convergence_threshold=threshold,
        Plot=False,
    )
    wall = time.perf_counter() - t0
    phase_np = SLM_Phase.cpu().clone().numpy()
    phase_wrapped = np.angle(np.exp(1j * phase_np))
    region = _measure_region(targetAmp.shape, targetAmp, margin=5)
    E_out = fft_propagate(SLM.initGaussianAmp * np.exp(1j * phase_wrapped))
    F = float(_fidelity(E_out, targetAmp, region))
    eta = float(_efficiency(E_out, region))
    print(
        f"{label:<40} F={F:.6f}  eta={eta*100:.2f}%  "
        f"wall={wall:.1f}s  max_iters={max_iters}"
    )


def main():
    SLM = SLM_class()
    SLM.arraySizeBit = [10, 10]
    SLM.image_init(
        initGaussianPhase_user_defined=np.zeros((1024, 1024)), Plot=False,
    )
    targetAmp = SLM.light_sheet_target(
        flat_width=SHEET_FLAT_WIDTH,
        gaussian_sigma=SHEET_GAUSSIAN_SIGMA,
        angle=0,
        edge_sigma=0.1,
    )
    phi_bowman = _initial_phase(
        (1024, 1024),
        CGMConfig(R=0, D=-np.pi / 6, theta=np.pi / 4),
    )
    phi_stat2d = SLM.stationary_phase_sheet(
        flat_width=SHEET_FLAT_WIDTH,
        gaussian_sigma=SHEET_GAUSSIAN_SIGMA,
        angle=0,
    )

    print("=" * 72)
    print("eta_min=0.3  (original benchmark constraint)")
    print("=" * 72)
    for threshold in (1e-5, 1e-9, 1e-12):
        run(SLM, targetAmp, phi_bowman,
            f"Bowman (thresh={threshold:.0e})",
            max_iters=4000, eta_min=0.3, threshold=threshold)
        run(SLM, targetAmp, phi_stat2d,
            f"Stationary 2D (thresh={threshold:.0e})",
            max_iters=4000, eta_min=0.3, threshold=threshold)
        print()

    print("=" * 72)
    print("eta_min=0  (unconstrained)")
    print("=" * 72)
    for threshold in (1e-5, 1e-9, 1e-12):
        run(SLM, targetAmp, phi_bowman,
            f"Bowman (thresh={threshold:.0e})",
            max_iters=4000, eta_min=0.0, threshold=threshold)
        run(SLM, targetAmp, phi_stat2d,
            f"Stationary 2D (thresh={threshold:.0e})",
            max_iters=4000, eta_min=0.0, threshold=threshold)
        print()


if __name__ == "__main__":
    main()
