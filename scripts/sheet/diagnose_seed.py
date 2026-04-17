"""Quick diagnostic: evaluate each seed alone (zero CGM) and probe the
plateau structure.  Answers:

  1. What fidelity / efficiency does each seed give without CGM?
  2. What is the output intensity's shape -- is it top-hat-ish along u?
  3. Is the plateau at F=0.878 real, or is it an eta_min artefact?
"""
from __future__ import annotations

import numpy as np

from slm.cgm import CGMConfig, _initial_phase
from slm.generation import SLM_class
from slm.metrics import efficiency as _efficiency
from slm.metrics import fidelity as _fidelity
from slm.propagation import fft_propagate
from slm.targets import measure_region as _measure_region


SHEET_FLAT_WIDTH = 34
SHEET_GAUSSIAN_SIGMA = 2.5


def eval_seed(SLM, targetAmp, phi_seed: np.ndarray, label: str):
    region = _measure_region(targetAmp.shape, targetAmp, margin=5)
    E_out = fft_propagate(SLM.initGaussianAmp * np.exp(1j * phi_seed))
    F = float(_fidelity(E_out, targetAmp, region))
    eta = float(_efficiency(E_out, region))
    I = np.abs(E_out) ** 2
    # Slice across the centre.
    cy, cx = I.shape[0] // 2, I.shape[1] // 2
    row_u = I[cy]
    col_v = I[:, cx]
    fp_x = SLM.Focalpitchx
    fp_y = SLM.Focalpitchy
    half_u_px = int(SHEET_FLAT_WIDTH / 2)
    inner_u = row_u[cx - half_u_px + 2 : cx + half_u_px - 2]
    outside_u = row_u[: cx - 3 * half_u_px]
    print(
        f"\n=== {label} ==="
        f"\n  F={F:.6f}  eta={eta*100:.2f}%"
        f"\n  phi range: [{phi_seed.min():+.2f}, {phi_seed.max():+.2f}] rad  "
        f"std={phi_seed.std():.2f}"
        f"\n  along-u slice (row {cy}):"
        f"\n    inner {inner_u.shape[0]}px mean={inner_u.mean():.3e} "
        f"std={inner_u.std():.3e} flatness={inner_u.min()/inner_u.max():.3f}"
        f"\n    outside mean={outside_u.mean():.3e}  "
        f"inner/outside ratio={inner_u.mean()/max(outside_u.mean(),1e-30):.1e}"
    )
    # Perpendicular Gaussian width: find 1/e^2 half-width along v.
    col_v_norm = col_v / col_v.max()
    threshold = np.exp(-2.0)
    above = np.where(col_v_norm >= threshold)[0]
    if len(above) > 0:
        half_w_px = (above[-1] - above[0]) / 2.0
        print(
            f"  perpendicular-v 1/e^2 half-width: {half_w_px:.1f} px "
            f"({half_w_px*fp_y:.1f} um)  "
            f"[target gauss_sigma={SHEET_GAUSSIAN_SIGMA} px"
            f" = {SHEET_GAUSSIAN_SIGMA*fp_y:.1f} um]"
        )


def main():
    SLM = SLM_class()
    SLM.arraySizeBit = [10, 10]
    SLM.image_init(
        initGaussianPhase_user_defined=np.zeros((1024, 1024)), Plot=False,
    )
    print(
        f"Compute grid: ({SLM.ImgResY}, {SLM.ImgResX})  "
        f"focal pitch = {SLM.Focalpitchx:.3f} um/px"
    )
    diff = SLM.wavelength * SLM.focallength / (np.pi * SLM.beamwaist)
    print(f"Diffraction 1/e^2 = {diff:.2f} um;  "
          f"natural focal Gaussian = {diff:.2f} um (same thing)")
    print(f"w0 = {SLM.beamwaist} um")

    targetAmp = SLM.light_sheet_target(
        flat_width=SHEET_FLAT_WIDTH,
        gaussian_sigma=SHEET_GAUSSIAN_SIGMA,
        angle=0,
        edge_sigma=0.1,
    )

    # Target self-comparison
    region = _measure_region(targetAmp.shape, targetAmp, margin=5)
    F_target_self = float(_fidelity(targetAmp, targetAmp, region))
    print(f"\nF(target, target) = {F_target_self:.6f}  (sanity check)")

    # Seeds
    phi_bowman = _initial_phase(
        (int(SLM.ImgResY), int(SLM.ImgResX)),
        CGMConfig(R=0, D=-np.pi / 6, theta=np.pi / 4),
    )
    phi_stationary_1d = SLM.stationary_phase_sheet(
        flat_width=SHEET_FLAT_WIDTH, angle=0, center=None,
    )
    phi_stationary_2d = SLM.stationary_phase_sheet(
        flat_width=SHEET_FLAT_WIDTH,
        gaussian_sigma=SHEET_GAUSSIAN_SIGMA,
        angle=0,
        center=None,
    )
    phi_zero = np.zeros_like(phi_bowman)

    eval_seed(SLM, targetAmp, phi_zero, "zero seed (no phase)")
    eval_seed(SLM, targetAmp, phi_bowman, "Bowman seed")
    eval_seed(SLM, targetAmp, phi_stationary_1d, "stationary 1D (u only)")
    eval_seed(SLM, targetAmp, phi_stationary_2d, "stationary 2D (u + perp lens)")


if __name__ == "__main__":
    main()
