"""Benchmark: stationary-phase vs Bowman initial guess for light-sheet CGM.

Runs the same CGM optimization (same target, same grid, same
optimizer hyperparams, same input Gaussian) twice with two different
initial phases, at a few iteration counts, and prints a comparison
table.  Sim-only, no hardware.

This is the Step-3 benchmark for the plan at
``~/.claude/plans/cheeky-hugging-bear.md``.  It quantifies the
warm-start speedup: if the stationary seed reaches baseline's final
fidelity in fewer iterations, the seed is working.

Usage::

    uv run python scripts/sheet/benchmark_stationary_vs_bowman.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from slm.cgm import CGM_phase_generate, CGMConfig, _initial_phase
from slm.generation import SLM_class
from slm.metrics import efficiency as _efficiency
from slm.metrics import fidelity as _fidelity
from slm.propagation import fft_propagate
from slm.targets import measure_region as _measure_region


sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# --- Fixed optimizer hyperparams (mirror testfile_sheet.py at its current state) ---
SHEET_FLAT_WIDTH = 34
SHEET_GAUSSIAN_SIGMA = 2.5
SHEET_ANGLE = 0
SHEET_EDGE_SIGMA = 0.1

CGM_STEEPNESS = 9
CGM_ETA_MIN = float(os.environ.get("BENCH_ETA_MIN", "0.3"))
CGM_R_BOWMAN = 0
CGM_D_BOWMAN = -np.pi / 6
CGM_THETA_BOWMAN = np.pi / 4

ITER_SCHEDULE = [100, 250, 500, 1000, 2000]
OUTPUT_PATH = "scripts/sheet/benchmark_stationary_vs_bowman.json"


def run_cgm(SLM, targetAmp, init_phi: np.ndarray, max_iters: int):
    """Run CGM on the given init_phi; return (F, eta, seconds)."""
    t0 = time.perf_counter()
    SLM_Phase = CGM_phase_generate(
        torch.tensor(SLM.initGaussianAmp),
        torch.from_numpy(init_phi),
        torch.from_numpy(targetAmp),
        max_iterations=max_iters,
        steepness=CGM_STEEPNESS,
        eta_min=CGM_ETA_MIN,
        Plot=False,
    )
    wall = time.perf_counter() - t0

    phase_np = SLM_Phase.cpu().clone().numpy()
    phase_wrapped = np.angle(np.exp(1j * phase_np))
    region = _measure_region(targetAmp.shape, targetAmp, margin=5)
    E_out = fft_propagate(SLM.initGaussianAmp * np.exp(1j * phase_wrapped))
    F = float(_fidelity(E_out, targetAmp, region))
    eta = float(_efficiency(E_out, region))
    return F, eta, wall


def main():
    SLM = SLM_class()
    SLM.arraySizeBit = [10, 10]
    SLM.image_init(
        initGaussianPhase_user_defined=np.zeros((1024, 1024)), Plot=False,
    )
    print(
        f"Compute grid: ({SLM.ImgResY}, {SLM.ImgResX})  "
        f"focal pitch = {SLM.Focalpitchx:.3f} um/px  "
        f"w0 = {SLM.beamwaist:.0f} um"
    )
    print(
        f"Diffraction 1/e^2 = "
        f"{SLM.wavelength * SLM.focallength / (np.pi * SLM.beamwaist):.2f} um"
    )

    targetAmp = SLM.light_sheet_target(
        flat_width=SHEET_FLAT_WIDTH,
        gaussian_sigma=SHEET_GAUSSIAN_SIGMA,
        angle=SHEET_ANGLE,
        edge_sigma=SHEET_EDGE_SIGMA,
    )
    print(
        f"Target: flat_width={SHEET_FLAT_WIDTH}px "
        f"({SHEET_FLAT_WIDTH * SLM.Focalpitchx:.1f} um), "
        f"gauss_sigma={SHEET_GAUSSIAN_SIGMA}px"
    )

    # --- Build both seeds once; iter loop only varies max_iterations. ---
    phi_bowman = _initial_phase(
        (int(SLM.ImgResY), int(SLM.ImgResX)),
        CGMConfig(R=CGM_R_BOWMAN, D=CGM_D_BOWMAN, theta=CGM_THETA_BOWMAN),
    )
    phi_stationary_1d = SLM.stationary_phase_sheet(
        flat_width=SHEET_FLAT_WIDTH, angle=SHEET_ANGLE, center=None,
    )
    phi_stationary_2d = SLM.stationary_phase_sheet(
        flat_width=SHEET_FLAT_WIDTH,
        gaussian_sigma=SHEET_GAUSSIAN_SIGMA,
        angle=SHEET_ANGLE,
        center=None,
    )
    print(
        f"Bowman seed:          min={phi_bowman.min():+.3f} max={phi_bowman.max():+.3f} "
        f"std={phi_bowman.std():.3f}"
    )
    print(
        f"Stationary 1D seed:   min={phi_stationary_1d.min():+.3f} "
        f"max={phi_stationary_1d.max():+.3f} std={phi_stationary_1d.std():.3f}"
    )
    print(
        f"Stationary 2D seed:   min={phi_stationary_2d.min():+.3f} "
        f"max={phi_stationary_2d.max():+.3f} std={phi_stationary_2d.std():.3f}"
    )

    # --- Run the sweep ---
    results = {"bowman": {}, "stationary_1d": {}, "stationary_2d": {}, "meta": {}}
    results["meta"] = {
        "compute_grid": [int(SLM.ImgResY), int(SLM.ImgResX)],
        "beamwaist_um": float(SLM.beamwaist),
        "wavelength_um": float(SLM.wavelength),
        "focallength_um": float(SLM.focallength),
        "focal_pitch_um_per_px": round(float(SLM.Focalpitchx), 4),
        "sheet_flat_width_px": SHEET_FLAT_WIDTH,
        "sheet_gaussian_sigma_px": SHEET_GAUSSIAN_SIGMA,
        "cgm_steepness": CGM_STEEPNESS,
        "cgm_eta_min": CGM_ETA_MIN,
        "cgm_R_bowman": CGM_R_BOWMAN,
        "cgm_D_bowman": CGM_D_BOWMAN,
        "cgm_theta_bowman": CGM_THETA_BOWMAN,
        "iter_schedule": ITER_SCHEDULE,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }

    print()
    header = f"{'iters':>6} | {'seed':<14} | {'F':>8} | {'1-F':>9} | {'eta%':>6} | {'wall_s':>7}"
    print(header)
    print("-" * len(header))

    seed_list = (
        ("bowman", phi_bowman),
        ("stationary_1d", phi_stationary_1d),
        ("stationary_2d", phi_stationary_2d),
    )
    for n_iters in ITER_SCHEDULE:
        for name, seed in seed_list:
            F, eta, wall = run_cgm(SLM, targetAmp, seed, n_iters)
            results[name][n_iters] = {
                "fidelity": F,
                "one_minus_fidelity": 1.0 - F,
                "efficiency": eta,
                "wall_seconds": wall,
            }
            print(
                f"{n_iters:>6} | {name:<14} | {F:>8.5f} | {1 - F:>9.2e} "
                f"| {eta * 100:>6.2f} | {wall:>7.2f}"
            )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")

    # --- Analysis summary ---
    print("\n=== Summary (fidelity by iter count) ===")
    print(f"{'iter':>6} | {'bowman':>10} | {'stat_1d':>10} | {'stat_2d':>10} |"
          f" {'d(2d-bow)':>10}")
    for n_iters in ITER_SCHEDULE:
        b = results["bowman"][n_iters]["fidelity"]
        s1 = results["stationary_1d"][n_iters]["fidelity"]
        s2 = results["stationary_2d"][n_iters]["fidelity"]
        print(
            f"{n_iters:>6} | {b:>10.6f} | {s1:>10.6f} | {s2:>10.6f} | "
            f"{s2 - b:>+10.6f}"
        )


if __name__ == "__main__":
    main()
