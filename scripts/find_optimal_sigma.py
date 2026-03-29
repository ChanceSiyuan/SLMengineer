"""Find the optimal input beam sigma for any target pattern.

Sweeps sigma_mm and reports fidelity + efficiency for the given pattern,
helping identify the beam width that breaks through the fidelity wall.

Usage:
    # Flat top (pattern e):
    uv run python scripts/find_optimal_sigma.py --patterns e

    # Light sheet (pattern h):
    uv run python scripts/find_optimal_sigma.py --patterns h

    # Custom sigma range:
    uv run python scripts/find_optimal_sigma.py --patterns e --sigma-min 0.5 --sigma-max 3.0 --sigma-steps 10

    # Small grid (faster):
    uv run python scripts/find_optimal_sigma.py --patterns e --n-slm 256 --pad 1

    # All continuous patterns:
    uv run python scripts/find_optimal_sigma.py --patterns e,f,h
"""

import argparse
import time

import numpy as np

from slm.beams import gaussian_beam
from slm.cgm import CGMConfig
from slm.cgm_lbfgsb import cgm_lbfgsb
from slm.propagation import pad_field
from slm.targets import measure_region

# Import pattern catalog from generate_hologram (same directory)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_hologram import PATTERNS, PIXEL_PITCH_MM, output_center  # noqa: E402


def sweep_sigma(
    pattern_name: str,
    sigma_range: np.ndarray,
    n_slm: int,
    pad_factor: int,
    max_iter: int,
    gs_iters: int,
) -> list[dict]:
    """Sweep sigma for a single pattern and return metrics."""
    n_pad = n_slm * pad_factor
    center = output_center(n_pad)
    cfg = PATTERNS[pattern_name]

    results = []
    for sigma_mm in sigma_range:
        sigma_px = sigma_mm / PIXEL_PITCH_MM
        slm_amp = gaussian_beam((n_slm, n_slm), sigma=sigma_px, normalize=False)
        input_amp = pad_field(slm_amp, (n_pad, n_pad))

        target = cfg["target_fn"]((n_pad, n_pad), center)
        region = measure_region((n_pad, n_pad), target, margin=5)

        config = CGMConfig(
            max_iterations=max_iter,
            steepness=9,
            R=cfg["R_mrad"] * 1e-3,
            eta_min=0.0,
        )

        if gs_iters > 0:
            from dataclasses import replace

            from slm.hybrid import gs_seed_phase

            seed = gs_seed_phase(input_amp, target, gs_iters)
            config = replace(config, initial_phase=seed)

        t0 = time.time()
        result = cgm_lbfgsb(input_amp, target, region, config)
        dt = time.time() - t0

        row = {
            "sigma_mm": sigma_mm,
            "sigma_px": sigma_px,
            "infidelity": 1 - result.final_fidelity,
            "efficiency": result.final_efficiency,
            "iters": result.n_iterations,
            "time": dt,
        }
        results.append(row)
        print(
            f"  σ={sigma_mm:5.2f}mm ({sigma_px:6.1f}px)  "
            f"1-F={row['infidelity']:10.2e}  "
            f"η={row['efficiency']*100:6.2f}%  "
            f"({dt:.1f}s)",
            flush=True,
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Find optimal beam sigma for a target pattern")
    parser.add_argument(
        "--patterns", type=str, required=True,
        help="Comma-separated pattern keys (e.g. 'e', 'e,f,h', or 'all')",
    )
    parser.add_argument("--sigma-min", type=float, default=0.5, help="Min sigma (mm)")
    parser.add_argument("--sigma-max", type=float, default=3.0, help="Max sigma (mm)")
    parser.add_argument("--sigma-steps", type=int, default=6, help="Number of sigma values")
    parser.add_argument("--n-slm", type=int, default=256, help="SLM resolution")
    parser.add_argument("--pad", type=int, default=2, help="Pad factor")
    parser.add_argument("--iters", type=int, default=200, help="Max L-BFGS-B iterations")
    parser.add_argument("--gs-iters", type=int, default=0, help="GS seed iterations")
    args = parser.parse_args()

    sigma_range = np.linspace(args.sigma_min, args.sigma_max, args.sigma_steps)

    # Parse pattern names
    pattern_keys = list(PATTERNS.keys())
    if args.patterns == "all":
        names = pattern_keys
    else:
        lookup = {k[0]: k for k in pattern_keys}  # 'a' -> 'a) LG01', etc.
        names = []
        for p in args.patterns.split(","):
            p = p.strip()
            if p in lookup:
                names.append(lookup[p])
            else:
                match = [k for k in pattern_keys if p.lower() in k.lower()]
                if match:
                    names.append(match[0])
                else:
                    print(f"Unknown pattern: {p}")
                    return

    n_pad = args.n_slm * args.pad
    print(f"Grid: {args.n_slm}→{n_pad}, sigma: {args.sigma_min}–{args.sigma_max}mm ({args.sigma_steps} steps)")
    print(f"Iterations: {args.iters}, GS seed: {args.gs_iters}\n")

    all_results = {}
    for name in names:
        print(f"=== {name} ===")
        results = sweep_sigma(name, sigma_range, args.n_slm, args.pad, args.iters, args.gs_iters)
        all_results[name] = results

        best = min(results, key=lambda r: r["infidelity"])
        print(f"  Best: σ={best['sigma_mm']:.2f}mm → 1-F={best['infidelity']:.2e}, η={best['efficiency']*100:.2f}%\n")

    # Summary
    print("=" * 60)
    print("SUMMARY: Optimal sigma per pattern")
    print(f"{'Pattern':<25} {'σ_opt(mm)':>10} {'1-F':>12} {'η(%)':>8}")
    print("-" * 60)
    for name, results in all_results.items():
        best = min(results, key=lambda r: r["infidelity"])
        paper = PATTERNS[name].get("paper")
        tag = ""
        if paper and best["infidelity"] < paper["1-F"]:
            tag = " << beats paper!"
        print(f"{name:<25} {best['sigma_mm']:>10.2f} {best['infidelity']:>12.2e} {best['efficiency']*100:>8.2f}{tag}")

    if any(PATTERNS[n].get("paper") for n in names):
        print("\nPaper reference:")
        for name in names:
            p = PATTERNS[name].get("paper")
            if p:
                print(f"  {name}: 1-F={p['1-F']:.2e}, η={p['eta']}%")


if __name__ == "__main__":
    main()
