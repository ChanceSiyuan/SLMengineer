"""Reproduce all numerical results from Bowman et al. top-hat paper.

Usage:
    uv run python scripts/reproduce_bowman.py --table1    # Table 1 metrics
    uv run python scripts/reproduce_bowman.py --figure2   # Intensity/phase mosaic
    uv run python scripts/reproduce_bowman.py --figure3   # Gaussian line diagnostics
    uv run python scripts/reproduce_bowman.py --all       # Everything
    uv run python scripts/reproduce_bowman.py --fast      # Reduced grid/iterations
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from slm.beams import gaussian_beam
from slm.cgm import CGMConfig, cgm
from slm.propagation import pad_field
from slm.targets import (
    chicken_egg_pattern,
    gaussian_line,
    graphene_lattice,
    lg_mode,
    measure_region,
    ring_lattice_vortex,
    square_lattice_vortex,
    top_hat,
)

# --- Paper constants ---
PIXEL_PITCH_MM = 0.024  # 24 um
N_SLM = 256
N_PAD = 512


def mm_to_px(sigma_mm: float) -> float:
    return sigma_mm / PIXEL_PITCH_MM


def output_center(n_pad: int = N_PAD) -> tuple[float, float]:
    """Compute where the initial phase (D=-pi/2, theta=pi/4) directs light."""
    D, theta = -np.pi / 2, np.pi / 4
    center = (n_pad - 1) / 2.0
    # Linear phase ramp D*cos(theta) shifts output by D*cos(theta)*N/(2*pi)
    shift = D * np.cos(theta) * n_pad / (2 * np.pi)
    return (center + shift, center + shift)


def make_padded_input(sigma_px: float) -> np.ndarray:
    """Create Gaussian beam on 256x256 SLM, zero-pad to 512x512."""
    amp = gaussian_beam((N_SLM, N_SLM), sigma=sigma_px, normalize=False)
    return pad_field(amp, (N_PAD, N_PAD))


# --- Pattern definitions ---

def build_patterns(shape: tuple[int, int], center: tuple[float, float]) -> dict:
    """Build all 7 target patterns at the offset center."""
    return {
        "a) LG01": lg_mode(shape, ell=1, p=0, w0=10.0, center=center),
        "b) Square Lattice": square_lattice_vortex(
            shape, rows=8, cols=8, spacing=14.0, peak_sigma=3.0, ell=1, center=center,
        ),
        "c) Ring Lattice": ring_lattice_vortex(
            shape, n_sites=12, ring_radius=25.0, peak_sigma=3.0, ell=1, center=center,
        ),
        "d) Graphene": graphene_lattice(
            shape, rows=4, cols=4, spacing=8.0, peak_sigma=2.5, center=center,
        ),
        "e) Flat Top": top_hat(shape, radius=25.0, center=center),
        "f) Gaussian Line": gaussian_line(
            shape, length=30.0, width_sigma=5.0, phase_gradient=0.1, center=center,
        ),
        "g) Chicken & Egg": chicken_egg_pattern(
            shape, radius=50.0, center=center, rng=np.random.default_rng(12345),
        ),
    }


# Paper Table 1 parameters per pattern
PARAMS = {
    "a) LG01":           {"sigma_mm": 1.0, "R_mrad": 4.5, "ROI_px": 42},
    "b) Square Lattice":  {"sigma_mm": 1.2, "R_mrad": 4.5, "ROI_px": 124},
    "c) Ring Lattice":    {"sigma_mm": 1.2, "R_mrad": 3.9, "ROI_px": 71},
    "d) Graphene":        {"sigma_mm": 1.4, "R_mrad": 2.7, "ROI_px": 78},
    "e) Flat Top":        {"sigma_mm": 1.0, "R_mrad": 4.5, "ROI_px": 63},
    "f) Gaussian Line":   {"sigma_mm": 1.4, "R_mrad": 2.9, "ROI_px": 45},
    "g) Chicken & Egg":   {"sigma_mm": 1.6, "R_mrad": 4.5, "ROI_px": 128},
}

# Paper Table 1 reference values
PAPER_VALUES = {
    "a) LG01":           {"1-F": 3.0e-6, "eta": 41.5, "eps_phi": 0.0003, "eps_nu": 0.005},
    "b) Square Lattice":  {"1-F": 1.6e-5, "eta": 10.6, "eps_phi": 0.00009, "eps_nu": 0.02},
    "c) Ring Lattice":    {"1-F": 1.5e-6, "eta": 24.6, "eps_phi": 0.00006, "eps_nu": 0.001},
    "d) Graphene":        {"1-F": 4.4e-4, "eta": 13.1, "eps_phi": 0.0003, "eps_nu": 0.010},
    "e) Flat Top":        {"1-F": 1.8e-4, "eta": 11.3, "eps_phi": 0.2,    "eps_nu": 0.007},
    "f) Gaussian Line":   {"1-F": 1.4e-5, "eta": 20.4, "eps_phi": 0.01,   "eps_nu": 0.002},
    "g) Chicken & Egg":   {"1-F": 7.1e-2, "eta": 2.0,  "eps_phi": 1.3,    "eps_nu": None},
}


def run_table1(max_iter: int = 200) -> dict:
    """Reproduce Table 1: run CGM for all 7 patterns, print metrics."""
    shape = (N_PAD, N_PAD)
    center = output_center()
    patterns = build_patterns(shape, center)

    print(f"\n{'Pattern':<22} {'1-F':>10} {'η(%)':>8} {'ε_Φ(%)':>10} {'ε_ν(%)':>10} {'Iter':>5}")
    print("-" * 70)

    results = {}
    for name, target in patterns.items():
        p = PARAMS[name]
        sigma_px = mm_to_px(p["sigma_mm"])
        R = p["R_mrad"] * 1e-3

        input_amp = make_padded_input(sigma_px)
        region = measure_region(shape, target, margin=5)

        config = CGMConfig(
            max_iterations=max_iter,
            steepness=9,
            R=R,
            D=-np.pi / 2,
            theta=np.pi / 4,
        )

        t0 = time.time()
        result = cgm(input_amp, target, region, config)
        dt = time.time() - t0

        results[name] = result
        one_minus_F = 1.0 - result.final_fidelity
        eta = result.final_efficiency * 100
        eps_phi = result.final_phase_error * 100
        eps_nu = result.final_non_uniformity * 100

        print(
            f"{name:<22} {one_minus_F:>10.2e} {eta:>8.1f} "
            f"{eps_phi:>10.4f} {eps_nu:>10.4f} {result.n_iterations:>5}  "
            f"({dt:.1f}s)"
        )

    # Print paper reference values
    print("\n--- Paper reference values ---")
    print(f"{'Pattern':<22} {'1-F':>10} {'η(%)':>8} {'ε_Φ(%)':>10} {'ε_ν(%)':>10}")
    print("-" * 65)
    for name, pv in PAPER_VALUES.items():
        eps_nu_str = f"{pv['eps_nu']:>10.4f}" if pv["eps_nu"] is not None else "         -"
        print(
            f"{name:<22} {pv['1-F']:>10.2e} {pv['eta']:>8.1f} "
            f"{pv['eps_phi']:>10.4f} {eps_nu_str}"
        )

    return results


def plot_figure2(results: dict):
    """Plot intensity and phase for all patterns in their ROI (Figure 2)."""
    center = output_center()
    n_patterns = len(results)
    fig, axes = plt.subplots(n_patterns, 2, figsize=(6, 3 * n_patterns))

    for i, (name, result) in enumerate(results.items()):
        roi = PARAMS[name]["ROI_px"]
        r = roi // 2
        cy, cx = int(center[0]), int(center[1])
        row_slice = slice(max(0, cy - r), min(N_PAD, cy + r))
        col_slice = slice(max(0, cx - r), min(N_PAD, cx + r))

        field_roi = result.output_field[row_slice, col_slice]
        intensity = np.abs(field_roi) ** 2
        phase = np.angle(field_roi)

        axes[i, 0].imshow(intensity, cmap="hot", aspect="equal")
        axes[i, 0].set_title(f"{name} — Intensity", fontsize=9)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        axes[i, 1].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi, aspect="equal")
        axes[i, 1].set_title(f"{name} — Phase", fontsize=9)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

    fig.suptitle("Bowman et al. Figure 2: CGM Output Patterns", fontsize=12)
    fig.tight_layout()
    fig.savefig("bowman_figure2.png", dpi=150, bbox_inches="tight")
    print("\nSaved bowman_figure2.png")


def run_figure3(max_iter: int = 200, grid_size: int = 8):
    """Reproduce Figure 3: Gaussian Line diagnostics."""
    shape = (N_PAD, N_PAD)
    center = output_center()

    # Fixed target (Gaussian line)
    target = gaussian_line(
        shape, length=30.0, width_sigma=5.0, phase_gradient=0.1, center=center,
    )
    region = measure_region(shape, target, margin=5)

    # Paper uses sigma=1.5mm, R=2.3 mrad/px^2 for Figure 3a,b
    sigma_diag = mm_to_px(1.5)
    R_diag = 2.3e-3

    # --- Panel (a): Fidelity vs iteration for different d ---
    print("\nFigure 3a: Fidelity vs iteration for d=1,2,3,6...")
    fig3a_data = {}
    for d in [1, 2, 3, 6]:
        input_amp = make_padded_input(sigma_diag)
        config = CGMConfig(
            max_iterations=max_iter, steepness=d, R=R_diag,
            track_fidelity=True,
        )
        result = cgm(input_amp, target, region, config)
        fig3a_data[d] = result.fidelity_history
        print(f"  d={d}: final F={result.final_fidelity:.6f}, iters={result.n_iterations}")

    # --- Panel (b): Final fidelity vs d ---
    print("\nFigure 3b: Final fidelity vs d=1..9...")
    d_values = list(range(1, 10))
    final_fidelities = []
    times_per_iter = []
    for d in d_values:
        input_amp = make_padded_input(sigma_diag)
        config = CGMConfig(
            max_iterations=max_iter, steepness=d, R=R_diag,
        )
        t0 = time.time()
        result = cgm(input_amp, target, region, config)
        dt = time.time() - t0
        final_fidelities.append(result.final_fidelity)
        t_per_iter = dt / max(result.n_iterations, 1)
        times_per_iter.append(t_per_iter)
        print(f"  d={d}: F={result.final_fidelity:.6f}, t/iter={t_per_iter:.3f}s")

    # --- Panels (c,d): 2D parameter sweep ---
    print(f"\nFigure 3c,d: σ×R parameter sweep ({grid_size}x{grid_size} grid)...")
    sigma_range_mm = np.linspace(0.6, 2.4, grid_size)
    R_range_mrad = np.linspace(1.0, 6.0, grid_size)
    fidelity_grid = np.zeros((grid_size, grid_size))
    efficiency_grid = np.zeros((grid_size, grid_size))

    for i, sigma_mm in enumerate(sigma_range_mm):
        input_amp = make_padded_input(mm_to_px(sigma_mm))
        for j, R_mrad in enumerate(R_range_mrad):
            config = CGMConfig(
                max_iterations=max_iter, steepness=9, R=R_mrad * 1e-3,
            )
            result = cgm(input_amp, target, region, config)
            fidelity_grid[i, j] = result.final_fidelity
            efficiency_grid[i, j] = result.final_efficiency * 100
        print(f"  Row {i+1}/{grid_size} done (σ={sigma_mm:.1f}mm)")

    # --- Plot all 4 panels ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Fidelity vs iteration
    ax = axes[0, 0]
    for d, hist in fig3a_data.items():
        ax.plot(hist, label=f"d={d}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fidelity F")
    ax.set_title("(a) Fidelity evolution vs steepness d")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Final fidelity and time vs d
    ax = axes[0, 1]
    color1 = "tab:blue"
    ax.plot(d_values, final_fidelities, "o-", color=color1)
    ax.set_xlabel("Steepness d")
    ax.set_ylabel("Final Fidelity F", color=color1)
    ax.tick_params(axis="y", labelcolor=color1)
    ax.set_title("(b) Final fidelity and time per iteration vs d")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    color2 = "tab:red"
    ax2.plot(d_values, [t * 1000 for t in times_per_iter], "s--", color=color2)
    ax2.set_ylabel("Time per iteration (ms)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # (c) Fidelity vs (sigma, R)
    ax = axes[1, 0]
    extent = [R_range_mrad[0], R_range_mrad[-1], sigma_range_mm[0], sigma_range_mm[-1]]
    im = ax.imshow(fidelity_grid, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    ax.set_xlabel("R (mrad/px²)")
    ax.set_ylabel("σ (mm)")
    ax.set_title("(c) Fidelity F(σ, R)")
    plt.colorbar(im, ax=ax, label="Fidelity")

    # (d) Efficiency vs (sigma, R)
    ax = axes[1, 1]
    im = ax.imshow(efficiency_grid, aspect="auto", origin="lower", extent=extent, cmap="plasma")
    ax.set_xlabel("R (mrad/px²)")
    ax.set_ylabel("σ (mm)")
    ax.set_title("(d) Efficiency η(σ, R) (%)")
    plt.colorbar(im, ax=ax, label="Efficiency (%)")

    fig.suptitle("Bowman et al. Figure 3: Gaussian Line Diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig("bowman_figure3.png", dpi=150, bbox_inches="tight")
    print("\nSaved bowman_figure3.png")


def main():
    parser = argparse.ArgumentParser(description="Reproduce Bowman et al. results")
    parser.add_argument("--table1", action="store_true", help="Run Table 1")
    parser.add_argument("--figure2", action="store_true", help="Generate Figure 2")
    parser.add_argument("--figure3", action="store_true", help="Generate Figure 3")
    parser.add_argument("--all", action="store_true", help="Run everything")
    parser.add_argument("--fast", action="store_true", help="Reduced iterations/grid")
    args = parser.parse_args()

    if not any([args.table1, args.figure2, args.figure3, args.all]):
        args.all = True

    max_iter = 50 if args.fast else 300
    grid_size = 4 if args.fast else 8

    if args.fast:
        print("*** FAST MODE: reduced iterations and grid size ***")

    results = None
    if args.table1 or args.figure2 or args.all:
        results = run_table1(max_iter=max_iter)

    if args.figure2 or args.all:
        if results is not None:
            plot_figure2(results)

    if args.figure3 or args.all:
        run_figure3(max_iter=max_iter, grid_size=grid_size)


if __name__ == "__main__":
    main()
