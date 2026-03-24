"""Reproduce all numerical results from Kim et al. WGS paper.

Usage:
    uv run python scripts/reproduce_kim.py --figure2   # CGH calculation comparison
    uv run python scripts/reproduce_kim.py --figure3   # Adaptive correction (50x30)
    uv run python scripts/reproduce_kim.py --figure5   # Hex + disordered adaptive
    uv run python scripts/reproduce_kim.py --all       # Everything
    uv run python scripts/reproduce_kim.py --fast       # Fewer ensembles/iterations
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from slm.beams import initial_slm_field
from slm.feedback import FeedbackConfig, adaptive_feedback_loop
from slm.gs import gs
from slm.metrics import uniformity
from slm.targets import (
    disordered_array,
    hexagonal_grid,
    mask_from_target,
    rectangular_grid,
)
from slm.transforms import generate_aberration
from slm.wgs import WGSConfig, wgs


SHAPE = (512, 512)
SIGMA = SHAPE[0] / 4  # Gaussian beam fills SLM
ABERRATION_COEFFS = {4: 0.8, 5: 0.5, 7: 0.3, 8: 0.2}


def run_figure2(n_iterations: int = 200):
    """Figure 2: CGH calculation — GS vs WGS vs Phase-Fixed WGS."""
    print("\n=== Figure 2: CGH Calculation ===\n")

    # --- (a) 50x30 rectangular grid comparison ---
    print("(a) 50x30 rectangular grid: 3-algorithm comparison...")
    target, positions = rectangular_grid(SHAPE, rows=50, cols=30, spacing=5)
    mask = mask_from_target(target)
    rng = np.random.default_rng(42)
    L0 = initial_slm_field(SHAPE, sigma=SIGMA, rng=rng)

    t0 = time.time()
    gs_result = gs(L0, target, n_iterations=n_iterations)
    print(f"  GS:           uniformity={gs_result.uniformity_history[-1]:.6f}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    wgs_result = wgs(L0, target, mask, WGSConfig(n_iterations=n_iterations))
    print(f"  WGS:          uniformity={wgs_result.uniformity_history[-1]:.6f}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    pf_result = wgs(L0, target, mask, WGSConfig(n_iterations=n_iterations, phase_fix_iteration=12))
    eff_at_12 = pf_result.efficiency_history[12] if len(pf_result.efficiency_history) > 12 else 0
    print(f"  Phase-Fixed:  uniformity={pf_result.uniformity_history[-1]:.6f}  "
          f"eff@iter12={eff_at_12:.3f}  ({time.time()-t0:.1f}s)")

    # --- (e) Hexagonal lattice ---
    print("\n(e) Hexagonal lattice (720 spots)...")
    hex_target, hex_pos = hexagonal_grid(SHAPE, rows=30, cols=24, spacing=5)
    hex_mask = mask_from_target(hex_target)
    L0_hex = initial_slm_field(SHAPE, sigma=SIGMA, rng=np.random.default_rng(42))
    n_hex = len(hex_pos)
    print(f"  Generated {n_hex} spots")

    wgs_hex = wgs(L0_hex, hex_target, hex_mask, WGSConfig(n_iterations=n_iterations))
    pf_hex = wgs(L0_hex, hex_target, hex_mask,
                 WGSConfig(n_iterations=n_iterations, phase_fix_iteration=12))
    print(f"  WGS:         uniformity={wgs_hex.uniformity_history[-1]:.6f}")
    print(f"  Phase-Fixed: uniformity={pf_hex.uniformity_history[-1]:.6f}")

    # --- (f) Disordered array ---
    print("\n(f) Disordered array (819 spots)...")
    dis_target, dis_pos = disordered_array(
        SHAPE, n_spots=819, extent=120, min_distance=3.0, rng=np.random.default_rng(42),
    )
    dis_mask = mask_from_target(dis_target)
    L0_dis = initial_slm_field(SHAPE, sigma=SIGMA, rng=np.random.default_rng(42))
    print(f"  Generated {len(dis_pos)} spots")

    wgs_dis = wgs(L0_dis, dis_target, dis_mask, WGSConfig(n_iterations=n_iterations))
    pf_dis = wgs(L0_dis, dis_target, dis_mask,
                 WGSConfig(n_iterations=n_iterations, phase_fix_iteration=12))
    print(f"  WGS:         uniformity={wgs_dis.uniformity_history[-1]:.6f}")
    print(f"  Phase-Fixed: uniformity={pf_dis.uniformity_history[-1]:.6f}")

    # --- Plot ---
    fig = plt.figure(figsize=(16, 14))
    gs_fig = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

    # (a) Top: modulation efficiency
    ax_eff = fig.add_subplot(gs_fig[0, 0:2])
    ax_eff.plot(pf_result.efficiency_history, "C0", label="Phase-Fixed")
    ax_eff.plot(wgs_result.efficiency_history, "C1", alpha=0.7, label="WGS")
    ax_eff.axvline(12, color="gray", linestyle="--", alpha=0.5, label="Phase fix (N=12)")
    ax_eff.set_ylabel("Modulation Efficiency")
    ax_eff.legend(fontsize=8)
    ax_eff.set_title("(a) 50×30 Grid: Efficiency")
    ax_eff.grid(True, alpha=0.3)

    # (a) Bottom: non-uniformity
    ax_uni = fig.add_subplot(gs_fig[1, 0:2])
    ax_uni.semilogy(gs_result.uniformity_history, "k", alpha=0.5, label="GS")
    ax_uni.semilogy(wgs_result.uniformity_history, "C1", label="WGS")
    ax_uni.semilogy(pf_result.uniformity_history, "C0", label="Phase-Fixed")
    ax_uni.axhline(0.005, color="r", linestyle=":", alpha=0.5, label="0.5% target")
    ax_uni.set_xlabel("Iteration")
    ax_uni.set_ylabel("Non-uniformity (std/mean)")
    ax_uni.legend(fontsize=8)
    ax_uni.set_title("(a) 50×30 Grid: Non-uniformity")
    ax_uni.grid(True, alpha=0.3)

    # (b) CGH phase
    ax_cgh = fig.add_subplot(gs_fig[0, 2])
    ax_cgh.imshow(pf_result.slm_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax_cgh.set_title("(b) CGH Phase Φ(x)")
    ax_cgh.set_xticks([])
    ax_cgh.set_yticks([])

    # (c) Focal-plane intensity
    ax_focal = fig.add_subplot(gs_fig[0, 3])
    focal_I = np.abs(pf_result.focal_field) ** 2
    ax_focal.imshow(focal_I, cmap="hot", vmax=np.percentile(focal_I[focal_I > 0], 99))
    ax_focal.set_title("(c) Expected LOFA")
    ax_focal.set_xticks([])
    ax_focal.set_yticks([])

    # (d) Per-spot phase & amplitude at u_30 and u_100
    ax_phase = fig.add_subplot(gs_fig[1, 2])
    ax_amp = fig.add_subplot(gs_fig[1, 3])
    if len(pf_result.spot_phase_history) > 0:
        n_spots_total = len(pf_result.spot_phase_history[0])
        idx_30 = min(29, n_spots_total - 1)
        idx_100 = min(99, n_spots_total - 1)
        phases_30 = [h[idx_30] for h in pf_result.spot_phase_history]
        phases_100 = [h[idx_100] for h in pf_result.spot_phase_history]
        amps_30 = [h[idx_30] for h in pf_result.spot_amplitude_history]
        amps_100 = [h[idx_100] for h in pf_result.spot_amplitude_history]

        ax_phase.plot(phases_30, "C0", label=f"u_{idx_30+1}")
        ax_phase.plot(phases_100, "C3", label=f"u_{idx_100+1}")
        ax_phase.set_ylabel("Phase (rad)")
        ax_phase.set_xlabel("Iteration")
        ax_phase.set_title("(d) Spot Phase")
        ax_phase.legend(fontsize=8)

        ax_amp.plot(amps_30, "C0", label=f"u_{idx_30+1}")
        ax_amp.plot(amps_100, "C3", label=f"u_{idx_100+1}")
        ax_amp.set_ylabel("Amplitude")
        ax_amp.set_xlabel("Iteration")
        ax_amp.set_title("(d) Spot Amplitude")
        ax_amp.legend(fontsize=8)

    # (e) Hexagonal comparison
    ax_hex = fig.add_subplot(gs_fig[2, 0])
    ax_hex.semilogy(wgs_hex.uniformity_history, "C1", label="WGS")
    ax_hex.semilogy(pf_hex.uniformity_history, "C0", label="Phase-Fixed")
    ax_hex.set_xlabel("Iteration")
    ax_hex.set_ylabel("Non-uniformity")
    ax_hex.set_title(f"(e) Hexagonal ({n_hex} spots)")
    ax_hex.legend(fontsize=8)
    ax_hex.grid(True, alpha=0.3)

    # (e) inset: hex focal intensity
    ax_hex_inset = fig.add_subplot(gs_fig[2, 1])
    hex_I = np.abs(pf_hex.focal_field) ** 2
    ax_hex_inset.imshow(hex_I, cmap="hot", vmax=np.percentile(hex_I[hex_I > 0], 99))
    ax_hex_inset.set_title("Hex LOFA")
    ax_hex_inset.set_xticks([])
    ax_hex_inset.set_yticks([])

    # (f) Disordered comparison
    ax_dis = fig.add_subplot(gs_fig[2, 2])
    ax_dis.semilogy(wgs_dis.uniformity_history, "C1", label="WGS")
    ax_dis.semilogy(pf_dis.uniformity_history, "C0", label="Phase-Fixed")
    ax_dis.set_xlabel("Iteration")
    ax_dis.set_ylabel("Non-uniformity")
    ax_dis.set_title(f"(f) Disordered ({len(dis_pos)} spots)")
    ax_dis.legend(fontsize=8)
    ax_dis.grid(True, alpha=0.3)

    # (f) inset: disordered focal intensity
    ax_dis_inset = fig.add_subplot(gs_fig[2, 3])
    dis_I = np.abs(pf_dis.focal_field) ** 2
    ax_dis_inset.imshow(dis_I, cmap="hot", vmax=np.percentile(dis_I[dis_I > 0], 99))
    ax_dis_inset.set_title("Disordered LOFA")
    ax_dis_inset.set_xticks([])
    ax_dis_inset.set_yticks([])

    fig.suptitle("Kim et al. Figure 2: CGH Calculation", fontsize=14)
    fig.savefig("kim_figure2.png", dpi=150, bbox_inches="tight")
    print("\nSaved kim_figure2.png")


def run_adaptive_correction(
    target, positions, mask, shape, aberration, n_correction_steps, n_ensembles,
    n_inner_iterations, label,
):
    """Run ensemble-averaged adaptive correction for both WGS and Phase-Fixed."""
    results = {}
    for method_name, phase_fix_iter in [("WGS", None), ("Phase-Fixed", 12)]:
        ensemble_uniformities = []
        last_results = None
        for seed in range(n_ensembles):
            L0 = initial_slm_field(shape, sigma=shape[0] / 4, rng=np.random.default_rng(seed))
            config = FeedbackConfig(
                n_correction_steps=n_correction_steps,
                inner_iterations=n_inner_iterations,
                phase_fix_iteration=phase_fix_iter if phase_fix_iter else 30,
                noise_level=0.02,
            )
            # For plain WGS (no phase fix), set phase_fix_iteration very high
            if phase_fix_iter is None:
                config.phase_fix_iteration = 9999

            step_results = adaptive_feedback_loop(
                L0, target, mask, positions, config,
                aberration_phase=aberration,
                rng=np.random.default_rng(1000 + seed),
            )
            step_uniformities = [r.uniformity_history[-1] for r in step_results]
            ensemble_uniformities.append(step_uniformities)
            if seed == 0:
                last_results = step_results

        ensemble_uniformities = np.array(ensemble_uniformities)
        mean_uni = np.mean(ensemble_uniformities, axis=0)
        std_uni = np.std(ensemble_uniformities, axis=0)
        results[method_name] = {
            "mean": mean_uni, "std": std_uni,
            "last_results": last_results,
        }
        print(f"  {method_name}: final={mean_uni[-1]:.4f} ± {std_uni[-1]:.4f}")

    return results


def run_figure3(n_inner_iterations: int = 100, n_ensembles: int = 8):
    """Figure 3: Adaptive CGH correction for 50x30 grid."""
    print("\n=== Figure 3: Adaptive CGH Correction (50x30) ===\n")

    target, positions = rectangular_grid(SHAPE, rows=50, cols=30, spacing=5)
    mask = mask_from_target(target)

    # Aberration tuned to produce ~20% initial non-uniformity
    aberration = generate_aberration(SHAPE, ABERRATION_COEFFS)

    results = run_adaptive_correction(
        target, positions, mask, SHAPE, aberration,
        n_correction_steps=3, n_ensembles=n_ensembles,
        n_inner_iterations=n_inner_iterations, label="50x30",
    )

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Non-uniformity vs correction step
    ax = axes[0, 0]
    steps = np.arange(1, 4)
    for method_name, color in [("WGS", "C2"), ("Phase-Fixed", "C1")]:
        r = results[method_name]
        ax.errorbar(steps, r["mean"], yerr=r["std"], fmt="o-", color=color,
                    capsize=4, label=method_name)
    ax.set_xlabel("Correction Step")
    ax.set_ylabel("Non-uniformity (std/mean)")
    ax.set_title("(a) Adaptive Correction Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) CGH correction difference
    ax = axes[0, 1]
    pf_results = results["Phase-Fixed"]["last_results"]
    if pf_results and len(pf_results) >= 2:
        diff = pf_results[-1].slm_phase - pf_results[0].slm_phase
        ax.imshow(diff, cmap="RdBu_r")
        ax.set_title("(b) CGH Correction Φ(3) − Φ(0)")
    ax.set_xticks([])
    ax.set_yticks([])

    # (c) Simulated focal-plane image
    ax = axes[1, 0]
    if pf_results:
        focal_I = np.abs(pf_results[-1].focal_field) ** 2
        ax.imshow(focal_I, cmap="hot", vmax=np.percentile(focal_I[focal_I > 0], 99))
        ax.set_title("(c) Corrected LOFA")
    ax.set_xticks([])
    ax.set_yticks([])

    # (d) Intensity histogram
    ax = axes[1, 1]
    if pf_results:
        initial_I = np.array([np.abs(pf_results[0].focal_field[r, c]) ** 2 for r, c in positions])
        corrected_I = np.array([np.abs(pf_results[-1].focal_field[r, c]) ** 2 for r, c in positions])
        ax.hist(initial_I / np.mean(initial_I), bins=30, alpha=0.5, label=f"Initial ({uniformity(initial_I):.1%})", color="C1")
        ax.hist(corrected_I / np.mean(corrected_I), bins=30, alpha=0.5, label=f"Corrected ({uniformity(corrected_I):.1%})", color="C3")
        ax.set_xlabel("Normalized Intensity")
        ax.set_ylabel("Count")
        ax.set_title("(d) Intensity Histogram")
        ax.legend(fontsize=8)

    fig.suptitle("Kim et al. Figure 3: Adaptive CGH Correction (50×30)", fontsize=13)
    fig.tight_layout()
    fig.savefig("kim_figure3.png", dpi=150, bbox_inches="tight")
    print("\nSaved kim_figure3.png")


def run_figure5(n_inner_iterations: int = 100, n_ensembles: int = 8):
    """Figure 5: Adaptive correction for hex and disordered geometries."""
    print("\n=== Figure 5: Hex & Disordered Adaptive Correction ===\n")

    aberration = generate_aberration(SHAPE, ABERRATION_COEFFS)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for row, (geom_name, target, positions) in enumerate([
        ("Hexagonal (720)", *hexagonal_grid(SHAPE, rows=30, cols=24, spacing=5)),
        ("Disordered (819)", *disordered_array(SHAPE, 819, extent=120, min_distance=3.0,
                                                rng=np.random.default_rng(42))),
    ]):
        mask = mask_from_target(target)
        n_spots = len(positions)
        print(f"\n{geom_name}: {n_spots} spots")

        results = run_adaptive_correction(
            target, positions, mask, SHAPE, aberration,
            n_correction_steps=5, n_ensembles=n_ensembles,
            n_inner_iterations=n_inner_iterations, label=geom_name,
        )

        # (a/d) Non-uniformity vs correction step
        ax = axes[row, 0]
        steps = np.arange(1, 6)
        for method_name, color in [("WGS", "C2"), ("Phase-Fixed", "C1")]:
            r = results[method_name]
            ax.errorbar(steps, r["mean"], yerr=r["std"], fmt="o-", color=color,
                        capsize=4, label=method_name)
        ax.set_xlabel("Correction Step")
        ax.set_ylabel("Non-uniformity")
        ax.set_title(f"({['a','d'][row]}) {geom_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (b/e) CGH and correction
        ax = axes[row, 1]
        pf_results = results["Phase-Fixed"]["last_results"]
        if pf_results and len(pf_results) >= 2:
            diff = pf_results[-1].slm_phase - pf_results[0].slm_phase
            ax.imshow(diff, cmap="RdBu_r")
            ax.set_title(f"({['b','e'][row]}) CGH Correction")
        ax.set_xticks([])
        ax.set_yticks([])

        # (c/f) Focal-plane image
        ax = axes[row, 2]
        if pf_results:
            focal_I = np.abs(pf_results[-1].focal_field) ** 2
            ax.imshow(focal_I, cmap="hot", vmax=np.percentile(focal_I[focal_I > 0], 99))
            ax.set_title(f"({['c','f'][row]}) Corrected LOFA")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Kim et al. Figure 5: Hex & Disordered Adaptive Correction", fontsize=13)
    fig.tight_layout()
    fig.savefig("kim_figure5.png", dpi=150, bbox_inches="tight")
    print("\nSaved kim_figure5.png")


def main():
    parser = argparse.ArgumentParser(description="Reproduce Kim et al. WGS results")
    parser.add_argument("--figure2", action="store_true")
    parser.add_argument("--figure3", action="store_true")
    parser.add_argument("--figure5", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Fewer ensembles/iterations")
    args = parser.parse_args()

    if not any([args.figure2, args.figure3, args.figure5, args.all]):
        args.all = True

    n_iter = 50 if args.fast else 200
    n_inner = 30 if args.fast else 100
    n_ens = 2 if args.fast else 8

    if args.fast:
        print("*** FAST MODE ***")

    if args.figure2 or args.all:
        run_figure2(n_iterations=n_iter)

    if args.figure3 or args.all:
        run_figure3(n_inner_iterations=n_inner, n_ensembles=n_ens)

    if args.figure5 or args.all:
        run_figure5(n_inner_iterations=n_inner, n_ensembles=n_ens)


if __name__ == "__main__":
    main()
