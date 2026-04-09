"""Local simulation test: does CGM reach high fidelity on an LG^0_1 donut?

Runs the same CGM configuration that ``scripts/testfile_lg.py`` uses on
hardware, but entirely in simulation (no SLM display, no camera, no
Fresnel post-hoc pipeline).  This isolates the **algorithmic** question
-- can the new torch/CUDA CGM reach paper-level fidelity on a continuous
phase-structured target -- from the **hardware** question that
``testfile_lg.py`` asks about calibration and Fresnel corrections.

The paper reports ``F = 0.999997`` (1-F = 3e-6) for LG^0_1 in
``references/top-hat.tex`` Table I.  This script asserts that our
implementation reaches ``F > 0.99`` (i.e. 1-F < 1e-2) on a reasonable
local grid, using the paper's analytical initial phase
``phi_0 = R*(p^2+q^2) + D*(p*cos theta + q*sin theta)`` with
``R=4.5e-3``, ``D=-pi/2``, ``theta=pi/4`` (the CGMConfig defaults).

Usage:

    uv run python scripts/test_lg_simulation.py           # fast: 1024x1024, 200 iter
    uv run python scripts/test_lg_simulation.py --full    # matches testfile_lg.py: 4096x4096
    uv run python scripts/test_lg_simulation.py --iter 500  # override iteration count

Output: a 6-panel PNG ``test_lg_simulation_<grid>.png`` showing input
amplitude, target intensity, target phase, output intensity, output
phase, and the cost history.  Exit code 0 on pass, 1 on fail.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slm.beams import gaussian_beam
from slm.cgm import cgm, CGMConfig
from slm.targets import lg_mode, mask_from_target, measure_region

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def log(msg=""):
    print(msg, flush=True)


def parse_args():
    full = "--full" in sys.argv
    max_iterations = 200
    if "--iter" in sys.argv:
        idx = sys.argv.index("--iter")
        max_iterations = int(sys.argv[idx + 1])
    return full, max_iterations


def run_lg_test(grid_size, sigma_px, lg_w0, max_iterations):
    """Build input/target and run CGM.  Returns (result, input_amp, target, region)."""
    shape = (grid_size, grid_size)

    # Input Gaussian amplitude (fills a fraction of the grid set by sigma_px)
    input_amp = gaussian_beam(shape, sigma=sigma_px, normalize=True)

    # LG^0_1 donut target (complex field with vortex phase)
    target = lg_mode(shape, ell=1, p=0, w0=lg_w0)
    region = measure_region(shape, target, margin=5)

    log(f"Target: LG^0_1 donut on {grid_size}x{grid_size} grid")
    log(f"  input Gaussian sigma = {sigma_px} px "
        f"(~{100 * sigma_px / grid_size:.1f}% of half-grid)")
    log(f"  LG waist w0 = {lg_w0} px")
    log(f"  |target| peak        = {np.abs(target).max():.4e}")
    log(f"  sum|target|^2        = {float(np.sum(np.abs(target) ** 2)):.6f} (normalised)")
    n_region = int(np.sum(region > 0))
    log(f"  measure region       = {n_region} px "
        f"({100 * n_region / (grid_size ** 2):.1f}% of grid)")

    # CGM with the paper's analytical initial phase (CGMConfig defaults R=4.5e-3,
    # D=-pi/2, theta=pi/4 are exactly the Bowman et al. LG^0_1 values from Table I)
    config = CGMConfig(
        max_iterations=max_iterations,
        steepness=9,
    )
    log("CGM config:")
    log(f"  max_iterations = {config.max_iterations}")
    log(f"  steepness      = {config.steepness}")
    log(f"  R              = {config.R:.2e}  (rad/px^2)")
    log(f"  D              = {config.D:.4f}  (= -pi/2)")
    log(f"  theta          = {config.theta:.4f}  (= pi/4)")
    log(f"  convergence    = {config.convergence_threshold:.1e}")

    log("")
    log("Running CGM...")
    t0 = time.perf_counter()
    result = cgm(input_amp, target, region, config)
    t_total = time.perf_counter() - t0
    per_iter = t_total / max(result.n_iterations, 1)
    log(f"  done: {result.n_iterations} iterations, "
        f"{t_total:.2f} s total, {per_iter * 1000:.1f} ms/iter")

    return result, input_amp, target, region


def print_metrics(result):
    log("")
    log("=== Final metrics ===")
    log(f"  Fidelity F              = {result.final_fidelity:.6f}")
    log(f"  1 - F                   = {1 - result.final_fidelity:.3e}")
    log(f"  Efficiency eta          = {result.final_efficiency:.4f}  "
        f"({100 * result.final_efficiency:.2f}%)")
    log(f"  Phase error eps_phi     = {result.final_phase_error:.3e}")
    log(f"  Non-uniformity eps_nu   = {result.final_non_uniformity:.3e}")
    log(f"  Cost first iter         = {result.cost_history[0]:.3e}")
    log(f"  Cost last iter          = {result.cost_history[-1]:.3e}")
    log(f"  Cost reduction factor   = {result.cost_history[0] / max(result.cost_history[-1], 1e-30):.2e}")

    log("")
    log("Paper reference (Bowman et al., top-hat.tex Table I, LG^0_1):")
    log("  1 - F = 3.0e-6  (F = 0.999997)")
    log("  eta   = 41.5%")


def save_visualisation(input_amp, target, region, result, path):
    """6-panel figure: input / target / target phase / output / output phase / cost."""
    E_out = result.output_field
    target_mask = mask_from_target(target)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(input_amp, cmap="viridis")
    axes[0, 0].set_title("Input amplitude |S|\n(Gaussian)")

    axes[0, 1].imshow(np.abs(target) ** 2, cmap="hot")
    axes[0, 1].set_title("Target intensity |tau|^2\n(LG^0_1 donut)")

    target_phase_masked = np.where(
        target_mask > 0, np.angle(target), np.nan
    )
    axes[0, 2].imshow(
        target_phase_masked, cmap="twilight", vmin=-np.pi, vmax=np.pi
    )
    axes[0, 2].set_title("Target phase arg(tau)\n(vortex, ell=1)")

    out_int = (np.abs(E_out) ** 2) * region
    axes[1, 0].imshow(out_int, cmap="hot")
    axes[1, 0].set_title(
        f"Output intensity |E_out|^2\n"
        f"F={result.final_fidelity:.4f}, eta={100 * result.final_efficiency:.1f}%"
    )

    out_phase_masked = np.where(
        target_mask > 0, np.angle(E_out), np.nan
    )
    axes[1, 1].imshow(out_phase_masked, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title(
        f"Output phase arg(E_out)\n"
        f"eps_phi={result.final_phase_error:.2e}"
    )

    axes[1, 2].plot(result.cost_history)
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_xlabel("Iteration")
    axes[1, 2].set_ylabel("Cost  C")
    axes[1, 2].set_title(f"CGM convergence ({result.n_iterations} iter)")
    axes[1, 2].grid(True, which="both", alpha=0.3)

    for ax in axes.flat[:5]:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def main():
    full, max_iterations = parse_args()

    if full:
        # Full-scale run matching scripts/testfile_lg.py (4096x4096)
        grid_size = 4096
        sigma_px = 400     # = beamwaist 5000um on 12.5um pitch
        lg_w0 = 100.0      # = 396 um on 4096 grid with focal pitch 3.96 um/px
    else:
        # Fast default for ~10 s local smoke test
        grid_size = 1024
        sigma_px = 100     # proportional
        lg_w0 = 25.0       # proportional

    log("=" * 72)
    log("CGM LG^0_1 local fidelity test")
    log("=" * 72)

    # Device info
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        device_str = (
            f"CUDA ({torch.cuda.get_device_name(0)})" if cuda_ok else "CPU"
        )
    except ImportError:
        device_str = "CPU (no torch)"
    log(f"Device: {device_str}")
    log("")

    result, input_amp, target, region = run_lg_test(
        grid_size, sigma_px, lg_w0, max_iterations,
    )
    print_metrics(result)

    # Pass/fail criterion -- any correct CGM implementation should reach
    # F > 0.99 on LG^0_1 within a few hundred iterations.  The paper's
    # result on 512x512 with comparable iteration count is F = 0.999997,
    # so we leave ~2 orders of magnitude of headroom.
    PASS_THRESHOLD = 0.99
    log("")
    if result.final_fidelity >= PASS_THRESHOLD:
        log(f"RESULT: PASS  (F = {result.final_fidelity:.4f} "
            f">= {PASS_THRESHOLD})")
        rc = 0
    else:
        log(f"RESULT: FAIL  (F = {result.final_fidelity:.4f} "
            f"<  {PASS_THRESHOLD})")
        rc = 1

    out_png = f"test_lg_simulation_{grid_size}.png"
    save_visualisation(input_amp, target, region, result, out_png)
    log("")
    log(f"Saved visualisation: {out_png}")

    sys.exit(rc)


if __name__ == "__main__":
    main()
