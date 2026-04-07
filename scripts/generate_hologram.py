"""Generate CGM holograms for various target beam profiles.

Computes phase-only holograms using conjugate gradient minimisation (CGM)
and reports quality metrics (fidelity, efficiency, intensity/phase error).
An intensity + phase mosaic plot is saved for every run.

Usage:
    uv run python scripts/generate_hologram.py                    # all patterns
    uv run python scripts/generate_hologram.py --patterns a,e,h   # selective
    uv run python scripts/generate_hologram.py --patterns h        # light sheet
    uv run python scripts/generate_hologram.py --list              # show patterns
    uv run python scripts/generate_hologram.py --fast              # 50 iterations
    uv run python scripts/generate_hologram.py --iters 500         # custom iters
    uv run python scripts/generate_hologram.py --no-plot           # metrics only
    uv run python scripts/generate_hologram.py --save out.png      # custom path
    uv run python scripts/generate_hologram.py --jax               # JAX backend
    uv run python scripts/generate_hologram.py --lbfgsb            # L-BFGS-B backend
    uv run python scripts/generate_hologram.py --gs-iters 100      # GS seeding
    uv run python scripts/generate_hologram.py --sigma 2.0         # override beam sigma
    uv run python scripts/generate_hologram.py --n-slm 256 --pad 1 # custom grid
"""

import argparse
import time
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from slm.beams import gaussian_beam
from slm.cgm import CGMConfig, cgm
from slm.cgm_lbfgsb import cgm_lbfgsb
from slm.hybrid import gs_seed_phase
from slm.propagation import pad_field
from slm.targets import (
    chicken_egg_pattern,
    gaussian_line,
    graphene_lattice,
    lg_mode,
    light_sheet,
    mask_from_target,
    measure_region,
    ring_lattice_vortex,
    square_lattice_vortex,
    top_hat,
)

# ============================================================================
# SLM hardware constants — change these to match your setup
# ============================================================================
PIXEL_PITCH_MM = 0.0125  # SLM pixel pitch (mm). 24 um for BNS P1920.
N_SLM = 1024            # SLM resolution (one side, square).
N_PAD = 2048            # Zero-padded output grid (>= N_SLM for resolution).

# ============================================================================
# Pattern configuration blocks
#
# Each block defines one target pattern. Edit parameters to customise.
#
# Keys explained:
#   target_fn   : callable(shape, center) -> complex array.  Builds the target
#                 field on the output grid.
#   sigma_mm    : Input Gaussian beam 1/e^2 radius on the SLM (mm).
#   R_mrad      : Quadratic initial-phase curvature (mrad/px^2).  Controls
#                 the envelope size of the initial output field.
#   ROI_px      : Region-of-interest diameter (px) for the mosaic plot crop.
#   eta_min     : Minimum efficiency floor (0-1).  The CGM cost function adds
#                 a penalty when efficiency drops below this value.  Higher
#                 values keep more light in the target region at the cost of
#                 reduced fidelity.  0 = no floor.
#   paper       : Reference values from Bowman et al. (None = no reference).
#   flat_region : (optional) Specification for flat-region intensity metrics.
#                 half_width : half-width of the flat region (px).
#                 margin     : pixels to exclude from each edge.
#                 axis       : "col" (horizontal) or "row" (vertical) slice
#                              through the pattern center.
# ============================================================================

PATTERNS = {
    # --- a) Laguerre-Gaussian LG01 mode ---
    # Vortex beam with orbital angular momentum l=1, radial order p=0.
    # w0 = beam waist of the LG mode (px in output plane).
    "a) LG01": {
        "target_fn": lambda s, c: lg_mode(s, ell=1, p=0, w0=10.0, center=c),
        "sigma_mm": 1.0,
        "R_mrad": 4.5,
        "ROI_px": 42,
        "eta_min": 0.05,
        "paper": {"1-F": 3.0e-6, "eta": 41.5, "eps_phi": 0.0003, "eps_nu": 0.005},
    },
    # --- b) 8x8 Square Lattice with vortex phase ---
    # rows, cols = lattice dimensions.  spacing = distance between sites (px).
    # peak_sigma = Gaussian width of each site (px).  ell = vortex charge.
    "b) Square Lattice": {
        "target_fn": lambda s, c: square_lattice_vortex(
            s, rows=8, cols=8, spacing=14.0, peak_sigma=3.0, ell=1, center=c,
        ),
        "sigma_mm": 1.2,
        "R_mrad": 4.5,
        "ROI_px": 124,
        "eta_min": 0.05,
        "paper": {"1-F": 1.6e-5, "eta": 10.6, "eps_phi": 0.00009, "eps_nu": 0.02},
    },
    # --- c) 12-site Ring Lattice with vortex phase ---
    # n_sites = number of spots on the ring.  ring_radius (px).
    "c) Ring Lattice": {
        "target_fn": lambda s, c: ring_lattice_vortex(
            s, n_sites=12, ring_radius=25.0, peak_sigma=3.0, ell=1, center=c,
        ),
        "sigma_mm": 1.2,
        "R_mrad": 3.9,
        "ROI_px": 71,
        "eta_min": 0.05,
        "paper": {"1-F": 1.5e-6, "eta": 24.6, "eps_phi": 0.00006, "eps_nu": 0.001},
    },
    # --- d) Graphene honeycomb lattice ---
    # rows, cols = unit cell count.  spacing = lattice constant (px).
    # Alternating 0/pi phase between sublattices.
    "d) Graphene": {
        "target_fn": lambda s, c: graphene_lattice(
            s, rows=4, cols=4, spacing=8.0, peak_sigma=2.5, center=c,
        ),
        "sigma_mm": 1.4,
        "R_mrad": 2.7,
        "ROI_px": 78,
        "eta_min": 0.05,
        "paper": {"1-F": 4.4e-4, "eta": 13.1, "eps_phi": 0.0003, "eps_nu": 0.010},
    },
    # --- e) Circular Flat Top ---
    # radius = disk radius (px).  Uniform intensity, flat phase inside.
    "e) Flat Top": {
        "target_fn": lambda s, c: top_hat(s, radius=25.0, center=c),
        "sigma_mm": 1.0,
        "R_mrad": 4.5,
        "ROI_px": 63,
        "eta_min": 0.05,
        "paper": {"1-F": 1.8e-4, "eta": 11.3, "eps_phi": 0.2, "eps_nu": 0.007},
        "flat_region": {"half_width": 20, "margin": 3, "axis": "row"},
    },
    # --- f) Gaussian Line with phase gradient ---
    # length = flat extent along the line (px).  width_sigma = Gaussian
    # perpendicular width (px).  phase_gradient = linear phase ramp (rad/px).
    "f) Gaussian Line": {
        "target_fn": lambda s, c: gaussian_line(
            s, length=30.0, width_sigma=5.0, phase_gradient=0.1, center=c,
        ),
        "sigma_mm": 1.4,
        "R_mrad": 2.9,
        "ROI_px": 45,
        "eta_min": 0.05,
        "paper": {"1-F": 1.4e-5, "eta": 20.4, "eps_phi": 0.01, "eps_nu": 0.002},
    },
    # --- g) Chicken & Egg (synthetic uncorrelated speckle) ---
    # radius = circular boundary (px).  Random smooth intensity + phase.
    "g) Chicken & Egg": {
        "target_fn": lambda s, c: chicken_egg_pattern(
            s, radius=50.0, center=c, rng=np.random.default_rng(12345),
        ),
        "sigma_mm": 1.6,
        "R_mrad": 4.5,
        "ROI_px": 128,
        "eta_min": 0.05,
        "paper": {"1-F": 7.1e-2, "eta": 2.0, "eps_phi": 1.3, "eps_nu": None},
    },
    # --- h) 1D Light Sheet (Rydberg beam shaping) ---
    # flat_width = full width of the uniform region (px).
    # gaussian_sigma = 1/e^2 Gaussian width perpendicular to line (px).
    # Uniform amplitude + flat phase inside the sheet.  Normalised to unit power.
    "h) Light Sheet": {
        "target_fn": lambda s, c: light_sheet(
            s, flat_width=50.0, gaussian_sigma=10.0, center=c,
        ),
        "sigma_mm": 1.0,
        "R_mrad": 4.5,
        "ROI_px": 80,
        "eta_min": 0.15,
        "paper": None,
        "flat_region": {"half_width": 25, "margin": 3, "axis": "col"},
    },
}


# ============================================================================
# Helpers
# ============================================================================


def mm_to_px(sigma_mm: float) -> float:
    return sigma_mm / PIXEL_PITCH_MM


def output_center(n_pad: int | None = None) -> tuple[float, float]:
    """Compute where the initial phase (D=-pi/2, theta=pi/4) directs light."""
    n = n_pad if n_pad is not None else N_PAD
    _defaults = CGMConfig()
    center = (n - 1) / 2.0
    shift = _defaults.D * np.cos(_defaults.theta) * n / (2 * np.pi)
    return (center + shift, center + shift)


def make_padded_input(sigma_px: float) -> np.ndarray:
    amp = gaussian_beam((N_SLM, N_SLM), sigma=sigma_px, normalize=False)
    return pad_field(amp, (N_PAD, N_PAD))


def flat_region_metrics(
    output_field: np.ndarray,
    center: tuple[float, float],
    half_width: int,
    margin: int,
    axis: str = "col",
) -> tuple[float, float, float]:
    """Compute I_rms(%), I_pk-pk(%), phi_rms(rad) along the flat direction."""
    ny, nx = output_field.shape
    cy = min(max(int(center[0]), 0), ny - 1)
    cx = min(max(int(center[1]), 0), nx - 1)
    hi = half_width - margin
    if axis == "col":
        sl = (cy, slice(max(0, cx - hi), min(nx, cx + hi)))
    else:
        sl = (slice(max(0, cy - hi), min(ny, cy + hi)), cx)

    I_flat = np.abs(output_field[sl]) ** 2
    I_mean = np.mean(I_flat)
    if I_mean == 0:
        return 0.0, 0.0, 0.0
    I_rms = float(np.std(I_flat) / I_mean * 100)
    I_pkpk = float((np.max(I_flat) - np.min(I_flat)) / I_mean * 100)

    phi_flat = np.angle(output_field[sl])
    phi_rms = float(np.std(phi_flat - np.mean(phi_flat)))
    return I_rms, I_pkpk, phi_rms


def resolve_pattern_keys(spec: str) -> list[str]:
    """Resolve 'a,e,h' or 'all' to list of full pattern names."""
    if spec.lower() == "all":
        return list(PATTERNS.keys())
    keys = []
    for token in spec.split(","):
        token = token.strip().lower()
        matched = [k for k in PATTERNS if k[0].lower() == token]
        if matched:
            keys.append(matched[0])
        else:
            print(f"Warning: no pattern matching '{token}'")
    return keys


# ============================================================================
# Core routines
# ============================================================================


def run_patterns(
    pattern_names: list[str],
    max_iter: int = 300,
    use_jax: bool = False,
    use_lbfgsb: bool = False,
    gs_iters: int = 0,
    sigma_override: float | None = None,
    eta_min_override: float | None = None,
    steepness_override: int | None = None,
) -> dict:
    """Run CGM on selected patterns, print metrics table, return results."""
    if use_lbfgsb:
        optimizer = cgm_lbfgsb
        print("(using L-BFGS-B)")
    elif use_jax:
        from slm.cgm_jax import cgm_jax as optimizer
        print("(using JAX autograd + scipy CG)")
    else:
        optimizer = cgm

    shape = (N_PAD, N_PAD)
    center = output_center()

    has_flat = any("flat_region" in PATTERNS[n] for n in pattern_names)
    hdr = f"{'Pattern':<22} {'1-F':>10} {'eta(%)':>8} {'eps_phi(%)':>10} {'eps_nu(%)':>10}"
    if has_flat:
        hdr += f" {'I_rms(%)':>10} {'I_pkpk(%)':>11} {'phi_rms':>8}"
    hdr += f" {'Iter':>5}"
    print(f"\n{hdr}")
    print("-" * len(hdr))

    results = {}
    inputs = {}
    targets = {}
    for name in pattern_names:
        cfg = PATTERNS[name]
        target = cfg["target_fn"](shape, center)
        targets[name] = target
        sigma_mm = sigma_override if sigma_override is not None else cfg["sigma_mm"]
        input_amp = make_padded_input(mm_to_px(sigma_mm))
        region = measure_region(shape, target, margin=5)

        eta_min = eta_min_override if eta_min_override is not None else cfg.get("eta_min", 0.0)
        steepness = steepness_override if steepness_override is not None else 9

        config = CGMConfig(
            max_iterations=max_iter,
            steepness=steepness,
            R=cfg["R_mrad"] * 1e-3,
            D=-np.pi / 2,
            theta=np.pi / 4,
            eta_min=eta_min,
        )

        if gs_iters > 0:
            seed = gs_seed_phase(input_amp, target, gs_iters)
            config = replace(config, initial_phase=seed)

        t0 = time.time()
        result = optimizer(input_amp, target, region, config)
        dt = time.time() - t0
        results[name] = result
        inputs[name] = input_amp

        one_minus_F = 1.0 - result.final_fidelity
        eta = result.final_efficiency * 100
        eps_phi = result.final_phase_error * 100
        eps_nu = result.final_non_uniformity * 100

        row = (
            f"{name:<22} {one_minus_F:>10.2e} {eta:>8.1f} "
            f"{eps_phi:>10.4f} {eps_nu:>10.4f}"
        )
        if has_flat:
            fr = cfg.get("flat_region")
            if fr:
                i_rms, i_pkpk, phi_rms = flat_region_metrics(
                    result.output_field, center,
                    fr["half_width"], fr["margin"], fr.get("axis", "col"),
                )
                row += f" {i_rms:>10.1f} {i_pkpk:>11.1f} {phi_rms:>8.4f}"
            else:
                row += " " * 32
        row += f" {result.n_iterations:>5}  ({dt:.1f}s)"
        print(row)

    # Paper reference
    has_ref = any(PATTERNS[n].get("paper") for n in pattern_names)
    if has_ref:
        print("\n--- Paper reference (Bowman et al.) ---")
        for name in pattern_names:
            pv = PATTERNS[name].get("paper")
            if not pv:
                continue
            nu = f"{pv['eps_nu']:>10.4f}" if pv.get("eps_nu") is not None else "         -"
            print(
                f"{name:<22} {pv['1-F']:>10.2e} {pv['eta']:>8.1f} "
                f"{pv['eps_phi']:>10.4f} {nu}"
            )

    return results, inputs, targets


def plot_mosaic(results: dict, inputs: dict, targets: dict,
               save_path: str = "hologram_output.png"):
    """Plot target, output intensity, input beam, output phase, convergence."""
    center = output_center()
    n = len(results)
    fig, axes = plt.subplots(n, 5, figsize=(20, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (name, result) in enumerate(results.items()):
        n_grid = result.output_field.shape[0]
        roi = min(PATTERNS[name]["ROI_px"], n_grid)
        r = roi // 2
        cy = min(max(int(center[0]), r), n_grid - r)
        cx = min(max(int(center[1]), r), n_grid - r)
        rs = slice(cy - r, cy + r)
        cs = slice(cx - r, cx + r)

        target = targets[name]

        # Target intensity
        axes[i, 0].imshow(np.abs(target[rs, cs]) ** 2, cmap="hot", aspect="equal")
        axes[i, 0].set_title(f"{name} — Target", fontsize=9)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Output intensity
        field = result.output_field[rs, cs]
        axes[i, 1].imshow(np.abs(field) ** 2, cmap="hot", aspect="equal")
        axes[i, 1].set_title("Output Intensity", fontsize=9)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        # Input beam
        input_amp = inputs[name]
        axes[i, 2].imshow(np.abs(input_amp) ** 2, cmap="viridis", aspect="equal")
        axes[i, 2].set_title("Input Beam (SLM)", fontsize=9)
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

        # Output phase
        phase = np.angle(field)
        amp = np.abs(field)
        threshold = 0.01 * np.max(amp) if np.max(amp) > 0 else 1.0
        phase_masked = np.where(amp > threshold, phase, np.nan)
        im = axes[i, 3].imshow(phase_masked, cmap="twilight", vmin=-np.pi, vmax=np.pi, aspect="equal")
        axes[i, 3].set_title("Output Phase", fontsize=9)
        axes[i, 3].set_xticks([])
        axes[i, 3].set_yticks([])
        cbar = fig.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
        cbar.set_ticks([-np.pi, 0, np.pi])
        cbar.set_ticklabels(["-\u03c0", "0", "\u03c0"])

        # Convergence
        if result.cost_history:
            axes[i, 4].semilogy(result.cost_history)
            axes[i, 4].set_title("Convergence", fontsize=9)
            axes[i, 4].set_xlabel("Eval")
            axes[i, 4].set_ylabel("Cost")
        else:
            axes[i, 4].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[i, 4].transAxes)
            axes[i, 4].set_title("Convergence", fontsize=9)

    fig.suptitle("CGM Hologram Output", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {save_path}")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate phase-only holograms via CGM and evaluate quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --patterns h              Light sheet only
  %(prog)s --patterns a,e,h          LG01, Flat Top, Light Sheet
  %(prog)s --patterns all             All 8 patterns
  %(prog)s --list                     Show available patterns
  %(prog)s --iters 500 --patterns a   LG01 with 500 iterations
  %(prog)s --fast --no-plot           Quick metrics, skip plot
""",
    )
    parser.add_argument(
        "--patterns", type=str, default="all",
        help="Comma-separated pattern keys (a-h) or 'all' (default: all)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available patterns and exit",
    )
    parser.add_argument(
        "--iters", type=int, default=None,
        help="Number of CGM iterations (default: 300, or 50 with --fast)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Quick mode: 50 iterations",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip the intensity/phase mosaic plot",
    )
    parser.add_argument(
        "--save", type=str, default="hologram_output.png",
        help="Save path for the mosaic plot (default: hologram_output.png)",
    )
    parser.add_argument(
        "--jax", action="store_true",
        help="Use JAX autograd + scipy CG backend",
    )
    parser.add_argument(
        "--lbfgsb", action="store_true",
        help="Use L-BFGS-B optimizer (recommended for top-hat/continuous patterns)",
    )
    parser.add_argument(
        "--gs-iters", type=int, default=0,
        help="GS seeding iterations before CGM (default: 0 = no seeding)",
    )
    parser.add_argument(
        "--sigma", type=float, default=None,
        help="Override input beam sigma (mm) for all patterns",
    )
    parser.add_argument(
        "--eta-min", type=float, default=None,
        help="Override efficiency floor for all patterns",
    )
    parser.add_argument(
        "--steepness", type=int, default=None,
        help="Override steepness parameter d (default: 9)",
    )
    parser.add_argument(
        "--n-slm", type=int, default=None,
        help="Override SLM resolution (default: 1024)",
    )
    parser.add_argument(
        "--pad", type=int, default=None,
        help="Override pad factor (default: 2)",
    )
    args = parser.parse_args()

    if args.list:
        print("Available patterns (use first letter with --patterns):\n")
        for key, cfg in PATTERNS.items():
            ref = cfg.get("paper")
            ref_str = f"paper: 1-F={ref['1-F']:.1e}, eta={ref['eta']}%" if ref else "no paper ref"
            print(f"  {key:<22} {ref_str}")
        return

    names = resolve_pattern_keys(args.patterns)
    if not names:
        parser.error("No patterns selected. Use --list to see available patterns.")

    if args.iters is not None:
        max_iter = args.iters
    elif args.fast:
        max_iter = 50
    else:
        max_iter = 300

    if args.fast:
        print("*** FAST MODE ***")

    # Apply grid overrides
    global N_SLM, N_PAD
    if args.n_slm is not None:
        N_SLM = args.n_slm
    if args.pad is not None:
        N_PAD = N_SLM * args.pad
    elif args.n_slm is not None:
        N_PAD = N_SLM * 2

    results, inputs, targets = run_patterns(
        names, max_iter=max_iter,
        use_jax=args.jax, use_lbfgsb=args.lbfgsb,
        gs_iters=args.gs_iters,
        sigma_override=args.sigma,
        eta_min_override=args.eta_min,
        steepness_override=args.steepness,
    )

    if not args.no_plot:
        plot_mosaic(results, inputs, targets, save_path=args.save)


if __name__ == "__main__":
    main()
