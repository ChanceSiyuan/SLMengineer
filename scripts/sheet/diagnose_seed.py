"""Diagnose why the stationary-phase seed produces a blob, not a line.

Issue #20. Runs entirely on the Linux side — no hardware, no CGM.

Tests hypotheses 1 (seed-math bug) and 2 (grid too coarse) by:

1. Rebuilding the light-sheet seed phase at N=1024 (production) and
   N=4096 (4× finer) for the *same physical* target (flat-top 142 μm
   full width, perp 1/e² = 45 μm, centred −475 μm off zero-order).
2. Propagating ``|fft_propagate(S · exp(iφ_seed))|²`` at each grid
   size with the same normalisation as ``slm.cgm``.
3. Saving a 2-row × 3-column diagnostic figure: 2D focal-plane
   intensity + along-line and perpendicular profile cuts, for each N.

Does NOT apply the post-hoc Fresnel lens (``fresnel_sd``) or
calibration BMP — the whole point is to isolate the seed's own
prediction.

Usage::

    uv run python scripts/sheet/diagnose_seed.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slm.generation import SLM_class
from slm.propagation import fft_propagate

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = "docs/sweep_sheet"
FIG_PATH   = f"{OUTPUT_DIR}/diagnose_seed_grid.png"
JSON_PATH  = f"{OUTPUT_DIR}/diagnose_seed_grid.json"
FIG_FSD    = f"{OUTPUT_DIR}/diagnose_seed_fresnel.png"
JSON_FSD   = f"{OUTPUT_DIR}/diagnose_seed_fresnel.json"


def build_case(array_size_bit: int, fresnel_sd_um: float = 0.0) -> dict:
    """Build the seed phase + propagate it at grid size 2**array_size_bit.

    Physical target is held fixed; only the grid density (and optionally
    the post-hoc Fresnel lens distance) changes.  At N=1024 focal pitch
    is 15.83 um/px; at N=4096 focal pitch is 3.96 um/px, so the flat-top
    (142 um) is 9 px at N=1024 and 36 px at N=4096.  Perp 1/e^2 amplitude
    radius is sigma*sqrt(2) = 45 um in both.

    ``fresnel_sd_um`` > 0 adds the same post-hoc spherical Fresnel lens
    that production applies in ``testfile_sheet.py`` (``fresnel_lens_phase_generate``).
    Set to 0 for the pure seed diagnostic.
    """
    SLM = SLM_class()
    SLM.arraySizeBit = [array_size_bit, array_size_bit]
    N = 2 ** array_size_bit
    SLM.image_init(
        initGaussianPhase_user_defined=np.zeros((N, N)), Plot=False,
        beam_center_um=(0.0, 0.0),
    )
    focal_pitch = float(SLM.Focalpitchx)
    # Physical target: flat_top_um = 9 * 15.83 = 142.45 um, sigma_um = 2 * 15.83 = 31.66 um
    flat_width_um = 9.0 * 15.8281
    gauss_sigma_um = 2.0 * 15.8281
    flat_width_px = flat_width_um / focal_pitch
    gauss_sigma_px = gauss_sigma_um / focal_pitch

    # Target center, shifted diagonally by the same physical amount as production
    # (30 * 15.83 um = 475 um) so the focal-plane shift ramp stays comparable.
    shift_um = 30.0 * 15.8281
    shift_px = shift_um / focal_pitch
    target_center = (
        (N - 1) / 2.0 - shift_px,
        (N - 1) / 2.0 - shift_px,
    )

    init_phi = SLM.stationary_phase_sheet(
        flat_width=flat_width_px,
        gaussian_sigma=gauss_sigma_px,
        angle=0.0,
        center=target_center,
    )

    # Optional: apply the same post-hoc Fresnel lens as production
    # (testfile_sheet.py:176-180).  Note production applies it at the
    # SLM native resolution (1024x1272), we apply it at the compute grid
    # since that's where FFT propagates.  The physical parameters are
    # identical.
    if fresnel_sd_um != 0.0:
        pitch_slm = float(SLM.pixelpitch)
        lam = float(SLM.wavelength)
        f_eff = float(SLM.focallength)
        mag = float(SLM.magnification)
        coords = (np.arange(N) - (N - 1) / 2.0) * pitch_slm  # um
        X, Y = np.meshgrid(coords, coords, indexing="xy")
        phi_fresnel = np.pi * (X ** 2 + Y ** 2) * fresnel_sd_um / (
            lam * f_eff ** 2
        ) * mag ** 2
        phi_fresnel = np.mod(phi_fresnel, 2.0 * np.pi)
        phi_total = init_phi + phi_fresnel
    else:
        phi_total = init_phi

    S = SLM.initGaussianAmp  # real Gaussian amplitude on the N x N grid
    E_in = S.astype(np.complex128) * np.exp(1j * phi_total.astype(np.float64))
    E_out = fft_propagate(E_in)
    I_out = np.abs(E_out) ** 2

    # Normalise for plotting + profile comparison
    P_tot = float(I_out.sum())
    I_norm = I_out / P_tot if P_tot > 0 else I_out

    # Extract a physical-coordinate window of ±(shift + 2*flat_width) um around
    # the target center, so the same physical slab is shown at both grid sizes.
    half_win_um = shift_um + 2.0 * flat_width_um
    half_win_px = int(round(half_win_um / focal_pitch))
    cy, cx = int(round(target_center[0])), int(round(target_center[1]))
    y0 = max(cy - half_win_px, 0)
    y1 = min(cy + half_win_px, N)
    x0 = max(cx - half_win_px, 0)
    x1 = min(cx + half_win_px, N)
    win = I_norm[y0:y1, x0:x1]

    # Profile cuts through the target center (sheet is horizontal at angle=0,
    # so major axis is x, minor axis is y).
    major_cut = I_norm[cy, x0:x1]
    minor_cut = I_norm[y0:y1, cx]

    # Efficiency-in-region using the same measure region logic as CGM.
    from slm.targets import measure_region
    targetAmp = SLM.light_sheet_target(
        flat_width=flat_width_px,
        gaussian_sigma=gauss_sigma_px,
        angle=0.0,
        edge_sigma=0.0,
        center=target_center,
    )
    region = measure_region(targetAmp.shape, targetAmp, margin=5)
    eta_region = float(np.sum(I_out * region) / max(P_tot, 1e-30))

    # Geometry of the ACTUAL intensity in the window:
    # peak location (inside window), per-axis FWHM.
    peak_flat = float(win.max())
    half = peak_flat / 2.0
    # Major-axis FWHM in focal pixels
    peak_idx_major = int(np.argmax(major_cut))
    above = major_cut > half
    major_fwhm_px = float(above.sum())
    # Minor-axis FWHM in focal pixels
    peak_idx_minor = int(np.argmax(minor_cut))
    above_m = minor_cut > half
    minor_fwhm_px = float(above_m.sum())
    # Focal-pitch-aware um widths
    major_fwhm_um = major_fwhm_px * focal_pitch
    minor_fwhm_um = minor_fwhm_px * focal_pitch

    return {
        "N": N,
        "focal_pitch_um_per_px": focal_pitch,
        "flat_width_px": flat_width_px,
        "flat_width_um": flat_width_um,
        "gauss_sigma_px": gauss_sigma_px,
        "gauss_sigma_um": gauss_sigma_um,
        "shift_px": shift_px,
        "shift_um": shift_um,
        "target_center_px": target_center,
        "I_norm_win": win,
        "major_cut": major_cut,
        "minor_cut": minor_cut,
        "eta_in_region": eta_region,
        "major_fwhm_px": major_fwhm_px,
        "major_fwhm_um": major_fwhm_um,
        "minor_fwhm_px": minor_fwhm_px,
        "minor_fwhm_um": minor_fwhm_um,
        "peak_flat": peak_flat,
        "P_total": P_tot,
        "win_shape": win.shape,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building N=1024 case...")
    case_lo = build_case(array_size_bit=10)
    print(f"  focal pitch = {case_lo['focal_pitch_um_per_px']:.3f} um/px  "
          f"eta_region = {case_lo['eta_in_region']:.6f}")
    print(f"  major FWHM = {case_lo['major_fwhm_px']:.1f} px  "
          f"({case_lo['major_fwhm_um']:.1f} um)")
    print(f"  minor FWHM = {case_lo['minor_fwhm_px']:.1f} px  "
          f"({case_lo['minor_fwhm_um']:.1f} um)")

    print("Building N=4096 case...")
    case_hi = build_case(array_size_bit=12)
    print(f"  focal pitch = {case_hi['focal_pitch_um_per_px']:.3f} um/px  "
          f"eta_region = {case_hi['eta_in_region']:.6f}")
    print(f"  major FWHM = {case_hi['major_fwhm_px']:.1f} px  "
          f"({case_hi['major_fwhm_um']:.1f} um)")
    print(f"  minor FWHM = {case_hi['minor_fwhm_px']:.1f} px  "
          f"({case_hi['minor_fwhm_um']:.1f} um)")

    # -------- Plot --------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for row, case, label in ((0, case_lo, "N=1024"), (1, case_hi, "N=4096")):
        pitch = case["focal_pitch_um_per_px"]
        win = case["I_norm_win"]
        # x-axis in um, zero at window center
        hy, hw = win.shape
        x_um = (np.arange(hw) - hw / 2) * pitch
        y_um = (np.arange(hy) - hy / 2) * pitch

        ax = axes[row, 0]
        ax.imshow(win, cmap="hot",
                  extent=[x_um[0], x_um[-1], y_um[-1], y_um[0]],
                  vmin=0, vmax=win.max())
        ax.set_title(f"{label}: |E_out|^2 (window around target)\n"
                     f"eta={case['eta_in_region']*100:.2f}%  "
                     f"focal pitch={pitch:.2f} um/px")
        ax.set_xlabel("focal x [um]")
        ax.set_ylabel("focal y [um]")

        ax = axes[row, 1]
        x_major_um = (np.arange(len(case["major_cut"])) - len(case["major_cut"]) / 2) * pitch
        ax.plot(x_major_um, case["major_cut"], "k-", lw=1)
        ax.axvline(-case["flat_width_um"] / 2, ls="--", color="r", lw=0.8,
                   label=f"target edges ({case['flat_width_um']:.0f} um)")
        ax.axvline( case["flat_width_um"] / 2, ls="--", color="r", lw=0.8)
        ax.set_title(f"{label} along-line cut\n"
                     f"major FWHM = {case['major_fwhm_um']:.1f} um "
                     f"({case['major_fwhm_px']:.1f} px)")
        ax.set_xlabel("focal x [um]")
        ax.set_ylabel("normalised intensity")
        ax.legend(fontsize=8)

        ax = axes[row, 2]
        y_minor_um = (np.arange(len(case["minor_cut"])) - len(case["minor_cut"]) / 2) * pitch
        ax.plot(y_minor_um, case["minor_cut"], "k-", lw=1)
        target_perp_fwhm_um = case["gauss_sigma_um"] * 2.0 * np.sqrt(2.0 * np.log(2.0))
        ax.axvline(-target_perp_fwhm_um / 2, ls="--", color="r", lw=0.8,
                   label=f"target FWHM ({target_perp_fwhm_um:.0f} um)")
        ax.axvline( target_perp_fwhm_um / 2, ls="--", color="r", lw=0.8)
        ax.set_title(f"{label} perpendicular cut\n"
                     f"minor FWHM = {case['minor_fwhm_um']:.1f} um "
                     f"({case['minor_fwhm_px']:.1f} px)")
        ax.set_xlabel("focal y [um]")
        ax.legend(fontsize=8)

    fig.suptitle("stationary_phase_sheet seed: simulated focal-plane intensity\n"
                 "(1024 vs 4096 grid, same physical target, no fresnel_sd, no LUT)")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG_PATH, dpi=120)
    plt.close(fig)
    print(f"[save] {FIG_PATH}")

    # -------- JSON --------
    summary = {
        "note": "Pure Linux-side simulation of stationary_phase_sheet seed. "
                "No fresnel_sd applied, no LUT, no camera. "
                "Tests issue #20 hypotheses 1 and 2.",
        "physical_target": {
            "flat_width_um":   case_lo["flat_width_um"],
            "gauss_sigma_um":  case_lo["gauss_sigma_um"],
            "shift_um":        case_lo["shift_um"],
        },
        "cases": {
            "N_1024": {k: v for k, v in case_lo.items()
                       if not isinstance(v, np.ndarray)},
            "N_4096": {k: v for k, v in case_hi.items()
                       if not isinstance(v, np.ndarray)},
        },
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print(f"[save] {JSON_PATH}")

    # --------------------------------------------------------------
    # Hypothesis 3 test: simulate seed + post-hoc Fresnel lens at
    # fresnel_sd ∈ {0, 500, 1000, 2000, 5000} μm on the 1024 grid.
    # --------------------------------------------------------------
    fsd_values = [0, 500, 1000, 2000, 5000]
    cases_fsd = []
    for fsd in fsd_values:
        print(f"Building fresnel_sd = {fsd} um case...")
        c = build_case(array_size_bit=10, fresnel_sd_um=float(fsd))
        cases_fsd.append((fsd, c))
        print(f"  major FWHM = {c['major_fwhm_um']:.1f} um  "
              f"minor FWHM = {c['minor_fwhm_um']:.1f} um  "
              f"eta_region = {c['eta_in_region']:.4f}")

    ncols = len(fsd_values)
    fig, axes = plt.subplots(3, ncols, figsize=(3 * ncols + 2, 9))
    for k, (fsd, case) in enumerate(cases_fsd):
        pitch = case["focal_pitch_um_per_px"]
        win = case["I_norm_win"]
        hy, hw = win.shape
        x_um = (np.arange(hw) - hw / 2) * pitch
        y_um = (np.arange(hy) - hy / 2) * pitch

        ax = axes[0, k]
        ax.imshow(win, cmap="hot",
                  extent=[x_um[0], x_um[-1], y_um[-1], y_um[0]],
                  vmin=0, vmax=win.max())
        ax.set_title(f"fresnel_sd = {fsd} um\n"
                     f"eta={case['eta_in_region']*100:.1f}%")
        if k == 0:
            ax.set_ylabel("|E_out|^2\nfocal y [um]")
        ax.set_xlabel("focal x [um]")

        ax = axes[1, k]
        cut = case["major_cut"]
        x_major = (np.arange(len(cut)) - len(cut) / 2) * pitch
        ax.plot(x_major, cut, "k-", lw=1)
        ax.axvline(-case["flat_width_um"] / 2, ls="--", color="r", lw=0.8)
        ax.axvline( case["flat_width_um"] / 2, ls="--", color="r", lw=0.8)
        ax.set_title(f"along-line cut  FWHM={case['major_fwhm_um']:.1f} um")
        if k == 0:
            ax.set_ylabel("along-line cut")
        ax.set_xlabel("focal x [um]")

        ax = axes[2, k]
        cut = case["minor_cut"]
        y_minor = (np.arange(len(cut)) - len(cut) / 2) * pitch
        ax.plot(y_minor, cut, "k-", lw=1)
        target_fwhm = case["gauss_sigma_um"] * 2.0 * np.sqrt(2.0 * np.log(2.0))
        ax.axvline(-target_fwhm / 2, ls="--", color="r", lw=0.8)
        ax.axvline( target_fwhm / 2, ls="--", color="r", lw=0.8)
        ax.set_title(f"perp cut  FWHM={case['minor_fwhm_um']:.1f} um")
        if k == 0:
            ax.set_ylabel("perp cut")
        ax.set_xlabel("focal y [um]")

    fig.suptitle("Hypothesis 3 test: seed + post-hoc Fresnel lens (N=1024)\n"
                 "If any fresnel_sd ≠ 0 column turns the line into a blob, "
                 "the post-hoc lens is the culprit.")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_FSD, dpi=120)
    plt.close(fig)
    print(f"[save] {FIG_FSD}")

    summary_fsd = {
        "note": "Hypothesis 3 test: seed + post-hoc Fresnel lens, N=1024.",
        "cases": {
            str(fsd): {k: v for k, v in c.items()
                       if not isinstance(v, np.ndarray)}
            for fsd, c in cases_fsd
        },
    }
    with open(JSON_FSD, "w", encoding="utf-8") as f:
        json.dump(summary_fsd, f, ensure_ascii=False, indent=2, default=float)
    print(f"[save] {JSON_FSD}")


if __name__ == "__main__":
    main()
