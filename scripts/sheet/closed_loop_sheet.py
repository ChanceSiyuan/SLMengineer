"""Camera-feedback closed-loop light-sheet uniformity optimizer.

Each iteration:

 1. Build the phase from the current *weighted* target via CGM.  After
    iter 0 CGM warm-starts from the previous iteration's wrapped phase
    so the polish is incremental.
 2. Compose Fresnel + calibration → push payload → capture ``after.bmp``.
 3. Call ``analysis_sheet.analyze`` in-process to extract the flat-top
    1D profile and RMS / Pk-Pk.
 4. Map the measured flat-top onto the target's ``flat_width`` columns,
    compute correction weights ``w[i] = sqrt(mean / measured[i])``, and
    multiply the target's along-axis amplitude by a damped version.

Stops when flat-top RMS% ≤ ``RMS_TOL`` or after ``MAX_ITER``.

Run::

    uv run python scripts/sheet/closed_loop_sheet.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from slm.cgm import CGM_phase_generate
from slm.generation import SLM_class
from slm import imgpy
from slm.metrics import fidelity as _fidelity, efficiency as _efficiency
from slm.propagation import fft_propagate
from slm.targets import measure_region

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Scripts/ is not a package — ensure imports from sibling files work.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from analysis_sheet import analyze as analyze_after_bmp  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
PAYLOAD_DIR = REPO / "payload" / "sheet"
PAYLOAD_PATH = PAYLOAD_DIR / "testfile_sheet_payload.npz"
PARAMS_PATH = PAYLOAD_DIR / "testfile_sheet_params.json"
AFTER_BMP = REPO / "data" / "sheet" / "testfile_sheet_after.bmp"
HISTORY_DIR = REPO / "docs" / "sweep_sheet" / "closed_loop"
BEST_PLOT = HISTORY_DIR / "best_plot.png"
BEST_JSON = HISTORY_DIR / "best_result.json"

# Physics / CGM config — kept in sync with scripts/sheet/testfile_sheet.py.
FRESNEL_SD_UM          = int(os.environ.get("SLM_FRESNEL_SD", 1200))
LUT                    = 207
CAL_BMP                = "calibration/CAL_LSH0905549_1013nm.bmp"
ETIME_US               = int(os.environ.get("SLM_ETIME_US", 1500))
N_AVG                  = int(os.environ.get("SLM_N_AVG", 20))
SHEET_FLAT_WIDTH       = int(os.environ.get("SLM_FLAT_WIDTH", 9))
SHEET_GAUSSIAN_SIGMA   = float(os.environ.get("SLM_GAUSS_SIGMA", 1))
SHEET_EDGE_SIGMA       = float(os.environ.get("SLM_EDGE_SIGMA", 0))
SHEET_ANGLE_RAD        = 0.0
TARGET_SHIFT_FPX       = int(os.environ.get("SLM_TARGET_SHIFT_FPX", 20))
CGM_STEEPNESS          = int(os.environ.get("SLM_CGM_STEEPNESS", 9))
CGM_MAX_ITER           = int(os.environ.get("SLM_CGM_MAX_ITER", 4000))
CGM_SETTING_ETA        = float(os.environ.get("SLM_SETTING_ETA", 0.1))
CGM_ETA_STEEPNESS      = int(os.environ.get("SLM_CGM_ETA_STEEPNESS", 7))

# Closed-loop knobs
MAX_ITER       = int(os.environ.get("CL_MAX_ITER", 5))
RMS_TOL        = float(os.environ.get("CL_RMS_TOL", 2.0))
WEIGHT_CLIP    = (0.5, 2.0)
DAMPING        = float(os.environ.get("CL_DAMPING", 0.5))
# After iter 0, CGM warm-starts from the last wrapped phase so polish is
# incremental; reuse a smaller iteration budget to save hardware time.
CGM_WARM_ITER  = int(os.environ.get("CL_CGM_WARM_ITER", CGM_MAX_ITER // 2))


def apply_weights(target_base: np.ndarray, flat_cols: np.ndarray,
                  weights: np.ndarray) -> np.ndarray:
    """Multiply each flat-top column by its per-column weight, then
    renormalize to unit power."""
    out = target_base.copy()
    for col_idx, w in zip(flat_cols, weights):
        out[:, col_idx] *= w
    power = float(np.sum(np.abs(out) ** 2))
    if power > 0:
        out /= np.sqrt(power)
    return out


def build_slm_screen(SLM: SLM_class, target_amp: np.ndarray,
                     init_phi: np.ndarray, n_iter: int
                     ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Run CGM, fold in Fresnel + calibration; return (uint8 SLM screen,
    wrapped phase for warm-starting the next iter, sim fidelity, sim
    efficiency)."""
    SLM_Phase = CGM_phase_generate(
        torch.tensor(SLM.initGaussianAmp),
        torch.from_numpy(init_phi),
        torch.from_numpy(target_amp),
        max_iterations=n_iter,
        steepness=CGM_STEEPNESS,
        eta_min=CGM_SETTING_ETA,
        eta_steepness=CGM_ETA_STEEPNESS,
        Plot=False,
    )
    phase_wrapped = np.angle(np.exp(1j * SLM_Phase.cpu().clone().numpy()))
    screen_raw = SLM.phase_to_screen(phase_wrapped)

    W, H = SLM.SLMRes
    cx, cy = W // 2, H // 2
    fresnel = SLM.fresnel_lens_phase_generate(FRESNEL_SD_UM, cx, cy)[0]
    screen_shift = (
        (screen_raw.astype(np.int32) + fresnel.astype(np.int32)) % 256
    ).astype(np.uint8)
    screen_final = imgpy.SLM_screen_Correct(
        screen_shift, LUT=LUT, correctionImgPath=CAL_BMP
    )

    region = measure_region(target_amp.shape, target_amp, margin=5)
    E_out = fft_propagate(SLM.initGaussianAmp * np.exp(1j * phase_wrapped))
    F = float(_fidelity(E_out, target_amp, region))
    eta = float(_efficiency(E_out, region))
    return screen_final, phase_wrapped, F, eta


def save_payload(screen: np.ndarray, SLM: SLM_class, extra: dict) -> None:
    np.savez_compressed(PAYLOAD_PATH, slm_screen=screen)
    params = {
        "algorithm": "CGM-closed-loop",
        "payload": str(PAYLOAD_PATH.relative_to(REPO)),
        "compute_grid": [int(SLM.ImgResY), int(SLM.ImgResX)],
        "slm_native": [int(SLM.SLMRes[1]), int(SLM.SLMRes[0])],
        "focal_pitch_x_um_per_px": round(float(SLM.Focalpitchx), 4),
        "focal_pitch_y_um_per_px": round(float(SLM.Focalpitchy), 4),
        "runner_defaults": {"etime_us": ETIME_US, "n_avg": N_AVG, "monitor": 1},
        "target": "light_sheet",
        "sheet_flat_width_px": SHEET_FLAT_WIDTH,
        "sheet_gaussian_sigma_px": SHEET_GAUSSIAN_SIGMA,
        "sheet_edge_sigma_px": SHEET_EDGE_SIGMA,
        "sheet_angle_rad": SHEET_ANGLE_RAD,
        "fresnel_applied_on_linux": True,
        "fresnel_shift_distance_um": FRESNEL_SD_UM,
        "calibration_applied_on_linux": True,
        "calibration_bmp": CAL_BMP,
        "LUT": LUT,
        "cgm_max_iterations": CGM_MAX_ITER,
        "cgm_steepness": CGM_STEEPNESS,
        "cgm_setting_eta": CGM_SETTING_ETA,
        "cgm_eta_steepness": CGM_ETA_STEEPNESS,
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        **extra,
    }
    with open(PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)


def push_and_capture() -> None:
    subprocess.run(
        ["./push_run.sh", str(PAYLOAD_PATH.relative_to(REPO))],
        cwd=str(REPO), check=True,
    )


def bin_profile(profile: np.ndarray, a: int, b: int, n: int) -> np.ndarray:
    """Split ``profile[a:b]`` into n equal-width bins and return the mean
    of each bin; falls back to profile mean if the slice is empty."""
    flat = profile[a:b]
    if flat.size == 0:
        return np.ones(n) * float(profile.mean())
    edges = np.linspace(0, len(flat), n + 1)
    out = np.zeros(n)
    for i in range(n):
        lo = int(round(edges[i]))
        hi = max(int(round(edges[i + 1])), lo + 1)
        out[i] = float(flat[lo:hi].mean())
    return out


def main():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    SLM = SLM_class()
    SLM.arraySizeBit = [10, 10]
    SLM.image_init(
        initGaussianPhase_user_defined=np.zeros((1024, 1024)),
        Plot=False, beam_center_um=(0, 0),
    )
    ny, nx = int(SLM.ImgResY), int(SLM.ImgResX)
    target_center = (
        (ny - 1) / 2.0 - TARGET_SHIFT_FPX,
        (nx - 1) / 2.0 - TARGET_SHIFT_FPX,
    )
    target_base = SLM.light_sheet_target(
        flat_width=SHEET_FLAT_WIDTH,
        gaussian_sigma=SHEET_GAUSSIAN_SIGMA,
        angle=SHEET_ANGLE_RAD,
        edge_sigma=SHEET_EDGE_SIGMA,
        center=target_center,
    )
    init_phi_seed = SLM.stationary_phase_sheet(
        flat_width=SHEET_FLAT_WIDTH,
        gaussian_sigma=None,
        angle=SHEET_ANGLE_RAD,
        center=target_center,
    )

    # Flat-top columns (angle=0 → flat region runs along x).
    x_off = np.arange(nx) - target_center[1]
    flat_cols = np.where(np.abs(x_off) <= SHEET_FLAT_WIDTH / 2.0)[0]
    N = int(flat_cols.size)
    print(f"[init] flat-top spans {N} target columns: "
          f"x={flat_cols[0]}..{flat_cols[-1]}")

    weights = np.ones(N)
    warm_phi = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    history: list[dict] = []
    best: dict | None = None

    for it in range(MAX_ITER):
        tag = f"iter{it:02d}"
        t0 = time.perf_counter()
        print(f"\n=== {tag}  (weights mean={weights.mean():.3f}, "
              f"range [{weights.min():.3f}, {weights.max():.3f}]) ===")

        target_mod = apply_weights(target_base, flat_cols, weights)
        # Warm-start from the previous iter's phase once we have one.
        init_phi = warm_phi if warm_phi is not None else init_phi_seed
        n_iter = CGM_MAX_ITER if warm_phi is None else CGM_WARM_ITER
        print(f"[{tag}] CGM ({n_iter} iters, device={device}, "
              f"warm={'no' if warm_phi is None else 'yes'})...")
        screen, warm_phi, F, eta = build_slm_screen(
            SLM, target_mod, init_phi, n_iter,
        )
        print(f"[{tag}] sim fidelity={F:.6f}  efficiency={eta*100:.2f}%")

        save_payload(screen, SLM, extra={
            "closed_loop_iter": it,
            "closed_loop_weights": [round(float(w), 5) for w in weights],
            "sim_fidelity": F,
            "sim_efficiency": eta,
        })

        print(f"[{tag}] pushing & capturing...")
        push_and_capture()

        plot_out = HISTORY_DIR / f"{tag}_plot.png"
        json_out = HISTORY_DIR / f"{tag}_result.json"
        result = analyze_after_bmp(AFTER_BMP, plot_out, json_out)
        rms = float(result["rms_percent"])
        ppk = float(result["pk_pk_percent"])
        a, b = result["flat_top_bounds_px"]
        profile = np.asarray(result["profile_values"])
        dt = time.perf_counter() - t0
        print(f"[{tag}] measured: RMS={rms:.2f}%   Pk-Pk={ppk:.2f}%   "
              f"flat=[{a},{b})   iter_time={dt:.1f}s")

        hist_entry = {
            "tag": tag, "iter": it,
            "rms_percent": rms, "pk_pk_percent": ppk,
            "flat_bounds_px": [a, b],
            "weights": [float(w) for w in weights],
            "sim_fidelity": F, "sim_efficiency": eta,
            "plot_path": str(plot_out),
        }
        history.append(hist_entry)
        if best is None or rms < best["rms_percent"]:
            best = hist_entry

        if rms <= RMS_TOL:
            print(f"  ✓ converged: RMS {rms:.2f}% ≤ tol {RMS_TOL}%")
            break

        bins = bin_profile(profile, a, b, N)
        mean = float(bins.mean()) if bins.mean() > 0 else 1.0
        raw_w = np.sqrt(mean / np.clip(bins, 1e-6, None))
        step = (1.0 - DAMPING) + DAMPING * raw_w
        weights = np.clip(weights * step, *WEIGHT_CLIP)
        weights /= weights.mean()
        print(f"[{tag}] bins={np.round(bins, 1).tolist()}")
        print(f"[{tag}] new weights={np.round(weights, 3).tolist()}")

    print("\n" + "=" * 78)
    print(f"Closed-loop summary ({len(history)} iterations)")
    print("=" * 78)
    print(f"{'tag':>8}  {'RMS%':>7}  {'Pk-Pk%':>8}  {'sim_F':>8}  {'sim_eta%':>8}")
    for h in history:
        mark = " ← best" if h is best else ""
        print(f"{h['tag']:>8}  {h['rms_percent']:>7.2f}  {h['pk_pk_percent']:>8.2f}  "
              f"{h['sim_fidelity']:>8.4f}  {h['sim_efficiency']*100:>8.2f}{mark}")

    if best is not None:
        shutil.copy(best["plot_path"], BEST_PLOT)
        with open(BEST_JSON, "w") as f:
            json.dump(best, f, indent=2)
        print(f"\nBest: {best['tag']}  RMS={best['rms_percent']:.2f}%  "
              f"Pk-Pk={best['pk_pk_percent']:.2f}%")
        print(f"Best plot → {BEST_PLOT.relative_to(REPO)}")
        print(f"Best meta → {BEST_JSON.relative_to(REPO)}")

    with open(HISTORY_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Full history → {(HISTORY_DIR / 'history.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
