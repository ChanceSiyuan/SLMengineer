"""Closed-loop uniformity correction for the light sheet (issue #23).

Workflow per iteration:
  1. Push the current payload to the SLM hardware and pull the camera bmp.
  2. Analyze the bmp and extract the flat-region intensity profile ``v``.
  3. Every ``STEPS`` iterations, average the collected ``v`` vectors,
     compute reweight ``w`` (a slight correction toward uniform), and
     refresh ``payload/sheet/testfile_sheet_payload.npz``.

Outputs land in ``data/sheet_inloop_<timestamp>/`` with per-iter bmp/png/json
plus a ``summary.json`` tracking rms / pk-pk across iterations.

    uv run python scripts/sheet/sheet_inloop.py
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "sheet"))

# ----- Hyperparameters -----
TOTAL_LOOP = 10      # push+analyze iterations (= PNGs generated)
STEPS      = 1       # refresh payload every STEPS iters using averaged v
STEEPNESS  = 0.2     # 0: no correction; 1: w = sqrt(1/v). Kept small + clipped.
CLIP_LO    = 0.85    # hard floor on w — guarantees "slight correction" invariant
CLIP_HI    = 1.15    # hard ceiling on w
FLAT_A_UM  = float(os.environ.get("SHEET_FLAT_A_UM", 50)) 
FLAT_B_UM  = float(os.environ.get("SHEET_FLAT_B_UM", 175))


SLM_FLAT_WIDTH_PX = 35
SLM_GAUSS_SIGMA   = 2
os.environ["SLM_FLAT_WIDTH"]  = str(SLM_FLAT_WIDTH_PX)
os.environ["SLM_GAUSS_SIGMA"] = str(SLM_GAUSS_SIGMA)

PUSH_RETRY_COUNT  = 3
PUSH_RETRY_SLEEP  = 10

IS_WINDOWS = platform.system() == "Windows"
AFTER_BMP = REPO_ROOT / "data" / "sheet" / "testfile_sheet_after.bmp"
PAYLOAD   = REPO_ROOT / "payload" / "sheet" / "testfile_sheet_payload.npz"
PUSH_RUN  = REPO_ROOT / ("push_run.ps1" if IS_WINDOWS else "push_run.sh")

from testfile_sheet import main as generate_sheet   # noqa: E402
from analysis_sheet import analyze                  # noqa: E402


def compute_reweight(v: np.ndarray, steepness: float,
                     clip_lo: float = CLIP_LO, clip_hi: float = CLIP_HI) -> np.ndarray:
    """Amplitude-domain correction, steepness-blended and hard-clipped.

    v is measured intensity; the target amplitude multiplier that produces a
    focal intensity change of factor k is sqrt(k). So w = sqrt(1/v_norm)
    matches the WGS adaptive formula in imgpy.py::fftLoop_adapt. Clipping
    guarantees the "slight correction" invariant even if v has outliers.
    """
    v = np.asarray(v, dtype=np.float64)
    v_norm = np.clip(v / v.mean(), 1e-6, None)
    inv_sqrt = 1.0 / np.sqrt(v_norm)
    inv_sqrt = inv_sqrt / inv_sqrt.mean()
    w = (1.0 - steepness) + steepness * inv_sqrt
    w = np.clip(w, clip_lo, clip_hi)
    return w / w.mean()


def push_run_retry(payload_arg: str) -> None:
    """Call push_run.{sh,ps1} with retry; transient SSH drops don't waste the CGM work."""
    for attempt in range(PUSH_RETRY_COUNT):
        try:
            if IS_WINDOWS:
                cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File",
                       str(PUSH_RUN), payload_arg]
            else:
                cmd = ["bash", str(PUSH_RUN), payload_arg]
            subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
            return
        except subprocess.CalledProcessError as e:
            if attempt == PUSH_RETRY_COUNT - 1:
                raise
            print(f"[RETRY] push_run attempt {attempt + 1}/{PUSH_RETRY_COUNT} "
                  f"failed (exit {e.returncode}); backing off {PUSH_RETRY_SLEEP}s")
            time.sleep(PUSH_RETRY_SLEEP)


def main() -> None:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    tag = f"_a{int(FLAT_A_UM)}_b{int(FLAT_B_UM)}"
    out_dir = REPO_ROOT / "data" / f"sheet_inloop_{ts}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INLOOP] total_loop={TOTAL_LOOP} steps={STEPS} steepness={STEEPNESS} "
          f"clip=[{CLIP_LO},{CLIP_HI}] slm_flat_width={SLM_FLAT_WIDTH_PX}px "
          f"flat=[{FLAT_A_UM},{FLAT_B_UM}]um")
    print(f"[INLOOP] output dir: {out_dir}")

    generate_sheet()

    flat_buffer: list[np.ndarray] = []
    history: list[dict] = []
    last_w: np.ndarray | None = None

    for i in range(TOTAL_LOOP):
        payload_arg = str(PAYLOAD.relative_to(REPO_ROOT))
        push_run_retry(payload_arg)

        # Archive the raw bmp for this iter before the next push overwrites it.
        iter_bmp = out_dir / f"iter_{i:02d}.bmp"
        shutil.copy2(AFTER_BMP, iter_bmp)

        result = analyze(
            str(iter_bmp),
            plot_path=str(out_dir / f"iter_{i:02d}.png"),
            result_path=str(out_dir / f"iter_{i:02d}.json"),
            flat_a=FLAT_A_UM, flat_b=FLAT_B_UM,
        )
        flat = np.asarray(result["flat_profile"], dtype=np.float64)
        
        flat_buffer.append(flat)
        history.append({
            "iter": i,
            "rms_percent": result["rms_percent"],
            "pk_pk_percent": result["pk_pk_percent"],
            "efficiency_observed": result["efficiency_observed"],
            "flat_mean": result["flat_top_mean_intensity"],
            "reweight_was_applied": last_w is not None,
        })
        print(f"[ITER {i:02d}] rms={result['rms_percent']:.3f}%  "
              f"pkpk={result['pk_pk_percent']:.3f}%  "
              f"eff={100*result['efficiency_observed']:.3f}%")

        if (i + 1) % STEPS == 0 and (i + 1) < TOTAL_LOOP:
            v_avg = np.mean(np.stack(flat_buffer), axis=0)
            w = compute_reweight(v_avg, STEEPNESS)
            print(f"[UPDATE] v_avg range=[{(v_avg/v_avg.mean()).min():.3f}, "
                  f"{(v_avg/v_avg.mean()).max():.3f}] "
                  f"→ w range=[{w.min():.3f}, {w.max():.3f}] "
                  f"(clip={CLIP_LO}/{CLIP_HI})")
            last_w = w
            generate_sheet(reweight=w)
            flat_buffer.clear()

    summary = {
        "timestamp": ts,
        "hyperparams": {
            "total_loop": TOTAL_LOOP, "steps": STEPS, "steepness": STEEPNESS,
            "flat_a_um": FLAT_A_UM, "flat_b_um": FLAT_B_UM,
            "slm_flat_width_px": SLM_FLAT_WIDTH_PX,
            "slm_gauss_sigma":   SLM_GAUSS_SIGMA,
        },
        "iterations": history,
        "final_reweight": None if last_w is None else last_w.tolist(),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INLOOP] done. summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
