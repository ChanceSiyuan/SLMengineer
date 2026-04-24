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

from testfile_sheet import main as generate_sheet   # noqa: E402
from analysis_sheet import analyze                  # noqa: E402

# ----- Hyperparameters -----
TOTAL_LOOP = 10     # push+analyze iterations (= PNGs generated)
STEPS      = 1      # refresh payload every STEPS iters using averaged v
STEEPNESS  = 0.5    # 0: uniform (no correction); 1: w = v^{-1}
FLAT_A_UM  = 50     # fixed per issue #23
FLAT_B_UM  = 200    # fixed per issue #23

IS_WINDOWS = platform.system() == "Windows"
AFTER_BMP = REPO_ROOT / "data" / "sheet" / "testfile_sheet_after.bmp"
PAYLOAD   = REPO_ROOT / "payload" / "sheet" / "testfile_sheet_payload.npz"
PUSH_RUN  = REPO_ROOT / ("push_run.ps1" if IS_WINDOWS else "push_run.sh")


def compute_reweight(v: np.ndarray, steepness: float) -> np.ndarray:
    """w close to 1 when steepness small; w = v^{-1} (normalized) when steepness=1.

    Linear blend between uniform-1 and inverse-normalized; mean(w) = 1.
    """
    v = np.asarray(v, dtype=np.float64)
    v_norm = np.clip(v / v.mean(), 1e-6, None)
    inv = 1.0 / v_norm
    inv = inv / inv.mean()
    w = (1.0 - steepness) + steepness * inv
    return w / w.mean()


def main() -> None:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = REPO_ROOT / "data" / f"sheet_inloop_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INLOOP] total_loop={TOTAL_LOOP} steps={STEPS} steepness={STEEPNESS}")
    print(f"[INLOOP] output dir: {out_dir}")

    # Iter 0 setup: uniform-target payload.
    generate_sheet()

    flat_buffer: list[np.ndarray] = []
    history: list[dict] = []
    last_w: np.ndarray | None = None

    for i in range(TOTAL_LOOP):
        # Push phase to hardware + pull back after.bmp.
        payload_arg = str(PAYLOAD.relative_to(REPO_ROOT))
        if IS_WINDOWS:
            cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File",
                   str(PUSH_RUN), payload_arg]
        else:
            cmd = ["bash", str(PUSH_RUN), payload_arg]
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

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
            "flat_mean": result["flat_top_mean_intensity"],
            "reweight_was_applied": last_w is not None,
        })
        print(f"[ITER {i:02d}] rms={result['rms_percent']:.3f}%  "
              f"pkpk={result['pk_pk_percent']:.3f}%")

        if (i + 1) % STEPS == 0 and (i + 1) < TOTAL_LOOP:
            v_avg = np.mean(np.stack(flat_buffer), axis=0)
            w = compute_reweight(v_avg, STEEPNESS)
            last_w = w
            generate_sheet(reweight=w)
            flat_buffer.clear()

    summary = {
        "timestamp": ts,
        "hyperparams": {
            "total_loop": TOTAL_LOOP, "steps": STEPS, "steepness": STEEPNESS,
            "flat_a_um": FLAT_A_UM, "flat_b_um": FLAT_B_UM,
        },
        "iterations": history,
        "final_reweight": None if last_w is None else last_w.tolist(),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INLOOP] done. summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
