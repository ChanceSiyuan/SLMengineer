"""Closed-loop camera feedback for the light-sheet payload (issue #20).

Drives the asymmetry of the on-camera light sheet toward zero by adjusting
``BEAM_CENTER_DX_UM`` between hardware iterations, with a strict shape-
regression guardrail so the loop cannot trade away the line shape to
reduce asymmetry (a small symmetric blob also has zero asymmetry).

Each iteration:

  1. Run ``scripts/sheet/testfile_sheet.py`` with the current
     ``SLM_BCM_DX_UM`` env override.
  2. ``./push_run.sh ... --png`` pushes the payload, runs the SLM, captures
     before/after PNGs from the Vimba camera.
  3. ``scripts/sheet/analysis_sheet.py`` extracts the signal ROI + fid_corr.
  4. Compute (a) left/right brightness asymmetry over the cropped ROI,
     (b) shape_score = aspect_ratio * fid_corr.
  5. Decide next step:
       * if |asym| ≤ ASYM_TOL → done
       * if shape_score < SHAPE_FLOOR * best_shape_score → revert to best
         and shrink the step
       * else: damped gradient on |asym|; halve step on sign flip

Stops when |asym| ≤ ASYM_TOL or step shrinks below MIN_STEP or MAX_ITER hit.

Run::

    uv run python scripts/sheet/closed_loop_sheet.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

import numpy as np
from PIL import Image


REPO = "/home/chance/SLMengineer"
TESTFILE = "scripts/sheet/testfile_sheet.py"
PAYLOAD = "payload/sheet/testfile_sheet_payload.npz"
AFTER_PNG = "data/sheet/testfile_sheet_after.png"
BEFORE_PNG = "data/sheet/testfile_sheet_before.png"
PARAMS_JSON = "payload/sheet/testfile_sheet_params.json"
ARCHIVE_DIR = "docs/sweep_sheet"

MAX_ITER = 6
ASYM_TOL = 0.05
INITIAL_BCM = -2000           # phase-2 manual winner
INITIAL_STEP = 500            # μm — small to stay near the known good point
MIN_STEP = 100                # μm
SHAPE_FLOOR = 0.6             # if shape_score < SHAPE_FLOOR * best, revert


def run_iteration(bcm_dx_um: int, tag: str) -> dict:
    env = {**os.environ, "SLM_BCM_DX_UM": str(int(bcm_dx_um))}
    print(f"\n=== iter {tag}: BCM_DX={bcm_dx_um} μm ===")
    t0 = time.perf_counter()
    subprocess.run(
        ["uv", "run", "python", TESTFILE], cwd=REPO, env=env, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"  payload built in {time.perf_counter() - t0:.1f}s; pushing...")
    t0 = time.perf_counter()
    subprocess.run(
        ["./push_run.sh", PAYLOAD, "--png"], cwd=REPO, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"  hardware capture in {time.perf_counter() - t0:.1f}s; analysing...")

    metrics_path = f"/tmp/cl_iter_{tag}_metrics.json"
    preview_path = f"/tmp/cl_iter_{tag}_preview.png"
    subprocess.run(
        ["uv", "run", "python", "scripts/sheet/analysis_sheet.py",
         "--after", AFTER_PNG, "--before", BEFORE_PNG, "--params", PARAMS_JSON,
         "--peak", "signal",
         "--out", metrics_path, "--preview", preview_path],
        cwd=REPO, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    metrics = json.load(open(os.path.join(REPO, metrics_path)))

    img = np.array(Image.open(os.path.join(REPO, AFTER_PNG)).convert("L"))
    roi = metrics["roi"]
    cy, cx, dx, dy = roi["cy"], roi["cx"], roi["dx"], roi["dy"]
    margin = 2
    y0 = max(0, cy - margin * dy); y1 = min(img.shape[0], cy + margin * dy)
    x0 = max(0, cx - margin * dx); x1 = min(img.shape[1], cx + margin * dx)
    crop = img[y0:y1, x0:x1].astype(np.float64)

    half = crop.shape[1] // 2
    left = crop[:, :half].sum()
    right = crop[:, half:].sum()
    asym = (left - right) / (left + right) if (left + right) > 0 else 0.0

    aspect = float(dx) / max(float(dy), 1.0)
    fid_corr = float(metrics["metrics"]["fidelity_corr"])
    eff = float(metrics["metrics"]["efficiency"])
    shape_score = aspect * fid_corr   # higher = better

    archive = os.path.join(REPO, ARCHIVE_DIR, f"closed_loop_{tag}_after.png")
    shutil.copy(os.path.join(REPO, AFTER_PNG), archive)
    Image.fromarray(crop.astype(np.uint8)).save(
        os.path.join(REPO, ARCHIVE_DIR, f"closed_loop_{tag}_zoom.png"))

    out = {
        "tag": tag,
        "bcm_dx_um": int(bcm_dx_um),
        "roi_dx": int(dx), "roi_dy": int(dy),
        "aspect": aspect, "fid_corr": fid_corr, "efficiency": eff,
        "asym_lr": asym, "shape_score": shape_score, "archive": archive,
    }
    print(
        f"  → ROI {dx}×{dy} aspect={aspect:.2f}  asym={asym:+.3f}  "
        f"fid={fid_corr:.3f}  eff={eff*100:.1f}%  shape_score={shape_score:.3f}"
    )
    return out


def main():
    history: list[dict] = []
    bcm = INITIAL_BCM
    step = INITIAL_STEP
    last_asym_sign = 0
    best = None

    print(f"Closed-loop start: BCM_DX={bcm} μm, step={step} μm, "
          f"|asym| target ≤ {ASYM_TOL}, shape_score floor {SHAPE_FLOOR}× best")

    for i in range(MAX_ITER):
        result = run_iteration(bcm, f"i{i:02d}")
        history.append(result)

        # Track best by shape_score; tie-break by lower |asym|.
        if (best is None
                or result["shape_score"] > best["shape_score"]
                or (abs(result["shape_score"] - best["shape_score"]) < 1e-6
                    and abs(result["asym_lr"]) < abs(best["asym_lr"]))):
            best = result

        if abs(result["asym_lr"]) <= ASYM_TOL and result["aspect"] >= 2.5:
            print(f"\n✓ converged: |asym|={abs(result['asym_lr']):.3f} ≤ {ASYM_TOL} "
                  f"AND aspect={result['aspect']:.2f} ≥ 2.5")
            break

        # Hard guardrail: if shape collapsed vs best so far, REVERT.
        if (best is not None and best is not result
                and result["shape_score"] < SHAPE_FLOOR * best["shape_score"]):
            print(f"  ! shape regression: {result['shape_score']:.3f} < "
                  f"{SHAPE_FLOOR}×{best['shape_score']:.3f}; reverting to "
                  f"BCM_DX={best['bcm_dx_um']} and shrinking step {step}→{step // 2}")
            bcm = best["bcm_dx_um"]
            step = max(step // 2, MIN_STEP)
            last_asym_sign = 0
            continue

        cur_sign = 1 if result["asym_lr"] > 0 else -1
        if last_asym_sign != 0 and cur_sign != last_asym_sign:
            step = max(step // 2, MIN_STEP)
            print(f"  asym sign flipped → halving step to {step}")
        last_asym_sign = cur_sign

        # asym>0 = brighter on left.  Empirically more-negative bcm reduces
        # the left bias on this rig.
        bcm -= cur_sign * step

        if step < MIN_STEP:
            print(f"  step shrunk below MIN_STEP={MIN_STEP}; stopping")
            break

    # Final report.
    print("\n" + "=" * 78)
    print("Closed-loop summary")
    print("=" * 78)
    print(f"{'iter':>4}  {'BCM_DX':>7}  {'asym':>7}  {'aspect':>7}  "
          f"{'fid':>6}  {'eff':>6}  {'shape_score':>11}")
    for h in history:
        marker = " ← best" if h is best else ""
        print(f"{h['tag']:>4}  {h['bcm_dx_um']:>7}  {h['asym_lr']:>+7.3f}  "
              f"{h['aspect']:>7.2f}  {h['fid_corr']:>6.3f}  "
              f"{h['efficiency']*100:>5.1f}%  {h['shape_score']:>11.3f}{marker}")

    if best is not None:
        print(f"\nBest: {best['tag']}  BCM_DX={best['bcm_dx_um']}  "
              f"shape_score={best['shape_score']:.3f}  asym={best['asym_lr']:+.3f}  "
              f"aspect={best['aspect']:.2f}  fid={best['fid_corr']:.3f}")
        # Re-run the best config one more time so the live testfile_sheet
        # state matches (and the archive after.png reflects the final).
        if best is not history[-1]:
            print(f"\nRe-running best ({best['tag']}) so final state matches…")
            final = run_iteration(best["bcm_dx_um"], "FINAL")
            history.append(final)

    out_path = os.path.join(REPO, ARCHIVE_DIR, "closed_loop_history.json")
    with open(out_path, "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"\nHistory saved: {out_path}")


if __name__ == "__main__":
    main()
