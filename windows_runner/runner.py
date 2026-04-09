"""Windows hardware runner for precomputed SLM phase payloads.

This script lives on the Windows lab box at::

    C:\\Users\\Galileo\\slm_runner\\runner.py

It is a minimal, **self-contained** hardware runner that knows how to:

  1. Load a payload .npz file (produced on Linux) containing a pre-
     computed uint8 SLM screen of shape (SLM_H, SLM_W).  All Fresnel
     lens / calibration BMP / LUT corrections have already been applied
     on the Linux side -- this runner does zero computation beyond
     camera-frame averaging.
  2. Open the SLM display on the requested monitor.
  3. Capture a "before" image with the SLM showing a blank screen.
  4. Upload the loaded screen and capture an "after" image.
  5. Compute the difference image.
  6. Save all three arrays (.npy), PNG previews, and per-capture stats
     (JSON) into ``data/<prefix>_*``.

It does **NOT**:
  - import torch, slm.cgm, scipy.optimize, or any CGM logic
  - compute Fresnel lenses or apply calibration BMPs
  - generate target patterns

Those responsibilities stay on the Linux side in
``scripts/testfile_lg.py`` (or whatever analog you write for the next
experiment).

Usage::

    python runner.py --payload incoming\\testfile_lg_payload.npz \\
                     --output-prefix testfile_lg \\
                     [--etime-us 4000] [--n-avg 10] [--monitor 1]

The Linux-side orchestrator ``scripts/testfile_lg.sh`` invokes this
script via ssh after scp'ing the payload into ``incoming\\``.  After the
runner completes, the orchestrator scp's ``data\\<prefix>_*`` back to
the Linux ``./data/`` directory.

See ``README.md`` in this directory for the one-time Windows setup
(venv, slmpy, Vimba SDK, PYTHONPATH).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The runner reuses the SLM display / camera wrappers from the main
# SLMengineer repo's src/slm/ package (already installed on the
# Windows lab box under C:\Users\Galileo\SLMengineer\).  These wrappers
# are CPU-only (no torch, no scipy.optimize) so importing them does not
# pull the heavy CGM stack onto the hardware box.
#
# If you prefer a fully standalone runner, replace these imports with
# direct calls to slmpy and the Vimba Python API.
_MAIN_REPO_SRC = r"C:\Users\Galileo\SLMengineer\src"
if os.path.isdir(_MAIN_REPO_SRC) and _MAIN_REPO_SRC not in sys.path:
    sys.path.insert(0, _MAIN_REPO_SRC)

from slm.display import SLMdisplay  # noqa: E402
from slm.camera import VimbaCamera   # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def multi_capture(camera, etime_us, n_frames):
    """Average ``n_frames`` camera frames at the given exposure."""
    acc = None
    for _ in range(n_frames):
        frame = camera.capture(etime_us)
        if acc is None:
            acc = np.zeros(frame.shape, dtype=np.float64)
        acc += frame.astype(np.float64)
    return (acc / n_frames).astype(np.float32)


def capture_stats(image, label):
    """Compact stats dict suitable for JSON output."""
    return {
        "label": label,
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "min": round(float(np.min(image)), 2),
        "max": round(float(np.max(image)), 2),
        "mean": round(float(np.mean(image)), 4),
        "std": round(float(np.std(image)), 4),
    }


def save_preview(name, img, out_dir, cmap="hot"):
    """Save a labelled PNG preview for one captured frame."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(img, cmap=cmap, vmin=img.min(), vmax=img.max())
    ax.set_title(
        f"{name}\nmin={img.min():.1f} max={img.max():.1f} mean={img.mean():.2f}"
    )
    plt.colorbar(im, ax=ax)
    fig.savefig(out_dir / f"{name}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Windows hardware runner for precomputed SLM phase payloads."
    )
    ap.add_argument(
        "--payload", required=True,
        help="Path to the payload .npz file (must contain 'slm_screen' uint8 array).",
    )
    ap.add_argument(
        "--output-prefix", required=True,
        help="Filename prefix for outputs under data/ (e.g. 'testfile_lg').",
    )
    ap.add_argument(
        "--etime-us", type=int, default=4000,
        help="Camera exposure time in microseconds (default: 4000 = 4 ms).",
    )
    ap.add_argument(
        "--n-avg", type=int, default=10,
        help="Number of frames to average per capture (default: 10).",
    )
    ap.add_argument(
        "--monitor", type=int, default=1,
        help="SLM display monitor index (default: 1).",
    )
    args = ap.parse_args()

    payload_path = Path(args.payload)
    if not payload_path.is_file():
        print(f"ERROR: payload not found: {payload_path}", file=sys.stderr)
        sys.exit(1)

    prefix = args.output_prefix
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    # ─── Load payload ───────────────────────────────────────────────
    print(f"[1/5] Loading payload: {payload_path}")
    data = np.load(payload_path)
    if "slm_screen" not in data:
        print(
            f"ERROR: payload {payload_path} missing 'slm_screen' array. "
            f"Available keys: {list(data.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)
    slm_screen = np.asarray(data["slm_screen"], dtype=np.uint8)
    if slm_screen.ndim != 2:
        print(
            f"ERROR: 'slm_screen' must be 2D; got shape={slm_screen.shape}",
            file=sys.stderr,
        )
        sys.exit(1)
    H, W = slm_screen.shape
    print(
        f"  slm_screen: shape=({H}, {W}) dtype={slm_screen.dtype} "
        f"range=[{slm_screen.min()}, {slm_screen.max()}] "
        f"mean={slm_screen.mean():.1f}"
    )

    # ─── Open SLM display ───────────────────────────────────────────
    print(f"\n[2/5] Opening SLM on monitor {args.monitor}...")
    slm = SLMdisplay(monitor=args.monitor, isImageLock=True)

    try:
        # ─── Capture "before" (SLM blank) ───────────────────────────
        print(f"\n[3/5] Capturing 'before' ({args.n_avg} frames @ "
              f"{args.etime_us} us)...")
        blank_screen = np.zeros((H, W), dtype=np.uint8)
        slm.updateArray(blank_screen)
        time.sleep(0.5)
        with VimbaCamera() as camera:
            img_before = multi_capture(camera, args.etime_us, args.n_avg)
        print(
            f"  img_before: shape={img_before.shape} "
            f"max={img_before.max():.1f} mean={img_before.mean():.2f}"
        )

        # ─── Display payload and capture "after" ────────────────────
        print(f"\n[4/5] Uploading payload screen and capturing 'after'...")
        slm.updateArray(slm_screen)
        time.sleep(0.5)
        with VimbaCamera() as camera:
            img_after = multi_capture(camera, args.etime_us, args.n_avg)
        print(
            f"  img_after:  shape={img_after.shape} "
            f"max={img_after.max():.1f} mean={img_after.mean():.2f}"
        )
    finally:
        slm.close()

    # ─── Save outputs to data/ ──────────────────────────────────────
    print(f"\n[5/5] Saving outputs to {out_dir}/...")
    np.save(out_dir / f"{prefix}_before.npy", img_before)
    np.save(out_dir / f"{prefix}_after.npy",  img_after)
    diff = img_after - img_before
    np.save(out_dir / f"{prefix}_diff.npy", diff)

    save_preview(f"{prefix}_before", img_before, out_dir, cmap="hot")
    save_preview(f"{prefix}_after",  img_after,  out_dir, cmap="hot")
    save_preview(f"{prefix}_diff",   diff,       out_dir, cmap="RdBu_r")

    run_meta = {
        "payload": str(payload_path),
        "output_prefix": prefix,
        "exposure_us": args.etime_us,
        "n_avg_frames": args.n_avg,
        "monitor": args.monitor,
        "slm_screen_shape": list(slm_screen.shape),
        "slm_screen_range": [int(slm_screen.min()), int(slm_screen.max())],
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "before": capture_stats(img_before, "SLM blank"),
        "after":  capture_stats(img_after,  "SLM payload screen"),
        "diff_max": round(float(diff.max()), 2),
        "diff_min": round(float(diff.min()), 2),
    }
    with open(out_dir / f"{prefix}_run.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print(
        f"  {prefix}_before.npy / .png\n"
        f"  {prefix}_after.npy  / .png\n"
        f"  {prefix}_diff.npy   / .png\n"
        f"  {prefix}_run.json"
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
