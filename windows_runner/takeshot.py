"""Windows-side camera snapshot + analysis utility (no SLM control).

Use when the SLM is already being held manually by another process
(typically ``runner.py --hold-on`` in a separate session) and you want
to record what the camera currently sees without touching the SLM.

Each invocation:

 1. Opens the Allied Vision camera.
 2. Averages ``--n-avg`` frames at ``--etime-us`` exposure.
 3. Saves the result to ``data/<prefix>.bmp`` plus a stats sidecar
    ``data/<prefix>_run.json``.
 4. Unless ``--no-analyze`` is passed, runs
    ``C:\\Users\\Galileo\\SLMengineer\\scripts\\sheet\\analysis_sheet.py``
    on the BMP and writes ``data/<prefix>_analysis.{png,json}``.

The script is self-contained: hardware wrappers live in sibling files
(``vimba_camera.py``) and the analysis helper is imported from the main
SLMengineer repo.  Deploy this file together with
``runner.py`` / ``slm_display.py`` / ``vimba_camera.py`` to
``C:\\Users\\Galileo\\SLMengineer\\windows_runner\\``.

Usage::

    python takeshot.py                             # auto-timestamp prefix
    python takeshot.py --output-prefix tweak_lens  # named capture
    python takeshot.py --etime-us 800 --n-avg 30 --no-analyze
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from vimba_camera import VimbaCamera  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    return np.clip(np.round(frame), 0, 255).astype(np.uint8)


def multi_capture(camera, etime_us, n_frames):
    acc = None
    for _ in range(n_frames):
        frame = camera.capture(etime_us)
        if acc is None:
            acc = np.zeros(frame.shape, dtype=np.float64)
        acc += frame.astype(np.float64)
    return (acc / n_frames).astype(np.float32)


def capture_stats(image, label):
    return {
        "label": label,
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "min": round(float(np.min(image)), 2),
        "max": round(float(np.max(image)), 2),
        "mean": round(float(np.mean(image)), 4),
        "std": round(float(np.std(image)), 4),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Snap the camera while the SLM is held externally.",
    )
    ap.add_argument(
        "--output-prefix", default=None,
        help="Filename prefix under data/ (defaults to "
             "'capture_YYYYMMDD_HHMMSS').",
    )
    ap.add_argument(
        "--etime-us", type=int, default=1500,
        help="Exposure in microseconds (default: 1500).",
    )
    ap.add_argument(
        "--n-avg", type=int, default=20,
        help="Frames to average (default: 20).",
    )
    ap.add_argument(
        "--no-analyze", action="store_true",
        help="Skip analysis_sheet.py; just save the BMP + stats.",
    )
    args = ap.parse_args()

    prefix = args.output_prefix or time.strftime("capture_%Y%m%d_%H%M%S")
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    print(f"[1/3] Capturing ({args.n_avg} frames @ {args.etime_us} us)...")
    with VimbaCamera() as camera:
        img = multi_capture(camera, args.etime_us, args.n_avg)
    print(
        f"  img: shape={img.shape} max={img.max():.1f} "
        f"mean={img.mean():.2f} std={img.std():.2f}"
    )

    bmp_path = out_dir / f"{prefix}.bmp"
    bmp_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[2/3] Saving {bmp_path} + {prefix}_run.json...")
    Image.fromarray(_to_uint8(img), mode="L").save(bmp_path, format="BMP")
    run_meta = {
        "output_prefix": prefix,
        "exposure_us": args.etime_us,
        "n_avg_frames": args.n_avg,
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "capture": capture_stats(img, "SLM held externally"),
    }
    with open(out_dir / f"{prefix}_run.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    if args.no_analyze:
        print("\nDone.")
        return

    # Deferred import: analysis_sheet pulls matplotlib/scipy/PIL.  The
    # sibling copy lives next to this file in windows_runner\ — keep it in
    # sync manually (or via scp) when the Linux copy changes.
    from analysis_sheet import analyze as _analyze_after

    plot_path = out_dir / f"{prefix}_analysis.png"
    json_path = out_dir / f"{prefix}_analysis.json"
    print(f"\n[3/3] Analysing {bmp_path.name}...")
    _analyze_after(bmp_path, plot_path, json_path)
    print(f"  {prefix}_analysis.png\n  {prefix}_analysis.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
