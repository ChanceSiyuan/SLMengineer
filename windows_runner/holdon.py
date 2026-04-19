"""Hold a precomputed SLM screen on the SLM until Ctrl+C.

Lives on the Windows lab box at:
    C:\\Users\\Galileo\\slm_runner\\holdon.py

Accepts the same payload files the runner.py / push_run.sh pipeline
produces:

    # positional path, .npz payload from Linux (contains key 'slm_screen')
    python holdon.py incoming\\wgs_square\\testfile_wgs_square_payload.npz

    # positional path, raw .npy uint8 screen
    python holdon.py some_screen.npy

    # optional: pick the display monitor (default 1)
    python holdon.py some_screen.npy --monitor 2

The path must refer to either:
  - a 2D uint8 .npy array of shape (SLM_H, SLM_W), or
  - an .npz archive containing a 'slm_screen' entry of the same shape
    (this is what `testfile_*.py` writes via `np.savez_compressed`).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_MAIN_REPO_SRC = r"C:\Users\Galileo\SLMengineer\src"
if os.path.isdir(_MAIN_REPO_SRC) and _MAIN_REPO_SRC not in sys.path:
    sys.path.insert(0, _MAIN_REPO_SRC)

from slm.display import SLMdisplay  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load_screen(path: Path) -> np.ndarray:
    """Load an SLM uint8 screen from .npy or .npz payload."""
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as z:
            if "slm_screen" not in z.files:
                raise SystemExit(
                    f"ERROR: {path} has no 'slm_screen' key "
                    f"(available: {list(z.files)})"
                )
            arr = z["slm_screen"]
    elif suffix == ".npy":
        arr = np.load(path)
    else:
        raise SystemExit(
            f"ERROR: unsupported extension '{suffix}' "
            f"(expected .npy or .npz): {path}"
        )
    if arr.ndim != 2:
        raise SystemExit(f"ERROR: array must be 2D, got shape={arr.shape}")
    return np.asarray(arr, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser(
        description="Load an SLM screen from .npy or .npz and hold it on the SLM."
    )
    ap.add_argument(
        "path",
        help="Path to a .npy (raw uint8 screen) or .npz (payload with "
             "'slm_screen' key) file.",
    )
    ap.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="SLM display monitor index (default: 1).",
    )
    args = ap.parse_args()

    path = Path(args.path)
    if not path.is_file():
        raise SystemExit(f"ERROR: file not found: {path}")

    print(f"[1/3] Loading screen: {path}")
    slm_screen = _load_screen(path)

    H, W = slm_screen.shape
    print(
        f"  shape=({H}, {W}), dtype={slm_screen.dtype}, "
        f"range=[{slm_screen.min()}, {slm_screen.max()}], "
        f"mean={slm_screen.mean():.2f}"
    )

    print(f"[2/3] Opening SLM on monitor {args.monitor}...")
    slm = SLMdisplay(monitor=args.monitor, isImageLock=True)

    print("[3/3] Uploading screen and holding on SLM.")
    print("Press Ctrl+C to release the SLM and exit.")

    try:
        slm.updateArray(slm_screen)
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nInterrupted by user, releasing SLM.")
    finally:
        slm.close()


if __name__ == "__main__":
    main()
