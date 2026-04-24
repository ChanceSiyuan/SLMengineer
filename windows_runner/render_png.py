"""Render <base>_before.bmp and <base>_after.bmp in <data_dir> to "hot"
colormap PNGs with colorbar and stats in the title.

Used by slm_local.bat (Windows-local dispatch, no SSH) and mirrors the
inline Python block in push_run.sh so both flows produce identical PNGs.
"""
import os
import sys

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def render(data_dir: str, base: str) -> int:
    ok = True
    for tag in ("before", "after"):
        bmp = os.path.join(data_dir, f"{base}_{tag}.bmp")
        png = os.path.join(data_dir, f"{base}_{tag}.png")
        if not os.path.exists(bmp):
            print(f"  {tag}: bmp not found ({bmp})", file=sys.stderr)
            ok = False
            continue
        arr = np.asarray(Image.open(bmp).convert("L"), dtype=np.uint8)
        vmax = max(int(arr.max()), 1)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(arr, cmap="hot", vmin=0, vmax=vmax)
        ax.set_title(
            f"{base}_{tag}  shape={arr.shape}  "
            f"min={int(arr.min())}  max={int(arr.max())}  "
            f"mean={arr.mean():.1f}"
        )
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(
            f"  {tag}: {os.path.getsize(bmp)//1024}KB bmp -> "
            f"{os.path.getsize(png)//1024}KB color png (hot cmap, vmax={vmax})"
        )
    return 0 if ok else 2


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: render_png.py <data_dir> <base>", file=sys.stderr)
        sys.exit(1)
    sys.exit(render(sys.argv[1], sys.argv[2]))
