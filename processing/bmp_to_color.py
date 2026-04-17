"""Convert 8-bit grayscale BMP camera captures into colored PNG heatmaps.

Usage::

    # Single file (output next to input, same stem, .png)
    uv run python processing/bmp_to_color.py data/sheet/testfile_sheet_after.bmp

    # Explicit output path
    uv run python processing/bmp_to_color.py input.bmp --out /tmp/out.png

    # Batch: every BMP under a directory (recursive)
    uv run python processing/bmp_to_color.py data/sheet/ --recursive

    # Custom colormap + label
    uv run python processing/bmp_to_color.py input.bmp --cmap inferno

Flags:
    --cmap NAME       matplotlib colormap (default: hot)
    --out PATH        output path (single-file mode only)
    --out-dir DIR     destination directory for batch mode (mirrors input layout)
    --recursive       when input is a directory, walk it recursively
    --no-colorbar     drop the colorbar / title / axes (pure heatmap image)
    --vmax INT        clamp color scale to [0, vmax] instead of per-image max
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_bmp(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def convert(
    bmp_path: Path,
    out_path: Path,
    cmap: str,
    colorbar: bool,
    vmax: int | None,
) -> None:
    img = load_bmp(bmp_path)
    vmax_use = int(vmax) if vmax is not None else int(img.max())
    vmax_use = max(vmax_use, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not colorbar:
        # Pure heatmap (no axes, no borders): identical pixel grid,
        # colors sampled from the requested cmap.
        cm = matplotlib.colormaps[cmap]
        norm = np.clip(img.astype(np.float32) / vmax_use, 0.0, 1.0)
        rgba = (cm(norm) * 255).astype(np.uint8)
        Image.fromarray(rgba[..., :3], mode="RGB").save(out_path, format="PNG")
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax_use)
        ax.set_title(
            f"{bmp_path.name}\n"
            f"shape={img.shape} min={int(img.min())} "
            f"max={int(img.max())} mean={img.mean():.2f}"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"{bmp_path} -> {out_path}")


def resolve_outputs(
    inputs: list[Path],
    root: Path | None,
    out: Path | None,
    out_dir: Path | None,
) -> list[tuple[Path, Path]]:
    if len(inputs) == 1 and out is not None:
        return [(inputs[0], out)]

    if out_dir is None:
        return [(p, p.with_suffix(".png")) for p in inputs]

    pairs = []
    for p in inputs:
        if root is not None:
            rel = p.relative_to(root)
        else:
            rel = Path(p.name)
        pairs.append((p, (out_dir / rel).with_suffix(".png")))
    return pairs


def main():
    ap = argparse.ArgumentParser(
        description="Convert grayscale BMP captures to colored PNG heatmaps."
    )
    ap.add_argument("input", type=Path,
                    help="BMP file or directory containing BMPs.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG path (single-file mode).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (batch mode, mirrors input layout).")
    ap.add_argument("--cmap", default="hot",
                    help="matplotlib colormap name (default: hot).")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse when input is a directory.")
    ap.add_argument("--no-colorbar", action="store_true",
                    help="Emit pure heatmap image (no axes/colorbar/title).")
    ap.add_argument("--vmax", type=int, default=None,
                    help="Clamp color scale to [0, VMAX] instead of per-image max.")
    args = ap.parse_args()

    src: Path = args.input
    if not src.exists():
        print(f"ERROR: input not found: {src}", file=sys.stderr)
        sys.exit(1)

    if src.is_file():
        inputs = [src]
        root = None
    else:
        pattern = "**/*.bmp" if args.recursive else "*.bmp"
        inputs = sorted(src.glob(pattern))
        root = src
        if not inputs:
            print(f"ERROR: no BMPs found under {src}", file=sys.stderr)
            sys.exit(1)

    pairs = resolve_outputs(inputs, root, args.out, args.out_dir)
    for bmp, png in pairs:
        convert(bmp, png, args.cmap, not args.no_colorbar, args.vmax)


if __name__ == "__main__":
    main()
