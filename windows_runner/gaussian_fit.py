"""2D elliptical-Gaussian fit for camera images.

Two input modes:

  * ``--capture``          grab a frame from the Vimba camera, save it as
                            a BMP, then fit.  Requires the Vimba SDK on
                            this (Windows lab) box.
  * ``--image <path>``     load an existing BMP/PNG/TIFF/NPY image and
                            fit it.  Works anywhere (Linux or Windows).

Both modes write a PNG overlay with the image, the fitted Gaussian
contour, and a text block listing the fit parameters.

Output files (under ``data/gauss/``, or ``--out-dir``):

    <name>.bmp         (capture mode only) raw camera frame
    <name>_fit.png     image + Gaussian contours + fit annotation
    <name>_fit.json    fit parameters + goodness-of-fit

Usage::

    # Capture a new frame and fit it
    python gaussian_fit.py --capture --name beam01 --etime-us 4000

    # Fit an existing BMP
    python gaussian_fit.py --image path/to/frame.bmp --name beam01
"""
from __future__ import annotations

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
from scipy.optimize import curve_fit

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _gaussian_2d(xy, amp, x0, y0, sx, sy, offset):
    x, y = xy
    return (amp * np.exp(-((x - x0) ** 2 / (2 * sx ** 2)
                           + (y - y0) ** 2 / (2 * sy ** 2)))
            + offset).ravel()


def load_image(path: Path) -> np.ndarray:
    """Load BMP/PNG/TIFF/NPY into a 2D float32 array (grayscale)."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        img = np.load(path)
    else:
        img = plt.imread(str(path))
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    return img.astype(np.float32)


def capture_bmp(out_bmp: Path, etime_us: int, n_avg: int) -> np.ndarray:
    """Grab ``n_avg`` averaged frames from the Vimba camera, save as BMP."""
    _MAIN_REPO_SRC = r"C:\Users\Galileo\SLMengineer\src"
    if os.path.isdir(_MAIN_REPO_SRC) and _MAIN_REPO_SRC not in sys.path:
        sys.path.insert(0, _MAIN_REPO_SRC)
    from slm.camera import VimbaCamera  # noqa: E402

    print(f"[capture] {n_avg} frames @ {etime_us} us")
    acc = None
    with VimbaCamera() as cam:
        for _ in range(n_avg):
            frame = cam.capture(etime_us)
            if acc is None:
                acc = np.zeros(frame.shape, dtype=np.float64)
            acc += frame.astype(np.float64)
    avg = acc / n_avg

    # Save BMP via matplotlib (uint8 grayscale).
    lo, hi = float(avg.min()), float(avg.max())
    norm = (avg - lo) / (hi - lo) if hi > lo else np.zeros_like(avg)
    bmp = (norm * 255).astype(np.uint8)
    plt.imsave(str(out_bmp), bmp, cmap="gray", format="bmp")
    print(f"[capture] saved {out_bmp}  range=[{lo:.1f}, {hi:.1f}]")
    return avg.astype(np.float32)


def fit_gaussian_2d(img: np.ndarray):
    """Fit axis-aligned 2D Gaussian.  Returns (params, perr, r2)."""
    H, W = img.shape
    y, x = np.mgrid[0:H, 0:W]

    # Initial guesses from image moments (background-subtracted).
    bg = float(np.median(img))
    amp0 = float(img.max() - bg)
    # Peak location via argmax (robust for a single-lobe beam).
    y0_init, x0_init = np.unravel_index(int(np.argmax(img)), img.shape)

    # Sigma estimate from second moment over a local window.
    win = max(20, int(0.1 * min(H, W)))
    ylo, yhi = max(0, y0_init - win), min(H, y0_init + win)
    xlo, xhi = max(0, x0_init - win), min(W, x0_init + win)
    sub = img[ylo:yhi, xlo:xhi] - bg
    sub = np.clip(sub, 0, None)
    total = sub.sum() + 1e-9
    yy, xx = np.mgrid[ylo:yhi, xlo:xhi]
    sx0 = float(np.sqrt((sub * (xx - x0_init) ** 2).sum() / total)) or 10.0
    sy0 = float(np.sqrt((sub * (yy - y0_init) ** 2).sum() / total)) or 10.0

    p0 = [amp0, float(x0_init), float(y0_init), sx0, sy0, bg]
    bounds = (
        [0, 0, 0, 0.5, 0.5, -np.inf],
        [np.inf, W, H, W, H, np.inf],
    )

    popt, pcov = curve_fit(
        _gaussian_2d, (x, y), img.ravel(), p0=p0, bounds=bounds, maxfev=5000,
    )
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))

    fit = _gaussian_2d((x, y), *popt).reshape(img.shape)
    ss_res = float(((img - fit) ** 2).sum())
    ss_tot = float(((img - img.mean()) ** 2).sum()) + 1e-30
    r2 = 1.0 - ss_res / ss_tot
    return popt, perr, r2, fit


def save_fit_png(img, popt, perr, r2, out_png: Path, title: str):
    amp, x0, y0, sx, sy, offset = popt
    fwhm_x = 2.0 * np.sqrt(2 * np.log(2)) * sx
    fwhm_y = 2.0 * np.sqrt(2 * np.log(2)) * sy

    H, W = img.shape
    y, x = np.mgrid[0:H, 0:W]
    model = _gaussian_2d((x, y), *popt).reshape(img.shape)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(img, cmap="hot", vmin=float(img.min()), vmax=float(img.max()))
    # 1-sigma, 2-sigma, FWHM contours of the fitted model.
    levels = offset + amp * np.exp(-0.5 * np.array([1.0, 4.0]) ** 2)
    ax.contour(model, levels=levels, colors=["cyan", "cyan"],
               linewidths=[1.2, 0.8], linestyles=["-", "--"])
    ax.plot([x0], [y0], "x", color="cyan", mew=2, ms=10)

    txt = (
        f"amp    = {amp:.2f} ({perr[0]:.2f})\n"
        f"x0     = {x0:.2f} ({perr[1]:.2f}) px\n"
        f"y0     = {y0:.2f} ({perr[2]:.2f}) px\n"
        f"sig_x  = {sx:.2f} ({perr[3]:.2f}) px\n"
        f"sig_y  = {sy:.2f} ({perr[4]:.2f}) px\n"
        f"FWHM_x = {fwhm_x:.2f} px\n"
        f"FWHM_y = {fwhm_y:.2f} px\n"
        f"offset = {offset:.2f} ({perr[5]:.2f})\n"
        f"R^2    = {r2:.5f}"
    )
    ax.text(
        0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
        family="monospace", fontsize=10, color="white",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=6),
    )
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="2D elliptical-Gaussian fit.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--capture", action="store_true",
                     help="Capture a new frame from the Vimba camera.")
    src.add_argument("--image", type=str,
                     help="Path to an existing BMP/PNG/TIFF/NPY image.")
    ap.add_argument("--name", default="gauss",
                    help="Output filename stem (default: 'gauss').")
    ap.add_argument("--out-dir", default="data/gauss",
                    help="Directory for outputs (default: data/gauss).")
    ap.add_argument("--etime-us", type=int, default=4000,
                    help="Exposure in us for --capture (default: 4000).")
    ap.add_argument("--n-avg", type=int, default=10,
                    help="Frames to average for --capture (default: 10).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.capture:
        bmp_path = out_dir / f"{args.name}.bmp"
        img = capture_bmp(bmp_path, args.etime_us, args.n_avg)
        src_label = str(bmp_path)
    else:
        src_path = Path(args.image)
        img = load_image(src_path)
        src_label = str(src_path)

    print(f"[fit] image shape={img.shape} range=[{img.min():.1f}, {img.max():.1f}]")
    popt, perr, r2, _fit = fit_gaussian_2d(img)
    amp, x0, y0, sx, sy, offset = popt
    print(f"[fit] x0={x0:.2f} y0={y0:.2f} sig_x={sx:.2f} sig_y={sy:.2f} R^2={r2:.5f}")

    png_path = out_dir / f"{args.name}_fit.png"
    json_path = out_dir / f"{args.name}_fit.json"
    save_fit_png(img, popt, perr, r2, png_path,
                 title=f"{args.name} — 2D Gaussian fit")

    result = {
        "source": src_label,
        "image_shape": [int(img.shape[0]), int(img.shape[1])],
        "amp": float(amp),     "amp_err": float(perr[0]),
        "x0_px": float(x0),    "x0_err_px": float(perr[1]),
        "y0_px": float(y0),    "y0_err_px": float(perr[2]),
        "sigma_x_px": float(sx), "sigma_x_err_px": float(perr[3]),
        "sigma_y_px": float(sy), "sigma_y_err_px": float(perr[4]),
        "fwhm_x_px": float(2.0 * np.sqrt(2 * np.log(2)) * sx),
        "fwhm_y_px": float(2.0 * np.sqrt(2 * np.log(2)) * sy),
        "offset": float(offset), "offset_err": float(perr[5]),
        "r_squared": float(r2),
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[save] {png_path}")
    print(f"[save] {json_path}")


if __name__ == "__main__":
    main()
