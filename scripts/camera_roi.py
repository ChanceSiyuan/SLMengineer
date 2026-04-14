"""Camera ROI detection for top-hat disc: auto-detect the disc boundary
from hardware camera captures, then compute uniformity and total intensity.

Used by sweep analysis to evaluate hardware quality without assuming
any pixel-scale mapping.
"""
from __future__ import annotations

import numpy as np


def find_target_center(
    after: np.ndarray, before: np.ndarray, min_separation: int = 100
) -> tuple[int, int, int, int]:
    """Find target spot center in after image, away from zero-order.

    Returns (target_cy, target_cx, zo_cy, zo_cx).
    """
    # Zero-order: brightest cluster in before
    flat_b = before.ravel()
    top_b = np.argpartition(flat_b, -50)[-50:]
    rows_b, cols_b = np.unravel_index(top_b, before.shape)
    zo_cy, zo_cx = int(rows_b.mean()), int(cols_b.mean())

    # Target: brightest cluster in (after - before), away from zero-order
    signal = after.astype(np.float64) - before.astype(np.float64)
    Y, X = np.ogrid[:signal.shape[0], :signal.shape[1]]
    R_zo = np.sqrt((X - zo_cx) ** 2 + (Y - zo_cy) ** 2)
    masked = np.where(R_zo > min_separation, signal, 0)
    flat = masked.ravel()
    top_idx = np.argpartition(flat, -100)[-100:]
    rows, cols = np.unravel_index(top_idx, signal.shape)
    cy, cx = int(rows.mean()), int(cols.mean())
    return cy, cx, zo_cy, zo_cx


def radial_profile(image: np.ndarray, center: tuple[int, int], max_r: int) -> np.ndarray:
    """Azimuthally averaged radial intensity profile (1-pixel bins)."""
    cy, cx = center
    Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r_int = R.astype(int)
    profile = np.zeros(max_r)
    for i in range(max_r):
        ring = r_int == i
        vals = image[ring]
        if len(vals) > 0:
            profile[i] = vals.mean()
    return profile


def detect_disc_edge(
    image: np.ndarray,
    center: tuple[int, int],
    max_r: int = 200,
) -> int:
    """Detect the disc edge radius from camera image.

    Works on the raw 'after' image (NOT bg-subtracted).  Subtracts the
    far-field background median, smooths the radial profile, then finds the
    first radius where the profile drops below 1% of the peak *or* below
    the background noise floor (whichever is higher).

    If a ring surrounds the disc (profile drops to ~0 then rises again),
    this returns the *first* drop, not the outer edge of the ring.
    """
    from scipy.ndimage import uniform_filter1d

    cy, cx = center
    Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # Background from far field
    far = image[R > max_r]
    if len(far) == 0:
        far = image[R > max_r // 2]
    bg_median = float(np.median(far))
    bg_std = float(np.std(far))

    profile = radial_profile(image, center, max_r)
    profile_sub = profile - bg_median  # subtract background
    smooth = uniform_filter1d(profile_sub, 3)

    peak = smooth[:5].max()
    threshold = max(0.01 * peak, 3 * bg_std)

    # Walk outward: disc edge = first radius where smoothed profile < threshold
    was_above = False
    for i in range(max_r):
        if smooth[i] > threshold:
            was_above = True
        elif was_above:
            # Check this isn't a single-pixel noise dip: require 3 consecutive below
            if i + 2 < max_r and smooth[i + 1] < threshold and smooth[i + 2] < threshold:
                return i
    return max_r


def disc_metrics(
    signal: np.ndarray,
    center: tuple[int, int],
    radius: int,
) -> dict:
    """Compute uniformity and total intensity within a circular disc ROI.

    Returns dict with:
      - uniformity: std/mean of pixel intensities inside disc
      - total_intensity: sum of all pixel values inside disc
      - mean_intensity: mean pixel value inside disc
      - peak_intensity: maximum pixel value inside disc
      - n_pixels: number of pixels inside disc
      - disc_radius: the radius used
    """
    cy, cx = center
    Y, X = np.ogrid[:signal.shape[0], :signal.shape[1]]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = R <= radius
    vals = signal[mask]
    # Use all pixels (including low values) for honest uniformity
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals))
    uniformity = std_val / mean_val if mean_val > 0 else float('inf')
    return {
        "uniformity": uniformity,
        "total_intensity": float(np.sum(vals)),
        "mean_intensity": mean_val,
        "peak_intensity": float(np.max(vals)),
        "std_intensity": std_val,
        "n_pixels": int(mask.sum()),
        "disc_radius_cam_px": radius,
    }


def analyze_camera_capture(
    after_path: str, before_path: str
) -> dict:
    """Full pipeline: load camera data, detect disc, compute metrics.

    Uses raw 'after' for disc-edge detection (sharp features aren't
    smeared by background subtraction), then bg-subtracted image for
    uniformity/intensity metrics.

    Returns dict with center, disc_radius, and all metrics.
    """
    after = np.load(after_path).astype(np.float64)
    before = np.load(before_path).astype(np.float64)
    signal = after - before

    cy, cx, zo_cy, zo_cx = find_target_center(after, before)
    radius = detect_disc_edge(after, (cy, cx))  # raw image for edge detection
    metrics = disc_metrics(signal, (cy, cx), radius)  # bg-sub for metrics

    metrics["target_center"] = (cy, cx)
    metrics["zero_order_center"] = (zo_cy, zo_cx)
    metrics["separation_cam_px"] = float(np.sqrt((cy - zo_cy)**2 + (cx - zo_cx)**2))
    return metrics
