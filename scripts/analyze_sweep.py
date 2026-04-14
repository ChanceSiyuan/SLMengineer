"""Post-sweep analysis: compute camera-based uniformity and total intensity
for each sweep point using auto-detected disc ROI.

Usage::

    uv run python scripts/analyze_sweep.py                    # default sweep dir
    uv run python scripts/analyze_sweep.py data/sweep_tophat  # custom data dir

Reads sweep_manifest.json from scripts/sweep_tophat/ and camera .npy files
from the data directory. Outputs a results table and summary PDF.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from camera_roi import analyze_camera_capture

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SWEEP_DIR = "scripts/sweep_tophat"
DATA_DIR = "data/sweep_tophat"
MANIFEST = f"{SWEEP_DIR}/sweep_manifest.json"
OUTPUT_JSON = f"{SWEEP_DIR}/sweep_camera_results.json"
OUTPUT_PDF = f"{SWEEP_DIR}/sweep_camera_analysis.pdf"


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR

    manifest = json.load(open(MANIFEST))
    print(f"Sweep: {len(manifest)} points, param={manifest[0]['sweep_param']}")
    print(f"Data dir: {data_dir}")

    results = []
    for entry in manifest:
        idx = entry["index"]
        prefix = f"sweep_tophat_{idx:03d}"
        after_path = f"{data_dir}/{prefix}_after.npy"
        before_path = f"{data_dir}/{prefix}_before.npy"

        if not os.path.exists(after_path) or not os.path.exists(before_path):
            print(f"  [{idx:03d}] SKIP — .npy files missing")
            results.append(None)
            continue

        metrics = analyze_camera_capture(after_path, before_path)
        metrics["index"] = idx
        metrics["sweep_param"] = entry["sweep_param"]
        metrics["sweep_value"] = entry["sweep_value"]
        metrics["sim_fidelity"] = entry.get("fidelity")
        metrics["sim_efficiency"] = entry.get("efficiency")
        results.append(metrics)

        print(f"  [{idx:03d}] {entry['sweep_param']}={entry['sweep_value']}"
              f"  disc_r={metrics['disc_radius_cam_px']}"
              f"  unif={metrics['uniformity']*100:.1f}%"
              f"  total_I={metrics['total_intensity']:.0f}"
              f"  peak={metrics['peak_intensity']:.1f}")

    # Filter valid results
    valid = [r for r in results if r is not None]
    if not valid:
        print("No valid results to analyze.")
        return

    # Save results
    with open(OUTPUT_JSON, "w") as f:
        json.dump(valid, f, indent=2, default=str)
    print(f"\n[SAVE] {OUTPUT_JSON}")

    # Summary table
    param = valid[0]["sweep_param"]
    print(f"\n{'='*80}")
    print(f"SWEEP SUMMARY: {param}")
    print(f"{'='*80}")
    print(f"{'value':>10}  {'disc_r':>6}  {'unif%':>7}  {'total_I':>8}  "
          f"{'peak':>6}  {'mean':>6}  {'sim_F':>6}")
    print(f"{'-'*62}")
    for r in valid:
        sf = f"{r['sim_fidelity']:.4f}" if r['sim_fidelity'] else "  n/a"
        print(f"{r['sweep_value']:>10}  {r['disc_radius_cam_px']:>6}  "
              f"{r['uniformity']*100:>6.1f}%  {r['total_intensity']:>8.0f}  "
              f"{r['peak_intensity']:>6.1f}  {r['mean_intensity']:>6.2f}  {sf}")

    # Best by uniformity (lower is better)
    best_unif = min(valid, key=lambda r: r["uniformity"])
    # Best by total intensity (higher is better)
    best_total = max(valid, key=lambda r: r["total_intensity"])
    print(f"\nBest uniformity:      {param}={best_unif['sweep_value']}  "
          f"(σ/μ={best_unif['uniformity']*100:.1f}%)")
    print(f"Best total intensity: {param}={best_total['sweep_value']}  "
          f"(total_I={best_total['total_intensity']:.0f})")

    # --- Plot ---
    vals = [r["sweep_value"] for r in valid]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(vals, [r["uniformity"] * 100 for r in valid], "ro-", linewidth=2)
    ax.set_ylabel("σ/μ (%)")
    ax.set_title(f"Uniformity vs {param}\n(lower is better)")
    ax.set_xlabel(param)

    ax = axes[0, 1]
    ax.plot(vals, [r["total_intensity"] for r in valid], "bs-", linewidth=2)
    ax.set_ylabel("Total intensity (ADU)")
    ax.set_title(f"Total intensity vs {param}\n(higher is better)")
    ax.set_xlabel(param)

    ax = axes[1, 0]
    ax.plot(vals, [r["peak_intensity"] for r in valid], "g^-", linewidth=2)
    ax.set_ylabel("Peak intensity (ADU)")
    ax.set_title(f"Peak intensity vs {param}")
    ax.set_xlabel(param)

    ax = axes[1, 1]
    ax.plot(vals, [r["disc_radius_cam_px"] for r in valid], "mD-", linewidth=2)
    ax.set_ylabel("Detected disc radius (cam-px)")
    ax.set_title(f"Disc radius vs {param}")
    ax.set_xlabel(param)

    plt.suptitle(f"Sweep analysis: {param} ({len(valid)} points)", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, dpi=120)
    plt.close(fig)
    print(f"[SAVE] {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
