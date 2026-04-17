"""Aggregate light-sheet sweep: run analysis_sheet on every hardware
capture, plot trends per swept parameter, and pick the best config.

Usage::

    uv run python scripts/sheet/analyze_sweep_sheet.py
    uv run python scripts/sheet/analyze_sweep_sheet.py --w-eff 1.0 --w-fid 2.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_capture_bmp(path) -> np.ndarray:
    """Load an 8-bit grayscale BMP capture into a float64 array."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analysis_sheet import analyze_capture  # noqa: E402

SWEEP_DIR = Path("payload/sheet")
DATA_DIR = Path("data/sweep_sheet")
OUT_DIR = Path("payload/sheet")


def _entry_to_params(entry: dict) -> dict:
    """Reshape a sweep-manifest entry into the flat dict that
    analysis_sheet expects as *params*."""
    return {
        "sheet_flat_width_px": entry["sheet_flat_width_px"],
        "sheet_gaussian_sigma_px": entry["sheet_gaussian_sigma_px"],
        "sheet_edge_sigma_px": entry["sheet_edge_sigma_px"],
        "sheet_angle_rad": entry["sheet_angle_rad"],
        "focal_pitch_x_um_per_px": entry.get("focal_pitch_x_um_per_px"),
        "focal_pitch_y_um_per_px": entry.get("focal_pitch_y_um_per_px"),
    }


def _pareto_front(points: list[tuple[float, float, int]]) -> list[int]:
    """Return indices on the Pareto frontier (maximize both axes)."""
    indices = []
    for i, (ei, fi, idx_i) in enumerate(points):
        dominated = False
        for j, (ej, fj, _) in enumerate(points):
            if j == i:
                continue
            if ej >= ei and fj >= fi and (ej > ei or fj > fi):
                dominated = True
                break
        if not dominated:
            indices.append(idx_i)
    return indices


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate light-sheet sweep + pick best config."
    )
    ap.add_argument("--manifest", default=str(SWEEP_DIR / "sweep_manifest.json"))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--w-eff", type=float, default=1.0,
                    help="Exponent on efficiency in composite score.")
    ap.add_argument("--w-fid", type=float, default=1.0,
                    help="Exponent on fidelity_corr in composite score.")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.is_file():
        print(f"ERROR: {manifest_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    results = []
    for entry in manifest:
        idx = entry["index"]
        prefix = f"sweep_sheet_{idx:03d}"
        after_p = data_dir / f"{prefix}_after.bmp"
        before_p = data_dir / f"{prefix}_before.bmp"
        if not after_p.is_file() or not before_p.is_file():
            print(f"[{idx:03d}] SKIP — missing bmp ({after_p.name} or {before_p.name})")
            continue

        try:
            res = analyze_capture(
                after_p, before_p,
                params_path=None,
                preview_path=None,
            )
            res["metrics"] = {**res["metrics"]}
            params = _entry_to_params(entry)
            from analysis_sheet import _build_reference, _intensity_fidelity
            y0, y1, x0, x1 = res["roi"]["bbox"]
            roi = _load_capture_bmp(after_p) - _load_capture_bmp(before_p)
            roi = roi[y0:y1, x0:x1]
            ref = _build_reference(roi.shape, params, res["fit"])
            res["metrics"].update(_intensity_fidelity(roi, ref))
        except Exception as e:
            print(f"[{idx:03d}] FAIL — {e}")
            continue

        m = res["metrics"]
        eff = float(m.get("efficiency", 0.0))
        fid = float(m.get("fidelity_corr", 0.0))
        entry_out = {
            **entry,
            "hardware_efficiency": eff,
            "hardware_fidelity_corr": fid,
            "hardware_fidelity_overlap": float(m.get("fidelity_overlap", 0.0)),
            "hardware_flat_region_rms": float(m.get("flat_region_rms", float("nan"))),
            "measured_flat_width_px": float(res["fit"].get("measured_flat_width_px", float("nan"))),
            "measured_gauss_sigma_px": float(res["fit"].get("measured_gauss_sigma_px", float("nan"))),
            "measured_edge_sigma_px": float(res["fit"].get("measured_edge_sigma_px", float("nan"))),
            "roi_warning": res["roi"]["warning"],
        }
        results.append(entry_out)
        print(f"[{idx:03d}] {entry['sweep_param']}={entry['sweep_value']}  "
              f"eff={eff*100:.2f}%  fid={fid:.4f}")

    if not results:
        print("No usable sweep points. Aborting.")
        sys.exit(1)

    analysis_path = out_dir / "analysis_manifest.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {analysis_path}")

    by_param: dict[str, list] = {}
    for r in results:
        by_param.setdefault(r["sweep_param"], []).append(r)

    for param_name, entries in by_param.items():
        entries_sorted = sorted(entries, key=lambda e: e["sweep_value"])
        xs = [e["sweep_value"] for e in entries_sorted]
        effs = [e["hardware_efficiency"] * 100 for e in entries_sorted]
        fids = [e["hardware_fidelity_corr"] for e in entries_sorted]
        rmss = [e["hardware_flat_region_rms"] for e in entries_sorted]
        widths = [e["measured_flat_width_px"] for e in entries_sorted]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].plot(xs, effs, "o-"); axes[0, 0].set_title("efficiency [%]")
        axes[0, 1].plot(xs, fids, "o-"); axes[0, 1].set_title("fidelity_corr")
        axes[1, 0].plot(xs, rmss, "o-"); axes[1, 0].set_title("flat_region_rms")
        axes[1, 1].plot(xs, widths, "o-"); axes[1, 1].set_title("measured flat_width [px]")
        for ax in axes.flat:
            ax.set_xlabel(param_name)
            ax.grid(alpha=0.3)
        fig.suptitle(f"sweep: {param_name}")
        fig.tight_layout()
        fig_path = out_dir / f"trend_{param_name}.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        print(f"[SAVE] {fig_path}")

    w_eff = args.w_eff
    w_fid = args.w_fid
    def _score(r):
        eff = max(r["hardware_efficiency"], 0.0)
        fid = max(r["hardware_fidelity_corr"], 0.0)
        return (eff ** w_eff) * (fid ** w_fid)

    ranked = sorted(results, key=_score, reverse=True)
    print("\n=== Top 5 by composite score "
          f"(eff^{w_eff} * fid^{w_fid}) ===")
    for r in ranked[:5]:
        print(
            f"  [{r['index']:03d}] {r['sweep_param']}={r['sweep_value']}  "
            f"eff={r['hardware_efficiency']*100:.2f}%  "
            f"fid={r['hardware_fidelity_corr']:.4f}  "
            f"score={_score(r):.4f}"
        )

    pareto_points = [
        (r["hardware_efficiency"], r["hardware_fidelity_corr"], r["index"])
        for r in results
    ]
    pareto_idx = set(_pareto_front(pareto_points))
    pareto_entries = [r for r in results if r["index"] in pareto_idx]
    pareto_entries.sort(key=lambda r: r["hardware_efficiency"])
    print(f"\n=== Pareto frontier ({len(pareto_entries)} points) ===")
    for r in pareto_entries:
        print(
            f"  [{r['index']:03d}] {r['sweep_param']}={r['sweep_value']}  "
            f"eff={r['hardware_efficiency']*100:.2f}%  "
            f"fid={r['hardware_fidelity_corr']:.4f}"
        )

    fig, ax = plt.subplots(figsize=(7, 6))
    all_eff = [r["hardware_efficiency"] * 100 for r in results]
    all_fid = [r["hardware_fidelity_corr"] for r in results]
    ax.scatter(all_eff, all_fid, c="0.6", label="all")
    pe = [r["hardware_efficiency"] * 100 for r in pareto_entries]
    pf = [r["hardware_fidelity_corr"] for r in pareto_entries]
    ax.plot(pe, pf, "ro-", label="Pareto")
    best = ranked[0]
    ax.plot(best["hardware_efficiency"] * 100,
            best["hardware_fidelity_corr"], "b*", ms=16, label="best composite")
    ax.set_xlabel("efficiency [%]")
    ax.set_ylabel("fidelity_corr")
    ax.set_title("Sweep: efficiency vs fidelity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    scatter_path = out_dir / "pareto.png"
    fig.savefig(scatter_path, dpi=120)
    plt.close(fig)
    print(f"[SAVE] {scatter_path}")

    best_path = out_dir / "best_config.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump({
            "composite_weights": {"w_eff": w_eff, "w_fid": w_fid},
            "best_composite": ranked[0],
            "pareto_frontier": pareto_entries,
        }, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {best_path}")


if __name__ == "__main__":
    main()
