"""Generate / inspect a single light-sheet sweep point.

Unlike ``sweep_sheet.py`` (which expands the whole manifest upfront
and runs ~10 CGM solves back-to-back), this helper drives ONE index
at a time so the outer loop can push → run → pull → analyze → verify
before the next CGM solve is paid for.

Usage::

    uv run python scripts/sheet/sweep_one.py --index 0
        # prints {sweep_param, sweep_value, ...}, generates payload if missing

    uv run python scripts/sheet/sweep_one.py --index 33 --force
        # regenerate payload even if file exists

    uv run python scripts/sheet/sweep_one.py --list
        # dump the deterministic (index -> param) expansion from the config

The deterministic expansion matches sweep_sheet.py's iteration order,
so index N refers to the same point whether built here or via the
batch script.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sweep_sheet import (  # noqa: E402
    DEFAULT_CONFIG, OUT_DIR, TIER1_PARAMS, SLM_REINIT_PARAMS,
    setup_slm, run_cgm, apply_post_processing, save_preview,
)
from slm.propagation import fft_propagate  # noqa: E402
from slm.targets import measure_region as _measure_region  # noqa: E402


def expand_manifest(config: dict) -> list[dict]:
    """Deterministic index -> flat param dict expansion, matching
    sweep_sheet.py iteration order.
    """
    base = config["base"]
    sweeps = config["sweep"]
    out: list[dict] = []
    idx = 0
    for param_name, values in sweeps.items():
        for val in values:
            params = {**base, param_name: val}
            out.append({
                "index": idx,
                "sweep_param": param_name,
                "sweep_value": val,
                "params": params,
            })
            idx += 1
    return out


def _entry_to_manifest_dict(entry: dict, payload_path: str,
                            preview_path: str, F: float, eta: float,
                            cgm_wall_time: float, cgm_device: str,
                            focal_pitch_x: float, focal_pitch_y: float,
                            beamwaist: float) -> dict:
    p = entry["params"]
    return {
        "index": entry["index"],
        "sweep_param": entry["sweep_param"],
        "sweep_value": entry["sweep_value"],
        "payload": payload_path,
        "preview": preview_path,
        "runner_defaults": {
            "etime_us": p["etime_us"],
            "n_avg": p["n_avg"],
            "monitor": 1,
        },
        "fresnel_shift_distance_um": p["fresnel_sd"],
        "LUT": p["LUT"],
        "sheet_flat_width_px": p["sheet_flat_width"],
        "sheet_gaussian_sigma_px": p["sheet_gaussian_sigma"],
        "sheet_angle_rad": p["sheet_angle"],
        "sheet_edge_sigma_px": p["sheet_edge_sigma"],
        "cgm_R": p["cgm_R"],
        "cgm_D": p["cgm_D"],
        "cgm_theta": p["cgm_theta"],
        "cgm_steepness": p["cgm_steepness"],
        "cgm_max_iterations": p["cgm_max_iterations"],
        "eta_min": p["eta_min"],
        "beam_center_dx_um": p.get("beam_center_dx_um", 0.0),
        "beam_center_dy_um": p.get("beam_center_dy_um", 0.0),
        "beamwaist": beamwaist,
        "focal_pitch_x_um_per_px": round(float(focal_pitch_x), 4),
        "focal_pitch_y_um_per_px": round(float(focal_pitch_y), 4),
        "cgm_wall_time_s": round(float(cgm_wall_time), 3),
        "cgm_device": cgm_device,
        "sim_fidelity": round(float(F), 6) if F is not None else None,
        "sim_efficiency": round(float(eta), 6) if eta is not None else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def generate_point(entry: dict, force: bool = False) -> dict:
    """Generate payload + preview for *entry* (or reuse existing)."""
    idx = entry["index"]
    p = entry["params"]
    payload_path = f"{OUT_DIR}/{idx:03d}_payload.npz"
    preview_path = f"{OUT_DIR}/{idx:03d}_preview.pdf"
    os.makedirs(OUT_DIR, exist_ok=True)

    existing = Path(payload_path).is_file() and not force
    if existing:
        print(f"[idx={idx:03d}] payload exists, reusing: {payload_path}")
        # We still need focal pitch / beamwaist for manifest. Build a
        # throwaway SLM without running CGM.
        bc = (float(p.get("beam_center_dx_um", 0.0)),
              float(p.get("beam_center_dy_um", 0.0)))
        SLM = setup_slm(beamwaist=p.get("beamwaist"), beam_center_um=bc)
        return _entry_to_manifest_dict(
            entry, payload_path, preview_path,
            F=None, eta=None, cgm_wall_time=0.0, cgm_device="cached",
            focal_pitch_x=SLM.Focalpitchx, focal_pitch_y=SLM.Focalpitchy,
            beamwaist=SLM.beamwaist,
        )

    print(f"[idx={idx:03d}] generating ({entry['sweep_param']}={entry['sweep_value']})")
    bc = (float(p.get("beam_center_dx_um", 0.0)),
          float(p.get("beam_center_dy_um", 0.0)))
    SLM = setup_slm(beamwaist=p.get("beamwaist"), beam_center_um=bc)
    phase_wrapped, screen_raw, F, eta, wall, dev = run_cgm(SLM, p)
    screen_final = apply_post_processing(SLM, screen_raw, p)
    np.savez_compressed(payload_path, slm_screen=screen_final)

    targetAmp = SLM.light_sheet_target(
        flat_width=p["sheet_flat_width"],
        gaussian_sigma=p["sheet_gaussian_sigma"],
        angle=p["sheet_angle"],
        edge_sigma=p["sheet_edge_sigma"],
    )
    region_np = _measure_region(targetAmp.shape, targetAmp, margin=5)
    E_out = fft_propagate(SLM.initGaussianAmp * np.exp(1j * phase_wrapped))
    save_preview(
        preview_path, SLM, targetAmp, E_out, region_np, screen_final, F, eta,
        entry["sweep_param"], entry["sweep_value"],
    )

    print(f"  F={F:.4f}  eta={eta*100:.2f}%  ({wall:.2f}s on {dev})")
    return _entry_to_manifest_dict(
        entry, payload_path, preview_path, F, eta, wall, dev,
        SLM.Focalpitchx, SLM.Focalpitchy, SLM.beamwaist,
    )


def upsert_manifest_entry(manifest_path: Path, entry_dict: dict) -> list[dict]:
    """Insert or replace *entry_dict* in the on-disk manifest by index."""
    manifest: list[dict] = []
    if manifest_path.is_file():
        with open(manifest_path) as f:
            manifest = json.load(f)
    manifest = [m for m in manifest if m["index"] != entry_dict["index"]]
    manifest.append(entry_dict)
    manifest.sort(key=lambda m: m["index"])
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Light-sheet sweep: one point at a time.")
    ap.add_argument("--index", type=int, help="Index to generate/inspect.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if payload exists.")
    ap.add_argument("--list", action="store_true",
                    help="Print the deterministic index expansion.")
    ap.add_argument("--manifest", default=f"{OUT_DIR}/sweep_manifest.json",
                    help="Manifest file to upsert into.")
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    expansion = expand_manifest(config)

    if args.list:
        for e in expansion:
            print(f"{e['index']:03d}  {e['sweep_param']:22s}  {e['sweep_value']}")
        return

    if args.index is None:
        print("ERROR: --index required (or use --list)", file=sys.stderr)
        sys.exit(2)

    entry = next((e for e in expansion if e["index"] == args.index), None)
    if entry is None:
        print(f"ERROR: index {args.index} out of range "
              f"(0..{len(expansion)-1})", file=sys.stderr)
        sys.exit(1)

    md = generate_point(entry, force=args.force)
    upsert_manifest_entry(Path(args.manifest), md)
    print(f"[manifest] upserted index {md['index']} into {args.manifest}")
    print(json.dumps({
        "index": md["index"],
        "sweep_param": md["sweep_param"],
        "sweep_value": md["sweep_value"],
        "sim_fidelity": md["sim_fidelity"],
        "sim_efficiency": md["sim_efficiency"],
        "payload": md["payload"],
    }, indent=2))


if __name__ == "__main__":
    main()
