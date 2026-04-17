"""Parameter sweep for top-hat hologram: generate multiple payloads varying
one parameter at a time, for hardware calibration on the Windows runner.

Usage::

    uv run python scripts/sweep_tophat.py                          # default config
    uv run python scripts/sweep_tophat.py scripts/my_sweep.json    # custom config

Reads a JSON config with ``"base"`` (default param dict) and ``"sweep"``
(param name -> list of values).  Outputs indexed payloads + a manifest into
``scripts/sweep_tophat/``.

See issue #15 for the sweep protocol and parameter tiers.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slm.cgm import CGM_phase_generate, CGMConfig, _initial_phase
from slm.generation import SLM_class
from slm import imgpy
from slm.metrics import fidelity as _fidelity, efficiency as _efficiency
from slm.propagation import fft_propagate
from slm.targets import measure_region as _measure_region, mask_from_target

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = "scripts/sweep_tophat"
DEFAULT_CONFIG = "scripts/sweep_tophat_config.json"

# Tier 1 params only affect post-CGM processing (Fresnel / LUT / camera).
# Everything else requires a full CGM recompute.
TIER1_PARAMS = {"fresnel_sd", "LUT", "etime_us", "n_avg"}

# Params that require re-initializing the SLM object (changes input beam).
SLM_REINIT_PARAMS = {"beamwaist"}


def setup_slm(beamwaist=None):
    """SLM setup (reads hamamatsu_test_config.json).

    If *beamwaist* is given (in µm), overrides the config value before
    computing the input Gaussian amplitude.
    """
    SLM = SLM_class()
    SLM.arraySizeBit = [10, 10]
    if beamwaist is not None:
        SLM.beamwaist = beamwaist
    SLM.image_init(
        initGaussianPhase_user_defined=np.zeros((1024, 1024)), Plot=False
    )
    return SLM


def run_cgm(SLM, params):
    """Run CGM with the given params dict.  Returns (phase_wrapped, screen_raw, F, eta)."""
    targetAmp = SLM.top_hat_target(radius=params["tophat_radius"])

    init_phi = _initial_phase(
        (int(SLM.ImgResY), int(SLM.ImgResX)),
        CGMConfig(R=params["cgm_R"], D=params["cgm_D"], theta=params["cgm_theta"]),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    SLM_Phase = CGM_phase_generate(
        torch.tensor(SLM.initGaussianAmp),
        torch.from_numpy(init_phi),
        torch.from_numpy(targetAmp),
        max_iterations=params["cgm_max_iterations"],
        steepness=params["cgm_steepness"],
        eta_min=params["eta_min"],
        Plot=False,
    )
    wall_time = time.perf_counter() - t0

    phase_np = SLM_Phase.cpu().clone().numpy()
    phase_wrapped = np.angle(np.exp(1j * phase_np))
    screen_raw = SLM.phase_to_screen(phase_wrapped)

    region_np = _measure_region(targetAmp.shape, targetAmp, margin=5)
    E_out = fft_propagate(SLM.initGaussianAmp * np.exp(1j * phase_wrapped))
    F = float(_fidelity(E_out, targetAmp, region_np))
    eta = float(_efficiency(E_out, region_np))

    return phase_wrapped, screen_raw, F, eta, wall_time, device


def apply_post_processing(SLM, screen_raw, params):
    """Apply Fresnel lens + LUT/calibration correction to a cached screen."""
    W, H = SLM.SLMRes
    cx, cy = W // 2, H // 2

    fresnel = SLM.fresnel_lens_phase_generate(params["fresnel_sd"], cx, cy)[0]
    screen_shift = (
        (screen_raw.astype(np.int32) + fresnel.astype(np.int32)) % 256
    ).astype(np.uint8)

    screen_final = imgpy.SLM_screen_Correct(
        screen_shift, LUT=params["LUT"],
        correctionImgPath="calibration/CAL_LSH0905549_1013nm.bmp",
    )
    return screen_final


def save_preview(path, SLM, targetAmp, E_out, region, screen_final, F, eta, sweep_param, sweep_val):
    """Single-panel preview PDF for a sweep point."""
    target_mask = mask_from_target(targetAmp)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(np.abs(targetAmp) ** 2, cmap="hot")
    axes[0].set_title("Target |tau|^2")

    out_int = (np.abs(E_out) ** 2) * region
    axes[1].imshow(out_int, cmap="hot")
    axes[1].set_title(f"Sim output\nF={F:.4f}, eta={100*eta:.2f}%")

    axes[2].imshow(screen_final, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"SLM screen\n{sweep_param}={sweep_val}")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    with open(config_path) as f:
        config = json.load(f)

    base = config["base"]
    sweeps = config["sweep"]

    os.makedirs(OUT_DIR, exist_ok=True)

    SLM = setup_slm()
    print(f"Compute grid: ({SLM.ImgResY}, {SLM.ImgResX})  "
          f"focal pitch = {SLM.Focalpitchx:.3f} um/px")
    print(f"SLM native:   ({SLM.SLMRes[1]}, {SLM.SLMRes[0]})")
    print(f"Config: {config_path}")
    print(f"Sweep params: {list(sweeps.keys())}")

    manifest = []
    idx = 0

    for param_name, values in sweeps.items():
        is_tier1 = param_name in TIER1_PARAMS
        tier_label = "Tier 1 (post-CGM)" if is_tier1 else "Tier 2 (CGM recompute)"
        print(f"\n{'='*60}")
        print(f"Sweeping {param_name}: {len(values)} points  [{tier_label}]")
        print(f"{'='*60}")

        # For Tier 1: run CGM once with base params, cache results
        cached_phase = cached_screen = None
        cached_F = cached_eta = cached_wall = cached_device = None
        if is_tier1:
            print(f"[CGM] computing base hologram (once)...")
            cached_phase, cached_screen, cached_F, cached_eta, cached_wall, cached_device = run_cgm(SLM, base)
            print(f"[CGM] F={cached_F:.6f}  eta={cached_eta*100:.2f}%  ({cached_wall:.2f}s)")

        for val in values:
            params = {**base, param_name: val}

            if is_tier1:
                phase_wrapped = cached_phase
                screen_raw = cached_screen
                F, eta = cached_F, cached_eta
                cgm_wall_time = cached_wall
                cgm_device = cached_device
            else:
                # Re-init SLM if sweeping a beam/hardware parameter
                if param_name in SLM_REINIT_PARAMS:
                    SLM = setup_slm(beamwaist=params.get("beamwaist"))
                print(f"\n[CGM] {param_name}={val} ...")
                phase_wrapped, screen_raw, F, eta, cgm_wall_time, cgm_device = run_cgm(SLM, params)
                print(f"  F={F:.6f}  eta={eta*100:.2f}%  ({cgm_wall_time:.2f}s)")

            screen_final = apply_post_processing(SLM, screen_raw, params)

            # Save payload
            payload_path = f"{OUT_DIR}/{idx:03d}_payload.npz"
            np.savez_compressed(payload_path, slm_screen=screen_final)

            # Save preview
            targetAmp = SLM.top_hat_target(radius=params["tophat_radius"])
            region_np = _measure_region(targetAmp.shape, targetAmp, margin=5)
            E_out = fft_propagate(SLM.initGaussianAmp * np.exp(1j * phase_wrapped))
            preview_path = f"{OUT_DIR}/{idx:03d}_preview.pdf"
            save_preview(preview_path, SLM, targetAmp, E_out, region_np, screen_final, F, eta, param_name, val)

            entry = {
                "index": idx,
                "sweep_param": param_name,
                "sweep_value": val,
                "payload": payload_path,
                "preview": preview_path,
                "runner_defaults": {
                    "etime_us": params["etime_us"],
                    "n_avg": params["n_avg"],
                    "monitor": 1,
                },
                "fresnel_shift_distance_um": params["fresnel_sd"],
                "LUT": params["LUT"],
                "tophat_radius_px": params["tophat_radius"],
                "cgm_R": params["cgm_R"],
                "cgm_D": params["cgm_D"],
                "cgm_theta": params["cgm_theta"],
                "cgm_steepness": params["cgm_steepness"],
                "cgm_max_iterations": params["cgm_max_iterations"],
                "eta_min": params["eta_min"],
                "beamwaist": params.get("beamwaist", SLM.beamwaist),
                "cgm_wall_time_s": round(cgm_wall_time, 3),
                "cgm_device": cgm_device,
                "fidelity": round(F, 6),
                "efficiency": round(eta, 6),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            manifest.append(entry)

            print(f"  [{idx:03d}] {param_name}={val}  F={F:.4f}  eta={eta*100:.1f}%  -> {payload_path}")
            idx += 1

    # Save manifest
    manifest_path = f"{OUT_DIR}/sweep_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Sweep complete: {idx} payloads in {OUT_DIR}/")
    print(f"Manifest: {manifest_path}")
    print(f"\nNext step:")
    print(f"    ./push_run.sh {OUT_DIR}/<payload_file>.npz   # per sweep point")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
