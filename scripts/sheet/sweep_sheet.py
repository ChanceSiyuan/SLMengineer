"""Shared helpers for the light-sheet sweep pipeline.

This module is imported by ``sweep_one.py`` (per-index generator) and
reused by ``analyze_sweep_sheet.py``.  It contains no CLI entry point;
use ``sweep_one.py --index N`` + ``run_one.sh N`` to drive sweeps.

Exports:
- constants: ``OUT_DIR``, ``DEFAULT_CONFIG``, ``TIER1_PARAMS``, ``SLM_REINIT_PARAMS``
- ``setup_slm(beamwaist, beam_center_um)`` — SLM_class ready for CGM
- ``run_cgm(SLM, params)`` — CGM phase + screen + sim F/eta
- ``apply_post_processing(SLM, screen_raw, params)`` — Fresnel + LUT + calibration
- ``save_preview(...)`` — 3-panel sim preview PDF per sweep point

The sweep grid is defined in ``scripts/sheet/sweep_sheet_config.json``
(``base`` + ``sweep``).  See ``sweep_journal.md`` for structure and
reproduction instructions.
"""
from __future__ import annotations

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

OUT_DIR = "scripts/sweep_sheet"
DEFAULT_CONFIG = "scripts/sheet/sweep_sheet_config.json"

# Tier 1 params only affect post-CGM processing (Fresnel / LUT / camera).
# Everything else requires a full CGM recompute.
TIER1_PARAMS = {"fresnel_sd", "LUT", "etime_us", "n_avg"}

# Params that require re-initializing the SLM object (changes input beam).
SLM_REINIT_PARAMS = {"beamwaist", "beam_center_dx_um", "beam_center_dy_um"}


def setup_slm(beamwaist=None, beam_center_um=(0.0, 0.0)):
    SLM = SLM_class()
    SLM.arraySizeBit = [10, 10]
    if beamwaist is not None:
        SLM.beamwaist = beamwaist
    SLM.image_init(
        initGaussianPhase_user_defined=np.zeros((1024, 1024)), Plot=False,
        beam_center_um=beam_center_um,
    )
    return SLM


def run_cgm(SLM, params):
    """Run CGM with the given params dict.  Returns (phase_wrapped, screen_raw, F, eta, wall_time, device)."""
    targetAmp = SLM.light_sheet_target(
        flat_width=params["sheet_flat_width"],
        gaussian_sigma=params["sheet_gaussian_sigma"],
        angle=params["sheet_angle"],
        edge_sigma=params["sheet_edge_sigma"],
    )

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


