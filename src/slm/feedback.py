"""Camera-feedback hardware orchestration for closed-loop SLM uniformity.

Algorithm core lives in :mod:`slm.gs` (see :func:`slm.gs.gs_phase_correction`).
This module handles:
  * loading camera BMP frames and extracting an ROI,
  * resampling/orienting the ROI from camera pitch to focal-plane pitch,
  * embedding the measured patch into a full focal-plane amplitude,
  * auto-detecting the dihedral camera-vs-focal orientation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.ndimage import zoom

from slm.dataio import load_bmp


def load_camera_roi(
    after_path: str | Path,
    before_path: str | Path | None,
    bbox: Sequence[int],
    dark_subtract: bool = True,
) -> np.ndarray:
    """Load camera BMP, optional dark-subtract, and crop to ``bbox``.

    Parameters
    ----------
    after_path : path to the SLM-on capture (BMP).
    before_path : path to the SLM-off capture (BMP); may be ``None`` to
        skip dark subtraction.
    bbox : ``[y0, y1, x0, x1]`` — half-open pixel ROI.
    dark_subtract : if True and ``before_path`` is given with matching
        shape, subtract before from after with ``clip(_, 0, None)``.

    Returns
    -------
    roi : float64 (y1-y0, x1-x0) intensity (camera grayscale, dark-corrected).
    """
    after = load_bmp(after_path).astype(np.float64)
    if dark_subtract and before_path is not None:
        before = load_bmp(before_path).astype(np.float64)
        if before.shape == after.shape:
            after = np.clip(after - before, 0.0, None)
    y0, y1, x0, x1 = (int(v) for v in bbox)
    return after[y0:y1, x0:x1]


def _orient(patch: np.ndarray, rotation_deg: int, flip_y: bool) -> np.ndarray:
    """Apply a dihedral D4 element to ``patch``.

    Rotation in {0, 90, 180, 270} composed with optional y-flip covers
    the full 8-element D4 group. ``flip_x`` is redundant: it equals
    ``rotate(180) ∘ flip_y`` and would only double-count candidates.
    """
    if rotation_deg % 90 != 0:
        raise ValueError(f"rotation_deg must be a multiple of 90, got {rotation_deg}")
    out = np.rot90(patch, k=(rotation_deg // 90) % 4)
    if flip_y:
        out = np.flip(out, axis=0)
    return out


def embed_camera_into_focal(
    roi_2d: np.ndarray,
    ideal_focal_amp: np.ndarray,
    target_center_yx_fpx: Sequence[float],
    cam_pitch_um: float,
    focal_pitch_um: float,
    rotation_deg: int = 0,
    flip_y: bool = False,
    energy_normalize: bool = True,
) -> tuple[np.ndarray, dict]:
    """Resample camera ROI onto focal grid and paste into ideal amplitude.

    Parameters
    ----------
    roi_2d : camera intensity ROI (already dark-subtracted). Will be
        ``sqrt``'d internally to get amplitude (``|E|`` from ``|E|^2``).
    ideal_focal_amp : (Ny, Nx) full focal-plane amplitude predicted by
        the current SLM phase. Acts as the surround for embedding.
    target_center_yx_fpx : ``(row, col)`` of the sheet center in focal
        pixel coordinates (typically grid_center + target_shift).
    cam_pitch_um, focal_pitch_um : physical pitch values.
    rotation_deg, flip_y : dihedral orientation alignment, see
        :func:`align_camera_to_focal`.
    energy_normalize : if True, scale the pasted patch so its energy
        equals the ideal amplitude's energy in the same window.

    Returns
    -------
    embedded : float64 (Ny, Nx) — copy of ``ideal_focal_amp`` with the
        patch overwritten in place.
    info : diagnostic dict with patch shape, paste origin, and the
        energy scale factor used.
    """
    cam_amp = np.sqrt(np.clip(np.asarray(roi_2d, dtype=np.float64), 0.0, None))
    cam_amp = _orient(cam_amp, rotation_deg, flip_y)

    zoom_factor = float(cam_pitch_um) / float(focal_pitch_um)
    patch = zoom(cam_amp, zoom_factor, order=1)
    ph, pw = patch.shape
    if ph == 0 or pw == 0:
        raise ValueError(
            f"Resampled patch is empty (cam ROI {cam_amp.shape}, zoom {zoom_factor})"
        )

    Ny, Nx = ideal_focal_amp.shape
    cy, cx = target_center_yx_fpx
    y0 = int(round(cy - ph / 2.0))
    x0 = int(round(cx - pw / 2.0))
    y1, x1 = y0 + ph, x0 + pw
    if y0 < 0 or x0 < 0 or y1 > Ny or x1 > Nx:
        raise ValueError(
            f"Patch [{y0}:{y1}, {x0}:{x1}] does not fit in focal {Ny}x{Nx}"
        )

    embedded = np.array(ideal_focal_amp, dtype=np.float64, copy=True)
    scale = 1.0
    if energy_normalize:
        patch_energy = float(np.sum(patch * patch))
        ideal_energy = float(np.sum(embedded[y0:y1, x0:x1] ** 2))
        if patch_energy > 0:
            scale = float(np.sqrt(ideal_energy / patch_energy))
            patch = patch * scale
    embedded[y0:y1, x0:x1] = patch

    info = {
        "patch_shape": (ph, pw),
        "paste_yx": (y0, x0),
        "energy_scale": scale,
        "rotation_deg": int(rotation_deg) % 360,
        "flip_y": bool(flip_y),
    }
    return embedded, info


def align_camera_to_focal(
    roi_2d: np.ndarray,
    ideal_focal_amp: np.ndarray,
    target_center_yx_fpx: Sequence[float],
    cam_pitch_um: float,
    focal_pitch_um: float,
) -> dict:
    """Search dihedral D4 orientations for best camera-to-focal alignment.

    Tries each of the 8 combinations ``rot ∈ {0, 90, 180, 270}`` ×
    ``flip_y ∈ {False, True}``. For each candidate, embeds the ROI
    using :func:`embed_camera_into_focal` and computes the normalized
    cross-correlation between the pasted patch and the ideal amplitude
    in the same window.

    Returns
    -------
    dict with keys::
        best : {"rotation_deg", "flip_y"} of the highest-scoring candidate
        scores : list of dicts (one per candidate) with score + params
        best_index : index into ``scores`` of the winner
    """
    candidates = []
    for rot in (0, 90, 180, 270):
        for flip in (False, True):
            embedded, info = embed_camera_into_focal(
                roi_2d=roi_2d,
                ideal_focal_amp=ideal_focal_amp,
                target_center_yx_fpx=target_center_yx_fpx,
                cam_pitch_um=cam_pitch_um,
                focal_pitch_um=focal_pitch_um,
                rotation_deg=rot,
                flip_y=flip,
                energy_normalize=True,
            )
            y0, x0 = info["paste_yx"]
            ph, pw = info["patch_shape"]
            patch = embedded[y0 : y0 + ph, x0 : x0 + pw]
            ref = ideal_focal_amp[y0 : y0 + ph, x0 : x0 + pw]
            num = float(np.sum(patch * ref))
            den = float(np.linalg.norm(patch) * np.linalg.norm(ref))
            score = num / den if den > 0 else 0.0
            candidates.append(
                {
                    "rotation_deg": rot,
                    "flip_y": flip,
                    "score": score,
                    "patch_shape": info["patch_shape"],
                    "paste_yx": info["paste_yx"],
                    "energy_scale": info["energy_scale"],
                }
            )

    best_index = int(np.argmax([c["score"] for c in candidates]))
    best = candidates[best_index]
    return {
        "best": {"rotation_deg": best["rotation_deg"], "flip_y": best["flip_y"]},
        "best_index": best_index,
        "scores": candidates,
    }
