# Initialisation comparison: stationary-phase vs Bowman guess-phase

Two payloads identical in every respect except the CGM warm-start phase.

- **Stationary-phase init** — exact geometric-optics ray redistribution (Dickey–Romero–Holswade closed form) from `references/Top Hat Beam.pdf`, implemented in `slm.initial_phase.stationary_phase_light_sheet`.  Script: `scripts/sheet/testfile_sheet.py`.
- **Bowman init** — structured guess phase `φ(p,q) = R·(p²+q²) + D·(p·cosθ + q·sinθ)` from `references/top-hat.tex` (Bowman et al., Opt. Express 2017).  Script: `scripts/sheet/testfile_sheet_bowman.py`.

Target, post-hoc Fresnel, LUT, calibration, capture settings, CGM hyperparameters (steepness, eta_min=0.6, eta_steepness=11) all identical.  Only the starting `init_phi` differs.

## Metrics (measured on camera, analysed against the `flat_width=25 px` reference)

| metric | stat cgm=5 | bow cgm=5 | stat cgm=4000 | bow cgm=4000 |
|---|---:|---:|---:|---:|
| efficiency | **20.14 %** | 0.71 % | 1.45 % | **2.97 %** |
| fidelity_corr | 0.195 | **0.350** | 0.151 | **0.473** |
| fidelity_overlap | 0.019 | 0.105 | 0.047 | **0.156** |
| aspect ratio | **2.21** | 0.42 | 0.45 | 0.67 |
| peak_count | 2 | 1 | 1 | 1 |
| asymmetry (abs) | **0.039** | 0.098 | 0.128 | **0.024** |

**See `INIT_COMPARE.png` (4-panel image + profile) and the matching `compare_*_preview.png` analysis previews.**

## Interpretation

**cgm = 5 (shallow polish).** The stationary-phase init already carries most of the solution: it places the target intensity in the right region with ≈ 20 % efficiency and produces an extended line (aspect 2.2, two visible peaks).  CGM only has to nudge it.  The Bowman init is a diffuse quadratic+linear guess — with only five iterations CGM hasn't had time to concentrate energy into the target region, so efficiency is just 0.7 % and the camera shows a small saturated blob.  **Winner at low iter: stationary.**

**cgm = 4000 (full convergence).** Both runs reach local minima of the cost function.  Here the stationary seed has pulled CGM into a tight single-peak minimum — the pathological chain-of-spots neighbourhood, compressed here into one dominant peak at 1.45 % efficiency, fid_corr 0.151.  The Bowman seed, starting from a simple quadratic+linear phase, converges instead to a broader shape that hugs the flat-top target: **3.1× higher fid_corr (0.473), 3.3× higher fid_overlap (0.156), 2.0× higher efficiency (2.97 %), 5.3× lower asymmetry (0.024)**.  **Winner at full iter: Bowman.**

This matches the paper's claim that the structured guess phase `R·(p²+q²) + D·(p·cosθ + q·sinθ)` actively suppresses optical vortices during optimisation, allowing CGM to relax into higher-fidelity minima.  Our prior working finding that the stationary-phase init plus only a handful of CGM iterations gives a clean line is *also* valid — different use cases reward different inits.

## Practical recommendation

| use case | init to pick | cgm_max_iter |
|---|---|---|
| fast iteration, hardware closed-loop (each push+capture costs ≈ 90 s) | **stationary** | 5 |
| highest-fidelity single shot, willing to let CGM run to convergence (~3 s on GPU) | **Bowman** | 4000 |
| absolute best shape uniformity with a smooth low-contrast sheet | **Bowman** | 4000 |

The project's current `testfile_sheet.py` default (cgm_max_iterations = 4000, stationary init) puts us in the *worst quadrant* of the table: spot-chain minimum with 1.45 % efficiency.  Either switching to Bowman init at 4000 iter (→ fid 0.473) or dropping to 5 iter with stationary (→ aspect 2.2, continuous line) is strictly better.

## Files

- `../../scripts/sheet/testfile_sheet_bowman.py` — new parallel driver (Bowman init)
- `../../scripts/sheet/testfile_sheet.py` — existing (stationary-phase init)
- `compare_STATIONARY_metrics.json`, `compare_STATIONARY_preview.png`, `compare_STATIONARY_cgm5_metrics.json`, `compare_STATIONARY_cgm5_preview.png`
- `compare_BOWMAN_metrics.json`, `compare_BOWMAN_preview.png`, `compare_BOWMAN_cgm5_metrics.json`, `compare_BOWMAN_cgm5_preview.png`
- `INIT_COMPARE.png` — 4-panel side-by-side (images + profiles)
- `BOWMAN_vs_STATIONARY.png` — 2-panel at cgm=4000 with metrics annotation
- Payloads + archived BMPs are at the payload/ and data/ locations; raw BMPs were pulled to `/tmp/{stat,bow}{,5}_{before,after}.bmp` for cross-run analysis.

## Reproduce

```bash
# Build + hardware-run both at cgm=4000 (default)
uv run python scripts/sheet/testfile_sheet.py
./push_run.sh payload/sheet/testfile_sheet_payload.npz

uv run python scripts/sheet/testfile_sheet_bowman.py
./push_run.sh payload/sheet/testfile_sheet_bowman_payload.npz

# Sweep cgm iterations via env override
SLM_CGM_MAX_ITER=5 uv run python scripts/sheet/testfile_sheet_bowman.py
```

Both scripts read `SLM_BCM_DX_UM`, `SLM_BCM_DY_UM`, `SLM_CGM_MAX_ITER`, `SLM_SETTING_ETA`, `SLM_CGM_ETA_STEEPNESS` env overrides — you can drive them from a single sweep script or the existing `scripts/sheet/closed_loop_sheet.py`.
