# Light-sheet uniformity sweep log

End state of this session (2026-04-19) is **not** at `demo.png` standard
(`RMS 1.02 %, Pk-Pk 4.36 %`).  Best on-camera result was `RMS 10.79 %,
Pk-Pk 37.08 %` with `fresnel_sd = 1200 µm`, all other parameters at the
baseline listed below.

Main blocker: the physical setup has a camera alignment / defocus issue
(the two sheet lobes are visibly asymmetric, left vs. right) that no
software-only sweep can fix.  Plan is to re-align the camera, then rerun
the sweeps below with the fresh alignment.

---

# Baseline

![baseline light sheet](Figures/baseline_sheet.png)

**Captured:** 2026-04-19 (`data/sheet/testfile_sheet_after.bmp`, `n_avg=20` frames averaged)
**Phase payload:** `payload/sheet/testfile_sheet_payload.npz`
**Params sidecar:** `payload/sheet/testfile_sheet_params.json`

### Optics / illumination (`hamamatsu_test_config.json`)

| key          | value   |
| ------------ | ------- |
| `beamwaist`  | 5500 µm |
| `focallength`| 200 000 µm |
| `wavelength` | 1.013 µm |
| `pixelpitch` | 12.5 µm |
| `SLMRes`     | 1272 × 1024 |

### CGM / target parameters (`scripts/sheet/testfile_sheet.py`)

| parameter                | value |
| ------------------------ | ----- |
| compute grid (`arraySizeBit`) | `[10, 10]` → 1024 × 1024 |
| `sheet_flat_width`       | 9 px (≈ 142.45 µm focal) |
| `sheet_gaussian_sigma`   | 1 px perpendicular |
| `sheet_edge_sigma`       | 0 px |
| `sheet_angle`            | 0 rad (horizontal) |
| `target_shift_fpx`       | 20 px (≈ 316 µm off zero-order) |
| `cgm_steepness`          | 9 |
| `cgm_max_iterations`     | 4000 (env `SLM_CGM_MAX_ITER`) |
| `setting_eta`            | 0.1 (env `SLM_SETTING_ETA`) |
| `cgm_eta_steepness`      | 7 (env `SLM_CGM_ETA_STEEPNESS`) |
| `fresnel_sd`             | **1200 µm**  (updated from 800 — see sweep below) |
| seed                     | `stationary_phase_sheet` (no cylindrical term) |
| camera exposure (`etime_us`) | 1500 µs (kept below the 8-bit clip) |
| camera averaging (`n_avg`) | 20 frames |

### Measured result on the Vimba camera (baseline = `fresnel_sd=800`)

- ROI bbox (y0,y1,x0,x1): `(1672, 1712, 1934, 2040)` → 40 × 106 px
- Major axis: x (horizontal sheet, as designed)
- Flat-top fit window: `[19, 106)`, width 87 px, mean 79.8
- **RMS: 24.65 %**  —  **Peak-to-peak: 111.88 %**
- 2-lobe spot-chain is visible in the heatmap (left peak ≈ 90, right peak ≈ 105, valley ≈ 70).  This is the "CGM-to-convergence" regime called out in `testfile_sheet.py` comments.

A previously reported 9.50 % RMS baseline was an artefact: the camera was
clipping at 255 (exposure 4000 µs), and the saturated region masqueraded
as a flat top in the fit.  Once exposure was lowered to 1500 µs and
averaging raised to 20 frames, the un-clipped profile revealed the true
two-lobe structure.

---

# Parameter sweeps explored this session

### 1. `etime_us` (camera exposure)

| etime_us | max signal | note                                        |
| -------- | ---------- | ------------------------------------------- |
| 3500     | 255 (clip) | caused the misleading 9.5 % "flat" baseline |
| 1500     | ~140       | no clipping — used for all following sweeps |
| 600      | 255 (clip) | too high once power concentrates (low iters) |
| 200      | ~140       | used for pure-seed trials                    |

`etime_us=1500` is the default going forward.  `scripts/sheet/testfile_sheet.py`
now accepts `SLM_ETIME_US` env override.

### 2. `cgm_max_iterations`

| iters | shape (@ flat_width=9, edge=0) | comment |
| ----- | ------------------------------ | ------- |
| 0     | single bright blob (Gaussian)  | seed alone — wider than flat_width; low RMS is fit artefact |
| 5     | 2 very bright saturated lobes  | documented "5-iter polish" did **not** produce a continuous sheet; claim in comments is stale |
| 50    | hops to a different focal spot | with `edge_sigma=3` the optimizer moved power to a second minimum |
| 4000  | 2 separated lobes with valley  | the stable "baseline" shape |

### 3. `sheet_flat_width`

| flat_width | effect |
| ---------- | ------ |
| 9  (baseline) | 2-lobe output |
| 12 | sharp central peak, fit collapses to 4 px window |
| 15 | very dim (mean 4 @ etime 200); power disperses beyond the camera’s dynamic range |

### 4. `sheet_edge_sigma`

Soft-edge targets pushed power into a completely different focal spot; the
detector found a ~40 × 30 Gaussian blob instead of a sheet.  Not useful
until the camera is realigned.

### 5. `fresnel_sd`  **(winner of this session)**

| `fresnel_sd` (µm) | RMS %   | Pk-Pk % | mean | fit window width |
| ----------------- | ------- | ------- | ---- | ---------------- |
| 400               | 25.00   | 110.33  | 92.8 | 90 |
| 600               | 24.75   | 110.92  | 99.5 | 89 |
| 800 (old baseline)| 24.65   | 111.88  | 79.8 | 87 |
| 1000              | 26.40   | 117.10  |100.8 | 87 |
| **1200**          | **10.79** | **37.08** | **112.4** | **63** |
| 1400              | 24.76   | 112.71  |105.3 | 85 |
| 1600              | 25.25   | 114.00  |102.3 | 85 |
| 1800              | 25.70   | 116.15  | 99.0 | 85 |
| 2000              | 24.30   | 111.89  | 94.7 | 83 |

![best sheet — fresnel_sd=1200](Figures/best_sheet_fresnel1200.png)

The dramatic RMS drop at `fresnel_sd=1200` is partially real (the sheet
defocused more cleanly) and partially an artefact of the tighter fit
window the fitter found; the underlying 2-lobe / valley shape is still
there, the fit just stopped including the rising edges.  `fresnel_sd=1200`
is the new baseline.

Raw per-point artefacts: `docs/sweep_sheet/param_sweep/fresnel_sd_*_{plot.png,result.json}`.

---

# Camera closed-loop optimization

**Script:** `scripts/sheet/closed_loop_sheet.py`
**Date:** 2026-04-19
**Best iteration artifacts:** `scripts/sheet/closed_loop_sheet.png` · `scripts/sheet/closed_loop_sheet_result.json` · per-iter history under `docs/sweep_sheet/closed_loop/`

![closed-loop best iteration](Figures/closed_loop_sheet.png)

### Algorithm

Target-amplitude re-weighting driven by the camera profile.  Each iteration:

1. Multiply the target’s 10 flat-top columns by the current per-bin weights.
2. Re-run CGM (4000 iters, same `steepness`/`eta_min`/`eta_steepness` as baseline).
3. Push payload, capture `after.bmp`, run `analysis_sheet.py`.
4. Bin the measured flat-top profile into 10 bins, compute `w_i = sqrt(mean / bin_i)`, apply damped update (`damping = 0.5`, clip `[0.5, 2.0]`, re-normalize to mean 1).

### Run summary (5 iterations, exposure was still clipping)

| tag     | weights range       | RMS %  | Pk-Pk % | sim_F  | sim_η %  |
| ------- | ------------------- | ------ | ------- | ------ | -------- |
| iter00  | [1.000, 1.000]      | 27.87  | 100.69  | 1.0000 | 14.91 |
| iter01  | [0.941, 1.449]      | 26.39  | 101.41  | 1.0000 | 15.72 |
| iter02  | [0.876, 1.921]      | 27.19  | 104.48  | 1.0000 | 17.02 |
| iter03  | [0.844, 2.006]      | **13.30** | **39.99** | 1.0000 | 17.35 | ← best |
| iter04  | [0.825, 1.996]      | 13.82  | 42.04   | 1.0000 | 17.24 |

### Observations

- Closed loop did **not** beat the (unclipped) fresnel_sd=1200 single-shot.
- Iter00 used all-ones weights yet measured RMS 27.87 %; the fit is
  fragile when the profile is not a clean top-hat.
- Camera saturation during this run (exposure was still 4000 µs via the
  Windows runner default) invalidated the feedback signal — the weight
  update cannot distinguish “right intensity” from “too high” once pixels
  clip.  `push_run.sh` now forwards `etime_us` / `n_avg` from the
  sidecar `params.json` so subsequent closed-loop runs respect the
  payload script's exposure choice.

### Next steps (blocked on hardware)

The amplitude-reweight loop is limited by (a) camera saturation (fixed
now via `runner_defaults` → `push_run.sh`) and (b) the camera alignment /
defocus observed at the end of this session.  Once the camera is
realigned the expected sequence is:

1. Re-run the fresnel_sd sweep (peak may shift).
2. Re-run the closed loop at the new optimum with 8–10 iterations.
3. If still above 5 % RMS, explore a gradient-descent-on-phase variant
   using the sheet-region camera intensity as loss
   (``scripts/sheet/closed_loop_sheet.py`` can be extended with a warm
   start from the best npz).

---

# Sweep of `arraySizeBit`  (new best — 2026-04-20)

Hypothesis: running CGM on a 4× finer compute grid and relying on
`SLM.phase_to_screen`'s built-in centre-crop (1024×1024 out of 4096²) gives
the optimiser higher spatial-frequency headroom to shape the flat-top edges
— same pattern as `scripts/wgs_square/testfile_wgs_square.py`.  Added an
`SLM_ARRAY_BIT` env-var knob to `scripts/sheet/testfile_sheet.py` (default
`10`, 1024²).

### Config (both runs use this session's locked-in `fresnel_sd=1200`, `etime_us=1500`, `n_avg=20`)

| env                    | `[10,10]` baseline | `[12,12]` candidate |
| ---------------------- | ------------------ | ------------------- |
| `SLM_ARRAY_BIT`        | 10 (default)       | 12                  |
| compute grid           | 1024 × 1024        | 4096 × 4096         |
| focal pitch            | 15.83 µm/px        | 3.96 µm/px          |
| `SLM_FLAT_WIDTH`       | 9 px  (≈142 µm)    | 36 px (≈142 µm)     |
| `SLM_TARGET_SHIFT_FPX` | 20                 | 80                  |
| `SLM_GAUSS_SIGMA`      | 1                  | 4                   |
| `SLM_CGM_MAX_ITER`     | 4000               | 4000                |

### Results

The analysis script's top-hat fit mis-bounds the wider softer edges the 4096
run produces (it pulls the lower vertical gray line into the rising edge),
so compare the two over the **intensity > 85 % of peak** core window:

| grid     | core px | mean  | RMS %  | Pk-Pk % |
| -------- | ------- | ----- | ----- | ------- |
| `[10,10]`| 63      | 103.8 | 10.97 | 38.85   |
| `[12,12]`| 53      | 193.8 | **4.01** | **15.48** |

Roughly **2.7 ×** better RMS and **2.5 ×** better Pk-Pk at 4096, with a
single continuous bright stripe on the 2D heatmap (no 2-lobe split).

![4096-grid light sheet](Figures/arraybit_12_sheet.png)

Per-point artefacts:

- `docs/sweep_sheet/param_sweep/arraybit_10_plot.png` + `_result.json`
- `docs/sweep_sheet/param_sweep/arraybit_12_plot.png` + `_result.json`
  (1000 iters, early confirmation)
- `docs/sweep_sheet/param_sweep/arraybit_12_cgm4000_plot.png` + `_result.json`
  (4000 iters, final)

### Follow-ups suggested by this result

1. Make `SLM_ARRAY_BIT=12` the new default after the camera alignment fix
   tomorrow, since the shape is still hardware-asymmetric.
2. The `analysis_sheet._fit_flat_top` top-hat + Gaussian-edge model is
   fragile on soft-edge profiles — consider switching to an "above-threshold
   core" metric to match what is actually usable for microscopy.
3. Re-run the `closed_loop_sheet.py` target re-weighting on top of the
   `[12,12]` base shape — much more headroom for feedback now that the
   base profile is already 4 % RMS.

---

# Tonight’s best configuration

Best payload this session:

- `payload/sheet/testfile_sheet_payload.npz`  (generated by
  `uv run python scripts/sheet/testfile_sheet.py` with the baseline env
  defaults and the locked-in `fresnel_sd = 1200` in `testfile_sheet.py`).
- Figure: `docs/Figures/best_sheet_fresnel1200.png`
- Metrics: RMS 10.79 %, Pk-Pk 37.08 %, mean 112.4.

This remains ~10× short of demo.png.  The physical setup, not the
software, is the limiting factor for this run.
