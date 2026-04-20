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

# 1D CGM dimension-decomposition (issue #21, 2026-04-20)

**Hypothesis** (issue #21): the SLM input is a 2D-separable Gaussian
`exp(-(x²+y²)/2σ²)`.  If we force the phase constant in `y`
(`φ(x,y)=φ(x)`), the focal-plane field factors as
`F_x{exp(iφ(x))·exp(-x²/2σ²)} · F_y{exp(-y²/2σ²)}` — the y factor is
automatically the natural focal Gaussian `λf/πw₀`, and we only need to
solve a 1D CGM on x to shape the top-hat.  Saves compute (1D vs 2D FFT
at the same length) and removes the perpendicular-axis cost term the
2D loop was partially optimising.

**Implementation.**  Added `CGM_phase_generate_1d` + `_run_cgm_torch_1d`
to `src/slm/cgm.py` (1D FFT / 1D sinc envelope; optimiser body is a
literal port of the 2D loop — only the FFT/sinc helpers differ).
Added `light_sheet_1d` and `measure_region_1d` to `src/slm/targets.py`
and `SLM_class.stationary_phase_sheet_1d` to `src/slm/generation.py`.
Rewrote `scripts/1d_sheet/testfile_1dsheet.py` as the 1D analog of
`scripts/sheet/testfile_sheet.py`: it slices the centre row of
`SLM.initGaussianAmp`, runs 1D CGM, then broadcasts φ(x)→φ(x,y)
before the existing `phase_to_screen` + Fresnel + calibration pipeline.
The 2D path (`CGM_phase_generate`, `_run_cgm_torch`) is untouched and
its SHA-256-verified payload is byte-identical to before the 1D patch.

**Measured result** (over the >85 % peak core, apples-to-apples):

| quantity                     | 2D reference | **1D new**    | delta |
| ---------------------------- | ------------ | ------------- | ------ |
| core width (camera px)       | 53           | **69**        | +30 % |
| flat-top mean intensity      | 193.8        | 141.5         | −27 % |
| RMS %                        | 4.01         | **3.99**      | tie |
| Pk-Pk %                      | 15.48        | **12.96**     | −2.52 (−16 %) |
| CGM wall time (4000 iters)   | 295 s (4096²) | **66 s** (length 4096) | 4.5× faster |

![1D-CGM light sheet](Figures/1d_cgm_sheet.png)

Camera heatmap shows a continuous bright plateau — no 2-lobe spot chain.
One mild dip near the right third of the flat top remains; compatible
with the 12.96 % Pk-Pk.  The lower absolute intensity is consistent
with the y-axis Gaussian being narrower than the 2D target (natural
σ = 11.73 µm vs 2D target σ = 15.83 µm; ratio 0.74×), so less total
power is concentrated into the detector's horizontal slice.

**Conclusion.**  Matches the 2D reference on uniformity, beats it on
flat-top width and Pk-Pk, and runs 4-5× faster.  Hypothesis confirmed.
Raw artefacts: `docs/sweep_sheet/1d_vs_2d/1d_{plot.png,result.json}`,
payload `payload/1d_sheet/testfile_1dsheet_payload.npz`.

## 1D follow-up: edge-induced ringing diagnosis + fix (2026-04-20)

The default (`SLM_EDGE_SIGMA=0`) 1D run shows periodic ripple along the
sheet axis that is not visible on the 2D capture.  The correct physical
picture is **edge-induced ringing from a band-limited system fitting a
hard-edged target** — not "discrete Fourier harmonics of a rectangle".
The ideal top-hat's Fourier transform is a continuous sinc; the CGM
residual against a Gaussian-input band piles up at the spatial
frequencies $k \sim n/W$ set by the target width, not by any discrete
spectrum of the rectangle itself.

In 1D the optimiser has no y-freedom, so the residual must sit in the
x-axis alone — it is visible on camera as a periodic pattern locked to
the target width.  In 2D the optimiser redistributes part of the
residual into y-dependent structure which is suppressed by the camera's
row-averaging, reducing (not halving) the observed along-x ripple.

### Experiment 1: edge_sigma sweep (mechanism test)

All other knobs at default (4096² grid, `SLM_FLAT_WIDTH=36`, 4000 CGM
iters).  PSD evaluated on a fixed 50-cam-px plateau-centred window at
the target-locked period `W_cam = 36·(3.957/2.4) ≈ 59` cam-px and
sub-multiples.  Amplitudes are % of plateau mean.

| σ (focal-px) | plateau mean | RMS % | Pk-Pk % | peak @ 59 px | peak @ 29.5 | peak @ 19.7 | 10-90 edge |
| ------------ | ------------ | ----- | ------- | ------------ | ----------- | ----------- | ---------- |
| 0            | 159.9        | 4.27  | 13.55   | **5.75**     | 1.73        | 0.29        | 17.5 px    |
| 1            | 198.9        | 2.59  | 8.88    | 3.32         | 1.19        | 0.66        | 27.5 px    |
| 2            | 180.6        | 1.97  | 7.20    | 2.49         | 0.98        | 0.55        | 25.0 px    |
| **3**        | 169.9        | **1.23** | **5.10** | **1.04**  | 1.12        | 0.48        | 25.0 px    |

The fundamental 59-px peak collapses **5.5×** (5.75 → 1.04 %) and core
RMS drops **3.5×** as the Gaussian edge taper grows.  Edge roll-off
widens ~8 cam-px (≈19 µm, <14 % of the 140 µm flat-top width).

### Experiment 2: flat_width sweep (period-locking test)

`SLM_EDGE_SIGMA=0`, otherwise default.  PSD evaluated at the *expected*
target-locked period for each run; if the period tracks the target
(not a fixed hardware artefact), these peaks stay dominant.

| `SLM_FLAT_WIDTH` (focal-px) | expected W_cam | RMS % | Pk-Pk % | peak @ W_target | peak @ W/2 |
| --------------------------- | -------------- | ----- | ------- | --------------- | ---------- |
| 27                          | 44.5 px        | 7.01  | 26.21   | **7.73**        | 5.64       |
| 36                          | 59.4 px        | 4.34  | 15.24   | **4.61**        | 2.37       |
| 45                          | 74.2 px        | 3.64  | 12.26   | **3.70**        | 3.33       |

The dominant ripple period follows the target width, not a fixed camera
period.  Narrower targets ring worse (~7 % at W=27 vs ~4 % at W=45) —
consistent with a fixed-bandwidth optical system having to represent
proportionally higher edge frequencies for a narrower top-hat.

### Verdict

The two experiments jointly nail the diagnosis: the "periodic dots" are
edge-induced ringing whose period is set by the target flat-top width.
A modest Gaussian edge taper (`SLM_EDGE_SIGMA=3`, i.e. ≈3 focal-px ≈
12 µm of rim) is the correct fix and cuts 1D-CGM plateau RMS from
4.27 % to **1.23 %** — comfortably better than the 2D reference
(4.01 %) and at the same ~66 s CGM cost.

Artefacts:
- `docs/sweep_sheet/1d_ringing/` — Group 1 (σ = 0…3) plots, results, ringing-metric JSONs
- `docs/sweep_sheet/1d_ringing_width/` — Group 2 (W = 27, 36, 45) plots and results
- `docs/Figures/1d_cgm_sheet_softedge3.png` — best single shot

![1D soft-edge σ=3 camera image](Figures/1d_cgm_sheet_softedge3.png)

## 2D follow-up: same ringing, same fix with larger σ (2026-04-20)

The 2D path has the same edge-induced ringing.  It's partially masked
by the y-redistribution channel, so the fundamental W-period peak is
slightly smaller at σ=0 than in 1D, but it's unmistakably present and
it responds to the same soft-edge fix — just with a **larger σ** needed
to fully kill the harmonics.

Same config as current 2D defaults (`SLM_ARRAY_BIT=12`, `SLM_FLAT_WIDTH=35`,
`SLM_GAUSS_SIGMA=5`, `SLM_FRESNEL_SD=1000`, 4000 CGM iters).  Fixed 50-
cam-px plateau-centred window; target-locked period W_cam ≈ 57.7 cam-px
(= 35 · 3.957 / 2.4).

| σ (focal-px) | plateau mean | RMS %  | Pk-Pk %  | peak @ W (57.7 px) | peak @ W/2 | edge 10-90 |
| ------------ | ------------ | ------ | -------- | ------------------ | ---------- | ---------- |
| 0            | 182.8        | 4.75   | 15.14    | 5.55               | 3.57       | 54.5 px    |
| 3            | 155.0        | 3.85   | 12.69    | 3.38               | **3.93**   | 26.0 px    |
| **5**        | 139.3        | **1.54** | **5.26** | **1.91**         | **0.79**   | 26.5 px    |

σ=3 suppresses the fundamental but pushes energy into W/2 — the 2D camera
image shows three visible bright lobes.  σ=5 kills both W and W/2
together: **RMS 4.75 % → 1.54 % (3.1×)**; fundamental peak 5.55 % → 1.91 %
(2.9×).  The plateau is visually a single clean top-hat with smooth
edges (see figure).

![2D soft-edge σ=5 camera image](Figures/2d_cgm_sheet_softedge5.png)

Default updated: `SLM_EDGE_SIGMA=5` in `scripts/sheet/testfile_sheet.py`
(and `=3` in `scripts/1d_sheet/testfile_1dsheet.py` — smaller σ suffices
when the 1D optimiser isn't competing with itself across axes).  The 2D
path needs a larger taper than the 1D path because the 2D CGM tends to
fall into a "spot-chain" local minimum (documented elsewhere in this
repo); a larger edge σ pre-filters the target hard enough that the
optimiser no longer has a spot-chain basin to slide into.

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
