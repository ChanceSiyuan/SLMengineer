# stationary_phase_sheet hardware fix — iteration log

Issue [#20](https://github.com/ChanceSiyuan/SLMengineer/issues/20) •
Hardware: Hamamatsu LCOS-SLM (1272×1024, 12.5 μm) + Allied Vision camera •
Date: 2026-04-17

## Phase 1 — seed-only delivers a real light sheet

Six closed-loop hardware iterations to make the seed-only payload deliver
a real light sheet on camera. Phase-1 winning configuration:
**`flat_width=25 px`, `gaussian_sigma=None` (no cylindrical lens),
`fresnel_sd=1000 μm`** → 166–188 × 50 cam-px line, aspect ≈ 3.5.

### Starting state

`scripts/sheet/testfile_sheet.py` ran with `cgm_max_iterations=0,
sheet_flat_width=9, sheet_gaussian_sigma=2, fresnel_sd=1000` produced a
single saturated round blob on the camera. See `seedonly_after.png`.

### Phase 1 iterations

| # | config change | ROI dx×dy | aspect | fit flat (px) | fid_corr | eff | qualitative |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | fsd: 1000→**5000** | 116×57 | 2.0 | 68.5 | 0.061 | 45.0 % | wider blob, 2 dim bumps |
| 2 | + gaussian_sigma:2→**None** (back fsd=1000) | 78×64 | 1.2 | 31.9 | 0.092 | 31.7 % | two distinct lobes side-by-side |
| 3 | + fsd 1000→**5000** | 137×76 | 1.8 | 72.6 | 0.059 | 50.8 % | rounder, more diffuse — wrong direction |
| 4 | + fsd 5000→**0** | 70×70 | 1.0 | 16.0 | **0.122** | 30.9 % | clean horizontal lobe but compact |
| 5 | + sheet_flat_width:9→**25** | 148×51 | 2.9 | 124.6 | 0.063 | 40.1 % | **clean horizontal streak** |
| **6** | **+ fsd 0→1000 (production)** | **188×50** | **3.76** | 129.1 | 0.083 | 40.4 % | **CLEAN ELONGATED LIGHT SHEET** |
| final-P1 | reload of iter 6 | 166×51 | 3.25 | 129.8 | 0.067 | 40.3 % | stable run-to-run |

(`fid_corr` is computed against a flat_width=9 reference target by
`analysis_sheet.py`, so it under-reports when we widened the commanded
flat to 25.)

## Phase 2 — fix asymmetry + warm-start CGM

After Phase 1 the line had a ~25 % asymmetric tail. Phase 2 reduces the
asymmetry with a beam-center offset and then runs a few CGM iterations
on top of the seed to polish it. Phase-2 winning configuration:
**`BEAM_CENTER_DX_UM=-2000`, `sheet_edge_sigma=5`, `cgm_max_iterations=5`,
`setting_eta=0.6`, `cgm_eta_steepness=11`** → 181 × 42 cam-px line,
aspect 4.31, with diffraction sidebands but no spot chain.

### Asymmetry calibration (left/right brightness imbalance, +ve = brighter on left)

| run | bcm (μm) | asym | aspect | fid_corr | qualitative |
|---|---:|---:|---:|---:|---|
| iter 6 (P1) | 0 | +0.256 | 3.76 | 0.083 | line, brighter on left |
| final-P1 | 0 | +0.191 | 3.25 | 0.067 | same |
| **A1** | **−2000** | **+0.143** | **3.29** | **0.146** | **cleanest line, mild left-skew** |
| A3 | −3500 | +0.151 | 2.64 | 0.167 | line shifts left, develops right tail |
| A2 | −5000 | −0.190 | 2.44 | 0.193 | overshoots — flips to right-heavy + breaks shape |
| CGM-4000 baseline | n/a | −0.066 | 1.23 | 0.324 | (chain of spots; mostly symmetric) |

`bcm = −2000 μm` was the cleanest visual: keeps aspect ≈ 3.3 and almost
doubles `fid_corr` over the bcm=0 case. Larger magnitudes either over-
correct or break the line shape.

### CGM warm-start sweep (warm-started from the corrected seed at bcm=−2000)

| run | cgm_max_iter | setting_eta | edge_sigma | aspect | fid_corr | eff | qualitative |
|---|---:|---:|---:|---:|---:|---:|---|
| **B4** | **5** | **0.6** | **5** | **4.28** | 0.127 | 40.1 % | **clean bright line, sidebands, no spots** |
| B2 | 10 | 0.6 | 5 | 3.96 | 0.151 | 41.2 % | clean line, slight internal modulation |
| B3 | 30 | 0.6 | 5 | 9.50 (artifact) | 0.193 | 23.3 % | line begins to break into wavy fringes |
| B1 | 100 | 0.4 | 0 | 1.59 | **0.238** | 10.1 % | **collapsed to spot chain — failure mode** |
| final-P2 | 5 | 0.6 | 5 | 4.31 | 0.155 | 40.1 % | reproduced B4 cleanly |

CGM's natural minimum is the chain-of-spots pattern (the prior
`00_baseline` 4000-iter result and B1 above both end there). Three
ingredients keep the warm-started run at a clean line instead:

1. **Few iterations** (5, not 4000): just polish the seed, don't run to convergence.
2. **High `setting_eta=0.6` with steep `cgm_eta_steepness=11`**: penalises the cost-function basin where energy concentrates into one or two spots.
3. **`sheet_edge_sigma=5`**: soft target edges, so the optimiser is rewarded for *any* energy near the edges instead of the flat-top inside.

## Iteration images

```
docs/sweep_sheet/
├── fix_iter_01_fsd5000_after.png     # P1 iter 1
├── fix_iter_02_nocyl_after.png       # P1 iter 2
├── fix_iter_03_nocyl_fsd5000_after.png  # P1 iter 3
├── fix_iter_04_pure_seed_after.png   # P1 iter 4
├── fix_iter_05_w25_after.png         # P1 iter 5
├── fix_iter_06_w25_fsd1000_after.png # P1 iter 6 (winner Phase 1)
├── fix_iter_A1_bcm2000_after.png     # P2 asym A1 (winner: bcm=-2000)
├── fix_iter_A2_bcm5000_after.png     # P2 asym A2 (overshoot)
├── fix_iter_A3_bcm3500_after.png     # P2 asym A3 (intermediate)
├── fix_iter_B1_cgm100_after.png      # P2 CGM B1 (failure: spot chain)
├── fix_iter_B2_cgm10_after.png       # P2 CGM B2 (clean line, 10 iter)
├── fix_iter_B3_cgm30_after.png       # P2 CGM B3 (begins to fragment)
├── fix_iter_B4_cgm5_after.png        # P2 CGM B4 (winner: 5 iter, cleanest)
├── fix_FINAL_after.png               # Phase 1 final reload
├── fix_FINAL_phase2_after.png        # Phase 2 final (winner overall)
└── fix_progression.png               # 6-panel side-by-side summary
```

## Final committed configuration (in `scripts/sheet/testfile_sheet.py`)

```python
BEAM_CENTER_DX_UM = -2000   # asymmetry compensation
BEAM_CENTER_DY_UM = 0
fresnel_sd       = 1000     # μm (production)
sheet_flat_width = 25       # px (~ 396 μm focal)
sheet_gaussian_sigma = 2    # (kept for target shape; passed as None to seed)
sheet_edge_sigma = 5        # soft taper
sheet_angle      = 0
target_shift_fpx = 30
cgm_max_iterations = 5
setting_eta        = 0.6
cgm_eta_steepness  = 11
```

Algorithm-side: no changes to `src/slm/initial_phase.py` or
`src/slm/generation.py`. `stationary_phase_light_sheet` is mathematically
correct and matches the published Dickey–Romero–Holswade closed form.
The bug was the on-rig coupling between its cylindrical-lens term and
the post-hoc Fresnel; omitting the cylindrical term avoids it.

## Open questions

- The fid_corr metric still understates the line shape because
  `analysis_sheet.py` rebuilds its reference from a hard-coded
  `flat_width=9`; reading `params.json` would close that gap.
- The remaining ~14–24 % left-bias is likely SLM flatness / LUT
  residuals on the left half of the active aperture. Not blocking;
  closed-loop feedback would address it cleanly.
- Option A from the original `SEED_DIAGNOSIS.md` (cylindrical-lens
  compensation for fresnel_sd) was not needed in the end — Option B
  (drop cylindrical) plus a wider commanded flat-top plus 5 CGM polish
  iterations gives a clean light sheet without algorithm changes.
