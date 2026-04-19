# CGM light-sheet benchmark on real SLM — unified report

Issue: [#19](https://github.com/ChanceSiyuan/SLMengineer/issues/19) •
Hardware: Hamamatsu LCOS-SLM (1272×1024, 12.5 μm pitch) + Allied Vision camera •
Date: 2026-04-17

This document reports 14 parameter-sweep hardware captures driven by
`scripts/sheet/run_sweep_point.sh`, plus the original baseline. Each
captured PNG, per-run metrics JSON, and 3-panel analysis preview is
preserved next to this file in `docs/sweep_sheet/`; filenames use the
tag from the tables below.

Workflow per point:

1. `sed` the chosen parameter in `scripts/sheet/testfile_sheet.py`.
2. `uv run python scripts/sheet/testfile_sheet.py` → payload + params.
3. `./push_run.sh … --png` → real SLM + camera capture, pulled back as PNG.
4. `scripts/sheet/analysis_sheet.py --peak signal` → ROI around the
   shifted sheet, top-hat+gauss-edge fit along the major axis, gauss
   fit along the minor axis, shape-correlation/overlap fidelity,
   efficiency (ROI/total-signal), flat-region RMS.

---

## Baseline

All other runs perturb exactly one entry of the following configuration:

| group | parameter | baseline value |
|---|---|---|
| target | `sheet_flat_width` | **9 px** (≈142 μm focal) |
| target | `sheet_gaussian_sigma` | **2 px** (≈32 μm focal) |
| target | `sheet_edge_sigma` | **0 px** |
| target | `sheet_angle` | 0 rad |
| target | `target_shift_fpx` | 30 (≈475 μm off zero-order) |
| CGM | `cgm_max_iterations` | 4000 |
| CGM | `cgm_steepness` | 9 |
| CGM | `setting_eta` | **0.2** |
| CGM | `cgm_eta_steepness` | **9** |
| CGM | `init_phase_method` | `stationary_phase_2d` |
| optics | `fresnel_shift_distance_um` | 1000 |
| optics | `LUT` | 207 |
| capture | `etime_us` | 4000 |
| capture | `n_avg` | 10 |

Baseline hardware capture:

![baseline](00_baseline_after.png)

Baseline analysis preview (signal = after − before, major-axis fit, minor-axis fit):

![baseline preview](00_baseline_preview.png)

Baseline measured metrics:

| sim F | sim η | meas flat (px) | meas σ⊥ (cam px) | meas edge σ (px) | fid_corr | fid_overlap | eff (ROI) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.928 | 17.2 % | 1.0 | 15.1 | 5.7 | **0.324** | 0.046 | 6.1 % |

The qualitative failure mode — baseline and every sweep point — is
that the 1D flat-top target is delivered to the camera as a **chain of
3–4 discrete bright spots**, not a continuous uniform line. The
top-hat + gauss-edge fit along the major axis collapses (`flat ≈ 1 px`,
`edge σ` absorbs the rest) whenever one of the spots dominates; it
pretends to match a wide plateau (`flat` shoots up) whenever two spots
happen to sit at similar heights across a wider region. Read
`measured flat_width_px` as *"where the fit landed"*, not *"how wide
the actual sheet is"*. The correlation and overlap metrics are the
honest shape scores.

---

## Raw sweep table

All 15 points (baseline + 14 perturbations). Columns: simulated
fidelity and efficiency (Linux-side, pre-hardware), then measured
camera metrics (post-hardware). See `_index.csv` for the machine
copy.

| tag | param | value | sim F | sim η | meas flat (px) | meas σ⊥ (px) | meas edge σ | fid_corr | fid_overlap | eff (ROI) | flat RMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | — | — | 0.928 | 17.2 % | 1.00 | 15.1 | 5.7 | **0.324** | 0.046 | 6.1 % | 0.000 |
| `eta_0p0` | `setting_eta` | 0.0 | 1.000 | 9.4 % | 48.41 | 15.2 | 16.5 | 0.154 | 0.019 | 6.9 % | 0.083 |
| `eta_0p1` | `setting_eta` | 0.1 | 0.999 | 10.0 % | 56.68 | 15.6 | 11.3 | 0.152 | 0.018 | 7.7 % | 0.094 |
| `eta_0p4` | `setting_eta` | 0.4 | 0.740 | 31.2 % | 76.50 | 12.9 | 10.7 | 0.099 | 0.012 | **19.7 %** | 0.224 |
| `etasteep_7` | `cgm_eta_steepness` | 7 | 0.995 | 11.4 % | 1.74 | 20.6 | 4.8 | **0.327** | 0.039 | 5.0 % | 0.034 |
| `etasteep_10` | `cgm_eta_steepness` | 10 | 0.892 | 19.6 % | 28.84 | 13.0 | 2.6 | 0.083 | 0.014 | 9.0 % | 0.318 |
| `etasteep_11` | `cgm_eta_steepness` | 11 | 0.886 | 20.0 % | 16.55 | 13.5 | 4.0 | 0.209 | 0.024 | 8.7 % | 0.069 |
| `flatw_5` | `sheet_flat_width` | 5 | 0.977 | 19.6 % | 37.56 | 16.9 | 7.1 | 0.117 | 0.012 | 12.8 % | 0.125 |
| `flatw_15` | `sheet_flat_width` | 15 | 0.899 | 13.5 % | 1.94 | 16.9 | 6.9 | 0.280 | 0.057 | 7.8 % | 0.004 |
| `flatw_25` | `sheet_flat_width` | 25 | 0.894 | 9.7 % | 4.28 | 17.3 | 5.8 | 0.293 | **0.113** | 5.2 % | 0.012 |
| `gsig_1` | `sheet_gaussian_sigma` | 1 | 0.998 | 20.0 % | 26.78 | 18.0 | 7.8 | 0.104 | 0.014 | 6.6 % | 0.231 |
| `gsig_4` | `sheet_gaussian_sigma` | 4 | 0.878 | 10.5 % | 1.69 | 35.4 | 4.3 | 0.303 | 0.041 | 4.1 % | 0.008 |
| `gsig_8` | `sheet_gaussian_sigma` | 8 | 0.905 | 4.8 % | 1.88 | 71.0 | 4.4 | **0.468** | 0.068 | 3.2 % | 0.036 |
| `esig_2` | `sheet_edge_sigma` | 2 | 0.891 | 11.5 % | 19.02 | 16.2 | 6.9 | 0.220 | 0.053 | 7.9 % | 0.220 |
| `esig_5` | `sheet_edge_sigma` | 5 | 0.902 | 6.7 % | 1.00 | 19.2 | 6.8 | 0.367 | 0.094 | 6.0 % | 0.000 |
| `seedonly` | `cgm_max_iterations` | 0 | 0.097 | **99.98 %** | 34.61 | 18.3 | 19.0 | 0.106 | 0.012 | **35.5 %** | 0.026 |

**Winners per metric** (bold above):

- Highest `fid_corr`: `gsig_8` (0.468) → soft perpendicular Gaussian traded efficiency for shape match.
- Highest `fid_overlap`: `flatw_25` (0.113) → wider target absorbs the fringed output better.
- Highest in-ROI efficiency: `seedonly` (35.5 %) → nearly 6× baseline, but the camera sees a broad saturated spot, not a line (see dedicated section below).
- Highest CGM-iterated efficiency: `eta_0p4` (19.7 %) → at the cost of `fid_corr` (0.099).
- Baseline already holds `etasteep_7`'s correlation spot (0.327 vs 0.324).

---

## Per-parameter trends

### 1. `setting_eta` (CGM efficiency floor)

| value | fid_corr | eff (ROI) | fit behaviour |
|---:|---:|---:|---|
| 0.0 | 0.154 | 6.9 % | fit sees a wide, lumpy envelope (flat ≈ 48) |
| 0.1 | 0.152 | 7.7 % | same |
| 0.2 (baseline) | **0.324** | 6.1 % | dominant single peak + side lobe |
| 0.4 | 0.099 | **19.7 %** | sheet spreads over ≈200 px, many lumps |

**Reading:** when `eta_min` is *below* the simulated η, the term is dormant — CGM runs the pure overlap-maximization cost, which concentrates energy into one or two sharp spots (matches `etasteep_7` pattern). When `eta_min` rises above what CGM can honestly deliver (0.4 vs sim η ≈ 10 %), the penalty forces the algorithm to smear energy out across a wider footprint to lift η on the simulation. On the camera this reads as higher measured efficiency but *worse* shape — the sheet becomes a lumpy blob. Baseline 0.2 sits at the knee: just active enough to keep the sheet longitudinally extended without smearing it beyond recognition.

![eta_0p4 preview — extended lumpy sheet at eff=19.7 %, fid_corr=0.099](eta_0p4_preview.png)

### 2. `cgm_eta_steepness` (10^N multiplier on the efficiency penalty)

| value | fid_corr | eff (ROI) | fit behaviour |
|---:|---:|---:|---|
| 7 | **0.327** | 5.0 % | single peak, small side lobe |
| 9 (baseline) | 0.324 | 6.1 % | same |
| 10 | 0.083 | 9.0 % | fragmented, wider spread |
| 11 | 0.209 | 8.7 % | fragmented, but slightly more coherent |

**Reading:** at steepness 7 the eta-penalty is ~100× weaker than the overlap-cost, so `setting_eta=0.2` becomes effectively inactive — result mirrors `setting_eta ≈ 0` (strong spot concentration). Baseline 9 is the sweet spot. Pushing to 10–11 over-weights the efficiency term; the optimisation prefers spreading energy and sacrifices the overlap term, which collapses the correlation. The two highest-steepness runs look similar to the high-`setting_eta` runs, confirming that the two knobs multiply rather than act independently.

### 3. `sheet_flat_width` (length of the commanded top-hat)

| value (px) | fid_corr | fid_overlap | eff (ROI) |
|---:|---:|---:|---:|
| 5 | 0.117 | 0.012 | 12.8 % |
| 9 (baseline) | 0.324 | 0.046 | 6.1 % |
| 15 | 0.280 | 0.057 | 7.8 % |
| 25 | 0.293 | **0.113** | 5.2 % |

**Reading:** very narrow targets (5 px) force CGM to emit almost a
point, then the fragmentation fills a much wider footprint than
commanded — big RMS, low `fid_corr`. At 9 px baseline the output
*is* close to a point; widening to 15 or 25 px drops sim η and gives
more room for the output's fringes to overlap the target envelope,
lifting `fid_overlap`. **Key finding:** `fid_overlap` is roughly
monotone with requested width, whereas `fid_corr` peaks at the
baseline — i.e., widening the target makes the measurement "contain"
the actual bright spots better, but doesn't improve the underlying
shape. If the physics target is "hit a 700 μm region", `flatw_25`
wins; if it's "be uniform across 700 μm", none of these sweeps do.

### 4. `sheet_gaussian_sigma` (perpendicular Gaussian 1-σ)

| value (px) | fid_corr | meas σ⊥ (cam px) | eff (ROI) |
|---:|---:|---:|---:|
| 1 | 0.104 | 18.0 | 6.6 % |
| 2 (baseline) | 0.324 | 15.1 | 6.1 % |
| 4 | 0.303 | 35.4 | 4.1 % |
| 8 | **0.468** | 71.0 | 3.2 % |

**Reading:** the measured perpendicular width scales roughly linearly
with the target σ (2 → 15 px, 4 → 35, 8 → 71). Perpendicular
delivery is well behaved — the optics are not fighting us on the
minor axis. `fid_corr` improves monotonically with target σ ≥ 2
because the target envelope gets more tolerant of the longitudinal
fringes: whatever bright spots CGM emits are *contained* inside a
bigger perpendicular Gaussian. At σ=8 we're effectively benchmarking
against a broader Gaussian, which is why `fid_corr` climbs to 0.47 —
but the real flat-top along the major axis is no closer to uniform.

![gsig_8 preview — σ⊥ = 71 px, fid_corr = 0.468, but only 3.2 % η](gsig_8_preview.png)

### 5. `sheet_edge_sigma` (soft taper at the ends of the flat region)

| value (px) | fid_corr | fid_overlap | eff (ROI) |
|---:|---:|---:|---:|
| 0 (baseline) | 0.324 | 0.046 | 6.1 % |
| 2 | 0.220 | 0.053 | 7.9 % |
| 5 | **0.367** | **0.094** | 6.0 % |

**Reading:** a soft taper of 5 px gives the best correlation *and*
overlap of any target-shape tweak — ≈13 % better than baseline on
`fid_corr` and 2× better on `fid_overlap`. The taper discourages the
stationary-phase seed from placing sharp phase discontinuities at
the sheet ends, so CGM starts from a gentler gradient and the output
fringing is slightly less aggressive. `edge_sigma=2` is too small to
change the seed meaningfully and ends up worse than baseline,
probably because the fitter now has three competing scales
(flat+edge+gauss) and the fit degenerates.

![esig_5 preview](esig_5_preview.png)

---

## Follow-up: seed-only (no CGM)

The Section B recommendation above was to test `cgm_max_iterations = 0`
— i.e. use the `stationary_phase_sheet` seed phase directly, with no
CGM iterations. The prediction from `testfile_sheet.py:137–144` was
"flatness ~0.64, η ≈ 99.9 %, contrast 10⁶" in simulation, which
should have translated to a clean line at the camera.

Hardware result:

![seedonly after](seedonly_after.png)

![seedonly preview](seedonly_preview.png)

| metric | value | vs. baseline | vs. `eta_0p4` (previous efficiency champion) |
|---|---:|---|---|
| sim F | 0.097 | 10× worse | 7× worse |
| sim η | **99.98 %** | 6× better | 3× better |
| meas efficiency (ROI) | **35.5 %** | **6× better** | **2× better** |
| meas fid_corr | 0.106 | worse | ≈ same |
| meas perpendicular σ | 18 cam px | ≈ same as CGM | — |

What the camera actually shows is **a single broad saturated spot (≈
136 × 156 cam px) plus a faint diagonal chain of fainter spots**,
NOT the predicted horizontal line. The major-axis projection is a
symmetric bell with a saturated plateau at the top — the fitter
reports `flat_width ≈ 35 px` and `edge_sigma ≈ 19 px` but that is an
artefact of camera saturation clipping a Gaussian, not a real
flat-top.

**Reading:**

1. The simulation metrics are now decorrelated. `sim η = 99.98 %`
   says "all energy lands inside the target region mask" — the
   region mask is a rectangular bounding box around the 9×2-px
   sheet plus 5-px margin, so a broad circular blob centred on the
   sheet still scores ~100 % region-energy while completely failing
   the shape. `sim F = 0.097` (very low) correctly flags the shape
   failure, but because CGM's cost function is overlap-based it
   would only get the chance to *fix* this shape mismatch by
   running iterations, and we've already shown iterations drive it
   back to a spot chain.

2. The camera's 35.5 % measured efficiency is 6× higher than CGM
   because it's dominated by a big saturated peak, not by good
   coverage of a 1D line. The useful metric here is not "efficiency
   in ROI" but "uniformity along the sheet axis" — and by that
   measure the seed-only output is worse than baseline, not better.

3. The contradiction with the code comment is significant. One of:
   (a) `stationary_phase_sheet` isn't delivering the phase it claims
   to — perhaps a sign / axis / scale bug in its cylindrical lens
   term, or the effective `f_cam` the lens is built for is wrong
   for the camera plane; (b) the 1024-grid compute size is too
   coarse for the ray-redistribution step to resolve a 9-px
   flat-top; (c) the Fresnel lens (`fresnel_sd=1000 um`) applied
   post-hoc on top of the seed is shifting the focal plane away
   from where the seed expects it — CGM runs also apply this lens,
   so if it's wrong it should affect both runs equally, but the
   seed is more sensitive because it has no line-search to absorb
   the defocus.

4. **Seed-only is not a drop-in replacement for CGM on this rig,
   contrary to what the code comments predict.** Before promoting
   it, we need to understand why the seed phase is producing a blob
   instead of a line.

Recommended next actions (outside this benchmark's scope):

- Open a new issue: re-verify `stationary_phase_sheet` on simulation
  (plot the seed's `|E_out|² = |fft_propagate(S · exp(iφ_seed))|²` at
  1024 and 4096 grid sizes; should be a 9-px bright line if the seed
  is correct).
- Run seed-only with `fresnel_sd=0` and sweep `fresnel_sd` ∈
  [0, 500, 1000, 2000, 5000] μm to check whether the post-hoc lens
  is responsible for the defocus.
- If the seed is bug-free, the only remaining path is Section C
  (add a non-uniformity regulariser to the CGM cost function).

---

## What's actually going on

Every run shows the same camera signature: **a short chain of 3–4
bright diffraction spots along the sheet axis, plus a broad
perpendicular Gaussian envelope**. The algorithm is *not* failing to
satisfy the simulation cost — `sim F ≈ 0.9` on most runs — it is
faithfully producing the solution the Linux-side cost favours. The
mismatch is between that cost and the "uniform line of light the
camera sees" notion of success.

Mechanism (restating the `testfile_sheet.py` comment at lines 73–77):

1. CGM's fidelity term is `(1 − Re⟨τ, E_out/|E_out|⟩)²`. A single
   bright spot at the correct location gives a near-perfect overlap
   with a 9-px top-hat because `⟨τ, δ_target⟩/|τ| ≈ 1` when the
   delta is normalised — a Dirac that lands inside the target
   region is as good as a flat-top.
2. Without a strong `eta_min` lower bound, the algorithm happily
   concentrates all energy into such a spot. This is what
   `etasteep_7` and `setting_eta=0.0/0.1` show: bright central peak,
   tiny simulated η, camera shows one dominant speck.
3. Turning `eta_min` up *forces* CGM to spread energy; but the path
   of least cost is not "make a flat-top", it's "make a wider blob
   that still overlaps the target envelope". Hence the `eta_0p4`
   behaviour — lots of energy in the ROI, still fringed.
4. The analysis-script top-hat fitter cannot distinguish between a
   1-px spike and a 50-px plateau when the plateau is itself lumpy —
   `flat_width_px` in the table is unreliable for qualitative
   judgement. `fid_corr` and visual inspection of the preview panels
   are the honest signal.

---

## How to optimise

Ordered from low-risk tweaks (stay within the current pipeline) to
bigger changes (new algorithm / new initial phase):

### A. Best parameter set from this sweep

Combining the two small wins that don't hurt each other:

| parameter | value | rationale |
|---|---:|---|
| `setting_eta` | 0.2 | keep baseline — it's the knee |
| `cgm_eta_steepness` | 9 | keep baseline — 7 and 9 are within noise; 10–11 hurt |
| `sheet_flat_width` | 9 | keep baseline — narrower widens the blob, wider doesn't fix uniformity |
| `sheet_gaussian_sigma` | 2 | keep baseline — wider gives cosmetic `fid_corr` gains only |
| `sheet_edge_sigma` | **5** | soft taper improves both shape metrics and costs almost no efficiency |

Expected: `fid_corr ≈ 0.37`, `fid_overlap ≈ 0.09` (vs. baseline 0.32 / 0.05). Real uniformity still won't be there — this is a ~15 % refinement, not a fix.

### B. Drop CGM altogether (tested — see seed-only section above)

We ran the experiment: `cgm_max_iterations = 0`, same hardware harness.
Result: **the seed phase does not deliver a line on the camera**, it
delivers a broad saturated Gaussian-like blob. 35.5 % efficiency but
`fid_corr = 0.106`. The prediction in the code comment is not
reproduced; see the dedicated "Follow-up: seed-only" section for
the full analysis and the three hypotheses for why.

Net: CGM is not the bottleneck anymore — the seed is. Do NOT
promote seed-only. Diagnose `stationary_phase_sheet` first.

### C. Change the CGM cost (needs code)

If CGM must stay in the pipeline (e.g., because we want to polish
other target shapes that need it), consider:

1. **Non-uniformity regulariser.** Add a term proportional to the
   RMS of `|E_out|² · region` relative to its mean inside the
   target. Would penalise exactly the spot-vs-flat failure mode.
   `slm.metrics.non_uniformity_error` is already defined.
2. **Higher `sheet_edge_sigma` AND `sheet_gaussian_sigma` together.**
   Make the target more "Gaussian-like" so CGM's preferred answer
   (a spot) is no longer maximally overlapping. The current sweep
   tested these axes separately; combining `esig=5, gsig=4` or
   `esig=5, gsig=8` is a natural follow-up.
3. **Initial-phase tweaks.** The current `stationary_phase_sheet`
   seed carries sharp discontinuities at the flat-top ends; a softer
   seed (e.g. linear ramp padded with a Gaussian) may prevent CGM
   from relaxing into the fringe basin.

### D. Hardware / capture improvements (orthogonal)

- Zero-order is clipping at 255 in both before and after; this
  limits how close to the zero-order we can place the sheet. If we
  lower exposure (`etime_us`) we'd see the tails better but kill
  SNR on the sheet itself. A better fix is an ND filter in front of
  the zero-order or a dedicated beam block — outside this issue's
  scope.
- The ROI fitter relies on `--peak signal` to find the shifted
  sheet; for future sweeps at different shift distances we should
  store `target_shift_fpx` in params.json and let the analyser pick
  `--peak` automatically.

---

## Recommended next hardware run

```python
sheet_flat_width = 9
sheet_gaussian_sigma = 2
sheet_edge_sigma = 5        # ← only change from baseline
setting_eta = 0.2
cgm_eta_steepness = 9
cgm_max_iterations = 4000
```

This is still the safest production setting after the seed-only follow-up:
- Seed-only produced a bright but shape-broken blob (`fid_corr = 0.106`,
  perpendicular extent ≈ 18 cam px matching the other CGM runs,
  longitudinal extent saturated blob), not a line.
- Of all 15 benchmarked settings, `esig_5` has the best combined
  (`fid_corr`, `fid_overlap`, eff) triple without introducing a new
  failure mode.

The real optimisation target is now upstream: **debug
`stationary_phase_sheet`** (and/or the `fresnel_sd` coupling) so the
seed delivers its promised line. See the seed-only follow-up section
for the three hypotheses to test.

---

## Files

```
docs/sweep_sheet/
├── SUMMARY.md           (this file)
├── _index.csv           (machine-readable per-run metrics)
├── 00_baseline_*        (baseline PNG, params, metrics, 3-panel preview)
├── eta_0p0_*            (setting_eta = 0.0 sweep point, same 4 files)
├── eta_0p1_*            (setting_eta = 0.1)
├── eta_0p4_*            (setting_eta = 0.4)
├── etasteep_7_*         (cgm_eta_steepness = 7)
├── etasteep_10_*        (cgm_eta_steepness = 10)
├── etasteep_11_*        (cgm_eta_steepness = 11)
├── flatw_5_*            (sheet_flat_width = 5)
├── flatw_15_*           (sheet_flat_width = 15)
├── flatw_25_*           (sheet_flat_width = 25)
├── gsig_1_*             (sheet_gaussian_sigma = 1)
├── gsig_4_*             (sheet_gaussian_sigma = 4)
├── gsig_8_*             (sheet_gaussian_sigma = 8)
├── esig_2_*             (sheet_edge_sigma = 2)
├── esig_5_*             (sheet_edge_sigma = 5)
└── seedonly_*           (cgm_max_iterations = 0, stationary-phase seed direct)
```
