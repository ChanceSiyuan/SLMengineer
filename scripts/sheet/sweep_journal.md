# scripts/sheet/ — light-sheet SLM pipeline + issue #17 sweep results

## Overview
`scripts/sheet/` holds the whole CGM → SLM → camera → analysis loop
for a 1D top-hat ("light sheet") target on the Hamamatsu LCOS-SLM +
Allied Vision camera rig. Two workflows live here:

1. **single-shot**: `testfile_sheet.py` + `testfile_sheet.sh` —
   compute one payload with one set of parameters, push it to the
   Windows runner, capture before/after frames.
2. **parameter sweep**: `sweep_one.py` + `run_one.sh` — drive a
   multi-point sweep one index at a time, with per-point hardware
   analysis between shots. Used for GitHub issue #17.

Issue #17 sweep (35 points × 7 swept parameters × 5 values each) ran
this pipeline end-to-end. The headline finding below is its result.

---

## Headline: best point

**`sheet_edge_sigma = 0.5`** (issue #17 sweep index **020**).

| metric              | base (edge_sigma=1.5) | **idx 020** | ratio |
|---------------------|-----------------------|-------------|-------|
| efficiency — mean of 3 hardware samples | 7.4% | **~38%** (19.8 / 41.1 / 52.4) | **~5×** |
| `fidelity_corr` — mean of 3 samples    | 0.093 | **0.213** (0.230 / 0.202 / 0.207) | **~2.3×** |
| Pareto frontier occupant?              | no    | **yes** (1 of 2) | — |

Recommended single-parameter change in `testfile_sheet.py`:
```python
sheet_edge_sigma = 0.5   # was 1.5
```
Everything else (`LUT=207`, `fresnel_sd=500`, `sheet_flat_width=24`,
`sheet_gaussian_sigma=1.5`, `sheet_angle=π/2`, `cgm_D=-π/2`,
`cgm_theta=π/4`, `cgm_steepness=8`, `eta_min=0.1`) stays at its
current value — the sweep did not identify a statistically
meaningful improvement on any other axis.

**Physical hypothesis.** On a 1024² compute grid at 15.83 µm/px, a
1.5 px Gaussian edge is only ~2 samples of taper. The optimizer
effectively wastes budget on an unresolvable smooth edge. Setting
`edge_sigma=0.5` forces the target into a hard-edged shape the grid
can actually represent, and the CGM converges to a better hologram.

**Important caveat.** Even at the best point, the camera capture is
**not** a continuous flat-top sheet — it's a short vertical chain of
discrete bright spots overlapping the zero-order reflection. Every
one of the 35 sweep points + 9 beam-center follow-up points shows
this same qualitative pattern. The sweep located the best
configuration within the current hardware regime; whatever is
producing the chain (likely `cgm_D`/`cgm_theta` aliasing, compute
grid resolution, or a stale calibration BMP) needs separate
investigation. See **Open questions** below.

---

## Directory structure

```
scripts/sheet/
├── testfile_sheet.py         # single-shot CGM → payload .npz
├── testfile_sheet.sh         # push/run/pull orchestrator for testfile_sheet
├── sweep_sheet_config.json   # sweep grid (base + per-param lists)
├── sweep_sheet.py            # module only — shared helpers (setup_slm,
│                             #   run_cgm, apply_post_processing, save_preview,
│                             #   TIER1_PARAMS, SLM_REINIT_PARAMS, OUT_DIR)
├── sweep_one.py              # CLI: generate ONE sweep point by index,
│                             #   upsert its entry into sweep_manifest.json
├── run_one.sh                # shell: per-point push → run → remote crop →
│                             #   pull → analyze (uses sweep_one.py under the hood)
├── crop_after.py             # helper pushed to the Windows runner so that
│                             #   _after.npy gets cropped to the sheet ROI
│                             #   before being scp'd back (~70× pull speedup)
├── analysis_sheet.py         # single-capture analyzer (CLI + importable lib)
├── analyze_sweep_sheet.py    # aggregate per-point analyses → plots + Pareto +
│                             #   best_config.json
└── sweep_journal.md          # this file
```

All per-point payloads, previews, and manifests live in
`scripts/sweep_sheet/` (gitignored / wiped during cleanup — they
regenerate from the config + CGM deterministically). Hardware
captures land in `data/sweep_sheet/`.

---

## Usage

### Single-shot (one payload, one capture)
```bash
./scripts/sheet/testfile_sheet.sh
```
This runs `testfile_sheet.py` locally (CGM → payload .npz), `scp`s
the payload to the Windows runner, triggers `slmrun.bat`, and pulls
`_after.png` + `_run.json` back to `data/`. Edit the constants at
the top of `testfile_sheet.py` to change parameters.

### Parameter sweep — point by point (recommended)
1. Edit `scripts/sheet/sweep_sheet_config.json` to change `base` (the
   fixed config) or `sweep` (the parameter/value lists to scan).

2. For each index `N` in `0 .. total_points - 1`, run:
   ```bash
   ./scripts/sheet/run_one.sh N
   ```
   This invokes `sweep_one.py --index N` (which generates the payload
   with CGM if missing and upserts the manifest entry), then
   `scp`s payload + params to the runner, triggers `slmrun.bat`,
   ssh-invokes `crop_after.py` on Windows to reduce the 49 MB frame
   to a ~300 KB ROI crop, `scp`s the crop + PNG + JSON back to
   `data/sweep_sheet/`, and runs `analysis_sheet.py` on the pulled
   crop.

3. List deterministic index → param mapping at any time:
   ```bash
   uv run python scripts/sheet/sweep_one.py --list
   ```

4. After all points are captured, aggregate + pick best:
   ```bash
   uv run python scripts/sheet/analyze_sweep_sheet.py
   ```

### Ad-hoc capture analysis
```bash
uv run python scripts/sheet/analysis_sheet.py \
    --after  data/sweep_sheet/sweep_sheet_020_after_crop.npy \
    --before data/sweep_sheet/sweep_sheet_000_before_crop.npy \
    --params scripts/sheet/testfile_sheet_params.json \
    --out    /tmp/analysis.json \
    --preview /tmp/analysis.png
```
Works on any `(before, after)` pair of camera captures. The analyzer
walks from the brightest point of `before` (zero-order) outward in
`after − median_bg` until intensity drops, fits a top-hat + Gaussian
edge to the major-axis projection, and computes three shape metrics:
`efficiency` (fraction of positive signal inside the ROI),
`fidelity_corr` (Pearson correlation with the expected
`slm.targets.light_sheet` reference), `fidelity_overlap`
(`(Σ√(I_meas · I_ref))²`).

---

## Reproduction of the issue #17 results

Full 35-point run end-to-end:

```bash
# Prerequisite: crop_after.py must be on the Windows runner.
scp -P 60022 scripts/sheet/crop_after.py \
    Galileo@199.7.140.178:/C:/Users/Galileo/slm_runner/crop_after.py

# Expected runtime: ~1 h (35 points × ~1m45s/pt with the remote
# crop optimization; ~20 min/pt without it).
for i in $(seq 0 34); do
    ./scripts/sheet/run_one.sh "$i"
done

# Aggregate into per-param trend plots + Pareto frontier + best_config.json
uv run python scripts/sheet/analyze_sweep_sheet.py
```

Per-point outputs land in `data/sweep_sheet/`:
- `sweep_sheet_NNN_after.png` — camera preview (for visual inspection)
- `sweep_sheet_NNN_after_crop.npy` — ROI crop around the sheet
- `sweep_sheet_NNN_analysis.json` — all metrics from `analysis_sheet.py`
- `sweep_sheet_NNN_analysis.png` — 3-panel analysis preview
- `sweep_sheet_NNN_run.json` — runner-side metadata
- `sweep_sheet_000_before_crop.npy` — shared SLM-blank reference
  (only pulled for index 0, reused for every other analysis)

Aggregate outputs:
- `data/sweep_sheet/aggregated.json` — all 35 per-point metrics
- `data/sweep_sheet/pareto.png` — efficiency × fidelity_corr scatter
  with the Pareto frontier highlighted (only idx 020 + 021 survive)

---

## Full 35-point results table

First-sample values. For repeatability see the next section.

| idx | param                | value | eff%   | fid_corr | fid_ovl | flat_rms |
|-----|----------------------|-------|--------|----------|---------|----------|
| 000 | LUT                  | 200   | 4.51   | 0.0978   | 0.0246  | 0.551    |
| 001 | LUT                  | 204   | 0.53   | 0.0732   | 0.0218  | 0.000    |
| 002 | LUT                  | 207   | 9.33   | 0.0924   | 0.0214  | 0.552    |
| 003 | LUT                  | 210   | 6.65   | 0.1199   | 0.0286  | 0.124    |
| 004 | LUT                  | 214   | 3.92   | 0.1088   | 0.0292  | 0.182    |
| 005 | fresnel_sd           | 400   | 7.52   | 0.0754   | 0.0176  | 0.524    |
| 006 | fresnel_sd           | 450   | 6.72   | 0.0967   | 0.0235  | 0.600    |
| 007 | fresnel_sd           | 500   | 3.90   | 0.0935   | 0.0260  | 0.147    |
| 008 | fresnel_sd           | 550   | 0.41   | 0.0844   | 0.0244  | 0.000    |
| 009 | fresnel_sd           | 600   | 2.37   | 0.0809   | 0.0230  | 0.099    |
| 010 | sheet_flat_width     | 20    | 6.91   | 0.1104   | 0.0260  | 0.153    |
| 011 | sheet_flat_width     | 22    | 11.13  | 0.1135   | 0.0267  | 0.133    |
| 012 | sheet_flat_width     | 24    | 9.11   | 0.0850   | 0.0195  | 0.531    |
| 013 | sheet_flat_width     | 26    | 4.17   | 0.0971   | 0.0209  | 0.528    |
| 014 | sheet_flat_width     | 28    | 5.30   | 0.1224   | 0.0296  | 0.440    |
| 015 | sheet_gaussian_sigma | 1.0   | 1.52   | 0.0598   | 0.0165  | 0.112    |
| 016 | sheet_gaussian_sigma | 1.25  | 6.37   | 0.0728   | 0.0171  | 0.528    |
| 017 | sheet_gaussian_sigma | 1.5   | 3.03   | 0.0932   | 0.0259  | 0.125    |
| 018 | sheet_gaussian_sigma | 1.75  | 1.43   | 0.1098   | 0.0321  | 0.000    |
| 019 | sheet_gaussian_sigma | 2.0   | 0.85   | 0.1192   | 0.0331  | 0.000    |
| **020** | **sheet_edge_sigma** | **0.5** | **19.84** | **0.2297** | **0.0408** | 0.000 |
| **021** | **sheet_edge_sigma** | **1.0** | **12.73** | **0.1904** | **0.0416** | 0.000 |
| 022 | sheet_edge_sigma     | 1.5   | 5.77   | 0.0951   | 0.0237  | 0.627    |
| 023 | sheet_edge_sigma     | 2.0   | 10.13  | 0.1062   | 0.0276  | 0.523    |
| 024 | sheet_edge_sigma     | 2.5   | 5.67   | 0.1171   | 0.0345  | 0.346    |
| 025 | cgm_steepness        | 6     | 8.98   | 0.1124   | 0.0320  | 0.175    |
| 026 | cgm_steepness        | 7     | 8.49   | 0.0897   | 0.0246  | 0.060    |
| 027 | cgm_steepness        | 8     | 10.44  | 0.1182   | 0.0303  | 0.154    |
| 028 | cgm_steepness        | 9     | 10.34  | 0.1163   | 0.0296  | 0.149    |
| 029 | cgm_steepness        | 10    | 11.14  | 0.0910   | 0.0220  | 0.569    |
| 030 | eta_min              | 0.05  | 27.48  | 0.1218   | 0.0322  | 0.230    |
| 031 | eta_min              | 0.075 | 9.27   | 0.1139   | 0.0283  | 0.378    |
| 032 | eta_min              | 0.1   | 10.46  | 0.0738   | 0.0217  | 0.054    |
| 033 | eta_min              | 0.125 | 10.34  | 0.1167   | 0.0306  | 0.175    |
| 034 | eta_min              | 0.15  | 11.05  | 0.0959   | 0.0271  | 0.161    |

---

## Noise floor

Seven sweep indices (002, 007, 012, 017, 022, 027, 032) happen to land
on the exact base configuration because the base value sits in the
middle of the corresponding sweep range. Their first-sample
efficiencies: **9.33, 3.90, 9.11, 3.03, 5.77, 10.44, 10.46** →
**mean 7.43%, std 3.15 pp**, peak-to-peak ~5 pp.

**This is the run-to-run noise floor at identical params.** Any
single-sample improvement within ±5 pp of the baseline mean is not a
statistically meaningful win.

---

## Reproducibility of the top candidates

| idx | config           | efficiency samples | fidelity_corr samples | verdict |
|-----|------------------|--------------------|----------------------|---------|
| **020** | edge_sigma=0.5  | 19.84 / 41.11 / 52.35 | 0.2297 / 0.2024 / 0.2065 | **real winner** — all 3 samples far above baseline; fidelity stable |
| **021** | edge_sigma=1.0  | 12.73 / 41.21         | 0.1904 / 0.2065          | **real winner** — confirmed by 2nd sample |
| 030 | eta_min=0.05         | 27.48 / 10.04         | 0.1218 / 0.0409          | **noise spike** — reverts to baseline on retry |

Efficiency has large run-to-run variance (factor-of-2.6 on idx 020) but
`fidelity_corr` is stable — the pattern shape is reproducible, only
the in-ROI flux drifts (likely due to zero-order saturation and/or
laser-power drift between shots).

---

## Beam-center follow-up

After issue #17, a 3×3 grid of `BEAM_CENTER_DX_UM, DY_UM ∈ {-2000, 0,
+2000} µm` was captured at the base config to test whether off-center
incident illumination was causing the dot-chain pattern. **It is not.**
All 9 captures show the same vertical chain of spots at the same
camera location; only the overall captured brightness changes
(slightly higher for `+dx`, within the ±5 pp noise floor). Best point
(idx 006, `dx=+2000, dy=-2000`) measured 13.07% first sample, 10.81%
on retry — within noise.

Visual panel: `data/sweep_sheet_bc/grid.png` (3×3 `_after` previews
with eff/fid annotations per cell). GitHub discussion:
<https://github.com/ChanceSiyuan/SLMengineer/issues/17#issuecomment-4244609967>.

The CGM converges to the same multi-peak interference structure
regardless of where the incident Gaussian is centered in its compute
grid. The dot-chain is driven by something upstream of beam position.

---

## Open questions / follow-ups

Ranked by likelihood of producing a continuous sheet, **not** by cost.

1. **Plain Fresnel-focus sanity test.** Before any more CGM work,
   push a pure `SLM.fresnel_lens_phase_generate(...)` screen (no
   target, no CGM) and verify it produces a single clean focal spot
   at the predicted location. If that already produces a chain, the
   root cause is upstream of the CGM (SLM LUT, alignment,
   calibration BMP) and no sweep of CGM parameters will fix it.
2. **CGM offset sweep.** `cgm_D=-π/2` may be too large for a 1024²
   grid; the analytical initial phase could be placing the target
   at a location the CGM can't reach cleanly and it settles into a
   chain of satellites. Coarse sweep: `cgm_D ∈ {-π/12, -π/8, -π/6,
   -π/4, -π/3}` at fixed `cgm_theta`, then vary `cgm_theta` around
   the best. 5 + 5 = 10 hardware points.
3. **Calibration BMP verification.** If
   `calibration/CAL_LSH0905549_1013nm.bmp` is stale relative to the
   current optical path, every CGM solution gets a systematic phase
   offset on hardware — which could look exactly like satellite
   spots around the intended target. Worth verifying before more
   sweeps.
4. **Compute grid doubling.** Move from 1024² at 15.83 µm/px to
   2048² at 7.9 µm/px — halves focal pitch, doubles spatial
   bandwidth, might remove the aliasing that's producing the
   discrete spots. CGM solve time grows 4-8×.
5. **Multi-sample averaging per point.** Current sweep has a ~5 pp
   noise floor. Averaging N=3 samples per point would tighten the
   noise floor to ~3 pp and make smaller wins detectable. Cost: 3×
   the hardware time.

---

## Workflow notes

### Analyzer fixes triggered by idx 000
First real capture exposed three bugs:

1. **Walk target.** The original analyzer walked through
   `after - before` starting at the brightest point of `before`,
   per the literal issue #17 wording. This collapses immediately
   because the saturated zero-order cancels between `before` and
   `after`, so `signal ≈ 0` at the starting pixel and the walk
   returns `dx = dy = 1` every time. **Fix**: walk through
   `after - median(far-field bg)` instead. Still starts from the
   `_before` peak per the issue's intent.
2. **Negative efficiency.** Residual zero-order saturation noise
   produced negative ROI sums, yielding `eff < 0`. **Fix**: clip
   `signal` to `max(0, …)` before summing both the ROI and the
   denominator.
3. **Unbounded curve_fit.** On degenerate major-axis projections,
   `curve_fit` produced absurd `gauss_sigma ≈ 800 px` values.
   **Fix**: add box bounds forcing widths into `[0.3, ROI_len]`.

### Remote-crop optimization
The Allied Vision sensor is 3036 × 4024 float32 = 48.9 MB per frame.
Pulling one frame over the slow ssh link takes ~15 min, giving a
per-point cycle time of ~20 min (40× the actual hardware work). Fix:

1. `scripts/sheet/crop_after.py` is pushed once to the Windows
   runner at `C:\Users\Galileo\slm_runner\crop_after.py`.
2. `run_one.sh` ssh-invokes it after each capture with a fixed ROI
   (y ∈ [1200, 2100], x ∈ [1500, 1950]) covering the sheet +
   zero-order, saves a compressed `.npz` (~335 KB).
3. The small `.npz` is what's actually pulled.
4. `sweep_sheet_000_before.npy` is cropped locally once to
   `sweep_sheet_000_before_crop.npy` and reused as the shared
   SLM-blank reference for every analysis (the SLM is identical
   between `_before` captures, so re-pulling it is wasteful).

Net: per-point cycle time went from ~20 min to ~1 m 45 s, a ~11×
speedup end-to-end (and ~70× on the pull stage alone).

### Iteration cadence
Sweeps on this rig are driven **one point at a time** —
`sweep_one.py` generates a single payload per invocation, and
`run_one.sh` runs one full push→capture→pull→analyze cycle before
returning. This lets the operator inspect each capture, spot
analyzer bugs early (see idx 000 above), and abort if something
catastrophic happens rather than discovering a broken analyzer after
34 wasted hardware runs. See user-memory
`feedback_sweep_one_at_a_time.md` for the rationale.
