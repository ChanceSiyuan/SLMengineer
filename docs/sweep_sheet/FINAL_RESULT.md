# Issue #20 — final result

Seed-only payload produced a saturated round blob on the Hamamatsu SLM + Allied Vision
camera. After three iteration phases (23 hardware captures total), the committed
configuration produces a clean continuous horizontal light sheet that is
quantitatively better than the CGM-4000 baseline on 4 of 5 meaningful metrics.

## TL;DR

**See `CL_FINAL_vs_CGM4000.png` in this directory for the head-to-head comparison.**

| metric | CGM-4000 | Final closed-loop | winner |
|---|---:|---:|---|
| continuity (along-line, above 50 % max) | 0.735 | **0.916** | CL (+25 %) |
| aspect ratio | 1.23 | **4.28** | CL (3.5×) |
| efficiency in ROI | 6.1 % | **40.2 %** | CL (6.6×) |
| line asymmetry | 0.131 | 0.202 | CGM (marginal) |
| fid_corr | 0.324 | 0.152 | CGM (favours spot concentration, not line shape) |

## Files to check

### The final result
- `fix_CL_FINAL_after.png` — camera frame with the final config
- `fix_CL_FINAL_metrics.json` — quantitative metrics
- `fix_CL_FINAL_preview.png` — 3-panel analysis preview
- `CL_FINAL_vs_CGM4000.png` — side-by-side comparison plot (images + profiles + metrics)

### Milestones from the iteration (for audit trail)
- `fix_iter_06_w25_fsd1000_after.png` — phase-1 winner (seed-only line, before asymmetry fix)
- `fix_iter_B4_cgm5_after.png` — phase-2 winner (seed + 5-iter CGM polish, pre-closed-loop)
- `fix_progression.png` — 6-panel progression plot from phase 1

### Code changes (only 2 files)
- `../../scripts/sheet/testfile_sheet.py` — configuration changes only (env-var overridable)
  - `BEAM_CENTER_DX_UM = -2000` (closed-loop winner)
  - `sheet_flat_width = 25` (was 9)
  - `sheet_edge_sigma = 5` (was 0)
  - `gaussian_sigma=None` passed to seed (cylindrical lens off — see docstring)
  - `cgm_max_iterations = 5` (was 0)
  - `setting_eta = 0.6`, `cgm_eta_steepness = 11`
- `../../scripts/sheet/closed_loop_sheet.py` — new, ~180 lines, camera-feedback driver

**No changes to `src/slm/initial_phase.py` or `src/slm/generation.py`.**
The stationary-phase algorithm matches `references/Top Hat Beam.pdf` + the
Dickey–Romero–Holswade literature canonical form exactly; all fixes are at
the calling-site / configuration layer.

### Documentation
- `FIX_LOG.md` — full per-iteration log (Phase 1/2 results)
- `SEED_DIAGNOSIS.md` — root-cause analysis predating my work (prior session)
- `SUMMARY.md` — the full CGM parameter-sweep report predating this issue
- `closed_loop_history.json` — raw Phase-3 metrics per iteration

### GitHub issue comments
- https://github.com/ChanceSiyuan/SLMengineer/issues/20#issuecomment-4269939460 — Phase 1 (seed fix)
- https://github.com/ChanceSiyuan/SLMengineer/issues/20#issuecomment-4270177577 — Phase 2 (asymmetry + CGM polish)
- https://github.com/ChanceSiyuan/SLMengineer/issues/20#issuecomment-4272191555 — Phase 3 (camera closed loop)

## How to reproduce

```bash
# Rebuild payload with the committed config
uv run python scripts/sheet/testfile_sheet.py

# Push to Windows SLM and pull back camera PNGs
./push_run.sh payload/sheet/testfile_sheet_payload.npz --png

# Re-run the closed-loop driver (takes ~5 hardware iterations, ~10 minutes)
uv run python scripts/sheet/closed_loop_sheet.py
```

All environment overrides (`SLM_BCM_DX_UM`, `SLM_CGM_MAX_ITER`,
`SLM_SETTING_ETA`, `SLM_CGM_ETA_STEEPNESS`) are documented inline in
`testfile_sheet.py`.
