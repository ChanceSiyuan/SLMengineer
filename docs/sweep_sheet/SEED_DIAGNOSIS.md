# Seed-only diagnosis: why does the stationary-phase seed produce a blob?

Issue: [#20](https://github.com/ChanceSiyuan/SLMengineer/issues/20) •
Hardware: same Hamamatsu LCOS-SLM + Allied Vision camera as #19 •
Date: 2026-04-17

This follow-up tests the three hypotheses from issue #19's seed-only
finding. **Verdict: hypotheses 1 and 2 are rejected by simulation;
hypothesis 3 is confirmed but is not the full story** — the post-hoc
Fresnel lens that CGM was calibrated against leaves the *seed's*
cylindrical-lens focal plane ~4 mm away from the camera, turning what
should be a line into a spot.

---

## Hypotheses recap

1. Bug in `SLM.stationary_phase_sheet` / `slm.initial_phase.stationary_phase_light_sheet` — rejected.
2. 1024-px compute grid too coarse for a 9-px flat-top — rejected.
3. Post-hoc `fresnel_sd=1000 μm` decouples u- and v-focal planes — confirmed, but the fix requires increasing `fresnel_sd` much further for seed-only, not the other way round.

---

## Step 1 — pure simulation, two grid sizes

`scripts/sheet/diagnose_seed.py` (top half).
Seed phase is built at **N=1024** (production) and **N=4096** (4× finer)
for the *same physical* target — flat-top 142 μm full width,
perpendicular 1/e² amplitude radius 45 μm, centred −475 μm off zero-order.
No post-hoc Fresnel lens, no SLM screen quantisation, no LUT — just
`|fft_propagate(S · exp(iφ_seed))|²`.

![seed grid diagnostic](diagnose_seed_grid.png)

| grid | major FWHM (meas) | major FWHM (target) | minor FWHM (meas) | minor FWHM (target) | sim η in region |
|---|---:|---:|---:|---:|---:|
| N=1024 | 142.5 μm | 142.5 μm | 47.5 μm | 74.6 μm | 99.98 % |
| N=4096 | 138.5 μm | 142.5 μm | 51.4 μm | 74.6 μm | 99.89 % |

**Reading.** The simulated focal-plane intensity is **a clean flat-top
line** at both grid sizes, matching the 142 μm target width along the
major axis. The minor axis is narrower than the Gaussian target FWHM
(47/51 μm vs 75 μm) — the cylindrical lens underbroadens slightly —
but this is a minor shape mismatch, not a geometry failure. The seed
math and the 1024 grid are both fine.

**Hypotheses 1 and 2 are both rejected.**

---

## Step 2 — simulation with post-hoc Fresnel lens

`scripts/sheet/diagnose_seed.py` (bottom half).
Same N=1024 seed, now with the production post-hoc Fresnel lens
(`np.pi * (X² + Y²) * fresnel_sd / (λ · f² · mag⁻²)` mod 2π) added to
the SLM phase before the FFT. Swept `fresnel_sd ∈ {0, 500, 1000, 2000,
5000}` μm.

![seed + fresnel_sd simulation](diagnose_seed_fresnel.png)

| `fresnel_sd` (μm) | major FWHM (μm) | minor FWHM (μm) | sim η |
|---:|---:|---:|---:|
| 0 | 142.5 | 47.5 | 99.98 % |
| 500 | 142.5 | 47.5 | 99.97 % |
| **1000** (production) | **174.1** | **15.8** | **99.94 %** |
| 2000 | 205.8 | 15.8 | 99.72 % |
| 5000 | 332.4 | 110.8 | 85.52 % |

**Reading.** Even in simulation the production `fresnel_sd=1000 μm`
distorts the line: the along-line FWHM stretches from 142 μm to 174 μm
and the perpendicular FWHM *collapses* from 47.5 μm to 15.8 μm — the
cylindrical-lens broadening is being undone by the spherical post-hoc
lens. But the simulation still shows a recognisable *line*, not a round
blob. So the hardware behaviour (round blob) must involve something
beyond the compute-grid FFT — SLM-screen quantisation, the padded
side-strips, or a physical defocus that the simulation's idealised FFT
does not capture.

---

## Step 3 — hardware sweep of `fresnel_sd` with seed-only

5 hardware runs with `cgm_max_iterations = 0`, sweeping `fresnel_sd`.
All other parameters identical to issue #19 baseline.

| tag | `fresnel_sd` (μm) | ROI (dx × dy, cam px) | major/minor ratio | eff (ROI) | what the camera shows |
|---|---:|---:|---:|---:|---|
| `seed_fsd_0` | 0 | 46 × 83 | 0.55 | 32.6 % | round saturated blob |
| `seed_fsd_500` | 500 | 58 × 79 | 0.73 | 35.8 % | round saturated blob |
| `seedonly` (≡ fsd_1000) | 1000 | 78 × 68 | 1.15 | 35.5 % | round saturated blob |
| `seed_fsd_2000` | 2000 | 85 × 65 | 1.31 | 29.7 % | slightly elongated |
| `seed_fsd_5000` | 5000 | 139 × 64 | **2.17** | 47.3 % | **clear horizontal line** |
| `seed_fsd_10000` | 10000 | 203 × 115 | 1.77 | **58.4 %** | over-defocused — wider but rounder again |

![seed_fsd_0: round blob](seed_fsd_0_after.png)
![seed_fsd_5000: line emerging](seed_fsd_5000_after.png)
![seed_fsd_10000: over-defocused](seed_fsd_10000_after.png)

**Reading.** On hardware there *is* a monotonic trend — bigger
`fresnel_sd` stretches the sheet horizontally — but:

1. At the production value `fresnel_sd = 1000 μm`, the seed output is a
   nearly circular saturated blob (aspect 1.15:1). The cylindrical lens
   is doing nothing useful.
2. The line first becomes recognisable around `fresnel_sd = 5000 μm`
   (aspect 2.17:1). At this point the major axis is still ~3× wider
   than the 142-μm target, so it's not yet a *good* line — just clearly
   a line rather than a dot.
3. Pushing to `fresnel_sd = 10000 μm` over-defocuses — the major/minor
   aspect drops back to 1.77:1 and the spot gains diffractive ripples.

**Hypothesis 3 is confirmed qualitatively**: the post-hoc Fresnel lens
and the seed's internal cylindrical lens couple through the optical
path, and at the production value they are badly misaligned for
seed-only. The hardware needs roughly 5× more defocus compensation
than production was calibrated for.

---

## Why production `fresnel_sd=1000 μm` is fine for CGM but wrong for the seed

| aspect | CGM-optimised phase | seed-only phase |
|---|---|---|
| perpendicular focus | naturally at the 2F plane — CGM's FFT runs at that plane | built by `cylindrical_lens_for_gaussian_width` at a *different* z |
| along-line focus | at the 2F plane | at the 2F plane (the ray-redistribution term is z-invariant in paraxial ray optics) |
| camera plane | offset ~1 mm from 2F — compensated by `fresnel_sd=1000 μm` | same ~1 mm offset, but the seed's perpendicular focus lives roughly 4 mm *further* from the camera, so the camera sees a defocused 2D blur |

The cylindrical-lens focal length our params resolve to is
`f_cyl = π·w0² / (λ · √((target_w/w_nat)² − 1))` ≈ 23 m
(`w0=5000 μm`, `λ=1.013 μm`, `f_eff=200 mm`, `w_nat=λ·f/(π·w0)=12.9 μm`,
`target_w=45 μm`). The post-hoc Fresnel lens at `fresnel_sd=5000 μm`
corresponds to an effective focal length of ~8 m, which is the same
order as f_cyl — so adding `fresnel_sd=5000 μm` brings the perp
focus to roughly the camera plane. Production's `fresnel_sd=1000 μm`
(f ≈ 41 m) is way too weak for that job.

---

## Proposed fix

**Don't tune `fresnel_sd` independently.** The seed's cylindrical-lens
focal length depends on the target σ, so the "right" `fresnel_sd` for
seed-only would change every time the target shape changes. Instead:

### Option A — build the cylindrical lens at the right plane inside the seed

Modify `stationary_phase_light_sheet` to accept the camera-plane offset
as a parameter and *compose* the cylindrical lens with the existing
1-mm offset rather than assume pure 2F. Concretely: pass `f_cyl_eff` =
series combination of the cylindrical lens and the camera-offset
spherical lens, so that `fresnel_sd=1000 μm` (which fixes the
along-line axis at the camera) *also* fixes the perpendicular axis.

This is the surgical fix. One-file change in `slm.initial_phase`; no
change to `testfile_sheet.py`, no behaviour change for CGM.

### Option B — skip the cylindrical lens (`gaussian_sigma=None`)

The seed then relies on natural diffraction perpendicular width
`w_nat ≈ 12.9 μm` (1/e² amp) = 30 μm FWHM. That's narrower than the
typical target but avoids the focal-plane decoupling entirely. Cheap to
test — one call-site change in `testfile_sheet.py`.

**Caveat:** `w_nat = 12.9 μm` is *very close* to the diffraction limit
for this optics (f=200 mm, λ=1.013 μm, w0=5 mm), so `_warn_if_near_diffraction_limit`
will fire.  Acceptable for a diagnostic run.

### Option C — recalibrate `fresnel_sd` on seed-only

Decide seed-only's best `fresnel_sd` (≈5000 μm from this sweep, could
refine at 3000/4000/5000/6000 μm). Store alongside production `fresnel_sd=1000 μm`
and pick based on whether CGM iterations are used. Fastest to
implement; but it drifts if the target shape changes, and it doesn't
explain *why* the two lenses need different values — so it's a patch,
not a fix.

---

## Recommended next action

Implement **Option A**: extend `stationary_phase_light_sheet` to take
the "effective working focal length" (including `fresnel_sd`) rather
than the bare 2F. This is a ~10-line change in
`slm/initial_phase.py::stationary_phase_light_sheet` + one call-site
update in `SLM.stationary_phase_sheet`. Verify first in simulation
(repeat `diagnose_seed.py` with the fix), then one hardware run. If
the seed now produces a clean line at `fresnel_sd=1000`, Option A is
confirmed and #19's seed-only recommendation becomes viable.

If that doesn't work, fall back to Option B for an immediate unblock
while Option A is debugged.

---

## Files

```
docs/sweep_sheet/
├── SEED_DIAGNOSIS.md                 (this file)
├── diagnose_seed_grid.png            (simulation, N=1024 vs 4096)
├── diagnose_seed_grid.json
├── diagnose_seed_fresnel.png         (simulation, fresnel_sd sweep on 1024)
├── diagnose_seed_fresnel.json
├── seed_fsd_0_*                      (hardware, seed + fresnel_sd = 0)
├── seed_fsd_500_*                    (hardware, seed + fresnel_sd = 500)
├── seed_fsd_2000_*                   (hardware, seed + fresnel_sd = 2000)
├── seed_fsd_5000_*                   (hardware, seed + fresnel_sd = 5000)
└── seed_fsd_10000_*                  (hardware, seed + fresnel_sd = 10000)
```

The `fresnel_sd=1000` seed-only run was captured as `seedonly_*` during
issue #19 and is reused as the reference mid-point of this sweep.
