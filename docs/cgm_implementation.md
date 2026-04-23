# How CGM is Realized in `slm.cgm`

This document walks through the Conjugate Gradient Minimisation (CGM) solver in
[`src/slm/cgm.py`](../src/slm/cgm.py) — the torch-based phase-retrieval engine that
backs every continuous-target hardware script in the repo. Read
[`algorithms.md` §3](algorithms.md#3-conjugate-gradient-minimization-cgm) first for
the mathematical statement (Bowman et al.); the goal here is to explain *how* the
code turns that statement into a fast, numerically stable optimiser.

## 1. Overview

GS and WGS are **constraint-satisfaction** algorithms: each iteration forces the
focal field to look more like the target and then forces the SLM amplitude back
to the physical incident beam. They are fast per iteration and work well on
discrete spot arrays, but for continuous targets (top-hat, Laguerre–Gauss, light
sheets) they converge to a visibly wrong minimum — the *chicken-egg* problem.

CGM replaces the alternating projections with **true optimisation**: a
differentiable scalar cost that penalises mismatch between the propagated field
and the target, minimised over the SLM phase by conjugate-gradient descent. The
cost is continuous in the optimised variable, so second-order information (via
conjugate directions) accelerates convergence, and soft penalty terms for beam
efficiency can be added without breaking the formulation.

## 2. Public entry points

| Symbol | Location | What it does |
|---|---|---|
| `CGMConfig` | [cgm.py:32](../src/slm/cgm.py#L32) | Dataclass holding every tunable. Callable-friendly defaults. |
| `CGM_phase_generate(initSLMAmp, initSLMPhase, targetAmp, ...)` | [cgm.py:380](../src/slm/cgm.py#L380) | 2D entry point. Accepts torch tensors, auto-selects CUDA, runs the optimiser, returns a real float32 phase tensor. Every hardware script (`scripts/sheet/testfile_sheet.py`, `scripts/tophat/testfile_tophat.py`, …) calls this. |
| `CGM_phase_generate_1d(initSLMAmp, initSLMPhase, targetAmp, ...)` | [cgm.py:762](../src/slm/cgm.py#L762) | 1D companion for separable light-sheet targets (issue #21). Same API with length-N tensors. |
| `_initial_phase(shape, config)` | [cgm.py:361](../src/slm/cgm.py#L361) | Bowman quadratic + linear seed `φ = R(p² + q²) + D(p cos θ + q sin θ)`. Exposed because hardware scripts build the seed themselves when they need custom geometry. |

The private workhorses `_run_cgm_torch` ([cgm.py:126](../src/slm/cgm.py#L126)) and
`_run_cgm_torch_1d` ([cgm.py:553](../src/slm/cgm.py#L553)) implement the iteration
loop and are invoked by the public wrappers.

## 3. Problem formulation

Let $A$ be the fixed real incident SLM amplitude, $\varphi \in \mathbb{R}^{n_y \times n_x}$
the optimised phase, and $\mathcal{F}$ the centred 2D FFT with `norm='ortho'`. The
propagated focal field is

$$E_{\text{out}}(\varphi) = S \odot \mathcal{F}\big(A \odot e^{i\varphi}\big),$$

where $S$ is an optional per-pixel sinc envelope (§4). Masking with a dilated
measure region $M$ gives $B = M \odot E_{\text{out}}$; the optimiser only sees
the target $T$ and the field through this mask:

- **Optimised**: the real phase $\varphi$, continuous in float32. No
  quantisation is enforced; the user may post-process to 8-bit for the SLM.
- **Fixed**: the incident amplitude $A$, the target $T$ (real or complex), the
  mask $M$, and the sinc envelope $S$.
- **Auto-derived**: $M$ is built inside `CGM_phase_generate` from the target via
  [`slm.targets.measure_region`](../src/slm/targets.py) with `margin=5` by
  default — dilating the target footprint so the cost sees a halo around the
  intended region rather than a pixel-tight mask.

## 4. Forward and adjoint operators

Forward propagation ([cgm.py:82](../src/slm/cgm.py#L82)):

```python
def _fft_propagate_t(field):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field), norm='ortho'))
```

`norm='ortho'` makes the transform unitary (Parseval-preserving), which keeps
the cost scale independent of grid size. The `ifftshift → fft2 → fftshift`
pattern matches `slm.propagation.fft_propagate` exactly so CGM results are
comparable with WGS/GS on the same grid.

The adjoint is the conjugate: IFFT with the same shift convention. When the
sinc envelope is active, it is applied symmetrically:

```python
def _forward_t(E_in, sinc_env):      # forward: multiply sinc AFTER FFT
    out = _fft_propagate_t(E_in)
    return out * sinc_env if sinc_env is not None else out

def _back_t(X, sinc_env):            # adjoint: multiply sinc BEFORE IFFT
    return _ifft_propagate_t(sinc_env * X) if sinc_env is not None else _ifft_propagate_t(X)
```

Both forward and adjoint cost one FFT each per iteration — this is the dominant
cost of the loop.

**Sinc pixel-envelope**
([`_sinc_envelope_t`, cgm.py:111](../src/slm/cgm.py#L111)): `sinc(iy·ff/ny)·sinc(ix·ff/nx)`
with `ff = config.fill_factor ∈ (0, 1]`. Set `fill_factor=1.0` (default) to
skip the multiplication entirely; set it below 1 to model the loss of focal
amplitude at high spatial frequency caused by finite SLM pixel aperture.

## 5. Cost function

The base cost is the Bowman overlap cost scaled by a steep power of 10:

$$C_{\text{base}} = 10^{d}\,\big(1 - \mathrm{Re}\,\mathrm{overlap}\big)^2, \qquad
\mathrm{overlap} = \frac{\langle T_\text{masked},\, B \rangle}{\lVert T_\text{masked}\rVert\,\lVert B\rVert}.$$

With `config.steepness = 9` (default) the cost sits around $10^9$ on
uninitialised phases and drops by ~4 orders of magnitude on convergence. The
large scale factor gives float32 gradients enough dynamic range for the line
search to discriminate step sizes.

Two extensions sit on top of $C_\text{base}$ that are **implemented in code but
not in the paper**:

1. **Efficiency penalty** ([cgm.py:188](../src/slm/cgm.py#L188)): when
   `efficiency_weight > 0`, add `ew · 10^{s_η} · (1 − η)²` where
   $\eta = \lVert B \rVert^2 / \sum |E_{\text{out}}|^2$ is the fraction of
   focal-plane power inside the measure region. Pushes the solver toward
   bright, peaked outputs rather than merely *shaped* but dim ones.

2. **Efficiency floor** ([cgm.py:190](../src/slm/cgm.py#L190)): when `eta_min > 0`,
   add `10^{s_η} · max(0, η_\min − η)²`. A one-sided penalty: no force as long
   as efficiency is above the floor, but it grows quadratically once you slip
   below. Preferred over `efficiency_weight` for hardware runs — you get a
   target fidelity without sacrificing efficiency below a guaranteed minimum.

3. **Decoupled `eta_steepness`** ([cgm.py:168](../src/slm/cgm.py#L168)): the η terms
   use `10^{eta_steepness}` instead of `10^{steepness}` when set. Useful when
   `steepness=9` pushes fidelity hard but the efficiency term needs a gentler
   scale like `10^8` to avoid swamping the fidelity signal.

The two penalty terms share one forward pass with the base cost and reuse its
intermediates, so they add only a handful of element-wise ops per iteration.

## 6. Analytic gradient (adjoint trick, not autodiff)

`_run_cgm_torch.cost_and_grad` ([cgm.py:195](../src/slm/cgm.py#L195)) computes $\nabla_\varphi C$
in closed form. Writing $E_\text{in} = A e^{i\varphi}$:

$$\frac{\partial E_\text{in}}{\partial \varphi} = i E_\text{in}.$$

For any focal-plane field $X$ applied via $\mathcal{F}$, the gradient of
$\mathrm{Re}\,\langle X, \mathcal{F}(E_\text{in})\rangle$ with respect to $\varphi$
is $\mathrm{Re}(i E_\text{in} \cdot \overline{\mathcal{F}^{-1} X})$. Applying
this identity to the three pieces of the cost gives

- `d_Re_r = Real(i · E_in · conj(back_A))` where `back_A = IFFT(T_masked)` is precomputed once
- `raw_B  = Real(i · E_in · conj(back_B))` where `back_B = IFFT(B)` is recomputed every step
- `d_‖B‖ = raw_B / ‖B‖`
- `d(overlap) = d_Re_r / (‖A‖ ‖B‖) − overlap · d_‖B‖ / ‖B‖`
- `∇C_base = −2 · 10^d · (1 − overlap) · d(overlap)`

The efficiency and floor gradients ([cgm.py:216](../src/slm/cgm.py#L216)) follow the
same pattern via $d\eta = 2\,\text{raw\_B}/P$.

Cost per iteration: **2 FFTs** (one forward, one adjoint on $B$) plus the
once-per-call `back_A`. No autodiff graph is built, which keeps peak memory
equal to the working set of a single iteration (~8× the phase tensor in complex64).

## 7. Search direction — Polak-Ribière+ with periodic restart

After the first iteration the direction is updated by
([cgm.py:338](../src/slm/cgm.py#L338)):

$$\beta = \max\!\left(0,\;\frac{(g_{k+1} - g_k)^{\!\top} g_{k+1}}{\lVert g_k \rVert^2}\right),\qquad
 d_{k+1} = -g_{k+1} + \beta\, d_k.$$

**Why PR+ instead of Fletcher–Reeves?** On non-convex cost landscapes, PR is
empirically more robust — it reacts to gradient changes rather than only
magnitudes — and the `max(0, β)` clip auto-restarts to steepest descent whenever
the gradient reverses direction (β would otherwise turn negative and build up
an uphill component). This is the standard textbook trick.

On top of that, a **forced restart** fires every 50 iterations (`(i + 1) % 50 == 0`)
to purge accumulated non-conjugacy from finite precision. The `max(0, β)` clip
is a soft safety net; the periodic restart is the hard one.

If the line search returns `α = 0` (no descent found), the direction is reset
to `-g` and the line search retried once ([cgm.py:307](../src/slm/cgm.py#L307)). Two
consecutive failures break the loop.

## 8. Line search — geometric probes + golden section

The step-size search ([cgm.py:241](../src/slm/cgm.py#L241)) is the most customised
part of the solver. Three stages:

1. **Base step size**: `α_base = (π/2) / max|direction|`. This is a
   dimensional scale: any step larger than this risks wrapping phase by more
   than π per update.

2. **Geometric probes**: evaluate the cost at nine multiples of `α_base`:
   ```python
   probe_factors = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.6, 1.0, 1.5)
   ```
   This spans six orders of magnitude; the min over the nine samples is a
   robust initial guess even when the gradient's true optimal step is orders
   of magnitude off `α_base`.

3. **Golden-section refinement**: if the best probe beats `cost0`, the bracket
   `[probe[best−1], probe[best+1]]` is contracted for 20 iterations of
   golden-section search ([cgm.py:262](../src/slm/cgm.py#L262)). Gives step sizes
   within ~0.5% of the exact line minimum.

If *no* main probe descends, a wider fallback sweep
`(1e-6 · α_base, …, 1e-2 · α_base)` is tried before returning zero
([cgm.py:256](../src/slm/cgm.py#L256)). This survives the flat patches that
appear when the cost is already near its floor.

## 9. Initial phase and warm starts

Two sources of initial phase, both supported:

**Bowman analytical seed** ([cgm.py:361](../src/slm/cgm.py#L361)):

```python
φ(p, q) = R · (p² + q²) + D · (p cos θ + q sin θ)
```

Quadratic lens term (R) widens the focus; linear tilt (D, θ) shifts it away
from the zero-order. `CGMConfig(R=4.5e-3, D=-π/2, theta=π/4)` is the default —
a modest diagonal off-axis target. Set all three to zero if you pass
`initial_phase` explicitly.

**User-supplied seed**: `CGM_phase_generate(initSLMPhase=...)` (or
`CGMConfig.initial_phase=...`) overrides the analytical formula. This is the
hardware path: hardware scripts bake in a Fresnel lens, beam-centring ramp, and
a geometric-optics warm start, then let CGM polish.

**Stationary-phase warm start** ([`src/slm/initial_phase.py`](../src/slm/initial_phase.py)):
closed-form geometric-optics solution for mapping a Gaussian onto a top-hat:

$$\phi_\text{1D}(x) = \frac{\pi b}{\lambda f}\left[x\,\mathrm{erf}\!\left(\tfrac{\sqrt 2\, x}{w_0}\right) + \tfrac{w_0}{\sqrt{2\pi}}\left(e^{-2x^2/w_0^2} - 1\right)\right].$$

Derived in `references/Top Hat Beam.pdf`; exact when the target size ≫
diffraction scale $\lambda f / (\pi w_0)$. Rings badly on its own but is a
dramatically better CGM seed than Bowman's quadratic+linear in the
flat-top direction. Helpers:

- `stationary_phase_1d` — 1D closed form
- `stationary_phase_light_sheet` — 2D wrapper for line targets, with optional
  cylindrical Fresnel lens perpendicular to the line
- `cylindrical_lens_for_gaussian_width` — focal length that widens the natural
  focal Gaussian to a requested width

**Alignment step** ([`align_initial_phase`, cgm.py:230](../src/slm/cgm.py#L230)):
regardless of the seed, the solver subtracts a global phase offset
$\arg\langle T, B\rangle$ before the first iteration to remove a trivially
incorrect phase mismatch that would otherwise show up in the gradient. It is
called once on the seed and once more after the final iteration.

## 10. Convergence criterion

Two stop conditions ([cgm.py:331](../src/slm/cgm.py#L331)):

1. **Relative cost change**: stop if
   $|c_{k-1} - c_k| < \text{threshold} \cdot \max(|c_k|, 1)$ with
   `threshold = convergence_threshold` (default `1e-5`). The `max(·, 1)` lower
   bound keeps the test meaningful even when the cost drops below 1.

2. **Line-search failure twice in a row**: `α = 0` on both the current
   direction and the steepest-descent fallback, handled at
   [cgm.py:306–311](../src/slm/cgm.py#L306).

Otherwise the loop runs to `max_iterations`.

## 11. 1D dimension decomposition (issue #21)

For a light-sheet whose long axis is aligned with a coordinate axis, the
SLM phase only needs to vary along that axis: $\varphi(x, y) = \varphi(x)$.
If the input beam is separable Gaussian and the target is
$T(x, y) = T_\text{line}(x) \cdot T_\text{perp}(y)$, then

$$\mathcal{F}\{A \cdot e^{i\varphi(x)}\} = \mathcal{F}_x\{A_x \cdot e^{i\varphi(x)}\} \cdot \mathcal{F}_y\{A_y\}$$

and the y-envelope is fixed by the natural 2F transform of the input Gaussian.
Only a **1D FFT** is needed per iteration — about 10× faster than the 2D path
at the same compute-grid resolution, with proportionally less memory.

This path is implemented by `CGM_phase_generate_1d`
([cgm.py:762](../src/slm/cgm.py#L762)) and `_run_cgm_torch_1d`
([cgm.py:553](../src/slm/cgm.py#L553)). Everything else — cost, gradient, PR+,
line search, alignment — is identical to 2D; only the FFT/sinc helpers are
1D. Targets are built with
[`light_sheet_1d` (targets.py:220)](../src/slm/targets.py#L220), which returns the
top-hat profile with optional Gaussian edge taper.

## 12. Edge-induced ringing diagnosis

Hard-edged 1D top-hat targets produce visible Gibbs ringing in the focal
plane: the FFT of an ideal rectangle has sinc sidelobes that CGM struggles
to suppress because the sidelobes are outside the measure region and do not
contribute to the cost.

**Mitigation**: soften the target edges via `light_sheet_1d(edge_sigma=σ)`,
which tapers the rectangle with a Gaussian roll-off outside the flat region.
`σ ≈ 5 px` (the default in `scripts/sheet/testfile_sheet.py`) is enough to
suppress visible ringing on our hardware without noticeably widening the
central flat.

The diagnostic sweep introduced in commit `45c8f57` lives under
`scripts/1d_sheet/` and logs per-`edge_sigma` quality metrics to
`data/1d_sheet/*_analysis.json`.

## 13. Device and dtypes

- **Device** ([cgm.py:64](../src/slm/cgm.py#L64)): `CUDA if torch.cuda.is_available() else CPU`.
  If the caller passes tensors on a device that matches the auto-pick (e.g. CUDA
  tensor on a CUDA host), the run stays on that device; otherwise inputs are
  moved to the auto-selected device and the result is moved back to the
  caller's device.
- **Hot-loop dtypes** ([cgm.py:72](../src/slm/cgm.py#L72)): `complex64` and `float32`.
  Not `complex128`/`float64` — 2× throughput and 2× less memory, and float32
  precision is sufficient because the cost is bounded above by $10^d$ and
  differences well above the float32 epsilon.
- **`track_fidelity=True`** ([cgm.py:291](../src/slm/cgm.py#L291)): records fidelity
  every iteration. Opt-in because it requires a GPU→CPU sync plus complex128
  promotion, which slows the hot loop. The returned `fidelity_history` is an
  empty list when `False`.

## 14. `CGMConfig` reference

| Field | Default | Meaning |
|---|---|---|
| `max_iterations` | 200 | Hard iteration limit. Typical hardware runs use 2 000–4 000. |
| `steepness` | 9 | Base cost scale $10^d$. Higher = more aggressive on fidelity. |
| `convergence_threshold` | 1e-5 | Relative cost-change stop tolerance. |
| `R` | 4.5e-3 | Quadratic-lens coefficient in the Bowman seed (rad/px²). |
| `D` | −π/2 | Linear-tilt magnitude in the Bowman seed. |
| `theta` | π/4 | Linear-tilt angle (radians) in the Bowman seed. |
| `track_fidelity` | `False` | If `True`, record fidelity per iteration (slower). |
| `efficiency_weight` | 0.0 | Weight of the `(1 − η)²` penalty. |
| `eta_min` | 0.0 | Lower bound on η. Triggers a one-sided `(η_min − η)²` penalty below this. |
| `eta_steepness` | `None` | Separate $10^{s_η}$ scale for the η terms. `None` → reuses `steepness`. |
| `initial_phase` | `None` | Pre-computed seed (e.g. stationary phase + Fresnel). Overrides `R`/`D`/`theta`. |
| `fill_factor` | 1.0 | SLM pixel fill-factor ∈ (0, 1]. Enables the sinc envelope when `< 1`. |
| `device` | `None` | Explicit device string. `None` = auto-select CUDA if available. |

In the `CGM_phase_generate` wrapper these fields are exposed as keyword
arguments; the wrapper also accepts `margin` (default 5) for the dilation of
the measure region and `Plot` (default `False`) for a built-in convergence
curve.

## 15. How the hardware scripts use CGM

The canonical hardware entry point is
[`scripts/sheet/testfile_sheet.py`](../scripts/sheet/testfile_sheet.py). It
demonstrates the full warm-start path:

1. Build an `SLM_class` instance configured for the physical Hamamatsu SLM
   (1272×1024 native, 12.5 µm pixel pitch) on a larger compute grid
   (default 4096²) for finer focal resolution.
2. Build the target via `SLM.light_sheet_target(flat_width, gaussian_sigma,
   edge_sigma, ...)`, with the target centre shifted off-axis so the
   first-order does not land on the zero-order beam.
3. Build the initial phase via `SLM.stationary_phase_sheet(...)` — the
   pixel-indexed wrapper around `stationary_phase_light_sheet`.
4. Call `CGM_phase_generate(amp, init_phi, target, max_iterations=4000,
   steepness=9, eta_min=0.1, eta_steepness=8, ...)` for the CGM polish.
5. Wrap the phase to [−π, π], convert to 8-bit via `SLM.phase_to_screen`,
   add a post-hoc Fresnel lens, and apply the calibration BMP correction.
6. Save `payload/sheet/testfile_sheet_payload.npz` plus a sidecar
   `params.json` (with `runner_defaults.etime_us`, `n_avg`) and a
   `preview.pdf` for inspection.

The 1D equivalent is [`scripts/1d_sheet/testfile_1dsheet.py`](../scripts/1d_sheet/testfile_1dsheet.py),
which uses `stationary_phase_1d` + `CGM_phase_generate_1d`. Other hardware
scripts (`scripts/tophat/`, `scripts/ring/`, `scripts/lg/`, `scripts/gline/`)
follow the same shape with different targets.

## 16. File and symbol index

Quick lookup for the constructs referenced above:

| Symbol | File | Rough line |
|---|---|---|
| `CGMConfig` dataclass | `src/slm/cgm.py` | 32 |
| `_fft_propagate_t`, `_ifft_propagate_t` | `src/slm/cgm.py` | 82, 89 |
| `_forward_t`, `_back_t` | `src/slm/cgm.py` | 96, 104 |
| `_sinc_envelope_t` | `src/slm/cgm.py` | 111 |
| `_run_cgm_torch` | `src/slm/cgm.py` | 126 |
| `cost_value`, `cost_and_grad` (closures) | `src/slm/cgm.py` | 174, 195 |
| `align_initial_phase` (closure) | `src/slm/cgm.py` | 230 |
| `line_search` (closure) | `src/slm/cgm.py` | 241 |
| PR+ direction update | `src/slm/cgm.py` | 338 |
| Convergence test | `src/slm/cgm.py` | 331 |
| `_initial_phase` (Bowman seed) | `src/slm/cgm.py` | 361 |
| `CGM_phase_generate` | `src/slm/cgm.py` | 380 |
| 1D path `_run_cgm_torch_1d` | `src/slm/cgm.py` | 553 |
| `CGM_phase_generate_1d` | `src/slm/cgm.py` | 762 |
| `stationary_phase_1d` | `src/slm/initial_phase.py` | 128 |
| `stationary_phase_light_sheet` | `src/slm/initial_phase.py` | 170 |
| `cylindrical_lens_for_gaussian_width` | `src/slm/initial_phase.py` | 95 |
| `light_sheet`, `light_sheet_1d` | `src/slm/targets.py` | 166, 220 |
| `measure_region`, `measure_region_1d` | `src/slm/targets.py` | 551, 568 |
