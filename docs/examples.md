# Examples

Hands-on walkthroughs for the scripts in `scripts/`. Two flavours of use:

- **Simulation only** — run the algorithms directly from Python, verify numerically, inspect with matplotlib. No hardware needed.
- **Hardware payload pipeline** — `scripts/<pattern>/testfile_<pattern>.py` computes the SLM screen locally, then [`push_run.sh`](../push_run.sh) uploads it to the Windows lab box, displays it on the Hamamatsu SLM, captures frames with the Allied Vision camera, and pulls the results back.

The hardware scripts are the primary way the codebase is used day-to-day; the simulation snippets are lightweight ways to play with the algorithms in isolation.

## Simulation snippets

Every snippet is self-contained and runs against the published `slm.*` modules. Nothing is re-exported at the top level — import from the submodule directly.

### Gerchberg–Saxton on a 4×4 spot array

```python
import numpy as np
from slm.gs import gs
from slm.targets import rectangular_grid, mask_from_target

rng = np.random.default_rng(42)
shape = (128, 128)

y = np.arange(shape[0]) - (shape[0] - 1) / 2
x = np.arange(shape[1]) - (shape[1] - 1) / 2
yy, xx = np.meshgrid(y, x, indexing="ij")
amp = np.exp(-(xx**2 + yy**2) / (2 * 20.0**2))
field = amp * np.exp(1j * rng.uniform(-np.pi, np.pi, shape))
field /= np.sqrt(np.sum(np.abs(field) ** 2))

target, positions = rectangular_grid(shape, rows=4, cols=4, spacing=10)

result = gs(field, target, n_iterations=200)
print(f"Non-uniformity: {result.uniformity_history[-1]:.4f}")
```

### Phase-fixed WGS tweezer array (Kim et al.)

```python
from slm.wgs import phase_fixed_wgs
from slm.targets import rectangular_grid, mask_from_target

target, _ = rectangular_grid(shape, rows=10, cols=10, spacing=10)
mask = mask_from_target(target)

result = phase_fixed_wgs(field, target, mask, phase_fix_iteration=12, n_iterations=200)
print(f"Final non-uniformity: {result.uniformity_history[-1]:.6f}")
print(f"Phase fixed at iteration: {result.phase_fixed_at}")
```

### CGM flat-top beam (Bowman et al.)

```python
import torch
from slm.cgm import CGM_phase_generate, CGMConfig
from slm.targets import top_hat

shape = (128, 128)
input_amp = torch.exp(-(torch.arange(shape[0]).float()[:, None] - 63.5) ** 2 / (2 * 30 ** 2))
input_amp = input_amp * torch.exp(-(torch.arange(shape[1]).float() - 63.5) ** 2 / (2 * 30 ** 2))
init_phi  = torch.zeros(shape)
target    = torch.from_numpy(top_hat(shape, radius=15.0)).real

phi = CGM_phase_generate(
    input_amp, init_phi, target,
    max_iterations=200,
    steepness=9,
    R=4.5e-3, D=-3.14/2, theta=3.14/4,
    Plot=False,
)
print(f"Output shape: {phi.shape}, device: {phi.device}")
```

See [`cgm_implementation.md`](cgm_implementation.md) for a detailed walkthrough of what CGM is actually doing inside that call.

## Hardware payload scripts

Each script under `scripts/<pattern>/` produces a triple in `payload/<pattern>/`:

```
payload/<pattern>/<base>_payload.npz       # uint8 SLM screen, ready to display
payload/<pattern>/<base>_params.json       # metadata + runner_defaults
payload/<pattern>/<base>_preview.pdf       # 6-panel diagnostic preview
```

The typical workflow is:

```bash
uv run python scripts/<pattern>/testfile_<pattern>.py     # 1. compute
./push_run.sh payload/<pattern>/<base>_payload.npz        # 2. display + capture
```

| Script | Target | Algorithm | Notes |
|---|---|---|---|
| [`scripts/sheet/testfile_sheet.py`](../scripts/sheet/testfile_sheet.py) | 2D light sheet (along-line top-hat × Gaussian perpendicular) | CGM on the full 2D grid | Canonical CGM hardware script. Uses `stationary_phase_sheet` warm start + 4 000-iteration CGM polish. Env vars: `SLM_FLAT_WIDTH`, `SLM_GAUSS_SIGMA`, `SLM_EDGE_SIGMA`, `SLM_CGM_STEEPNESS`, `SLM_CGM_MAX_ITER`, `SLM_SETTING_ETA`, `SLM_CGM_ETA_STEEPNESS`, `SLM_TARGET_SHIFT_FPX`, `SLM_FRESNEL_SD`, `SLM_ETIME_US`, `SLM_N_AVG`, `SLM_BCM_DX_UM`, `SLM_BCM_DY_UM`, `SLM_ARRAY_BIT`. |
| [`scripts/1d_sheet/testfile_1dsheet.py`](../scripts/1d_sheet/testfile_1dsheet.py) | 1D top-hat along the line axis | 1D CGM (`CGM_phase_generate_1d`) | Dimension-decomposed path (issue #21). ~10× faster than the 2D sheet at equivalent compute-grid resolution; same env-var surface. |
| [`scripts/tophat/testfile_tophat.py`](../scripts/tophat/testfile_tophat.py) | Circular flat-top | CGM | Parallel to the sheet script; swaps the target for `SLM.top_hat_target`. |
| [`scripts/tophat_optimized/testfile_tophat_optimized.py`](../scripts/tophat_optimized/testfile_tophat_optimized.py) | Circular flat-top | CGM with tuned warm start | Tophat variant with an optimised initial phase. |
| [`scripts/lg/testfile_lg.py`](../scripts/lg/testfile_lg.py) | LG⁰₁ donut (Laguerre–Gauss) | CGM | Bowman analytic seed + CGM polish. Demonstrates simultaneous amplitude + vortex-phase control. |
| [`scripts/ring/testfile_ring.py`](../scripts/ring/testfile_ring.py) | Ring of Gaussian peaks with global vortex phase | CGM | Uses `SLM.ring_lattice_vortex_target`. |
| [`scripts/gline/testfile_gline.py`](../scripts/gline/testfile_gline.py) | 1D line with Gaussian cross-section | CGM | Uses `SLM.gaussian_line_target`. |
| [`scripts/wgs_square/testfile_wgs_square.py`](../scripts/wgs_square/testfile_wgs_square.py) | 4×4 square trap array | WGS (`WGS_phase_generate`) | Shows the WGS path through the same payload format used by the CGM scripts — the Windows runner is shape-agnostic. |

## Analysis + processing

- [`scripts/sheet/analysis_sheet.py`](../scripts/sheet/analysis_sheet.py) — post-capture analysis for light-sheet runs. Takes `--after`, `--before` (for dark-frame subtraction), `--plot`, `--result` and emits a 2-panel PNG + JSON with uniformity / flat-top bounds / RMS. Automatically invoked by `./push_run.sh … --png-analy`.
- [`scripts/sheet/param_sweep.sh`](../scripts/sheet/param_sweep.sh) — light-sheet parameter sweep harness. Drives `testfile_sheet.py` with a grid of env-var overrides and compares the resulting analysis JSONs.
- [`scripts/processing/bmp_to_color.py`](../scripts/processing/bmp_to_color.py) — render a raw BMP capture to a hot-colormap PNG. Useful for sharing a frame without transferring the full uint8 BMP. Run locally; the `--png` flag of `push_run.sh` wires the same thing up on the Windows side.

## End-to-end light-sheet example

The canonical hardware loop, from a fresh checkout:

```bash
# 1. Compute the payload locally (CGM on GPU if available)
SLM_FLAT_WIDTH=40 SLM_EDGE_SIGMA=5 SLM_CGM_MAX_ITER=4000 \
  uv run python scripts/sheet/testfile_sheet.py
# -> payload/sheet/testfile_sheet_payload.npz + params.json + preview.pdf

# 2. Display on the SLM, capture, run the analysis script on the Windows side,
#    and pull the analysis PNG + JSON + run metadata back.
./push_run.sh payload/sheet/testfile_sheet_payload.npz --png-analy
# -> data/sheet/testfile_sheet_analysis.png
#    data/sheet/testfile_sheet_analysis.json
#    data/sheet/testfile_sheet_run.json
```

Alternate `push_run.sh` modes:

| Flag | Behaviour | Pulled artefacts |
|---|---|---|
| *(none)* | Display + capture; pull raw BMPs | `*_before.bmp`, `*_after.bmp`, `*_run.json` |
| `--png` | Also render each BMP to a hot-colormap PNG on Windows | `*_before.png`, `*_after.png`, `*_run.json` |
| `--png-analy` | Run `analysis_sheet.py` remotely; pull its outputs | `*_analysis.png`, `*_analysis.json`, `*_run.json` |
| `--hold-on` | Display on the SLM and hold indefinitely; no capture, no pull | *(none)* |

`--png` and `--png-analy` are mutually exclusive.
