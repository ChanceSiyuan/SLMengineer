# SLMengineer

Iterative optimisation algorithms for **Spatial Light Modulator** (SLM) holographic
beam shaping — tweezer arrays, flat-top beams, light sheets, Laguerre–Gauss modes
— with a closed-loop runner for the Hamamatsu LCOS-SLM + Allied Vision Vimba
camera on Rydberg-atom quantum-simulator hardware.

The repo covers both ends of the workflow:

1. **Algorithms** (`src/slm/`) — GS, WGS, Phase-Fixed WGS (Kim et al. 2019),
   Conjugate Gradient Minimisation (Bowman et al. 2017), stationary-phase warm
   starts, propagation + metrics utilities. NumPy and PyTorch backends;
   CUDA-aware.
2. **Hardware pipeline** — `scripts/<pattern>/testfile_<pattern>.py` computes a
   ready-to-display uint8 SLM screen locally; [`push_run.sh`](push_run.sh)
   pushes it to the Windows lab box, runs the capture, and pulls the result
   back. See [`windows_runner/README.md`](windows_runner/README.md) for the
   lab-side runner.

## Three machine roles

The workflow is split across up to three machines, each with a distinct job.
Pick the row that matches the box you're on:

| Machine | Job | Compute entry point | Reaches the SLM via |
|---|---|---|---|
| **Remote Linux dev box** | Compute CGM + orchestrate over SSH (WAN / Tailscale) | `scripts/<shape>/testfile_*.py` → [`push_run.sh`](push_run.sh) | SSH to the `windows_remote` block in [`hamamatsu_test_config.json`](hamamatsu_test_config.json) |
| **In-lab Windows workstation** (no hardware attached) | Compute CGM + orchestrate over SSH (LAN) | `scripts/<shape>/testfile_*.py` → [`push_run.ps1`](push_run.ps1) | SSH to the `windows_local` block in [`hamamatsu_test_config.json`](hamamatsu_test_config.json) |
| **SLM-attached Windows** (lab hardware) | Receive the payload, drive the Hamamatsu SLM + Vimba camera, capture frames | [`windows_runner/slm_local.bat`](windows_runner/slm_local.bat) (standalone) **or** [`slmrun.bat`](slmrun.bat) (invoked over SSH by the first two) | direct USB/DVI to the hardware |

The first two compute the `.npz` SLM screen locally and ship it to the third.
Only the third machine needs the Hamamatsu driver + Vimba SDK + `wxPython` +
`vmbpy` (the `hardware` extra in [`pyproject.toml`](pyproject.toml)).

## Quickstart — pick the machine you're on

### 1. Remote Linux dev box (compute + drive over WAN SSH)

```bash
git clone https://github.com/ChanceSiyuan/SLMengineer.git
cd SLMengineer
uv sync --extra gpu --extra dev              # no hardware extra needed

# Compute the payload locally (CUDA-aware)
uv run python scripts/sheet/testfile_sheet.py
# → payload/sheet/testfile_sheet_payload.npz

# Push to the lab SLM, capture, pull analysis back
./push_run.sh payload/sheet/testfile_sheet_payload.npz --png-analy
# → data/sheet/testfile_sheet_analysis.{png,json}

# Closed-loop uniformity correction (issue #23) — 10 push/analyse iterations
uv run python scripts/sheet/sheet_inloop.py
# → data/sheet_inloop_<timestamp>/iter_{00..09}.{bmp,png,json} + summary.json
```

`push_run.sh` reads the **`windows_remote`** block of
[`hamamatsu_test_config.json`](hamamatsu_test_config.json) for SSH host/port/user.
Passwordless (key-based) SSH required.

### 2. In-lab Windows workstation (no SLM, drive over LAN SSH)

From **PowerShell** at the repo root:

```powershell
git clone https://github.com/ChanceSiyuan/SLMengineer.git
cd SLMengineer
uv sync --extra gpu --extra dev

# Compute the payload locally
uv run python scripts\sheet\testfile_sheet.py

# Push over LAN SSH to the SLM-attached Windows, capture, pull analysis
.\push_run.ps1 payload\sheet\testfile_sheet_payload.npz -PngAnaly

# Closed-loop — sheet_inloop.py auto-selects push_run.ps1 on Windows
uv run python scripts\sheet\sheet_inloop.py
```

`push_run.ps1` reads the **`windows_local`** block of
[`hamamatsu_test_config.json`](hamamatsu_test_config.json). Flag spelling
differs from the bash script: `-HoldOn`, `-Png`, `-PngAnaly` (single dash,
PowerShell switches).

### 3. SLM-attached Windows (hardware directly connected, no SSH)

Install the hardware extras once — this is the only machine that needs them:

```powershell
uv sync --all-extras          # wxPython + vmbpy + Pillow + zernike
```

Then drive any payload locally — same flag surface as `push_run.sh` but
without the push/pull step:

```powershell
.\windows_runner\slm_local.bat payload\sheet\testfile_sheet_payload.npz --png-analy
```

This reads the `.npz` directly from disk, invokes `windows_runner\runner.py`,
displays on the SLM, and captures with the Vimba camera. See
[`windows_runner/README.md`](windows_runner/README.md) for driver install,
Vimba SDK setup, and the OpenSSH server config required if this box also needs
to serve machines (1) or (2).

## Algorithms

| Algorithm | Use case | Module | Reference |
|---|---|---|---|
| Gerchberg–Saxton (GS) | Baseline phase retrieval | `slm.gs` | Gerchberg & Saxton 1972 |
| Weighted GS (WGS) | Uniform tweezer arrays | `slm.wgs` | Di Leonardo et al. 2007 |
| Phase-Fixed WGS | Fast-converging tweezer arrays | `slm.wgs` | Kim et al. 2019 |
| Conjugate Gradient Minimisation (CGM) | Top-hat, LG modes, light sheets, continuous patterns | `slm.cgm` | Bowman et al. 2017 |
| Stationary-phase warm start | Geometric-optics seed for CGM | `slm.initial_phase` | `references/Top Hat Beam.pdf` |
| Zernike aberration | Wavefront correction | `slm.aberration` | — |

Every hardware script in `scripts/` calls `WGS_phase_generate` or
`CGM_phase_generate` (the torch entry points). The numpy `gs` / `wgs` /
`phase_fixed_wgs` functions are the reference implementations used by the test
suite.

## Installation

Requires **Python ≥ 3.12** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ChanceSiyuan/SLMengineer.git
cd SLMengineer
uv sync --all-extras
```

Optional extras exposed in [`pyproject.toml`](pyproject.toml):

| Extra | What it adds | Needed when |
|---|---|---|
| `gpu` | `torch ≥ 2.0` | You want to run WGS/CGM on CUDA. Without it, CGM raises `ImportError`. |
| `jax` | `jax`, `jaxlib` | Alternative autodiff backend (experimental). |
| `hardware` | `wxPython`, `vmbpy`, `Pillow`, `zernike` | Only on the Windows lab box — Linux dev machines don't need this. |
| `dev` | `pytest`, `pytest-cov`, `ruff` | Running tests or linting. |
| `app` | `streamlit` | Legacy; unused on the current critical path. |

`uv sync --all-extras` installs everything at once. For a simulation-only Linux
box, `uv sync --extra gpu --extra dev` is enough.

For the Windows lab machine setup (SLM driver, Vimba SDK, SSH/SCP access,
runner invocation), see [`windows_runner/README.md`](windows_runner/README.md).

## Repository layout

```
src/slm/                       core algorithms
    gs.py                      Gerchberg–Saxton (numpy)
    wgs.py                     Weighted GS (numpy) + WGS_phase_generate (torch)
    cgm.py                     Conjugate Gradient Minimisation (torch)
    initial_phase.py           Stationary-phase warm starts for CGM
    targets.py                 Target constructors (spot grids, top-hat, LG, light sheet, …)
    propagation.py             FFT/IFFT (ortho-normalised, sinc envelope)
    metrics.py                 Uniformity, fidelity, efficiency, phase/NU error
    generation.py              SLM_class — physical SLM configuration wrapper
    aberration.py              Zernike polynomials
    imgpy.py                   Legacy calibration + quantisation helpers

scripts/
    sheet/                     2D light-sheet CGM + analysis + sweep harness
    1d_sheet/                  1D dimension-decomposed CGM (issue #21)
    tophat/                    circular flat-top CGM
    tophat_optimized/          tophat with tuned warm start
    lg/                        Laguerre–Gauss mode CGM
    ring/                      ring-of-spots + vortex CGM
    gline/                     Gaussian-line CGM
    wgs_square/                4×4 square trap WGS
    processing/                BMP → colour-heatmap PNG helper

windows_runner/                Windows-side SLM display + camera capture
    runner.py                  loads payload.npz → display → capture → save
    slm_display.py             wxPython wrapper around Hamamatsu LCOS-SLM
    vimba_camera.py            Allied Vision Vimba SDK wrapper
    README.md                  Windows setup + per-experiment workflow

payload/<pattern>/             locally-generated payloads: *.npz + *.json + *.pdf
data/<pattern>/                captured frames + run metadata pulled from Windows

tests/                         pytest suite (numpy algorithms, torch hot paths, SLM_class, targets, metrics, initial_phase)
docs/                          algorithms.md, api.md, examples.md, cgm_implementation.md
calibration/                   SLM LUT + correction BMPs (not all checked in)
references/                    Background PDFs referenced by the code (e.g. Top Hat Beam.pdf)

push_run.sh                    Linux-side orchestrator: local → Windows → local
slmrun.bat / run_in_session1.bat  Windows-side entry points invoked over SSH
hamamatsu_test_config.json     Default SLM optical configuration
pyproject.toml                 Package + dependencies
```

## Simulation quickstart

Nothing is re-exported at the package level — import from submodules directly.

### Gerchberg–Saxton + WGS on a tweezer array

```python
import numpy as np
from slm.gs import gs
from slm.wgs import phase_fixed_wgs
from slm.targets import rectangular_grid, mask_from_target

rng   = np.random.default_rng(42)
shape = (128, 128)

# Gaussian amplitude × random phase, normalised to unit power
yy, xx = np.indices(shape)
cy, cx = (s // 2 for s in shape)
amp    = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 20.0 ** 2))
field  = amp * np.exp(1j * rng.uniform(-np.pi, np.pi, shape))
field /= np.sqrt(np.sum(np.abs(field) ** 2))

target, _ = rectangular_grid(shape, rows=10, cols=10, spacing=10)
mask      = mask_from_target(target)

gs_result = gs(field, target, n_iterations=200)
pf_result = phase_fixed_wgs(field, target, mask, phase_fix_iteration=12, n_iterations=200)

print(f"GS  non-uniformity: {gs_result.uniformity_history[-1]:.4f}")
print(f"WGS non-uniformity: {pf_result.uniformity_history[-1]:.4f}")
```

### CGM flat-top beam

```python
import math
import torch
from slm.cgm import CGM_phase_generate
from slm.targets import top_hat

shape = (128, 128)
cy, cx = (s / 2 - 0.5 for s in shape)
yy, xx = torch.meshgrid(
    torch.arange(shape[0]).float() - cy,
    torch.arange(shape[1]).float() - cx,
    indexing="ij",
)
input_amp = torch.exp(-(xx ** 2 + yy ** 2) / (2 * 30.0 ** 2))
init_phi  = torch.zeros(shape)
target    = torch.from_numpy(top_hat(shape, radius=15.0)).abs().float()

phi = CGM_phase_generate(
    input_amp, init_phi, target,
    max_iterations=200,
    steepness=9,
    R=4.5e-3, D=-math.pi / 2, theta=math.pi / 4,
    Plot=True,
)
```

`Plot=True` draws the cost-history curve and prints final fidelity + efficiency.
For a walkthrough of what CGM is doing internally, see
[`docs/cgm_implementation.md`](docs/cgm_implementation.md).

## Hardware workflow details

See **[Quickstart](#quickstart--pick-the-machine-youre-on)** above for the
end-to-end recipe on each machine. This section documents the shared
artefacts and mode flags.

A single payload run produces:

```
payload/sheet/testfile_sheet_payload.npz       (uint8 SLM screen)
payload/sheet/testfile_sheet_params.json       (metadata + runner_defaults)
payload/sheet/testfile_sheet_preview.pdf       (6-panel diagnostic)
```

After `push_run.sh` / `push_run.ps1` / `slm_local.bat` completes:

```
data/sheet/testfile_sheet_{before,after}.bmp   (raw camera frames)
data/sheet/testfile_sheet_analysis.{png,json}  (with --png-analy / -PngAnaly)
data/sheet/testfile_sheet_run.json             (exposure, frame count, timestamps)
```

Mode flags (bash vs PowerShell spelling):

| bash flag | PowerShell switch | Behaviour | Pulled artefacts |
|---|---|---|---|
| *(none)* | *(none)* | Display + capture; pull raw BMP frames | `*_before.bmp`, `*_after.bmp`, `*_run.json` |
| `--png` | `-Png` | Render each BMP to a hot-colormap PNG on Windows; pull only PNGs | `*_before.png`, `*_after.png`, `*_run.json` |
| `--png-analy` | `-PngAnaly` | Run [`analysis_sheet.py`](scripts/sheet/analysis_sheet.py) on Windows; pull its outputs | `*_analysis.png`, `*_analysis.json`, `*_run.json` |
| `--hold-on` | `-HoldOn` | Display on the SLM and hold indefinitely; no capture, no pull | *(none)* |

`--png` / `-Png` and `--png-analy` / `-PngAnaly` are mutually exclusive.
`slm_local.bat` accepts the bash-style long flags (it's a thin local wrapper
around the same runner).

Other payload generators share the same env-var surface — see
[`docs/examples.md`](docs/examples.md) for the full list.

## Configuration

### Payload-generator env vars

All hardware scripts expose their knobs via environment variables so a sweep
harness (e.g. `scripts/sheet/param_sweep.sh`) can perturb one parameter without
editing the file. Defaults apply to
[`scripts/sheet/testfile_sheet.py`](scripts/sheet/testfile_sheet.py); other
scripts reuse the subset relevant to their target.

| Variable | Default | Meaning |
|---|---|---|
| `SLM_ETIME_US` | 1500 | Camera exposure time (µs). Propagates via `params.json → runner_defaults.etime_us`. |
| `SLM_N_AVG` | 20 | Number of frames to average per capture. |
| `SLM_FLAT_WIDTH` | 40 | Flat-top width in compute-grid pixels. |
| `SLM_GAUSS_SIGMA` | 3 | Gaussian envelope σ perpendicular to the sheet (px). |
| `SLM_EDGE_SIGMA` | 5 | Gaussian edge-taper σ for the top-hat (px). `0` = hard edges. |
| `SLM_TARGET_SHIFT_FPX` | 80 | Diagonal shift of the target from the grid centre (px) — prevents first-order overlap with the zero-order. |
| `SLM_FRESNEL_SD` | 1000 | Post-hoc Fresnel-lens shift distance (µm). |
| `SLM_CGM_STEEPNESS` | 9 | CGM cost scale: $10^{d}$ multiplies `(1 − overlap)²`. |
| `SLM_CGM_MAX_ITER` | 4000 | CGM iteration budget. `0` = use the stationary-phase seed directly. |
| `SLM_SETTING_ETA` | 0.1 | Efficiency floor for the `eta_min` penalty in CGM. |
| `SLM_CGM_ETA_STEEPNESS` | 8 | Separate scale for the efficiency penalty term. |
| `SLM_BCM_DX_UM`, `SLM_BCM_DY_UM` | 0, 0 | Measured incident-beam centre on the SLM plane (µm, relative to grid centre). |
| `SLM_ARRAY_BIT` | 12 | Compute-grid size exponent: `grid = 1 << bit`, so 12 → 4096². |

### Sidecar `params.json`

Each payload is accompanied by `<base>_params.json`. `push_run.sh` reads
`runner_defaults.{etime_us, n_avg}` and forwards them as `--etime-us` /
`--n-avg` to the Windows runner, so adjusting exposure in the generator
actually reaches the camera without editing the Windows-side script.

The rest of `params.json` is metadata (compute grid, focal pitch, CGM timing,
final fidelity/efficiency, calibration paths) useful for post-hoc analysis
and for comparing captures across sweeps.

### Algorithm configs

- [`CGMConfig`](src/slm/cgm.py) — every tunable exposed as a dataclass field.
  See [`docs/cgm_implementation.md §14`](docs/cgm_implementation.md#14-cgmconfig-reference)
  for the full table.
- [`WGSConfig`](src/slm/wgs.py) — `n_iterations`, `uniformity_threshold`,
  `phase_fix_iteration` for the numpy WGS path.
- [`SLM_class`](src/slm/generation.py) — reads
  [`hamamatsu_test_config.json`](hamamatsu_test_config.json) for optics
  (pixel pitch 12.5 µm, native resolution 1272 × 1024, focal length,
  wavelength, magnification).

## Outputs

A hardware run produces, on the local box:

| Artefact | Path | Format | Notes |
|---|---|---|---|
| SLM screen payload | `payload/<pattern>/<base>_payload.npz` | compressed uint8 (1024 × 1272) | Calibration + Fresnel lens already baked in; ready for direct display. |
| Generator metadata | `payload/<pattern>/<base>_params.json` | JSON | CGM timing, fidelity, efficiency, optics, target knobs, `runner_defaults`. |
| Preview | `payload/<pattern>/<base>_preview.pdf` | PDF | 6-panel: input amplitude, target intensity+phase, output intensity+phase, final SLM screen. |
| Before/after frames | `data/<pattern>/<base>_{before,after}.bmp` | uint8 greyscale | Captured by the Vimba camera; `before` is blank SLM (dark-frame). Pulled unless `--png` or `--png-analy`. |
| Colour PNGs | `data/<pattern>/<base>_{before,after}.png` | RGB | Hot-colormap render of the BMPs. Produced by `--png`. |
| Analysis | `data/<pattern>/<base>_analysis.{png,json}` | PNG + JSON | 2-panel sheet ROI + profile and uniformity / flat-top / RMS numbers. Produced by `--png-analy`. |
| Run metadata | `data/<pattern>/<base>_run.json` | JSON | Exposure, frame count, wall time, runner version. Always pulled. |

## Testing

```bash
uv run pytest tests/ -v                 # full suite
uv run pytest tests/ --cov=slm          # with coverage
uv run pytest tests/ -k cgm             # filter by name

uv run ruff check src/ tests/           # linting
```

The test modules cover the numpy algorithms (`test_gs.py`, `test_wgs.py`), the
torch hot paths (`test_wgs_torch.py`, `test_cgm.py`, `test_slm_cgm_class.py`),
`SLM_class` end-to-end (`test_generation.py`), target constructors
(`test_targets.py`, `test_bowman_targets.py`, `test_kim_targets.py`), the
stationary-phase warm starts (`test_initial_phase.py`), metrics, propagation,
and aberration. Shared fixtures live in [`tests/conftest.py`](tests/conftest.py).

## Troubleshooting

- **`ImportError: PyTorch is required for CGM`** — install the `gpu` extra:
  `uv sync --extra gpu`. CPU-only torch is fine; CGM auto-selects CUDA only
  when it's available.
- **Visible Gibbs ringing on the light-sheet capture** — increase
  `SLM_EDGE_SIGMA` (the target edge taper), or switch to the 1D path
  (`scripts/1d_sheet/testfile_1dsheet.py`) which avoids the 2D sidelobe
  coupling. See
  [`docs/cgm_implementation.md §12`](docs/cgm_implementation.md#12-edge-induced-ringing-diagnosis).
- **Low efficiency, good shape** — set `efficiency_weight > 0` or
  `eta_min > 0` (env var `SLM_SETTING_ETA`) to penalise power outside the
  target region. `eta_min` with a separate `eta_steepness` is usually
  preferable to a fidelity/efficiency trade-off.
- **Line-search stalls early / cost plateaus** — CGM does a forced
  steepest-descent restart every 50 iterations; if you see the cost not
  moving for >50 iters, the solver is already at its local minimum. Try a
  better seed (`stationary_phase_light_sheet` for line targets) rather than
  adding iterations.
- **SCP errors in `push_run.sh` / `push_run.ps1`** — SSH host/port/user
  come from [`hamamatsu_test_config.json`](hamamatsu_test_config.json)
  (`windows_remote` block for the bash script, `windows_local` block for
  the PowerShell script). Edit that file if the lab network changes. Both
  scripts require passwordless (key-based) SSH.
- **Payload displayed but capture is blank** — check `*_run.json` for the
  actual exposure used; the camera may be saturating or starving. Adjust
  `SLM_ETIME_US` in the payload script (not on the Windows side).

## Documentation

- [docs/algorithms.md](docs/algorithms.md) — mathematical reference for each algorithm.
- [docs/cgm_implementation.md](docs/cgm_implementation.md) — detailed walkthrough of the CGM solver as implemented in `src/slm/cgm.py`.
- [docs/api.md](docs/api.md) — module-level API reference.
- [docs/examples.md](docs/examples.md) — walkthroughs of every script in `scripts/` plus simulation snippets.
- [windows_runner/README.md](windows_runner/README.md) — Windows lab-machine runner: SLM driver, Vimba SDK, SSH setup, per-experiment workflow.

## References

- Kim, D. et al. *Large-Scale Uniform Optical Focus Array Generation with a Phase Spatial Light Modulator.* (2019)
- Bowman, D. et al. *High-fidelity phase and amplitude control of phase-only computer generated holograms using conjugate gradient minimisation.* (2017)
- Di Leonardo, R. et al. *Computer generation of optimal holograms for optical trap arrays.* (2007)
- Gerchberg, R. W. & Saxton, W. O. *A practical algorithm for the determination of phase from image and diffraction plane pictures.* (1972)
- `references/Top Hat Beam.pdf` — stationary-phase derivation for the Gaussian → top-hat warm start.

## License

Apache 2.0 — see [LICENSE](LICENSE).
