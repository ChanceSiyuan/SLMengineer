# SLMengineer

Offline simulation of Spatial Light Modulator (SLM) iterative optimization algorithms for holographic beam shaping and optical tweezer array generation in Rydberg atom quantum simulators.

## Algorithms

| Algorithm | Use Case | Module | Reference |
|---|---|---|---|
| Gerchberg-Saxton (GS) | Baseline phase retrieval | `slm.gs` | Gerchberg & Saxton 1972 |
| Weighted GS (WGS) | Uniform tweezer arrays | `slm.wgs` | Di Leonardo et al. 2007 |
| Phase-Fixed WGS | Fast-converging tweezer arrays | `slm.wgs` | Kim et al. 2019 |
| Conjugate Gradient (CGM) | Top-hat, LG modes, continuous patterns | `slm.cgm` | Bowman et al. 2017 |
| Adaptive Feedback | Closed-loop correction with camera | `slm.feedback` | Kim et al. 2019 |
| Zernike / Affine Transforms | Hologram alignment and correction | `slm.transforms` | Manovitz et al. 2025 |

## Installation

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ChanceSiyuan/SLMengineer.git
cd SLMengineer
uv sync --all-extras
```

## Quick Start

### Tweezer Array with Phase-Fixed WGS

```python
import numpy as np
from slm import initial_slm_field, rectangular_grid, mask_from_target, phase_fixed_wgs

L0 = initial_slm_field((256, 256), sigma=40.0, rng=np.random.default_rng(42))
target, positions = rectangular_grid((256, 256), rows=10, cols=10, spacing=10)
mask = mask_from_target(target)

result = phase_fixed_wgs(L0, target, mask, phase_fix_iteration=12, n_iterations=200)
print(f"Non-uniformity: {result.uniformity_history[-1]:.4f}")
```

### Flat-Top Beam with CGM

```python
from slm import gaussian_beam, top_hat, measure_region, cgm, CGMConfig

input_amp = gaussian_beam((128, 128), sigma=30.0, normalize=False)
target = top_hat((128, 128), radius=15.0)
region = measure_region((128, 128), target, margin=5)

result = cgm(input_amp, target, region, CGMConfig(max_iterations=200, steepness=6))
print(f"Fidelity: {result.final_fidelity:.6f}")
```

## Project Structure

```
src/slm/
    propagation.py    FFT/IFFT (ortho-normalized, Parseval-preserving)
    beams.py          Gaussian beam, random phase, initial SLM field
    targets.py        Spot arrays (rect/hex/disordered), top-hat, LG modes,
                      lattice patterns with vortex/alternating phase
    metrics.py        Uniformity, fidelity, efficiency, phase/NU error
    gs.py             Gerchberg-Saxton
    wgs.py            Weighted GS + Phase-Fixed WGS (Kim et al.)
    cgm.py            Conjugate Gradient Minimization (Bowman et al.)
    feedback.py       Simulated adaptive camera feedback
    transforms.py     Zernike polynomials, anti-aliased affine transform
    visualization.py  Plotting utilities

scripts/
    demo_*.py             6 standalone demo scripts
    reproduce_bowman.py   Reproduce Bowman et al. Table 1, Figures 2-3
    reproduce_kim.py      Reproduce Kim et al. Figures 2, 3, 5

tests/                    81 tests across 11 test files
docs/                     Algorithm reference, API docs, examples
```

## Paper Reproductions

Reproduce all numerical results from the reference papers:

```bash
# Kim et al. - WGS tweezer arrays (Figures 2, 3, 5)
uv run python scripts/reproduce_kim.py --all

# Bowman et al. - CGM beam shaping (Table 1, Figures 2, 3)
uv run python scripts/reproduce_bowman.py --all

# Fast mode for quick verification
uv run python scripts/reproduce_kim.py --all --fast
uv run python scripts/reproduce_bowman.py --all --fast
```

## Running Tests

```bash
uv run pytest tests/ -v                    # all tests
uv run pytest tests/ --cov=slm             # with coverage
uv run ruff check src/ tests/              # linting
```

## Demo Scripts

```bash
uv run python scripts/demo_gs.py           # Basic GS on 4x4 spot array
uv run python scripts/demo_wgs.py          # GS vs WGS vs Phase-Fixed WGS
uv run python scripts/demo_cgm_tophat.py   # CGM flat-top beam shaping
uv run python scripts/demo_cgm_lg_mode.py  # CGM Laguerre-Gaussian mode
uv run python scripts/demo_feedback.py     # Adaptive feedback with aberration
uv run python scripts/demo_transforms.py   # Zernike + anti-aliased transforms
```

See [docs/examples.md](docs/examples.md) for detailed descriptions.

## Documentation

- [docs/algorithms.md](docs/algorithms.md) -- Algorithm descriptions with equations
- [docs/api.md](docs/api.md) -- Module-level API reference
- [docs/examples.md](docs/examples.md) -- Demo script walkthroughs

## References

- Kim, D. et al. "Large-Scale Uniform Optical Focus Array Generation with a Phase Spatial Light Modulator." (2019)
- Bowman, D. et al. "High-fidelity phase and amplitude control of phase-only computer generated holograms using conjugate gradient minimisation." (2017)

## License

Apache 2.0 -- see [LICENSE](LICENSE).
