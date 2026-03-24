# Examples

All scripts are in the `scripts/` directory. Run from the repo root:

```bash
uv run python scripts/<script_name>.py
```

## Demo Scripts

### `demo_gs.py` -- Basic Gerchberg-Saxton

Runs GS on a 4x4 spot array (256x256 grid, 200 iterations). Shows GS limitations: spots converge but with significant non-uniformity.

**Output:** `gs_result.png`, `gs_convergence.png`

### `demo_wgs.py` -- Algorithm Comparison

Compares GS, WGS, and Phase-Fixed WGS on a 10x10 spot array. Reproduces the key result from Kim et al.: Phase-Fixed WGS achieves sub-percent non-uniformity much faster than standard WGS.

**Output:** `wgs_comparison.png`, `wgs_histograms.png`

### `demo_cgm_tophat.py` -- CGM Flat-Top Beam

Shapes a Gaussian beam into a circular flat-top using CGM. Prints the Bowman metrics table (fidelity, efficiency, phase error, non-uniformity).

**Output:** `cgm_tophat.png`

### `demo_cgm_lg_mode.py` -- CGM Laguerre-Gaussian Mode

Generates an LG^0_1 mode (ring with vortex phase) using CGM. Demonstrates simultaneous amplitude and phase control.

**Output:** `cgm_lg_mode.png`

### `demo_feedback.py` -- Adaptive Feedback Loop

Simulates the full experimental workflow: generate hologram, measure through simulated aberration and noise, adjust weights, repeat.

**Output:** `feedback_result.png`

### `demo_transforms.py` -- Hologram Transformations

Demonstrates Zernike tilt/defocus corrections and anti-aliased rotation vs naive rotation.

**Output:** `transforms_result.png`

---

## Paper Reproductions

### `reproduce_kim.py` -- Kim et al. WGS Paper

Reproduces all numerical results from the WGS paper:

```bash
uv run python scripts/reproduce_kim.py --figure2   # CGH calculation comparison
uv run python scripts/reproduce_kim.py --figure3   # Adaptive correction (50x30)
uv run python scripts/reproduce_kim.py --figure5   # Hex + disordered adaptive
uv run python scripts/reproduce_kim.py --all        # Everything
uv run python scripts/reproduce_kim.py --fast       # Quick verification
```

**Figure 2:** GS vs WGS vs Phase-Fixed WGS convergence on 50x30 rectangular, hexagonal (720 spots), and disordered (819 spots) arrays. Includes CGH phase visualization, focal-plane intensity, and per-spot phase/amplitude evolution.

**Figure 3:** Ensemble-averaged adaptive CGH correction (8 runs) for 50x30 grid with simulated aberration. Compares WGS vs Phase-Fixed correction convergence with error bars, CGH correction visualization, and intensity histograms.

**Figure 5:** Same adaptive correction for hexagonal and disordered geometries (5 correction steps).

**Output:** `kim_figure2.png`, `kim_figure3.png`, `kim_figure5.png`

### `reproduce_bowman.py` -- Bowman et al. CGM Paper

Reproduces all numerical results from the top-hat paper:

```bash
uv run python scripts/reproduce_bowman.py --table1   # Table 1 metrics (7 patterns)
uv run python scripts/reproduce_bowman.py --figure2   # Intensity/phase mosaic
uv run python scripts/reproduce_bowman.py --figure3   # Gaussian line diagnostics
uv run python scripts/reproduce_bowman.py --all        # Everything
uv run python scripts/reproduce_bowman.py --fast       # Quick verification
```

**Table 1:** Runs CGM on 7 target patterns (LG mode, square/ring/graphene lattices, flat-top, Gaussian line, chicken & egg) with paper-exact parameters (256x256 SLM padded to 512x512, d=9). Reports fidelity, efficiency, phase error, and non-uniformity.

**Figure 2:** 7x2 grid showing intensity and phase for each pattern in the region of interest.

**Figure 3:** Gaussian line diagnostics -- fidelity vs iteration for different steepness d, final fidelity vs d, and fidelity/efficiency 2D parameter scans over beam size and phase curvature.

**Output:** `bowman_figure2.png`, `bowman_figure3.png`
