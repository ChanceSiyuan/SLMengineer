# API Reference

The `slm` package intentionally re-exports nothing at the top level — scripts
import from submodules directly:

```python
from slm.cgm import CGM_phase_generate, CGMConfig
from slm.wgs import phase_fixed_wgs, WGS_phase_generate
from slm.gs  import gs
from slm.targets import rectangular_grid, top_hat, light_sheet, mask_from_target, measure_region
from slm.metrics import uniformity, fidelity, efficiency
from slm.propagation import fft_propagate, ifft_propagate
from slm.generation import SLM_class
from slm.initial_phase import stationary_phase_light_sheet
```

## `slm.propagation`

| Function | Description |
|---|---|
| `fft_propagate(field)` | SLM -> focal plane via ortho-normalized FFT (centered) |
| `ifft_propagate(field)` | Focal -> SLM plane via ortho-normalized IFFT |
| `sinc_envelope(shape, fill_factor)` | Per-pixel sinc envelope modelling finite SLM fill factor |
| `zero_order_field(shape, amplitude, phase)` | Centre-only DC field (for zero-order diagnostics) |
| `realistic_propagate(field, sinc_env)` | Forward propagation with a pre-multiplied sinc envelope |
| `realistic_ifft_propagate(field, sinc_env)` | Adjoint of `realistic_propagate` |
| `pad_field(field, target_shape)` | Zero-pad for increased focal resolution |

## `slm.targets`

### Discrete spot targets

| Function | Description |
|---|---|
| `spot_array(shape, positions, amplitudes)` | Discrete spot array at given positions |
| `rectangular_grid(shape, rows, cols, spacing)` | Rectangular lattice of spots |
| `hexagonal_grid(shape, rows, cols, spacing)` | Hexagonal lattice of spots |
| `disordered_array(shape, n_spots, extent, min_distance)` | Random non-overlapping spots in circular region |

### Continuous targets

| Function | Description |
|---|---|
| `top_hat(shape, radius, center)` | Circular flat-top target |
| `gaussian_line(shape, length, width_sigma, angle, phase_gradient)` | 1D Gaussian line with optional phase ramp |
| `light_sheet(shape, flat_width, gaussian_sigma, angle, edge_sigma, center)` | 2D light-sheet: along-line top-hat with Gaussian transverse envelope and optional soft edge taper |
| `light_sheet_1d(length, flat_width, center, edge_sigma)` | 1D top-hat companion for the dimension-decomposed CGM path |
| `lg_mode(shape, l, p, w0, center)` | Laguerre-Gaussian mode LG^p_l |
| `gaussian_lattice(shape, positions, peak_sigma, phases, center)` | Sum of Gaussian peaks with per-site phases |
| `square_lattice_vortex(shape, rows, cols, spacing, peak_sigma, l)` | Square grid + vortex phase |
| `ring_lattice_vortex(shape, n_sites, ring_radius, peak_sigma, l)` | Ring of spots + vortex phase |
| `graphene_lattice(shape, rows, cols, spacing, peak_sigma)` | Honeycomb with alternating sublattice phase |
| `chicken_egg_pattern(shape, radius, rng)` | Synthetic uncorrelated intensity/phase |

### Utilities

| Function | Description |
|---|---|
| `mask_from_target(target, threshold)` | Binary mask from target field |
| `measure_region(shape, target, margin)` | Dilated measure region for CGM |
| `measure_region_1d(length, target, margin)` | 1D companion for the decomposed CGM path |

## `slm.metrics`

| Function | Description |
|---|---|
| `uniformity(intensities)` | std/mean non-uniformity (Kim et al.) |
| `efficiency(output_field, region_mask)` | Power fraction in target region |
| `modulation_efficiency(output_field, spot_positions)` | Power fraction at discrete spots |
| `fidelity(output_field, target_field, region)` | Complex-field overlap fidelity (Bowman et al.) |
| `phase_error(output_phase, target_phase, region)` | Relative phase error with cyclic correction |
| `non_uniformity_error(output_I, target_I, mask)` | Flat-region intensity non-uniformity |

## `slm.gs`

| Item | Description |
|---|---|
| `GSResult` | Dataclass: `slm_phase`, `focal_field`, `uniformity_history`, `efficiency_history`, `n_iterations` |
| `gs(initial_field, target, n_iterations, callback)` | Basic Gerchberg-Saxton algorithm |

## `slm.wgs`

| Item | Description |
|---|---|
| `WGSConfig` | `n_iterations`, `uniformity_threshold`, `phase_fix_iteration` |
| `WGSResult` | Extends GSResult: `weight_history`, `phase_fixed_at`, `spot_phase_history`, `spot_amplitude_history` |
| `wgs(initial_field, target, mask, config, callback)` | NumPy-based Weighted GS with optional phase fixing |
| `phase_fixed_wgs(initial_field, target, mask, ...)` | NumPy convenience wrapper with phase fixing |
| `WGS_phase_generate(initSLMAmp, initSLMPhase, targetAmp, Loop, threshold, Plot)` | PyTorch GPU-accelerated WGS; returns the SLM phase tensor. Used by every hardware script. |
| `WGS3D_phase_generate(...)` | 3D variant for multi-layer target stacks |
| `phase_to_screen(slm_phase)` | Quantise a wrapped phase to the uint8 SLM screen format |
| `fresnel_lens_phase_generate(shift_distance, ...)` | Post-hoc Fresnel lens used to offset the beam from the zero-order |

## `slm.cgm`

Torch-based conjugate gradient minimisation. Pass torch tensors in, receive a real
float32 phase tensor out. See [`cgm_implementation.md`](cgm_implementation.md) for
the full walkthrough.

| Item | Description |
|---|---|
| `CGMConfig` | Dataclass: `max_iterations`, `steepness`, `convergence_threshold`, `R`, `D`, `theta`, `track_fidelity`, `efficiency_weight`, `eta_min`, `eta_steepness`, `initial_phase`, `fill_factor`, `device` |
| `CGM_phase_generate(initSLMAmp, initSLMPhase, targetAmp, **kwargs)` | 2D entry point. Auto-selects CUDA; builds the measure region from the target via `measure_region(margin=5)`; returns the SLM phase as a real float32 torch tensor on the caller's device. |
| `CGM_phase_generate_1d(initSLMAmp, initSLMPhase, targetAmp, **kwargs)` | 1D companion for separable light-sheet targets (issue #21). Same API with length-N tensors. |

## `slm.initial_phase`

Closed-form stationary-phase (geometric-optics) warm starts for CGM.

| Function | Description |
|---|---|
| `stationary_phase_1d(x_um, b_um, w0_um, wavelength_um, focal_length_um)` | 1D closed-form phase mapping a Gaussian to a top-hat (see `references/Top Hat Beam.pdf`) |
| `stationary_phase_light_sheet(shape, flat_width_um, w0_um, wavelength_um, focal_length_um, pixel_pitch_um, angle, center_um, beam_center_um, perp_target_w_um)` | 2D wrapper for light-sheet targets with optional cylindrical Fresnel lens perpendicular to the line |
| `cylindrical_lens_for_gaussian_width(target_w_um, w0_um, wavelength_um, focal_length_um)` | Focal length of a cylindrical Fresnel lens that widens the natural focal Gaussian to a target width |

## `slm.aberration`

| Item | Description |
|---|---|
| `Zernike` | Zernike polynomial class (radial/azimuthal indices, evaluation, combination into aberration phase maps) |

## `slm.generation`

| Item | Description |
|---|---|
| `SLM_class` | Top-level wrapper around the physical SLM configuration: reads `hamamatsu_test_config.json`, builds the input Gaussian amplitude, computes focal-plane pitches, and exposes convenience constructors (`light_sheet_target`, `stationary_phase_sheet`, `phase_to_screen`, `fresnel_lens_phase_generate`). Used by every hardware script in `scripts/<pattern>/`. |

## `slm.imgpy`

Legacy utilities kept for hardware-side calibration and quantisation.

| Function | Description |
|---|---|
| `SLM_screen_Correct(slm_screen, LUT, correctionImgPath)` | Apply LUT + calibration-BMP correction to an 8-bit SLM screen |
| `camera_Amp_generate(init_target_amp, camera_intensity_array)` | Build an updated target amplitude from a camera frame for closed-loop feedback |
