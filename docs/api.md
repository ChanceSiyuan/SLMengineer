# API Reference

## `slm.propagation`

| Function | Description |
|---|---|
| `fft_propagate(field)` | SLM -> focal plane via ortho-normalized FFT (centered) |
| `ifft_propagate(field)` | Focal -> SLM plane via ortho-normalized IFFT |
| `pad_field(field, target_shape)` | Zero-pad for increased focal resolution |

## `slm.beams`

| Function | Description |
|---|---|
| `gaussian_beam(shape, sigma, center, normalize)` | 2D Gaussian amplitude profile |
| `uniform_beam(shape)` | Uniform amplitude (normalized) |
| `random_phase(shape, rng)` | Random phasor exp(i*phase) |
| `initial_slm_field(shape, sigma, rng)` | Gaussian amplitude + random phase (standard L_0) |

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
| `wgs(initial_field, target, mask, config, callback)` | Weighted GS with optional phase fixing |
| `phase_fixed_wgs(initial_field, target, mask, ...)` | Convenience wrapper with phase fixing |

## `slm.cgm`

| Item | Description |
|---|---|
| `CGMConfig` | `max_iterations`, `steepness`, `convergence_threshold`, `R`, `D`, `theta`, `track_fidelity` |
| `CGMResult` | `slm_phase`, `output_field`, `cost_history`, `fidelity_history`, final metrics |
| `cgm(input_amplitude, target_field, measure_region, config, callback)` | Conjugate Gradient Minimization |

## `slm.feedback`

| Item | Description |
|---|---|
| `FeedbackConfig` | `n_correction_steps`, `inner_iterations`, `phase_fix_iteration`, `noise_level` |
| `simulate_camera_measurement(focal_field, positions, noise_level)` | Noisy intensity measurement |
| `adjust_target_weights(target, measured, positions)` | Adaptive weight correction |
| `adaptive_feedback_loop(initial_field, target, mask, positions, config, ...)` | Full feedback loop |

## `slm.transforms`

| Item | Description |
|---|---|
| `zernike(n, m, shape, radius)` | Zernike polynomial Z_n^m |
| `zernike_from_noll(j, shape, radius)` | Zernike by Noll index |
| `apply_zernike_correction(phase, coefficients)` | Add Zernike correction to hologram |
| `anti_aliased_affine_transform(phase, rotation, stretch, sigma)` | Anti-aliased rotation/stretch |
| `generate_aberration(shape, coefficients)` | Generate aberration phase for simulation |

## `slm.visualization`

| Function | Description |
|---|---|
| `plot_phase(phase, title, ax)` | Phase map with cyclic colormap |
| `plot_intensity(field, title, ax, log_scale)` | Intensity map |
| `plot_convergence(history, ylabel, title, ax)` | Convergence curve |
| `plot_comparison(results, ylabel, title)` | Multi-algorithm comparison |
| `plot_spot_histogram(intensities, title, ax)` | Spot intensity histogram |
| `plot_hologram_summary(slm_phase, focal_field, target)` | Four-panel summary |
