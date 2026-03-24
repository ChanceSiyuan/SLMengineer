"""Demo: CGM Laguerre-Gaussian mode generation."""

import matplotlib.pyplot as plt
import numpy as np

from slm.beams import gaussian_beam
from slm.cgm import CGMConfig, cgm
from slm.targets import lg_mode, measure_region
from slm.visualization import plot_intensity, plot_phase

# Setup
shape = (128, 128)
input_amp = gaussian_beam(shape, sigma=30.0, normalize=False)

# Target: LG^0_1 mode (ring with vortex phase)
target = lg_mode(shape, l=1, p=0, w0=15.0)
# Scale target to have comparable power
target = target * np.sqrt(np.sum(input_amp**2) / np.sum(np.abs(target) ** 2))
region = measure_region(shape, target, margin=5)

config = CGMConfig(max_iterations=200, steepness=6, R=3e-3)

print("Running CGM for LG^0_1 mode generation...")
result = cgm(input_amp, target, region, config)

print(f"\n{'Metric':<25} {'Value':>12}")
print("-" * 40)
print(f"{'Fidelity F':<25} {result.final_fidelity:>12.6f}")
print(f"{'Efficiency η':<25} {result.final_efficiency:>12.6f}")
print(f"{'Phase error ε_Φ':<25} {result.final_phase_error:>12.6f}")
print(f"{'Iterations':<25} {result.n_iterations:>12d}")

# Plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

plot_phase(result.slm_phase, "SLM Phase", ax=axes[0, 0])
plot_intensity(result.output_field, "Output Intensity", ax=axes[0, 1])
plot_intensity(target, "Target Intensity", ax=axes[0, 2])

plot_phase(np.angle(result.output_field), "Output Phase", ax=axes[1, 0])
plot_phase(np.angle(target), "Target Phase", ax=axes[1, 1])

# Phase difference in target region
phase_diff = np.angle(result.output_field) - np.angle(target)
phase_diff[np.abs(target) < np.max(np.abs(target)) * 0.1] = np.nan
axes[1, 2].imshow(phase_diff, cmap="RdBu_r", vmin=-np.pi, vmax=np.pi)
axes[1, 2].set_title("Phase Difference")
plt.colorbar(axes[1, 2].images[0], ax=axes[1, 2])

fig.suptitle("CGM: LG$^0_1$ Mode Generation", fontsize=14)
fig.tight_layout()
plt.savefig("cgm_lg_mode.png", dpi=150, bbox_inches="tight")
print("\nSaved cgm_lg_mode.png")

plt.show()
