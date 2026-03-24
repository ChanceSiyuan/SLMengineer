"""Demo: CGM flat-top beam shaping."""

import matplotlib.pyplot as plt
import numpy as np

from slm.beams import gaussian_beam
from slm.cgm import CGMConfig, cgm
from slm.targets import measure_region, top_hat
from slm.visualization import plot_convergence, plot_intensity, plot_phase

# Setup: 128x128 SLM
shape = (128, 128)
input_amp = gaussian_beam(shape, sigma=30.0, normalize=False)
target = top_hat(shape, radius=15.0)
region = measure_region(shape, target, margin=5)

config = CGMConfig(max_iterations=200, steepness=6, R=3e-3)

print("Running CGM for flat-top beam shaping...")
print(f"  Grid: {shape}, Target radius: 15 px, Steepness: 10^{config.steepness}")

result = cgm(input_amp, target, region, config)

# Print metrics table
print(f"\n{'Metric':<25} {'Value':>12}")
print("-" * 40)
print(f"{'Fidelity F':<25} {result.final_fidelity:>12.6f}")
print(f"{'Efficiency η':<25} {result.final_efficiency:>12.6f}")
print(f"{'Phase error ε_Φ':<25} {result.final_phase_error:>12.6f}")
print(f"{'Non-uniformity ε_ν':<25} {result.final_non_uniformity:>12.6f}")
print(f"{'Iterations':<25} {result.n_iterations:>12d}")

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

plot_phase(result.slm_phase, "Optimized SLM Phase", ax=axes[0, 0])
plot_intensity(result.output_field, "Output Intensity", ax=axes[0, 1])
plot_intensity(target, "Target", ax=axes[0, 2])

# Output phase in target region
out_phase = np.angle(result.output_field)
out_phase[np.abs(target) == 0] = np.nan
axes[1, 0].imshow(out_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
axes[1, 0].set_title("Output Phase (target region)")

# Cross-section comparison
center = shape[0] // 2
target_profile = np.abs(target[center, :]) ** 2
output_profile = np.abs(result.output_field[center, :]) ** 2
output_profile *= np.max(target_profile) / np.max(output_profile) if np.max(output_profile) > 0 else 1
axes[1, 1].plot(target_profile, "k--", label="Target")
axes[1, 1].plot(output_profile, "r-", label="Output")
axes[1, 1].set_title("Cross-section (center row)")
axes[1, 1].legend()

plot_convergence(result.cost_history, ylabel="Cost", title="CGM Convergence", ax=axes[1, 2])

fig.suptitle("CGM: Flat-Top Beam Shaping", fontsize=14)
fig.tight_layout()
plt.savefig("cgm_tophat.png", dpi=150, bbox_inches="tight")
print("\nSaved cgm_tophat.png")

plt.show()
