"""Demo: Basic Gerchberg-Saxton on a 4x4 spot array."""

import matplotlib.pyplot as plt
import numpy as np

from slm.beams import initial_slm_field
from slm.gs import gs
from slm.targets import rectangular_grid
from slm.visualization import plot_convergence, plot_hologram_summary

# Setup
shape = (256, 256)
rng = np.random.default_rng(42)

# Create initial SLM field (Gaussian beam, random phase)
L0 = initial_slm_field(shape, sigma=40.0, rng=rng)

# Create 4x4 spot array target
target, positions = rectangular_grid(shape, rows=4, cols=4, spacing=20)

# Run GS algorithm
print("Running Gerchberg-Saxton (200 iterations)...")
result = gs(L0, target, n_iterations=200)

# Print results
print(f"Final non-uniformity: {result.uniformity_history[-1]:.4f}")
print(f"Final efficiency: {result.efficiency_history[-1]:.4f}")

# Plot results
fig = plot_hologram_summary(result.slm_phase, result.focal_field, target)
fig.suptitle("Gerchberg-Saxton: 4x4 Spot Array", fontsize=14)
plt.savefig("gs_result.png", dpi=150, bbox_inches="tight")
print("Saved gs_result.png")

fig2, ax = plt.subplots()
plot_convergence(result.uniformity_history, ylabel="Non-uniformity (std/mean)", title="GS Convergence", ax=ax)
plt.savefig("gs_convergence.png", dpi=150, bbox_inches="tight")
print("Saved gs_convergence.png")

plt.show()
