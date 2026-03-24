"""Demo: GS vs WGS vs Phase-Fixed WGS comparison on a spot array."""

import matplotlib.pyplot as plt
import numpy as np

from slm.beams import initial_slm_field
from slm.gs import gs
from slm.targets import mask_from_target, rectangular_grid
from slm.visualization import plot_comparison, plot_spot_histogram
from slm.wgs import WGSConfig, phase_fixed_wgs, wgs

# Setup
shape = (256, 256)
rng = np.random.default_rng(42)
n_iterations = 200

# Create initial SLM field
L0 = initial_slm_field(shape, sigma=40.0, rng=rng)

# Create 10x10 spot array
target, positions = rectangular_grid(shape, rows=10, cols=10, spacing=10)
mask = mask_from_target(target)

print(f"Target: {len(positions)} spots on {shape} grid")
print(f"Running {n_iterations} iterations for each algorithm...\n")

# Run all three algorithms
print("1. Gerchberg-Saxton...")
gs_result = gs(L0, target, n_iterations=n_iterations)
print(f"   Final non-uniformity: {gs_result.uniformity_history[-1]:.6f}")

print("2. Weighted GS...")
wgs_result = wgs(L0, target, mask, WGSConfig(n_iterations=n_iterations))
print(f"   Final non-uniformity: {wgs_result.uniformity_history[-1]:.6f}")

print("3. Phase-Fixed WGS (fix at iter 12)...")
pfwgs_result = phase_fixed_wgs(
    L0, target, mask, phase_fix_iteration=12, n_iterations=n_iterations
)
print(f"   Final non-uniformity: {pfwgs_result.uniformity_history[-1]:.6f}")
print(f"   Phase fixed at iteration: {pfwgs_result.phase_fixed_at}")

# Convergence comparison
fig = plot_comparison(
    {
        "GS": gs_result.uniformity_history,
        "WGS": wgs_result.uniformity_history,
        "Phase-Fixed WGS": pfwgs_result.uniformity_history,
    },
    ylabel="Non-uniformity (std/mean)",
    title="Algorithm Comparison: Convergence",
)
plt.savefig("wgs_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved wgs_comparison.png")

# Spot intensity histograms
fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, result, name in zip(
    axes,
    [gs_result, wgs_result, pfwgs_result],
    ["GS", "WGS", "Phase-Fixed WGS"],
):
    spot_intensities = np.array(
        [np.abs(result.focal_field[r, c]) ** 2 for r, c in positions]
    )
    plot_spot_histogram(spot_intensities, title=name, ax=ax)
fig2.tight_layout()
plt.savefig("wgs_histograms.png", dpi=150, bbox_inches="tight")
print("Saved wgs_histograms.png")

plt.show()
