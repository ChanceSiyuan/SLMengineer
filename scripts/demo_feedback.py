"""Demo: Simulated adaptive feedback loop with optical aberration."""

import matplotlib.pyplot as plt
import numpy as np

from slm.beams import initial_slm_field
from slm.feedback import FeedbackConfig, adaptive_feedback_loop
from slm.metrics import uniformity
from slm.targets import mask_from_target, rectangular_grid
from slm.transforms import generate_aberration

# Setup
shape = (128, 128)
rng = np.random.default_rng(42)

# Create initial field and target
L0 = initial_slm_field(shape, sigma=25.0, rng=rng)
target, positions = rectangular_grid(shape, rows=6, cols=6, spacing=8)
mask = mask_from_target(target)

# Generate optical aberration (astigmatism + coma)
aberration = generate_aberration(shape, {5: 0.5, 6: 0.3, 7: 0.2})

print(f"Target: {len(positions)} spots ({6}x{6} grid)")
print("Aberration: astigmatism + coma (Zernike j=5,6,7)")
print()

# Run feedback loop
config = FeedbackConfig(
    n_correction_steps=5,
    inner_iterations=100,
    phase_fix_iteration=10,
    noise_level=0.01,
)

print(f"Running {config.n_correction_steps} adaptive correction steps...")
results = adaptive_feedback_loop(
    L0, target, mask, positions, config,
    aberration_phase=aberration, rng=rng,
)

# Print results per step
print(f"\n{'Step':<6} {'Non-uniformity':>15} {'Efficiency':>12}")
print("-" * 35)
for i, r in enumerate(results):
    nu = r.uniformity_history[-1]
    eff = r.efficiency_history[-1]
    print(f"{i+1:<6} {nu:>15.6f} {eff:>12.6f}")

# Plot convergence across feedback steps
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Non-uniformity per step
step_uniformities = [r.uniformity_history[-1] for r in results]
axes[0].plot(range(1, len(step_uniformities) + 1), step_uniformities, "bo-")
axes[0].set_xlabel("Correction Step")
axes[0].set_ylabel("Non-uniformity")
axes[0].set_title("Adaptive Feedback: Non-uniformity")
axes[0].grid(True, alpha=0.3)

# Efficiency per step
step_efficiencies = [r.efficiency_history[-1] for r in results]
axes[1].plot(range(1, len(step_efficiencies) + 1), step_efficiencies, "ro-")
axes[1].set_xlabel("Correction Step")
axes[1].set_ylabel("Modulation Efficiency")
axes[1].set_title("Adaptive Feedback: Efficiency")
axes[1].grid(True, alpha=0.3)

fig.suptitle("Adaptive Feedback Loop with Optical Aberration", fontsize=14)
fig.tight_layout()
plt.savefig("feedback_result.png", dpi=150, bbox_inches="tight")
print("\nSaved feedback_result.png")

plt.show()
