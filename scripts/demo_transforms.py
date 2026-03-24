"""Demo: Zernike corrections and anti-aliased affine hologram transforms."""

import matplotlib.pyplot as plt
import numpy as np

from slm.beams import initial_slm_field
from slm.propagation import fft_propagate
from slm.targets import mask_from_target, rectangular_grid
from slm.transforms import anti_aliased_affine_transform, apply_zernike_correction
from slm.visualization import plot_intensity, plot_phase
from slm.wgs import phase_fixed_wgs

# Setup: generate a hologram with Phase-Fixed WGS
shape = (128, 128)
rng = np.random.default_rng(42)
L0 = initial_slm_field(shape, sigma=25.0, rng=rng)
target, positions = rectangular_grid(shape, rows=5, cols=5, spacing=10)
mask = mask_from_target(target)

print("Generating base hologram with Phase-Fixed WGS...")
result = phase_fixed_wgs(L0, target, mask, phase_fix_iteration=12, n_iterations=100)
base_phase = result.slm_phase
slm_amp = np.abs(L0)

print("Applying transformations...\n")

# 1. Zernike tilt correction (shifts pattern laterally)
tilt_phase = apply_zernike_correction(base_phase, {2: 3.0})  # x-tilt
E_tilt = fft_propagate(slm_amp * np.exp(1j * tilt_phase))

# 2. Zernike defocus (shifts focal plane)
defocus_phase = apply_zernike_correction(base_phase, {4: 2.0})
E_defocus = fft_propagate(slm_amp * np.exp(1j * defocus_phase))

# 3. Anti-aliased rotation (5 degrees)
rotated_phase = anti_aliased_affine_transform(
    base_phase, rotation_angle=np.radians(5), gaussian_sigma=1.5
)
E_rotated = fft_propagate(slm_amp * np.exp(1j * rotated_phase))

# 4. Naive rotation for comparison (direct phase manipulation)
from scipy.ndimage import rotate as scipy_rotate
naive_rotated = scipy_rotate(base_phase, 5, reshape=False, order=3)
E_naive = fft_propagate(slm_amp * np.exp(1j * naive_rotated))

# Plot all results
fig, axes = plt.subplots(3, 3, figsize=(14, 14))

# Row 1: Original
plot_phase(base_phase, "Original Phase", ax=axes[0, 0])
plot_intensity(result.focal_field, "Original Focal", ax=axes[0, 1])
axes[0, 2].axis("off")

# Row 2: Zernike corrections
plot_intensity(E_tilt, "Tilt (Z₂=3.0)", ax=axes[1, 0])
plot_intensity(E_defocus, "Defocus (Z₄=2.0)", ax=axes[1, 1])
axes[1, 2].axis("off")

# Row 3: Rotation comparison
plot_intensity(E_naive, "Naive Rotation (5°)", ax=axes[2, 0])
plot_intensity(E_rotated, "Anti-aliased Rotation (5°)", ax=axes[2, 1])

# Difference
diff = np.abs(E_rotated) ** 2 - np.abs(E_naive) ** 2
im = axes[2, 2].imshow(diff, cmap="RdBu_r")
axes[2, 2].set_title("Anti-aliased − Naive")
plt.colorbar(im, ax=axes[2, 2])

fig.suptitle("Hologram Transformations", fontsize=14)
fig.tight_layout()
plt.savefig("transforms_result.png", dpi=150, bbox_inches="tight")
print("Saved transforms_result.png")

plt.show()
