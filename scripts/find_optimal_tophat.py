"""Find optimal (sigma, radius) for top-hat on 256x256 padded to 512x512.

Goal: beat Bowman et al. on BOTH metrics: 1-F < 1.8e-4 AND η > 11.3%.
"""

import time

from slm.beams import gaussian_beam
from slm.cgm import CGMConfig
from slm.cgm_lbfgsb import cgm_lbfgsb
from slm.propagation import pad_field
from slm.targets import measure_region, top_hat

slm_shape = (256, 256)
fft_shape = (512, 512)

sigmas = [40, 50, 60, 70, 80, 100]
radii = [30, 40, 50, 60, 70, 80, 100]

print("=== Padded Top-Hat Sweep: 256x256 SLM → 512x512 FFT ===")
print(f"{'σ':>5} {'r':>5} {'1-F':>12} {'η':>10} {'beat?':>6} {'time':>8}")
print("-" * 52)

winners = []

for sigma in sigmas:
    slm_amp = gaussian_beam(slm_shape, sigma=float(sigma), normalize=False)
    input_amp = pad_field(slm_amp, fft_shape)

    for radius in radii:
        target = top_hat(fft_shape, radius=float(radius))
        region = measure_region(fft_shape, target, margin=5)
        config = CGMConfig(max_iterations=2000, steepness=9, R=4.5e-3)

        t0 = time.time()
        result = cgm_lbfgsb(input_amp, target, region, config)
        dt = time.time() - t0

        inf = 1 - result.final_fidelity
        eta = result.final_efficiency
        beats = inf < 1.8e-4 and eta > 0.113
        tag = "YES!" if beats else ""

        print(f"{sigma:5d} {radius:5d} {inf:12.2e} {eta:10.4f} {tag:>6} {dt:7.1f}s")

        if beats:
            winners.append((sigma, radius, inf, eta))

print()
if winners:
    print("=== WINNERS (beat paper on both metrics) ===")
    for s, r, inf, eta in winners:
        print(f"  σ={s}, r={r}: 1-F={inf:.2e}, η={eta:.4f}")
else:
    print("No configuration beat the paper on both metrics.")
    print("Best fidelity and best efficiency shown above.")
