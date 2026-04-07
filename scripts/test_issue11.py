"""Test Issue #11: trace the fidelity-efficiency Pareto frontier.

Strategy: GS seed gives high η, then run L-BFGS-B iteration by iteration,
tracking (1-F, η) at each step. Find the iteration where η crosses the target.
"""

from dataclasses import replace

import numpy as np

from slm.beams import gaussian_beam
from slm.cgm import CGMConfig
from slm.cgm_lbfgsb import cgm_lbfgsb
from slm.metrics import efficiency, fidelity
from slm.hybrid import gs_seed_phase
from slm.propagation import fft_propagate, pad_field
from slm.targets import measure_region, top_hat

n_slm, n_pad = 256, 512
pixel_pitch_mm = 0.024
_d = CGMConfig()
offset = _d.D * np.cos(_d.theta) * n_pad / (2 * np.pi)
c = (n_pad - 1) / 2.0
center = (c + offset, c + offset)

target = top_hat((n_pad, n_pad), radius=25.0, center=center)
region = measure_region((n_pad, n_pad), target, margin=5)

eta_target = 0.113
print("=== Issue #11: Pareto Frontier via Incremental Optimization ===\n")

for sig_mm in [1.0, 1.5]:
    sigma_px = sig_mm / pixel_pitch_mm
    slm_amp = gaussian_beam((n_slm, n_slm), sigma=sigma_px, normalize=False)
    input_amp = pad_field(slm_amp, (n_pad, n_pad))

    print(f"--- sigma={sig_mm}mm ---")

    # Strategy 1: GS seed then incremental L-BFGS-B (no eta_min)
    seed = gs_seed_phase(input_amp, target, 100)
    E_seed = fft_propagate(input_amp * np.exp(1j * seed))
    eta_seed = efficiency(E_seed, region)
    fid_seed = fidelity(E_seed, target, region)
    print(f"GS(100) seed: 1-F={1-fid_seed:.2e}, eta={eta_seed*100:.1f}%")

    best_at_target = None
    for max_it in [5, 10, 20, 50, 100, 200, 500]:
        cfg = CGMConfig(max_iterations=max_it, steepness=9, R=4.5e-3)
        cfg = replace(cfg, initial_phase=seed)
        r = cgm_lbfgsb(input_amp, target, region, cfg)
        inf = 1 - r.final_fidelity
        eta = r.final_efficiency
        tag = ""
        if eta >= eta_target and (best_at_target is None or inf < best_at_target[0]):
            best_at_target = (inf, eta)
            tag = " <-- best at target eta"
        print(f"  iters={max_it:4d}: 1-F={inf:.2e}, eta={eta*100:.2f}%{tag}", flush=True)

    # Strategy 2: Standard + eta_min (current approach for reference)
    cfg2 = CGMConfig(max_iterations=500, steepness=9, R=4.5e-3, eta_min=0.113)
    cfg2 = replace(cfg2, initial_phase=seed)
    r2 = cgm_lbfgsb(input_amp, target, region, cfg2)
    print(f"  eta_min=0.113 (500it): 1-F={1-r2.final_fidelity:.2e}, eta={r2.final_efficiency*100:.2f}%")

    # Strategy 3: Bidirectional eta penalty (penalize deviation from target)
    for mu in [0.5, 1.0, 2.0]:
        cfg3 = CGMConfig(max_iterations=500, steepness=9, R=4.5e-3,
                         efficiency_weight=mu, eta_min=0.113)
        cfg3 = replace(cfg3, initial_phase=seed)
        r3 = cgm_lbfgsb(input_amp, target, region, cfg3)
        print(f"  ew={mu} + eta_min=0.113: 1-F={1-r3.final_fidelity:.2e}, eta={r3.final_efficiency*100:.2f}%",
              flush=True)

    if best_at_target:
        print(f"  ** Best at eta>={eta_target*100:.1f}%: 1-F={best_at_target[0]:.2e}, eta={best_at_target[1]*100:.2f}%")
    print()

print("Paper reference: 1-F=1.80e-04, eta=11.3%")
