"""Hybrid GS-then-CGM and multi-start optimization strategies.

These strategies address the CGM fidelity wall (Issue #1) by seeding
CGM with a GS-derived phase that starts in a basin with better
efficiency, avoiding the local minimum where CGM reduces region power.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from slm.cgm import CGMConfig, CGMResult, cgm
from slm.gs import gs


def gs_seed_phase(
    input_amplitude: np.ndarray,
    target_field: np.ndarray,
    n_gs_iterations: int = 50,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Run GS and return the SLM phase as a CGM initial-phase seed.

    GS preserves total power (Parseval's theorem) while reshaping
    intensity, producing a phase that encodes a good intensity
    distribution without the efficiency collapse that plagues CGM.

    Parameters
    ----------
    input_amplitude : real (ny, nx) -- incident beam amplitude.
    target_field : complex (ny, nx) -- desired focal-plane field.
    n_gs_iterations : GS iterations before extracting phase.
    rng : random number generator for the initial random phase.

    Returns
    -------
    phase : real (ny, nx) -- SLM phase compatible with CGMConfig.initial_phase.
    """
    if rng is None:
        rng = np.random.default_rng()

    random_phase = rng.uniform(-np.pi, np.pi, size=input_amplitude.shape)
    initial_field = input_amplitude * np.exp(1j * random_phase)

    result = gs(initial_field, target_field, n_gs_iterations)
    return result.slm_phase


def hybrid_gs_cgm(
    input_amplitude: np.ndarray,
    target_field: np.ndarray,
    measure_region: np.ndarray,
    config: CGMConfig = CGMConfig(),
    n_gs_iterations: int = 50,
    rng: np.random.Generator | None = None,
    callback: Callable[[int, float], None] | None = None,
) -> CGMResult:
    """Hybrid GS-then-CGM: seed CGM with a GS-derived phase.

    Runs GS for *n_gs_iterations* to establish a phase with good
    efficiency and approximate intensity match, then hands off to
    CGM for fine phase optimization.  Drop-in replacement for
    :func:`slm.cgm.cgm`.

    Parameters
    ----------
    input_amplitude : real (ny, nx) -- incident beam amplitude.
    target_field : complex (ny, nx) -- desired focal-plane field.
    measure_region : binary (ny, nx) -- region of interest.
    config : CGM configuration (initial_phase is overridden by GS seed).
    n_gs_iterations : GS iterations before switching to CGM.
    rng : random number generator for GS initial phase.
    callback : optional function called each CGM iteration with (i, cost).
    """
    seed = gs_seed_phase(input_amplitude, target_field, n_gs_iterations, rng)
    seeded_config = replace(config, initial_phase=seed)
    return cgm(input_amplitude, target_field, measure_region, seeded_config, callback)


def multistart_cgm(
    input_amplitude: np.ndarray,
    target_field: np.ndarray,
    measure_region: np.ndarray,
    config: CGMConfig = CGMConfig(),
    n_starts: int = 5,
    seed_mode: str = "gs",
    n_gs_iterations: int = 50,
    rng: np.random.Generator | None = None,
    callback: Callable[[int, float], None] | None = None,
) -> CGMResult:
    """Run CGM from multiple starting phases, return the best result.

    Parameters
    ----------
    input_amplitude : real (ny, nx) -- incident beam amplitude.
    target_field : complex (ny, nx) -- desired focal-plane field.
    measure_region : binary (ny, nx) -- region of interest.
    config : CGM configuration.
    n_starts : number of independent CGM runs.
    seed_mode : ``"gs"`` (all GS-seeded), ``"random"`` (all random phase),
        or ``"both"`` (one analytical, one GS, rest random).
    n_gs_iterations : GS iterations for GS-seeded starts.
    rng : random number generator.
    callback : optional function called each CGM iteration with (i, cost).
    """
    if rng is None:
        rng = np.random.default_rng()

    if seed_mode not in ("gs", "random", "both"):
        raise ValueError(f"seed_mode must be 'gs', 'random', or 'both', got {seed_mode!r}")

    best: CGMResult | None = None

    for k in range(n_starts):
        child_rng = np.random.default_rng(rng.integers(0, 2**63))

        if seed_mode == "gs":
            seed = gs_seed_phase(input_amplitude, target_field, n_gs_iterations, child_rng)
            run_config = replace(config, initial_phase=seed)

        elif seed_mode == "random":
            seed = child_rng.uniform(-np.pi, np.pi, size=input_amplitude.shape)
            run_config = replace(config, initial_phase=seed)

        else:  # "both"
            if k == 0:
                run_config = replace(config, initial_phase=None)
            elif k == 1:
                seed = gs_seed_phase(input_amplitude, target_field, n_gs_iterations, child_rng)
                run_config = replace(config, initial_phase=seed)
            else:
                seed = child_rng.uniform(-np.pi, np.pi, size=input_amplitude.shape)
                run_config = replace(config, initial_phase=seed)

        result = cgm(input_amplitude, target_field, measure_region, run_config, callback)
        if best is None or result.final_fidelity > best.final_fidelity:
            best = result

    if best is None:
        raise ValueError("multistart_cgm: n_starts must be >= 1")
    return best
