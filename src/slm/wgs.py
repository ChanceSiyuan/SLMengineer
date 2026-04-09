"""Weighted Gerchberg-Saxton and Phase-Fixed WGS algorithms (Kim et al.).

Includes both the NumPy/SciPy implementation (wgs, phase_fixed_wgs) and
the PyTorch GPU-accelerated implementation (WGS_phase_generate, WGS3D_phase_generate)
from ~/slm-code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    import torch.fft as fft
except ImportError:
    torch = None
    fft = None

import matplotlib.pyplot as plt

from slm.gs import GSResult
from slm.metrics import uniformity
from slm.propagation import (
    fft_propagate,
    ifft_propagate,
    realistic_ifft_propagate,
    realistic_propagate,
)


@dataclass
class WGSConfig:
    """Configuration for WGS algorithm."""

    n_iterations: int = 200
    uniformity_threshold: float = 0.005
    phase_fix_iteration: int | None = (
        None  # N for phase-fixed variant; None = never fix
    )


@dataclass
class WGSResult(GSResult):
    """Extended result with WGS-specific data."""

    weight_history: list[float] = field(default_factory=list)
    phase_fixed_at: int | None = None
    spot_phase_history: list[np.ndarray] = field(default_factory=list)
    spot_amplitude_history: list[np.ndarray] = field(default_factory=list)


def wgs(
    initial_field: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    config: WGSConfig = WGSConfig(),
    callback: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
    sinc_env: np.ndarray | None = None,
) -> WGSResult:
    """Weighted Gerchberg-Saxton with optional phase fixing.

    Parameters
    ----------
    initial_field : L_0, complex (ny, nx) -- Gaussian amplitude, random phase.
    target : E, complex (ny, nx) -- desired focal plane amplitudes.
    mask : E_mask, binary (ny, nx) -- 1 at target spot positions, 0 elsewhere.
    config : algorithm parameters.
    callback : optional per-iteration callback(i, slm_field, focal_field).

    The algorithm follows Kim et al.:
        1. R_i = FFT(L_i)
        2. R_mask_i = mask * R_i
        3. g_i update: cumulative weight correction
        4. Phase decision (update or freeze based on uniformity threshold)
        5. Optional phase fix at iteration N (Kim's key innovation)
        6. R_i' = target * g_i * exp(i * phase_i)
        7. L' = IFFT(R_i')
        8. L_{i+1} = |L_0| * exp(i * angle(L'))
    """
    target_amp = np.abs(target)
    slm_amp = np.abs(initial_field)
    mask_bool = mask > 0

    L = initial_field.copy()
    g = np.ones_like(target_amp)  # cumulative weight
    fixed_phase = None
    phase_fixed_at = None
    current_phase = None

    uniformity_hist = []
    efficiency_hist = []
    weight_hist = []
    spot_phase_hist = []
    spot_amp_hist = []

    # Parseval: total power is constant under ortho FFT
    total_power = float(np.sum(slm_amp**2))

    _fwd = (
        (lambda f: realistic_propagate(f, sinc_env))
        if sinc_env is not None
        else fft_propagate
    )
    _inv = (
        (lambda f: realistic_ifft_propagate(f, sinc_env))
        if sinc_env is not None
        else ifft_propagate
    )

    for i in range(config.n_iterations):
        R = _fwd(L)

        spot_amps = np.abs(R[mask_bool])
        if len(spot_amps) == 0:
            break
        mean_amp = np.mean(spot_amps)

        spot_intensities = spot_amps**2
        uniformity_hist.append(uniformity(spot_intensities))
        if total_power > 0:
            efficiency_hist.append(float(np.sum(spot_intensities) / total_power))
        else:
            efficiency_hist.append(0.0)
        weight_hist.append(float(np.std(g[mask_bool])))
        spot_phase_hist.append(np.angle(R[mask_bool]))
        spot_amp_hist.append(spot_amps)

        if callback is not None:
            callback(i, L, R)

        # Cumulative weight update (only at spot positions)
        if mean_amp > 0:
            g[mask_bool] *= mean_amp / np.maximum(spot_amps, 1e-30)

        # Phase decision
        if config.phase_fix_iteration is not None and i >= config.phase_fix_iteration:
            if fixed_phase is None:
                fixed_phase = np.angle(R)
                phase_fixed_at = i
            current_phase = fixed_phase
        elif i == 0 or (
            mean_amp > 0
            and np.max(spot_amps) / mean_amp - 1.0 > config.uniformity_threshold
        ):
            current_phase = np.angle(R)
        # else: keep current_phase from previous iteration

        R_prime = target_amp * g * np.exp(1j * current_phase)
        L_prime = _inv(R_prime)
        L = slm_amp * np.exp(1j * np.angle(L_prime))

    # Final propagation
    focal_field = _fwd(L)

    return WGSResult(
        slm_phase=np.angle(L),
        focal_field=focal_field,
        uniformity_history=uniformity_hist,
        efficiency_history=efficiency_hist,
        n_iterations=config.n_iterations,
        weight_history=weight_hist,
        phase_fixed_at=phase_fixed_at,
        spot_phase_history=spot_phase_hist,
        spot_amplitude_history=spot_amp_hist,
    )


def phase_fixed_wgs(
    initial_field: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    phase_fix_iteration: int = 12,
    n_iterations: int = 200,
    uniformity_threshold: float = 0.005,
    callback: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
    sinc_env: np.ndarray | None = None,
) -> WGSResult:
    """Convenience wrapper: WGS with phase fixing at iteration N.

    Kim et al. fix phase at iteration 12 with ~91.2% modulation efficiency,
    then continue for ~200 total iterations to reach <0.5% non-uniformity.
    """
    config = WGSConfig(
        n_iterations=n_iterations,
        uniformity_threshold=uniformity_threshold,
        phase_fix_iteration=phase_fix_iteration,
    )
    return wgs(initial_field, target, mask, config, callback, sinc_env)


# ---------------------------------------------------------------------------
# PyTorch GPU-accelerated WGS (from ~/slm-code)
# ---------------------------------------------------------------------------


def WGS_phase_generate(initSLMAmp, initSLMPhase, targetAmp, Loop=5, threshold=0.01, Plot=False):
    '''
    This function uses WGS algorithm to generate hologram according to your input targetAmp.

    Parameters
    ----------
    initSLMAmp : 2D_tensor
        The amplitude distrubution of the incident laser, typically gaussian distrubution.

    initSLMPhase : 2D_tensor
        The phase distrubution of SLM plane for the initial calculation. If generate a new target, use a random initPhase.
        If generate a series of dynamic targets, use the phase generated by the last WGS.

    targetAmp : 2D_tensor
        The amplitude distrubution of the target.

    Loop :
        The number of iteration loops in WGS calculation, typically 5~20 is enough.

    threshold :
        The phase fix threshold in WGS calculation. If the non-uniformity < threshold, the phase in focal plane will be fixed.

    Plot :
        Whether to plot non-uniformity vs iteration. Default to be False.

    '''
    if torch is None:
        raise ImportError("PyTorch is required for WGS_phase_generate. Install with: pip install torch")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    initSLMAmp = initSLMAmp.to(device)
    initSLMPhase = initSLMPhase.to(device)
    targetAmp = targetAmp.to(device)

    SLM_Field = torch.multiply(initSLMAmp, torch.exp(1j*initSLMPhase))
    targetAmp = targetAmp/torch.sqrt(torch.sum(torch.square(targetAmp)))
    targetAmpmask = (targetAmp != 0)*1
    totalsites = torch.count_nonzero(targetAmp)
    count = 0
    g_coeff0 = torch.ones(1).to(device)
    Focal_phase = torch.zeros_like(targetAmp).to(device)
    fftAmp = torch.zeros_like(targetAmp).to(device)
    non_uniform = torch.zeros(Loop).to(device)

    targetAmp_weightfactor = torch.abs(targetAmp) / torch.sum(torch.abs(targetAmp))

    while count < Loop:
        fftSLM = fft.fft2(SLM_Field)
        fftSLMShift = fft.fftshift(fftSLM)
        fftSLM_norm = torch.sqrt(torch.sum(torch.square(torch.abs(fftSLMShift))))
        fftSLMShift_norm = fftSLMShift / fftSLM_norm

        fftAmp = torch.abs(fftSLMShift_norm)
        fftAmp_foci = torch.multiply(fftAmp, targetAmpmask)

        non_uniform[count] = nonUniformity_adapt(fftAmp_foci, targetAmp, totalsites)

        fftAmp_foci_avg = torch.multiply(torch.sum(fftAmp_foci) / totalsites, targetAmpmask)

        g_coeff = torch.where(fftAmp_foci != 0,
                              torch.multiply(torch.div(torch.multiply(fftAmp_foci_avg, targetAmp_weightfactor), fftAmp_foci),g_coeff0),
                              torch.zeros_like(fftAmp_foci_avg).to(device))

        Focal_Amp = torch.multiply(targetAmp, g_coeff)

        if non_uniform[count] > threshold or count == 0:
            Focal_phase0 = torch.angle(fftSLMShift_norm)
        else:
            Focal_phase0 = Focal_phase

        Focal_phase = Focal_phase0
        Focal_Field = torch.multiply(Focal_Amp, torch.exp(1j * Focal_phase))

        SLM_Field = fft.ifft2(fft.ifftshift(Focal_Field))
        SLM_Phase = torch.angle(SLM_Field)

        SLM_Field = torch.multiply(initSLMAmp, torch.exp(1j * SLM_Phase))
        g_coeff0 = g_coeff
        count += 1

    SLM_Phase = torch.angle(SLM_Field)

    if Plot:
            plt.plot(non_uniform.cpu())
            plt.grid()
            plt.yscale('log')
            plt.xlabel('Iteration')
            plt.ylabel('Non-uniformity')
            plt.show()

    return SLM_Phase


def WGS3D_phase_generate(initSLMAmp, initSLMPhase, targetAmp, targetLayer, Loop=5, threshold=0.01, Plot=False):
    '''
    This function uses 3D WGS algorithm to generate hologram according to your input targetAmp.

    Parameters
    ----------
    initSLMAmp : 2D_tensor
        The amplitude distrubution of the incident laser, typically gaussian distrubution.

    initSLMPhase : 2D_tensor
        The phase distrubution of SLM plane for the initial calculation. If generate a new target, use a random initPhase.
        If generate a series of dynamic targets, use the phase generated by the last WGS.

    targetAmp : 3D_tensor
        Containing multi layers of targetAmp. [targetAmp1, targetAmp2, targetAmp3...]

    targetLayer : 1D_tensor
        Representing the position of each target layer relative to the actual focal plane. The unit is um. [z1, z2, z3...]

    Loop :
        The number of iteration loops in WGS calculation, typically 5~20 is enough.

    threshold :
        The phase fix threshold in WGS calculation. If the non-uniformity < threshold, the phase in focal plane will be fixed.

    Plot :
        Whether to plot non-uniformity vs iteration. Default to be False.

    '''
    if torch is None:
        raise ImportError("PyTorch is required for WGS3D_phase_generate. Install with: pip install torch")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    initSLMAmp = initSLMAmp.to(device)
    initSLMPhase = initSLMPhase.to(device)
    targetAmp = targetAmp.to(device)

    layer_num = targetLayer.shape[0]

    SLM_Field = torch.multiply(initSLMAmp, torch.exp(1j*initSLMPhase))
    targetAmp = targetAmp/torch.sqrt(torch.sum(torch.square(targetAmp)))
    targetAmpmask = (targetAmp > 0)*1
    totalsites = torch.count_nonzero(targetAmp)

    count = 0
    g_coeff0 = torch.ones(1).to(device)
    Focal_phase = torch.zeros_like(targetAmp).to(device)
    Propagation_Field = torch.zeros_like(targetAmp, dtype=torch.complex128).to(device)
    Backpropagation_Field = torch.zeros_like(targetAmp, dtype=torch.complex128).to(device)
    non_uniform = torch.zeros(Loop).to(device)

    targetAmp_weightfactor = torch.abs(targetAmp) / torch.sum(torch.abs(targetAmp))

    while count < Loop:

        for i in range(layer_num):
            layer = targetLayer[i]
            Propagation_Field[i] = fft.fftshift(fft.fft2( torch.multiply(SLM_Field, torch.exp(-1j*fresnel_lens_phase_generate(layer))) ))

        Propagation_Field_norm_coeff = torch.sqrt(torch.sum(torch.square(torch.abs(Propagation_Field))))
        Propagation_Field_norm = Propagation_Field/Propagation_Field_norm_coeff

        Propagation_Amp = torch.abs(Propagation_Field_norm)
        Propagation_Amp_foci = torch.multiply(Propagation_Amp, targetAmpmask)

        non_uniform[count] = nonUniformity_adapt(Propagation_Amp_foci, targetAmp, totalsites)

        Propagation_Amp_foci_avg = torch.multiply(torch.sum(Propagation_Amp_foci) / totalsites, targetAmpmask)
        g_coeff = torch.where(Propagation_Amp_foci != 0, torch.multiply(torch.div(torch.multiply(Propagation_Amp_foci_avg, targetAmp_weightfactor), Propagation_Amp_foci),g_coeff0), torch.zeros_like(Propagation_Amp_foci_avg).to(device))
        Focal_Amp = torch.multiply(targetAmp, g_coeff)

        if non_uniform[count] > threshold or count == 0:
            Focal_phase0 = torch.angle(Propagation_Field_norm)
        else:
            Focal_phase0 = Focal_phase

        Focal_phase = Focal_phase0
        Focal_Field = torch.multiply(Focal_Amp, torch.exp(1j * Focal_phase))

        for i in range(layer_num):
            layer = targetLayer[i]
            Backpropagation_Field[i] = torch.multiply(fft.ifft2(fft.ifftshift(Focal_Field[i])), torch.exp(1j*fresnel_lens_phase_generate(layer)))

        SLM_Field = Backpropagation_Field.sum(axis=0)
        SLM_Phase = torch.angle(SLM_Field)

        SLM_Field = torch.multiply(initSLMAmp, torch.exp(1j * SLM_Phase))
        g_coeff0 = g_coeff
        count += 1

    SLM_Phase = torch.angle(SLM_Field)

    if Plot:
            plt.plot(non_uniform.cpu())
            plt.grid()
            plt.yscale('log')
            plt.xlabel('Iteration')
            plt.ylabel('Non-uniformity')
            plt.show()

    return SLM_Phase


def fresnel_lens_phase_generate(shift_distance, SLMRes=(4096,4096), x0=2048, y0=2048, pixelpitch=17, wavelength=0.813, focallength=4000, magnification=0.375):
    '''
    the fresnel lens phase, see notion for more details.
    '''
    if torch is None:
        raise ImportError("PyTorch is required for fresnel_lens_phase_generate. Install with: pip install torch")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    Xps, Yps = torch.meshgrid(torch.linspace(0, SLMRes[0], SLMRes[0]), torch.linspace(0, SLMRes[1], SLMRes[1]))
    Xps = Xps.to(device)
    Yps = Yps.to(device)

    X = (Xps-x0)*pixelpitch
    Y = (Yps-y0)*pixelpitch

    fresnel_lens_phase = torch.fmod(torch.pi*(X**2+Y**2)*shift_distance/(wavelength*focallength**2)*magnification**2,2*torch.pi)

    return fresnel_lens_phase


def nonUniformity_adapt(Amp_foci, targetAmp_adapt, totalsites):
    '''
    This function calculates the nonUniformity with the non-uniform targetAmp distribution.
    Amp_foci is the field amplitude at the focal plane. It is a tensor.
    targetAmp_adapt is the non-uniform targetAmp distribution.
    '''
    if torch is None:
        raise ImportError("PyTorch is required for nonUniformity_adapt. Install with: pip install torch")

    Inten_foci = torch.square(Amp_foci) / torch.sum(torch.square(Amp_foci))
    Inten_foci_nonzero = torch.abs(Inten_foci[Inten_foci != 0])
    Inten_adapt = torch.square(targetAmp_adapt) / torch.sum(torch.square(targetAmp_adapt))
    Inten_adapt_nonzero = torch.abs(Inten_adapt[Inten_adapt != 0])
    non_Uniform = torch.sqrt(torch.sum(torch.square(Inten_foci_nonzero - Inten_adapt_nonzero))) / totalsites / torch.mean(Inten_adapt_nonzero)

    return non_Uniform


def phase_to_screen(SLM_Phase):
    '''
    This function converts the SLM_Phase calculated by WGS to fit to the SLM screen.
    Temporarily for 4096*4096 phase to 1024*1272 screen.
    '''
    SLM_IMG = SLM_Phase[1536:2560, 1536:2560]
    SLM_bit = np.around((SLM_IMG+np.pi)/(2*np.pi)*256).astype('uint8')

    return SLM_bit


def phase_to_screen_cuda(SLM_Phase):
    if torch is None:
        raise ImportError("PyTorch is required for phase_to_screen_cuda. Install with: pip install torch")
    device = "cuda"
    SLM_Phase = SLM_Phase.to(device)
    SLM_IMG = SLM_Phase[1536:2560, 1536:2560]
    SLM_bit = torch.round((SLM_IMG + torch.pi) / (2 * torch.pi) * 256).to(dtype=torch.uint8)

    return SLM_bit


def slm_screen_correct(slm_screen, fresnel_lens_screen, slm_correction, LUT):
    if torch is None:
        raise ImportError("PyTorch is required for slm_screen_correct. Install with: pip install torch")
    slm_screen_f = slm_screen + fresnel_lens_screen
    slm_screen_f_corrected = slm_screen_f + slm_correction
    slm_screen_f_corrected_LUT = torch.round(slm_screen_f_corrected * LUT).to(torch.uint8)
    slm_screen_f_corrected_LUT_cpu = slm_screen_f_corrected_LUT.clone().cpu().numpy()

    return slm_screen_f_corrected_LUT_cpu
