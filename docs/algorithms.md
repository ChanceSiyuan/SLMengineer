# Algorithm Reference

This document describes the iterative optimization algorithms implemented in `slm`.

## 1. Gerchberg-Saxton (GS)

The basic iterative Fourier transform algorithm for phase retrieval.

**Module:** `slm.gs`

### Algorithm

Given initial SLM field $L_0$ (Gaussian amplitude, random phase) and target focal-plane amplitude $|E|$:

1. $R_i = \text{FFT}(L_i)$ — forward propagate
2. $R_i' = |E| \cdot \exp(i \cdot \angle R_i)$ — replace amplitude, keep phase
3. $L_i' = \text{IFFT}(R_i')$ — backward propagate
4. $L_{i+1} = |L_0| \cdot \exp(i \cdot \angle L_i')$ — restore SLM amplitude, keep phase

### Limitations

GS converges slowly for large spot arrays and cannot guarantee uniform spot intensities. It is included as a baseline for comparison.

---

## 2. Weighted Gerchberg-Saxton (WGS)

Extends GS with adaptive intensity weighting to redistribute power to underperforming spots.

**Module:** `slm.wgs`
**Reference:** Kim et al., "In situ single-atom array synthesis using dynamic holographic optical tweezers"

### Algorithm

Adds cumulative weight correction $g_i$ to the GS loop:

1. $R_i = \text{FFT}(L_i)$
2. $R_{\text{mask},i} = E_{\text{mask}} \odot R_i$ — extract at spot positions
3. $g_i(u_m) = \frac{\langle B_i \rangle}{B_i(u_m)} \cdot g_{i-1}(u_m)$ — cumulative weight update
4. Phase update/freeze based on uniformity threshold
5. $R_i' = E \cdot g_i \cdot \exp(i \cdot \varphi_i)$
6. $L_{i+1} = |L_0| \cdot \exp(i \cdot \angle \text{IFFT}(R_i'))$

### Phase-Fixed WGS (Kim et al.)

The key innovation: after $N$ iterations of standard WGS (when modulation efficiency is high), fix the focal-plane phase $\psi_i = \psi_N$ for all subsequent iterations. This removes the parasitic phase rotation $\delta\psi_i$ introduced by the amplitude substitution step, enabling much faster convergence to target uniformity.

Typical parameters: fix phase at iteration 12, run 200 total iterations.

---

## 3. Conjugate Gradient Minimization (CGM)

Gradient-based optimization for continuous beam shaping (top-hat, LG modes).

**Module:** `slm.cgm`
**Reference:** Bowman et al., "Efficient generation of complex optical patterns with optimised holograms"

### Cost Function

$$C = 10^d \left(1 - \sum_\Omega \sqrt{\tilde{I} \cdot \tilde{T}} \cos(\Phi - \varphi)\right)^2$$

where $\tilde{I}$ and $\tilde{T}$ are intensities normalized over the measure region $\Omega$, $\Phi$ and $\varphi$ are the target and output phases, and $d$ controls steepness (typically 9).

### Algorithm

1. Initialize $\varphi = R(p^2 + q^2) + D(p\cos\theta + q\sin\theta)$ — quadratic + linear phase
2. Compute $E_\text{out} = \text{FFT}(S \cdot e^{i\varphi})$
3. Compute gradient $\nabla C$ analytically via back-propagation of sensitivity field
4. Conjugate direction: $\alpha_i = -g_i + \beta_i \alpha_{i-1}$ (Polak-Ribiere)
5. Line search: minimize $C$ along $\alpha_i$
6. Update $\varphi$; repeat until convergence

### Quality Metrics

- **Fidelity** $F = |\sum \tau^* E_\text{out}|^2$
- **Efficiency** $\eta$: fraction of power in target region
- **Phase error** $\varepsilon_\Phi$: relative phase deviation with cyclic correction
- **Non-uniformity** $\varepsilon_\nu$: intensity flatness error

---

## 4. Adaptive Feedback

Simulated closed-loop correction using camera measurements.

**Module:** `slm.feedback`

For each correction step:
1. Run Phase-Fixed WGS with current target weights
2. Measure spot intensities (with simulated noise)
3. Adjust: $T^{(j)}(u_m) = \sqrt{\langle I \rangle / I(u_m)} \cdot T^{(j-1)}(u_m)$
4. Repeat

---

## 5. Hologram Transforms

**Module:** `slm.transforms`

- **Zernike corrections:** Add Zernike polynomial phase corrections for optical alignment (tilt, defocus, astigmatism)
- **Anti-aliased affine transform:** FFT → Gaussian convolution → affine resampling → IFFT. Prevents aliasing artifacts from sharp diffraction spots during rotation/stretching.
