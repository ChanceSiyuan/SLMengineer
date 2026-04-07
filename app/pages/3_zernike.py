"""Zernike Aberration Explorer — interactive visualization of aberration modes."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "src"))

from slm.transforms import generate_aberration, zernike_from_noll

st.set_page_config(page_title="Zernike Explorer", layout="wide")
st.title("Zernike Aberration Explorer")
st.caption("Adjust sliders to visualise individual Zernike modes and their combination.")

NOLL_NAMES = {
    1: "Piston",
    2: "Tilt X",
    3: "Tilt Y",
    4: "Defocus",
    5: "Astigmatism 45",
    6: "Astigmatism 0",
    7: "Coma Y",
    8: "Coma X",
    9: "Trefoil Y",
    10: "Trefoil X",
    11: "Spherical",
}

shape = (128, 128)

st.sidebar.header("Zernike Coefficients")
n_modes = st.sidebar.slider("Number of modes", 4, 11, 6)

coeffs = {}
for j in range(1, n_modes + 1):
    name = NOLL_NAMES.get(j, f"Z{j}")
    val = st.sidebar.slider(f"j={j}: {name}", -2.0, 2.0, 0.0, step=0.05, key=f"z{j}")
    if abs(val) > 1e-6:
        coeffs[j] = val

# Generate combined aberration
aberration = generate_aberration(shape, coeffs) if coeffs else np.zeros(shape)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Combined Aberration")
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(aberration, cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_title(f"Sum of {len(coeffs)} active modes")
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="Phase (rad)", shrink=0.8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Individual Modes")
    active = {j: v for j, v in coeffs.items() if abs(v) > 1e-6}
    if not active:
        st.info("Move a slider to see individual mode shapes.")
    else:
        n_active = len(active)
        ncols = min(3, n_active)
        nrows = (n_active + ncols - 1) // ncols
        fig2, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes_flat = np.atleast_1d(axes).ravel()
        for idx, (j, v) in enumerate(active.items()):
            Z = zernike_from_noll(j, shape)
            axes_flat[idx].imshow(v * Z, cmap="RdBu_r", vmin=-3, vmax=3)
            name = NOLL_NAMES.get(j, f"Z{j}")
            axes_flat[idx].set_title(f"j={j}: {name} ({v:+.2f})")
            axes_flat[idx].axis("off")
        for idx in range(len(active), len(axes_flat)):
            axes_flat[idx].axis("off")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

# Summary
if coeffs:
    st.markdown("### Active coefficients")
    summary = " | ".join(
        f"**j={j}** ({NOLL_NAMES.get(j, '?')}): {v:+.2f}" for j, v in coeffs.items()
    )
    st.markdown(summary)
    rms = np.std(aberration[aberration != 0]) if np.any(aberration != 0) else 0
    st.metric("Wavefront RMS", f"{rms:.3f} rad ({rms / (2 * np.pi) * 1e3:.1f} milli-waves)")
