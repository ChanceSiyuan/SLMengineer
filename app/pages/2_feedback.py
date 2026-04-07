"""Feedback Loop — closed-loop intensity correction with simulated camera."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "src"))

from slm.beams import gaussian_beam
from slm.camera import SimulatedCamera
from slm.feedback import experimental_feedback_loop
from slm.propagation import pad_field
from slm.targets import light_sheet, measure_region, top_hat

st.set_page_config(page_title="Feedback Loop", layout="wide")
st.title("Closed-Loop Feedback Demo")
st.caption(
    "Run iterative CGM with a simulated camera to correct intensity inhomogeneities."
)

# Sidebar
st.sidebar.header("Setup")
target_type = st.sidebar.radio("Target", ["Flat Top (circle)", "Light Sheet (1D)"])
n_slm = st.sidebar.selectbox("SLM pixels", [64, 128, 256], index=1)
n_pad = n_slm * 2
sigma_mm = st.sidebar.slider("Beam sigma (mm)", 0.5, 2.0, 1.0, step=0.1)
pixel_pitch_mm = 0.024

st.sidebar.header("Feedback")
n_steps = st.sidebar.slider("Correction steps", 1, 10, 3)
inner_iters = st.sidebar.slider("CGM iterations per step", 10, 200, 50, step=10)
noise = st.sidebar.slider("Camera noise level", 0.0, 0.1, 0.02, step=0.005)

shape = (n_pad, n_pad)
D_val, theta_val = -np.pi / 2, np.pi / 4
center_off = D_val * np.cos(theta_val) * n_pad / (2 * np.pi)
center_px = (n_pad - 1) / 2.0
center = (center_px + center_off, center_px + center_off)

if target_type.startswith("Flat"):
    target = top_hat(shape, radius=n_pad // 20, center=center)
else:
    target = light_sheet(shape, flat_width=n_pad // 10, gaussian_sigma=n_pad // 50, center=center)
region = measure_region(shape, target, margin=3)

input_amp = pad_field(
    gaussian_beam((n_slm, n_slm), sigma=sigma_mm / pixel_pitch_mm, normalize=False),
    (n_pad, n_pad),
)

run = st.button("Run Feedback Loop", type="primary", use_container_width=True)

if run:
    camera = SimulatedCamera(input_amp, noise_level=noise)
    progress = st.progress(0, text="Starting...")

    results = []
    for step in range(n_steps):
        progress.progress((step + 1) / n_steps, text=f"Step {step + 1}/{n_steps}...")

    # Run the full loop
    results = experimental_feedback_loop(
        input_amp, target, region, camera,
        n_steps=n_steps, max_iter=inner_iters,
    )

    progress.empty()

    # Show convergence across steps
    st.subheader("Step-by-step metrics")
    fids = [r.final_fidelity for r in results]
    effs = [r.final_efficiency * 100 for r in results]

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax = plt.subplots(figsize=(5, 3))
        ax.plot(range(1, n_steps + 1), [1 - f for f in fids], "o-")
        ax.set_xlabel("Feedback step")
        ax.set_ylabel("1 - F")
        ax.set_yscale("log")
        ax.set_title("Fidelity improvement")
        ax.grid(True, alpha=0.3)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        fig2, ax = plt.subplots(figsize=(5, 3))
        ax.plot(range(1, n_steps + 1), effs, "s-", color="tab:orange")
        ax.set_xlabel("Feedback step")
        ax.set_ylabel("Efficiency (%)")
        ax.set_title("Efficiency across steps")
        ax.grid(True, alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # Final result mosaic
    st.subheader("Final output")
    final = results[-1]
    E_out = final.output_field
    cy, cx = int(center[0]), int(center[1])
    roi_r = n_pad // 6
    rs = slice(max(0, cy - roi_r), min(n_pad, cy + roi_r))
    cs = slice(max(0, cx - roi_r), min(n_pad, cx + roi_r))

    fig3, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].imshow(np.abs(target[rs, cs]) ** 2, cmap="hot")
    axes[0].set_title("Target")
    axes[0].axis("off")
    axes[1].imshow(np.abs(E_out[rs, cs]) ** 2, cmap="hot")
    axes[1].set_title("Output (after feedback)")
    axes[1].axis("off")
    axes[2].imshow(np.angle(E_out[rs, cs]), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[2].set_title("Output Phase")
    axes[2].axis("off")
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
