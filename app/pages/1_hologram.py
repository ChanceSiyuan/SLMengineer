"""Hologram Generator — target pattern -> GS-seeded CGM (torch/CUDA) -> results."""

import io
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "src"))

from slm.beams import gaussian_beam
from slm.cgm import CGMConfig, cgm
from slm.device import SLMDevice
from slm.hybrid import gs_seed_phase
from slm.propagation import pad_field
from slm.targets import (
    gaussian_line,
    graphene_lattice,
    lg_mode,
    light_sheet,
    measure_region,
    rectangular_grid,
    ring_lattice_vortex,
    square_lattice_vortex,
    top_hat,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.presets import (
    DEVICE_PRESETS,
    GRADIENT_PARAMS,
    PATTERN_CATALOG,
    SPATIAL_PARAMS,
)


# -- Helpers ------------------------------------------------------------------


def _estimate_roi(params: dict) -> int:
    """Compute a good ROI size from pattern parameters (in pixels)."""
    candidates = [
        params.get("flat_width", 0),
        params.get("radius", 0) * 2,
        params.get("ring_radius", 0) * 2 + 30,
        params.get("spacing", 0)
        * max(params.get("rows", 1), params.get("cols", 1)),
        6 * params.get("gaussian_sigma", 0),
        params.get("w0", 0) * 6,
        params.get("length", 0),
        80,  # minimum
    ]
    return int(max(candidates) * 1.5)


def _masked_phase(field: np.ndarray) -> np.ndarray:
    """Phase array with NaN where amplitude is < 1% of peak (renders transparent)."""
    amp = np.abs(field)
    phase = np.angle(field)
    threshold = 0.01 * amp.max() if amp.max() > 0 else 0
    return np.where(amp > threshold, phase, np.nan)


def _roi_slices(center, roi, n_pad):
    cy, cx = int(center[0]), int(center[1])
    r = roi // 2
    return (
        slice(max(0, cy - r), min(n_pad, cy + r)),
        slice(max(0, cx - r), min(n_pad, cx + r)),
    )


def _physical_extent(rs, cs, focal_pitch):
    """Compute imshow extent in um for a pixel ROI.

    Returns [left, right, bottom, top] with y-axis flipped for imshow.
    """
    ny = rs.stop - rs.start
    nx = cs.stop - cs.start
    hw = nx * focal_pitch / 2.0
    hh = ny * focal_pitch / 2.0
    return [-hw, hw, hh, -hh]


def _plot_mosaic(target_roi, output_roi, extent=None, input_amp=None,
                 slm_pitch_um=None, figsize=None):
    """Create target/output-intensity/input-beam/output-phase mosaic figure."""
    has_beam = input_amp is not None
    ncols = 4 if has_beam else 3
    if figsize is None:
        figsize = (4 * ncols, 3.5)
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    kw = {"extent": extent} if extent is not None else {}

    axes[0].imshow(np.abs(target_roi) ** 2, cmap="hot", **kw)
    axes[0].set_title("Target")
    axes[1].imshow(np.abs(output_roi) ** 2, cmap="hot", **kw)
    axes[1].set_title("Output Intensity")

    if has_beam:
        beam_img = np.abs(input_amp) ** 2
        ny, nx = beam_img.shape
        if slm_pitch_um is not None:
            hw = nx * slm_pitch_um / 2.0
            hh = ny * slm_pitch_um / 2.0
            beam_ext = [-hw, hw, hh, -hh]
            axes[2].imshow(beam_img, cmap="viridis", extent=beam_ext)
            axes[2].set_xlabel("x (\u00b5m)")
            axes[2].set_ylabel("y (\u00b5m)")
        else:
            axes[2].imshow(beam_img, cmap="viridis")
        axes[2].set_title("Input Beam (SLM)")

    ph_ax = axes[3] if has_beam else axes[2]
    im_ph = ph_ax.imshow(
        _masked_phase(output_roi), cmap="twilight",
        vmin=-np.pi, vmax=np.pi, **kw,
    )
    ph_ax.set_title("Output Phase")
    cbar = fig.colorbar(im_ph, ax=ph_ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels(["-\u03c0", "-\u03c0/2", "0", "\u03c0/2", "\u03c0"])
    cbar.set_label("Phase (rad)")

    for ax in axes:
        if ax.get_title() != "Input Beam (SLM)" and extent is not None:
            ax.set_xlabel("x (\u00b5m)")
            ax.set_ylabel("y (\u00b5m)")
        elif ax.get_title() != "Input Beam (SLM)":
            ax.axis("off")

    fig.tight_layout()
    return fig


# -- Page ---------------------------------------------------------------------

st.set_page_config(page_title="Hologram Generator", layout="wide")
st.title("Hologram Generator")

# -- Sidebar: Device + Algorithm ----------------------------------------------

st.sidebar.header("SLM Device")
device_name = st.sidebar.selectbox("Hardware preset", list(DEVICE_PRESETS.keys()))
if device_name == "Custom":
    pixel_pitch = st.sidebar.number_input("Pixel pitch (\u00b5m)", value=24.0, step=0.5)
    n_px = st.sidebar.number_input("SLM pixels (one side)", value=256, step=64)
    wl = st.sidebar.number_input("Wavelength (nm)", value=1070.0, step=10.0)
    fl = st.sidebar.number_input("Focal length (mm)", value=150.0, step=10.0)
    device = SLMDevice(pixel_pitch, (n_px, n_px), wl, fl)
else:
    device = DEVICE_PRESETS[device_name]

pad_factor = st.sidebar.selectbox("Pad factor", [1, 2, 4], index=1)
n_slm = min(device.n_pixels)
n_pad = n_slm * pad_factor

focal_pitch = device.focal_plane_pitch_um(n_pad)  # um per focal-plane pixel

st.sidebar.info(
    f"Focal-plane resolution: {focal_pitch:.4f} \u00b5m/px\n\n"
    f"SLM aperture: {n_slm * device.pixel_pitch_um:.0f} \u00b5m "
    f"({n_slm * device.pixel_pitch_um / 1000:.2f} mm)"
)

st.sidebar.header("Optimizer")
max_iters = st.sidebar.slider("Max iterations", 10, 1000, 200, step=10)
n_gs_iters = st.sidebar.slider(
    "GS seed iterations", 0, 200, 50, step=10,
    help="Gerchberg-Saxton warm-up iterations before L-BFGS-B. "
         "Set to 0 to use analytical initial phase only.",
)
steepness = st.sidebar.slider("Steepness d", 1, 12, 9)
eta_min = st.sidebar.slider("Min efficiency floor", 0.0, 0.3, 0.20, step=0.01)
sigma_mm = st.sidebar.slider(
    "Input beam \u03c3 (mm on SLM)", 0.5, 3.0, 1.5, step=0.1,
    help="1/e\u00b2 Gaussian radius of the illumination beam on the SLM surface.",
)
st.sidebar.caption(
    f"\u2248 {sigma_mm / device.pixel_pitch_mm:.0f} px on SLM"
)
R_mrad = st.sidebar.slider(
    "Initial phase R (mrad/px\u00b2)", 0.5, 8.0, 4.5, step=0.5,
)

# -- Main: Pattern selector ---------------------------------------------------

col_pat, col_res = st.columns([1, 1])

with col_pat:
    st.subheader("Target Pattern")
    pattern_name = st.selectbox("Pattern", list(PATTERN_CATALOG.keys()))
    pat = PATTERN_CATALOG[pattern_name]
    st.caption(pat["desc"])

    # Dynamic parameter sliders with physical units
    params_px = {}
    for k, default_px in pat["params"].items():
        if k in SPATIAL_PARAMS:
            default_um = default_px * focal_pitch
            max_um = max(default_um * 4, 100 * focal_pitch)
            step_um = max(round(focal_pitch, 4), 0.001)
            val_um = st.slider(
                f"{k} (\u00b5m)",
                min_value=step_um,
                max_value=round(max_um, 4),
                value=round(default_um, 4),
                step=step_um,
                key=f"p_{k}",
            )
            params_px[k] = val_um / focal_pitch

        elif k in GRADIENT_PARAMS:
            default_mrad_um = default_px / focal_pitch * 1000.0
            max_mrad_um = max(default_mrad_um * 4, 50.0)
            val_mrad_um = st.slider(
                f"{k} (mrad/\u00b5m)",
                min_value=0.0,
                max_value=round(max_mrad_um, 2),
                value=round(default_mrad_um, 2),
                step=1.0,
                key=f"p_{k}",
            )
            params_px[k] = val_mrad_um * focal_pitch / 1000.0

        elif isinstance(default_px, int):
            params_px[k] = st.slider(
                k, 1, max(default_px * 3, 20), default_px, key=f"p_{k}",
            )
        elif isinstance(default_px, float):
            params_px[k] = st.slider(
                k, 0.1, max(default_px * 4, 50.0), default_px,
                step=0.5, key=f"p_{k}",
            )

    # Build target
    shape = (n_pad, n_pad)
    _defaults = CGMConfig()
    D_val, theta_val = _defaults.D, _defaults.theta
    center_offset = D_val * np.cos(theta_val) * n_pad / (2 * np.pi)
    center_px = (n_pad - 1) / 2.0
    center = (center_px + center_offset, center_px + center_offset)

    BUILDERS = {
        "lg_mode": lambda s, c, p: lg_mode(s, center=c, **p),
        "square_lattice_vortex": lambda s, c, p: square_lattice_vortex(s, center=c, **p),
        "ring_lattice_vortex": lambda s, c, p: ring_lattice_vortex(s, center=c, **p),
        "graphene_lattice": lambda s, c, p: graphene_lattice(s, center=c, **p),
        "top_hat": lambda s, c, p: top_hat(s, center=c, **p),
        "gaussian_line": lambda s, c, p: gaussian_line(s, center=c, **p),
        "light_sheet": lambda s, c, p: light_sheet(s, center=c, **p),
        "rectangular_grid": lambda s, c, p: rectangular_grid(s, center=c, **p),
    }

    builder = BUILDERS[pat["key"]]
    result = builder(shape, center, params_px)
    target = result[0] if isinstance(result, tuple) else result
    region = measure_region(shape, target, margin=5)

    # Adaptive ROI from pattern params (in pixels)
    roi = _estimate_roi(params_px)
    rs, cs = _roi_slices(center, roi, n_pad)

    # Preview target with physical axes
    ext = _physical_extent(rs, cs, focal_pitch)
    fig_t, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    ax1.imshow(np.abs(target[rs, cs]) ** 2, cmap="hot", extent=ext)
    ax1.set_title("Target Intensity")
    ax1.set_xlabel("x (\u00b5m)")
    ax1.set_ylabel("y (\u00b5m)")
    ax2.imshow(
        _masked_phase(target[rs, cs]), cmap="twilight",
        vmin=-np.pi, vmax=np.pi, extent=ext,
    )
    ax2.set_title("Target Phase")
    ax2.set_xlabel("x (\u00b5m)")
    ax2.set_ylabel("y (\u00b5m)")
    fig_t.tight_layout()
    st.pyplot(fig_t)
    plt.close(fig_t)

# -- Reproduce command (always visible, updates with sliders) ----------------

_sigma_px = sigma_mm / device.pixel_pitch_mm
_gs_snippet = (
    f"gs=gs_seed_phase(a,t,{n_gs_iters}); "
    f"c=replace(c,initial_phase=gs); "
    if n_gs_iters > 0 else ""
)
reproduce_cmd = (
    f"python -c \""
    f"import matplotlib.pyplot as plt; import numpy as np; "
    f"from dataclasses import replace; "
    f"from slm.beams import gaussian_beam; "
    f"from slm.cgm import CGMConfig, cgm; "
    f"from slm.hybrid import gs_seed_phase; "
    f"from slm.propagation import pad_field; "
    f"from slm.targets import measure_region, {pat['key']}; "
    f"a=pad_field(gaussian_beam(({n_slm},{n_slm}),sigma={_sigma_px:.1f},normalize=False),({n_pad},{n_pad})); "
    f"t={pat['key']}(({n_pad},{n_pad}),center=({center[0]:.1f},{center[1]:.1f}),{','.join(f'{k}={v:.4g}' for k,v in params_px.items())}); "
    f"r=measure_region(({n_pad},{n_pad}),t,margin=5); "
    f"c=CGMConfig(max_iterations={max_iters},steepness={steepness},"
    f"R={R_mrad*1e-3},eta_min={eta_min}); "
    f"{_gs_snippet}"
    f"res=cgm(a,t,r,c); "
    f"print(f'1-F={{1-res.final_fidelity:.2e}} eta={{res.final_efficiency:.4f}}'); "
    f"E=res.output_field; ph=np.where(np.abs(E)>0.01*np.max(np.abs(E)),np.angle(E),np.nan); "
    f"fig,ax=plt.subplots(1,4,figsize=(16,3.5)); "
    f"ax[0].imshow(np.abs(t)**2,cmap='hot'); ax[0].set_title('Target'); "
    f"ax[1].imshow(np.abs(E)**2,cmap='hot'); ax[1].set_title('Output Intensity'); "
    f"ax[2].imshow(a**2,cmap='viridis'); ax[2].set_title('Input Beam'); "
    f"ax[3].imshow(ph,cmap='twilight',vmin=-np.pi,vmax=np.pi); ax[3].set_title('Output Phase'); "
    f"fig.colorbar(ax[3].images[0],ax=ax[3],fraction=0.046).set_label('rad'); "
    f"[x.axis('off') for x in ax[:3]]; fig.tight_layout(); "
    f"fig.savefig('hologram_output.png',dpi=150,bbox_inches='tight'); "
    f"print('Saved hologram_output.png'); "
    f"fig2,ax2=plt.subplots(figsize=(6,2.5)); ax2.semilogy(res.cost_history); "
    f"ax2.set(xlabel='Eval',ylabel='Cost',title='Convergence'); fig2.tight_layout(); "
    f"fig2.savefig('hologram_convergence.png',dpi=150,bbox_inches='tight'); "
    f"print('Saved hologram_convergence.png')"
    f"\""
)
st.caption("Reproduce in terminal:")
st.code(reproduce_cmd, language="bash")

# -- Run button ---------------------------------------------------------------

run_clicked = st.button("Run Optimisation", type="primary", use_container_width=True)

if run_clicked:
    input_amp = pad_field(
        gaussian_beam(
            (n_slm, n_slm),
            sigma=sigma_mm / device.pixel_pitch_mm,
            normalize=False,
        ),
        (n_pad, n_pad),
    )
    config = CGMConfig(
        max_iterations=max_iters,
        steepness=steepness,
        R=R_mrad * 1e-3,
        D=D_val,
        theta=theta_val,
        eta_min=eta_min,
    )

    # GS seeding (avoids fidelity wall)
    if n_gs_iters > 0:
        with st.spinner(f"GS seeding ({n_gs_iters} iterations)..."):
            seed_phase = gs_seed_phase(input_amp, target, n_gs_iters)
        config = replace(config, initial_phase=seed_phase)

    with st.spinner(
        f"Running CGM ({max_iters} iterations on {n_pad}\u00d7{n_pad} grid)..."
    ):
        t0 = time.time()
        cgm_result = cgm(input_amp, target, region, config)
        dt = time.time() - t0

    st.session_state["result"] = cgm_result
    st.session_state["dt"] = dt
    st.session_state["target"] = target
    st.session_state["center"] = center
    st.session_state["n_pad"] = n_pad
    st.session_state["roi"] = roi
    st.session_state["focal_pitch"] = focal_pitch
    st.session_state["input_amp"] = input_amp


# -- Results display ----------------------------------------------------------

if "result" in st.session_state:
    cgm_result = st.session_state["result"]
    dt = st.session_state["dt"]
    target_saved = st.session_state["target"]
    center_saved = st.session_state["center"]
    n_pad_saved = st.session_state["n_pad"]
    roi_saved = st.session_state.get("roi", 80)
    focal_pitch_saved = st.session_state.get("focal_pitch", focal_pitch)

    with col_res:
        st.subheader("Results")

        one_minus_F = 1.0 - cgm_result.final_fidelity
        eta_pct = cgm_result.final_efficiency * 100
        eps_phi = cgm_result.final_phase_error * 100
        eps_nu = cgm_result.final_non_uniformity * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("1 - F", f"{one_minus_F:.2e}")
        m2.metric("Efficiency", f"{eta_pct:.1f}%")
        m3.metric("Phase Error", f"{eps_phi:.4f}%")
        m4.metric("Non-uniformity", f"{eps_nu:.4f}%")
        st.caption(f"{cgm_result.n_iterations} L-BFGS-B iterations in {dt:.1f}s")

        # Mosaic with adaptive ROI, physical axes, and masked phase
        E_out = cgm_result.output_field
        rs2, cs2 = _roi_slices(center_saved, roi_saved, n_pad_saved)
        ext2 = _physical_extent(rs2, cs2, focal_pitch_saved)

        # Focal-plane mosaic (target, output intensity, output phase)
        fig = _plot_mosaic(
            target_saved[rs2, cs2], E_out[rs2, cs2], extent=ext2,
        )
        st.pyplot(fig)
        plt.close(fig)

        # SLM plane: input beam + optimized phase side-by-side
        input_amp_saved = st.session_state.get("input_amp")
        slm_phase = cgm_result.slm_phase
        if slm_phase.shape[0] > n_slm:
            y0 = (slm_phase.shape[0] - n_slm) // 2
            x0 = (slm_phase.shape[1] - n_slm) // 2
            slm_phase_crop = slm_phase[y0:y0 + n_slm, x0:x0 + n_slm]
        else:
            slm_phase_crop = slm_phase
        hw = n_slm * device.pixel_pitch_um / 2.0
        slm_ext = [-hw, hw, hw, -hw]

        fig_slm, (ax_beam, ax_ph) = plt.subplots(1, 2, figsize=(10, 4))
        # Input beam intensity
        if input_amp_saved is not None:
            beam_img = np.abs(input_amp_saved) ** 2
            ny, nx = beam_img.shape
            beam_hw = nx * device.pixel_pitch_um / 2.0
            beam_hh = ny * device.pixel_pitch_um / 2.0
            ax_beam.imshow(beam_img, cmap="viridis",
                           extent=[-beam_hw, beam_hw, beam_hh, -beam_hh])
        ax_beam.set_title("Input Beam (SLM plane)")
        ax_beam.set_xlabel("x (\u00b5m)")
        ax_beam.set_ylabel("y (\u00b5m)")
        # SLM phase
        im_slm = ax_ph.imshow(
            np.mod(slm_phase_crop, 2 * np.pi),
            cmap="twilight", vmin=0, vmax=2 * np.pi,
            extent=slm_ext,
        )
        ax_ph.set_title("SLM Phase (experiment)")
        ax_ph.set_xlabel("x (\u00b5m)")
        ax_ph.set_ylabel("y (\u00b5m)")
        cbar_slm = fig_slm.colorbar(im_slm, ax=ax_ph, fraction=0.046, pad=0.04)
        cbar_slm.set_ticks([0, np.pi, 2 * np.pi])
        cbar_slm.set_ticklabels(["0", "\u03c0", "2\u03c0"])
        cbar_slm.set_label("Phase (rad)")
        fig_slm.tight_layout()
        st.pyplot(fig_slm)
        plt.close(fig_slm)

        # Convergence plot
        if cgm_result.cost_history:
            fig_c, ax = plt.subplots(figsize=(6, 2.5))
            ax.semilogy(cgm_result.cost_history)
            ax.set_xlabel("Function evaluation")
            ax.set_ylabel("Cost")
            ax.set_title("Convergence")
            ax.grid(True, alpha=0.3)
            fig_c.tight_layout()
            st.pyplot(fig_c)
            plt.close(fig_c)

        # Downloads
        st.subheader("Download")
        dc1, dc2 = st.columns(2)
        with dc1:
            buf = io.BytesIO()
            np.save(buf, cgm_result.slm_phase)
            st.download_button(
                "Phase pattern (.npy)", buf.getvalue(),
                file_name="slm_phase.npy", mime="application/octet-stream",
            )
        with dc2:
            buf2 = io.BytesIO()
            fig_dl = _plot_mosaic(
                target_saved[rs2, cs2], E_out[rs2, cs2], extent=ext2,
            )
            fig_dl.savefig(buf2, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig_dl)
            st.download_button(
                "Mosaic image (.png)", buf2.getvalue(),
                file_name="hologram_mosaic.png", mime="image/png",
            )

