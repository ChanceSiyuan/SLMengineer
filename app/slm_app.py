"""SLM Hologram Generator — Interactive Web UI.

Launch with:  streamlit run app/slm_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="SLM Hologram Generator",
    page_icon="🔬",
    layout="wide",
)

st.title("SLM Hologram Generator")
st.markdown(
    "Generate phase-only holograms for optical tweezer arrays and beam shaping. "
    "Select a page from the sidebar to get started."
)

st.sidebar.success("Select a page above.")

st.markdown("### Pages")
st.markdown("""
- **Hologram Generator** — Choose a target pattern, configure the optimizer, run CGM, and download the phase mask.
- **Feedback Loop** — Run closed-loop intensity correction with a simulated camera.
- **Zernike Explorer** — Interactively visualise Zernike aberration modes and their effect on the focal plane.
""")
