import base64
from pathlib import Path

import streamlit as st

from dashboard_utils import data_analysis, hypothesis, references
from dashboard_utils.theme import apply_theme

st.set_page_config(
    page_title="BYOS: Bring Your Own Sustainability",
    page_icon="🍎",
    layout="wide",
)

apply_theme()

_favicon_path = (
    Path(__file__).parent / "dashboard_utils" / "images" / "apple_favicon.png"
)
_favicon_b64 = base64.b64encode(_favicon_path.read_bytes()).decode()

st.markdown(
    f"""
<nav class="byos-navbar">
  <img src="data:image/png;base64,{_favicon_b64}" width="22" />
</nav>
<div class="byos-page-content">
  <div style="text-align:center; padding-top: 2.5rem;">
    <h1 style="font-size:2.8rem; font-weight:700; letter-spacing:-0.03em; color:#1d1d1f; margin-bottom:0.2rem;">BYOS: Bring Your Own Sustainability</h1>
    <p style="font-size:1.2rem; font-weight:400; color:#6e6e73; margin-top:0;">An Agentic Mobility Pipeline for Personal Carbon Insights</p>
  </div>
  <hr style="border:none; border-top:1px solid #e0e0e5; margin:2rem 0;">
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Hypothesis",
        "Data & Analysis",
        "Modeling",
        "Agentic Rating",
        "Agentic Simulation",
        "Potential Products",
        "References",
    ]
)

with tab1:
    hypothesis.render()

with tab2:
    data_analysis.render()

with tab3:
    st.header("Modeling")

with tab4:
    st.header("Agentic Rating")

with tab5:
    st.header("Agentic Simulation")

with tab6:
    st.header("Potential Products")

with tab7:
    references.render()
