import base64
from pathlib import Path

import streamlit as st

from dashboard_utils import (
    agentic_eval,
    agentic_simulation,
    data_analysis,
    hypothesis,
    modeling,
    products,
    references,
)
from dashboard_utils.theme import apply_theme

st.set_page_config(
    page_title="BYOB: Bring Your Own Bike",
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
<div style="text-align:center; padding-top: 2rem;">
  <img src="data:image/png;base64,{_favicon_b64}" width="22" style="margin-bottom:1.2rem;" />
  <h1 style="font-size:2.8rem; font-weight:700; letter-spacing:-0.03em; color:#1d1d1f; margin-bottom:0.2rem;">BYOB: Bring Your Own Bike</h1>
  <p style="font-size:1.2rem; font-weight:400; color:#6e6e73; margin-top:0;">Understanding How Apple ESG Features Could Reshape How We Move Using AI Agents</p>
</div>
<hr style="border:none; border-top:1px solid #e0e0e5; margin:2rem 0;">
""",
    unsafe_allow_html=True,
)

with st.sidebar.container(border=True):
    st.markdown(
        "**Researched & Engineered by** [Minh Nguyen](https://minhbtnguyen.com/)"
    )

page = st.sidebar.selectbox(
    "",
    [
        "Hypothesis",
        "Exploratory Data Analysis (EDA)",
        "Commute Mode Predictor",
        "Agentic Evaluation vs Human Evaluation",
        "Agentic Behaviour Simulation",
        "Proposed Apple Products",
        "References",
    ],
)

st.sidebar.divider()

if page == "Hypothesis":
    hypothesis.render()
elif page == "Exploratory Data Analysis (EDA)":
    data_analysis.render()
elif page == "Commute Mode Predictor":
    modeling.render()
elif page == "Agentic Evaluation vs Human Evaluation":
    agentic_eval.render()
elif page == "Agentic Behaviour Simulation":
    agentic_simulation.render()
elif page == "Proposed Apple Products":
    products.render()
elif page == "References":
    references.render()
