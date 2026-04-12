import base64
from pathlib import Path

import streamlit as st

from dashboard_utils.theme import apply_theme

st.set_page_config(
    page_title="BYOS: Bring Your Own Sustainability",
    page_icon="🍎",
    layout="wide",
)

apply_theme()

_favicon_path = Path(__file__).parent / "dashboard_utils" / "images" / "apple_favicon.png"
_favicon_b64 = base64.b64encode(_favicon_path.read_bytes()).decode()

st.markdown(f"""
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
""", unsafe_allow_html=True)