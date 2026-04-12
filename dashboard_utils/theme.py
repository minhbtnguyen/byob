import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
<style>
  /* ── Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }

  /* ── Page background — warm paper ── */
  .stApp {
    background-color: #f5f5f7;
    color: #1d1d1f;
  }

  /* ── Main content container ── */
  .block-container {
    max-width: 1100px;
    padding: 3rem 2rem 4rem;
    margin: 0 auto;
  }

  /* ── Headings ── */
  h1 {
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    color: #1d1d1f !important;
    margin-bottom: 0.2rem !important;
  }

  h2 {
    font-size: 1.6rem !important;
    font-weight: 500 !important;
    color: #6e6e73 !important;
    letter-spacing: -0.01em !important;
  }

  h3 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #1d1d1f !important;
  }

  /* ── Cards ── */
  .card {
    background: #ffffff;
    border-radius: 18px;
    padding: 2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
  }

  /* ── Divider ── */
  hr {
    border: none;
    border-top: 1px solid #e0e0e5;
    margin: 2rem 0;
  }

  /* ── Metric labels ── */
  [data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #6e6e73 !important;
  }

  [data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 600 !important;
    color: #1d1d1f !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background-color: #fbfbfd;
    border-right: 1px solid #e0e0e5;
  }

  /* ── Hide sidebar collapse/expand buttons ── */
  [data-testid="stSidebarCollapseButton"],
  [data-testid="stSidebarCollapsedControl"] {
    display: none !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background-color: #0071e3;
    color: white;
    border: none;
    border-radius: 980px;
    padding: 0.5rem 1.4rem;
    font-size: 0.9rem;
    font-weight: 500;
    letter-spacing: -0.01em;
    transition: background 0.2s ease;
  }

  .stButton > button:hover {
    background-color: #0077ed;
  }

  /* ── Select boxes & inputs ── */
  .stSelectbox > div > div,
  .stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 1px solid #d2d2d7 !important;
    background: #fff !important;
    color: #1d1d1f !important;
  }

  /* ── Dataframe text ── */
  [data-testid="stDataFrame"] * {
    color: #1d1d1f !important;
  }

  /* ── Make header transparent, keep sidebar toggle visible ── */
  header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.5rem !important;
  }

  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stStatusWidget"] {
    display: none !important;
  }

  /* ── Remove top padding left by hidden header ── */
  .block-container {
    padding-top: 0 !important;
  }



  /* ── Hide map attribution ── */
  .maplibregl-ctrl-attrib,
  .maplibregl-ctrl-bottom-left,
  .maplibregl-ctrl-bottom-right {
    display: none !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #e0e0e5;
    background: transparent;
  }

  .stTabs [data-baseweb="tab"] {
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 0.875rem;
    font-weight: 500;
    color: #6e6e73;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 0.6rem 1.2rem;
    margin-bottom: -1px;
    letter-spacing: -0.01em;
    transition: color 0.15s ease;
  }

  .stTabs [data-baseweb="tab"]:hover {
    color: #1d1d1f;
    background: transparent;
  }

  .stTabs [aria-selected="true"] {
    color: #1d1d1f !important;
    border-bottom: 2px solid #1d1d1f !important;
    background: transparent !important;
  }

  .stTabs [data-baseweb="tab-highlight"] {
    display: none;
  }

  .stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
    background: transparent;
  }

  /* ── Expanders ── */
  [data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e0e0e5 !important;
    border-radius: 12px !important;
  }

  [data-testid="stExpander"] summary {
    background: #ffffff !important;
    color: #1d1d1f !important;
  }

  [data-testid="stExpander"] summary:hover {
    background: #f5f5f7 !important;
  }

  [data-testid="stExpander"] > div > div {
    background: #ffffff !important;
    color: #1d1d1f !important;
  }
</style>
""",
        unsafe_allow_html=True,
    )
