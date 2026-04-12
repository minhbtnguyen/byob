import pickle
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import shap
import streamlit as st

_MODEL_DIR = Path(__file__).parent.parent / "models"

_FEATURE_META = {
    "speed_mean": {
        "label": "Speed Mean (km/h)",
        "min": 0.0,
        "max": 120.0,
        "step": 0.5,
        "default": 5.0,
        "help": "Average speed across the 30s window",
    },
    "speed_max": {
        "label": "Speed Max (km/h)",
        "min": 0.0,
        "max": 200.0,
        "step": 1.0,
        "default": 8.0,
        "help": "Peak speed in the window",
    },
    "speed_std": {
        "label": "Speed Std (km/h)",
        "min": 0.0,
        "max": 50.0,
        "step": 0.1,
        "default": 1.5,
        "help": "Speed variability — high for bus (stop/go)",
    },
    "accel_mean": {
        "label": "Accel Mean (km/h/s)",
        "min": 0.0,
        "max": 5.0,
        "step": 0.01,
        "default": 0.05,
        "help": "Average acceleration magnitude",
    },
    "accel_std": {
        "label": "Accel Std (km/h/s)",
        "min": 0.0,
        "max": 5.0,
        "step": 0.01,
        "default": 0.08,
        "help": "Acceleration variability",
    },
    "jerk_mean": {
        "label": "Jerk Mean (km/h/s²)",
        "min": 0.0,
        "max": 2.0,
        "step": 0.01,
        "default": 0.03,
        "help": "Rate of acceleration change — high for car/bus",
    },
    "stop_ratio": {
        "label": "Stop Ratio (0–1)",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "default": 0.25,
        "help": "Fraction of points at speed < 1 km/h",
    },
    "bearing_variance": {
        "label": "Bearing Variance (°²)",
        "min": 0.0,
        "max": 5000.0,
        "step": 10.0,
        "default": 1800.0,
        "help": "Direction change variance — high for walk/bike",
    },
    "distance_total_m": {
        "label": "Distance Total (m)",
        "min": 0.0,
        "max": 2000.0,
        "step": 1.0,
        "default": 37.0,
        "help": "Total metres covered in the 30s window",
    },
}

_MODE_COLORS = {
    "walk": "#4C72B0",
    "bike": "#55A868",
    "bus": "#C44E52",
    "car": "#8172B2",
}

_MODE_ICONS = {"walk": "🚶", "bike": "🚲", "bus": "🚌", "car": "🚗"}


@st.cache_resource(show_spinner=False)
def _load_model():
    path = _MODEL_DIR / "lgbm_mode_classifier.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def _load_explainer(_model):
    return shap.TreeExplainer(_model)


def render() -> None:
    st.markdown("### Interactive Mode Predictor")
    st.sidebar.info(
        "Adjust the sliders to describe a 30-second GPS window and see the predicted transport mode."
    )

    artifact = _load_model()
    if artifact is None:
        st.error(
            "Model not found. Run `eda/modeling.ipynb` to train and save the model."
        )
        return

    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    le = artifact["label_encoder"]
    explainer = _load_explainer(model)

    col_a, col_b = st.columns([1, 1])

    inputs = {}
    with col_a:
        for feat in feature_cols:
            meta = _FEATURE_META[feat]
            inputs[feat] = st.slider(
                meta["label"],
                min_value=meta["min"],
                max_value=meta["max"],
                value=meta["default"],
                step=meta["step"],
                help=meta["help"],
            )

    X = np.array([[inputs[f] for f in feature_cols]])
    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    pred_mode = le.classes_[pred_idx]
    confidence = float(proba[pred_idx])

    # SHAP values for this single prediction
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        sv = shap_vals[0, :, pred_idx]  # (n_features,) for predicted class
    else:
        sv = shap_vals[pred_idx][0]

    with col_b:
        icon = _MODE_ICONS.get(pred_mode, "")
        st.markdown(
            f"<div style='text-align:center; padding:0.6rem 1rem; background:#fff; "
            f"border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,0.07); margin-bottom:0.75rem;'>"
            f"<span style='font-size:1.4rem'>{icon}</span> "
            f"<span style='font-size:1.1rem; font-weight:700; color:#1d1d1f'>{pred_mode.upper()}</span>"
            f"<span style='font-size:0.85rem; color:#6e6e73; margin-left:0.5rem'>{confidence*100:.1f}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        fig_proba = go.Figure(
            go.Bar(
                x=le.classes_.tolist(),
                y=proba.tolist(),
                marker_color=[
                    "#0071e3" if c == pred_mode else _MODE_COLORS.get(c, "#888")
                    for c in le.classes_
                ],
                text=[f"{p*100:.1f}%" for p in proba],
                textposition="outside",
            )
        )
        fig_proba.update_layout(
            yaxis=dict(title="Probability", range=[0, 1.15]),
            height=260,
            margin=dict(t=10, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig_proba, use_container_width=True)

        st.divider()
        # SHAP contribution chart for predicted class
        st.markdown(f"**Feature contributions to {pred_mode}**")
        labels = [_FEATURE_META[f]["label"] for f in feature_cols]
        order = np.argsort(np.abs(sv))
        sorted_labels = [labels[i] for i in order]
        sorted_sv = [float(sv[i]) for i in order]
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in sorted_sv]

        fig_shap = go.Figure(
            go.Bar(
                x=sorted_sv,
                y=sorted_labels,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in sorted_sv],
                textposition="outside",
            )
        )
        fig_shap.update_layout(
            xaxis_title="SHAP value (pushes toward predicted class)",
            height=300,
            margin=dict(t=10, b=30, l=10, r=60),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption("Red = pushes toward prediction · Blue = pushes away")
