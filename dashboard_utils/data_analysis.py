import glob
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_DATASET_ROOT = (
    Path.home()
    / ".cache/kagglehub/datasets/arashnic/microsoft-geolife-gps-trajectory-dataset"
    / "versions/1/Geolife Trajectories 1.3/Data"
)


def _load_plt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        skiprows=6,
        header=None,
        names=["lat", "lon", "zero", "altitude_ft", "days", "date", "time"],
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    return df[["datetime", "lat", "lon", "altitude_ft"]].reset_index(drop=True)


def render() -> None:
    st.markdown(
        "### Dataset [Microsoft Geolife GPS Trajectories](https://www.kaggle.com/datasets/arashnic/microsoft-geolife-gps-trajectory-dataset/data)"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Users", "182")
    col2.metric("Trajectories", "17,621")
    col3.metric("Total Distance", "1.29M km")
    col4.metric("Total Duration", "50,176 hrs")

    st.markdown("---")

    st.markdown(
        "Collected by Microsoft Research Asia from **April 2007 to August 2012**. "
        "Each trajectory is a sequence of time-stamped GPS points (latitude, longitude, altitude). "
        "91.5% of trajectories are logged at high density - every 1-5 seconds or every 5-10 meters."
    )

    st.markdown(
        "The dataset captures a broad range of outdoor movements: daily commutes, shopping, sightseeing, "
        "hiking, and cycling. Data spans 30+ cities in China plus some in the USA and Europe, "
        "though the majority originates from **Beijing**."
    )

    st.markdown("---")
    st.markdown("#### Sample Trajectory")

    if not _DATASET_ROOT.exists():
        st.warning("Dataset not found. Run the kagglehub download cell first.")
        return

    users = sorted([p.name for p in _DATASET_ROOT.iterdir() if p.is_dir()])

    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        selected_user = st.selectbox("Select user", users, index=0)

    plt_files = sorted((_DATASET_ROOT / selected_user / "Trajectory").glob("*.plt"))
    if not plt_files:
        st.warning("No trajectory files found for this user.")
        return

    with sel_col2:
        selected_file = st.selectbox(
            "Select trajectory", [f.name for f in plt_files], index=0
        )
    df = _load_plt(_DATASET_ROOT / selected_user / "Trajectory" / selected_file)

    col_map, col_table = st.columns([3, 2])

    with col_map:
        fig = go.Figure(
            go.Scattermap(
                lat=df["lat"].tolist(),
                lon=df["lon"].tolist(),
                mode="lines+markers",
                line=dict(width=3, color="#0071e3"),
                marker=dict(size=4, color="#0071e3"),
                hovertext=df["datetime"].astype(str).tolist(),
            )
        )
        fig.update_layout(
            map=dict(
                style="open-street-map",
                center=dict(lat=float(df["lat"].mean()), lon=float(df["lon"].mean())),
                zoom=13,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=420,
            showlegend=False,
        )
        st.plotly_chart(fig)

    with col_table:
        st.dataframe(
            df.head(50).rename(
                columns={
                    "datetime": "Datetime",
                    "lat": "Latitude",
                    "lon": "Longitude",
                    "altitude_ft": "Altitude (ft)",
                }
            ),
            height=420,
        )
