from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

_DATASET_ROOT = (
    Path.home()
    / ".cache/kagglehub/datasets/arashnic/microsoft-geolife-gps-trajectory-dataset"
    / "versions/1/Geolife Trajectories 1.3/Data"
)

_TARGET_MODES = ["walk", "bike", "bus", "car", "subway", "taxi"]
_EMISSION_FACTORS = {
    "car": 170,
    "taxi": 170,
    "bus": 89,
    "subway": 41,
    "bike": 0,
    "walk": 0,
}
_MODE_COLORS = {
    "walk": "#4C72B0",
    "bike": "#55A868",
    "bus": "#C44E52",
    "car": "#8172B2",
    "subway": "#CCB974",
    "taxi": "#64B5CD",
}


# -- Parsers -------------------------------------------------------------------


def _load_plt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        skiprows=6,
        header=None,
        names=["lat", "lon", "zero", "altitude_ft", "days", "date", "time"],
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    return df[["datetime", "lat", "lon", "altitude_ft"]].reset_index(drop=True)


def _parse_plt_full(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        skiprows=6,
        header=None,
        names=["lat", "lon", "_zero", "altitude", "days", "date", "time"],
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df["user"] = filepath.parts[-3]
    return df[["user", "lat", "lon", "altitude", "datetime"]]


def _parse_labels(user_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        user_dir / "labels.txt",
        sep="\t",
        skiprows=1,
        header=None,
        names=["start", "end", "mode"],
    )
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df["duration_min"] = (df["end"] - df["start"]).dt.total_seconds() / 60
    df["mode"] = df["mode"].str.lower().str.strip()
    df["user"] = user_dir.name
    return df


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# -- Cached data loaders -------------------------------------------------------


@st.cache_data(show_spinner=False, persist="disk")
def _compute_sampling_gaps(data_root: str) -> np.ndarray:
    root = Path(data_root)
    gaps = []
    for user_dir in sorted(root.iterdir()):
        if not user_dir.is_dir():
            continue
        for f in sorted((user_dir / "Trajectory").glob("*.plt"))[:5]:
            df = _parse_plt_full(f)
            dt = df["datetime"].diff().dt.total_seconds().dropna()
            gaps.extend(dt[(dt > 0) & (dt < 300)].tolist())
    return np.array(gaps)


@st.cache_data(show_spinner=False, persist="disk")
def _compute_label_coverage(data_root: str) -> pd.DataFrame:
    root = Path(data_root)
    labeled = [
        u for u in sorted(root.iterdir()) if u.is_dir() and (u / "labels.txt").exists()
    ]
    rows = []
    for user_dir in labeled:
        labels = _parse_labels(user_dir)
        labeled_s = (labels["end"] - labels["start"]).dt.total_seconds().sum()
        all_times = []
        for f in sorted((user_dir / "Trajectory").glob("*.plt")):
            all_times.extend(_parse_plt_full(f)["datetime"].tolist())
        if not all_times:
            continue
        total_s = (max(all_times) - min(all_times)).total_seconds()
        rows.append(
            {
                "user": user_dir.name,
                "pct_labeled": 100 * labeled_s / total_s if total_s > 0 else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("pct_labeled")


@st.cache_data(show_spinner=False, persist="disk")
def _compute_speed_anomalies(data_root: str) -> pd.DataFrame:
    root = Path(data_root)
    rows = []
    for user_dir in sorted(root.iterdir()):
        if not user_dir.is_dir():
            continue
        for f in sorted((user_dir / "Trajectory").glob("*.plt"))[:10]:
            df = _parse_plt_full(f).reset_index(drop=True)
            if len(df) < 2:
                continue
            dist = _haversine_m(
                df["lat"].values[:-1],
                df["lon"].values[:-1],
                df["lat"].values[1:],
                df["lon"].values[1:],
            )
            dt_s = df["datetime"].diff().dt.total_seconds().values[1:]
            with np.errstate(divide="ignore", invalid="ignore"):
                speed_kmh = np.where(dt_s > 0, (dist / dt_s) * 3.6, 0)
            n = int((speed_kmh > 250).sum())
            if n:
                rows.append({"user": user_dir.name, "file": f.name, "anomalies": n})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, persist="disk")
def _load_all_labels(data_root: str) -> pd.DataFrame:
    root = Path(data_root)
    labeled = [
        u for u in sorted(root.iterdir()) if u.is_dir() and (u / "labels.txt").exists()
    ]
    return pd.concat([_parse_labels(u) for u in labeled], ignore_index=True)


@st.cache_data(show_spinner=False, persist="disk")
def _compute_speed_by_mode(data_root: str) -> dict:
    root = Path(data_root)
    labeled = [
        u for u in sorted(root.iterdir()) if u.is_dir() and (u / "labels.txt").exists()
    ]
    speed_by_mode: dict = {m: [] for m in _TARGET_MODES}
    for user_dir in labeled:
        labels = _parse_labels(user_dir)
        labels = labels[labels["mode"].isin(_TARGET_MODES)]
        if labels.empty:
            continue
        plts = sorted((user_dir / "Trajectory").glob("*.plt"))
        traj = pd.concat(
            [_parse_plt_full(f) for f in plts], ignore_index=True
        ).sort_values("datetime")
        dist = np.zeros(len(traj))
        dist[1:] = _haversine_m(
            traj["lat"].values[:-1],
            traj["lon"].values[:-1],
            traj["lat"].values[1:],
            traj["lon"].values[1:],
        )
        dt_s = traj["datetime"].diff().dt.total_seconds().fillna(0).values
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = np.where(dt_s > 0, (dist / dt_s) * 3.6, 0)
        traj = traj.copy()
        traj["speed_kmh"] = speed
        traj = traj[traj["speed_kmh"] < 250]
        for _, row in labels.iterrows():
            mask = (traj["datetime"] >= row["start"]) & (traj["datetime"] <= row["end"])
            pts = traj.loc[mask, "speed_kmh"]
            if len(pts) > 2:
                speed_by_mode[row["mode"]].extend(pts.tolist())
    return speed_by_mode


@st.cache_data(show_spinner=False, persist="disk")
def _compute_emissions(data_root: str) -> pd.DataFrame:
    speed_by_mode = _compute_speed_by_mode(data_root)
    all_labels = _load_all_labels(data_root)

    def est_dist(mode, duration_min):
        speeds = speed_by_mode.get(mode, [])
        spd = float(np.median(speeds)) if speeds else 30.0
        return (duration_min / 60) * spd

    rows = []
    for _, row in all_labels[all_labels["mode"].isin(_EMISSION_FACTORS)].iterrows():
        dist = est_dist(row["mode"], row["duration_min"])
        rows.append(
            {
                "user": row["user"],
                "mode": row["mode"],
                "dist_km": dist,
                "co2_g": dist * _EMISSION_FACTORS[row["mode"]],
            }
        )
    return pd.DataFrame(rows)


# -- Section renderers ---------------------------------------------------------


def _render_health(data_root: str) -> None:
    st.markdown("#### Label Coverage per Labeled Users")

    with st.spinner("Computing label coverage..."):
        cov_df = _compute_label_coverage(data_root)

    median_cov = float(cov_df["pct_labeled"].median())
    fig = go.Figure(
        go.Bar(
            x=cov_df["user"].tolist(),
            y=cov_df["pct_labeled"].tolist(),
            marker_color="#0071e3",
        )
    )
    fig.add_hline(
        y=median_cov,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Median {median_cov:.1f}%",
    )
    fig.update_layout(
        xaxis_title="User",
        yaxis_title="% GPS time labeled",
        height=350,
        margin=dict(t=20, b=60),
        xaxis=dict(
            categoryorder="total ascending",
            tickangle=90,
            tickfont=dict(size=7),
        ),
    )

    col_chart, col_text = st.columns([3, 1])
    with col_chart:
        st.plotly_chart(fig, use_container_width=True)
    with col_text:
        st.info(
            "Label coverage is typically low in this dataset — many users labeled only specific "
            "trips rather than their full recording history. Expect median coverage around 10–30%. "
            "Users with very low coverage (<5%) should be excluded from supervised training — "
            "they contribute too few labeled windows to learn reliable patterns and will silently "
            "degrade the training set."
        )


def _render_mode_distribution(data_root: str) -> None:
    with st.spinner("Loading labels..."):
        all_labels = _load_all_labels(data_root)

    mc = all_labels["mode"].value_counts().reset_index()
    mc.columns = ["mode", "count"]
    mc = mc.sort_values("count", ascending=True)

    md = (
        all_labels.groupby("mode")["duration_min"]
        .sum()
        .reset_index()
        .sort_values("duration_min", ascending=True)
    )
    md["hours"] = md["duration_min"] / 60

    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure(
            go.Bar(
                y=mc["mode"].tolist(),
                x=mc["count"].tolist(),
                orientation="h",
                marker_color="#0071e3",
            )
        )
        fig.update_layout(
            title="Trip Count by Mode",
            xaxis_title="Segments",
            height=300,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = go.Figure(
            go.Bar(
                y=md["mode"].tolist(),
                x=md["hours"].tolist(),
                orientation="h",
                marker_color="#ff6b35",
            )
        )
        fig2.update_layout(
            title="Total Duration by Mode",
            xaxis_title="Hours",
            height=300,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    um = all_labels.groupby("user")["mode"].nunique().reset_index().sort_values("mode")
    fig3 = go.Figure(
        go.Bar(
            x=um["user"].tolist(),
            y=um["mode"].tolist(),
            marker_color="#0071e3",
        )
    )
    fig3.update_layout(
        title="Mode Diversity per Labeled User",
        xaxis_title="User",
        yaxis_title="Distinct modes",
        height=300,
        margin=dict(t=40, b=60),
        xaxis_tickangle=90,
        xaxis_tickfont_size=7,
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        f"Median distinct modes per user: **{int(um['mode'].median())}** | "
        f"Max: **{int(um['mode'].max())}**"
    )


def _render_speed_profiles(data_root: str) -> None:
    with st.spinner("Computing speed profiles — may take a minute on first run..."):
        sbm = _compute_speed_by_mode(data_root)

    colors = [_MODE_COLORS[m] for m in _TARGET_MODES]
    fig = make_subplots(rows=2, cols=3, subplot_titles=list(_TARGET_MODES))
    for i, mode in enumerate(_TARGET_MODES):
        r, c = divmod(i, 3)
        data = np.array(sbm[mode])
        if len(data) == 0:
            continue
        clipped = data[(data >= 0) & (data <= 120)]
        med = float(np.median(data))
        fig.add_trace(
            go.Histogram(
                x=clipped.tolist(),
                nbinsx=50,
                marker_color=colors[i],
                marker_line_color="white",
                marker_line_width=0.3,
                histnorm="probability density",
                name=mode,
                showlegend=False,
            ),
            row=r + 1,
            col=c + 1,
        )
        fig.add_vline(
            x=med,
            line_dash="dash",
            line_color="black",
            line_width=1,
            row=r + 1,
            col=c + 1,
        )
        fig.layout.annotations[i].text = f"{mode}  (n={len(data):,})  med={med:.1f}"
    fig.update_layout(
        title="Speed Distribution by Transport Mode",
        height=500,
        margin=dict(t=60, b=40),
    )
    fig.update_xaxes(title_text="Speed (km/h)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    stop_rows = []
    for mode in _TARGET_MODES:
        speeds = sbm.get(mode, [])
        if speeds:
            arr = np.array(speeds)
            stop_rows.append(
                {"mode": mode, "pct_stopped": float((arr < 1.0).mean() * 100)}
            )
    stop_df = pd.DataFrame(stop_rows).sort_values("pct_stopped", ascending=False)
    fig2 = go.Figure(
        go.Bar(
            x=stop_df["mode"].tolist(),
            y=stop_df["pct_stopped"].tolist(),
            marker_color="#ff6b35",
        )
    )
    fig2.update_layout(
        title="Stop Density by Mode (% points at speed < 1 km/h)",
        yaxis_title="% stopped",
        height=300,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)


def _render_temporal(data_root: str) -> None:
    with st.spinner("Loading labels..."):
        all_labels = _load_all_labels(data_root)
        sbm = _compute_speed_by_mode(data_root)

    al = all_labels.copy()
    al["hour"] = al["start"].dt.hour

    fig = go.Figure()
    for mode in _TARGET_MODES:
        subset = al[al["mode"] == mode]
        if subset.empty:
            continue
        hourly = subset.groupby("hour").size().reindex(range(24), fill_value=0)
        fig.add_trace(
            go.Scatter(
                x=list(hourly.index),
                y=hourly.values.tolist(),
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(color=_MODE_COLORS.get(mode, "#888")),
                name=mode,
            )
        )
    fig.update_layout(
        title="Trip Start Time by Mode",
        xaxis_title="Hour of day",
        yaxis_title="Trip count",
        xaxis=dict(tickmode="linear", dtick=2),
        height=350,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    car_speeds = sbm.get("car", [])
    car_med_kmh = float(np.median(car_speeds)) if car_speeds else 30.0
    car_trips = al[al["mode"] == "car"].copy()
    car_trips["est_dist_km"] = (car_trips["duration_min"] / 60) * car_med_kmh
    sub3 = car_trips[car_trips["est_dist_km"] < 3]
    pct = 100 * len(sub3) / len(car_trips) if len(car_trips) else 0

    fig2 = go.Figure(
        go.Histogram(
            x=car_trips["est_dist_km"].clip(upper=50).tolist(),
            nbinsx=40,
            marker_color="#0071e3",
            marker_line_color="white",
            marker_line_width=0.5,
        )
    )
    fig2.add_vline(
        x=3,
        line_dash="dash",
        line_color="red",
        annotation_text=f"3 km ({pct:.1f}% of trips)",
    )
    fig2.update_layout(
        title="Car Trip Distance Distribution",
        xaxis_title="Estimated trip distance (km)",
        yaxis_title="Trip count",
        height=300,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        f"Car trips under 3 km: **{len(sub3):,} / {len(car_trips):,}** "
        f"({pct:.1f}%) — potentially bike-replaceable"
    )


def _render_emissions(data_root: str) -> None:
    with st.spinner("Computing emissions — may take a minute on first run..."):
        em_df = _compute_emissions(data_root)

    mode_co2 = (
        em_df.groupby("mode")["co2_g"]
        .sum()
        .reset_index()
        .sort_values("co2_g", ascending=True)
    )
    mode_co2["co2_kg"] = mode_co2["co2_g"] / 1000

    user_co2 = (
        em_df.groupby("user")["co2_g"]
        .sum()
        .reset_index()
        .sort_values("co2_g", ascending=True)
        .tail(20)
    )
    user_co2["co2_kg"] = user_co2["co2_g"] / 1000

    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure(
            go.Bar(
                y=mode_co2["mode"].tolist(),
                x=mode_co2["co2_kg"].tolist(),
                orientation="h",
                marker_color="#ff6b35",
            )
        )
        fig.update_layout(
            title="Total CO\u2082 by Mode (all users)",
            xaxis_title="Total CO\u2082 (kg)",
            height=300,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = go.Figure(
            go.Bar(
                y=user_co2["user"].tolist(),
                x=user_co2["co2_kg"].tolist(),
                orientation="h",
                marker_color="#0071e3",
            )
        )
        fig2.update_layout(
            title="Per-User CO\u2082 Footprint (top 20)",
            xaxis_title="Total CO\u2082 (kg)",
            height=300,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    car_em = em_df[em_df["mode"] == "car"].copy()
    sub3_em = car_em[car_em["dist_km"] < 3]
    saved_kg = float(sub3_em["co2_g"].sum() / 1000)
    total_kg = float(car_em["co2_g"].sum() / 1000)
    pct_saved = 100 * saved_kg / total_kg if total_kg else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total car CO\u2082", f"{total_kg:.1f} kg")
    c2.metric("Savings if sub-3 km \u2192 bike", f"{saved_kg:.1f} kg")
    c3.metric("% of car emissions saved", f"{pct_saved:.1f}%")


# -- Main render ---------------------------------------------------------------


def _render_map():
    st.markdown("#### Raw Trajectory Viewer")
    users = sorted([p.name for p in _DATASET_ROOT.iterdir() if p.is_dir()])
    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        selected_user = st.selectbox("Select user", users, index=0)
    plt_files = sorted((_DATASET_ROOT / selected_user / "Trajectory").glob("*.plt"))
    if not plt_files:
        st.warning("No trajectory files found for this user.")
    else:
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
                    center=dict(
                        lat=float(df["lat"].mean()), lon=float(df["lon"].mean())
                    ),
                    zoom=13,
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                height=420,
                showlegend=False,
            )
            st.plotly_chart(fig, config={"displayModeBar": False})
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


def render() -> None:
    st.markdown(
        "### Dataset [Microsoft Geolife GPS Trajectories](https://www.kaggle.com/datasets/arashnic/microsoft-geolife-gps-trajectory-dataset/data)"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Users", "182")
    col2.metric("Trajectories", "17,621")
    col3.metric("Total Distance", "1.29M km")
    col4.metric("Total Duration", "50,176 hrs")

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

    if not _DATASET_ROOT.exists():
        st.warning("Dataset not found. Run the kagglehub download cell first.")
        return

    _root_str = str(_DATASET_ROOT)

    # -- Raw trajectory viewer -------------------------------------------------
    _render_map()

    st.markdown("---")
    _render_health(_root_str)

    st.markdown("---")
    _render_mode_distribution(_root_str)

    st.markdown("---")

    st.markdown("---")

    # with st.expander("⚡ 3. Speed & Motion Profiles by Mode"):
    #     _render_speed_profiles(_root_str)

    # with st.expander("🕐 4. Temporal Patterns — Time of Day & Short Car Trips"):
    #     _render_temporal(_root_str)

    # with st.expander("🌿 5. Emissions Preview — CO₂ by Mode & Counterfactual Savings"):
    #     _render_emissions(_root_str)
