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

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Analysis", expanded=True):
        st.write(
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

    fig = go.Figure(
        go.Bar(
            y=mc["mode"].tolist(),
            x=mc["count"].tolist(),
            orientation="h",
            marker_color="#0071e3",
        )
    )
    fig.update_layout(
        xaxis_title="Segments",
        height=300,
        margin=dict(t=20, b=40),
    )
    st.markdown("### Trip Count by Mode")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Analysis", expanded=True):
        st.write(
            "Walk and bike tend to dominate trip count due to short, frequent trips. "
            "If we train on trip count, walk and bike are overrepresented relative to their "
            "real-world impact. Modes with fewer than ~200 trip segments should be considered "
            "for grouping or dropping — e.g. merge taxi into car, merge all rail into a single "
            "'rail' class — to avoid producing a classifier that technically learns rare labels "
            "from only a handful of examples."
        )

    st.divider()
    fig2 = go.Figure(
        go.Bar(
            y=md["mode"].tolist(),
            x=md["hours"].tolist(),
            orientation="h",
            marker_color="#ff6b35",
        )
    )
    fig2.update_layout(
        xaxis_title="Hours",
        height=300,
        margin=dict(t=20, b=40),
    )
    st.markdown("### Total Duration by Mode")
    st.plotly_chart(fig2, use_container_width=True)
    with st.expander("Analysis", expanded=True):
        st.write(
            "Walk and bike dominate total duration as well as trip count, which is somewhat "
            "surprising — it means participants in this dataset accumulated thousands of hours "
            "on foot or by bicycle, not just short frequent trips. This is consistent with the "
            "Beijing urban context where walking and cycling are primary modes for short-haul "
            "movement. The implication for modeling is that duration-based weighting does not "
            "rescue minority classes like subway or taxi; class rebalancing will be needed "
            "regardless of whether we weight by trip count or total time."
        )

    st.divider()

    um = all_labels.groupby("user")["mode"].nunique().reset_index().sort_values("mode")
    fig3 = go.Figure(
        go.Bar(
            x=um["user"].tolist(),
            y=um["mode"].tolist(),
            marker_color="#0071e3",
        )
    )
    fig3.update_layout(
        xaxis_title="User",
        yaxis_title="Distinct modes",
        height=300,
        margin=dict(t=20, b=60),
        xaxis_tickangle=90,
        xaxis_tickfont_size=7,
    )
    st.markdown("### Mode Diversity per Labeled User")
    st.plotly_chart(fig3, use_container_width=True)
    with st.expander("Analysis", expanded=True):
        st.write(
            "If most users only have 2–3 distinct modes, subject-independent cross-validation "
            "becomes tricky — holding out a user leaves a gap in class coverage for that fold. "
            "Users with only 1 mode contribute nothing to multi-class learning and should be "
            "excluded from training; they inflate accuracy on their dominant class without "
            "teaching the model anything about the others. Users with 4+ modes are the most "
            "valuable training examples and should be weighted more heavily when constructing "
            "train/validation splits."
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
        height=500,
        margin=dict(t=60, b=40),
    )
    fig.update_xaxes(title_text="Speed (km/h)")
    st.markdown("### Speed Distribution by Transport Mode")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Analysis", expanded=True):
        st.write(
            "We expect clearly separated medians: walk ~5 km/h, bike ~12 km/h, bus/car ~25–40 km/h, "
            "subway ~35–50 km/h. The hard classification problems are:"
        )
        st.markdown(
            "- **Bus vs car**: similar speed ranges with substantial overlap — stop density and "
            "bearing variance will be the separating features\n"
            "- **Subway vs car/bus**: similar speeds but subway has no traffic-light stops — "
            "stop density helps here\n"
            "- **Taxi vs car**: nearly identical — likely not separable from speed alone"
        )
        st.write(
            "If walk and car distributions overlap significantly, it indicates label noise in the dataset."
        )


def _render_temporal(data_root: str) -> None:
    with st.spinner("Loading labels..."):
        all_labels = _load_all_labels(data_root)

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
        xaxis_title="Hour of day",
        yaxis_title="Trip count",
        xaxis=dict(tickmode="linear", dtick=2),
        height=350,
        margin=dict(t=20, b=40),
    )
    st.markdown("### Trip Start Time by Mode")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Analysis", expanded=True):
        st.write(
            "We expect morning (8–9am) and evening (5–7pm) peaks for car, bus, and subway — typical "
            "commute patterns. Walk and bike are likely more distributed throughout the day. A flat "
            "temporal distribution across all modes would suggest the Geolife users were not typical "
            "commuters, which is plausible — many were university researchers. If commute peaks are "
            "visible, this strengthens the counterfactual story: the highest-emissions car trips are "
            "concentrated in predictable windows, making them ideal targets for behavior-change "
            "recommendations."
        )


def _render_emissions(data_root: str) -> None:
    with st.spinner("Computing emissions — may take a minute on first run..."):
        em_df = _compute_emissions(data_root)

    car_em = em_df[em_df["mode"] == "car"].copy()
    sub3_em = car_em[car_em["dist_km"] < 3]
    saved_kg = float(sub3_em["co2_g"].sum() / 1000)
    total_kg = float(car_em["co2_g"].sum() / 1000)
    pct_saved = 100 * saved_kg / total_kg if total_kg else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total car CO\u2082", f"{total_kg:.1f} kg")
    c2.metric("Savings if sub-3 km \u2192 bike", f"{saved_kg:.1f} kg")
    c3.metric("% of car emissions saved", f"{pct_saved:.1f}%")

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

    fig = go.Figure(
        go.Bar(
            y=mode_co2["mode"].tolist(),
            x=mode_co2["co2_kg"].tolist(),
            orientation="h",
            marker_color="#ff6b35",
        )
    )
    fig.update_layout(
        xaxis_title="Total CO\u2082 (kg)",
        height=300,
        margin=dict(t=20, b=40),
    )
    st.markdown("### Total CO\u2082 by Mode (all users)")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Analysis", expanded=True):
        st.write(
            "Car will dominate total CO\u2082 despite not necessarily having the most trips — its "
            "emission factor is 2\u20134\u00d7 higher than bus/subway per km. The counterfactual saving "
            "percentage is the number to watch. If replacing sub-3 km car trips saves >20% of total "
            "car emissions, the Green Commute Coach pitch is very strong. If it's <5%, the intervention "
            "is marginal and the product story needs reframing toward longer-trip alternatives "
            "(e.g. car-to-bus substitution)."
        )
        st.write(
            "**Important caveat:** Distance here is estimated from duration \u00d7 median speed, not "
            "computed from GPS coordinates. These numbers are directionally correct but not precise. "
            "The full pipeline will compute actual haversine distances along each trajectory."
        )

    st.divider()

    fig2 = go.Figure(
        go.Bar(
            y=user_co2["user"].tolist(),
            x=user_co2["co2_kg"].tolist(),
            orientation="h",
            marker_color="#0071e3",
        )
    )
    fig2.update_layout(
        xaxis_title="Total CO\u2082 (kg)",
        height=300,
        margin=dict(t=20, b=40),
    )
    st.markdown("### Per-User CO\u2082 Footprint (top 20)")
    st.plotly_chart(fig2, use_container_width=True)
    with st.expander("Analysis", expanded=True):
        st.write(
            "The per-user chart will show high variance: a few heavy car users will account for a "
            "disproportionate share of total emissions, which is the classic Pareto pattern seen in "
            "real mobility data. These high-emitters are the primary targets for the Green Commute "
            "Coach — a small behavior change in this group yields outsized aggregate impact."
        )


# -- Main render ---------------------------------------------------------------


def render() -> None:
    with st.sidebar.container(border=True):
        st.markdown(
            "[Microsoft Geolife GPS Trajectories Dataset](https://www.kaggle.com/datasets/arashnic/microsoft-geolife-gps-trajectory-dataset/data)"
        )
        st.caption(
            "Microsoft Research Asia · Apr 2007 - Aug 2012 · "
            "GPS points every 1-5 s · 30+ cities, mostly Beijing."
        )
        st.markdown("**Number of Users:** 182")
        st.markdown("**Number of Trajectories:** 17,621")
        st.markdown("**Total Distance:** 1.29M km")
        st.markdown("**Total Duration:** 50,176 hrs")

    if not _DATASET_ROOT.exists():
        st.warning("Dataset not found. Run the kagglehub download cell first.")
        return

    _root_str = str(_DATASET_ROOT)

    # -- Raw trajectory viewer -------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Raw Trajectory Viewer",
            "User Analysis",
            "Commute Mode Analysis",
            "Motion Analysis",
            "Temporal Analysis",
            "Emission Review",
        ]
    )

    with tab1:
        _render_map()

    with tab2:
        _render_health(_root_str)

    with tab3:
        _render_mode_distribution(_root_str)

    with tab4:
        _render_speed_profiles(_root_str)

    with tab5:
        _render_temporal(_root_str)

    with tab6:
        _render_emissions(_root_str)
