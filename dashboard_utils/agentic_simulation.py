import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

_REPORTS_DIR = Path(__file__).parent.parent / "reports"

_MODE_COLOR = {
    "car": "#e74c3c",
    "transit": "#f0a500",
    "bike": "#2ecc71",
    "walk": "#3498db",
}
_MODE_ICON = {"car": "🚗", "transit": "🚌", "bike": "🚲", "walk": "🚶"}
_MOOD_COLOR = {
    "stressed": "#e74c3c",
    "neutral": "#adb5bd",
    "content": "#74c69d",
    "energized": "#0071e3",
}
_MOOD_ICON = {"stressed": "😟", "neutral": "😐", "content": "😊", "energized": "⚡"}

SUSTAINABLE_MODES = {"transit", "bike", "walk"}


def _strength_label(s) -> str:
    """Return display label for a strength value (accepts float or string)."""
    v = float(s)
    if v == 0.0:
        return "Level 0 — Off (no feature)"
    if v <= 0.25:
        return "Level 1 — Subtle (quiet Maps suggestion)"
    if v <= 0.5:
        return "Level 2 — Moderate (weekly summary + route)"
    if v <= 0.75:
        return "Level 3 — Strong (leaderboard + badges)"
    return "Level 4 — Full (Watch nudge + social feed)"


# Keep for the sidebar expander (static display only)
_STRENGTH_LABEL = {
    "0.0": "Level 0 — Off (no feature)",
    "0.25": "Level 1 — Subtle (quiet Maps suggestion)",
    "0.5": "Level 2 — Moderate (weekly summary + route)",
    "0.75": "Level 3 — Strong (leaderboard + badges)",
    "1.0": "Level 4 — Full (Watch nudge + social feed)",
}


@st.cache_data(show_spinner=False)
def _load_index():
    # Prefer the explicit index file if it exists
    path = _REPORTS_DIR / "simulation_index.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)

    # Auto-discover sim files so the dashboard works even before the full sweep is done
    import re

    found = []
    for p in sorted(_REPORTS_DIR.glob("sim_strength_*.json")):
        m = re.search(r"sim_strength_([\d.]+)\.json", p.name)
        if m:
            found.append(float(m.group(1)))
    if not found:
        return None

    # Infer n_days from the first file
    with open(_REPORTS_DIR / f"sim_strength_{found[0]:.2f}.json") as f:
        sample = json.load(f)
    n_days = sample.get("config", {}).get(
        "n_days", len(sample["population"]["car_pct_over_time"]) - 1
    )

    return {
        "sweep_values": found,
        "files": {str(s): f"sim_strength_{s:.2f}.json" for s in found},
        "n_agents": sample.get("config", {}).get(
            "n_agents", len(sample["agent_profiles"])
        ),
        "n_days": n_days,
    }


@st.cache_data(show_spinner=False)
def _load_sim(strength_str: str):
    fname = f"sim_strength_{float(strength_str):.2f}.json"
    path = _REPORTS_DIR / fname
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── Circular layout for 15 nodes ─────────────────────────────────────────────
def _node_positions(n: int):
    positions = []
    for i in range(n):
        angle = 2 * math.pi * i / n - math.pi / 2
        positions.append((math.cos(angle), math.sin(angle)))
    return positions


def _render_pipeline() -> None:
    st.sidebar.markdown("**Simulation Design**")

    with st.sidebar.expander("Research Question"):
        st.markdown(
            "**H5:** Apple ESG features accelerate sustainable commute adoption via social "
            "norm diffusion — with downstream benefits in population health and mood.\n\n"
            "Sweep: 5 Apple feature strength levels, 15 agents, 7 days each."
        )

    with st.sidebar.expander("Agent Profiles"):
        st.markdown(
            "Each agent is a full person with:\n"
            "- Age, occupation, backstory\n"
            "- Conformity level (extremely → non-conformist)\n"
            "- Environmental consciousness (0–1)\n"
            "- Fitness level (0–1)\n"
            "- Commute distance, car ownership\n"
            "- Social openness (determines peer count: 2–5)\n\n"
            "Each day, the agent calls **Claude Haiku** and returns:\n"
            "`Reasoning` · `Mood` · `Choice`"
        )

    with st.sidebar.expander("Dynamic Social Network"):
        st.markdown(
            "Peer connections are **weighted** and **evolve daily**:\n\n"
            "- Peers who chose sustainably gain +10% salience weight\n"
            "- Peers who drove decay -3% each day\n"
            "- Weights normalised to sum = 1 per agent\n\n"
            "This models how visible green behaviour becomes more influential over time."
        )

    with st.sidebar.expander("Apple Nudge Levels"):
        for k, v in _STRENGTH_LABEL.items():
            st.markdown(f"**{v}**")

    with st.sidebar.expander("Tracked Metrics"):
        st.markdown(
            "**Per agent per day:**\n"
            "- Commute mode (car / transit / bike / walk)\n"
            "- Mood (stressed / neutral / content / energized)\n"
            "- Cumulative health score (+2 walk/bike · +0.5 transit · 0 car)\n"
            "- Daily CO₂ kg\n\n"
            "**Population per day:** car% · total CO₂"
        )


def _render_summary(index: dict) -> None:
    st.markdown("### H5 Summary — Apple Feature Strength vs Car Usage Reduction")
    cols = st.columns(5)
    rows = []
    for col, s in zip(cols, index["sweep_values"]):
        data = _load_sim(str(s))
        if data is None:
            col.metric(label=f"strength={s}", value="N/A")
            continue
        pop = data["population"]
        start = pop["car_pct_over_time"][0]
        end = pop["car_pct_over_time"][-1]
        delta = end - start
        col.metric(
            label=_strength_label(s).split("—")[0].strip(),
            value=f"{end:.1f}%",
            delta=f"{delta:+.1f}% vs day 0",
            delta_color="inverse",
        )
        rows.append((s, start, end))

    if rows:
        best = min(rows, key=lambda r: r[2])
        worst = max(rows, key=lambda r: r[2])
        if best[2] < worst[2]:
            with st.expander("H5 Result", expanded=True):
                st.write(
                    f"H5 **supported** — Full ESG feature (strength={best[0]}) achieved "
                    f"{best[2]:.1f}% car usage by day 7, vs {worst[2]:.1f}% with no feature. "
                    f"Apple social nudges accelerated sustainable adoption."
                )
        else:
            with st.expander("H5 Result", expanded=True):
                st.write(
                    "H5 **not supported** — car usage did not decrease with Apple feature. "
                    "Both results are reported regardless."
                )


def _render_trajectories(index: dict) -> None:
    st.markdown("### Population Trajectories Across Feature Levels")
    _color_palette = ["#adb5bd", "#74c69d", "#f0a500", "#e76f51", "#0071e3"]
    all_strengths = sorted(set(index["sweep_values"]))
    colors = {
        str(s): _color_palette[i % len(_color_palette)]
        for i, s in enumerate(all_strengths)
    }
    days = list(range(index["n_days"] + 1))

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Car Usage (%)**")
        fig = go.Figure()
        for s in index["sweep_values"]:
            data = _load_sim(str(s))
            if data is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=data["population"]["car_pct_over_time"],
                    mode="lines+markers",
                    name=_strength_label(s),
                    line=dict(color=colors[str(s)], width=2),
                    marker=dict(size=5),
                )
            )
        fig.update_layout(
            yaxis=dict(title="% by Car", range=[0, 100]),
            xaxis_title="Day",
            height=300,
            margin=dict(t=10, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**Total CO₂ (kg/day)**")
        fig2 = go.Figure()
        for s in index["sweep_values"]:
            data = _load_sim(str(s))
            if data is None:
                continue
            fig2.add_trace(
                go.Scatter(
                    x=days,
                    y=data["population"]["co2_over_time"],
                    mode="lines+markers",
                    name=_strength_label(s),
                    line=dict(color=colors[str(s)], width=2),
                    marker=dict(symbol="square", size=5),
                )
            )
        fig2.update_layout(
            yaxis_title="kg CO₂",
            xaxis_title="Day",
            height=300,
            margin=dict(t=10, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Health + mood comparison (baseline vs full only)
    col3, col4 = st.columns(2)
    base = _load_sim("0.0")
    full = _load_sim("1.0")

    if base and full:
        profiles = base["agent_profiles"]
        n_days = index["n_days"]

        with col3:
            st.markdown("**Avg Health Score — Baseline vs Full Feature**")
            fig3 = go.Figure()
            for label, data, color in [
                ("Baseline", base, "#adb5bd"),
                ("Full Feature", full, "#0071e3"),
            ]:
                avg = [
                    float(
                        np.mean(
                            [
                                data["agents"][p["name"]]["health_history"][d]
                                for p in profiles
                            ]
                        )
                    )
                    for d in range(n_days + 1)
                ]
                fig3.add_trace(
                    go.Scatter(
                        x=days,
                        y=avg,
                        mode="lines+markers",
                        name=label,
                        line=dict(color=color, width=2),
                        marker=dict(symbol="triangle-up", size=6),
                    )
                )
            fig3.update_layout(
                yaxis_title="Avg health points",
                xaxis_title="Day",
                height=280,
                margin=dict(t=10, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            st.markdown("**Mood Distribution — Baseline vs Full Feature**")
            all_moods = ["stressed", "neutral", "content", "energized"]
            mood_counts = {}
            for label, data in [("Baseline", base), ("Full Feature", full)]:
                cnt = {m: 0 for m in all_moods}
                for p in profiles:
                    for m in data["agents"][p["name"]]["mood_history"]:
                        if m in cnt:
                            cnt[m] += 1
                mood_counts[label] = cnt

            fig4 = go.Figure()
            for label, color in [("Baseline", "#adb5bd"), ("Full Feature", "#0071e3")]:
                fig4.add_trace(
                    go.Bar(
                        name=label,
                        x=all_moods,
                        y=[mood_counts[label][m] for m in all_moods],
                        marker_color=color,
                    )
                )
            fig4.update_layout(
                barmode="group",
                yaxis_title="Count",
                height=280,
                margin=dict(t=10, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig4, use_container_width=True)


def _render_network(index: dict) -> None:
    st.markdown("### Social Network — Time-Lapse")
    st.caption(
        "Each node = one agent. Colour = commute mode that day. Size = cumulative health score. "
        "Edge thickness = peer influence weight. Use the slider to step through days."
    )

    strength_options = {_strength_label(s): str(s) for s in index["sweep_values"]}
    selected_label = st.selectbox("Apple feature level", list(strength_options.keys()))
    strength_str = strength_options[selected_label]
    data = _load_sim(strength_str)
    if data is None:
        st.warning("Results not found for this level. Run the notebook first.")
        return

    n_days = data["config"]["n_days"]
    day = st.slider("Day", 0, n_days, 0, key="network_day")
    # Share state so the agent inspector below can read the same day + strength
    st.session_state["sim_day"] = day
    st.session_state["sim_strength"] = strength_str

    profiles = data["agent_profiles"]
    n = len(profiles)
    pos = _node_positions(n)

    snapshots = data["network_snapshots"]
    net = snapshots[day] if day < len(snapshots) else snapshots[-1]

    # Build edge traces (one per edge, weighted thickness)
    edge_traces = []
    for i_str, peers in net.items():
        i = int(i_str)
        xi, yi = pos[i]
        for j_str, weight in peers.items():
            j = int(j_str)
            xj, yj = pos[j]
            edge_traces.append(
                go.Scatter(
                    x=[xi, xj, None],
                    y=[yi, yj, None],
                    mode="lines",
                    line=dict(
                        width=max(0.5, weight * 8), color="rgba(150,150,150,0.4)"
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Node data
    agent_data_map = data["agents"]
    node_x, node_y, node_colors, node_sizes = [], [], [], []
    node_text, hover_texts = [], []

    for i, p in enumerate(profiles):
        name = p["name"]
        a = agent_data_map[name]
        choice = a["history"][day] if day < len(a["history"]) else a["history"][-1]
        mood = (
            a["mood_history"][day]
            if day < len(a["mood_history"])
            else a["mood_history"][-1]
        )
        health = (
            a["health_history"][day]
            if day < len(a["health_history"])
            else a["health_history"][-1]
        )
        co2 = (
            a["co2_history"][day]
            if day < len(a["co2_history"])
            else a["co2_history"][-1]
        )

        xi, yi = pos[i]
        node_x.append(xi)
        node_y.append(yi)
        node_colors.append(_MODE_COLOR.get(choice, "#888"))
        node_sizes.append(max(20, 18 + health * 1.5))
        node_text.append(name)
        hover_texts.append(
            f"<b>{name}</b><br>"
            f"{p['age']}y · {p['occupation']}<br>"
            f"Mode: {_MODE_ICON.get(choice,'')} {choice}<br>"
            f"Mood: {_MOOD_ICON.get(mood,'')} {mood}<br>"
            f"Health: {health:.1f} pts<br>"
            f"CO₂: {co2:.2f} kg<br>"
            f"Conformity: {p['conformity']}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes, color=node_colors, line=dict(width=2, color="#fff")
        ),
        text=node_text,
        textposition="top center",
        textfont=dict(size=10),
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
    )

    # Legend annotation
    legend_shapes = []
    legend_text = []
    for k, (mode, color) in enumerate(
        [
            ("car", "#e74c3c"),
            ("transit", "#f0a500"),
            ("bike", "#2ecc71"),
            ("walk", "#3498db"),
        ]
    ):
        legend_shapes.append(
            dict(
                type="circle",
                xref="paper",
                yref="paper",
                x0=0.01 + k * 0.12,
                y0=0.01,
                x1=0.03 + k * 0.12,
                y1=0.04,
                fillcolor=color,
                line_color=color,
            )
        )
        legend_text.append(
            dict(
                xref="paper",
                yref="paper",
                showarrow=False,
                x=0.04 + k * 0.12,
                y=0.025,
                text=f"{_MODE_ICON[mode]} {mode}",
                font=dict(size=10),
            )
        )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        height=520,
        margin=dict(t=30, b=10, l=10, r=10),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, range=[-1.4, 1.4]
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, range=[-1.4, 1.4]
        ),
        plot_bgcolor="#f9f9f9",
        annotations=legend_text,
        shapes=legend_shapes,
        title=dict(text=f"Day {day} — Social Network", font=dict(size=14)),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_agent_inspector(index: dict) -> None:
    st.markdown("### Agent Inspector")
    st.caption(
        "Select an agent to see their profile and reasoning for the day selected in the time-lapse above."
    )

    # Read day + strength set by the network slider above
    day = st.session_state.get("sim_day", 0)
    strength_str = st.session_state.get("sim_strength", str(index["sweep_values"][0]))

    data = _load_sim(strength_str)
    if data is None:
        st.warning("Results not found.")
        return

    profiles = data["agent_profiles"]
    profile_map = {p["name"]: p for p in profiles}
    agent_name = st.selectbox("Agent", [p["name"] for p in profiles])

    p = profile_map[agent_name]
    a = data["agents"][agent_name]

    choice = a["history"][day] if day < len(a["history"]) else "—"
    mood = a["mood_history"][day] if day < len(a["mood_history"]) else "—"
    health = a["health_history"][day] if day < len(a["health_history"]) else 0
    co2 = a["co2_history"][day] if day < len(a["co2_history"]) else 0
    # reasoning index: day 0 has no reasoning (initial state), day N uses index N-1
    reasoning = (
        a["reasoning_history"][day - 1]
        if day > 0 and day - 1 < len(a["reasoning_history"])
        else ""
    )

    m_color = _MODE_COLOR.get(choice, "#ccc")

    col_card, col_day = st.columns([1, 1])

    with col_card:
        st.markdown(
            f"<div style='background:#f5f5f7; border-radius:10px; padding:1rem; font-size:0.9rem;'>"
            f"<b style='font-size:1.1rem'>{agent_name}</b><br>"
            f"{p['age']}y · {p['occupation']}<br><br>"
            f"<i>{p['backstory']}</i><br><br>"
            f"Conformity: <b>{p['conformity']}</b><br>"
            f"Env. consciousness: <b>{p['env_consciousness']:.0%}</b><br>"
            f"Fitness: <b>{p['fitness']:.0%}</b><br>"
            f"Commute: <b>{p['commute_km']} km</b><br>"
            f"Car owner: <b>{'Yes' if p['car_owner'] else 'No'}</b><br>"
            f"Social openness: <b>{p['social_openness']} peers</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_day:
        st.markdown(
            f"<div style='background:{m_color}22; border:2px solid {m_color}; border-radius:10px; "
            f"padding:1rem; text-align:center;'>"
            f"<div style='font-size:0.85rem; color:#6e6e73; margin-bottom:0.3rem'>Day {day}</div>"
            f"<div style='font-size:1.6rem'>{_MODE_ICON.get(choice, '')} "
            f"<b style='font-size:1.1rem'>{choice}</b></div>"
            f"<div style='margin-top:0.5rem; font-size:0.9rem'>"
            f"{_MOOD_ICON.get(mood, '')} {mood} &nbsp;·&nbsp; "
            f"❤️ {health:.0f} pts &nbsp;·&nbsp; 🌿 {co2:.1f} kg CO₂"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        if reasoning:
            st.markdown(" ")
            with st.expander("Reasoning", expanded=True):
                st.markdown(reasoning)
        elif day == 0:
            st.caption("Day 0 is the initial state — no reasoning yet.")

    # ── Time-series charts for this agent ─────────────────────────────────────
    st.markdown(" ")
    n_days_total = data["config"]["n_days"]
    all_days = list(range(n_days_total + 1))

    _mood_score = {"stressed": 0, "neutral": 1, "content": 2, "energized": 3}

    mood_vals   = [_mood_score.get(m, 1) for m in a["mood_history"]]
    health_vals = a["health_history"]
    co2_vals    = a["co2_history"]

    def _marker_colors(n, selected):
        return ["#0071e3" if i == selected else "#adb5bd" for i in range(n)]

    gc1, gc2 = st.columns(2)
    gc3, gc4 = st.columns(2)

    with gc1:
        fig = go.Figure(go.Scatter(
            x=all_days, y=mood_vals, mode="lines+markers",
            line=dict(color="#9b59b6", width=2),
            marker=dict(color=_marker_colors(len(mood_vals), day), size=9,
                        line=dict(width=1, color="#fff")),
            hovertemplate="Day %{x}<br>%{text}<extra></extra>",
            text=a["mood_history"],
        ))
        fig.update_layout(
            title="Mood Over Time",
            yaxis=dict(tickvals=[0, 1, 2, 3],
                       ticktext=["😟 stressed", "😐 neutral", "😊 content", "⚡ energized"],
                       range=[-0.4, 3.4]),
            xaxis_title="Day", height=220, margin=dict(t=35, b=30, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with gc2:
        fig = go.Figure(go.Scatter(
            x=all_days, y=health_vals, mode="lines+markers",
            line=dict(color="#e74c3c", width=2),
            marker=dict(color=_marker_colors(len(health_vals), day), size=9,
                        line=dict(width=1, color="#fff")),
            hovertemplate="Day %{x}<br>Health: %{y:.1f} pts<extra></extra>",
        ))
        fig.update_layout(
            title="Cumulative Health Score",
            yaxis_title="pts", xaxis_title="Day",
            height=220, margin=dict(t=35, b=30, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with gc3:
        fig = go.Figure(go.Scatter(
            x=all_days, y=co2_vals, mode="lines+markers",
            line=dict(color="#2ecc71", width=2),
            marker=dict(color=_marker_colors(len(co2_vals), day), size=9,
                        line=dict(width=1, color="#fff")),
            hovertemplate="Day %{x}<br>CO₂: %{y:.2f} kg<extra></extra>",
            fill="tozeroy", fillcolor="rgba(46,204,113,0.1)",
        ))
        fig.update_layout(
            title="Daily CO₂ Emissions",
            yaxis_title="kg CO₂", xaxis_title="Day",
            height=220, margin=dict(t=35, b=30, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with gc4:
        mode_colors = [_MODE_COLOR.get(c, "#adb5bd") for c in a["history"]]
        fig = go.Figure(go.Bar(
            x=all_days, y=[1] * len(all_days),
            marker_color=mode_colors,
            hovertemplate="Day %{x}<br>%{text}<extra></extra>",
            text=a["history"],
        ))
        fig.add_vline(x=day, line_width=2, line_dash="dash", line_color="#1d1d1f")
        fig.update_layout(
            title="Commute Mode per Day",
            yaxis=dict(showticklabels=False, range=[0, 1.5]),
            xaxis_title="Day", height=220, margin=dict(t=35, b=30, l=80, r=10),
            showlegend=False,
        )
        # Compact mode legend on the left y-axis label area
        for k, (mode, color) in enumerate(_MODE_COLOR.items()):
            fig.add_annotation(
                xref="paper", yref="paper", x=-0.12,
                y=0.95 - k * 0.22,
                text=f"<span style='color:{color}'>■</span> {_MODE_ICON[mode]}",
                showarrow=False, font=dict(size=10), xanchor="left",
            )
        st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    _render_pipeline()

    index = _load_index()
    if index is None:
        st.warning(
            "No simulation results found. "
            "Run `eda/agentic_simulation.ipynb` to generate the log files."
        )
        return

    _render_summary(index)
    st.markdown("---")
    _render_trajectories(index)
    st.markdown("---")
    _render_network(index)
    st.markdown("---")
    _render_agent_inspector(index)
