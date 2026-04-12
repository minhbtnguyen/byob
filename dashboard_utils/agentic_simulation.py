import json
import math
import re
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
_MOOD_SCORE = {"stressed": 0, "neutral": 1, "content": 2, "energized": 3}
_SWEEP_COLORS = ["#adb5bd", "#74c69d", "#f0a500", "#e76f51", "#0071e3"]
SUSTAINABLE_MODES = {"transit", "bike", "walk"}


def _strength_label(s) -> str:
    v = float(s)
    if v == 0.0:
        return "Level 0 - Off"
    if v <= 0.25:
        return "Level 1 - Subtle"
    if v <= 0.5:
        return "Level 2 - Moderate"
    if v <= 0.75:
        return "Level 3 - Strong"
    return "Level 4 - Full"


@st.cache_data(show_spinner=False)
def _load_index():
    path = _REPORTS_DIR / "simulation_index.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    found = sorted(
        float(m.group(1))
        for p in _REPORTS_DIR.glob("sim_strength_*.json")
        if (m := re.search(r"sim_strength_([\d.]+)\.json", p.name))
    )
    if not found:
        return None
    with open(_REPORTS_DIR / f"sim_strength_{found[0]:.2f}.json") as f:
        sample = json.load(f)
    return {
        "sweep_values": found,
        "n_agents": sample["config"].get("n_agents", len(sample["agent_profiles"])),
        "n_days": sample["config"].get(
            "n_days", len(sample["population"]["car_pct_over_time"]) - 1
        ),
    }


@st.cache_data(show_spinner=False)
def _load_sim(strength_str: str):
    path = _REPORTS_DIR / f"sim_strength_{float(strength_str):.2f}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _node_positions(n: int):
    return [
        (
            math.cos(2 * math.pi * i / n - math.pi / 2),
            math.sin(2 * math.pi * i / n - math.pi / 2),
        )
        for i in range(n)
    ]


# ── Sidebar ───────────────────────────────────────────────────────────────────


def _render_pipeline() -> None:
    st.sidebar.markdown("**Simulation Design**")

    with st.sidebar.expander("Research Question"):
        st.markdown(
            "If Apple rolls out an ESG feature (Green Commute Score in Apple Maps / Health), "
            "how does social influence diffuse sustainable transport choices - and what are the "
            "downstream effects on population health and mood?\n\n"
            "Sweep: 5 feature strength levels, 15 agents, 7 days."
        )

    with st.sidebar.expander("Agent Profiles"):
        st.markdown(
            "Each agent has: age, occupation, backstory, conformity level, "
            "environmental consciousness (0-1), fitness (0-1), commute distance, "
            "car ownership, and social openness (2-5 peers).\n\n"
            "Each day the agent calls **Claude Haiku** and returns: "
            "Reasoning, Mood, Choice."
        )

    with st.sidebar.expander("Dynamic Social Network"):
        st.markdown(
            "Peer influence weights update daily:\n"
            "- Peers who chose sustainably gain +10% salience\n"
            "- Peers who drove decay -3% per day\n"
            "- Weights normalised to sum = 1 per agent"
        )

    with st.sidebar.expander("Apple Nudge Levels"):
        st.markdown(
            "**Level 0** - Feature off  \n"
            "**Level 1** - Quiet route suggestion in Maps  \n"
            "**Level 2** - Weekly carbon summary + route nudge  \n"
            "**Level 3** - Leaderboard + carbon badges  \n"
            "**Level 4** - Real-time Watch nudge + social sharing"
        )

    with st.sidebar.expander("Tracked Metrics"):
        st.markdown(
            "Per agent per day: commute mode, mood, cumulative health score "
            "(+2 walk/bike, +0.5 transit, 0 car), daily CO2 kg\n\n"
            "Population per day: car%, total CO2"
        )


# ── H5 Summary ────────────────────────────────────────────────────────────────


def _render_summary(index: dict) -> None:
    st.markdown("### Apple Feature Strength vs Car Usage Reduction")
    cols = st.columns(5)
    rows = []
    for col, s in zip(cols, index["sweep_values"]):
        data = _load_sim(str(s))
        if data is None:
            col.metric(label=_strength_label(s), value="N/A")
            continue
        pop = data["population"]
        start, end = pop["car_pct_over_time"][0], pop["car_pct_over_time"][-1]
        col.metric(
            label=_strength_label(s),
            value=f"{end:.1f}%",
            delta=f"{end - start:+.1f}% vs day 0",
            delta_color="inverse",
        )
        rows.append((s, start, end))

    if not rows:
        return
    best = min(rows, key=lambda r: r[2])
    worst = max(rows, key=lambda r: r[2])
    with st.expander("Analysis", expanded=True):
        if best[2] < worst[2]:
            st.write(
                f"{_strength_label(best[0])} achieved {best[2]:.1f}% car usage "
                f"by day 7, vs {worst[2]:.1f}% with no feature. "
                f"Apple social nudges accelerated sustainable adoption."
            )
        else:
            st.write("Car usage did not decrease with the Apple feature.")


# ── Trajectories ──────────────────────────────────────────────────────────────


def _render_trajectories(index: dict) -> None:
    st.markdown("### Population Trajectories Across Feature Levels")

    all_strengths = sorted(index["sweep_values"])
    colors = {
        str(s): _SWEEP_COLORS[i % len(_SWEEP_COLORS)]
        for i, s in enumerate(all_strengths)
    }
    days = list(range(index["n_days"] + 1))

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Car Usage (%)**")
        fig = go.Figure()
        for s in all_strengths:
            d = _load_sim(str(s))
            if d is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=d["population"]["car_pct_over_time"],
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
        st.markdown("**Total CO2 (kg/day)**")
        fig = go.Figure()
        for s in all_strengths:
            d = _load_sim(str(s))
            if d is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=d["population"]["co2_over_time"],
                    mode="lines+markers",
                    name=_strength_label(s),
                    line=dict(color=colors[str(s)], width=2),
                    marker=dict(symbol="square", size=5),
                )
            )
        fig.update_layout(
            yaxis_title="kg CO2",
            xaxis_title="Day",
            height=300,
            margin=dict(t=10, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Health + mood: baseline vs full only
    base = _load_sim("0.0")
    full = _load_sim("1.0")
    if not (base and full):
        return

    profiles = base["agent_profiles"]
    n_days = index["n_days"]
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Avg Health Score - Baseline vs Full Feature**")
        fig = go.Figure()
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
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=avg,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=2),
                    marker=dict(symbol="triangle-up", size=6),
                )
            )
        fig.update_layout(
            yaxis_title="Avg health pts",
            xaxis_title="Day",
            height=280,
            margin=dict(t=10, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("**Mood Distribution - Baseline vs Full Feature**")
        all_moods = ["stressed", "neutral", "content", "energized"]
        fig = go.Figure()
        for label, data, color in [
            ("Baseline", base, "#adb5bd"),
            ("Full Feature", full, "#0071e3"),
        ]:
            cnt = {m: 0 for m in all_moods}
            for p in profiles:
                for m in data["agents"][p["name"]]["mood_history"]:
                    if m in cnt:
                        cnt[m] += 1
            fig.add_trace(
                go.Bar(
                    name=label,
                    x=all_moods,
                    y=[cnt[m] for m in all_moods],
                    marker_color=color,
                )
            )
        fig.update_layout(
            barmode="group",
            yaxis_title="Count",
            height=280,
            margin=dict(t=10, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    _render_trajectory_analysis(index, base, full, profiles, n_days)


def _render_trajectory_analysis(index, base, full, profiles, n_days) -> None:

    level_stats = []
    for s in sorted(index["sweep_values"]):
        d = _load_sim(str(s))
        if d is None:
            continue
        pop = d["population"]
        level_stats.append(
            {
                "s": s,
                "label": _strength_label(s),
                "car_start": pop["car_pct_over_time"][0],
                "car_end": pop["car_pct_over_time"][-1],
                "co2_end": pop["co2_over_time"][-1],
                "co2_start": pop["co2_over_time"][0],
                "health_end": float(
                    np.mean(
                        [
                            d["agents"][p["name"]]["health_history"][-1]
                            for p in d["agent_profiles"]
                        ]
                    )
                ),
            }
        )
    if not level_stats:
        return

    base_stat = level_stats[0]
    best_car = min(level_stats, key=lambda x: x["car_end"])
    best_co2 = min(level_stats, key=lambda x: x["co2_end"])
    best_hlth = max(level_stats, key=lambda x: x["health_end"])

    all_moods = ["stressed", "neutral", "content", "energized"]
    b_cnt = {m: 0 for m in all_moods}
    f_cnt = {m: 0 for m in all_moods}
    for p in profiles:
        for m in base["agents"][p["name"]]["mood_history"]:
            if m in b_cnt:
                b_cnt[m] += 1
        for m in full["agents"][p["name"]]["mood_history"]:
            if m in f_cnt:
                f_cnt[m] += 1
    total_b = sum(b_cnt.values()) or 1
    total_f = sum(f_cnt.values()) or 1
    pos_base = (b_cnt["content"] + b_cnt["energized"]) / total_b * 100
    pos_full = (f_cnt["content"] + f_cnt["energized"]) / total_f * 100

    car_reduction = base_stat["car_end"] - best_car["car_end"]
    co2_reduction = base_stat["co2_end"] - best_co2["co2_end"]
    health_gain = best_hlth["health_end"] - base_stat["health_end"]
    is_monotonic = all(
        level_stats[i]["co2_end"] >= level_stats[i + 1]["co2_end"]
        for i in range(len(level_stats) - 1)
    )

    with st.expander("Analysis", expanded=True):
        st.markdown(
            f"**Car Usage.** Without any Apple feature, car commuting settled at "
            f"**{base_stat['car_end']:.1f}%** by day {n_days} (starting from "
            f"{base_stat['car_start']:.1f}%). Social dynamics alone produced "
            f"{'a modest reduction' if base_stat['car_end'] < base_stat['car_start'] else 'no meaningful reduction'} "
            f"in driving. {best_car['label']} brought car usage down to "
            f"**{best_car['car_end']:.1f}%** - a reduction of **{car_reduction:.1f} percentage points** "
            f"vs baseline. This suggests Apple's social nudge shifts commute norms beyond "
            f"what peer influence alone achieves.\n\n"
            f"**CO2 Emissions.** Daily population CO2 dropped by **{co2_reduction:.1f} kg** "
            f"at the strongest available level ({best_co2['label']}) vs baseline "
            f"({base_stat['co2_end']:.1f} to {best_co2['co2_end']:.1f} kg/day). "
            f"The relationship is {'monotonic - each nudge level produces further reductions' if is_monotonic else 'non-linear - intermediate levels do not always produce proportional gains'}.\n\n"
            f"**Health.** Average cumulative health rose from **{base_stat['health_end']:.1f} pts** "
            f"(baseline) to **{best_hlth['health_end']:.1f} pts** ({best_hlth['label']}), "
            f"a gain of **{health_gain:.1f} pts per agent** over {n_days} days. "
            f"This is a direct consequence of agents switching to walking and cycling, "
            f"which contribute +2 health points per day vs 0 for car.\n\n"
            f"**Mood.** Positive moods (content + energized) rose from **{pos_base:.0f}%** "
            f"(no feature) to **{pos_full:.0f}%** (full feature) across all agents and days.\n\n"
            f"**Note on health vs CO2.** The Apple nudge is more effective at reducing CO2 than "
            f"improving health scores. When agents switch away from car, they tend to choose transit "
            f"(+0.5 health pts/day) rather than walking or cycling (+2 pts/day). Baseline agents who "
            f"were already walking or biking naturally continue accumulating high health scores. "
            f"A stronger product intervention would nudge short trips specifically toward active "
            f"transport rather than any sustainable mode."
        )


# ── Network time-lapse ────────────────────────────────────────────────────────


def _render_network(index: dict) -> None:
    st.markdown("### Social Network - Time-Lapse")
    st.info(
        "Node colour = commute mode, Node size = cumulative health score, "
        "Edge thickness = peer influence weight"
    )

    strength_options = {_strength_label(s): str(s) for s in index["sweep_values"]}
    selected = st.selectbox("Apple feature level", list(strength_options.keys()))
    strength_str = strength_options[selected]
    data = _load_sim(strength_str)
    if data is None:
        st.warning("Results not found. Run the notebook first.")
        return

    n_days = data["config"]["n_days"]
    day = st.slider("Day", 0, n_days, 0, key="network_day")
    st.session_state["sim_day"] = day
    st.session_state["sim_strength"] = strength_str

    profiles = data["agent_profiles"]
    pos = _node_positions(len(profiles))
    net = data["network_snapshots"][min(day, len(data["network_snapshots"]) - 1)]

    edge_traces = [
        go.Scatter(
            x=[pos[int(i)][0], pos[int(j)][0], None],
            y=[pos[int(i)][1], pos[int(j)][1], None],
            mode="lines",
            line=dict(width=max(0.5, w * 8), color="rgba(150,150,150,0.4)"),
            hoverinfo="skip",
            showlegend=False,
        )
        for i, peers in net.items()
        for j, w in peers.items()
    ]

    node_x, node_y, node_colors, node_sizes, node_text, hover_texts = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i, p in enumerate(profiles):
        a = data["agents"][p["name"]]
        choice = a["history"][min(day, len(a["history"]) - 1)]
        mood = a["mood_history"][min(day, len(a["mood_history"]) - 1)]
        health = a["health_history"][min(day, len(a["health_history"]) - 1)]
        co2 = a["co2_history"][min(day, len(a["co2_history"]) - 1)]
        x, y = pos[i]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(_MODE_COLOR.get(choice, "#888"))
        node_sizes.append(max(20, 18 + health * 1.5))
        node_text.append(p["name"])
        hover_texts.append(
            f"<b>{p['name']}</b><br>{p['age']}y, {p['occupation']}<br>"
            f"Mode: {choice}<br>Mood: {mood}<br>"
            f"Health: {health:.1f} pts<br>CO2: {co2:.2f} kg<br>"
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

    # Mode legend
    legend_annots = [
        dict(
            xref="paper",
            yref="paper",
            showarrow=False,
            x=0.01 + k * 0.13,
            y=0.02,
            text=f"<b style='color:{color}'>■</b> {mode}",
            font=dict(size=10),
            xanchor="left",
        )
        for k, (mode, color) in enumerate(_MODE_COLOR.items())
    ]

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
        annotations=legend_annots,
        title=dict(text=f"Day {day} - Social Network", font=dict(size=14)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-agent winners table
    base_data = _load_sim("0.0")
    if base_data:
        _render_agent_winners(index, base_data)


def _render_agent_winners(index: dict, base_data: dict) -> None:
    with st.expander("Analysis", expanded=True):
        profiles = base_data["agent_profiles"]
        base_agents = base_data["agents"]

        def _pos_pct(hist):
            total = len(hist) or 1
            return sum(m in ("content", "energized") for m in hist) / total * 100

        winners = []
        for s in sorted(index["sweep_values"]):
            if float(s) == 0.0:
                continue
            d = _load_sim(str(s))
            if d is None:
                continue
            agents_d = d["agents"]
            health_gains = {
                p["name"]: agents_d[p["name"]]["health_history"][-1]
                - base_agents[p["name"]]["health_history"][-1]
                for p in profiles
            }
            mood_gains = {
                p["name"]: _pos_pct(agents_d[p["name"]]["mood_history"])
                - _pos_pct(base_agents[p["name"]]["mood_history"])
                for p in profiles
            }
            co2_savings = {
                p["name"]: sum(base_agents[p["name"]]["co2_history"])
                - sum(agents_d[p["name"]]["co2_history"])
                for p in profiles
            }
            best_health = max(health_gains, key=health_gains.get)
            best_mood = max(mood_gains, key=mood_gains.get)
            best_co2 = max(co2_savings, key=co2_savings.get)
            winners.append(
                {
                    "label": _strength_label(s),
                    "health": best_health,
                    "health_val": health_gains[best_health],
                    "mood": best_mood,
                    "mood_val": mood_gains[best_mood],
                    "co2": best_co2,
                    "co2_val": co2_savings[best_co2],
                }
            )

        if not winners:
            return

        header, *_ = [st.columns([1.4, 1, 1, 1])]
        header[0].markdown("**Level**")
        header[1].markdown("**Most health gain**")
        header[2].markdown("**Most mood improvement**")
        header[3].markdown("**Most CO2 saved**")
        for w in winners:
            row = st.columns([1.4, 1, 1, 1])
            row[0].markdown(w["label"])
            row[1].markdown(f"{w['health']} ({w['health_val']:+.1f} pts)")
            row[2].markdown(f"{w['mood']} ({w['mood_val']:+.1f}%)")
            row[3].markdown(f"{w['co2']} ({w['co2_val']:+.2f} kg)")


# ── Agent Inspector ───────────────────────────────────────────────────────────


def _render_agent_inspector(index: dict) -> None:
    st.markdown("### Agent Inspector")
    st.info(
        "Select an agent to see their profile and reasoning for the day selected above."
    )

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

    choice = a["history"][min(day, len(a["history"]) - 1)]
    mood = a["mood_history"][min(day, len(a["mood_history"]) - 1)]
    health = a["health_history"][min(day, len(a["health_history"]) - 1)]
    co2 = a["co2_history"][min(day, len(a["co2_history"]) - 1)]
    reasoning = (
        a["reasoning_history"][day - 1]
        if day > 0 and day - 1 < len(a["reasoning_history"])
        else ""
    )
    m_color = _MODE_COLOR.get(choice, "#ccc")

    col_card, col_day = st.columns(2)

    with col_card:
        st.markdown(
            f"<div style='background:#f5f5f7; border-radius:10px; padding:1rem; font-size:0.9rem;'>"
            f"<b style='font-size:1.1rem'>{agent_name}</b><br>"
            f"{p['age']}y, {p['occupation']}<br><br>"
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
            f"<div style='font-size:1.4rem; font-weight:700'>{choice}</div>"
            f"<div style='margin-top:0.5rem; font-size:0.9rem'>"
            f"{mood} &nbsp;·&nbsp; {health:.0f} health pts &nbsp;·&nbsp; {co2:.1f} kg CO2"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        if reasoning:
            st.markdown(" ")
            with st.expander("Reasoning", expanded=True):
                st.markdown(reasoning)
        elif day == 0:
            st.caption("Day 0 is the initial state - no reasoning yet.")

    # Time-series charts
    st.markdown(" ")
    n_days_total = data["config"]["n_days"]
    all_days = list(range(n_days_total + 1))
    dot_colors = ["#0071e3" if i == day else "#adb5bd" for i in all_days]

    gc1, gc2 = st.columns(2)
    gc3, gc4 = st.columns(2)

    with gc1:
        fig = go.Figure(
            go.Scatter(
                x=all_days,
                y=[_MOOD_SCORE.get(m, 1) for m in a["mood_history"]],
                mode="lines+markers",
                line=dict(color="#9b59b6", width=2),
                marker=dict(color=dot_colors, size=9, line=dict(width=1, color="#fff")),
                hovertemplate="Day %{x}: %{text}<extra></extra>",
                text=a["mood_history"],
            )
        )
        fig.update_layout(
            title="Mood",
            xaxis_title="Day",
            yaxis=dict(
                tickvals=[0, 1, 2, 3],
                ticktext=["stressed", "neutral", "content", "energized"],
                range=[-0.4, 3.4],
            ),
            height=220,
            margin=dict(t=35, b=30, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with gc2:
        fig = go.Figure(
            go.Scatter(
                x=all_days,
                y=a["health_history"],
                mode="lines+markers",
                line=dict(color="#e74c3c", width=2),
                marker=dict(color=dot_colors, size=9, line=dict(width=1, color="#fff")),
                hovertemplate="Day %{x}: %{y:.1f} pts<extra></extra>",
            )
        )
        fig.update_layout(
            title="Cumulative Health Score",
            xaxis_title="Day",
            yaxis_title="pts",
            height=220,
            margin=dict(t=35, b=30, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with gc3:
        fig = go.Figure(
            go.Scatter(
                x=all_days,
                y=a["co2_history"],
                mode="lines+markers",
                line=dict(color="#2ecc71", width=2),
                marker=dict(color=dot_colors, size=9, line=dict(width=1, color="#fff")),
                hovertemplate="Day %{x}: %{y:.2f} kg<extra></extra>",
                fill="tozeroy",
                fillcolor="rgba(46,204,113,0.1)",
            )
        )
        fig.update_layout(
            title="Daily CO2 Emissions",
            xaxis_title="Day",
            yaxis_title="kg CO2",
            height=220,
            margin=dict(t=35, b=30, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with gc4:
        fig = go.Figure(
            go.Bar(
                x=all_days,
                y=[1] * len(all_days),
                marker_color=[_MODE_COLOR.get(c, "#adb5bd") for c in a["history"]],
                hovertemplate="Day %{x}: %{text}<extra></extra>",
                text=a["history"],
            )
        )
        fig.add_vline(x=day, line_width=2, line_dash="dash", line_color="#1d1d1f")
        fig.update_layout(
            title="Commute Mode per Day",
            xaxis_title="Day",
            yaxis=dict(showticklabels=False, range=[0, 1.5]),
            height=220,
            margin=dict(t=35, b=30, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Entry point ───────────────────────────────────────────────────────────────


def render() -> None:
    _render_pipeline()

    index = _load_index()
    if index is None:
        st.warning(
            "No simulation results found. Run `eda/agentic_simulation.ipynb` first."
        )
        return

    _render_summary(index)
    st.markdown("---")
    _render_trajectories(index)
    st.markdown("---")
    _render_network(index)
    st.markdown("---")
    _render_agent_inspector(index)
