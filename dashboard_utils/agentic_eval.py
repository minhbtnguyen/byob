import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import cohen_kappa_score

_REPORTS_DIR = Path(__file__).parent.parent / "reports"


@st.cache_data(show_spinner=False)
def _load_data():
    with open(_REPORTS_DIR / "user_reports.json") as f:
        reports = json.load(f)
    with open(_REPORTS_DIR / "eval_results.json") as f:
        eval_results = json.load(f)
    labels_df = pd.read_csv(_REPORTS_DIR / "human_labels.csv")
    return reports, eval_results, labels_df


def _render_pipeline() -> None:
    st.sidebar.markdown("**Experiment Design & Steps**")

    with st.sidebar.expander("1. Report Generation"):
        st.markdown(
            "**User selection:** from the held-out test fold, keep only users with ≥ 2 distinct "
            "predicted transport modes — single-mode users produce uninformative reports.\n\n"
            "**Aggregation:** per-user stats computed from classified windows: "
            "`mode_km`, `mode_co2_kg`, `total_co2_kg`, `car_pct_of_co2`.\n\n"
            "**Prompt (1 Claude call per user):**\n"
            "```\n"
            "You are a sustainability analyst writing a personal mobility report.\n"
            "Rules:\n"
            "  1. Write exactly 4-6 sentences.\n"
            "  2. Every number must come directly from the data below —\n"
            "     do not round, estimate, or invent figures.\n"
            "  3. End with one concrete, actionable recommendation.\n\n"
            "User mobility data: {mode_km, mode_co2_kg, total_co2_kg, car_pct_of_co2}\n"
            "```\n"
            "**Output:** report text → saved alongside raw stats for evaluation."
        )

    with st.sidebar.expander("2. Judge + Critic + Verdict"):
        st.markdown(
            "Three sequential Claude calls per user.\n\n"
            "**Judge (call 1) — initial fact-check:**\n"
            "```\n"
            "You are a fact-checker for sustainability reports.\n"
            "Given the raw data and the report, identify each numeric claim\n"
            "and label it: correct / incorrect / unverifiable.\n"
            "Cite the specific data value that supports or contradicts each claim.\n\n"
            "Raw data: {stats}    Report: {report}\n"
            "```\n\n"
            "**Critic (call 2) — review the Judge:**\n"
            "```\n"
            "You are reviewing a fact-checker's assessment.\n"
            "Identify any claims accepted too easily, verdicts that lack\n"
            "clear data support, or any missed claims.\n\n"
            "Fact-checker assessment: {judge_output}\n"
            "```\n\n"
            "**Revised Judge (call 3) — final verdict:**\n"
            "```\n"
            "You previously assessed a sustainability report.\n"
            "A critic has flagged issues. Revise your verdicts where\n"
            "the critic raises valid points.\n\n"
            "Raw data: {stats}    Report: {report}\n"
            "Critic flags: {critic_flags}\n"
            "```\n"
            "**Output:** final per-claim verdicts → `judge_final` in `eval_results.json`\n\n"
            "**Total calls — loop:** 4 per user (1 report + 3 judge/critic/revised)  \n"
        )

    with st.sidebar.expander("3. Baseline"):
        st.markdown(
            "Single-prompt judge with no Critic pass — used as the comparison baseline.\n\n"
            "**Prompt (1 Claude call per user):**\n"
            "```\n"
            "Given the data and the report, is each numeric claim\n"
            "correct, incorrect, or unverifiable?\n"
            "List each claim with its verdict.\n\n"
            "Data: {stats}    Report: {report}\n"
            "```\n"
            "**Output:** per-claim verdicts → `baseline_judge` in `eval_results.json`\n\n"
            "**Total calls — baseline:** 2 per user (1 report + 1 judge)  \n"
        )

    with st.sidebar.expander("4. Human Evaluation"):
        st.markdown(
            "After the notebook runs, a CSV template is auto-generated with:\n"
            "- one row per verifiable claim extracted from each report\n"
            "- `source_data` showing the full stats for that user\n"
            "- `baseline_label` and `loop_label` pre-filled from the LLM outputs\n\n"
            "**Human task:** fill in only `human_label` for each claim:  \n"
            "`correct` / `incorrect` / `unverifiable`\n\n"
            "Labeling criteria:\n"
            "- **correct** — every number in the sentence matches `source_data` exactly\n"
            "- **incorrect** — a number is wrong or a comparison is false\n"
            "- **unverifiable** — the claim cannot be confirmed or denied from the data alone "
            "(e.g. inferences about feasibility or future behaviour)"
        )

    with st.sidebar.expander("5. Benchmark"):
        st.markdown(
            "**Cohen's κ** measures agreement between two raters beyond what chance would produce.\n\n"
            "κ = 1.0 → perfect agreement  \n"
            "κ = 0.0 → agreement at chance level  \n"
            "κ < 0.0 → worse than chance\n\n"
            "Two κ values are computed:\n"
            "- **κ_baseline** — single-prompt judge vs human labels\n"
            "- **κ_loop** — Judge+Critic loop vs human labels\n\n"
            "**H3 is supported if κ_loop > κ_baseline.** Both values are reported "
            "regardless of direction."
        )


def _render_kappa(labels_df: pd.DataFrame) -> None:
    st.markdown("### Cohen's κ Evaluation")

    filled = labels_df[
        labels_df["human_label"].notna() & (labels_df["human_label"] != "")
    ]
    if filled.empty:
        st.warning(
            "No human labels found. Fill in `reports/human_labels.csv` to see results."
        )
        return

    human = filled["human_label"].tolist()
    baseline = filled["baseline_label"].tolist()
    loop = filled["loop_label"].tolist()

    k_baseline = cohen_kappa_score(human, baseline)
    k_loop = cohen_kappa_score(human, loop)
    delta = k_loop - k_baseline

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Claims evaluated", len(filled))
    c2.metric("Baseline κ (single-prompt)", f"{k_baseline:.3f}")
    c3.metric("Loop κ (Judge+Critic)", f"{k_loop:.3f}")
    c4.metric("Δ κ", f"{delta:+.3f}")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Cohen's κ — Baseline vs Loop**")
        fig = go.Figure(
            go.Bar(
                x=["Baseline (single-prompt)", "Loop (Judge+Critic)"],
                y=[k_baseline, k_loop],
                marker_color=["#adb5bd", "#0071e3"],
                text=[f"{k_baseline:.3f}", f"{k_loop:.3f}"],
                textposition="outside",
                width=0.4,
            )
        )
        fig.update_layout(
            yaxis=dict(
                title="Cohen's κ", range=[0, max(k_baseline, k_loop) * 1.4 + 0.05]
            ),
            height=280,
            margin=dict(t=20, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("**Label distribution across all claims**")
        labels = ["correct", "incorrect", "unverifiable"]
        dist = pd.DataFrame(
            {
                "Label": labels,
                "Human": [sum(l == v for l in human) for v in labels],
                "Baseline": [sum(l == v for l in baseline) for v in labels],
                "Loop": [sum(l == v for l in loop) for v in labels],
            }
        )
        fig2 = go.Figure()
        for col, color in [
            ("Human", "#1d1d1f"),
            ("Baseline", "#adb5bd"),
            ("Loop", "#0071e3"),
        ]:
            fig2.add_trace(
                go.Bar(
                    name=col,
                    x=dist["Label"],
                    y=dist[col],
                    marker_color=color,
                    text=dist[col],
                    textposition="outside",
                )
            )
        fig2.update_layout(
            barmode="group",
            height=280,
            margin=dict(t=20, b=40),
            yaxis_title="Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)

    if k_loop > k_baseline:
        with st.expander("H3 Result", expanded=True):
            st.write(
                f"H3 supported — Judge+Critic loop (κ={k_loop:.3f}) agrees with human labels "
                f"more than the single-prompt baseline (κ={k_baseline:.3f})."
            )
    else:
        with st.expander("H3 Result", expanded=True):
            st.write(
                f"H3 not supported — loop (κ={k_loop:.3f}) does not improve over baseline "
                f"(κ={k_baseline:.3f}). Both results are reported regardless."
            )


def _render_claim_table(labels_df: pd.DataFrame) -> None:
    st.markdown("### Claim-Level Verdicts")
    st.markdown(
        "Each row is one verifiable claim extracted from a generated report. "
        "Green = LLM verdict agrees with human label · Red = disagrees."
    )

    def highlight(row):
        styles = [""] * len(row)
        cols = list(row.index)
        for col in ["baseline_label", "loop_label"]:
            if col in cols and "human_label" in cols:
                idx = cols.index(col)
                color = "#d4edda" if row[col] == row["human_label"] else "#f8d7da"
                styles[idx] = f"background-color: {color}"
        return styles

    display = labels_df[
        ["user", "claim", "human_label", "baseline_label", "loop_label"]
    ].copy()
    display["claim"] = display["claim"].str[:90] + "…"
    st.dataframe(
        display.style.apply(highlight, axis=1),
        use_container_width=True,
        height=420,
    )


def _parse_nums(raw: str) -> set:
    """Expand a string like '1–7, 10 and 11' into a set of ints."""
    import re

    nums: set = set()
    for part in re.split(r"[,\s]+", raw):
        range_m = re.match(r"(\d+)\s*[-–]\s*(\d+)", part)
        if range_m:
            nums.update(range(int(range_m.group(1)), int(range_m.group(2)) + 1))
        elif part.isdigit():
            nums.add(int(part))
    return nums


def _extract_claim_section(text: str, claim_number: int, silent_msg: str = "") -> str:
    """Extract only the portion of an LLM output relevant to a specific claim number.

    Handles formats seen across all pipeline outputs:
    - Markdown table row:  | N | claim | verdict | data |
    - Section headers:     ## Claim N: / **Claim N:** / **Claim N –** / ### Claim N:
    - Grouped ranges:      ## Claims 1–7 and 10:
    - Summary tables:      | N. Claim text | verdict |

    If the claim number is not found (claim was not flagged/revised), returns
    `silent_msg` if provided, otherwise the full text as fallback.
    """
    import re

    lines = text.split("\n")

    # ── Format 1: standard table row  | N | ... |  ────────────────────────────
    table_row_re = re.compile(r"^\|\s*" + str(claim_number) + r"[\s\.\)]*\|")
    for line in lines:
        if table_row_re.match(line):
            return line.strip()

    # ── Format 2: section / header blocks ────────────────────────────────────
    # Matches: "Claim N:", "Claim N –", "Claims 1–7", numbered flags "1. Claim N"
    header_re = re.compile(
        r"(?:^|\s)(?:\d+\.\s*)?[Cc]laim[s]?\s+([\d,\s\-–and]+)",
        re.IGNORECASE,
    )
    headers = []
    for i, line in enumerate(lines):
        m = header_re.search(line)
        if not m:
            continue
        nums = _parse_nums(m.group(1))
        if nums:
            headers.append((i, nums))

    for idx, (line_i, nums) in enumerate(headers):
        if claim_number in nums:
            end_line = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines)
            return "\n".join(lines[line_i:end_line]).strip()

    # ── Not found ─────────────────────────────────────────────────────────────
    return silent_msg if silent_msg else text


def _render_example(eval_results: dict, labels_df: pd.DataFrame) -> None:
    st.markdown("### Report + Judge+Critic Trace")

    # ── User selector ─────────────────────────────────────────────────────────
    users = list(eval_results.keys())
    selected_user = st.selectbox("Select user", users)
    data = eval_results[selected_user]

    st.markdown("**Generated report**")
    st.info(data["report"])

    st.markdown("---")

    # ── Claim selector ────────────────────────────────────────────────────────
    user_claims = labels_df[
        labels_df["user"].astype(str).str.lstrip("0") == str(selected_user).lstrip("0")
    ].reset_index(drop=True)

    if user_claims.empty:
        st.warning("No claims found for this user in human_labels.csv.")
        return

    claim_options = {
        f"Claim {i+1}: {row['claim'][:80]}…": i for i, row in user_claims.iterrows()
    }
    selected_claim_key = st.selectbox("Select claim", list(claim_options.keys()))
    claim_idx = claim_options[selected_claim_key]
    claim_row = user_claims.iloc[claim_idx]
    claim_number = claim_idx + 1  # 1-based

    st.markdown(f"**Full claim:** {claim_row['claim']}")
    st.caption(f"Source data: {claim_row['source_data']}")

    st.markdown(" ")

    # ── Three-column verdict display ──────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Baseline judge**")
        verdict = claim_row.get("baseline_label", "—")
        color = {
            "correct": "#d4edda",
            "incorrect": "#f8d7da",
            "unverifiable": "#fff3cd",
        }.get(verdict, "#f5f5f7")
        st.markdown(
            f"<div style='background:{color}; border-radius:8px; padding:0.6rem 0.8rem; "
            f"font-weight:600; text-align:center; margin-bottom:0.8rem'>{verdict}</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Analysis"):
            st.markdown(_extract_claim_section(data["baseline_judge"], claim_number))

    with col_b:
        st.markdown("**Judge + Critic + Verdict**")
        verdict = claim_row.get("loop_label", "—")
        color = {
            "correct": "#d4edda",
            "incorrect": "#f8d7da",
            "unverifiable": "#fff3cd",
        }.get(verdict, "#f5f5f7")
        st.markdown(
            f"<div style='background:{color}; border-radius:8px; padding:0.6rem 0.8rem; "
            f"font-weight:600; text-align:center; margin-bottom:0.8rem'>{verdict}</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Judge — initial"):
            st.markdown(_extract_claim_section(data["judge_initial"], claim_number))
        with st.expander("Critic — flags"):
            st.markdown(
                _extract_claim_section(
                    data["critic"],
                    claim_number,
                    silent_msg="*This claim was not flagged by the Critic — initial verdict stands.*",
                )
            )
        with st.expander("Judge — final verdict"):
            st.markdown(
                _extract_claim_section(
                    data["judge_final"],
                    claim_number,
                    silent_msg="*This claim was not revised — verdict unchanged from initial Judge pass.*",
                )
            )

    with col_c:
        st.markdown("**Human label**")
        verdict = claim_row.get("human_label", "")
        label_text = verdict if verdict else "not labeled"
        color = {
            "correct": "#d4edda",
            "incorrect": "#f8d7da",
            "unverifiable": "#fff3cd",
        }.get(verdict, "#f5f5f7")
        st.markdown(
            f"<div style='background:{color}; border-radius:8px; padding:0.6rem 0.8rem; "
            f"font-weight:600; text-align:center; margin-bottom:0.8rem'>{label_text}</div>",
            unsafe_allow_html=True,
        )
        reasoning = claim_row.get("human_reasoning_label", "")
        if pd.notna(reasoning) and str(reasoning).strip():
            st.markdown(f"**Reasoning:** {reasoning}")


def render() -> None:
    _reports, eval_results, labels_df = _load_data()

    _render_pipeline()
    _render_kappa(labels_df)
    st.markdown("---")
    _render_claim_table(labels_df)
    st.markdown("---")
    _render_example(eval_results, labels_df)
