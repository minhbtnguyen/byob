import streamlit as st


def render() -> None:
    st.markdown("### Primary Hypothesis")
    st.info(
        "**H1:** Using only GPS data - no extra sensors, no user input - we can classify how people "
        "commute, estimate their carbon footprint, and generate a personalised sustainability report "
        "verified by an AI agent pipeline. Layered on top, a simulation of agentic commuters shows "
        "that Apple ESG nudges can measurably shift behaviour, reduce emissions, and improve health - "
        "forming an end-to-end system from raw GPS to human behaviour change."
    )

    st.markdown("### Sub-Hypotheses")

    with st.expander("H2: Commute Mode Predictor"):
        st.markdown(
            "Using only GPS data - no phone sensors, no user input - we can train a model to tell apart "
            "walking, biking, taking the bus, and driving."
        )
        st.caption("Analysis is in **Commute Mode Predictor** tab.")

    with st.expander("H3: Emissions Attribution"):
        st.markdown(
            "Cars produce far more CO₂ than their share of trips would suggest. "
            "Swapping short car trips (under 3 km) to bike would eliminate a measurable chunk of each user's carbon footprint."
        )
        st.caption("Analysis is in **EDA** tab.")

    with st.expander("H4: Agentic Evaluation"):
        st.markdown(
            "Running a 3-step pipeline - Judge to Critic to Revised Judge - produces fact-check verdicts "
            "that match human labels more closely than a simple single-prompt judge, "
            "measured on 28 claims hand-labeled across 6 user reports using Cohen's κ."
        )
        st.caption("Analysis is in **Agentic Evaluation vs Human Evaluation** tab.")

    with st.expander("H5: Human Behaviour Simulation"):
        st.markdown(
            "Simulating 15 agentic commuters over 7 days shows that stronger Apple ESG nudges "
            "(quiet route suggestion to leaderboard to real-time Watch alert) progressively reduce "
            "car usage, lower CO2 emissions, and improve population health and mood - "
            "with the effect growing at each feature level."
        )
        st.caption("Analysis is in **Agentic Behaviour Simulation** tab.")

    with st.expander("Extra: Potential Apple Products"):
        st.markdown(
            "Based on the analyses above, we propose three Apple features that put this pipeline into real products\n\n"
        )
        st.caption("Analysis is in **Proposed Apple Products** tab.")
