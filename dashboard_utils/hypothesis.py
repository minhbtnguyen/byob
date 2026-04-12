import streamlit as st


def render() -> None:
    st.markdown("### Primary Hypothesis")
    st.info(
        "**H1:** Personal mobility carbon footprints can be reliably estimated from raw GPS traces alone - "
        "without IMU or user input - by combining heuristic feature engineering, gradient-boosted mode "
        "classification, and a Judge+Critic LLM agent applied at the evaluation boundary of the pipeline. "
        "The agentic layer measurably improves output factual reliability over a single-prompt baseline, "
        "while the deterministic ML core remains unchanged."
    )

    st.markdown("### Sub-Hypotheses")

    with st.expander("H2: Classification"):
        st.markdown(
            "Heuristic GPS features + GBDT classify {walk, bike, bus, car} at >75% macro-F1 "
            "under subject-independent CV."
        )
        st.caption("Metric: Macro-F1 on held-out users")

    with st.expander("H3: Emissions"):
        st.markdown(
            "Car contributes disproportionate CO₂ relative to distance travelled, "
            "and a mode-shift counterfactual (car to bus) yields a quantifiable reduction."
        )
        st.caption(
            "Metric: Car CO₂ share vs. distance share; kg CO₂ saved if car to bus"
        )

    with st.expander("H4: Agent Evaluation"):
        st.markdown(
            "A Judge+Critic LLM eval loop agrees with human ratings more than a single-prompt judge baseline, "
            "measured on ~20–30 hand-labeled report segments."
        )
        st.caption("Metric: Cohen's κ vs. human labels")

    with st.expander("H5: Human Psychology Simulation"):
        st.markdown(
            "A generative-agent simulation of 10–20 LLM personas under a biking incentive produces "
            "directionally meaningful mode-share shifts versus a control. "
        )
        st.caption("Metric: Simulated mode-share delta")
