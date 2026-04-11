# BYOS: Bring Your Own Sustainability

### An Agentic Mobility Pipeline for Personal Carbon Insights

---

## 1. Idea

Build a transportation-mode classification pipeline on the **Geolife GPS dataset** using heuristic feature engineering and gradient-boosted trees, bracketed by two LLM agent systems:

- A **Judge + Critic Agent** that evaluates generated sustainability reports for factual accuracy.

**Thesis:** *Agents earn their keep at the data-quality and eval boundaries of an ML pipeline — where judgment is required — not in the deterministic middle where code wins.*

**End goal:** Estimate per-user carbon footprint from GPS mobility patterns and surface actionable insights about where emissions reductions are achievable.

---

## 2. Pipeline

```
Raw Geolife GPS traces
  → Time-based windowing + heuristic features
    (speed, jerk, stop density, bearing variance)
  → Data quality checks (impossible speeds, label misalignment)
  → GBDT mode classifier (LightGBM, subject-independent CV)
  → Emissions attribution
    (haversine distance × mode × emission factor)
  → Counterfactual analysis
    ("what if sub-3km car trips were biked?")
  → Per-user sustainability report (LLM-generated)
  → [Judge + Critic Agent] verifies factual claims
  → Final scored report
```

**Emission factors (illustrative):** car 170 g CO₂/km, bus 89 g/km, subway 41 g/km, bike/walk 0.

---

## 3. Hypotheses

| ID | Hypothesis | Metric |
|---|---|---|
| **H2** | Heuristic GPS features + GBDT classify {walk, bike, bus, car} at >80% macro-F1 under subject-independent CV | Macro-F1 on held-out users |
| **H3** | A Judge+Critic loop agrees with human ratings more than a single-prompt judge, measured on ~20–30 hand-labeled report segments | Cohen's κ vs human labels |
| **H4** | A non-trivial fraction of car trips are under 3 km and bike-replaceable, yielding a quantifiable CO₂ reduction | % car trips <3km + kg CO₂/user saved |
| **H5** *(stretch)* | A generative-agent simulation under a biking incentive produces directionally meaningful mode-share shifts vs a control | Simulated mode-share delta |

---

## 4. Agent

### Judge + Critic Agent — report factuality eval
Evaluates generated sustainability reports against the underlying computed data:

> *"You emitted 42 kg CO₂ this week, 78% from car trips, 31% of which were under 3km."*

1. **Judge** verifies each numeric claim against the source data
2. **Critic** reviews the Judge's reasoning, flags weak justifications
3. **Judge** revises if needed → final verdict

Used to measure whether the loop outperforms a single-prompt judge on a hand-labeled eval set.

---

## 5. Caveats

- **Geolife is old and Beijing-centric.** Generalization claims need explicit scope limits.
- **Judge–human agreement requires hand-labeling ~20–30 report segments.** Small but necessary.
- **H5 is stretch.** Cut if time runs short — H2–H4 form the complete report spine.

---

## 6. Proposed Feature: *Green Commute Coach*

An on-device feature that passively detects mobility patterns, estimates personal carbon footprint, and surfaces specific, feasible alternatives grounded in actual trip history:

> *"You drove 2.1 km to the gym 4 times this week — biking would save ~1.4 kg CO₂/week."*

LLM-generated suggestions are grounded in user data and verified by the Judge + Critic loop. Maps to a real product gap: no current consumer product does personal mobility carbon tracking at this fidelity.
