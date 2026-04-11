# BYOS: Bring Your Own Sustainability

### An Agentic Mobility Pipeline for Personal Carbon Insights

---

## 1. Idea

Build a transportation-mode classification pipeline on the **Geolife GPS dataset** using heuristic feature engineering and gradient-boosted trees, and bracket the ML model with two LLM agent systems:

- A **Data-Quality Triage Agent** that audits inputs (label noise, sensor anomalies, suspicious windows) before training.
- A **Judge + Critic Agent** that rigorously evaluates downstream outputs (sustainability insight reports) with self-critique to improve reliability.

**Thesis:** *Agents earn their keep at the data-quality and eval boundaries of an ML pipeline — where judgment is required and rules are brittle — not in the deterministic middle where code wins.*

**End goal:** Estimate per-trip and per-user carbon footprint from mobility patterns and surface actionable insights about where emissions reductions are achievable. Mode classification is the *means*; you can't attribute CO₂ without knowing whether someone drove, biked, or took the bus.

---

## 2. Pipeline

```
Raw Geolife traces
  → Time-based windowing + heuristic features
    (speed, jerk, stop density, bearing variance)
  → [Triage Agent] audits labels & flags anomalies
  → GBDT mode classifier (subject-independent CV)
  → Emissions attribution layer
    (distance × mode × emission factor — EPA / EEA / DEFRA)
  → Per-user & aggregate footprint analytics
  → Counterfactual analysis
    ("what if sub-3km car trips were biked?")
  → Sustainability insight reports
  → [Judge + Critic Agent] evaluates factuality, completeness, grounding
  → Final report + proposed feature
```

**Reference emission factors (illustrative):** ~170 g CO₂/km for an average car, ~100 g CO₂/km per passenger for bus, ~0 for walk/bike.

---

## 3. Agents

### 3.1 Triage Agent — input quality control
Audits data quality with a sustainability lens. A mislabeled walk-as-car trip inflates the carbon estimate significantly, so label quality directly affects the headline metric.

> **Quantifiable finding:** *"X% of emission-estimate variance traces to label noise in the training set."*

### 3.2 Judge + Critic Agent — output quality control
Evaluates per-user sustainability insight reports such as:

> *"You emitted 42 kg CO₂ this week, 78% from car trips, 31% of which were under 3km and bike-replaceable."*

The Judge verifies every numeric claim against the underlying data (did the model actually see 31%? is the 3km threshold applied correctly?). The Critic reviews the Judge's reasoning and flags weak justifications, prompting revision.

This directly addresses the JD's *"information and content integrity"* bullet — hallucinated sustainability numbers are a real reputational risk for any company shipping climate features.

### 3.3 Counterfactual Analyst — stretch agent
Given a user's trip history, an agent explores "what-if" scenarios using tools: filter trips by distance, recompute emissions under alternative mode assignments, check feasibility via reverse-geocoded terrain/weather. It reports the highest-leverage behavior changes. Branching logic is real here — the analyst decides which counterfactuals are worth exploring based on what it finds.

---

## 4. Why This Solves a Real Business Problem

This project maps directly onto the three pillars the hiring manager described:

| Team pillar | Project component |
|---|---|
| Heuristic feature collection | Windowing + hand-crafted GPS features |
| Traditional ML training | GBDT mode classifier, subject-independent CV |
| LLM-centric eval (LLM-as-judge) | Judge + Critic loop with measured human agreement |

Beyond pillar alignment, the agents address two unsolved industry pain points:

- **Label noise and data quality** is the #1 bottleneck on sensor-ML teams. A triage agent producing auditable root-cause reports is exactly the *"tools to interrogate data in depth"* the JD asks for.
- **LLM eval reliability** is an open problem. Single-shot LLM-as-judge has known failure modes (sycophancy, shallow rubric application, inconsistent scoring). A Judge + Critic loop with measured human-agreement lift is a methodologically honest contribution, not a demo.

---

## 5. What I Bring to the Table

- **A coherent thesis, not a tech demo.** A defensible principle for *when* to use agents (judgment-heavy boundaries) and when not to (deterministic middle). Signals senior judgment.
- **Quantified agent value.** Each agent is compared against a non-agentic baseline (single-prompt judge, rule-based anomaly detection) with reported deltas. Most portfolio projects assume agents are better; this one measures.
- **Methodological rigor.** Subject-independent splits, calibration, label-quality audits, judge–human agreement on a hand-labeled subset. Echoes my BlackRock validation-automation background.
- **Artifacts, not just metrics.** Triage reports with citations, judge rationales with evidence, a Geolife data-quality audit. Concrete deliverables that mirror real internal team outputs.
- **Direct pillar alignment.** Heuristic features → GBDT → LLM-as-judge with self-critique.

---

## 6. Why This Idea Matters

> *On sensor-data ML teams, the hardest problems aren't in model training — they're in knowing whether your labels are trustworthy going in and whether your generative outputs are trustworthy coming out. This project demonstrates that lightweight agent systems, applied surgically at those two boundaries, can measurably improve both — without replacing the deterministic ML core that should stay deterministic.*

---

## 7. Risks & Honest Caveats

- **Geolife is old, Beijing-centric, and label-sparse.** Generalization claims need explicit caveats.
- **Judge–human agreement requires hand-labeling a small eval set.** Budget time for this — it's the most important number in the report.
- **Agent value must be proven, not asserted.** If the triage agent doesn't beat a rule-based anomaly detector, report it honestly. Negative results handled well are a senior signal.

---

## 8. Proposed Feature: *Green Commute Coach*

**Primary framing — ESG measurement, Apple as a better company.**

An on-device feature that passively detects mobility patterns, estimates personal mobility carbon footprint with privacy-preserving aggregation, and proactively surfaces specific, feasible alternatives grounded in actual trip history:

> *"You drove 2.1 km to the gym 4 times this week — biking would save ~1.4 kg CO₂/week and add ~6 minutes per trip."*

LLM-generated suggestions are grounded in user data and evaluated by the Judge + Critic loop for factual accuracy.

**Why it fits Apple.** No current Apple product does personal mobility carbon tracking. Apple Maps added cycling routes in iOS 14, and Apple's sustainability work is entirely *corporate-side* (supply chain, manufacturing, Restore Fund). User-side personal footprint tracking is a real product gap. Framing: *"Apple enables users to measure and improve their mobility footprint, and reports the aggregate impact as part of its sustainability story."*

The technical work (classification → emissions → privacy-preserving aggregation) maps perfectly, and the feature hits multiple JD bullets at once: on-device sensors, motion understanding, ML + LLM, privacy-preserving aggregation, sustainability, immersive/intelligent experiences.

### Secondary — Gamified behavior change
Visible rewards move the needle on sustainable choices. Defensible framing: points unlock *features* or fund *verified offsets / tree planting*, or recommend lower-impact alternatives to things the user already does (e.g., a greener route to an existing destination). **Lead with behavior change, not commerce** — anything resembling "buy more stuff to offset your footprint" reads as greenwashing.

---

## 9. Extension: Generative-Agent Policy Simulation

Drawing on Park et al. 2023 (*Generative Agents: Interactive Simulacra of Human Behavior*), simulate 10–20 LLM-driven agents in a single neighborhood under one policy intervention (e.g., *"biking under 3km earns 2× points"*) and measure simulated behavior change over N days.

**Epistemic guardrails:**
- Simulated agents are *not* evidence about real human outcomes — they're evidence about what an LLM thinks humans would do.
- Frame as *"exploring policy design space"* or *"generating hypotheses for real-world study,"* never *"predicting health impact."*
- Belongs in an **Extensions** section, positioned as *"if given more time, I would extend the analysis with agent-based simulation to explore policy design space before real-world piloting."*

---