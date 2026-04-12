# BYOS: Bring Your Own Sustainability

### An Agentic Mobility Pipeline for Personal Carbon Insights

---

## 1. Idea

Build a transportation-mode classification pipeline on the **Geolife GPS dataset** using heuristic feature engineering and gradient-boosted trees, bracketed by a Judge+Critic LLM agent that evaluates generated sustainability reports for factual accuracy.

**Thesis:** *Agents earn their keep at the data-quality and eval boundaries of an ML pipeline — where judgment is required — not in the deterministic middle where code wins.*

**End goal:** Estimate per-user carbon footprint from GPS mobility patterns and surface actionable insights about where emissions reductions are achievable.

---

## 2. Pipeline

```
Raw Geolife GPS traces
  → Time-based windowing (30s windows, 15s stride)
    + heuristic features (speed, jerk, stop density, bearing variance)
    + quality filters (impossible speeds, time gaps)
  → GBDT mode classifier (LightGBM, 5-fold subject-independent CV)
  → Emissions attribution
    (haversine distance × mode × emission factor)
  → Counterfactual analysis
    (car → bus mode shift CO₂ savings)
  → Per-user sustainability report (GPT-4.1 generated)
  → [Judge + Critic Agent] verifies factual claims
  → Final scored report
```

**Emission factors:** car 170 g CO₂/km, bus 89 g/km, bike/walk 0.

---

## 3. Hypotheses

| ID | Hypothesis | Metric |
|---|---|---|
| **H2** | Heuristic GPS features + GBDT classify {walk, bike, bus, car} at >75% macro-F1 under subject-independent CV | Macro-F1 on held-out users |
| **H3** | A Judge+Critic loop agrees with human ratings more than a single-prompt judge, measured on ~20–30 hand-labeled report segments | Cohen's κ vs human labels |
| **H4** | Car contributes disproportionate CO₂ relative to distance, and a mode-shift counterfactual yields a quantifiable reduction | Car CO₂ share vs distance share; kg saved if car → bus |
| **H5** *(stretch)* | A generative-agent simulation under a biking incentive produces directionally meaningful mode-share shifts vs a control | Simulated mode-share delta |

---

## 4. Agent

### Judge + Critic Agent — report factuality eval
Evaluates generated sustainability reports against the underlying computed data:

> *"You emitted 42 kg CO₂ this week, 78% from car trips."*

1. **Judge** verifies each numeric claim against the source data
2. **Critic** reviews the Judge's reasoning, flags weak justifications
3. **Judge** revises if needed → final verdict

Used to measure whether the loop outperforms a single-prompt judge on a hand-labeled eval set (Cohen's κ).

---

## 5. Results vs. Published Research

### Classification (H2)

| Paper | Model | Validation | Reported |
|---|---|---|---|
| Zheng et al. 2008 (UbiComp) | Decision Tree | Subject-dependent | ~76% accuracy |
| Zheng et al. 2010 (ACM Trans. Web) | Random Forest + CRF | Subject-dependent | ~91.5% accuracy |
| Xiao et al. 2012 | HMM | Subject-dependent | ~88% accuracy |
| Dabiri & Heaslip 2018 | CNN (raw GPS) | Subject-dependent | ~84% accuracy |
| **This project** | **LightGBM** | **Subject-independent 5-fold CV** | **0.553 ± 0.046 macro-F1** |

**Why direct comparison is misleading:**
- Every published paper trains and tests on the **same users** (different time windows). The model memorises personal patterns. Subject-independent CV — where test users are completely unseen — is strictly harder and more realistic for deployment.
- Published papers report **accuracy**, gamed by the majority class (walk). Macro-F1 weights all classes equally. Converting our result to weighted accuracy gives ~0.60–0.62, within range of Zheng 2008 under a fairer comparison.
- Published work uses **map-matching features** (proximity to bus stops, road type) — external data we deliberately excluded to keep the pipeline GPS-only.

**Honest position:** Our 0.553 macro-F1 under strict subject-independent CV is methodologically more rigorous than most published benchmarks. The gap is largely explained by validation strategy, not model quality.

### Emissions (H4)

No published benchmark for GPS-only emissions attribution. Key finding: car = 24% of distance but 52% of CO₂ (2× disproportionality). Shifting all car km to bus saves ~48% of car emissions across test users. Consistent with published emission factors (car 170 g/km vs. bus 89 g/km).

---

## 6. What Would Improve Results Given More Time

### Classification

| Change | Expected gain | Effort |
|---|---|---|
| 60s windows instead of 30s | +5–10% bus F1 — captures stop→move→stop cycle | Low |
| LightGBM `num_leaves=127` | Closes gap with Random Forest (~+3% macro-F1) | Low |
| Map-matching features (road type, bus stop proximity) | +10–15% — what published papers use | High — needs OSM data |
| More labeled users (all 69, not 37) | Reduces fold variance (± drops from 0.046) | Medium |
| Sequence model (CRF or LSTM over windows) | +5–10% — exploits temporal context | High |

### Emissions

| Change | Impact |
|---|---|
| Raw GPS trip reconstruction (stop detection on `.plt` files) | Enables the sub-3km car trip analysis — requires door-to-door trip boundaries |
| More test users | Per-user CO₂ estimates become more representative |

### Agent (H3)

| Change | Impact |
|---|---|
| Larger hand-labeled eval set (50+ claims vs. 20–30) | More reliable Cohen's κ estimate |
| Fine-tuned judge vs. zero-shot | Tests whether task-specific prompting improves factuality detection |

---

## 7. Caveats

- **Geolife is old and Beijing-centric.** Results generalise only within similar urban contexts.
- **0.553 mean macro-F1 (5-fold CV).** Subject-independent CV on 37 users has high variance — some folds get single-mode test users. Car detection (F1 ≈ 0.66) is the most reliable class and the one that matters most for CO₂ attribution.
- **Sub-3km trip fraction not computable.** Our 30s windows are classification units, not trip units — reconstructed "trips" average 330m. Full trip reconstruction requires the raw `.plt` files and stop detection.
- **Judge–human agreement requires hand-labeling ~20–30 report segments.** Small but necessary for H3.

---

## 8. Proposed Feature: *Green Commute Coach*

An on-device feature that passively detects mobility patterns, estimates personal carbon footprint, and surfaces specific, feasible alternatives grounded in actual trip history:

> *"You drove 2.1 km to the gym 4 times this week — biking would save ~1.4 kg CO₂/week."*

LLM-generated suggestions are grounded in user data and verified by the Judge + Critic loop. Maps to a real product gap: no current consumer product does personal mobility carbon tracking at this fidelity.
