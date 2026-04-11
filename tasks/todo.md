# BYOS: Project Todo

## Phase 1 — Data Ingestion & EDA
- [ ] Download Geolife dataset via kagglehub
- [ ] Verify folder structure (182 users, ~69 with labels.txt)
- [ ] Parse .plt trajectory files into DataFrame
- [ ] Parse labels.txt files for labeled users
- [ ] EDA: trip count, duration, distance distributions
- [ ] EDA: class distribution across labeled users
- [ ] EDA: GPS sampling rate consistency

## Phase 2 — Feature Engineering
- [ ] Implement time-based windowing (sliding windows over trajectories)
- [ ] Compute heuristic features per window:
  - Speed (mean, max, std)
  - Acceleration / jerk
  - Stop density (% time < 1 m/s)
  - Bearing variance
  - Distance covered
- [ ] Align windows to mode labels
- [ ] Train/test split (subject-independent — held-out users)

## Phase 3 — Triage Agent
- [ ] Define anomaly rules (impossible speeds, GPS dropouts, label conflicts)
- [ ] Build rule-based baseline anomaly detector
- [ ] Build LLM Triage Agent (audits label quality, surfaces root causes)
- [ ] Compare: agent vs rule-based on hand-labeled anomaly set
- [ ] Quantify: % of emission-estimate variance from label noise

## Phase 4 — GBDT Classifier
- [ ] Train LightGBM on engineered features
- [ ] Subject-independent cross-validation
- [ ] Feature importance analysis
- [ ] Calibration check (confidence vs accuracy)
- [ ] Confusion matrix by mode class

## Phase 5 — Emissions Attribution
- [ ] Implement emission factor lookup (EPA/EEA/DEFRA)
- [ ] Per-trip CO₂ estimation (distance × mode × factor)
- [ ] Per-user weekly/monthly footprint aggregation
- [ ] Counterfactual: "what if sub-3km car trips were biked?"

## Phase 6 — Judge + Critic Agent
- [ ] Generate sample sustainability insight reports per user
- [ ] Build Judge agent (verifies numeric claims against raw data)
- [ ] Build Critic agent (reviews Judge's reasoning, flags weak justifications)
- [ ] Hand-label a small eval set for human-agreement measurement
- [ ] Compare: single-shot LLM judge vs Judge+Critic loop

## Phase 7 — Dashboard & Analysis
- [ ] Notebook with full pipeline walkthrough
- [ ] Charts: mode distribution, emissions by user, counterfactual impact
- [ ] Written analysis with quantified agent value

## Phase 8 — Report & Website (optional)
- [ ] Host results as static site or Streamlit dashboard
- [ ] Write final report
