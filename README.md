# BYOB: Bring Your Own Bike

### Understanding How Apple ESG Features Could Reshape How We Move Using AI Agents

---

## Project Overview

An end-to-end agentic mobility pipeline built on the Microsoft Geolife GPS dataset:

1. **Classify** how people commute from raw GPS alone - no extra sensors, no user input
2. **Estimate** each user's carbon footprint by mode
3. **Generate** a personalised sustainability report using Claude, fact-checked by a Judge+Critic agent loop
4. **Simulate** how Apple ESG nudges could shift commute behaviour using LLM-powered agents
5. **Propose** three Apple product features grounded in the above work

---

## Project Structure

```
geolife-human-psychology/
│
├── dashboard.py                  # Streamlit app entry point
├── requirements.txt
│
├── dashboard_utils/              # One module per dashboard page
│   ├── hypothesis.py             # Hypothesis page
│   ├── data_analysis.py          # EDA page
│   ├── modeling.py               # Commute Mode Predictor page
│   ├── agentic_eval.py           # Agentic Evaluation vs Human page
│   ├── agentic_simulation.py     # Agentic Behaviour Simulation page
│   ├── products.py               # Proposed Apple Products page
│   ├── references.py             # References page
│   └── theme.py                  # Apple-style CSS theme
│
├── eda/                          # Notebooks (analysis only - no rerun needed)
│   ├── analysis.ipynb            # EDA: data quality, mode distribution, emissions
│   ├── modeling.ipynb            # Feature engineering + LightGBM classifier
│   ├── agentic_evaluation.ipynb  # Report generation + Judge+Critic loop
│   └── agentic_simulation.ipynb  # 5-level Apple nudge simulation (525 Claude calls)
│
├── data/
│   └── processed/
│       └── features.parquet      # Cached feature dataset (built by modeling.ipynb)
│
├── models/
│   └── lgbm_mode_classifier.pkl  # Trained LightGBM model + label encoder
│
└── reports/                      # All generated outputs
    ├── user_reports.json         # Claude-generated sustainability reports
    ├── eval_results.json         # Judge+Critic + baseline verdicts
    ├── human_labels.csv          # Hand-labeled claims for Cohen's κ
    ├── simulation_index.json     # Simulation sweep metadata
    └── sim_strength_X.XX.json    # One file per nudge level (0.00 – 1.00)
```

---

## Hypotheses

| ID | Hypothesis |
|---|---|
| **H2** | Using only GPS data - no phone sensors, no user input - we can train a model to tell apart walking, biking, taking the bus, and driving. |
| **H3** | Cars produce far more CO₂ than their share of trips would suggest. Swapping short car trips (under 3 km) to bike would eliminate a measurable chunk of each user's carbon footprint. |
| **H4** | Running a 3-step pipeline - Judge to Critic to Revised Judge - produces fact-check verdicts that match human labels more closely than a simple single-prompt judge, measured on 28 claims hand-labeled across 6 user reports using Cohen's κ. |
| **H5** | Simulating 15 agentic commuters over 7 days shows that stronger Apple ESG nudges (quiet route suggestion to leaderboard to real-time Watch alert) progressively reduce car usage, lower CO2 emissions, and improve population health and mood - with the effect growing at each feature level. |

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_key_here
```

### 3. Download the dataset

Run the first cell of `eda/analysis.ipynb` - it downloads the Geolife dataset via `kagglehub` automatically.

### 4. Run the notebooks (optional - outputs already saved)

The `reports/` and `models/` directories are pre-populated. You only need to rerun notebooks if you want to regenerate results from scratch:

```
eda/analysis.ipynb          : EDA (no outputs saved, read-only)
eda/modeling.ipynb          : builds features.parquet + lgbm_mode_classifier.pkl
eda/agentic_evaluation.ipynb: builds user_reports.json + eval_results.json
eda/agentic_simulation.ipynb: builds sim_strength_*.json (costs ~$0.08 in API calls)
```

### 5. Launch the dashboard

```bash
streamlit run dashboard.py
```

The dashboard runs at `http://localhost:8501`.

---

## Docker

```bash
# Build
docker build -t geolife-dashboard .

# Run
docker run -p 8501:8501 \
  -e ANTHROPIC_API_KEY=your_key_here \
  -v ~/.cache/kagglehub:/root/.cache/kagglehub \
  geolife-dashboard
```

Open `http://localhost:8501`.

> **Notes:**
> - `-e ANTHROPIC_API_KEY` - never bake the key into the image
> - `-v ~/.cache/kagglehub` - mounts the local dataset cache so the EDA tab works; omit this flag if you only need the other tabs

---

## Results Summary

| Hypothesis | Result |
|---|---|
| H2 - Mode classification | CV macro-F1 = **0.553 ± 0.037** across 5 folds (subject-independent) |
| H3 - Emissions attribution | Car = 24% of distance but 52% of CO₂. Sub-3km car: bike saves a measurable share of per-user emissions |
| H4 - Agentic evaluation | Judge+Critic loop (κ) vs. single-prompt baseline - see **Agentic Evaluation** tab |
| H5 - Behaviour simulation | Car usage and CO₂ dropped at every nudge level; leaderboard level produced the biggest shift |

---

## Proposed Apple Products

| Feature | What it does |
|---|---|
| **Commute Copilot** | On-device trip classification + weekly Claude-generated carbon report, fact-checked by Judge+Critic |
| **Carbon Ring** | A new Apple Health ring that rewards low-carbon travel with points for carbon offsets or Watch features |
| **Apple Green Impact** | Privately aggregated user footprints: stronger Apple ESG score: broader investor base |
