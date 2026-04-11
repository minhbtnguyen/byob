# BYOS: Tries and Lessons

A running log of what we built, what broke, and what we learned.

---

## Phase 1 — Dataset & EDA

### What we did
- Downloaded Geolife dataset via `kagglehub` (182 users, 69 labeled, Beijing 2007–2012)
- Built `eda/analysis.ipynb` with 5 EDA sections: dataset health, mode distribution, speed profiles, temporal patterns, emissions preview
- Identified 4 usable modes: walk, bike, bus, car. Merged taxi→car, dropped subway/boat/airplane (too rare)

### What we learned
- Only 69/182 users have mode labels — the rest are unlabeled GPS traces, unusable for supervised classification
- GPS sampling rate is noisy: median ~1s but long tail up to 60s+ gaps. Gaps within a window corrupt speed/accel features
- User 068 had 51% of all windows in the initial build — one heavy user can dominate the entire dataset
- Subway and taxi together were <2% of windows — not worth modeling as separate classes

---

## Phase 2 — Feature Engineering (`eda/features.py`)

### What we did
- 30s sliding windows, 15s stride (50% overlap) over labeled intervals only
- 9 heuristic features: speed_mean/max/std, accel_mean/std, jerk_mean, stop_ratio, bearing_variance, distance_total_m
- Parquet cache at `data/processed/features.parquet`
- Quality filters: <5 points → drop, speed >250 km/h → drop, internal gap >15s → drop

### What broke and how we fixed it

**Zero windows for users 010/020/021**
- Cause: label assignment required full containment (window must be 100% inside a label interval). Label intervals shorter than the window = no matches.
- Fix: switched to majority overlap (≥50% of window covered by the label interval).

**Processing took 35+ min and stalled after user 161**
- Cause 1: sliding windows over the full recording span (years of GPS) instead of just labeled intervals — millions of empty iterations.
- Fix 1: iterate only over label intervals.
- Cause 2: user 163 had 3,112 label rows. `assign_label()` was called per window, doing a pandas scan over all 3,112 labels each time — O(n_windows × n_labels) = ~486M ops before hitting the 5K cap.
- Fix 2: since we are already sliding within a known label interval, use `label_row["mode"]` directly — eliminated the `assign_label()` call entirely.
- Cause 3: `multiprocessing.Pool` blocks until all workers complete, so no per-user progress appeared.
- Fix 3: removed multiprocessing, used sequential tqdm loop.

**User 068 dominated 51% of dataset**
- Fix: added `MAX_WINDOWS_USER = 5000` cap with early break.

**Coverage <5% filter too aggressive (kept only 15 users)**
- Fix: replaced percentage threshold with `MIN_WINDOWS_USER = 50` absolute count.

### Final dataset stats
- 130,343 windows across 37 users
- Mode split: walk 43.8%, bus 25%, car 18.3%, bike 12.9%
- 22/37 users hit the 5,000-window cap

---

## Phase 3 — ML Pipeline (`eda/pipeline.ipynb`)

### What we built
- Section 1: Feature engineering analysis — dataset overview, windows-per-user chart, feature box plots by mode, written observations
- Section 2: Classifier — baselines, model comparison, feature selection, fixed params, 5-fold CV, confusion matrix, SHAP
- Section 3: Emissions attribution (H4) — CO₂ per trip, sub-3km car counterfactual
- Section 4: Sustainability report generation (Claude API, stubbed)

### Model comparison results (macro-F1, subject-independent CV)

| Model | Macro-F1 |
|---|---|
| Baseline (most_frequent) | 0.174 |
| Baseline (stratified) | 0.238 |
| KNN | 0.553 |
| LightGBM (default) | 0.562 |
| Logistic Regression | 0.577 |
| Random Forest | 0.658 |
| **LightGBM 5-fold CV mean** | **0.553 ± 0.046** |

### What we learned

**Random Forest beat LightGBM at default settings**
- LightGBM with `num_leaves=31` was underfitting. RF's deeper trees captured non-linear interactions better.
- Fix: LightGBM needs `num_leaves=127+` to compete. At defaults, RF is the safer choice.

**Grid search was removed**
- 12 combinations × 3-fold GroupKFold on 130K rows was too slow to run interactively.
- Default params + documented rationale is sufficient for a portfolio project.

**The test split was skewed (walk 53% test vs 41% train)**
- Cause: `GroupShuffleSplit` picks random users — if a walk-heavy user lands in test, the whole split is unbalanced.
- Fix attempt: `StratifiedGroupKFold` — stratifies by row labels while keeping users together.
- Result: distribution still skewed because individual users are single-mode. No split algorithm can fix a dataset where user 067 is 93% walk and user 115 is 99% car.
- Real fix: **5-fold CV, report mean ± std**. CV macro-F1 = 0.553 ± 0.046. The variance is from user composition per fold, not model instability.

**The H2 target of >0.80 macro-F1 is unrealistic for this setup**
- Subject-independent CV on 37 users with single-mode test users makes 0.80 unachievable.
- Revised target: 0.75. Literature values for GPS mode detection with heuristic features and independent CV sit at 0.70–0.75.
- The model consistently crushes both baselines across all folds (+0.38 over most_frequent).

**Bus is the hardest class (F1 ≈ 0.35–0.48 across folds)**
- Bus shares high `stop_ratio` with walk (bus stops) and high speed with car (between stops).
- A 30s window catches a bus either stopped or cruising — each looks like a different mode.
- Potential fix: 60s windows to capture the stop→accelerate→cruise→stop pattern.

**Car detection is the strongest (F1 ≈ 0.66)**
- High speed is a clean signal. Good news: car is the high-CO₂ mode, so emissions attribution is reliable where it matters most.

**Feature selection found nothing to drop**
- All 9 features cleared the 5% importance threshold — none were truly redundant from LightGBM's perspective.
- `distance_total_m` is correlated with `speed_mean` but the model uses it anyway for edge cases.

---

---

## Phase 4 — Emissions Attribution (H4)

### What we did
- Computed CO₂ per window: `distance_total_m × 0.5 (overlap correction) × emission_factor`
- Produced per-user CO₂ breakdown by mode
- Computed car → bus counterfactual

### What broke

**Sub-3km trip fraction returned 100%**
- Cause 1 (window level): Every 30s window covers at most ~2km even at highway speed. All windows trivially pass the <3km filter — the check is meaningless at window granularity.
- Cause 2 (trip level): Reconstructing trips by grouping consecutive same-mode windows produces "trips" averaging ~330m. These are labelled recording segments, not door-to-door journeys. The 3km threshold still can't be applied.
- Fix: abandoned sub-3km analysis. Replaced with mode-shift counterfactual (car → bus).

**Distance double-counting from overlapping windows**
- Cause: 30s windows with 15s stride → 50% overlap. Summing `distance_total_m` across all windows counts each 15s of real travel twice.
- Fix: multiply `dist_km` by 0.5 before aggregating.

### H4 results
- Car = 23.8% of distance, 52.3% of CO₂ — **2× disproportionality**
- Shifting all car km to bus would save ~140 kg CO₂ (~48% reduction) across 6 test users
- Per-user car CO₂ share: 32–67% — high variance supports personalised reports
- Sub-3km trip fraction: **not computable** from window-level features. Requires raw GPS trip reconstruction with stop detection on `.plt` files.

---

## Open Questions / Next Steps

- [ ] Implement Judge+Critic agent (H3) in `agents/`
- [ ] Generate sustainability reports via GPT-4.1 API
- [ ] Run H3 eval: extract claims, hand-label, compute Cohen's κ
- [ ] Try 60s windows — likely improves bus F1
- [ ] Sub-3km trip reconstruction from raw `.plt` files (future work)
