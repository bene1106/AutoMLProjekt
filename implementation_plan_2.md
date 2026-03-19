# Implementation Plan — Week 2: Regime Classifier Deep-Dive & Diagnostics

## Context

Milestone 1 (Reflex Agent Baseline) is complete. Key findings:
- Regime Classifier: 84.5% accuracy (but potentially trivial due to VIX autocorrelation)
- Oracle Gap: -0.009 Sharpe (regime prediction is NOT the bottleneck)
- Reflex Agent loses to Equal Weight on 2021–2024 test set (Sharpe -0.29 vs +0.37)

This week we investigate WHY and build the analytical foundation for the professor meeting.

The existing codebase is in `regime_algo_selection/` with the structure from Milestone 1.

---

## Part 1: Regime Classifier Deep-Dive

### Task 1.1: Naive Baseline Measurement

**File:** `regimes/classifier.py` (extend existing) or new file `experiments/classifier_analysis.py`

**What to do:**
Measure the accuracy of the simplest possible "classifier": just use yesterday's regime label.

```
Function: naive_baseline_accuracy(regime_labels: pd.Series) -> dict
- Create predictions: s_hat_t = s*_{t-1} (shift regime_labels by 1)
- Compute on the SAME train/test split as the current classifier:
  - Overall accuracy
  - Per-regime accuracy (precision, recall, F1)
  - Confusion matrix
- Return dict with all metrics

Compare side-by-side:
| Method              | Overall Acc | Calm Recall | Normal Recall | Tense Recall | Crisis Recall |
|---------------------|-------------|-------------|---------------|--------------|---------------|
| Naive (yesterday)   | ???         | ???         | ???           | ???          | ???           |
| LogReg (current)    | 84.5%       | ...         | ...           | ...          | ...           |
```

**Key question this answers:** How much does the Logistic Regression classifier actually learn beyond "VIX doesn't change much day-to-day"?

---

### Task 1.2: Feature Ablation Study

**File:** `experiments/classifier_analysis.py`

**What to do:**
Train the regime classifier with progressively richer feature sets and measure accuracy for each.

```
Feature Sets to test:

Set A — "VIX Only" (current):
  - vix_prev, vix_change_1d, vix_change_5d, vix_ma5, vix_ma20, vix_std20, vix_relative
  
Set B — "VIX + Asset Returns":
  - Everything in Set A, plus:
  - Per-asset lagged returns: ret_1d, ret_5d, ret_20d for each of the 5 ETFs (15 features)

Set C — "VIX + Asset Returns + Volatilities":
  - Everything in Set B, plus:
  - Per-asset rolling volatility: vol_20d, vol_60d for each ETF (10 features)
  
Set D — "VIX + Asset Returns + Volatilities + Momentum":
  - Everything in Set C, plus:
  - Per-asset momentum: mom_20d, mom_60d, mom_120d for each ETF (15 features)

Set E — "Asset Features Only" (NO VIX):
  - Only the asset-derived features from Sets B/C/D (without any VIX features)
  - This answers: can you predict the regime WITHOUT looking at VIX?

Set F — "VIX + Cross-Asset Stress Signals":
  - Everything in Set A (VIX features), plus engineered cross-asset stress indicators:
  - SPY-TLT rolling correlation (20d): rises during crises (flight to safety)
  - SPY-GLD rolling correlation (20d): gold as safe haven signal
  - Cross-asset realized dispersion: std of daily returns across all 5 ETFs (high = stress)
  - SPY drawdown from 20d high: (SPY_price / SPY_20d_max) - 1
  - TLT-SPY return spread (1d): positive = flight to bonds
  - GLD-SPY return spread (1d): positive = flight to gold
  - Average absolute return across all ETFs (1d): high = market turbulence
  - All features lagged by 1 day
  - ~14 features total (7 VIX + 7 stress signals)
  - Rationale: these capture the MECHANISM behind regime changes (stress propagation
    across asset classes), not just VIX autocorrelation

Set G — "Kitchen Sink" (everything):
  - All features from Sets D + F combined
  - VIX features + asset returns + volatilities + momentum + cross-asset stress signals
  - ~54 features total
  - This is the upper bound — if this doesn't beat Set F significantly, more features don't help
```

**Implementation note for Set F features** — add to `data/features.py`:

```
Function: compute_cross_asset_features(prices: pd.DataFrame) -> pd.DataFrame
- Input: daily adjusted close prices for all assets
- Compute (all lagged by 1 day):
  - spy_tlt_corr_20d: rolling 20-day correlation of SPY and TLT daily returns
  - spy_gld_corr_20d: rolling 20-day correlation of SPY and GLD daily returns
  - cross_asset_dispersion: rolling std across all 5 ETF daily returns (per day)
  - spy_drawdown_20d: SPY price / 20-day rolling max price - 1
  - tlt_spy_spread_1d: TLT 1d return - SPY 1d return
  - gld_spy_spread_1d: GLD 1d return - SPY 1d return
  - avg_abs_return_1d: mean of |ret_1d| across all 5 ETFs
- Return: DataFrame with index=date, columns=feature names
- CRITICAL: all features must use only t-1 information (lagged!)
```

For each feature set:
- Train Logistic Regression on the training period (2006–2018)
- Evaluate on the test period (2021–2024)
- Record: overall accuracy, per-regime precision/recall/F1, confusion matrix

```
Output: Feature Ablation Results Table

| Feature Set              | # Features | Overall Acc | Calm   | Normal | Tense  | Crisis |
|--------------------------|------------|-------------|--------|--------|--------|--------|
| Naive Baseline           | 0          | ???         | ???    | ???    | ???    | ???    |
| A: VIX Only              | 7          | 84.5%       | ...    | ...    | ...    | ...    |
| B: + Asset Returns       | 22         | ???         | ???    | ???    | ???    | ???    |
| C: + Volatilities        | 32         | ???         | ???    | ???    | ???    | ???    |
| D: + Momentum            | 47         | ???         | ???    | ???    | ???    | ???    |
| E: Assets Only (no VIX)  | 40         | ???         | ???    | ???    | ???    | ???    |
| F: VIX + Stress Signals  | 14         | ???         | ???    | ???    | ???    | ???    |
| G: Kitchen Sink (all)    | 54         | ???         | ???    | ???    | ???    | ???    |
```

Save this table as CSV to `results/feature_ablation.csv` and as a formatted plot to `results/07_feature_ablation.png`.

---

### Task 1.3: Model Comparison

**File:** `experiments/classifier_analysis.py`

**What to do:**
Using the BEST feature set from Task 1.2, compare different classifier architectures.

```
Models to compare:

1. Logistic Regression (current baseline)
   - sklearn LogisticRegression with StandardScaler
   
2. Random Forest
   - sklearn RandomForestClassifier
   - n_estimators=200, max_depth=10
   
3. XGBoost
   - xgboost.XGBClassifier
   - n_estimators=200, max_depth=6, learning_rate=0.1
   
4. Gradient Boosting
   - sklearn GradientBoostingClassifier
   - n_estimators=200, max_depth=5

5. (Optional) LSTM / GRU
   - Only if time permits
   - Small network: input → LSTM(32) → Dense(4) → softmax
   - Use sequences of length 20 (last 20 days of features)
   - Train with cross-entropy loss
   - This captures temporal patterns that flat classifiers miss
```

For each model:
- Train on training period with the best feature set
- Evaluate on test period
- Record: accuracy, per-regime metrics, training time
- Also record: predicted probability calibration (are the confidence scores meaningful?)

```
Output: Model Comparison Table

| Model               | Accuracy | Calm F1 | Normal F1 | Tense F1 | Crisis F1 | Train Time |
|---------------------|----------|---------|-----------|----------|-----------|------------|
| Naive Baseline      | ???      | —       | —         | —        | —         | 0s         |
| Logistic Regression | 84.5%    | ...     | ...       | ...      | ...       | ...        |
| Random Forest       | ???      | ???     | ???       | ???      | ???       | ???        |
| XGBoost             | ???      | ???     | ???       | ???      | ???       | ???        |
| Gradient Boosting   | ???      | ???     | ???       | ???      | ???       | ???        |
```

Save to `results/model_comparison.csv` and `results/08_model_comparison.png`.

---

### Task 1.4: Confidence Analysis

**File:** `experiments/classifier_analysis.py`

**What to do:**
Analyze how classifier confidence relates to prediction quality.

```
For the best model from Task 1.3:

1. Confidence Distribution
   - For each prediction, record max(p_hat_t) as the confidence score
   - Plot histogram of confidence scores
   - Split by correct vs incorrect predictions
   - Question: Are high-confidence predictions actually more accurate?

2. Confidence-Accuracy Curve (Reliability Diagram)
   - Bin predictions by confidence level (e.g., 50-60%, 60-70%, ..., 90-100%)
   - For each bin: compute actual accuracy
   - Plot: expected confidence vs actual accuracy
   - Perfect calibration = diagonal line
   
3. Regime Transition Analysis
   - When does the classifier fail? 
   - Hypothesis: failures cluster around regime TRANSITIONS (day before/after VIX crosses a threshold)
   - Compute accuracy for "stable" days (regime unchanged for ±3 days) vs "transition" days
   - This is directly relevant to the professor's question about regime change prediction
```

```
Output:
- results/09_confidence_histogram.png
- results/10_reliability_diagram.png  
- results/11_transition_analysis.png
- Print: "Accuracy on stable days: X%, Accuracy on transition days: Y%"
```

---

## Part 2: Diagnostics & Quick Fixes

### Task 2.1: Per-Algorithm Analysis

**File:** `experiments/algorithm_analysis.py`

**What to do:**
Backtest every single one of the 48 Tier-1 algorithms individually on the test period (2021–2024).

```
For each algorithm A_k (k = 1, ..., 48):
- Run it standalone on the test period (it always produces its own weights, ignoring regimes)
- Compute: Sharpe, cumulative return, max drawdown, average turnover, switching cost
- All with kappa = 0.001

Output: Ranked table of all 48 algorithms

| Rank | Algorithm        | Family      | Sharpe | Cum Return | Max DD  | Turnover |
|------|------------------|-------------|--------|------------|---------|----------|
| 1    | ???              | ???         | ???    | ???        | ???     | ???      |
| 2    | ???              | ???         | ???    | ???        | ???     | ???      |
| ...  | ...              | ...         | ...    | ...        | ...     | ...      |
| 48   | ???              | ???         | ???    | ???        | ???     | ???      |

Also include Equal Weight in the ranking for reference.

Key questions this answers:
- Does ANY Tier-1 algorithm beat Equal Weight on 2021-2024?
- Which algorithm families work best in this period?
- Is the Reflex Agent's failure due to bad algorithm SELECTION or bad algorithm POOL?
```

Save to `results/algorithm_ranking.csv` and generate:
- `results/12_algorithm_ranking.png` (bar chart of Sharpe ratios, color-coded by family)
- `results/13_algorithm_families.png` (box plot of Sharpe by family)

---

### Task 2.2: Reflex Agent with Net-Sharpe Fitting

**File:** `agents/reflex_agent.py` (modify existing)

**What to do:**
The current ReflexAgent.fit() selects the best algorithm per regime by gross Sharpe ratio.
Add an option to fit by NET Sharpe (after switching costs).

```
Modify ReflexAgent.fit():
- Add parameter: metric="sharpe_net" (default, new) vs "sharpe_gross" (current behavior)
- When metric="sharpe_net":
  - For each algorithm in each regime, compute Sharpe AFTER subtracting switching costs
  - This penalizes high-turnover algorithms like Trend_L10_B1
  
Run both versions and compare:
- What mapping does the net-Sharpe agent choose? (probably different from current)
- Does it perform better on the test set?

Output:
| Regime  | Gross-Sharpe Pick  | Net-Sharpe Pick    |
|---------|--------------------|--------------------|
| Calm    | Trend_L10_B1       | ???                |
| Normal  | MinVar_L40         | ???                |
| Tense   | MaxDiv_L120        | ???                |
| Crisis  | RiskParity_L60     | ???                |

And performance comparison:
| Agent              | Sharpe | Cum Return | Turnover |
|--------------------|--------|------------|----------|
| Equal Weight       | +0.37  | +18.3%     | 0        |
| Reflex (gross fit) | -0.29  | -12.2%     | 226      |
| Reflex (net fit)   | ???    | ???        | ???      |
| Oracle             | -0.30  | -12.5%     | 238      |
```

---

### Task 2.3: Switching Cost Sensitivity Analysis

**File:** `experiments/sensitivity_analysis.py`

**What to do:**
Run the full pipeline (classifier + reflex agent + backtest) for multiple values of kappa.

```
kappa_values = [0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]

For each kappa:
- Re-fit the Reflex Agent (with net-Sharpe, so the mapping may change per kappa)
- Backtest on test period
- Record: Sharpe, cumulative return, total switching cost, turnover
- Also record: which algorithm was selected per regime (does mapping change with kappa?)

Output:
| kappa  | Sharpe | Cum Return | Total Cost | Turnover | Calm Algo       | Normal Algo  |
|--------|--------|------------|------------|----------|-----------------|--------------|
| 0      | ???    | ???        | 0          | ???      | ???             | ???          |
| 0.0005 | ???    | ???        | ???        | ???      | ???             | ???          |
| 0.001  | ???    | ???        | ???        | 226      | Trend_L10_B1    | MinVar_L40   |
| ...    | ...    | ...        | ...        | ...      | ...             | ...          |

Plots:
- results/14_kappa_sensitivity_sharpe.png (Sharpe vs kappa, with EW horizontal line)
- results/15_kappa_sensitivity_turnover.png (Turnover vs kappa)
- results/16_kappa_regime_mapping.png (heatmap: which algo selected per regime per kappa)
```

---

## Execution Instructions

Run all experiments from the project root:

```bash
cd regime_algo_selection

# Part 1: Classifier Deep-Dive
python -m experiments.classifier_analysis

# Part 2: Diagnostics
python -m experiments.algorithm_analysis
python -m experiments.sensitivity_analysis
```

Each script should:
1. Load cached data (from Milestone 1 — no re-downloading)
2. Run all analyses
3. Print results to console
4. Save tables as CSV to `results/`
5. Save plots as PNG to `results/`

Create an `experiments/` directory with an `__init__.py` file.

---

## Expected Outputs After This Week

```
results/
├── (Milestone 1 plots 01–06, already exist)
├── 07_feature_ablation.png
├── 08_model_comparison.png
├── 09_confidence_histogram.png
├── 10_reliability_diagram.png
├── 11_transition_analysis.png
├── 12_algorithm_ranking.png
├── 13_algorithm_families.png
├── 14_kappa_sensitivity_sharpe.png
├── 15_kappa_sensitivity_turnover.png
├── 16_kappa_regime_mapping.png
├── feature_ablation.csv
├── model_comparison.csv
├── algorithm_ranking.csv
└── kappa_sensitivity.csv
```

---

## What to Present to the Professor

With these results, you can tell a complete story:

1. **"The regime classifier has X% accuracy, but the naive baseline already achieves Y%."**
   → Shows you understand what the classifier actually learns.

2. **"Adding asset features improves accuracy to Z%, with the biggest gains from [specific features]."**
   → Shows systematic feature engineering.

3. **"XGBoost/RF achieves the highest accuracy at W%, but the Oracle Gap shows this barely matters for portfolio performance."**
   → Connects classifier quality to the actual objective.

4. **"Of the 48 Tier-1 algorithms, only N beat Equal Weight on the test set. The best is [X] with Sharpe [Y]."**
   → Identifies whether the algorithm pool is the bottleneck.

5. **"Switching costs explain [amount] of the performance gap. With kappa=0, the Reflex Agent achieves Sharpe [X]."**
   → Isolates the cost effect from the selection effect.

6. **"Next steps: replace the rigid lookup table with a learned meta-learner that can use asset features to differentiate within regimes, and expand the algorithm pool with Tier 2/3."**
   → Clear roadmap motivated by evidence.
