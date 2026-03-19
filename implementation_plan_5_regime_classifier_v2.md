# Implementation Plan: Improved Regime Classifier v2

## Context & Motivation

The current regime classifier (Logistic Regression on lagged VIX features) achieves ~84.5% accuracy — but the naive baseline ("yesterday's regime") already achieves ~84.7%. The classifier learns nothing beyond VIX autocorrelation.

**Root cause:** The label `"which regime is today?"` is dominated by stable days (~85% of all days). Any model can achieve high accuracy by predicting "no change". The classifier fails exactly where it matters: at regime transition points.

**Solution:** Change the label to `"will there be a regime shift in the next H days?"` (binary). This forces the model to detect transitions — the only thing the downstream Meta-Learner actually needs.

This experiment runs in a **separate directory**. No existing code is modified.

---

## File Structure

```
experiments/
  regime_classifier_v2/
    01_build_dataset.py       # Features + Labels
    02_train_evaluate.py      # Training + Walk-Forward Evaluation
    03_analyze_results.py     # Plots + Metrics
results/
  regime_classifier_v2/
    (all outputs go here)
```

**Existing files in `agents/`, `algorithms/`, `evaluation/` are NOT touched.**

---

## Phase 1: Core Setup (Strategies 1 + 2)

This is the foundation. Everything else builds on this.

### Step 1 — Load Data & Compute Existing Regime Labels

Load existing data from `data/`:
- ETF prices: SPY, TLT, GLD, EFA, VNQ (2006–2024)
- VIX closing prices

Compute the existing 4-class regime label using the existing VIX thresholds:
```python
# Existing thresholds — do NOT change these
regime_t = g(VIX_t)  # {1: Calm, 2: Normal, 3: Tense, 4: Crisis}
```

These thresholds are defined elsewhere in the codebase. Import or replicate them exactly.

### Step 2 — Compute Binary Shift Label (Strategy 1, Variant A)

```python
H = 10  # Horizon: look H trading days into the future

# IMPORTANT: Check if the regime deviates from today's regime
# at ANY point in the next H days — not just at day t+H.
# This catches shifts that revert within the window.

for each day t:
    future_regimes = regime[t+1 : t+H+1]   # next H days (exclusive of today)
    label_t = 1 if any(future_regimes != regime_t) else 0
```

**Why `any(...)` instead of just checking `regime_{t+H} != regime_t`:**
If the regime goes Calm → Tense → Calm within 10 days, comparing only endpoints gives label=0 (no shift). But a shift clearly happened. The `any(...)` formulation catches this.

**Edge cases:**
- The last H rows of the dataset have no complete future window → drop them (no label available).
- This is NOT look-ahead bias because the label is only used during training, never as an input feature.

**Log the class distribution:**
```
label=0 (no shift):  X% of days  (expected: ~70-80%)
label=1 (shift):     Y% of days  (expected: ~20-30%)
```

### Step 3 — Compute Features (Strategies 2 + 4, first batch)

All features use data up to and including time t. No look-ahead.

**Lookback window L = 20 trading days** is used for window-summary features.

#### VIX-Level Feature (one representative, not three correlated ones):
```python
VIX_MA20 = VIX.rolling(20).mean()
```
Do NOT include VIX_MA5 and VIX_MA60 in the first run — they are highly correlated with VIX_MA20 and the strategy document explicitly warns against this. They can be tested as optional additions later.

#### VIX Outlier Feature (highest priority new feature):
```python
VIX_rolling_std_20 = VIX.rolling(20).std()
z_VIX = (VIX - VIX_MA20) / VIX_rolling_std_20
```
This is the z-score of VIX: how many standard deviations the current VIX is from its 20-day mean. z_VIX > 2 signals a potential breakout. This is the single most important new feature.

#### VIX Dynamics Features:
```python
delta_VIX   = VIX - VIX.shift(1)          # daily change
VIX_slope_5 = VIX - VIX.shift(5)          # 5-day trend
VIX_slope_20 = VIX - VIX.shift(20)        # 20-day trend (longer-term context)
VIX_rolling_std_10 = VIX.rolling(10).std() # VIX's own volatility
```

#### Window Summary Features (over L=20 days):
```python
max_VIX_window = VIX.rolling(20).max()
min_VIX_window = VIX.rolling(20).min()
std_VIX_window = VIX.rolling(20).std()   # same as VIX_rolling_std_20
```
Note: `std_VIX_window` is identical to `VIX_rolling_std_20`. Include it only once.

#### Return-Based Features (from SPY):
```python
SPY_returns = SPY_price.pct_change()
SPY_return_5 = SPY_price / SPY_price.shift(5) - 1  # cumulative 5-day return
rolling_vol_5  = SPY_returns.rolling(5).std()
rolling_vol_60 = SPY_returns.rolling(60).std()
vol_ratio = rolling_vol_5 / rolling_vol_60           # short vs long-term vol
```

#### Final feature list for Phase 1:
```python
features = [
    'VIX_MA20',
    'z_VIX',              # outlier detection
    'delta_VIX',          # daily change
    'VIX_slope_5',        # 5-day trend
    'VIX_slope_20',       # 20-day trend
    'VIX_rolling_std_10', # VIX instability
    'max_VIX_window',     # 20-day max
    'min_VIX_window',     # 20-day min
    'SPY_return_5',       # market momentum
    'vol_ratio',          # short/long vol ratio
]
```
That is 10 features. All NaN rows at the start (from rolling windows needing warm-up, max warm-up = 60 days for rolling_vol_60) are dropped.

### Step 4 — Walk-Forward Validation Setup

Use the same 12 folds as the existing pipeline:
```
Fold 1:  Train 2006–2012, Test 2013
Fold 2:  Train 2006–2013, Test 2014
...
Fold 12: Train 2006–2023, Test 2024
```

**IMPORTANT timing detail:** Because of the label horizon H=10:
- The last H rows of each training set have no label → drop them from training.
- The last H rows of each test set have no label → drop them from evaluation.
- This means each test year effectively has ~250 - 10 = ~240 evaluable days.

Per fold:
1. Split data into train and test by year boundaries.
2. Drop rows without labels (last H rows of each set).
3. Fit StandardScaler on training features ONLY.
4. Transform test features with the fitted scaler (no re-fit on test).
5. Train model on training set.
6. Predict on test set.
7. Evaluate.

### Step 5 — Train Models (Strategy 3 + 5)

Train three models and compare:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = {
    "LogisticRegression": LogisticRegression(
        class_weight='balanced',  # Strategy 5: handle class imbalance
        max_iter=1000
    ),
    "RandomForest": RandomForestClassifier(
        class_weight='balanced',  # Strategy 5
        n_estimators=100,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        scale_pos_weight=None,  # computed per fold below
        n_estimators=100,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ),
}
```

**For XGBoost**: `scale_pos_weight` must be computed **per fold** from the training data:
```python
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
xgb_model.set_params(scale_pos_weight=n_neg / n_pos)
```
This is NOT a global constant — it changes per fold as the training set grows.

### Step 6 — Evaluation Metrics

**Primary metric — Shift Detection (label=1):**
```python
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

precision_shift = precision_score(y_test, y_pred, pos_label=1)
recall_shift    = recall_score(y_test, y_pred, pos_label=1)
f1_shift        = f1_score(y_test, y_pred, pos_label=1)
```

**Recall on label=1 is the single most important metric.** The classifier must not miss regime shifts.

**Naive Baselines (on the NEW binary label):**

Baseline 1 — "Always predict 0 (no shift)":
```python
# This is the trivial majority-class baseline
baseline_always_0_accuracy = (y_test == 0).mean()
baseline_always_0_recall   = 0.0   # misses ALL shifts
baseline_always_0_f1       = 0.0
```

Baseline 2 — "Always predict 1 (always shift)":
```python
baseline_always_1_recall    = 1.0   # catches all shifts
baseline_always_1_precision = (y_test == 1).mean()  # but very imprecise
```

These two baselines bracket the trivial performance range.

**Comparison with old classifier:**
Also compute what the OLD regime classifier's accuracy looks like when evaluated on the NEW metric. Specifically: the old classifier predicts regime labels; convert these to binary shift predictions by checking if predicted regime differs from previous predicted regime. This shows how the old system performs on the new task.

**Save per fold:**
```
fold, model, precision_shift, recall_shift, f1_shift, accuracy, n_shifts_true, n_shifts_predicted
```

### Step 7 — Plots & Outputs

**Plot 1 — Confusion Matrix (aggregated across all 12 folds):**
- One 2x2 matrix per model (3 total)
- Save as `results/regime_classifier_v2/confusion_matrix_{model}.png`

**Plot 2 — Shift Recall per Fold:**
- Line plot: Recall on label=1 across 12 folds, one line per model
- Include horizontal line at 0.0 (naive baseline "always 0")
- Save as `results/regime_classifier_v2/shift_recall_per_fold.png`

**Plot 3 — Feature Importance (RF + XGBoost):**
- Bar plot of all 10 features, importance averaged across 12 folds
- Save as `results/regime_classifier_v2/feature_importance.png`

**CSV Outputs:**
```
results/regime_classifier_v2/fold_results.csv             # per fold + model
results/regime_classifier_v2/shift_detection_summary.csv  # aggregated across folds
results/regime_classifier_v2/class_distribution.csv       # label balance per fold
```

### Step 8 — Print Summary

At the end of execution, print:
```
=== Binary Shift Detection Summary ===
Label: "regime shift in next H=10 days?" (any != current)
Class distribution: X% no-shift, Y% shift

Naive Baseline (always predict "no shift"):
  Accuracy: X.XX   Recall: 0.00   F1: 0.00

Model: LogisticRegression (class_weight='balanced')
  Avg Recall (shift):    0.XX  ± 0.XX
  Avg Precision (shift): 0.XX  ± 0.XX
  Avg F1 (shift):        0.XX  ± 0.XX
  Avg Accuracy:          0.XX  ± 0.XX

Model: RandomForest (class_weight='balanced')
  ...

Model: XGBoost (scale_pos_weight per fold)
  ...

Top 3 Features (by avg importance across RF + XGB):
  1. feature_name: 0.XXX
  2. feature_name: 0.XXX
  3. feature_name: 0.XXX
```

---

## Constraints & Implementation Notes

1. **Existing code is read-only.** Do not modify files in `agents/`, `algorithms/`, `evaluation/`.
2. **No look-ahead bias.** Features use data up to time t only. Labels use future data but are ONLY used during training.
3. **StandardScaler fits on training set only.** Never fit on test data.
4. **XGBoost `scale_pos_weight` is computed per fold** from training data, not globally.
5. **Drop NaN rows** at the start (rolling window warm-up, ~60 rows) and at the end (label horizon, 10 rows).
6. **Drop rows without labels** at the end of each train/test split (last H=10 rows).
7. **Code comments in English**, clear and thesis-quality.
8. **Use existing data loading** if utility functions exist in `data/`, otherwise load directly from CSV.
9. **Random seeds** fixed at 42 for reproducibility.
10. **Install xgboost** if not already available: `pip install xgboost`.
