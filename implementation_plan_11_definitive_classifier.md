# Implementation Plan 11: Definitive Regime Shift Classifier

## Context & Motivation

Plans 7–10 established:
- Original 4-regime definition (VIX thresholds 15/20/30) confirmed as best (Plan 10 Step 1).
- LR is the cleanest model: F1=0.656 with zero train-test gap (Plan 10 Step 2).
- The `any(...)` binary shift label is too noisy: 54.5% of days are "shift" days at H=10.
- Simplified regime definitions (2 regimes, hysteresis, smoothing) all performed WORSE.
- More models and bigger grids did not help — the bottleneck is label design and features.

**This plan fixes the two root causes:**
1. The label counts 1-day flickers as "shifts" → replace with persistence-aware label.
2. Features describe market state, not crossing mechanics → add transition-geometry features.

**Target:** F1 > 0.85 at H=5 or H=7, significantly better than all baselines (including strong heuristic baselines), with stable performance across folds and no overfitting. Hard constraints: FAR < 0.50 and Precision > 0.50 — the classifier must not achieve high F1 through alarm spam.

---

## File Structure

```
experiments/
  regime_classifier_v2/
    08_definitive_classifier.py         # NEW — single script, all phases
results/
  regime_classifier_v2/
    definitive/
      # Phase 0 — Label Analysis
      phase0_label_comparison.csv
      phase0_label_distribution.png
      phase0_label_comparison.png
      # Phase 1 — Feature Ablation
      phase1_feature_ablation.csv
      phase1_feature_ablation.png
      phase1_feature_importance.png
      # Phase 2 — Model Optimization
      phase2_model_results.csv
      phase2_model_comparison.png
      phase2_confusion_matrices.png
      phase2_roc_curves.png
      phase2_overfitting_diagnostics.csv
      # Phase 3 — Horizon Sweep
      phase3_horizon_sweep.csv
      phase3_horizon_sweep.png
      phase3_confusion_matrices_per_H.png
      # Phase 4 — Final Model
      phase4_final_model.csv
      phase4_confusion_matrix.png
      phase4_precision_recall_curve.png
      phase4_per_fold_barplot.png
      phase4_shift_timeline.png
      phase4_learning_curve.png
      # Summary
      full_summary.csv
```

**Do NOT modify any existing scripts (01–07).**

---

## Phase 0 — Label Redesign (THE critical change)

### Why this matters most

The current label `1 if any(regime[t+1:t+H+1] != regime_t) else 0` counts every 1-day flicker across a VIX threshold as a "shift". At H=10 with 4 regimes, 54.5% of all days are labeled as shift — the classifier is trying to predict something that happens more than half the time. That is too easy AND too noisy.

### New: Persistence-Aware Shift Label

A shift counts ONLY if the new regime persists for at least `min_persistence` consecutive days:

```python
def compute_persistent_shift_label(regime_series, H, min_persistence):
    """
    label_t = 1 if there exists tau in [t+1, t+H] such that:
      - regime[tau] != regime[t]  (a different regime starts)
      - regime[tau+k] == regime[tau] for all k in [0, min_persistence-1]
        (the new regime persists for min_persistence consecutive days)
    label_t = 0 otherwise.
    
    Edge cases:
    - Last H + min_persistence - 1 rows of the dataset get no label (dropped).
    - The persistence check may look beyond H. This is intentional:
      we want to verify the shift is real, not just that it occurred.
      The persistence window is part of the label definition, not a feature.
    """
    labels = pd.Series(index=regime_series.index, dtype=float)
    regimes = regime_series.values
    n = len(regimes)
    
    for t in range(n):
        # Need at least H + min_persistence - 1 future days
        if t + H + min_persistence - 1 >= n:
            labels.iloc[t] = np.nan
            continue
        
        found_shift = False
        for tau in range(t + 1, t + H + 1):
            if regimes[tau] != regimes[t]:
                # Check: does the new regime persist?
                if tau + min_persistence - 1 < n:
                    persists = all(
                        regimes[tau + k] == regimes[tau] 
                        for k in range(min_persistence)
                    )
                    if persists:
                        found_shift = True
                        break
        
        labels.iloc[t] = 1 if found_shift else 0
    
    return labels
```

### Label variants to test

```python
label_configs = {
    "any_shift":       {"min_persistence": 1},   # current label (baseline for comparison)
    "persist_2":       {"min_persistence": 2},   # PRIMARY candidate — removes 1-day flickers
    "persist_3":       {"min_persistence": 3},   # SECONDARY candidate — stricter
    "persist_5":       {"min_persistence": 5},   # EXPLORATORY only — may be too strict (too few positives)
}
# Priority order: persist_2 > persist_3 > persist_5
# persist_5 is expected to have very few positive labels — if shift_pct < 10%, skip it.
```

### For each label variant × H ∈ {5, 7} (primary) and H ∈ {3, 10} (reference only):

H=5 and H=7 are the primary targets. H=3 is a secondary check (may be too hard). H=10 is included only as a benchmark against previous plans — it is NOT a candidate for the final model because larger H produces artificially easy labels.

1. Compute labels on full dataset.
2. Log class distribution: what % of days are now labeled as shift?
3. Run 12-fold walk-forward with baseline LR (C=0.001, class_weight='balanced') using the existing Group B features (8 features from Plan 9).
4. Compute: Recall, Precision, F1, PR-AUC, Accuracy.

### Naive Baselines (compute for EVERY label variant × H):

#### Trivial Baselines:
```python
from sklearn.dummy import DummyClassifier

# Baseline 1: Always predict 0 (no shift)
y_pred_always0 = np.zeros_like(y_test)

# Baseline 2: Always predict 1 (always shift)  
y_pred_always1 = np.ones_like(y_test)

# Baseline 3: Random proportional to class distribution
dummy_stratified = DummyClassifier(strategy="stratified", random_state=42)
dummy_stratified.fit(X_train, y_train)
y_pred_stratified = dummy_stratified.predict(X_test)
```

#### Strong Heuristic Baselines (IMPORTANT — the model MUST beat these):
```python
# Baseline 4: Yesterday's shift
# "If a regime change happened yesterday, predict shift for today"
# Logic: shifts often cluster — if one just happened, another may follow.
y_pred_yesterday_shift = (regime_test != regime_test.shift(1)).astype(int)
# For the first day of test set: use last training day as reference.

# Baseline 5: Threshold proximity heuristic
# "Predict shift if VIX is within 'margin' points of any threshold"
# This is the simplest possible version of crossing_pressure — no ML needed.
# Test margin values: 1.0, 1.5, 2.0, 3.0
# Choose the margin that maximizes F1 on the TRAINING set, then apply to test.
margins_to_test = [1.0, 1.5, 2.0, 3.0]
for margin in margins_to_test:
    y_pred_proximity = (VIX_dist_nearest_test < margin).astype(int)
    # evaluate on training set to pick best margin (no test-set leak)

# Baseline 6: Persistence baseline
# "Predict shift if VIX has been in the current regime for fewer than X days"
# Logic: short regime tenure → more likely to shift soon
# Test tenure thresholds: 3, 5, 10 days
# regime_tenure_t = number of consecutive days in current regime up to day t
```

Log all 6 baselines per fold. The model must beat ALL of them — especially Baselines 4, 5, and 6 — to be considered genuinely useful. If the model only beats the trivial baselines but not the heuristic ones, it has not learned anything beyond simple rules.

### Select best label variant

Choose the variant where:
- F1 is highest at H=5 or H=7
- The shift rate is between 15% and 45% (not too rare, not too common)
- Performance is stable across folds (std(F1) < 0.15)

### Output:
- `phase0_label_comparison.csv` — one row per label_variant × H: shift_pct, recall, precision, f1, pr_auc, plus naive baseline metrics
- `phase0_label_distribution.png` — bar chart: shift_pct per label_variant × H
- `phase0_label_comparison.png` — grouped bar chart: F1 per label_variant × H, with horizontal lines for naive baselines

### Print summary:
```
=== Phase 0: Label Comparison ===

Label Variant  | H  | Shift% | Recall | Prec  | F1    | PR-AUC | vs Naive(strat)
any_shift      | 5  | XX.X%  | 0.XXX  | 0.XXX | 0.XXX | 0.XXX  | +0.XXX
persist_2      | 5  | XX.X%  | 0.XXX  | 0.XXX | 0.XXX | 0.XXX  | +0.XXX
persist_3      | 5  | XX.X%  | 0.XXX  | 0.XXX | 0.XXX | 0.XXX  | +0.XXX
persist_5      | 5  | XX.X%  | 0.XXX  | 0.XXX | 0.XXX | 0.XXX  | +0.XXX

Best label: [variant] at H=[value], F1 = 0.XXX, shift_pct = XX.X%
```

---

## Phase 1 — Feature Ablation with Crossing-Geometry Features

### New Features to Compute

All features use data up to time t only. No look-ahead. Thresholds are [15, 20, 30].

```python
# --- Existing features (from previous plans) ---
VIX_MA20           = VIX.rolling(20).mean()
max_VIX_window     = VIX.rolling(20).max()
min_VIX_window     = VIX.rolling(20).min()
VIX_slope_20       = VIX - VIX.shift(20)
VIX_rolling_std_10 = VIX.rolling(10).std()

# Threshold distances (existing)
VIX_dist_nearest   = VIX.apply(lambda v: min(abs(v - t) for t in [15, 20, 30]))
VIX_dist_upper     = ...  # distance to next higher threshold
VIX_dist_lower     = ...  # distance to next lower threshold

# --- NEW: Crossing-Geometry Features ---

# 1. window_position: where is VIX within its recent range? (0=bottom, 1=top)
window_range = max_VIX_window - min_VIX_window
window_position = (VIX - min_VIX_window) / (window_range + 1e-8)

# 2. crossing_pressure: trend strength relative to distance to threshold
#    High value = strong trend toward a nearby threshold = likely crossing
crossing_pressure = VIX_slope_20 / (VIX_dist_nearest + 1e-8)

# 3. threshold_instability: volatility relative to distance to threshold
#    High value = volatile AND near a threshold = unstable, likely to cross
threshold_instability = VIX_rolling_std_10 / (VIX_dist_nearest + 1e-8)

# 4. recent_threshold_touch_5: was VIX within 1.0 point of any threshold in last 5 days?
#    Binary flag. c=1.0 (adjustable).
c = 1.0
recent_touch = VIX_dist_nearest.rolling(5).min()
recent_threshold_touch_5 = (recent_touch < c).astype(int)

# 5. signed_distance_to_relevant_boundary (OPTIONAL — may be unstable):
#    Conceptually interesting but potentially overfit or hard to interpret.
#    Include in Set E only. Do not include in Set D or Set F.
#    If it has low feature importance in Phase 1, drop it from final model.
#    Calm (VIX<=15): relevant boundary is 15 (upward)
#    Normal (15<VIX<=20): closer of 15 (downward) or 20 (upward)
#    Tense (20<VIX<=30): closer of 20 (downward) or 30 (upward)
#    Crisis (VIX>30): relevant boundary is 30 (downward)
#    Sign: positive = VIX is moving TOWARD the boundary, negative = away
def signed_distance(vix_val, slope):
    thresholds = [15, 20, 30]
    distances = [(t - vix_val) for t in thresholds]
    # Find the threshold we're moving toward (same sign as slope)
    relevant = [(abs(d), d) for d in distances 
                if (d > 0 and slope > 0) or (d < 0 and slope < 0)]
    if relevant:
        _, d = min(relevant, key=lambda x: x[0])
        return d  # positive = approaching from below, negative = approaching from above
    else:
        return min(distances, key=abs)  # fallback: nearest threshold

signed_dist = pd.Series(
    [signed_distance(v, s) for v, s in zip(VIX, VIX_slope_20)],
    index=VIX.index
)
```

### Feature Sets for Ablation

```python
# Set A — Level Only (baseline, minimal)
set_A = ['VIX_MA20', 'max_VIX_window', 'min_VIX_window']

# Set B — Level + Boundary Distance  
set_B = set_A + ['VIX_dist_nearest']

# Set C — Core-6 (Level + Trend + Volatility + Distance)
set_C = ['VIX_MA20', 'max_VIX_window', 'min_VIX_window',
         'VIX_dist_nearest', 'VIX_slope_20', 'VIX_rolling_std_10']

# Set D — Transition Only (NEW features only, no level features)
#          Hardest test: can crossing-geometry alone predict shifts?
set_D = ['VIX_dist_nearest', 'crossing_pressure', 'threshold_instability',
         'window_position', 'recent_threshold_touch_5']

# Set E — Core-6 + Transition (best of both worlds)
#          signed_dist included here as optional test
set_E = ['VIX_MA20', 'max_VIX_window', 'min_VIX_window',
         'VIX_dist_nearest', 'VIX_slope_20', 'VIX_rolling_std_10',
         'crossing_pressure', 'threshold_instability', 'window_position',
         'recent_threshold_touch_5', 'signed_dist']

# Set F — Minimal Transition (only the 3 strongest new features)
set_F = ['VIX_dist_nearest', 'crossing_pressure', 'threshold_instability']

# Set G — Set E without signed_dist (if signed_dist proves unhelpful)
set_G = ['VIX_MA20', 'max_VIX_window', 'min_VIX_window',
         'VIX_dist_nearest', 'VIX_slope_20', 'VIX_rolling_std_10',
         'crossing_pressure', 'threshold_instability', 'window_position',
         'recent_threshold_touch_5']
```

### Run ablation

Use the best label variant from Phase 0, at H=5 and H=7.
Model: LR (C=0.001, class_weight='balanced', max_iter=2000).
12-fold walk-forward. StandardScaler on training only.

For each feature set × H:
- Compute Recall, Precision, F1, PR-AUC, Accuracy.
- Compare against naive baselines.

### Feature correlation check per fold:
Before training, drop features with |r| > 0.90 (keep the one with higher F-statistic). Log which features are dropped.

### Output:
- `phase1_feature_ablation.csv` — one row per feature_set × H
- `phase1_feature_ablation.png` — grouped bar chart: F1 per feature set, grouped by H
- `phase1_feature_importance.png` — for sets C, E: bar plot of LR coefficients (absolute value, averaged across 12 folds). Shows which features LR actually uses.

---

## Phase 2 — Model Optimization (focused)

### Models: LR + XGBoost (mandatory), RF (optional)

LR and XGBoost are the primary models. RF is only run if LR and XGB results are unclear (within 0.02 F1 of each other) or if runtime allows. If time is tight, skip RF entirely.

Use the best feature set from Phase 1, best label from Phase 0, at H=5 and H=7.

### LR Grid (full, with Pipeline + SelectKBest):

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

pipe_lr = Pipeline([
    ('select', SelectKBest(f_classif)),
    ('model', LogisticRegression(solver='saga', max_iter=2000, random_state=42))
])

lr_grid = {
    'select__k': [3, 5, 'all'],
    'model__C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    'model__class_weight': [
        'balanced',
        {0: 1, 1: 1},
        {0: 1, 1: 2},
        {0: 1, 1: 3},
        {0: 1, 1: 5},
        {0: 1, 1: 10},
        {0: 1, 1: 15},
        {0: 1, 1: 20},
    ],
    'model__penalty': ['l1', 'l2'],
}
# = 3 × 6 × 8 × 2 = 288 combos × 3-fold inner CV = 864 fits per outer fold
# GridSearchCV, scoring='f1', cv=StratifiedKFold(3)
```

### XGBoost Grid (RandomizedSearchCV):

```python
xgb_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'scale_pos_weight': [],  # per fold: base_ratio * [0.5, 1.0, 2.0, 3.0, 5.0]
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [1.0, 5.0, 10.0],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}
# RandomizedSearchCV, n_iter=150, scoring='f1', cv=StratifiedKFold(3)
```

### RF Grid (GridSearchCV, moderate):

```python
rf_grid = {
    'n_estimators': [100, 300],
    'max_depth': [5, 10, None],
    'class_weight': ['balanced', {0: 1, 1: 5}, {0: 1, 1: 10}],
    'min_samples_leaf': [1, 5],
}
# = 2 × 3 × 3 × 2 = 36 combos × 3-fold inner CV = 108 fits per outer fold
```

### Per fold:
1. Compute class distribution.
2. For XGB: fill scale_pos_weight candidates based on training data ratio.
3. Fit StandardScaler on training data.
4. Run GridSearchCV / RandomizedSearchCV.
5. Log best params.
6. Predict on test set.
7. Log BOTH train metrics and test metrics (overfitting diagnostic).

### Overfitting rules:
- Log train-test F1 gap per fold per model → `phase2_overfitting_diagnostics.csv`
- If avg gap > 0.20 across 12 folds: model is FLAGGED as likely overfit. Print warning. Do NOT automatically reject — instead, report the flag and let the final summary interpret it.
- If std(F1) > 0.20 across 12 folds: model is FLAGGED as unstable.
- If any single fold has F1 < 0.20: log as WARNING (that fold may have unusual market conditions).
- These are diagnostic thresholds for interpretation, NOT hard rejection rules. A model with avg gap = 0.21 but otherwise strong and stable results should still be considered.

### Output:
- `phase2_model_results.csv` — per fold × model: best_params, train_f1, test_f1, gap, recall, precision, f1, pr_auc
- `phase2_model_comparison.png` — bar chart: Recall, Precision, F1, PR-AUC per model
- `phase2_confusion_matrices.png` — one CM per model (aggregated across 12 folds), 1 row × 3 columns
- `phase2_roc_curves.png` — ROC curve per model
- `phase2_overfitting_diagnostics.csv`

---

## Phase 3 — Horizon Sweep

Take the best model from Phase 2 (with its optimized weights). Sweep H ∈ {3, 5, 7, 10}.

Use the best label variant from Phase 0 (same min_persistence).

For each H:
1. Recompute labels.
2. Run 12-fold walk-forward with best model.
3. Find optimal threshold via PR curve on training data per fold.
4. Record metrics at threshold=0.5 AND at optimal threshold.
5. Compute naive baselines for comparison.

### Output:
- `phase3_horizon_sweep.csv` — per H: shift_pct, recall, precision, f1, pr_auc, far, optimal_threshold, plus naive baseline metrics
- `phase3_horizon_sweep.png` — line plot: F1 and PR-AUC vs H (two lines: default threshold, optimal threshold)
- `phase3_confusion_matrices_per_H.png` — one CM per H (1 row × 4 columns: H=3,5,7,10)

### Stability check:
For each H, report std(F1) across 12 folds. Flag any H where std > 0.15.

---

## Phase 4 — Final Model Report

Take the absolute best configuration from Phases 0–3:
- Best label variant + min_persistence
- Best feature set
- Best model + hyperparameters
- Best H
- Optimal threshold (from PR curve on training data)

Run it one final time with full logging.

### Per fold, log:
- train_recall, train_precision, train_f1
- test_recall, test_precision, test_f1
- gap_f1
- optimal_threshold
- n_shift_true, n_shift_predicted, TP, FP, TN, FN

### Confusion Matrix:
Aggregated across all 12 folds. Save as `phase4_confusion_matrix.png`.

### Precision-Recall Curve:
Aggregated across all 12 folds. Show the chosen operating point. Save as `phase4_precision_recall_curve.png`.

### Per-Fold Barplot:
Bar chart of F1 per fold (12 bars), with horizontal line at mean F1 and shaded band at ±1 std. Shows stability. Save as `phase4_per_fold_barplot.png`.

### Shift Timeline:
For the test year with MEDIAN F1 (most representative fold):
- X-axis: trading days of the year
- Y-axis top: VIX level with regime threshold lines (15, 20, 30)
- Y-axis bottom: actual shifts (green dots) vs predicted shifts (red/blue dots)
- Shows WHERE the model succeeds and fails
Save as `phase4_shift_timeline.png`.

### Learning Curve:
On Fold 6 (middle fold, ~3000 training days):
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train, y_train,
    train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
    cv=3, scoring='f1', random_state=42
)
```
Plot train F1 vs test F1 as function of training set size. Shows if overfitting persists.
Save as `phase4_learning_curve.png`.

### Summary CSV:
`full_summary.csv` — single row with the final configuration and all metrics:
```
label_variant, min_persistence, H, feature_set, n_features, model, best_params,
avg_recall, std_recall, avg_precision, std_precision, avg_f1, std_f1, 
avg_pr_auc, avg_accuracy, avg_far,
naive_always0_f1, naive_always1_f1, naive_stratified_f1,
improvement_vs_naive, improvement_vs_plan10_LR,
avg_train_test_gap, max_fold_gap, min_fold_f1
```

### Print comprehensive summary:

```
======================================================================
  PLAN 11 — DEFINITIVE REGIME SHIFT CLASSIFIER — FINAL RESULTS
======================================================================

Configuration:
  Label:       persistent shift (min_persistence=[X])
  Horizon:     H=[X]
  Features:    Set [X] ([N] features): [list]
  Model:       [name]
  Best Params: [params]
  Threshold:   [X] (optimized via PR curve)

Class Distribution:
  Shift:     XX.X% of days
  No-shift:  XX.X% of days

Performance (avg ± std across 12 folds):
  Recall:     0.XXX ± 0.XXX
  Precision:  0.XXX ± 0.XXX
  F1:         0.XXX ± 0.XXX
  PR-AUC:     0.XXX ± 0.XXX
  Accuracy:   0.XXX ± 0.XXX
  FAR:        0.XXX ± 0.XXX

Overfitting Check:
  Avg train-test gap:  0.XXX  (< 0.20 ✓/✗)
  Max single-fold gap: 0.XXX
  Min single-fold F1:  0.XXX  (> 0.30 ✓/✗)
  Std(F1):             0.XXX  (< 0.15 ✓/✗)

Naive Baselines:
  Always predict 0:        F1 = 0.XXX
  Always predict 1:        F1 = 0.XXX
  Stratified random:       F1 = 0.XXX
  Yesterday's shift:       F1 = 0.XXX
  Threshold proximity:     F1 = 0.XXX  (margin=[X])
  Regime tenure:           F1 = 0.XXX  (tenure=[X])
  Model vs best heuristic: +0.XXX F1

Comparison to Previous Plans:
  vs. Plan 10 LR (any_shift, H=5):       F1 +0.XXX
  vs. Plan 9 best (LR, any_shift, H=10): F1 +0.XXX
  vs. Plan 7 Phase 1 (LR, 3 feat, H=10): F1 +0.XXX

Progression within Plan 11:
  Phase 0 — Label fix:     F1 went from 0.XXX (any) to 0.XXX (persist_[X])
  Phase 1 — Feature fix:   F1 went from 0.XXX (set A) to 0.XXX (set [X])
  Phase 2 — Model opt:     F1 went from 0.XXX (default LR) to 0.XXX (tuned [model])
  Phase 3 — Best horizon:  H=[X], F1 = 0.XXX

Key Findings:
  1. [Was the persistence label the biggest improvement?]
  2. [Did crossing-geometry features outperform level-only features?]
  3. [Which individual feature was most important?]
  4. [Is there overfitting? What does the learning curve show?]
======================================================================
```

---

## Constraints

### Overfitting Prevention (MANDATORY — apply everywhere)

A. **Feature correlation handling:** Before training, drop features with |r| > 0.90 per fold. Keep the one with higher F-statistic. Log drops.

B. **Train-test gap logging:** Every model evaluation logs train metrics AND test metrics. Gap > 0.20 = FLAGGED (warning, not automatic rejection). Interpret in context.

C. **Feature selection in LR pipeline:** SelectKBest with k ∈ {3, 5, 'all'} inside GridSearchCV.

D. **Stability check:** std(F1) > 0.20 across folds = FLAGGED. Single fold F1 < 0.20 = WARNING. These are diagnostic thresholds, not hard cutoffs.

E. **Learning curve:** Generated for final model to verify convergence.

### General Constraints

1. Do NOT modify existing scripts (01–07).
2. Walk-forward: 12 folds, test years 2013–2024, expanding training window.
3. StandardScaler fit on training data only per fold.
4. XGBoost scale_pos_weight computed per fold from training data.
5. Threshold optimization: PR curve on training data only, never test data.
6. Random seed = 42 everywhere.
7. Code comments in English, thesis-quality.
8. Naive baselines computed for EVERY configuration, not just the final one.
9. Print progress per phase, per model, per fold with ETA estimates.
10. All output files listed in the file structure MUST be generated.
11. If runtime exceeds 2 hours: Phase 2 LR must complete fully. XGB can use n_iter=100 instead of 150. RF can be skipped.
