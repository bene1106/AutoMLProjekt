# Optimization Plan: Regime Classifier v2 — Maximizing Shift Detection (Variant A)

## Context

The baseline experiment (`experiments/regime_classifier_v2/`) and ablation study (`04_ablation_study.py`) are complete. Key findings:

- **Phase 1** (3 level features, bare LR): Recall 0.828, F1 0.646
- **Phase 2** (10 features, bare LR): Recall 0.680, F1 0.629
- **Phase 4** (10 features, RF balanced): Recall 0.697, F1 0.674 (best F1)
- Adding features LOWERED recall because LR became overselective
- Level features (VIX_MA20, max/min_VIX_window) dominate; dynamics features contribute little

**Problem:** The models were run with default hyperparameters, no threshold tuning, no feature selection, no cross-validation. There is significant room for improvement.

**Goal:** Systematically optimize shift detection to maximize BOTH recall and F1. Extract every bit of predictive signal from the data. Produce a final "best model" configuration with rigorous validation.

---

## File Structure

```
experiments/
  regime_classifier_v2/
    01_build_dataset.py          # (exists — do not modify)
    02_train_evaluate.py         # (exists — do not modify)
    03_analyze_results.py        # (exists — do not modify)
    04_ablation_study.py         # (exists — do not modify)
    05_optimize_classifier.py    # NEW — all optimization experiments
results/
  regime_classifier_v2/
    optimization/
      feature_selection_results.csv
      hyperparameter_tuning_results.csv
      threshold_tuning_results.csv
      horizon_sensitivity_results.csv
      final_model_results.csv
      final_model_fold_results.csv
      optimization_summary.png
      threshold_curve.png
      horizon_sensitivity.png
      feature_selection_comparison.png
```

**Do NOT modify existing scripts 01–04.**

---

## Step 1 — Feature Selection (fix the "more features = worse recall" problem)

### 1A: Correlation Analysis

Before any model training, compute the Pearson correlation matrix of all 10 features across the full dataset. Print and save it. Identify pairs with |correlation| > 0.8 — these are candidates for removal.

### 1B: Systematic Feature Selection via Walk-Forward

Test these feature subsets using LogisticRegression (no class_weight, max_iter=1000) on all 12 walk-forward folds:

```python
feature_subsets = {
    "level_only": [
        'VIX_MA20', 'max_VIX_window', 'min_VIX_window'
    ],
    "level_plus_zscore": [
        'VIX_MA20', 'max_VIX_window', 'min_VIX_window',
        'z_VIX'
    ],
    "level_plus_dynamics_core": [
        'VIX_MA20', 'max_VIX_window', 'min_VIX_window',
        'z_VIX', 'delta_VIX', 'VIX_slope_5'
    ],
    "level_plus_returns": [
        'VIX_MA20', 'max_VIX_window', 'min_VIX_window',
        'SPY_return_5', 'vol_ratio'
    ],
    "dynamics_only": [
        'z_VIX', 'delta_VIX', 'VIX_slope_5', 'VIX_slope_20',
        'VIX_rolling_std_10'
    ],
    "best_from_importance": [
        # Top 5 features from RF/XGBoost feature importance (from existing results)
        # Read feature_importances.csv and pick top 5
    ],
    "all_10": [
        'VIX_MA20', 'max_VIX_window', 'min_VIX_window',
        'z_VIX', 'delta_VIX', 'VIX_slope_5', 'VIX_slope_20',
        'VIX_rolling_std_10', 'SPY_return_5', 'vol_ratio'
    ],
}
```

For the "best_from_importance" subset: read `results/regime_classifier_v2/feature_importances.csv` (or `feature_importance.png` data), rank features by average importance across RF + XGBoost, pick top 5.

**Evaluate each subset:** avg recall, avg precision, avg F1, avg accuracy across 12 folds.

**Save:** `optimization/feature_selection_results.csv` with columns: `subset_name, n_features, features, avg_recall, std_recall, avg_precision, std_precision, avg_f1, std_f1, avg_accuracy`

**Select the winning subset** = the one with the best F1 among subsets that maintain recall >= 0.75. If no subset achieves recall >= 0.75 with acceptable F1, take the one with highest recall. Call this `best_features`.

---

## Step 2 — Hyperparameter Tuning

Using `best_features` from Step 1, tune each model type. Tuning is done via inner cross-validation WITHIN each walk-forward training set (3-fold stratified CV inside the training set). The outer walk-forward folds remain the evaluation.

### 2A: Logistic Regression

```python
from sklearn.model_selection import StratifiedKFold, GridSearchCV

param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'class_weight': [None, 'balanced'],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],  # supports both l1 and l2
    'max_iter': [1000],
}
```

Per outer fold:
1. Split into train/test by year.
2. Fit StandardScaler on train.
3. Run GridSearchCV with StratifiedKFold(n_splits=3) on the training set.
4. Scoring metric for GridSearchCV: `'f1'` (for positive class).
5. Record best params, then evaluate best model on the test set.

### 2B: Random Forest

```python
param_grid_rf = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_leaf': [1, 5, 10],
    'class_weight': ['balanced', 'balanced_subsample'],
}
```

Same inner CV procedure as LR.

### 2C: XGBoost

```python
param_grid_xgb = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [1, 5, 10],
    'scale_pos_weight': [None],  # computed per fold, see below
    'eval_metric': ['logloss'],
    'use_label_encoder': [False],
}
```

For XGBoost, `scale_pos_weight` is always set to `n_neg/n_pos` from the training fold — do NOT include it in the grid. Set it before running GridSearchCV:
```python
xgb = XGBClassifier(scale_pos_weight=n_neg/n_pos, random_state=42)
grid = GridSearchCV(xgb, param_grid_xgb, cv=inner_cv, scoring='f1')
```

### 2D: Gradient Boosting (sklearn)

As an additional model option:
```python
from sklearn.ensemble import GradientBoostingClassifier

param_grid_gb = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_samples_leaf': [5, 10, 20],
}
```
Note: GradientBoostingClassifier does not support `class_weight`. Use the data as-is (the class split is ~55/45, close enough to balanced).

**Save:** `optimization/hyperparameter_tuning_results.csv` with columns: `model, fold, best_params, recall, precision, f1, accuracy`

Plus an aggregated summary with: `model, avg_recall, std_recall, avg_precision, std_precision, avg_f1, std_f1, most_common_best_params`

---

## Step 3 — Decision Threshold Tuning

All classifiers above use the default threshold of 0.5. But the optimal threshold for shift detection may be different.

Using the best model + best hyperparameters from Step 2:

Per outer fold:
1. Train on training set with best hyperparameters.
2. Get predicted probabilities on test set: `y_proba = model.predict_proba(X_test)[:, 1]`
3. Sweep thresholds from 0.1 to 0.9 in steps of 0.05.
4. For each threshold, compute recall, precision, F1.
5. Record all results.

```python
thresholds = np.arange(0.10, 0.91, 0.05)
for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    # compute metrics
```

**Save:** `optimization/threshold_tuning_results.csv` with columns: `fold, threshold, recall, precision, f1, accuracy`

**Plot:** `optimization/threshold_curve.png`
- X-axis: threshold (0.1 to 0.9)
- Y-axis: metric value
- Three lines: Recall, Precision, F1 (averaged across 12 folds, with std as shaded area)
- Vertical dashed line at the optimal threshold (highest avg F1)
- Second vertical dashed line at the threshold that achieves recall >= 0.90 with best F1

**Select:** Two optimal thresholds:
- `threshold_f1`: maximizes avg F1 across folds
- `threshold_recall90`: lowest threshold where avg recall >= 0.90, report corresponding F1

---

## Step 4 — Horizon Sensitivity (H variation)

The current experiment uses H=10. Test whether a different horizon gives better results.

Using the best model + best hyperparameters + best threshold from Steps 2-3:

```python
horizons = [3, 5, 10, 15, 20]
```

For each H:
1. Recompute the binary shift label with the new H (same `any(...)` formulation).
2. Recompute class distribution (log it).
3. Run the full 12-fold walk-forward with the best model configuration.
4. Evaluate with the best threshold from Step 3 AND with threshold 0.5 for comparison.

**Save:** `optimization/horizon_sensitivity_results.csv` with columns: `H, class_distribution_shift_pct, threshold, avg_recall, std_recall, avg_precision, std_precision, avg_f1, std_f1`

**Plot:** `optimization/horizon_sensitivity.png`
- X-axis: H values (3, 5, 10, 15, 20)
- Two sets of bars or lines: F1 and Recall
- Show both threshold=0.5 and threshold=optimal

---

## Step 5 — Final Model Assembly

Take the best combination from Steps 1-4:
- Best feature subset (from Step 1)
- Best model + hyperparameters (from Step 2)
- Best threshold (from Step 3)
- Best horizon H (from Step 4, if different from 10)

Run the full 12-fold walk-forward evaluation with this final configuration.

**Save:**
- `optimization/final_model_results.csv`: aggregated metrics
- `optimization/final_model_fold_results.csv`: per-fold metrics

**Compare against all previous baselines in a single table:**

```
=== Final Comparison ===

Configuration                              | Recall | Precision | F1    | Accuracy
-------------------------------------------|--------|-----------|-------|--------
Naive baseline (always predict 0)          | 0.000  | NaN       | 0.000 | 0.XXX
Naive baseline (always predict 1)          | 1.000  | 0.XXX     | 0.XXX | 0.XXX
Ablation Phase 1 (3 feat, LR default)     | 0.828  | 0.XXX     | 0.646 | 0.XXX
Ablation Phase 4 RF (10 feat, balanced)    | 0.697  | 0.XXX     | 0.674 | 0.XXX
Optimized model (this experiment)          | 0.XXX  | 0.XXX     | 0.XXX | 0.XXX
```

Include the values from the existing ablation study for direct comparison. Read them from `results/regime_classifier_v2/ablation_summary.csv`.

---

## Step 6 — Summary Plot

**Plot:** `optimization/optimization_summary.png`

A single figure with 2 subplots:

**Subplot 1 (left):** Bar chart comparing the key configurations:
- Ablation Phase 1 (LR, 3 features, default)
- Ablation Phase 4 RF (10 features, balanced)
- Optimized model (best from this experiment)
- Bars for: Recall, Precision, F1

**Subplot 2 (right):** The threshold curve from Step 3 (the most actionable result).

---

## Step 7 — Print Summary

```
================================================================
         REGIME CLASSIFIER v2 — OPTIMIZATION RESULTS
================================================================

Label: Binary shift detection (Variant A)
  "Is there a regime shift in the next H days?"
  Formulation: any(regime[t+1:t+H+1] != regime_t)

--- FEATURE SELECTION ---
Best subset: {name} ({N} features)
Features: [list]
Reason: highest F1 with recall >= 0.75

Correlation analysis: [note any removed features and why]

--- HYPERPARAMETER TUNING ---
Best model: {model_name}
Best params: {params_dict}
Performance (12-fold avg, threshold=0.5):
  Recall: 0.XXX ± 0.XXX
  Precision: 0.XXX ± 0.XXX
  F1: 0.XXX ± 0.XXX

--- THRESHOLD TUNING ---
Optimal threshold (max F1): {threshold_f1}
  Recall: 0.XXX | Precision: 0.XXX | F1: 0.XXX

High-recall threshold (recall >= 0.90): {threshold_recall90}
  Recall: 0.XXX | Precision: 0.XXX | F1: 0.XXX

--- HORIZON SENSITIVITY ---
Best H: {best_H}
Class distribution at best H: {shift_pct}% shift / {no_shift_pct}% no-shift
Performance at best H + optimal threshold:
  Recall: 0.XXX | F1: 0.XXX

--- FINAL OPTIMIZED MODEL ---
Model: {model_name}
Features: {feature_list}
Horizon: H={best_H}
Threshold: {best_threshold}
  Recall:    0.XXX ± 0.XXX
  Precision: 0.XXX ± 0.XXX
  F1:        0.XXX ± 0.XXX
  Accuracy:  0.XXX ± 0.XXX

--- IMPROVEMENT OVER BASELINES ---
vs. Ablation Phase 1 (LR default):  Recall {+/-}0.XXX, F1 {+/-}0.XXX
vs. Ablation Phase 4 RF:            Recall {+/-}0.XXX, F1 {+/-}0.XXX
================================================================
```

---

## Constraints

1. Do NOT modify existing scripts 01–04.
2. All optimization is in a single new script: `05_optimize_classifier.py`.
3. Walk-forward folds are identical to existing experiment (same year splits).
4. StandardScaler fits on training set only.
5. Inner CV for hyperparameter tuning uses StratifiedKFold(n_splits=3) WITHIN the training set.
6. XGBoost `scale_pos_weight` computed per fold from training data, NOT part of the grid.
7. Random seed = 42 everywhere.
8. Code comments in English.
9. Install any missing packages: `pip install xgboost scikit-learn`.
10. The script should be runnable end-to-end and take ~10-30 minutes (depending on machine). If hyperparameter grids are too large, reduce by removing less promising values — but document what was removed and why.
11. When recomputing labels for different H values (Step 4), reuse the same feature computation — only the label changes.
