# Implementation Plan 10: Regime Definition Simplification + Full Weight Optimization

## Context

Plans 7–9 established that with 4 VIX regimes (thresholds at 15, 20, 30) and the binary shift label, the best achievable F1 at small H is ~0.58 (H=3) to ~0.75 (H=10). The core problem is not the model — it's that the shift label is noisy. The VIX frequently oscillates across thresholds (e.g., 19.8 → 20.2 → 19.9), creating "shifts" that are not real regime changes.

Additionally, Plan 9's weight optimization grids were reduced by 83–99% for runtime. The most aggressive class weights ({0:1, 1:15}, {0:1, 1:20}), L1 penalty, and XGBoost regularization were all skipped.

**This plan has two goals:**
1. Test simplified regime definitions that produce cleaner shift labels.
2. Re-run weight optimization with full grids on the best regime definition.

## File Structure

```
experiments/
  regime_classifier_v2/
    07_regime_simplification.py     # NEW
results/
  regime_classifier_v2/
    regime_simplification/
      # Step 1 — Regime Definition Comparison
      step1_regime_comparison.csv
      step1_regime_comparison.png
      step1_confusion_matrices.png
      step1_shift_distributions.png
      # Step 2 — Full Weight Optimization
      step2_full_weight_optimization.csv
      step2_best_weights_per_fold.csv
      step2_model_comparison_barplot.png
      step2_confusion_matrices.png
      step2_roc_curves.png
      overfitting_diagnostics.csv
      # Step 3 — Horizon Sweep
      step3_horizon_sweep.csv
      step3_horizon_sweep.png
      step3_confusion_matrices_per_H.png
      # Step 4 — Final Model
      step4_final_model.csv
      step4_confusion_matrix.png
      step4_precision_recall_curve.png
      step4_learning_curve.png
      step4_per_fold_barplot.png
      step4_shift_timeline.png
```

**Do NOT modify any existing scripts (01–06).**

---

## Step 1 — Compare Regime Definitions

Test 5 regime definitions. For each, compute the binary shift label and run the same walk-forward evaluation (LR, class_weight='balanced', C=0.001) across H values to see which definition produces the most predictable shifts.

### Regime Definition A: Original 4 Regimes (baseline)
```python
# 3 thresholds: 15, 20, 30
# regime = 1 (Calm) if VIX <= 15
# regime = 2 (Normal) if 15 < VIX <= 20
# regime = 3 (Tense) if 20 < VIX <= 30
# regime = 4 (Crisis) if VIX > 30
```
This is the current setup. Lots of boundary crossings, especially around VIX=15 and VIX=20.

### Regime Definition B: 2 Regimes, threshold at 20
```python
# 1 threshold: 20
# regime = 0 (Calm) if VIX <= 20
# regime = 1 (Volatile) if VIX > 20
regime_t = (VIX_t > 20).astype(int)
```
Only one boundary. VIX=20 is the most commonly used threshold in practice. Fewer shift events, but each one is more meaningful.

### Regime Definition C: 2 Regimes, threshold at 25
```python
regime_t = (VIX_t > 25).astype(int)
```
Higher threshold — only triggers in genuinely stressed markets. Even fewer shift events but potentially very clean signal.

### Regime Definition D: 2 Regimes with Hysteresis (threshold 20, buffer 2)
```python
# Hysteresis: regime changes only if VIX crosses threshold by at least 'buffer' points.
# Once in Volatile: stays Volatile until VIX drops below (threshold - buffer) = 18
# Once in Calm: stays Calm until VIX rises above (threshold + buffer) = 22
buffer = 2
regime = np.zeros(len(VIX), dtype=int)
regime[0] = int(VIX.iloc[0] > 20)
for t in range(1, len(VIX)):
    if regime[t-1] == 0:  # currently Calm
        regime[t] = 1 if VIX.iloc[t] > (20 + buffer) else 0
    else:  # currently Volatile
        regime[t] = 0 if VIX.iloc[t] < (20 - buffer) else 1
```
This eliminates the oscillation problem entirely. VIX must move 2 points beyond the threshold before a shift is registered.

### Regime Definition E: Smoothed 4 Regimes (5-day majority vote)
```python
# Use original 4-class regime, but smooth with 5-day majority vote.
# regime_smoothed_t = mode(regime[t-2:t+3])  — centered 5-day window
# For look-ahead safety: use backward-looking window instead:
# regime_smoothed_t = mode(regime[t-4:t+1])  — last 5 days including today
from scipy.stats import mode
regime_smoothed = regime_raw.rolling(5, min_periods=1).apply(
    lambda x: mode(x, keepdims=False)[0]
).astype(int)
```
Keeps the 4-regime structure but eliminates 1-2 day flickers.

### For each regime definition:

Compute the binary shift label with `any(...)` formulation for H ∈ {3, 5, 7, 10}:
```python
label_t = 1 if any(regime[t+1:t+H+1] != regime_t) else 0
```

Log per regime definition × H:
```
regime_def, H, n_days, n_shift, n_no_shift, shift_pct,
avg_recall, avg_precision, avg_f1, avg_accuracy
```

Use the same LR baseline model for all comparisons (to isolate the effect of the regime definition):
```python
LogisticRegression(C=0.001, class_weight='balanced', penalty='l2', 
                   max_iter=2000, random_state=42)
```

Use the Group B features from Plan 9 (the winner):
```python
features = ['VIX_MA20', 'max_VIX_window', 'min_VIX_window', 'VIX_slope_20', 
            'VIX_rolling_std_10', 'VIX_dist_nearest', 'VIX_dist_upper', 'VIX_dist_lower']
```

**IMPORTANT:** For definitions B, C, D — the threshold distance features must be recomputed relative to the NEW thresholds (not the old 15/20/30). For definition B: distance to threshold 20 only. For definition C: distance to threshold 25. For definition D: distance to thresholds 18 and 22 (the effective hysteresis boundaries).

### Walk-forward setup: same 12 folds as always.

### Output:
- `step1_regime_comparison.csv` — one row per regime_def × H
- `step1_regime_comparison.png` — line plot: F1 vs H, one line per regime definition
- `step1_confusion_matrices.png` — combined figure: one confusion matrix per regime definition, at the best H. Shows how many shifts each definition produces and how well the baseline LR detects them. Layout: 1 row × 5 columns (one per definition). Aggregated across 12 folds.
- `step1_shift_distributions.png` — bar chart showing shift% per regime definition × H. Visualizes how much "cleaner" simplified definitions are.

### Print summary:
```
=== Step 1: Regime Definition Comparison ===

Regime Def    | Thresholds  | H=3          | H=5          | H=10
              |             | shift% | F1  | shift% | F1  | shift% | F1
--------------+-------------+--------+-----+--------+-----+--------+----
A: 4 regimes  | 15,20,30    | XX%    | 0.XX| XX%    | 0.XX| XX%    | 0.XX
B: 2r, thr=20 | 20          | XX%    | 0.XX| XX%    | 0.XX| XX%    | 0.XX
C: 2r, thr=25 | 25          | XX%    | 0.XX| XX%    | 0.XX| XX%    | 0.XX
D: hysteresis | 18/22       | XX%    | 0.XX| XX%    | 0.XX| XX%    | 0.XX
E: smoothed 4r| 15,20,30 sm | XX%    | 0.XX| XX%    | 0.XX| XX%    | 0.XX

Best for H=5: [definition X] with F1 = 0.XX
```

---

## Step 2 — Full Weight Optimization on Best Regime Definition

Take the best regime definition from Step 1 (at the professor's target H — try H=5 first, fall back to H=3 or H=7) and run the complete weight optimization that was cut short in Plan 9.

### Full grids (what Plan 9 was supposed to do):

```python
# --- Logistic Regression: FULL grid ---
lr_grid = {
    'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    'class_weight': [
        'balanced',
        {0: 1, 1: 1},
        {0: 1, 1: 2},
        {0: 1, 1: 3},
        {0: 1, 1: 5},
        {0: 1, 1: 10},
        {0: 1, 1: 15},
        {0: 1, 1: 20},
    ],
    'penalty': ['l1', 'l2'],
    'solver': ['saga'],
    'max_iter': [2000],
}
# = 6C × 8cw × 2pen = 96 combos × 3-fold inner CV = 288 fits per outer fold

# --- Random Forest: FULL grid ---
rf_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 10, None],
    'class_weight': [
        'balanced',
        'balanced_subsample',
        {0: 1, 1: 2},
        {0: 1, 1: 5},
        {0: 1, 1: 10},
    ],
    'min_samples_leaf': [1, 5, 10],
}
# = 3 × 4 × 5 × 3 = 180 combos

# --- XGBoost: FULL grid (with regularization) ---
# scale_pos_weight computed per fold: base_ratio * factor
xgb_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'scale_pos_weight': [],  # filled per fold: [ratio*0.5, ratio*1.0, ratio*1.5, ratio*2.0, ratio*3.0, ratio*5.0]
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [1.0, 5.0, 10.0],
}
# Very large grid. To manage runtime: use RandomizedSearchCV with n_iter=200 instead of full GridSearchCV.
# This samples 200 random combinations per fold instead of exhaustive search.

# --- Gradient Boosting: via WeightedGB wrapper ---
gb_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'shift_weight': [1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0],
    'min_samples_leaf': [5, 10],
}
# = 3 × 3 × 3 × 7 × 2 = 378 combos → use RandomizedSearchCV n_iter=100

# --- SVM: FULL grid ---
svm_grid = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'class_weight': [
        'balanced',
        {0: 1, 1: 2},
        {0: 1, 1: 5},
        {0: 1, 1: 10},
    ],
}
# = 5 × 4 × 4 = 80 combos

# --- MLP: via WeightedMLP wrapper ---
mlp_grid = {
    'hidden_layer_sizes': [(32,), (64, 32), (128, 64, 32), (64, 32, 16)],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'shift_weight': [1, 2, 5, 10, 15, 20],
    'learning_rate_init': [0.001, 0.01],
    'max_iter': [500],
}
# = 4 × 4 × 6 × 2 = 192 combos → use RandomizedSearchCV n_iter=80
```

### Implementation:
- Use **GridSearchCV** for LR, RF, SVM (manageable grid sizes).
- Use **RandomizedSearchCV** for XGB, GB, MLP (large grids, sample n_iter combinations).
- All with **3-fold StratifiedKFold** inner CV, **scoring='f1'**.
- Run **per fold** (12 outer folds).
- For XGBoost: compute `scale_pos_weight` candidates per fold based on training data ratio.

### LSTM re-optimization:
```python
# Run architecture search on FOLDS 1, 6, and 12 (not just fold 1)
# to capture different data sizes. Take config that has best AVERAGE F1 across the 3 folds.
lstm_configs = [
    {'hidden_dim': 16, 'num_layers': 1, 'dropout': 0.2},
    {'hidden_dim': 32, 'num_layers': 1, 'dropout': 0.3},
    {'hidden_dim': 64, 'num_layers': 1, 'dropout': 0.3},
    {'hidden_dim': 32, 'num_layers': 2, 'dropout': 0.3},
    {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.4},
]
pos_weight_values = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0]
lr_values = [0.0005, 0.001, 0.005]
# = 5 × 7 × 3 = 105 configs × 3 search folds = 315 training runs
# Then run best config across all 12 folds.
```

### Output:
- `step2_full_weight_optimization.csv` — per fold × model: best params, metrics
- `step2_best_weights_per_fold.csv` — per fold × model: which class_weight won
- `step2_model_comparison_barplot.png` — grouped bar chart: Recall, Precision, F1 per model (only non-rejected models)
- `step2_confusion_matrices.png` — combined figure: one confusion matrix per model (aggregated across 12 folds). Layout: 2 rows × 5 columns (10 models). Rejected models shown with red border and "REJECTED (overfit)" label.
- `step2_roc_curves.png` — ROC curve per model (from predict_proba), aggregated across folds
- `overfitting_diagnostics.csv` — per fold × model: train_recall, train_f1, test_recall, test_f1, gap_f1

---

## Step 3 — Horizon Sweep with Best Model + Regime Definition

Take the best model from Step 2 (with its optimized weights) and sweep H ∈ {3, 5, 7, 10, 15, 20} using the best regime definition.

For each H:
- Run 12-fold walk-forward.
- Find optimal threshold via PR curve on training data per fold.
- Record metrics at both threshold=0.5 and optimal threshold.

### Output:
- `step3_horizon_sweep.csv`
- `step3_horizon_sweep.png`
- `step3_confusion_matrices_per_H.png` — combined figure: one confusion matrix per H value (at the best model). Shows how the shift detection changes with horizon. Layout: 1 row × 6 columns (H=3,5,7,10,15,20).

---

## Step 4 — Final Model Report

Run the best configuration (regime definition + model + weights + H + threshold) one final time with full logging.

### Output:
- `step4_final_model.csv` — per-fold detailed results (including train metrics and gap)
- `step4_confusion_matrix.png` — final model confusion matrix (aggregated across 12 folds)
- `step4_precision_recall_curve.png`
- `step4_learning_curve.png` — learning curve on Fold 6 (training size vs. train/test F1)
- `step4_per_fold_barplot.png` — bar chart showing F1 per fold (12 bars), with horizontal line at mean. Shows stability across different test years.
- `step4_shift_timeline.png` — timeline plot for one representative test year: actual shifts (ground truth) vs. predicted shifts. Shows WHERE the model succeeds and fails. Use the test year with median F1 as the representative.

### Print comprehensive summary:
```
=== Plan 10 Final Results ===

Regime Definition: [name] (thresholds: [values])
Model:             [name]
Best Params:       [params including class_weight]
Features:          [list] (N features)
Horizon:           H=[value]
Threshold:         [value]

Performance (avg ± std, 12 folds):
  Accuracy:  0.XXX ± 0.XXX
  Recall:    0.XXX ± 0.XXX
  Precision: 0.XXX ± 0.XXX
  F1:        0.XXX ± 0.XXX
  FAR:       0.XXX ± 0.XXX

vs. Plan 9 best (4 regimes, LR, H=20):  F1 change: +/- 0.XXX
vs. Plan 7 RF (4 regimes, H=10):        F1 change: +/- 0.XXX

Key finding: [Does the simplified regime definition enable >95% accuracy at small H?]
```

---

## Constraints

### Overfitting Prevention (apply to ALL steps)

These measures are MANDATORY and must be implemented throughout the script:

**A. Feature Correlation Handling:**
Before training any model, compute the correlation matrix of the feature set. If any pair has |r| > 0.90, drop one of the two (keep the one with higher univariate F1 with the target). Log which features were dropped and why. This applies per fold (correlations can shift over time).

```python
from sklearn.feature_selection import f_classif

def drop_correlated(X_train, y_train, feature_names, threshold=0.90):
    corr = np.abs(np.corrcoef(X_train.T))
    to_drop = set()
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if corr[i, j] > threshold and feature_names[j] not in to_drop:
                # Keep the feature with higher F-statistic
                f_i = f_classif(X_train[:, [i]], y_train)[0][0]
                f_j = f_classif(X_train[:, [j]], y_train)[0][0]
                drop = feature_names[j] if f_i >= f_j else feature_names[i]
                to_drop.add(drop)
    kept = [f for f in feature_names if f not in to_drop]
    return kept, to_drop
```

**B. Train-Test Performance Gap Logging:**
For EVERY model evaluation, log BOTH training set performance and test set performance. A gap > 0.15 in F1 is a warning sign.

```python
# After fitting model m on X_tr_s, y_tr:
y_pred_train = m.predict(X_tr_s)
train_metrics = wf_metrics(y_tr, y_pred_train)
test_metrics  = wf_metrics(y_te, y_pred)
gap = train_metrics['f1'] - test_metrics['f1']
# Log: fold, model, train_f1, test_f1, gap
```

Save to: `overfitting_diagnostics.csv` with columns:
```
step, fold, model, train_recall, train_f1, test_recall, test_f1, gap_f1
```

**C. Feature Selection within Pipeline (Step 2 only):**
For the LR grid search, add a SelectKBest step inside the pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

pipe = Pipeline([
    ('select', SelectKBest(f_classif)),
    ('model', LogisticRegression(random_state=42))
])
pipe_grid = {
    'select__k': [3, 5, 'all'],  # test with fewer features too
    'model__C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    'model__class_weight': ['balanced', {0:1,1:2}, {0:1,1:5}, {0:1,1:10}, {0:1,1:15}, {0:1,1:20}],
    'model__penalty': ['l1', 'l2'],
    'model__solver': ['saga'],
    'model__max_iter': [2000],
}
```

This lets the inner CV decide whether using fewer features generalizes better.

**D. Regime Definition Robustness Check:**
In Step 1, after finding the best regime definition, verify it is not overfit by checking:
- Is it the best definition for EACH H value, or only for one specific H?
- Is it the best in the majority of individual folds (>= 8 out of 12)?
Log these checks.

**E. Automatic Overfitting Rejection Rule:**
After Step 2, for each model: if the average train-test F1 gap across 12 folds exceeds 0.20, that model is DISQUALIFIED from Steps 3–4. It is logged as "rejected due to overfitting" and not considered for the final model. This is an active prevention measure, not just a diagnostic.

```python
# After all folds for a given model:
avg_gap = np.mean([r['gap_f1'] for r in fold_diagnostics])
if avg_gap > 0.20:
    print(f"  WARNING: {model_name} REJECTED — avg train-test gap = {avg_gap:.3f} > 0.20")
    rejected_models.add(model_name)
```

**F. Stability Check via Fold-Variance:**
For each model, compute the standard deviation of F1 across 12 folds. If std(F1) > 0.20, the model is unstable and flagged (not rejected, but flagged). Overfitting often manifests as high variance across folds — the model fits training data well in some folds but fails in others.

**G. Learning Curve for Final Model (Step 4 only):**
After selecting the final model in Step 4, generate a learning curve: train on 20%, 40%, 60%, 80%, 100% of the training data (for a single representative fold, e.g., Fold 6) and plot training F1 vs. test F1. If the lines converge, the model generalizes well. If they diverge, overfitting persists even with the best hyperparameters.

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train, y_train,
    train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
    cv=3, scoring='f1', random_state=42
)
```
Save as `learning_curve.png`.

### Additional Models: HMM and GRU

Add these two models to the comparison in Steps 2–3:

**HMM (Hidden Markov Model):**
```python
from hmmlearn import GaussianHMM

# HMM is unsupervised — it models the sequence of observations and infers
# hidden states. We use it differently from the other models:
# 1. Fit GaussianHMM with n_components=2 on training VIX sequence.
# 2. Decode the most likely state sequence (Viterbi) for the test set.
# 3. A "shift" is predicted when the decoded state at t differs from state at t-1.
# 4. Tune n_components ∈ {2, 3, 4} and covariance_type ∈ {'diag', 'full'}.
# Note: HMM does NOT use class_weight — it learns transition probabilities directly.
# Install if needed: pip install hmmlearn --break-system-packages
```

**GRU (Gated Recurrent Unit):**
```python
# Same architecture as LSTM but replace nn.LSTM with nn.GRU.
# GRU is simpler (fewer parameters) and often generalizes better on small datasets.
# Use the same training procedure: BCEWithLogitsLoss + pos_weight + early stopping.
class ShiftGRU(nn.Module):
    def __init__(self, seq_dim=4, tab_dim=5, hidden_dim=32,
                 num_layers=1, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden_dim, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + tab_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
    def forward(self, seq, tab):
        _, h = self.gru(seq)
        h_last = h[-1]
        return self.head(torch.cat([h_last, tab], dim=1)).squeeze(1)

# Same grid search as LSTM (architecture × pos_weight × lr),
# same fold search strategy (folds 1, 6, 12).
```

### Other Constraints

1. Do NOT modify existing scripts (01–06).
2. Walk-forward setup identical: 12 folds, test years 2013–2024.
3. StandardScaler fit on training data only per fold.
4. XGBoost scale_pos_weight computed per fold.
5. Threshold optimization on training data only (PR curve).
6. LSTM early stopping on validation split, no test data leakage.
7. Random seed = 42 everywhere.
8. Code comments in English.
9. For RandomizedSearchCV: set `random_state=42` and `n_iter` as specified.
10. Print progress updates per step, per model, per fold.
11. Expected runtime: potentially several hours due to full grids. Print ETA estimates.
12. If total runtime exceeds 3 hours, prioritize: LR full grid (most important) > XGB randomized > RF full > SVM > GB > MLP > LSTM. Skip lower-priority models if time is critical, but always complete LR and XGB.
