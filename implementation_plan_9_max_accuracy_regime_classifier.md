# Implementation Plan 9: Maximum Regime Classifier Accuracy

## Context

Previous experiments (Plans 7–8) established:
- Binary shift label (Variant A) with `any(...)` formulation works.
- Best result so far: RF with H=10 achieves Recall 0.741, F1 0.715 (honest classifier).
- "Optimized" LR with H=20, threshold=0.35 achieves Recall 0.996 but FAR=0.984 (nearly trivial).
- Level features (VIX_MA20, max/min window) dominate; dynamics features hurt LR recall.
- Professor's input: >95% accuracy is achievable with smaller H.

**Goal of this plan:** Push accuracy and F1 above 95% at H=5, using every available technique. Systematic, incremental, with clear ablation at each step.

## File Structure

```
experiments/
  regime_classifier_v2/
    06_max_accuracy.py              # NEW — main script, runs all experiments
results/
  regime_classifier_v2/
    max_accuracy/
      step1_sequence_features.csv
      step2_weight_optimization.csv
      step2_best_weights_per_fold.csv
      step3_model_comparison.csv
      step4_ensemble.csv
      step5_horizon_sweep.csv
      step6_final_model.csv
      final_confusion_matrix.png
      final_precision_recall_curve.png
      model_comparison_barplot.png
      horizon_sweep_plot.png
      roc_curves.png
```

**Do NOT modify any existing scripts (01–05).**

---

## Design Principle: Incremental with Ablation

Each step adds ONE thing. Results are logged per step so we can see exactly what helped.

---

## Step 1 — Richer Feature Engineering

The current features are mostly point-in-time summaries. This step adds features designed to capture the DYNAMICS leading up to a shift.

### 1a. Sequence-Aware Features (what LSTM would see, but as tabular features)

Compute over the lookback window L=20:

```python
# VIX trajectory features
VIX_range_20     = VIX.rolling(20).max() - VIX.rolling(20).min()  # range within window
VIX_pct_above_MA = (VIX > VIX_MA20).rolling(20).mean()           # % of days above MA
VIX_trend_strength = abs(VIX_slope_20) / VIX_rolling_std_20       # normalized trend

# Rate of change features
delta_VIX_MA5    = delta_VIX.rolling(5).mean()    # smoothed daily change
delta_VIX_std5   = delta_VIX.rolling(5).std()     # volatility of daily changes
VIX_acceleration = VIX_slope_5 - VIX_slope_5.shift(5)  # is the trend accelerating?

# Distance to regime thresholds (critical for shift prediction)
# Import or replicate the exact VIX thresholds from the codebase
# Example thresholds (adjust to match actual values):
#   Calm/Normal boundary:  threshold_1
#   Normal/Tense boundary: threshold_2
#   Tense/Crisis boundary: threshold_3
VIX_dist_to_nearest_threshold = min(|VIX - threshold_i|) for each threshold_i
VIX_dist_to_upper_threshold   = next_higher_threshold - VIX  # how close to upgrading
VIX_dist_to_lower_threshold   = VIX - next_lower_threshold   # how close to downgrading

# Cross-asset features
TLT_return_5  = TLT_price / TLT_price.shift(5) - 1   # bond momentum (flight to safety)
GLD_return_5  = GLD_price / GLD_price.shift(5) - 1    # gold momentum (fear indicator)
SPY_VIX_corr_20 = SPY_returns.rolling(20).corr(VIX.pct_change())  # correlation regime
```

### 1b. Feature Groups

Organize features into testable groups:

```python
# Group A: Existing best (from Plan 7 optimization)
group_A = ['VIX_MA20', 'max_VIX_window', 'min_VIX_window', 'VIX_slope_20', 'VIX_rolling_std_10']

# Group B: Group A + threshold distance features
group_B = group_A + ['VIX_dist_to_nearest_threshold', 'VIX_dist_to_upper_threshold', 'VIX_dist_to_lower_threshold']

# Group C: Group B + trajectory features
group_C = group_B + ['VIX_range_20', 'VIX_pct_above_MA', 'VIX_trend_strength', 'VIX_acceleration']

# Group D: Group C + cross-asset features
group_D = group_C + ['TLT_return_5', 'GLD_return_5', 'SPY_return_5', 'vol_ratio', 'SPY_VIX_corr_20']

# Group E: Group D + smoothed dynamics
group_E = group_D + ['delta_VIX_MA5', 'delta_VIX_std5']

# Group F: Best subset (determined by Step 2 feature importance — filled in after running)
```

Test each group with the same model (LogisticRegression, C=0.001, balanced) on H=10, 12-fold WF.

**Output:** `step1_sequence_features.csv` — one row per group with avg Recall, Precision, F1, Accuracy.

---

## Step 2 — Systematic Weight & Hyperparameter Optimization

**This is critical.** Previous experiments used default or single-value class weights. The weights must be PERFECTLY tuned — not just `class_weight='balanced'`, but a full search over the weight space to find the exact configuration that maximizes F1 (or Recall at acceptable Precision).

### 2a. Class Weight Grid Search

For each model type, sweep over class weights systematically. This is done per fold on training data using internal 3-fold cross-validation (not on the test set).

```python
from sklearn.model_selection import GridSearchCV

# --- Logistic Regression ---
lr_param_grid = {
    'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    'class_weight': [
        'balanced',
        {0: 1, 1: 1},       # no weighting
        {0: 1, 1: 2},       # shift 2x more important
        {0: 1, 1: 3},       # shift 3x more important
        {0: 1, 1: 5},       # shift 5x more important
        {0: 1, 1: 10},      # shift 10x more important
        {0: 1, 1: 15},      # aggressive shift focus
        {0: 1, 1: 20},      # very aggressive
    ],
    'penalty': ['l1', 'l2'],
    'solver': ['saga'],      # supports both l1 and l2
}
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=2000, random_state=42),
    lr_param_grid, scoring='f1', cv=3, n_jobs=-1
)

# --- Random Forest ---
rf_param_grid = {
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
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid, scoring='f1', cv=3, n_jobs=-1
)

# --- XGBoost ---
# scale_pos_weight is the equivalent of class_weight for XGBoost
# Compute base ratio per fold, then multiply by a factor
xgb_param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'scale_pos_weight': [],  # filled per fold: [ratio*0.5, ratio*1.0, ratio*1.5, ratio*2.0, ratio*3.0]
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1, 1.0],       # L1 regularization
    'reg_lambda': [1.0, 5.0, 10.0],   # L2 regularization
}
# Per fold:
base_ratio = n_neg / n_pos
xgb_param_grid['scale_pos_weight'] = [base_ratio * f for f in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]]

xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
    xgb_param_grid, scoring='f1', cv=3, n_jobs=-1
)

# --- Gradient Boosting ---
gb_param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 4, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_samples_leaf': [5, 10, 20],
    'subsample': [0.8, 1.0],
}
# GradientBoostingClassifier does not support class_weight directly.
# Use sample_weight in fit(): weight_i = class_weight[y_i]
# Sweep: shift_weight in [1, 2, 3, 5, 10, 15, 20]
# sample_weights = np.where(y_train == 1, shift_weight, 1.0)

# --- SVM ---
svm_param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'class_weight': [
        'balanced',
        {0: 1, 1: 2},
        {0: 1, 1: 5},
        {0: 1, 1: 10},
    ],
}
svm_grid = GridSearchCV(
    SVC(kernel='rbf', probability=True, random_state=42),
    svm_param_grid, scoring='f1', cv=3, n_jobs=-1
)

# --- MLP ---
mlp_param_grid = {
    'hidden_layer_sizes': [(32,), (64, 32), (128, 64, 32), (64, 32, 16)],
    'alpha': [0.0001, 0.001, 0.01, 0.1],      # L2 regularization
    'learning_rate_init': [0.001, 0.01],
    'batch_size': [32, 64],
}
# MLP does not support class_weight.
# Use sample_weight in fit(): weight_i = class_weight[y_i]
# Sweep: shift_weight in [1, 2, 5, 10]
```

### 2b. LSTM Weight Optimization

For the LSTM, class weights are implemented via `pos_weight` in BCEWithLogitsLoss:

```python
# Sweep pos_weight values:
pos_weight_values = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0]

# Also sweep architecture:
lstm_configs = [
    {'hidden_dim': 16, 'num_layers': 1, 'dropout': 0.2},
    {'hidden_dim': 32, 'num_layers': 1, 'dropout': 0.3},
    {'hidden_dim': 64, 'num_layers': 1, 'dropout': 0.3},
    {'hidden_dim': 32, 'num_layers': 2, 'dropout': 0.3},
    {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.4},
]

# Also sweep learning rate:
lr_values = [0.0005, 0.001, 0.005]

# Use first fold only for architecture/weight search (to save time).
# Then run the best config across all 12 folds.
```

### 2c. Per-Fold Weight Adaptation

CRITICAL: The class distribution changes across folds as the training set grows. The optimal class weight in Fold 1 (2006–2012) may differ from Fold 12 (2006–2023). Therefore:

```python
# For each fold:
# 1. Compute class distribution in training data
# 2. Run GridSearchCV with 3-fold internal CV on training data
# 3. Select best hyperparameters INCLUDING best class weight
# 4. Retrain on full training data with best params
# 5. Predict on test data
# 6. Log: fold, best_params, best_class_weight, metrics
```

This means the grid search runs INSIDE each walk-forward fold. Yes, this is slow. But it guarantees the weights are perfectly optimized for each fold's data distribution.

**Output:** `step2_weight_optimization.csv` — per fold, per model: best_params (including class_weight), Recall, Precision, F1, Accuracy.

Also log the best class weights per fold to see if they are stable or vary:
```
step2_best_weights_per_fold.csv:
fold, model, best_class_weight_or_scale_pos_weight, shift_ratio_in_training
```

---

## Step 3 — Model Comparison (with optimized weights)

Now compare all models using their INDIVIDUALLY OPTIMIZED weights from Step 2.

### Models (all with their best hyperparameters from Step 2):

```python
# The models dictionary is filled dynamically from Step 2 results.
# Each model uses its best hyperparameters found per fold.
# This is NOT a fixed config — it adapts per fold.
```

Additionally include models that were not grid-searched but use fixed good defaults:

```python
extra_models = {
    "ExtraTrees": ExtraTreesClassifier(class_weight='balanced', n_estimators=200, random_state=42),
}
```

### LSTM (separate implementation):

```python
# LSTM gets the RAW SEQUENCE as input, not just summary features.
# Input shape: (batch_size, sequence_length=20, n_raw_features)
# Raw features per timestep: VIX_close, delta_VIX, SPY_return, VIX_rolling_std_10
# Plus the tabular features from the best group appended after the LSTM output.

class ShiftLSTM(torch.nn.Module):
    def __init__(self, seq_input_dim, tabular_input_dim, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(seq_input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + tabular_input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, seq_input, tabular_input):
        lstm_out, (h_n, _) = self.lstm(seq_input)
        h_last = h_n[-1]  # last hidden state
        combined = torch.cat([h_last, tabular_input], dim=1)
        return self.classifier(combined)

# Training:
# - Binary cross-entropy loss with pos_weight (class imbalance)
# - Adam optimizer, lr=0.001
# - Early stopping on validation loss (use last 20% of training data as validation)
# - Max 50 epochs, patience=5
# - Batch size 64

# Sequence input per timestep t:
# For each day in [t-19, t-18, ..., t]:
#   [VIX_close_normalized, delta_VIX, SPY_daily_return, VIX_rolling_std_10]
# → shape (20, 4) per sample

# Tabular input:
# The best feature group from Step 1 (already computed summary features)
# → shape (n_features,) per sample
```

For all models: StandardScaler on training data only. 12-fold walk-forward. Threshold = 0.5 first.

**Output:** `step2_model_comparison.csv` — one row per model with avg Recall, Precision, F1, Accuracy, plus std.

Also generate:
- `model_comparison_barplot.png` — grouped bar chart (Recall, Precision, F1 per model)
- `roc_curves.png` — ROC curve per model (from predict_proba), aggregated across folds
- Feature importance for tree-based models → identify `group_F` (best subset)

---

## Step 4 — Ensemble Methods

Take the top 3 models from Step 3 and combine them.

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier

# Soft Voting: average predicted probabilities
voting_soft = VotingClassifier(
    estimators=[('model1', best1), ('model2', best2), ('model3', best3)],
    voting='soft'
)

# Stacking: use a meta-classifier on top of base model predictions
stacking = StackingClassifier(
    estimators=[('model1', best1), ('model2', best2), ('model3', best3)],
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=5  # internal cross-validation for stacking
)
```

If LSTM is among the top 3: create a manual ensemble by averaging LSTM's predicted probabilities with sklearn models' predicted probabilities. Then apply threshold.

**Output:** `step4_ensemble.csv`

---

## Step 5 — Horizon Sweep with Best Model

Take the best single model OR ensemble from Steps 3–4 and sweep across horizons:

```python
H_values = [3, 5, 7, 10, 15, 20]
```

For each H:
- Recompute labels with `any(future_regimes != regime_t)` and horizon H.
- Run 12-fold walk-forward with the best model.
- Record: class distribution, Recall, Precision, F1, Accuracy, FAR.

Also for each H: find the optimal threshold via Precision-Recall curve on training data (per fold), then apply to test data.

```python
from sklearn.metrics import precision_recall_curve

# Per fold, on training predictions:
precisions, recalls, thresholds = precision_recall_curve(y_train, y_prob_train)
# Find threshold that maximizes F1:
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]
# Apply to test:
y_pred_test = (y_prob_test >= best_threshold).astype(int)
```

**Output:** 
- `step5_horizon_sweep.csv` — one row per H with metrics at both threshold=0.5 and optimal threshold
- `horizon_sweep_plot.png` — F1 and Recall vs H, with two lines (default threshold vs optimal)

---

## Step 6 — Final Model Selection & Report

Based on Steps 1–5, select the final best configuration. Run it one more time with full logging.

**Output:**
- `step6_final_model.csv` — per-fold detailed results
- `final_confusion_matrix.png` — 2x2 CM aggregated across 12 folds
- `final_precision_recall_curve.png` — PR curve aggregated across folds

**Print comprehensive summary:**

```
=== Maximum Accuracy Experiment — Final Results ===

Configuration:
  Model:     [name]
  Features:  [list] (N features)
  Horizon:   H=[value]
  Threshold: [value] (optimized via PR curve / default 0.5)
  Label:     binary shift, any(...) formulation

Performance (avg ± std across 12 folds):
  Accuracy:  0.XXX ± 0.XXX
  Recall:    0.XXX ± 0.XXX
  Precision: 0.XXX ± 0.XXX
  F1:        0.XXX ± 0.XXX
  FAR:       0.XXX ± 0.XXX

Comparison to previous best results:
  vs. Plan 7 Phase 1 (LR, 3 feat, H=10):     F1 +0.XXX
  vs. Plan 7 Phase 4 RF (10 feat, H=10):      F1 +0.XXX
  vs. Plan 7 Optimized (LR, H=20, thr=0.35):  F1 +0.XXX (but FAR -0.XXX)

Ablation progression (this plan):
  Step 1 — Best feature group: [group X], F1 = 0.XXX
  Step 2 — Weight optimization: best weights for [model], F1 = 0.XXX
  Step 3 — Best model (with optimized weights): [name], F1 = 0.XXX (+0.XXX vs Step 1)
  Step 4 — Best ensemble: [type], F1 = 0.XXX (+0.XXX vs Step 3)
  Step 5 — Best horizon: H=[value], F1 = 0.XXX (+0.XXX vs Step 4)
  Step 6 — Final (with optimal threshold): F1 = 0.XXX
```

---

## Constraints

1. Do NOT modify existing scripts (01–05).
2. All walk-forward, labeling, and feature logic must be consistent with previous experiments.
3. StandardScaler fit on training data only per fold.
4. XGBoost `scale_pos_weight` computed per fold from training data.
5. LSTM: early stopping on validation loss, no test data leakage.
6. Threshold optimization: done on training data only (via internal CV or train split), never on test data.
7. Random seed = 42 everywhere (including PyTorch: `torch.manual_seed(42)`).
8. Code comments in English.
9. If LSTM requires PyTorch and it is not installed: `pip install torch --break-system-packages`.
10. Expected runtime: may be long (LSTM training × 12 folds × multiple H values). Print progress updates per step.
