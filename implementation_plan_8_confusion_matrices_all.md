# Implementation Plan 8: Confusion Matrices — Ablation + Optimized Model

## Context

We have two completed experiments for the regime classifier v2:

1. **Ablation study** (`04_ablation_study.py`): 5 phase-model combinations, showing isolated strategy effects.
2. **Optimization** (in `results/regime_classifier_v2/optimization/`): feature selection, hyperparameter tuning, threshold tuning, horizon sensitivity, final optimized model.

We need confusion matrices that tell the full story from baseline to optimized model.

## Goal

Generate confusion matrices (aggregated across 12 walk-forward folds) for **7 configurations** that show the complete progression:

```
1. Naive Baseline: always predict 0 (no shift)
2. Ablation Phase 1: LR, 3 level features, H=10, threshold=0.5
3. Ablation Phase 2: LR, 10 features, H=10, threshold=0.5
4. Ablation Phase 3: LR balanced, 10 features, H=10, threshold=0.5
5. Ablation Phase 4: RandomForest balanced, 10 features, H=10, threshold=0.5
6. Ablation Phase 4: XGBoost, 10 features, H=10, threshold=0.5
7. Optimized: LR (C=0.001, balanced, L2), 5 features, H=20, threshold=0.35
```

## File Structure

```
experiments/
  regime_classifier_v2/
    05_confusion_matrices.py        # NEW
results/
  regime_classifier_v2/
    cm_all_phases.png               # main output: all 7 CMs in one figure
    cm_individual/                  # folder with individual PNGs
      cm_1_naive_baseline.png
      cm_2_phase1_LR_level.png
      cm_3_phase2_LR_allfeat.png
      cm_4_phase3_LR_balanced.png
      cm_5_phase4_RF.png
      cm_6_phase4_XGB.png
      cm_7_optimized.png
    cm_summary.csv                  # TP, FP, TN, FN counts per configuration
```

**Do NOT modify any existing scripts (01–04, optimization script).**

## Implementation

### Step 1 — Data Preparation

Reuse the same data loading and feature computation logic from the existing scripts. Compute ALL features (the full set of 10), then select subsets per configuration.

Two label variants are needed:
- **H=10** for configurations 1–6
- **H=20** for configuration 7

Both use the `any(future_regimes != regime_t)` formulation.

### Step 2 — Re-run Predictions

For each of the 7 configurations, run the full 12-fold walk-forward loop and collect `y_true` and `y_pred` per fold. Aggregate by concatenating all folds.

Configuration details:

```python
configs = [
    {
        "name": "Naive Baseline (always 0)",
        "short": "naive_baseline",
        "H": 10,
        "model": None,  # just predict 0 for everything
        "features": None,
        "threshold": None,
    },
    {
        "name": "Phase 1: LR, level features",
        "short": "phase1_LR_level",
        "H": 10,
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "features": ['VIX_MA20', 'max_VIX_window', 'min_VIX_window'],
        "threshold": 0.5,
    },
    {
        "name": "Phase 2: LR, all features",
        "short": "phase2_LR_allfeat",
        "H": 10,
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "features": "all_10",  # all 10 features
        "threshold": 0.5,
    },
    {
        "name": "Phase 3: LR balanced, all features",
        "short": "phase3_LR_balanced",
        "H": 10,
        "model": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "features": "all_10",
        "threshold": 0.5,
    },
    {
        "name": "Phase 4: RandomForest",
        "short": "phase4_RF",
        "H": 10,
        "model": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
        "features": "all_10",
        "threshold": 0.5,
    },
    {
        "name": "Phase 4: XGBoost",
        "short": "phase4_XGB",
        "H": 10,
        "model": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        # scale_pos_weight computed per fold
        "features": "all_10",
        "threshold": 0.5,
    },
    {
        "name": "Optimized: LR, 5 feat, H=20, thr=0.35",
        "short": "optimized",
        "H": 20,
        "model": LogisticRegression(C=0.001, class_weight='balanced', penalty='l2', max_iter=1000, random_state=42),
        "features": ['VIX_MA20', 'max_VIX_window', 'min_VIX_window', 'VIX_slope_20', 'VIX_rolling_std_10'],
        "threshold": 0.35,
    },
]
```

**Important notes:**
- For the naive baseline: `y_pred = np.zeros_like(y_true)` — no model needed.
- For XGBoost: compute `scale_pos_weight = n_neg / n_pos` per fold from training data.
- For threshold != 0.5: use `model.predict_proba()` and apply custom threshold:
  ```python
  y_prob = model.predict_proba(X_test)[:, 1]
  y_pred = (y_prob >= threshold).astype(int)
  ```
- For threshold = 0.5: `model.predict()` is equivalent, but use `predict_proba` + threshold for consistency across all configurations.
- StandardScaler fit on training set only per fold.
- Drop NaN rows (rolling window warm-up) and rows without labels (last H rows).
- Configuration 7 uses H=20 → different label and different row count. Recompute the label with H=20 for this configuration only.

### Step 3 — Compute Aggregated Confusion Matrices

For each configuration, sum the confusion matrices across all 12 folds:

```python
from sklearn.metrics import confusion_matrix
import numpy as np

cm_total = np.zeros((2, 2), dtype=int)
for fold in range(12):
    cm_fold = confusion_matrix(y_true_fold, y_pred_fold, labels=[0, 1])
    cm_total += cm_fold
```

Labels: `[0, 1]` → `["No Shift", "Shift"]`.

### Step 4 — Plot: Combined Figure (main output)

Create a single figure with all 7 confusion matrices. Use 2 rows: top row = 4 matrices (naive + phases 1-3), bottom row = 3 matrices (phases 4a, 4b, optimized).

```python
fig, axes = plt.subplots(2, 4, figsize=(28, 10))

# Row 1: Naive, Phase 1, Phase 2, Phase 3
# Row 2: Phase 4 RF, Phase 4 XGB, Optimized, (empty subplot hidden)

# Use consistent color scale: same vmin=0, vmax=max across all CMs
# Use seaborn heatmap with annot=True, fmt='d', cmap='Blues'
# X-axis: Predicted (No Shift, Shift)
# Y-axis: Actual (No Shift, Shift)
# Title per subplot: configuration name

# Hide the empty 4th subplot in row 2
axes[1, 3].set_visible(False)

fig.suptitle('Regime Shift Detection: Confusion Matrices Across All Configurations\n(aggregated over 12 walk-forward folds)', fontsize=14)
plt.tight_layout()
plt.savefig('results/regime_classifier_v2/cm_all_phases.png', dpi=150, bbox_inches='tight')
plt.close()
```

### Step 5 — Plot: Individual Confusion Matrices

Save each of the 7 confusion matrices as a separate PNG in `results/regime_classifier_v2/cm_individual/`:

```python
# Create directory if not exists
os.makedirs('results/regime_classifier_v2/cm_individual', exist_ok=True)

for config in configs:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Shift', 'Shift'],
                yticklabels=['No Shift', 'Shift'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(config['name'])
    plt.tight_layout()
    plt.savefig(f'results/regime_classifier_v2/cm_individual/cm_{i+1}_{config["short"]}.png', dpi=150)
    plt.close()
```

### Step 6 — CSV Summary

Save `results/regime_classifier_v2/cm_summary.csv`:

```
config_name, H, threshold, n_features, TP, FP, TN, FN, recall, precision, f1, accuracy, false_alarm_rate
```

Where:
- `TP` = correctly detected shifts
- `FN` = missed shifts
- `FP` = false alarms (predicted shift, actually no shift)
- `TN` = correctly identified stable days
- `false_alarm_rate` = FP / (FP + TN)

### Step 7 — Print Summary

```
=== Confusion Matrix Summary (aggregated across 12 folds) ===

Config                                   | TP     | FN     | FP     | TN     | Recall | Prec   | F1     | FalseAlarm
-----------------------------------------|--------|--------|--------|--------|--------|--------|--------|----------
1. Naive (always 0)                      | 0      | XXXX   | 0      | XXXX   | 0.000  | —      | 0.000  | 0.000
2. Phase 1: LR, 3 feat, H=10            | XXXX   | XXXX   | XXXX   | XXXX   | 0.XXX  | 0.XXX  | 0.XXX  | 0.XXX
3. Phase 2: LR, 10 feat, H=10           | XXXX   | XXXX   | XXXX   | XXXX   | 0.XXX  | 0.XXX  | 0.XXX  | 0.XXX
4. Phase 3: LR bal, 10 feat, H=10       | XXXX   | XXXX   | XXXX   | XXXX   | 0.XXX  | 0.XXX  | 0.XXX  | 0.XXX
5. Phase 4: RF, 10 feat, H=10           | XXXX   | XXXX   | XXXX   | XXXX   | 0.XXX  | 0.XXX  | 0.XXX  | 0.XXX
6. Phase 4: XGB, 10 feat, H=10          | XXXX   | XXXX   | XXXX   | XXXX   | 0.XXX  | 0.XXX  | 0.XXX  | 0.XXX
7. Optimized: LR, 5 feat, H=20, thr=0.35| XXXX   | XXXX   | XXXX   | XXXX   | 0.XXX  | 0.XXX  | 0.XXX  | 0.XXX

Key progression:
  Label change effect (Naive → Phase 1):        Recall +X.XXX
  Feature effect (Phase 1 → Phase 2):           Recall X.XXX (change)
  Class weight effect (Phase 2 → Phase 3):      Recall X.XXX (change)
  Model effect (Phase 3 → Phase 4 RF):          Recall X.XXX (change)
  Full optimization (Phase 1 → Optimized):      Recall +X.XXX, F1 +X.XXX
```

## Constraints

1. Do NOT modify existing scripts (01–04, optimization script).
2. Identical walk-forward setup, feature computation, and label logic as existing scripts.
3. StandardScaler fit on training data only.
4. XGBoost `scale_pos_weight` computed per fold.
5. Random seed = 42.
6. Code comments in English.
