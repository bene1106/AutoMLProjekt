# Ablation Study: Regime Classifier v2 — Isolating Strategy Effects

## Context

The full experiment (`experiments/regime_classifier_v2/`) has already been run with ALL strategies combined (new label + new features + class_weight + XGBoost). Results are in `results/regime_classifier_v2/`.

**Problem:** We cannot tell which improvement came from which strategy. This ablation study fills in the missing intermediate steps.

**Goal:** Produce a clean ablation table showing the isolated contribution of each strategy.

## File Structure

```
experiments/
  regime_classifier_v2/
    01_build_dataset.py          # (already exists — do not modify)
    02_train_evaluate.py         # (already exists — do not modify)
    03_analyze_results.py        # (already exists — do not modify)
    04_ablation_study.py         # NEW — runs all 4 phases, produces comparison
results/
  regime_classifier_v2/
    ablation_results.csv         # NEW — per phase, per fold, per model
    ablation_summary.csv         # NEW — aggregated comparison table
    ablation_comparison.png      # NEW — bar chart comparing phases
```

**Do NOT modify the existing 3 scripts.** The ablation study is a single new script that reuses the same data loading and regime label computation.

---

## The 4 Phases

Each phase uses the SAME walk-forward setup (12 folds, same year splits) and the SAME binary shift label (Strategy 1, H=10, `any(...)` formulation). The label is not a variable — it stays fixed across all phases.

What changes between phases is: which features are used, whether class_weight is applied, and which model is trained.

---

### Phase 1 — New Label Only (Strategies 1+2, minimal features)

**Purpose:** Isolate the effect of the label change. Use only the features that the OLD classifier had available (or close equivalents), but with the NEW binary shift label.

**Model:** LogisticRegression, NO class_weight (default), max_iter=1000

**Features (old-style, VIX level only):**
```python
phase1_features = [
    'VIX_MA20',         # VIX level context
    'max_VIX_window',   # window summary (from Strategy 2, but simple)
    'min_VIX_window',   # window summary
]
```

These are basic VIX-level features — no dynamics, no z-scores, no return-based features. This is the closest approximation to "old features on new label" while still including the window structure from Strategy 2 (since Strategy 1 and 2 are inseparable by design).

**What this answers:** "Does changing the label alone already improve shift detection compared to the old 4-class approach?"

---

### Phase 2 — Add Dynamics Features (Strategy 4)

**Purpose:** Isolate the effect of the new dynamics features on top of Phase 1.

**Model:** LogisticRegression, NO class_weight, max_iter=1000

**Features (Phase 1 + dynamics):**
```python
phase2_features = [
    # Phase 1 features
    'VIX_MA20',
    'max_VIX_window',
    'min_VIX_window',
    # NEW: dynamics features (Strategy 4)
    'z_VIX',
    'delta_VIX',
    'VIX_slope_5',
    'VIX_slope_20',
    'VIX_rolling_std_10',
    'SPY_return_5',
    'vol_ratio',
]
```

This is the full feature set from the original experiment.

**What this answers:** "Do the dynamics features (z_VIX, slopes, vol_ratio) improve over level-only features?"

---

### Phase 3 — Add Class Weighting (Strategy 5)

**Purpose:** Isolate the effect of handling class imbalance.

**Model:** LogisticRegression, class_weight='balanced', max_iter=1000

**Features:** Same as Phase 2 (full feature set).

**What this answers:** "Does class_weight='balanced' improve recall on the minority class?"

---

### Phase 4 — Stronger Models (Strategy 3)

**Purpose:** Show the effect of using non-linear models.

**Models:** RandomForest (class_weight='balanced', n_estimators=100) and XGBoost (scale_pos_weight per fold, n_estimators=100)

**Features:** Same as Phase 2/3 (full feature set).

**What this answers:** "Do non-linear models extract more signal from the same features?"

Note: Phase 4 results should match the existing results from `02_train_evaluate.py` (same setup). This serves as a sanity check. Small differences due to random seeds are acceptable.

---

## Implementation Details

### Data Preparation

Reuse the same data loading logic from `01_build_dataset.py`. Either:
- Import from it directly if structured as importable functions, OR
- Re-implement the same loading and feature computation inline.

All features from Phase 2 onwards must be computed regardless — Phase 1 simply selects a subset.

The label computation must be identical: H=10, `any(future_regimes != regime_t)`.

### Walk-Forward Setup

Identical to the existing experiment:
```
Fold 1:  Train 2006–2012, Test 2013
...
Fold 12: Train 2006–2023, Test 2024
```

- StandardScaler fit on training set only, transform test set.
- Drop last H=10 rows of each train/test split (no label available).
- Drop NaN rows from rolling window warm-up.
- XGBoost `scale_pos_weight` computed per fold from training data.

### Evaluation

Per phase, per fold, compute:
```python
precision_shift  = precision_score(y_test, y_pred, pos_label=1)
recall_shift     = recall_score(y_test, y_pred, pos_label=1)
f1_shift         = f1_score(y_test, y_pred, pos_label=1)
accuracy         = accuracy_score(y_test, y_pred)
```

### Output 1 — ablation_results.csv

One row per phase × fold × model:
```
phase, model, fold, precision_shift, recall_shift, f1_shift, accuracy, n_features, features_used
```

### Output 2 — ablation_summary.csv

Aggregated across 12 folds (mean ± std):
```
phase, model, description, n_features, avg_recall, std_recall, avg_precision, std_precision, avg_f1, std_f1, avg_accuracy, std_accuracy
```

With these rows:
```
Phase 1, LogReg,              "New label + level features only",           3,  ...
Phase 2, LogReg,              "Phase 1 + dynamics features",              10, ...
Phase 3, LogReg (balanced),   "Phase 2 + class_weight balanced",          10, ...
Phase 4, RandomForest,        "Phase 3 features + RF model",              10, ...
Phase 4, XGBoost,             "Phase 3 features + XGBoost model",         10, ...
```

### Output 3 — ablation_comparison.png

Grouped bar chart:
- X-axis: the 5 rows from the summary table (Phase 1 LR, Phase 2 LR, Phase 3 LR balanced, Phase 4 RF, Phase 4 XGB)
- Y-axis: metric value (0 to 1)
- Bars: Recall, Precision, F1 (grouped per phase)
- Horizontal dashed line at the naive "always predict 0" baseline accuracy (~0.455)
- Title: "Ablation Study: Regime Shift Detection"
- Save as `results/regime_classifier_v2/ablation_comparison.png`

### Print Summary

```
=== Ablation Study Summary ===
Binary shift label: H=10, any(...) formulation

Phase 1 — New label + level features (LR, no balancing)
  Features: VIX_MA20, max_VIX_window, min_VIX_window
  Recall: 0.XX ± 0.XX | Precision: 0.XX ± 0.XX | F1: 0.XX ± 0.XX

Phase 2 — + dynamics features (LR, no balancing)
  Added: z_VIX, delta_VIX, VIX_slope_5, VIX_slope_20, VIX_rolling_std_10, SPY_return_5, vol_ratio
  Recall: 0.XX ± 0.XX | Precision: 0.XX ± 0.XX | F1: 0.XX ± 0.XX
  Delta vs Phase 1:  Recall +0.XX | F1 +0.XX

Phase 3 — + class_weight='balanced' (LR)
  Recall: 0.XX ± 0.XX | Precision: 0.XX ± 0.XX | F1: 0.XX ± 0.XX
  Delta vs Phase 2:  Recall +0.XX | F1 +0.XX

Phase 4 — Non-linear models (RF, XGBoost)
  RandomForest:  Recall 0.XX ± 0.XX | F1 0.XX ± 0.XX | Delta vs Phase 3: Recall +0.XX
  XGBoost:       Recall 0.XX ± 0.XX | F1 0.XX ± 0.XX | Delta vs Phase 3: Recall +0.XX

Naive Baseline ("always predict no-shift"):
  Accuracy: 0.XX | Recall: 0.00 | F1: 0.00
```

---

## Constraints

1. Do NOT modify existing files (`01_build_dataset.py`, `02_train_evaluate.py`, `03_analyze_results.py`).
2. All walk-forward, labeling, and feature logic must match the existing experiment exactly.
3. StandardScaler fit on training data only.
4. XGBoost `scale_pos_weight` computed per fold.
5. Random seed = 42 everywhere.
6. Code comments in English.
