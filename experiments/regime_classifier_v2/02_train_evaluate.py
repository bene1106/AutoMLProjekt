"""
02_train_evaluate.py -- Regime Classifier v2: Walk-Forward Training & Evaluation

Walk-forward folds (expanding window, step=1 year):
  Fold  1: Train 2006-2012, Test 2013
  Fold  2: Train 2006-2013, Test 2014
  ...
  Fold 12: Train 2006-2023, Test 2024

Models trained per fold:
  - LogisticRegression  (class_weight='balanced')
  - RandomForest        (class_weight='balanced')
  - XGBoost             (scale_pos_weight computed per fold from training data)

Primary metric: Recall on label=1 (the classifier must not miss regime shifts).
Features are standardized using StandardScaler fitted on training data only.

Outputs saved to results/regime_classifier_v2/:
  fold_results.csv             - per-fold, per-model metrics
  shift_detection_summary.csv  - aggregated (mean +/- std across 12 folds)
  class_distribution.csv       - label balance per fold (train + test)
  feature_importances.csv      - RF + XGBoost importance per fold
  predictions_<model>.csv      - y_true / y_pred for confusion matrix generation

Run from project root (Implementierung1/):
  python experiments/regime_classifier_v2/02_train_evaluate.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (
    precision_score, recall_score, f1_score, accuracy_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    print("WARNING: xgboost not installed. Run: pip install xgboost")
    print("         XGBoost model will be skipped.")
    HAS_XGB = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results", "regime_classifier_v2")
DATASET_PATH = os.path.join(RESULTS_DIR, "dataset.csv")

FEATURE_COLS = [
    "VIX_MA20", "z_VIX", "delta_VIX", "VIX_slope_5", "VIX_slope_20",
    "VIX_rolling_std_10", "max_VIX_window", "min_VIX_window",
    "SPY_return_5", "vol_ratio",
]
TARGET_COL = "shift_label"

# Walk-forward: test years 2013 to 2024 (12 folds)
TEST_YEARS  = list(range(2013, 2025))
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_models(n_neg: int, n_pos: int) -> dict:
    """
    Instantiate all three models.
    XGBoost scale_pos_weight is computed per fold from training class counts
    to compensate for class imbalance.  This value changes fold-by-fold as
    the training set grows.
    """
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100,
            random_state=RANDOM_SEED,
        ),
    }
    if HAS_XGB:
        scale = n_neg / n_pos if n_pos > 0 else 1.0
        models["XGBoost"] = XGBClassifier(
            scale_pos_weight=scale,   # per-fold imbalance correction
            n_estimators=100,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            verbosity=0,
        )
    return models


# ---------------------------------------------------------------------------
# Per-fold evaluation
# ---------------------------------------------------------------------------

def evaluate_fold(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute shift-detection metrics for a single fold."""
    return {
        "precision_shift":   precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_shift":      recall_score(   y_true, y_pred, pos_label=1, zero_division=0),
        "f1_shift":          f1_score(       y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy":          accuracy_score( y_true, y_pred),
        "n_shifts_true":     int((y_true == 1).sum()),
        "n_shifts_predicted":int((y_pred == 1).sum()),
    }


# ---------------------------------------------------------------------------
# Main walk-forward loop
# ---------------------------------------------------------------------------

def main():
    # -----------------------------------------------------------------------
    # Load dataset produced by 01_build_dataset.py
    # -----------------------------------------------------------------------
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        print("Run 01_build_dataset.py first.")
        sys.exit(1)

    print(f"Loading dataset from {DATASET_PATH} ...")
    dataset = pd.read_csv(DATASET_PATH, index_col=0, parse_dates=True)
    print(f"  Shape      : {dataset.shape}")
    print(f"  Date range : {dataset.index[0].date()} to {dataset.index[-1].date()}")

    X_all = dataset[FEATURE_COLS].values
    y_all = dataset[TARGET_COL].values.astype(int)
    dates = dataset.index

    # -----------------------------------------------------------------------
    # Containers for results
    # -----------------------------------------------------------------------
    fold_records       = []
    class_dist_records = []
    importance_records = []

    # Accumulate predictions per model for post-hoc confusion matrices
    model_names_base = ["LogisticRegression", "RandomForest"] + (["XGBoost"] if HAS_XGB else [])
    all_preds = {
        name: {"y_true": [], "y_pred": [], "fold": []}
        for name in model_names_base
    }

    # -----------------------------------------------------------------------
    # Walk-forward loop
    # -----------------------------------------------------------------------
    print(f"\nStarting walk-forward evaluation ({len(TEST_YEARS)} folds) ...\n")
    print(f"{'Fold':>4} | {'Test':>4} | {'Train days':>10} | {'Test days':>9} | "
          f"{'Train pos':>9} | {'Test pos':>8}")
    print("-" * 60)

    for fold_idx, test_year in enumerate(TEST_YEARS):
        fold_num      = fold_idx + 1
        train_end_yr  = test_year - 1

        # Temporal split: all years up to (test_year-1) for training, test_year for test
        train_mask = dates.year <= train_end_yr
        test_mask  = dates.year == test_year

        X_train = X_all[train_mask];  y_train = y_all[train_mask]
        X_test  = X_all[test_mask];   y_test  = y_all[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Fold {fold_num}: insufficient data for test_year={test_year}, skipping.")
            continue

        # Class counts (used for scale_pos_weight and logging)
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        n_test_neg = int((y_test == 0).sum())
        n_test_pos = int((y_test == 1).sum())

        print(f"{fold_num:>4} | {test_year:>4} | {len(X_train):>10} | "
              f"{len(X_test):>9} | {n_pos:>9} | {n_test_pos:>8}")

        # Record class distribution for this fold
        class_dist_records.append({
            "fold": fold_num, "test_year": test_year,
            "train_n0": n_neg,  "train_n1": n_pos,
            "train_pct_shift": round(100 * n_pos / (n_neg + n_pos), 2),
            "test_n0":  n_test_neg, "test_n1": n_test_pos,
            "test_pct_shift": round(
                100 * n_test_pos / max(n_test_neg + n_test_pos, 1), 2
            ),
        })

        # Standardize: fit scaler on training set ONLY, transform both sets
        scaler     = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        # Instantiate models (XGBoost gets fold-specific scale_pos_weight)
        models = make_models(n_neg, n_pos)

        for model_name, model in models.items():
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)

            metrics = evaluate_fold(y_test, y_pred)
            fold_records.append({
                "fold":      fold_num,
                "test_year": test_year,
                "model":     model_name,
                **metrics,
            })

            # Accumulate predictions for confusion matrix
            all_preds[model_name]["y_true"].extend(y_test.tolist())
            all_preds[model_name]["y_pred"].extend(y_pred.tolist())
            all_preds[model_name]["fold"].extend([fold_num] * len(y_test))

            # Feature importances (tree-based models only)
            if model_name in ("RandomForest", "XGBoost"):
                for feat, imp in zip(FEATURE_COLS, model.feature_importances_):
                    importance_records.append({
                        "fold":       fold_num,
                        "test_year":  test_year,
                        "model":      model_name,
                        "feature":    feat,
                        "importance": imp,
                    })

        # Per-fold model summary line
        for rec in [r for r in fold_records if r["fold"] == fold_num]:
            print(f"        {rec['model']:22s}  "
                  f"recall={rec['recall_shift']:.3f}  "
                  f"prec={rec['precision_shift']:.3f}  "
                  f"f1={rec['f1_shift']:.3f}  "
                  f"acc={rec['accuracy']:.3f}")

    # -----------------------------------------------------------------------
    # Save CSV outputs
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Saving results ...")

    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv(os.path.join(RESULTS_DIR, "fold_results.csv"), index=False)
    print(f"  fold_results.csv           ({len(fold_df)} rows)")

    class_dist_df = pd.DataFrame(class_dist_records)
    class_dist_df.to_csv(os.path.join(RESULTS_DIR, "class_distribution.csv"), index=False)
    print(f"  class_distribution.csv     ({len(class_dist_df)} rows)")

    imp_df = pd.DataFrame(importance_records)
    imp_df.to_csv(os.path.join(RESULTS_DIR, "feature_importances.csv"), index=False)
    print(f"  feature_importances.csv    ({len(imp_df)} rows)")

    for model_name, pred_dict in all_preds.items():
        if not pred_dict["y_true"]:
            continue
        pred_df   = pd.DataFrame(pred_dict)
        safe_name = model_name.replace(" ", "_").lower()
        pred_df.to_csv(
            os.path.join(RESULTS_DIR, f"predictions_{safe_name}.csv"), index=False
        )
    print("  predictions_*.csv          (one file per model)")

    # -----------------------------------------------------------------------
    # Aggregate summary across folds
    # -----------------------------------------------------------------------
    summary_records = []
    for model_name in fold_df["model"].unique():
        sub = fold_df[fold_df["model"] == model_name]
        summary_records.append({
            "model":               model_name,
            "avg_recall_shift":    sub["recall_shift"].mean(),
            "std_recall_shift":    sub["recall_shift"].std(),
            "avg_precision_shift": sub["precision_shift"].mean(),
            "std_precision_shift": sub["precision_shift"].std(),
            "avg_f1_shift":        sub["f1_shift"].mean(),
            "std_f1_shift":        sub["f1_shift"].std(),
            "avg_accuracy":        sub["accuracy"].mean(),
            "std_accuracy":        sub["accuracy"].std(),
        })

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "shift_detection_summary.csv"), index=False)
    print("  shift_detection_summary.csv")

    # -----------------------------------------------------------------------
    # Naive baselines (computed from aggregated test set class distribution)
    # -----------------------------------------------------------------------
    total_test_n0 = class_dist_df["test_n0"].sum()
    total_test_n1 = class_dist_df["test_n1"].sum()
    total_test    = total_test_n0 + total_test_n1

    baseline_always0_acc   = total_test_n0 / total_test   # accuracy of "always predict 0"
    baseline_pct_shift     = 100 * total_test_n1 / total_test

    # -----------------------------------------------------------------------
    # Feature importance: average across RF + XGBoost, across all folds
    # -----------------------------------------------------------------------
    top_features = []
    if not imp_df.empty:
        imp_combined = (
            imp_df.groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
        )
        top_features = list(imp_combined.items())

    # -----------------------------------------------------------------------
    # Print summary (Step 8 of the implementation plan)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("=== Binary Shift Detection Summary ===")
    print(f"Label: 'regime shift in next H=10 days?' (any != current)")
    print(f"Class distribution: "
          f"{100 - baseline_pct_shift:.1f}% no-shift, "
          f"{baseline_pct_shift:.1f}% shift")
    print()
    print("Naive Baseline (always predict 'no shift'):")
    print(f"  Accuracy: {baseline_always0_acc:.3f}   Recall: 0.000   F1: 0.000")
    print()
    print("Naive Baseline (always predict 'shift'):")
    print(f"  Accuracy: {1 - baseline_always0_acc:.3f}   "
          f"Recall: 1.000   "
          f"Precision: {baseline_pct_shift / 100:.3f}")
    print()

    for rec in summary_records:
        print(f"Model: {rec['model']}"
              + (" (class_weight='balanced')" if rec["model"] != "XGBoost" else
                 " (scale_pos_weight per fold)"))
        print(f"  Avg Recall (shift)    : {rec['avg_recall_shift']:.3f}"
              f" +/- {rec['std_recall_shift']:.3f}")
        print(f"  Avg Precision (shift) : {rec['avg_precision_shift']:.3f}"
              f" +/- {rec['std_precision_shift']:.3f}")
        print(f"  Avg F1 (shift)        : {rec['avg_f1_shift']:.3f}"
              f" +/- {rec['std_f1_shift']:.3f}")
        print(f"  Avg Accuracy          : {rec['avg_accuracy']:.3f}"
              f" +/- {rec['std_accuracy']:.3f}")
        print()

    if top_features:
        print("Top 3 Features (avg importance across RF + XGBoost, all folds):")
        for rank, (feat, imp) in enumerate(top_features[:3], 1):
            print(f"  {rank}. {feat}: {imp:.4f}")

    print("=" * 60)
    print("\nDone. Run 03_analyze_results.py to generate plots.")


if __name__ == "__main__":
    main()
