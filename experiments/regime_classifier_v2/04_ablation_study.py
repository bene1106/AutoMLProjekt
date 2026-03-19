"""
04_ablation_study.py -- Ablation Study: Regime Classifier v2

Isolates the contribution of each strategy by running 4 phases in sequence.
The binary shift label (H=10, any-formulation) is FIXED across all phases.
Only features, class weighting, and model type vary.

Phase 1 -- New label + level features only (LR, no balancing)
           Answers: does the label change alone improve shift detection?

Phase 2 -- + dynamics features (LR, no balancing)
           Answers: do z_VIX, slopes, vol_ratio add signal on top of level features?

Phase 3 -- + class_weight='balanced' (LR)
           Answers: does balancing the loss improve recall on the minority class?

Phase 4 -- Non-linear models: RandomForest + XGBoost (same full feature set)
           Answers: do non-linear models extract more signal from the same features?

Walk-forward setup is identical to 02_train_evaluate.py:
  Fold 1:  Train 2006-2012, Test 2013
  ...
  Fold 12: Train 2006-2023, Test 2024

Outputs (results/regime_classifier_v2/):
  ablation_results.csv      -- per phase x fold x model
  ablation_summary.csv      -- aggregated (mean +/- std across 12 folds)
  ablation_comparison.png   -- grouped bar chart

Run from project root (Implementierung1/):
  python experiments/regime_classifier_v2/04_ablation_study.py
"""

import importlib.util
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    print("WARNING: xgboost not installed. XGBoost phase will be skipped.")
    HAS_XGB = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results", "regime_classifier_v2")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Import functions from 01_build_dataset.py via importlib.
# The "01_" prefix makes standard import impossible; importlib handles this.
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "build_dataset",
    os.path.join(SCRIPT_DIR, "01_build_dataset.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_raw_data             = _mod.load_raw_data
compute_regime_labels     = _mod.compute_regime_labels
compute_binary_shift_label = _mod.compute_binary_shift_label
compute_features          = _mod.compute_features

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
ALL_FEATURES = [
    "VIX_MA20", "z_VIX", "delta_VIX", "VIX_slope_5", "VIX_slope_20",
    "VIX_rolling_std_10", "max_VIX_window", "min_VIX_window",
    "SPY_return_5", "vol_ratio",
]

PHASE1_FEATURES = ["VIX_MA20", "max_VIX_window", "min_VIX_window"]
PHASE2_FEATURES = ALL_FEATURES   # full set, no balancing
PHASE3_FEATURES = ALL_FEATURES   # full set + balanced
PHASE4_FEATURES = ALL_FEATURES   # full set + non-linear models

# Walk-forward: test years 2013-2024
TEST_YEARS  = list(range(2013, 2025))
H           = 10
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Phase configuration
# Each entry defines one (phase, model) combination to evaluate.
# XGBoost gets scale_pos_weight computed per fold — see run_phase() below.
# ---------------------------------------------------------------------------
PHASES = [
    {
        "phase":       1,
        "model_name":  "LogisticRegression",
        "description": "New label + level features (LR, no balancing)",
        "features":    PHASE1_FEATURES,
        "balanced":    False,
        "model_type":  "lr",
    },
    {
        "phase":       2,
        "model_name":  "LogisticRegression",
        "description": "Phase 1 + dynamics features (LR, no balancing)",
        "features":    PHASE2_FEATURES,
        "balanced":    False,
        "model_type":  "lr",
    },
    {
        "phase":       3,
        "model_name":  "LogisticRegression",
        "description": "Phase 2 + class_weight='balanced' (LR)",
        "features":    PHASE3_FEATURES,
        "balanced":    True,
        "model_type":  "lr",
    },
    {
        "phase":       4,
        "model_name":  "RandomForest",
        "description": "Phase 3 features + RF model (balanced)",
        "features":    PHASE4_FEATURES,
        "balanced":    True,
        "model_type":  "rf",
    },
]

if HAS_XGB:
    PHASES.append({
        "phase":       4,
        "model_name":  "XGBoost",
        "description": "Phase 3 features + XGBoost (scale_pos_weight per fold)",
        "features":    PHASE4_FEATURES,
        "balanced":    True,
        "model_type":  "xgb",
    })


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(cfg: dict, n_neg: int, n_pos: int):
    """Instantiate the appropriate model for the given phase config."""
    t = cfg["model_type"]
    if t == "lr":
        return LogisticRegression(
            class_weight="balanced" if cfg["balanced"] else None,
            max_iter=1000,
            random_state=RANDOM_SEED,
        )
    if t == "rf":
        return RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100,
            random_state=RANDOM_SEED,
        )
    if t == "xgb":
        scale = n_neg / n_pos if n_pos > 0 else 1.0
        return XGBClassifier(
            scale_pos_weight=scale,
            n_estimators=100,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            verbosity=0,
        )
    raise ValueError(f"Unknown model_type: {t}")


# ---------------------------------------------------------------------------
# Core: run one phase across all 12 walk-forward folds
# ---------------------------------------------------------------------------

def run_phase(cfg: dict, dataset: pd.DataFrame) -> list:
    """
    Run walk-forward evaluation for one phase configuration.
    Returns a list of per-fold result dicts.
    """
    feat_cols = cfg["features"]
    X_all     = dataset[feat_cols].values
    y_all     = dataset["shift_label"].values.astype(int)
    dates     = dataset.index

    records = []

    for fold_idx, test_year in enumerate(TEST_YEARS):
        fold_num     = fold_idx + 1
        train_mask   = dates.year <= (test_year - 1)
        test_mask    = dates.year == test_year

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())

        # Standardize: fit on training set ONLY
        scaler     = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        model  = make_model(cfg, n_neg, n_pos)
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)

        records.append({
            "phase":           cfg["phase"],
            "model":           cfg["model_name"],
            "description":     cfg["description"],
            "fold":            fold_num,
            "test_year":       test_year,
            "n_features":      len(feat_cols),
            "features_used":   "|".join(feat_cols),
            "precision_shift": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "recall_shift":    recall_score(   y_test, y_pred, pos_label=1, zero_division=0),
            "f1_shift":        f1_score(       y_test, y_pred, pos_label=1, zero_division=0),
            "accuracy":        accuracy_score( y_test, y_pred),
            "n_shifts_true":   int((y_test == 1).sum()),
            "n_shifts_pred":   int((y_pred == 1).sum()),
        })

    return records


# ---------------------------------------------------------------------------
# Plot: grouped bar chart
# ---------------------------------------------------------------------------

def plot_ablation_comparison(summary_df: pd.DataFrame, baseline_acc: float) -> None:
    """
    Grouped bar chart with Recall, Precision, F1 for each phase/model row.
    X-axis labels are compact phase descriptions.
    """
    # Build a short label per row in display order
    labels = []
    for _, row in summary_df.iterrows():
        if row["phase"] == 1:
            labels.append("Phase 1\nLR (level only)")
        elif row["phase"] == 2:
            labels.append("Phase 2\nLR (all features)")
        elif row["phase"] == 3:
            labels.append("Phase 3\nLR (balanced)")
        elif row["phase"] == 4 and "RandomForest" in row["model"]:
            labels.append("Phase 4\nRandomForest")
        elif row["phase"] == 4 and "XGBoost" in row["model"]:
            labels.append("Phase 4\nXGBoost")
        else:
            labels.append(f"Phase {row['phase']}\n{row['model']}")

    n      = len(labels)
    x      = np.arange(n)
    width  = 0.25

    recalls    = summary_df["avg_recall"].values
    precisions = summary_df["avg_precision"].values
    f1s        = summary_df["avg_f1"].values

    fig, ax = plt.subplots(figsize=(max(10, n * 2), 6))

    bars_r = ax.bar(x - width, recalls,    width, label="Recall (shift)",    color="#1f77b4", alpha=0.85)
    bars_p = ax.bar(x,          precisions, width, label="Precision (shift)", color="#ff7f0e", alpha=0.85)
    bars_f = ax.bar(x + width, f1s,        width, label="F1 (shift)",        color="#2ca02c", alpha=0.85)

    # Annotate bars with values
    for bars in (bars_r, bars_p, bars_f):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7.5,
            )

    # Naive baseline: accuracy of "always predict no-shift"
    ax.axhline(
        baseline_acc, color="black", linestyle="--", linewidth=1.5,
        label=f"Naive baseline accuracy = {baseline_acc:.3f}",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Metric value (avg across 12 folds)", fontsize=11)
    ax.set_title("Ablation Study: Regime Shift Detection\n"
                 "(Binary shift label H=10, walk-forward 2013-2024)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "ablation_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: ablation_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -----------------------------------------------------------------------
    # Build dataset (same logic as 01_build_dataset.py, not re-loaded from CSV
    # to guarantee exact parity regardless of CSV state)
    # -----------------------------------------------------------------------
    print("Building dataset ...")
    prices, vix = load_raw_data()
    regime      = compute_regime_labels(vix)
    shift_label = compute_binary_shift_label(regime, H)
    features    = compute_features(prices, vix)

    dataset              = features.copy()
    dataset["regime"]    = regime
    dataset["shift_label"] = shift_label
    dataset = dataset.dropna()

    n_neg_total = int((dataset["shift_label"] == 0).sum())
    n_pos_total = int((dataset["shift_label"] == 1).sum())
    n_total     = n_neg_total + n_pos_total
    baseline_acc = n_neg_total / n_total   # "always predict 0" accuracy

    print(f"  Dataset: {len(dataset)} rows | "
          f"shift={n_pos_total} ({100*n_pos_total/n_total:.1f}%) | "
          f"no-shift={n_neg_total} ({100*n_neg_total/n_total:.1f}%)")
    print(f"  Naive baseline accuracy (always predict 0): {baseline_acc:.3f}\n")

    # -----------------------------------------------------------------------
    # Run all phases
    # -----------------------------------------------------------------------
    all_records = []

    for cfg in PHASES:
        print(f"Phase {cfg['phase']} | {cfg['model_name']:22s} | "
              f"{len(cfg['features'])} features | {cfg['description']}")
        fold_records = run_phase(cfg, dataset)
        all_records.extend(fold_records)

        # Quick per-phase summary
        sub = pd.DataFrame(fold_records)
        print(f"  -> Avg Recall={sub['recall_shift'].mean():.3f}  "
              f"Precision={sub['precision_shift'].mean():.3f}  "
              f"F1={sub['f1_shift'].mean():.3f}  "
              f"Acc={sub['accuracy'].mean():.3f}\n")

    # -----------------------------------------------------------------------
    # Save ablation_results.csv
    # -----------------------------------------------------------------------
    results_df = pd.DataFrame(all_records)
    results_df.to_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"), index=False)
    print(f"Saved ablation_results.csv  ({len(results_df)} rows)")

    # -----------------------------------------------------------------------
    # Build ablation_summary.csv
    # -----------------------------------------------------------------------
    summary_rows = []
    for cfg in PHASES:
        sub = results_df[
            (results_df["phase"] == cfg["phase"]) &
            (results_df["model"] == cfg["model_name"])
        ]
        if sub.empty:
            continue
        summary_rows.append({
            "phase":          cfg["phase"],
            "model":          cfg["model_name"],
            "description":    cfg["description"],
            "n_features":     len(cfg["features"]),
            "avg_recall":     sub["recall_shift"].mean(),
            "std_recall":     sub["recall_shift"].std(),
            "avg_precision":  sub["precision_shift"].mean(),
            "std_precision":  sub["precision_shift"].std(),
            "avg_f1":         sub["f1_shift"].mean(),
            "std_f1":         sub["f1_shift"].std(),
            "avg_accuracy":   sub["accuracy"].mean(),
            "std_accuracy":   sub["accuracy"].std(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "ablation_summary.csv"), index=False)
    print(f"Saved ablation_summary.csv  ({len(summary_df)} rows)")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    print("\nGenerating ablation_comparison.png ...")
    plot_ablation_comparison(summary_df, baseline_acc)

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    # Helper to look up a row from summary_df
    def get_row(phase: int, model: str) -> dict:
        mask = (summary_df["phase"] == phase) & (summary_df["model"] == model)
        rows = summary_df[mask]
        return rows.iloc[0].to_dict() if not rows.empty else None

    p1 = get_row(1, "LogisticRegression")
    p2 = get_row(2, "LogisticRegression")
    p3 = get_row(3, "LogisticRegression")
    p4_rf  = get_row(4, "RandomForest")
    p4_xgb = get_row(4, "XGBoost") if HAS_XGB else None

    print("\n" + "=" * 65)
    print("=== Ablation Study Summary ===")
    print(f"Binary shift label: H=10, any(...) formulation")
    print()

    if p1:
        print(f"Phase 1 -- New label + level features (LR, no balancing)")
        print(f"  Features: {', '.join(PHASE1_FEATURES)}")
        print(f"  Recall: {p1['avg_recall']:.3f} +/- {p1['std_recall']:.3f} | "
              f"Precision: {p1['avg_precision']:.3f} +/- {p1['std_precision']:.3f} | "
              f"F1: {p1['avg_f1']:.3f} +/- {p1['std_f1']:.3f}")
        print()

    if p2:
        delta_r = p2["avg_recall"] - p1["avg_recall"] if p1 else float("nan")
        delta_f = p2["avg_f1"]     - p1["avg_f1"]     if p1 else float("nan")
        added = [f for f in PHASE2_FEATURES if f not in PHASE1_FEATURES]
        print(f"Phase 2 -- + dynamics features (LR, no balancing)")
        print(f"  Added: {', '.join(added)}")
        print(f"  Recall: {p2['avg_recall']:.3f} +/- {p2['std_recall']:.3f} | "
              f"Precision: {p2['avg_precision']:.3f} +/- {p2['std_precision']:.3f} | "
              f"F1: {p2['avg_f1']:.3f} +/- {p2['std_f1']:.3f}")
        print(f"  Delta vs Phase 1:  Recall {delta_r:+.3f} | F1 {delta_f:+.3f}")
        print()

    if p3:
        delta_r = p3["avg_recall"] - p2["avg_recall"] if p2 else float("nan")
        delta_f = p3["avg_f1"]     - p2["avg_f1"]     if p2 else float("nan")
        print(f"Phase 3 -- + class_weight='balanced' (LR)")
        print(f"  Recall: {p3['avg_recall']:.3f} +/- {p3['std_recall']:.3f} | "
              f"Precision: {p3['avg_precision']:.3f} +/- {p3['std_precision']:.3f} | "
              f"F1: {p3['avg_f1']:.3f} +/- {p3['std_f1']:.3f}")
        print(f"  Delta vs Phase 2:  Recall {delta_r:+.3f} | F1 {delta_f:+.3f}")
        print()

    print(f"Phase 4 -- Non-linear models (RF, XGBoost)")
    if p4_rf:
        delta_r = p4_rf["avg_recall"] - p3["avg_recall"] if p3 else float("nan")
        delta_f = p4_rf["avg_f1"]     - p3["avg_f1"]     if p3 else float("nan")
        print(f"  RandomForest:  Recall {p4_rf['avg_recall']:.3f} +/- {p4_rf['std_recall']:.3f} | "
              f"F1 {p4_rf['avg_f1']:.3f} +/- {p4_rf['std_f1']:.3f} | "
              f"Delta vs Phase 3: Recall {delta_r:+.3f}")
    if p4_xgb:
        delta_r = p4_xgb["avg_recall"] - p3["avg_recall"] if p3 else float("nan")
        delta_f = p4_xgb["avg_f1"]     - p3["avg_f1"]     if p3 else float("nan")
        print(f"  XGBoost:       Recall {p4_xgb['avg_recall']:.3f} +/- {p4_xgb['std_recall']:.3f} | "
              f"F1 {p4_xgb['avg_f1']:.3f} +/- {p4_xgb['std_f1']:.3f} | "
              f"Delta vs Phase 3: Recall {delta_r:+.3f}")
    print()

    print(f"Naive Baseline ('always predict no-shift'):")
    print(f"  Accuracy: {baseline_acc:.3f} | Recall: 0.000 | F1: 0.000")
    print("=" * 65)
    print(f"\nAll outputs saved to: {RESULTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
