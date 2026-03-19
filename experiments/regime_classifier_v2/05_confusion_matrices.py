"""
05_confusion_matrices.py -- Confusion Matrices: Full Progression from Baseline to Optimized

Generates confusion matrices for 7 configurations that tell the complete story
of the regime classifier v2 experiment:

  1. Naive Baseline:          always predict 0 (no shift)
  2. Ablation Phase 1:        LR, 3 level features, H=10, thr=0.5
  3. Ablation Phase 2:        LR, all 10 features, H=10, thr=0.5
  4. Ablation Phase 3:        LR balanced, 10 features, H=10, thr=0.5
  5. Ablation Phase 4a:       RandomForest balanced, 10 features, H=10, thr=0.5
  6. Ablation Phase 4b:       XGBoost (scale_pos_weight per fold), 10 features, H=10, thr=0.5
  7. Optimized:               LR (C=0.001, balanced, L2), 5 features, H=20, thr=0.35

All confusion matrices are aggregated across the same 12 walk-forward folds
(test years 2013-2024) used throughout the experiment.

Outputs:
  results/regime_classifier_v2/cm_all_phases.png       -- combined 2x4 figure
  results/regime_classifier_v2/cm_individual/          -- 7 individual PNGs
  results/regime_classifier_v2/cm_summary.csv          -- TP, FP, TN, FN per config

Does NOT modify any existing scripts (01-05_optimize_classifier).

Run from project root (Implementierung1/):
  python experiments/regime_classifier_v2/05_confusion_matrices.py
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
import seaborn as sns

from sklearn.base            import clone
from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing   import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    print("WARNING: xgboost not installed.  Config 6 (XGBoost) will be skipped.")
    HAS_XGB = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results", "regime_classifier_v2")
CM_IND_DIR   = os.path.join(RESULTS_DIR, "cm_individual")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.makedirs(CM_IND_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Import data-building functions from 01_build_dataset.py via importlib
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "build_dataset",
    os.path.join(SCRIPT_DIR, "01_build_dataset.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_raw_data              = _mod.load_raw_data
compute_regime_labels      = _mod.compute_regime_labels
compute_binary_shift_label = _mod.compute_binary_shift_label
compute_features           = _mod.compute_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_FEATURES = [
    "VIX_MA20", "z_VIX", "delta_VIX", "VIX_slope_5", "VIX_slope_20",
    "VIX_rolling_std_10", "max_VIX_window", "min_VIX_window",
    "SPY_return_5", "vol_ratio",
]
TARGET_COL  = "shift_label"
TEST_YEARS  = list(range(2013, 2025))
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Configuration table (7 configurations)
# ---------------------------------------------------------------------------
# model=None means naive baseline (predict all zeros).
# features="all_10" is resolved to ALL_FEATURES at runtime.
# For XGBoost: scale_pos_weight is set per fold in collect_predictions().
# ---------------------------------------------------------------------------
CONFIGS = [
    {
        "idx":      1,
        "name":     "Naive Baseline (always 0)",
        "short":    "naive_baseline",
        "H":        10,
        "model":    None,
        "features": None,
        "threshold": None,
    },
    {
        "idx":      2,
        "name":     "Phase 1: LR, level features",
        "short":    "phase1_LR_level",
        "H":        10,
        "model":    LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "features": ["VIX_MA20", "max_VIX_window", "min_VIX_window"],
        "threshold": 0.5,
    },
    {
        "idx":      3,
        "name":     "Phase 2: LR, all features",
        "short":    "phase2_LR_allfeat",
        "H":        10,
        "model":    LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "features": "all_10",
        "threshold": 0.5,
    },
    {
        "idx":      4,
        "name":     "Phase 3: LR balanced, all features",
        "short":    "phase3_LR_balanced",
        "H":        10,
        "model":    LogisticRegression(class_weight="balanced", max_iter=1000,
                                       random_state=RANDOM_SEED),
        "features": "all_10",
        "threshold": 0.5,
    },
    {
        "idx":      5,
        "name":     "Phase 4: RandomForest",
        "short":    "phase4_RF",
        "H":        10,
        "model":    RandomForestClassifier(class_weight="balanced", n_estimators=100,
                                           random_state=RANDOM_SEED),
        "features": "all_10",
        "threshold": 0.5,
    },
    {
        "idx":      6,
        "name":     "Phase 4: XGBoost",
        "short":    "phase4_XGB",
        "H":        10,
        # scale_pos_weight is set per fold; base config has default value
        "model":    XGBClassifier(n_estimators=100, random_state=RANDOM_SEED,
                                  eval_metric="logloss",
                                  verbosity=0) if HAS_XGB else None,
        "features": "all_10",
        "threshold": 0.5,
    },
    {
        "idx":      7,
        "name":     "Optimized: LR, 5 feat, H=20, thr=0.35",
        "short":    "optimized",
        "H":        20,
        "model":    LogisticRegression(C=0.001, class_weight="balanced", penalty="l2",
                                       solver="liblinear", max_iter=1000,
                                       random_state=RANDOM_SEED),
        "features": ["VIX_MA20", "max_VIX_window", "min_VIX_window",
                     "VIX_slope_20", "VIX_rolling_std_10"],
        "threshold": 0.35,
    },
]


# ---------------------------------------------------------------------------
# Walk-forward prediction collector
# ---------------------------------------------------------------------------

def collect_predictions(dataset: pd.DataFrame,
                         feature_cols: list,
                         model_template,
                         threshold: float,
                         is_xgb: bool = False) -> tuple:
    """
    Run 12-fold walk-forward and return concatenated (y_true, y_pred) arrays.

    Parameters
    ----------
    dataset        : DataFrame with feature columns + TARGET_COL
    feature_cols   : list of feature column names to use
    model_template : unfitted sklearn estimator (cloned per fold)
    threshold      : decision probability threshold (0.5 = standard predict)
    is_xgb         : if True, set scale_pos_weight = n_neg/n_pos per fold

    Returns (y_true: ndarray, y_pred: ndarray)
    """
    X_all = dataset[feature_cols].values
    y_all = dataset[TARGET_COL].values.astype(int)
    dates = dataset.index

    y_true_list = []
    y_pred_list = []

    for fold_idx, test_year in enumerate(TEST_YEARS):
        train_mask = dates.year <= (test_year - 1)
        test_mask  = dates.year == test_year

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        scaler     = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        # Clone model and optionally set per-fold XGBoost weight
        model = clone(model_template)
        if is_xgb:
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            model.set_params(scale_pos_weight=n_neg / n_pos if n_pos > 0 else 1.0)

        model.fit(X_train_sc, y_train)

        # Apply threshold via predict_proba for all non-naive configs
        y_proba = model.predict_proba(X_test_sc)[:, 1]
        y_pred  = (y_proba >= threshold).astype(int)

        y_true_list.append(y_test)
        y_pred_list.append(y_pred)

    return np.concatenate(y_true_list), np.concatenate(y_pred_list)


# ---------------------------------------------------------------------------
# Single confusion-matrix plot (used both in combined and individual figures)
# ---------------------------------------------------------------------------

def plot_single_cm(ax, cm: np.ndarray, title: str,
                   vmin: int = 0, vmax: int = None,
                   fontsize: int = 10) -> None:
    """
    Draw a labeled 2x2 confusion matrix heatmap on the given axes.
    Rows = Actual (No Shift / Shift), Cols = Predicted.
    """
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        xticklabels=["No Shift", "Shift"],
        yticklabels=["No Shift", "Shift"],
        ax=ax,
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": fontsize + 1},
    )
    ax.set_xlabel("Predicted", fontsize=fontsize)
    ax.set_ylabel("Actual",    fontsize=fontsize)
    ax.set_title(title,        fontsize=fontsize, pad=6)
    ax.tick_params(labelsize=fontsize - 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -----------------------------------------------------------------------
    # Load raw data and compute features
    # -----------------------------------------------------------------------
    print("Loading data ...")
    prices, vix  = load_raw_data()
    regime       = compute_regime_labels(vix)
    features_df  = compute_features(prices, vix)

    # Build one dataset per unique H value used across configurations
    unique_H = sorted(set(c["H"] for c in CONFIGS if c["model"] is not None or c["idx"] == 1))
    datasets = {}
    for h in unique_H:
        shift_label = compute_binary_shift_label(regime, h)
        ds = features_df.copy()
        ds["regime"]      = regime
        ds["shift_label"] = shift_label
        datasets[h] = ds.dropna()
        n1 = int((datasets[h][TARGET_COL] == 1).sum())
        n0 = int((datasets[h][TARGET_COL] == 0).sum())
        print(f"  H={h:2d}: {len(datasets[h])} rows  "
              f"shift={n1} ({100*n1/(n0+n1):.1f}%)  "
              f"no-shift={n0} ({100*n0/(n0+n1):.1f}%)")

    # -----------------------------------------------------------------------
    # Collect predictions for each configuration
    # -----------------------------------------------------------------------
    print("\nRunning walk-forward for each configuration ...")
    cms       = {}   # idx -> 2x2 ndarray
    all_stats = []   # rows for CSV summary

    for cfg in CONFIGS:
        idx  = cfg["idx"]
        h    = cfg["H"]
        ds   = datasets[h]

        # Resolve feature list
        if cfg["features"] == "all_10":
            feat_cols = ALL_FEATURES
        else:
            feat_cols = cfg["features"]

        # Get y_true (ground truth) for the test folds from this dataset
        # For naive baseline: predict all zeros without any model
        if cfg["model"] is None:
            # Naive baseline: aggregate y_true from all 12 test folds
            y_all  = ds[TARGET_COL].values.astype(int)
            dates  = ds.index
            y_true_list = []
            for test_year in TEST_YEARS:
                mask = dates.year == test_year
                if mask.sum() > 0:
                    y_true_list.append(y_all[mask])
            y_true = np.concatenate(y_true_list)
            y_pred = np.zeros_like(y_true)
            print(f"  [{idx}] {cfg['name']:45s}  (naive — no model)")
        else:
            # Skip XGBoost if not available
            if not HAS_XGB and cfg["short"] == "phase4_XGB":
                print(f"  [{idx}] {cfg['name']:45s}  (SKIPPED — xgboost not installed)")
                continue

            is_xgb = HAS_XGB and isinstance(cfg["model"], XGBClassifier)
            y_true, y_pred = collect_predictions(
                ds, feat_cols, cfg["model"],
                cfg["threshold"], is_xgb=is_xgb,
            )
            print(f"  [{idx}] {cfg['name']:45s}  "
                  f"recall={recall_score(y_true, y_pred, pos_label=1, zero_division=0):.3f}  "
                  f"prec={precision_score(y_true, y_pred, pos_label=1, zero_division=0):.3f}  "
                  f"f1={f1_score(y_true, y_pred, pos_label=1, zero_division=0):.3f}")

        # Compute aggregated confusion matrix (labels=[0,1] fixes TN/FP/FN/TP order)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cms[idx] = cm

        # Extract cells: cm[true, pred]
        TN = int(cm[0, 0]);  FP = int(cm[0, 1])
        FN = int(cm[1, 0]);  TP = int(cm[1, 1])

        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        prec = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        acc  = (TP + TN) / (TP + TN + FP + FN)
        far  = FP / (FP + TN) if (FP + TN) > 0 else 0.0   # false alarm rate

        all_stats.append({
            "config_idx":       idx,
            "config_name":      cfg["name"],
            "H":                h,
            "threshold":        cfg["threshold"] if cfg["threshold"] is not None else "—",
            "n_features":       len(feat_cols) if feat_cols is not None else 0,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "recall":    round(rec,  4),
            "precision": round(prec, 4) if not np.isnan(prec) else float("nan"),
            "f1":        round(f1,   4),
            "accuracy":  round(acc,  4),
            "false_alarm_rate": round(far, 4),
        })

    # -----------------------------------------------------------------------
    # Shared color scale: vmax = max cell value across all CMs
    # -----------------------------------------------------------------------
    all_cell_values = np.concatenate([cm.flatten() for cm in cms.values()])
    vmax_global = int(all_cell_values.max())

    # -----------------------------------------------------------------------
    # Plot 1: Combined figure (2 rows x 4 cols, 7 matrices + 1 hidden)
    # -----------------------------------------------------------------------
    print("\nPlotting combined figure ...")

    # Layout: row 0 = configs 1-4, row 1 = configs 5-7 + hidden
    layout_order = [1, 2, 3, 4, 5, 6, 7]

    fig, axes = plt.subplots(2, 4, figsize=(28, 10))

    for plot_pos, cfg_idx in enumerate(layout_order):
        row = plot_pos // 4
        col = plot_pos  % 4
        ax  = axes[row, col]

        if cfg_idx not in cms:
            ax.set_visible(False)
            continue

        cfg   = next(c for c in CONFIGS if c["idx"] == cfg_idx)
        cm    = cms[cfg_idx]
        stats = next(s for s in all_stats if s["config_idx"] == cfg_idx)

        # Subtitle: key stats below the config name
        rec_str  = f"Recall={stats['recall']:.3f}"
        f1_str   = f"F1={stats['f1']:.3f}"
        far_str  = f"FAR={stats['false_alarm_rate']:.3f}"
        subtitle = f"{cfg['name']}\n{rec_str}  {f1_str}  {far_str}"

        plot_single_cm(ax, cm, subtitle,
                       vmin=0, vmax=vmax_global, fontsize=9)

    # Hide the unused 4th subplot in row 2
    axes[1, 3].set_visible(False)

    fig.suptitle(
        "Regime Shift Detection: Confusion Matrices Across All Configurations\n"
        "(aggregated over 12 walk-forward folds, test years 2013-2024)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "cm_all_phases.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: cm_all_phases.png")

    # -----------------------------------------------------------------------
    # Plot 2: Individual confusion matrices
    # -----------------------------------------------------------------------
    print("Plotting individual figures ...")

    for cfg in CONFIGS:
        idx = cfg["idx"]
        if idx not in cms:
            continue

        cm    = cms[idx]
        stats = next(s for s in all_stats if s["config_idx"] == idx)
        rec_str = f"Recall={stats['recall']:.3f}  F1={stats['f1']:.3f}"

        fig, ax = plt.subplots(figsize=(5, 4))
        plot_single_cm(ax, cm, f"{cfg['name']}\n{rec_str}",
                       vmin=0, vmax=None, fontsize=10)
        plt.tight_layout()
        fname = f"cm_{idx}_{cfg['short']}.png"
        plt.savefig(os.path.join(CM_IND_DIR, fname), dpi=150)
        plt.close()
        print(f"  Saved: cm_individual/{fname}")

    # -----------------------------------------------------------------------
    # CSV summary
    # -----------------------------------------------------------------------
    summary_df = pd.DataFrame(all_stats)
    csv_path = os.path.join(RESULTS_DIR, "cm_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved: cm_summary.csv")

    # -----------------------------------------------------------------------
    # Print summary table (Step 7)
    # -----------------------------------------------------------------------
    # Helper to look up stats by config index
    def s(idx):
        rows = [r for r in all_stats if r["config_idx"] == idx]
        return rows[0] if rows else None

    s1 = s(1);  s2 = s(2);  s3 = s(3);  s4 = s(4)
    s5 = s(5);  s6 = s(6);  s7 = s(7)

    def fmt_prec(v):
        return f"{v:.3f}" if v is not None and not np.isnan(v) else "  —  "

    hdr_width = 41
    print("\n" + "=" * 95)
    print("=== Confusion Matrix Summary (aggregated across 12 walk-forward folds) ===")
    print()
    print(f"{'Config':<{hdr_width}} | {'TP':>6} | {'FN':>6} | {'FP':>6} | {'TN':>6} "
          f"| {'Recall':>6} | {'Prec':>6} | {'F1':>6} | {'FAR':>6}")
    print("-" * 95)

    rows_to_print = [
        (1, "1. Naive (always 0)"),
        (2, "2. Phase 1: LR, 3 feat, H=10"),
        (3, "3. Phase 2: LR, 10 feat, H=10"),
        (4, "4. Phase 3: LR bal, 10 feat, H=10"),
        (5, "5. Phase 4: RF, 10 feat, H=10"),
        (6, "6. Phase 4: XGB, 10 feat, H=10"),
        (7, "7. Optimized: LR, 5 feat, H=20, thr=0.35"),
    ]

    for idx, label in rows_to_print:
        st = s(idx)
        if st is None:
            print(f"  {label:<{hdr_width-2}}  (not available)")
            continue
        print(f"{label:<{hdr_width}} | {st['TP']:>6} | {st['FN']:>6} | "
              f"{st['FP']:>6} | {st['TN']:>6} | "
              f"{st['recall']:>6.3f} | {fmt_prec(st['precision']):>6} | "
              f"{st['f1']:>6.3f} | {st['false_alarm_rate']:>6.3f}")

    print()
    print("Key progression:")

    def delta(a, b, key):
        if a and b:
            return b[key] - a[key]
        return float("nan")

    def fmt_d(v):
        return f"{v:+.3f}" if not np.isnan(v) else "  n/a"

    print(f"  Label change effect  (Naive -> Phase 1):       "
          f"Recall {fmt_d(delta(s1, s2, 'recall'))}")
    print(f"  Feature effect       (Phase 1 -> Phase 2):     "
          f"Recall {fmt_d(delta(s2, s3, 'recall'))}")
    print(f"  Class weight effect  (Phase 2 -> Phase 3):     "
          f"Recall {fmt_d(delta(s3, s4, 'recall'))}")
    print(f"  Model effect         (Phase 3 -> Phase 4 RF):  "
          f"Recall {fmt_d(delta(s4, s5, 'recall'))}")
    if s6:
        print(f"  Model effect         (Phase 3 -> Phase 4 XGB): "
              f"Recall {fmt_d(delta(s4, s6, 'recall'))}")
    print(f"  Full optimization    (Phase 1 -> Optimized):   "
          f"Recall {fmt_d(delta(s2, s7, 'recall'))}, "
          f"F1 {fmt_d(delta(s2, s7, 'f1'))}")
    print("=" * 95)
    print(f"\nAll outputs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
