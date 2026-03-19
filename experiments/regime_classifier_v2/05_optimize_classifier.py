"""
05_optimize_classifier.py -- Regime Classifier v2: Systematic Optimization

Runs 7 optimization steps in sequence, each building on the previous:

  Step 1 -- Feature selection     (fixes "more features = worse recall")
  Step 2 -- Hyperparameter tuning (inner 3-fold CV per walk-forward fold)
  Step 3 -- Threshold tuning      (probability threshold sweep 0.10 -- 0.90)
  Step 4 -- Horizon sensitivity   (H = 3, 5, 10, 15, 20)
  Step 5 -- Final model assembly  (best combination from Steps 1-4)
  Step 6 -- Summary plot          (2-panel figure)
  Step 7 -- Print summary

Walk-forward folds: identical to 02_train_evaluate.py (12 folds, 2013-2024).
Does NOT modify scripts 01-04.

Hyperparameter grids are reduced vs. the full plan to keep runtime manageable:
  - n_estimators=500 removed (slow, marginal gains over 200)
  - learning_rate=0.01 removed (slow convergence)
  - max_depth=10 removed from XGB (overfitting risk on time-series data)
  - min_child_weight=10 removed from XGB (too conservative)
  - max_depth=20 removed from RF (nearly identical to None)
  - min_samples_leaf=20 removed from GB (too conservative)
  These removals are documented per-grid below.

Run from project root (Implementierung1/):
  python experiments/regime_classifier_v2/05_optimize_classifier.py
"""

import collections
import importlib.util
import json
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble       import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model   import LogisticRegression
from sklearn.metrics        import (
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing  import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    print("WARNING: xgboost not installed.  XGBoost steps will be skipped.")
    HAS_XGB = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_BASE = os.path.join(PROJECT_ROOT, "results", "regime_classifier_v2")
OPT_DIR      = os.path.join(RESULTS_BASE, "optimization")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.makedirs(OPT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Import data-building functions from 01_build_dataset.py via importlib.
# The "01_" prefix makes standard import syntax illegal.
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
H_BASE      = 10
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Reduced hyperparameter grids
# ---------------------------------------------------------------------------

# LR: full grid as specified (24 combinations -- fast)
PARAM_GRID_LR = {
    "C":            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "class_weight": [None, "balanced"],
    "penalty":      ["l1", "l2"],
    "solver":       ["liblinear"],   # supports both L1 and L2
    "max_iter":     [1000],
}

# RF: removed n_estimators=500 (slow) and max_depth=20 (same as None)
# Remaining: 2 * 3 * 3 * 2 = 36 combinations per fold
PARAM_GRID_RF = {
    "n_estimators":     [100, 200],
    "max_depth":        [5, 10, None],
    "min_samples_leaf": [1, 5, 10],
    "class_weight":     ["balanced", "balanced_subsample"],
}

# XGB: scale_pos_weight NOT in grid (set per-fold on base estimator).
# Removed: n_estimators=500, learning_rate=0.01, max_depth=10, min_child_weight=10.
# Remaining: 2 * 3 * 2 * 2 = 24 combinations per fold
PARAM_GRID_XGB = {
    "n_estimators":     [100, 200],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.05, 0.1],
    "min_child_weight": [1, 5],
}

# GB: removed n_estimators=500, learning_rate=0.01, min_samples_leaf=20, max_depth=7.
# Remaining: 2 * 2 * 2 * 2 = 16 combinations per fold
PARAM_GRID_GB = {
    "n_estimators":     [100, 200],
    "max_depth":        [3, 5],
    "learning_rate":    [0.05, 0.1],
    "min_samples_leaf": [5, 10],
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def safe_json(d: dict) -> str:
    """JSON-serialize a param dict, converting numpy scalar types to Python natives."""
    def _cast(v):
        if isinstance(v, np.integer):   return int(v)
        if isinstance(v, np.floating):  return float(v)
        if isinstance(v, np.bool_):     return bool(v)
        return v
    return json.dumps({k: _cast(v) for k, v in d.items()}, sort_keys=True)


def most_common_params(params_list: list) -> dict:
    """Return the most frequently occurring param dict across a list of dicts."""
    strs     = [safe_json(p) for p in params_list]
    mode_str = collections.Counter(strs).most_common(1)[0][0]
    return json.loads(mode_str)


def agg_folds(records: list) -> dict:
    """Aggregate recall/precision/f1/accuracy mean and std from a list of fold dicts."""
    df = pd.DataFrame(records)
    return {
        "avg_recall":    df["recall_shift"].mean(),
        "std_recall":    df["recall_shift"].std(),
        "avg_precision": df["precision_shift"].mean(),
        "std_precision": df["precision_shift"].std(),
        "avg_f1":        df["f1_shift"].mean(),
        "std_f1":        df["f1_shift"].std(),
        "avg_accuracy":  df["accuracy"].mean(),
        "std_accuracy":  df["accuracy"].std(),
    }


def build_dataset_for_h(h: int, features_df: pd.DataFrame,
                         regime: pd.Series) -> pd.DataFrame:
    """
    Build a clean dataset for a given horizon H.
    Only the shift label changes; feature columns are identical across H values.
    """
    shift_label = compute_binary_shift_label(regime, h)
    ds                 = features_df.copy()
    ds["regime"]       = regime
    ds["shift_label"]  = shift_label
    return ds.dropna()


def walk_forward_basic(dataset: pd.DataFrame, feature_cols: list,
                        make_model_fn, threshold: float = 0.5) -> list:
    """
    Generic 12-fold walk-forward evaluation.

    Parameters
    ----------
    make_model_fn : callable(n_neg, n_pos) -> fitted-less sklearn model
    threshold     : decision threshold for predict_proba (0.5 uses predict())

    Returns list of per-fold metric dicts.
    """
    X_all  = dataset[feature_cols].values
    y_all  = dataset[TARGET_COL].values.astype(int)
    dates  = dataset.index
    records = []

    for fold_idx, test_year in enumerate(TEST_YEARS):
        fold_num   = fold_idx + 1
        train_mask = dates.year <= (test_year - 1)
        test_mask  = dates.year == test_year

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())

        scaler     = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        model = make_model_fn(n_neg, n_pos)
        model.fit(X_train_sc, y_train)

        if threshold == 0.5:
            y_pred = model.predict(X_test_sc)
        else:
            y_proba = model.predict_proba(X_test_sc)[:, 1]
            y_pred  = (y_proba >= threshold).astype(int)

        records.append({
            "fold":            fold_num,
            "test_year":       test_year,
            "recall_shift":    recall_score(   y_test, y_pred, pos_label=1, zero_division=0),
            "precision_shift": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "f1_shift":        f1_score(       y_test, y_pred, pos_label=1, zero_division=0),
            "accuracy":        accuracy_score( y_test, y_pred),
        })
    return records


def walk_forward_proba(dataset: pd.DataFrame, feature_cols: list,
                        make_model_fn) -> pd.DataFrame:
    """
    Walk-forward returning predicted class-1 probabilities for each test sample.
    Returns DataFrame with columns: fold, test_year, y_true, y_proba_1
    """
    X_all  = dataset[feature_cols].values
    y_all  = dataset[TARGET_COL].values.astype(int)
    dates  = dataset.index
    rows   = []

    for fold_idx, test_year in enumerate(TEST_YEARS):
        fold_num   = fold_idx + 1
        train_mask = dates.year <= (test_year - 1)
        test_mask  = dates.year == test_year

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())

        scaler     = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        model = make_model_fn(n_neg, n_pos)
        model.fit(X_train_sc, y_train)
        y_proba = model.predict_proba(X_test_sc)[:, 1]

        for yt, yp in zip(y_test, y_proba):
            rows.append({"fold": fold_num, "test_year": test_year,
                         "y_true": int(yt), "y_proba_1": float(yp)})

    return pd.DataFrame(rows)


def make_model(model_name: str, params: dict, n_neg: int, n_pos: int):
    """Instantiate any supported model with given params (post-grid-search usage)."""
    p = dict(params)
    if model_name == "LogisticRegression":
        return LogisticRegression(random_state=RANDOM_SEED, **p)
    if model_name == "RandomForest":
        return RandomForestClassifier(random_state=RANDOM_SEED, **p)
    if model_name == "XGBoost" and HAS_XGB:
        scale = n_neg / n_pos if n_pos > 0 else 1.0
        return XGBClassifier(scale_pos_weight=scale, eval_metric="logloss",
                             verbosity=0, random_state=RANDOM_SEED, **p)
    if model_name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=RANDOM_SEED, **p)
    raise ValueError(f"Unknown model: {model_name}")


# ===========================================================================
# STEP 1 — Feature Selection
# ===========================================================================

def step1_feature_selection(dataset: pd.DataFrame) -> tuple:
    """
    1A: Pearson correlation matrix of all 10 features.
    1B: Walk-forward evaluation of 7 feature subsets using LR (no balancing).

    Returns (best_features: list, results_df: pd.DataFrame)
    """
    print("\n" + "=" * 65)
    print("STEP 1 -- Feature Selection")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # 1A: Correlation analysis
    # -----------------------------------------------------------------------
    print("\n[1A] Pearson correlation matrix ...")
    corr = dataset[ALL_FEATURES].corr()
    high_corr_pairs = []
    for i, f1 in enumerate(ALL_FEATURES):
        for j, f2 in enumerate(ALL_FEATURES):
            if j <= i:
                continue
            if abs(corr.loc[f1, f2]) > 0.8:
                high_corr_pairs.append((f1, f2, round(corr.loc[f1, f2], 3)))

    if high_corr_pairs:
        print("  High-correlation pairs (|r| > 0.8):")
        for f1, f2, r in high_corr_pairs:
            print(f"    {f1} <-> {f2} : r = {r}")
    else:
        print("  No pairs with |r| > 0.8 found.")

    # Save correlation matrix
    corr.to_csv(os.path.join(OPT_DIR, "feature_correlation_matrix.csv"))

    # -----------------------------------------------------------------------
    # 1B: Feature subset evaluation
    # -----------------------------------------------------------------------

    # "best_from_importance": top-5 features by avg importance from RF + XGBoost
    imp_path = os.path.join(RESULTS_BASE, "feature_importances.csv")
    if os.path.exists(imp_path):
        imp_df = pd.read_csv(imp_path)
        top5 = (imp_df.groupby("feature")["importance"]
                .mean()
                .sort_values(ascending=False)
                .head(5)
                .index.tolist())
        print(f"\n  Top-5 from RF+XGB importance: {top5}")
    else:
        # Fallback based on ablation findings
        top5 = ["VIX_MA20", "max_VIX_window", "min_VIX_window",
                "VIX_slope_20", "VIX_rolling_std_10"]
        print(f"\n  feature_importances.csv not found -- using fallback top-5: {top5}")

    feature_subsets = {
        "level_only": [
            "VIX_MA20", "max_VIX_window", "min_VIX_window",
        ],
        "level_plus_zscore": [
            "VIX_MA20", "max_VIX_window", "min_VIX_window", "z_VIX",
        ],
        "level_plus_dynamics_core": [
            "VIX_MA20", "max_VIX_window", "min_VIX_window",
            "z_VIX", "delta_VIX", "VIX_slope_5",
        ],
        "level_plus_returns": [
            "VIX_MA20", "max_VIX_window", "min_VIX_window",
            "SPY_return_5", "vol_ratio",
        ],
        "dynamics_only": [
            "z_VIX", "delta_VIX", "VIX_slope_5", "VIX_slope_20",
            "VIX_rolling_std_10",
        ],
        "best_from_importance": top5,
        "all_10": ALL_FEATURES,
    }

    print(f"\n[1B] Evaluating {len(feature_subsets)} feature subsets "
          f"(LR, no balancing, 12 folds) ...")

    def make_lr_default(n_neg, n_pos):
        return LogisticRegression(class_weight=None, max_iter=1000,
                                  random_state=RANDOM_SEED)

    rows = []
    for name, feats in feature_subsets.items():
        fold_recs = walk_forward_basic(dataset, feats, make_lr_default)
        agg       = agg_folds(fold_recs)
        rows.append({
            "subset_name": name,
            "n_features":  len(feats),
            "features":    "|".join(feats),
            **agg,
        })
        print(f"  {name:30s} ({len(feats):2d} feats)  "
              f"recall={agg['avg_recall']:.3f}  "
              f"prec={agg['avg_precision']:.3f}  "
              f"f1={agg['avg_f1']:.3f}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(OPT_DIR, "feature_selection_results.csv"), index=False)
    print("  Saved: feature_selection_results.csv")

    # -----------------------------------------------------------------------
    # Plot: feature subset comparison
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    x       = np.arange(len(results_df))
    w       = 0.25
    ax.bar(x - w, results_df["avg_recall"],    w, label="Recall",    color="#1f77b4", alpha=0.85)
    ax.bar(x,     results_df["avg_precision"], w, label="Precision", color="#ff7f0e", alpha=0.85)
    ax.bar(x + w, results_df["avg_f1"],        w, label="F1",        color="#2ca02c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["subset_name"], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Metric value (avg, 12 folds)")
    ax.set_title("Feature Selection: LR Performance by Subset")
    ax.axhline(0.75, color="gray", linestyle="--", linewidth=1, label="Recall threshold = 0.75")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OPT_DIR, "feature_selection_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: feature_selection_comparison.png")

    # -----------------------------------------------------------------------
    # Select best subset
    # -----------------------------------------------------------------------
    # Winner: highest F1 among subsets with recall >= 0.75;
    # fallback: highest recall if none qualifies
    candidates = results_df[results_df["avg_recall"] >= 0.75]
    if not candidates.empty:
        best_row   = candidates.loc[candidates["avg_f1"].idxmax()]
        win_reason = "highest F1 with recall >= 0.75"
    else:
        best_row   = results_df.loc[results_df["avg_recall"].idxmax()]
        win_reason = "highest recall (no subset reached recall >= 0.75)"

    best_features = best_row["features"].split("|")
    print(f"\n  --> Best subset: '{best_row['subset_name']}' ({len(best_features)} features)")
    print(f"       Reason   : {win_reason}")
    print(f"       Features : {best_features}")
    print(f"       Recall={best_row['avg_recall']:.3f}  F1={best_row['avg_f1']:.3f}")

    return best_features, results_df


# ===========================================================================
# STEP 2 — Hyperparameter Tuning
# ===========================================================================

def step2_hyperparameter_tuning(dataset: pd.DataFrame,
                                  best_features: list) -> tuple:
    """
    Inner 3-fold stratified CV per walk-forward fold.
    Trains LR, RF, XGBoost, GradientBoosting with GridSearchCV(scoring='f1').

    Returns (best_model_name: str, best_params: dict, results_df: pd.DataFrame)
    """
    print("\n" + "=" * 65)
    print("STEP 2 -- Hyperparameter Tuning")
    print(f"  Feature set : {best_features}")
    print(f"  Inner CV    : StratifiedKFold(n_splits=3), scoring='f1'")
    print("=" * 65)

    X_all = dataset[best_features].values
    y_all = dataset[TARGET_COL].values.astype(int)
    dates = dataset.index

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    # Define model configs: (name, base_estimator_factory, param_grid)
    model_configs = [
        ("LogisticRegression",
         lambda n_neg, n_pos: LogisticRegression(random_state=RANDOM_SEED),
         PARAM_GRID_LR),
        ("RandomForest",
         lambda n_neg, n_pos: RandomForestClassifier(random_state=RANDOM_SEED),
         PARAM_GRID_RF),
        ("GradientBoosting",
         lambda n_neg, n_pos: GradientBoostingClassifier(random_state=RANDOM_SEED),
         PARAM_GRID_GB),
    ]
    if HAS_XGB:
        model_configs.append((
            "XGBoost",
            lambda n_neg, n_pos: XGBClassifier(
                scale_pos_weight=n_neg / n_pos if n_pos > 0 else 1.0,
                eval_metric="logloss", verbosity=0, random_state=RANDOM_SEED,
            ),
            PARAM_GRID_XGB,
        ))

    all_fold_rows = []
    per_model_params = collections.defaultdict(list)   # model -> list of best_params dicts
    per_model_folds  = collections.defaultdict(list)   # model -> list of fold metric dicts

    for model_name, base_factory, param_grid in model_configs:
        n_combos = 1
        for v in param_grid.values():
            n_combos *= len(v)
        print(f"\n  [{model_name}]  {n_combos} combinations x 3-fold x 12 outer folds "
              f"= {n_combos * 3 * 12} fits ...")
        t0 = time.time()

        for fold_idx, test_year in enumerate(TEST_YEARS):
            fold_num   = fold_idx + 1
            train_mask = dates.year <= (test_year - 1)
            test_mask  = dates.year == test_year

            X_train = X_all[train_mask];  y_train = y_all[train_mask]
            X_test  = X_all[test_mask];   y_test  = y_all[test_mask]

            if len(X_train) == 0 or len(X_test) == 0:
                continue

            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())

            scaler     = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc  = scaler.transform(X_test)

            # Build base estimator (with per-fold scale_pos_weight for XGB)
            base_est = base_factory(n_neg, n_pos)

            grid = GridSearchCV(
                base_est, param_grid,
                cv=inner_cv, scoring="f1",
                refit=True, n_jobs=1,
            )
            grid.fit(X_train_sc, y_train)

            y_pred = grid.best_estimator_.predict(X_test_sc)

            fold_metrics = {
                "model":           model_name,
                "fold":            fold_num,
                "test_year":       test_year,
                "best_params":     safe_json(grid.best_params_),
                "recall_shift":    recall_score(   y_test, y_pred, pos_label=1, zero_division=0),
                "precision_shift": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
                "f1_shift":        f1_score(       y_test, y_pred, pos_label=1, zero_division=0),
                "accuracy":        accuracy_score( y_test, y_pred),
            }
            all_fold_rows.append(fold_metrics)
            per_model_params[model_name].append(grid.best_params_)
            per_model_folds[model_name].append(fold_metrics)

        elapsed = time.time() - t0
        agg = agg_folds(per_model_folds[model_name])
        print(f"    Done ({elapsed:.0f}s)  recall={agg['avg_recall']:.3f}  "
              f"prec={agg['avg_precision']:.3f}  f1={agg['avg_f1']:.3f}")

    # Save per-fold results
    results_df = pd.DataFrame(all_fold_rows)
    results_df.to_csv(os.path.join(OPT_DIR, "hyperparameter_tuning_results.csv"), index=False)
    print("\n  Saved: hyperparameter_tuning_results.csv")

    # Build aggregated summary with most common best params
    summary_rows = []
    for model_name, fold_recs in per_model_folds.items():
        agg         = agg_folds(fold_recs)
        modal_params = most_common_params(per_model_params[model_name])
        summary_rows.append({
            "model": model_name,
            "most_common_best_params": safe_json(modal_params),
            **agg,
        })
        print(f"  {model_name:22s}  recall={agg['avg_recall']:.3f}  "
              f"f1={agg['avg_f1']:.3f}  "
              f"best_params={safe_json(modal_params)}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(OPT_DIR, "hyperparameter_tuning_summary.csv"), index=False
    )

    # Select winner: model with highest avg F1
    best_row       = summary_df.loc[summary_df["avg_f1"].idxmax()]
    best_model_name = best_row["model"]
    best_params     = json.loads(best_row["most_common_best_params"])

    print(f"\n  --> Best model: '{best_model_name}'  avg_f1={best_row['avg_f1']:.3f}")
    print(f"       Params: {best_params}")

    return best_model_name, best_params, results_df


# ===========================================================================
# STEP 3 — Decision Threshold Tuning
# ===========================================================================

def step3_threshold_tuning(dataset: pd.DataFrame, best_features: list,
                             best_model_name: str, best_params: dict) -> tuple:
    """
    Sweep probability thresholds 0.10 -- 0.90 in steps of 0.05.
    Uses the best model + modal best params from Step 2.

    Returns (threshold_f1: float, threshold_recall90: float or None,
             thresh_df: pd.DataFrame)
    """
    print("\n" + "=" * 65)
    print("STEP 3 -- Decision Threshold Tuning")
    print(f"  Model: {best_model_name}  |  params: {best_params}")
    print("=" * 65)

    def make_best_model(n_neg, n_pos):
        return make_model(best_model_name, best_params, n_neg, n_pos)

    # Collect per-sample probabilities across all 12 folds
    proba_df = walk_forward_proba(dataset, best_features, make_best_model)

    # Sweep thresholds, compute per-fold metrics
    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    rows = []

    for fold_num in sorted(proba_df["fold"].unique()):
        fold_data = proba_df[proba_df["fold"] == fold_num]
        y_true    = fold_data["y_true"].values
        y_proba   = fold_data["y_proba_1"].values

        for thr in thresholds:
            y_pred = (y_proba >= thr).astype(int)
            rows.append({
                "fold":      fold_num,
                "threshold": thr,
                "recall":    recall_score(   y_true, y_pred, pos_label=1, zero_division=0),
                "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
                "f1":        f1_score(       y_true, y_pred, pos_label=1, zero_division=0),
                "accuracy":  accuracy_score( y_true, y_pred),
            })

    thresh_df = pd.DataFrame(rows)
    thresh_df.to_csv(os.path.join(OPT_DIR, "threshold_tuning_results.csv"), index=False)
    print("  Saved: threshold_tuning_results.csv")

    # Aggregate across folds
    thr_agg = (thresh_df
               .groupby("threshold")[["recall", "precision", "f1", "accuracy"]]
               .agg(["mean", "std"])
               .reset_index())
    # Flatten MultiIndex columns
    thr_agg.columns = [
        "_".join(c).strip("_") if c[1] else c[0]
        for c in thr_agg.columns
    ]

    # threshold_f1: highest avg F1
    best_f1_idx  = thr_agg["f1_mean"].idxmax()
    threshold_f1 = float(thr_agg.loc[best_f1_idx, "threshold"])
    f1_at_tf1    = float(thr_agg.loc[best_f1_idx, "f1_mean"])
    rec_at_tf1   = float(thr_agg.loc[best_f1_idx, "recall_mean"])
    prec_at_tf1  = float(thr_agg.loc[best_f1_idx, "precision_mean"])

    # threshold_recall90: lowest threshold where avg recall >= 0.90
    candidates_r90 = thr_agg[thr_agg["recall_mean"] >= 0.90]
    if not candidates_r90.empty:
        threshold_recall90 = float(candidates_r90["threshold"].min())
        row_r90 = thr_agg[thr_agg["threshold"] == threshold_recall90].iloc[0]
        f1_at_r90   = float(row_r90["f1_mean"])
        rec_at_r90  = float(row_r90["recall_mean"])
        prec_at_r90 = float(row_r90["precision_mean"])
    else:
        threshold_recall90 = None
        f1_at_r90 = rec_at_r90 = prec_at_r90 = float("nan")
        print("  NOTE: no threshold achieved avg recall >= 0.90 across all folds.")

    print(f"\n  threshold_f1 = {threshold_f1:.2f}  "
          f"(recall={rec_at_tf1:.3f}, prec={prec_at_tf1:.3f}, f1={f1_at_tf1:.3f})")
    if threshold_recall90 is not None:
        print(f"  threshold_recall90 = {threshold_recall90:.2f}  "
              f"(recall={rec_at_r90:.3f}, prec={prec_at_r90:.3f}, f1={f1_at_r90:.3f})")

    # -----------------------------------------------------------------------
    # Plot: threshold curve
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    for metric, color, label in [
        ("recall",    "#1f77b4", "Recall"),
        ("precision", "#ff7f0e", "Precision"),
        ("f1",        "#2ca02c", "F1"),
    ]:
        means = thr_agg[f"{metric}_mean"].values
        stds  = thr_agg[f"{metric}_std"].values
        ax.plot(thresholds, means, color=color, label=label, linewidth=2)
        ax.fill_between(thresholds, means - stds, means + stds,
                        alpha=0.15, color=color)

    ax.axvline(threshold_f1, color="black", linestyle="--", linewidth=1.5,
               label=f"Optimal F1 threshold = {threshold_f1:.2f}")
    if threshold_recall90 is not None:
        ax.axvline(threshold_recall90, color="red", linestyle=":", linewidth=1.5,
                   label=f"Recall>=0.90 threshold = {threshold_recall90:.2f}")

    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Metric Value (avg across 12 folds)", fontsize=11)
    ax.set_title(f"Threshold Tuning: {best_model_name}\n"
                 f"(Shaded = ±1 std, walk-forward 2013-2024)", fontsize=11)
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OPT_DIR, "threshold_curve.png"), dpi=150)
    plt.close()
    print("  Saved: threshold_curve.png")

    return threshold_f1, threshold_recall90, thresh_df, thr_agg


# ===========================================================================
# STEP 4 — Horizon Sensitivity
# ===========================================================================

def step4_horizon_sensitivity(features_df: pd.DataFrame, regime: pd.Series,
                               best_features: list, best_model_name: str,
                               best_params: dict, threshold_f1: float) -> tuple:
    """
    Test H in {3, 5, 10, 15, 20}.  For each H: recompute labels, run walk-forward
    with both threshold=0.5 and threshold=threshold_f1.

    Returns (best_H: int, horizon_df: pd.DataFrame)
    """
    print("\n" + "=" * 65)
    print("STEP 4 -- Horizon Sensitivity")
    print(f"  Model: {best_model_name}  |  thresholds: 0.50 and {threshold_f1:.2f}")
    print("=" * 65)

    horizons = [3, 5, 10, 15, 20]

    def make_best_model(n_neg, n_pos):
        return make_model(best_model_name, best_params, n_neg, n_pos)

    rows = []
    for h in horizons:
        ds      = build_dataset_for_h(h, features_df, regime)
        n0      = int((ds[TARGET_COL] == 0).sum())
        n1      = int((ds[TARGET_COL] == 1).sum())
        pct_sh  = round(100 * n1 / (n0 + n1), 1)

        for thr in [0.5, threshold_f1]:
            fold_recs = walk_forward_basic(ds, best_features, make_best_model, thr)
            agg       = agg_folds(fold_recs)
            rows.append({
                "H":                        h,
                "class_distribution_shift_pct": pct_sh,
                "threshold":                thr,
                **agg,
            })

        # Print summary for this H
        r_t5  = next(r for r in rows if r["H"] == h and r["threshold"] == 0.5)
        r_tf1 = next(r for r in rows if r["H"] == h and r["threshold"] == threshold_f1)
        print(f"  H={h:2d} ({pct_sh:4.1f}% shift)  "
              f"thr=0.50: recall={r_t5['avg_recall']:.3f}  f1={r_t5['avg_f1']:.3f}  ||  "
              f"thr={threshold_f1:.2f}: recall={r_tf1['avg_recall']:.3f}  "
              f"f1={r_tf1['avg_f1']:.3f}")

    horizon_df = pd.DataFrame(rows)
    horizon_df.to_csv(os.path.join(OPT_DIR, "horizon_sensitivity_results.csv"), index=False)
    print("  Saved: horizon_sensitivity_results.csv")

    # Select best_H: highest avg_f1 at threshold_f1
    at_best_thr = horizon_df[horizon_df["threshold"] == threshold_f1]
    best_H      = int(at_best_thr.loc[at_best_thr["avg_f1"].idxmax(), "H"])
    print(f"\n  --> Best H = {best_H}  "
          f"(highest F1 at threshold={threshold_f1:.2f})")

    # -----------------------------------------------------------------------
    # Plot: horizon sensitivity
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, thr, title_suffix in [
        (axes[0], 0.5,         "threshold = 0.50 (default)"),
        (axes[1], threshold_f1, f"threshold = {threshold_f1:.2f} (optimal F1)"),
    ]:
        sub = horizon_df[horizon_df["threshold"] == thr]
        ax.plot(sub["H"], sub["avg_recall"], "o-", color="#1f77b4",
                label="Recall", linewidth=2, markersize=7)
        ax.plot(sub["H"], sub["avg_f1"],     "s-", color="#2ca02c",
                label="F1",     linewidth=2, markersize=7)
        ax.fill_between(sub["H"],
                        sub["avg_recall"] - sub["std_recall"],
                        sub["avg_recall"] + sub["std_recall"],
                        alpha=0.12, color="#1f77b4")
        ax.fill_between(sub["H"],
                        sub["avg_f1"] - sub["std_f1"],
                        sub["avg_f1"] + sub["std_f1"],
                        alpha=0.12, color="#2ca02c")
        ax.axvline(best_H, color="black", linestyle="--", linewidth=1.2,
                   label=f"Best H = {best_H}")
        ax.set_xticks(horizons)
        ax.set_xlabel("Horizon H (trading days)", fontsize=10)
        ax.set_ylabel("Metric value (avg, 12 folds)", fontsize=10)
        ax.set_title(f"Horizon Sensitivity\n{title_suffix}", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Horizon Sensitivity: {best_model_name}", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OPT_DIR, "horizon_sensitivity.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: horizon_sensitivity.png")

    return best_H, horizon_df


# ===========================================================================
# STEP 5 — Final Model Assembly
# ===========================================================================

def step5_final_model(features_df: pd.DataFrame, regime: pd.Series,
                       best_features: list, best_model_name: str,
                       best_params: dict, best_threshold: float,
                       best_H: int) -> dict:
    """
    Run the full 12-fold walk-forward with the best combination:
      features=best_features, model=best_model_name, params=best_params,
      threshold=best_threshold, H=best_H.

    Returns a summary dict of aggregated metrics.
    """
    print("\n" + "=" * 65)
    print("STEP 5 -- Final Model Assembly")
    print(f"  Model     : {best_model_name}")
    print(f"  Features  : {best_features}")
    print(f"  Horizon   : H={best_H}")
    print(f"  Threshold : {best_threshold:.2f}")
    print("=" * 65)

    dataset = build_dataset_for_h(best_H, features_df, regime)

    def make_best_model(n_neg, n_pos):
        return make_model(best_model_name, best_params, n_neg, n_pos)

    fold_recs = walk_forward_basic(dataset, best_features, make_best_model, best_threshold)
    agg       = agg_folds(fold_recs)

    # Save per-fold results
    fold_df = pd.DataFrame(fold_recs)
    fold_df.to_csv(os.path.join(OPT_DIR, "final_model_fold_results.csv"), index=False)

    # Save summary
    summary = {
        "model":         best_model_name,
        "features":      "|".join(best_features),
        "n_features":    len(best_features),
        "H":             best_H,
        "threshold":     best_threshold,
        "best_params":   safe_json(best_params),
        **agg,
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(OPT_DIR, "final_model_results.csv"), index=False
    )

    print(f"\n  Recall    : {agg['avg_recall']:.3f} ± {agg['std_recall']:.3f}")
    print(f"  Precision : {agg['avg_precision']:.3f} ± {agg['std_precision']:.3f}")
    print(f"  F1        : {agg['avg_f1']:.3f} ± {agg['std_f1']:.3f}")
    print(f"  Accuracy  : {agg['avg_accuracy']:.3f} ± {agg['std_accuracy']:.3f}")
    print("  Saved: final_model_results.csv, final_model_fold_results.csv")

    return agg


# ===========================================================================
# STEP 6 — Summary Plot
# ===========================================================================

def step6_summary_plot(feat_sel_df: pd.DataFrame, thr_agg: pd.DataFrame,
                        best_model_name: str, threshold_f1: float,
                        threshold_recall90, final_agg: dict) -> None:
    """
    2-panel figure:
      Left  — bar chart comparing ablation baselines vs. optimized model
      Right — threshold curve (reused from Step 3)
    """
    print("\n" + "=" * 65)
    print("STEP 6 -- Summary Plot")
    print("=" * 65)

    # Read ablation baselines
    ablation_path = os.path.join(RESULTS_BASE, "ablation_summary.csv")
    if os.path.exists(ablation_path):
        ab_df = pd.read_csv(ablation_path)
        p1 = ab_df[(ab_df["phase"] == 1) & (ab_df["model"] == "LogisticRegression")].iloc[0]
        p4_rf_rows = ab_df[(ab_df["phase"] == 4) & (ab_df["model"] == "RandomForest")]
        p4_rf = p4_rf_rows.iloc[0] if not p4_rf_rows.empty else None
    else:
        p1 = p4_rf = None

    # Bar data: configurations to compare
    configs   = []
    recalls   = []
    precisions = []
    f1s       = []

    if p1 is not None:
        configs.append("Ablation P1\n(LR, 3 feat)")
        recalls.append(p1["avg_recall"]);  precisions.append(p1["avg_precision"]);  f1s.append(p1["avg_f1"])

    if p4_rf is not None:
        configs.append("Ablation P4\n(RF, 10 feat)")
        recalls.append(p4_rf["avg_recall"]); precisions.append(p4_rf["avg_precision"]); f1s.append(p4_rf["avg_f1"])

    configs.append(f"Optimized\n({best_model_name[:4]})")
    recalls.append(final_agg["avg_recall"])
    precisions.append(final_agg["avg_precision"])
    f1s.append(final_agg["avg_f1"])

    fig, (ax_bar, ax_thr) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: bar chart
    x = np.arange(len(configs))
    w = 0.25
    ax_bar.bar(x - w, recalls,    w, label="Recall",    color="#1f77b4", alpha=0.85)
    ax_bar.bar(x,     precisions, w, label="Precision", color="#ff7f0e", alpha=0.85)
    ax_bar.bar(x + w, f1s,        w, label="F1",        color="#2ca02c", alpha=0.85)

    for bars, vals in [(ax_bar.containers[0], recalls),
                       (ax_bar.containers[1], precisions),
                       (ax_bar.containers[2], f1s)]:
        for bar, v in zip(bars, vals):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(configs, fontsize=9)
    ax_bar.set_ylim(0, 1.12)
    ax_bar.set_ylabel("Metric value (avg, 12 folds)", fontsize=10)
    ax_bar.set_title("Optimization Progress:\nBaselines vs. Final Model", fontsize=10)
    ax_bar.legend(fontsize=9)
    ax_bar.grid(True, axis="y", alpha=0.3)

    # Right: threshold curve
    thresholds = thr_agg["threshold"].values
    for metric, color, label in [
        ("recall",    "#1f77b4", "Recall"),
        ("precision", "#ff7f0e", "Precision"),
        ("f1",        "#2ca02c", "F1"),
    ]:
        means = thr_agg[f"{metric}_mean"].values
        stds  = thr_agg[f"{metric}_std"].values
        ax_thr.plot(thresholds, means, color=color, label=label, linewidth=2)
        ax_thr.fill_between(thresholds, means - stds, means + stds,
                            alpha=0.12, color=color)

    ax_thr.axvline(threshold_f1, color="black", linestyle="--", linewidth=1.5,
                   label=f"Optimal F1 = {threshold_f1:.2f}")
    if threshold_recall90 is not None:
        ax_thr.axvline(threshold_recall90, color="red", linestyle=":", linewidth=1.5,
                       label=f"Recall>=0.90 = {threshold_recall90:.2f}")

    ax_thr.set_xlabel("Decision Threshold", fontsize=10)
    ax_thr.set_ylabel("Metric Value (avg, 12 folds)", fontsize=10)
    ax_thr.set_title(f"Threshold Curve: {best_model_name}\n"
                     f"(±1 std shaded)", fontsize=10)
    ax_thr.set_xlim(0.08, 0.92)
    ax_thr.set_ylim(0, 1.05)
    ax_thr.legend(fontsize=8)
    ax_thr.grid(True, alpha=0.3)

    plt.suptitle("Regime Classifier v2 — Optimization Summary", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OPT_DIR, "optimization_summary.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: optimization_summary.png")


# ===========================================================================
# STEP 7 — Print Summary
# ===========================================================================

def step7_print_summary(best_features, best_model_name, best_params,
                         hp_agg, threshold_f1, threshold_recall90,
                         thr_agg, best_H, horizon_df, final_agg,
                         baseline_acc, feat_sel_df, high_corr_pairs) -> None:
    """
    Print the full optimization summary report to stdout.
    """
    # Helper: look up threshold row
    def thr_row(t):
        row = thr_agg[np.isclose(thr_agg["threshold"], t)]
        return row.iloc[0] if not row.empty else None

    tf1_row = thr_row(threshold_f1)
    tr90_row = thr_row(threshold_recall90) if threshold_recall90 else None

    # Best H row at threshold_f1
    hor_row = horizon_df[
        (horizon_df["H"] == best_H) &
        np.isclose(horizon_df["threshold"], threshold_f1)
    ]

    # Best model row from hp tuning
    best_hp_row = hp_agg[hp_agg["model"] == best_model_name].iloc[0]

    # Ablation baselines
    ab_path = os.path.join(RESULTS_BASE, "ablation_summary.csv")
    if os.path.exists(ab_path):
        ab = pd.read_csv(ab_path)
        p1_row   = ab[(ab["phase"] == 1) & (ab["model"] == "LogisticRegression")]
        p4rf_row = ab[(ab["phase"] == 4) & (ab["model"] == "RandomForest")]
        p1   = p1_row.iloc[0]   if not p1_row.empty   else None
        p4rf = p4rf_row.iloc[0] if not p4rf_row.empty else None
    else:
        p1 = p4rf = None

    print("\n" + "=" * 64)
    print("         REGIME CLASSIFIER v2 -- OPTIMIZATION RESULTS")
    print("=" * 64)
    print()
    print("Label: Binary shift detection (Variant A)")
    print('  "Is there a regime shift in the next H days?"')
    print("  Formulation: any(regime[t+1:t+H+1] != regime_t)")
    print()

    # --- Feature selection ---
    print("--- FEATURE SELECTION ---")
    match = feat_sel_df[feat_sel_df["features"] == "|".join(best_features)]
    subset_name = match.iloc[0]["subset_name"] if not match.empty else "custom"
    print(f"Best subset: '{subset_name}'  ({len(best_features)} features)")
    print(f"Features: {best_features}")
    print(f"Reason: highest F1 with recall >= 0.75 (or fallback: highest recall)")
    if high_corr_pairs:
        print(f"Correlation note: {len(high_corr_pairs)} pairs with |r|>0.8 detected "
              f"({', '.join([f'{a}<->{b}' for a, b, _ in high_corr_pairs])})")
    else:
        print("Correlation note: no feature pairs with |r| > 0.8")
    print()

    # --- Hyperparameter tuning ---
    print("--- HYPERPARAMETER TUNING ---")
    print(f"Best model : {best_model_name}")
    print(f"Best params: {best_params}")
    print(f"Performance (12-fold avg, threshold=0.5):")
    print(f"  Recall    : {best_hp_row['avg_recall']:.3f} +/- {best_hp_row['std_recall']:.3f}")
    print(f"  Precision : {best_hp_row['avg_precision']:.3f} +/- {best_hp_row['std_precision']:.3f}")
    print(f"  F1        : {best_hp_row['avg_f1']:.3f} +/- {best_hp_row['std_f1']:.3f}")
    print()

    # --- Threshold tuning ---
    print("--- THRESHOLD TUNING ---")
    if tf1_row is not None:
        print(f"Optimal threshold (max F1): {threshold_f1:.2f}")
        print(f"  Recall={tf1_row['recall_mean']:.3f} | "
              f"Precision={tf1_row['precision_mean']:.3f} | "
              f"F1={tf1_row['f1_mean']:.3f}")
    if tr90_row is not None:
        print(f"High-recall threshold (recall >= 0.90): {threshold_recall90:.2f}")
        print(f"  Recall={tr90_row['recall_mean']:.3f} | "
              f"Precision={tr90_row['precision_mean']:.3f} | "
              f"F1={tr90_row['f1_mean']:.3f}")
    else:
        print("High-recall threshold (recall >= 0.90): not achievable at any tested threshold")
    print()

    # --- Horizon sensitivity ---
    print("--- HORIZON SENSITIVITY ---")
    print(f"Best H: {best_H}")
    if not hor_row.empty:
        hr = hor_row.iloc[0]
        print(f"Class distribution at H={best_H}: "
              f"{hr['class_distribution_shift_pct']:.1f}% shift / "
              f"{100 - hr['class_distribution_shift_pct']:.1f}% no-shift")
        print(f"Performance at H={best_H} + threshold={threshold_f1:.2f}:")
        print(f"  Recall={hr['avg_recall']:.3f} | F1={hr['avg_f1']:.3f}")
    print()

    # --- Final optimized model ---
    print("--- FINAL OPTIMIZED MODEL ---")
    print(f"Model     : {best_model_name}")
    print(f"Features  : {best_features}")
    print(f"Horizon   : H={best_H}")
    print(f"Threshold : {threshold_f1:.2f}")
    print(f"  Recall    : {final_agg['avg_recall']:.3f} +/- {final_agg['std_recall']:.3f}")
    print(f"  Precision : {final_agg['avg_precision']:.3f} +/- {final_agg['std_precision']:.3f}")
    print(f"  F1        : {final_agg['avg_f1']:.3f} +/- {final_agg['std_f1']:.3f}")
    print(f"  Accuracy  : {final_agg['avg_accuracy']:.3f} +/- {final_agg['std_accuracy']:.3f}")
    print()

    # --- Final comparison table ---
    print("--- FINAL COMPARISON TABLE ---")
    n0_pct    = baseline_acc
    n1_pct    = 1 - baseline_acc
    print(f"{'Configuration':<44} | {'Recall':>6} | {'Prec':>6} | {'F1':>6} | {'Acc':>6}")
    print("-" * 75)
    print(f"{'Naive (always predict 0)':<44} | {'0.000':>6} | {'  NaN':>6} | {'0.000':>6} | {n0_pct:6.3f}")
    print(f"{'Naive (always predict 1)':<44} | {'1.000':>6} | {n1_pct:6.3f} | {2*n1_pct/(1+n1_pct):6.3f} | {n1_pct:6.3f}")

    if p1 is not None:
        print(f"{'Ablation Phase 1 (3 feat, LR default)':<44} | "
              f"{p1['avg_recall']:6.3f} | {p1['avg_precision']:6.3f} | "
              f"{p1['avg_f1']:6.3f} | {p1['avg_accuracy']:6.3f}")
    if p4rf is not None:
        print(f"{'Ablation Phase 4 RF (10 feat, balanced)':<44} | "
              f"{p4rf['avg_recall']:6.3f} | {p4rf['avg_precision']:6.3f} | "
              f"{p4rf['avg_f1']:6.3f} | {p4rf['avg_accuracy']:6.3f}")
    print(f"{'Optimized model (this experiment)':<44} | "
          f"{final_agg['avg_recall']:6.3f} | {final_agg['avg_precision']:6.3f} | "
          f"{final_agg['avg_f1']:6.3f} | {final_agg['avg_accuracy']:6.3f}")
    print()

    # --- Improvement deltas ---
    print("--- IMPROVEMENT OVER BASELINES ---")
    if p1 is not None:
        dr = final_agg["avg_recall"] - p1["avg_recall"]
        df = final_agg["avg_f1"]     - p1["avg_f1"]
        print(f"vs. Ablation Phase 1 (LR default) :  "
              f"Recall {dr:+.3f}, F1 {df:+.3f}")
    if p4rf is not None:
        dr = final_agg["avg_recall"] - p4rf["avg_recall"]
        df = final_agg["avg_f1"]     - p4rf["avg_f1"]
        print(f"vs. Ablation Phase 4 RF           :  "
              f"Recall {dr:+.3f}, F1 {df:+.3f}")

    print("=" * 64)
    print(f"\nAll outputs saved to: {OPT_DIR}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    t_start = time.time()

    # -----------------------------------------------------------------------
    # Load raw data and build base dataset (H=10, same as existing experiment)
    # -----------------------------------------------------------------------
    print("Loading raw data and building base dataset ...")
    prices, vix      = load_raw_data()
    regime           = compute_regime_labels(vix)
    features_df      = compute_features(prices, vix)

    # Base dataset (H=10) used in Steps 1-3
    dataset          = build_dataset_for_h(H_BASE, features_df, regime)

    n0 = int((dataset[TARGET_COL] == 0).sum())
    n1 = int((dataset[TARGET_COL] == 1).sum())
    baseline_acc = n0 / (n0 + n1)   # "always predict 0" accuracy
    print(f"  {len(dataset)} rows | shift={n1} ({100*n1/(n0+n1):.1f}%) | "
          f"no-shift={n0} ({100*n0/(n0+n1):.1f}%)")

    # -----------------------------------------------------------------------
    # Run all steps sequentially
    # -----------------------------------------------------------------------

    # Step 1: Feature selection
    best_features, feat_sel_df = step1_feature_selection(dataset)

    # Step 2: Hyperparameter tuning
    best_model_name, best_params, hp_df = step2_hyperparameter_tuning(
        dataset, best_features
    )

    # Build summary DF for Step 7
    hp_summary = []
    for mname in hp_df["model"].unique():
        sub = hp_df[hp_df["model"] == mname]
        hp_summary.append({
            "model":       mname,
            "avg_recall":    sub["recall_shift"].mean(),
            "std_recall":    sub["recall_shift"].std(),
            "avg_precision": sub["precision_shift"].mean(),
            "std_precision": sub["precision_shift"].std(),
            "avg_f1":        sub["f1_shift"].mean(),
            "std_f1":        sub["f1_shift"].std(),
            "avg_accuracy":  sub["accuracy"].mean(),
            "std_accuracy":  sub["accuracy"].std(),
        })
    hp_agg_df = pd.DataFrame(hp_summary)

    # Step 3: Threshold tuning
    threshold_f1, threshold_recall90, thresh_df, thr_agg = step3_threshold_tuning(
        dataset, best_features, best_model_name, best_params
    )

    # Step 4: Horizon sensitivity
    best_H, horizon_df = step4_horizon_sensitivity(
        features_df, regime, best_features,
        best_model_name, best_params, threshold_f1
    )

    # Step 5: Final model (use best threshold from Step 3, best H from Step 4)
    # Rebuild dataset with best_H if it differs from baseline
    final_agg = step5_final_model(
        features_df, regime, best_features,
        best_model_name, best_params, threshold_f1, best_H
    )

    # Step 6: Summary plot
    # Extract correlation info from Step 1 output
    corr_path = os.path.join(OPT_DIR, "feature_correlation_matrix.csv")
    corr      = pd.read_csv(corr_path, index_col=0)
    high_corr = []
    for i, f1 in enumerate(ALL_FEATURES):
        for j, f2 in enumerate(ALL_FEATURES):
            if j <= i:
                continue
            if abs(corr.loc[f1, f2]) > 0.8:
                high_corr.append((f1, f2, round(corr.loc[f1, f2], 3)))

    step6_summary_plot(feat_sel_df, thr_agg, best_model_name,
                       threshold_f1, threshold_recall90, final_agg)

    # Step 7: Print summary
    step7_print_summary(
        best_features, best_model_name, best_params,
        hp_agg_df, threshold_f1, threshold_recall90,
        thr_agg, best_H, horizon_df, final_agg,
        baseline_acc, feat_sel_df, high_corr,
    )

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
