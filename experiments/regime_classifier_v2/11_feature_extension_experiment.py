"""
11_feature_extension_experiment.py

Tests whether 3 new transition-specific features improve the current best
regime-classifier setup.

Setup:
  label           = any_shift
  min_persistence = 1
  primary horizon = H=5
  secondary       = H=7

Current best features (Set F):
  VIX_dist_nearest, crossing_pressure, threshold_instability

New features:
  1. time_since_last_threshold_touch
  2. recent_threshold_cross_attempts
  3. directional_consistency

Feature sets compared:
  F          = current best 3 features
  G          = F + all 3 new features
  H_new_only = only the 3 new features
  I_F+T1     = F + time_since_last_threshold_touch
  I_F+T2     = F + recent_threshold_cross_attempts
  I_F+T3     = F + directional_consistency

Models: Logistic Regression, XGBoost, Random Forest
Baselines: 6 (3 trivial + 3 heuristic) -- reused from definitive setup.
Fixed representative parameters, no GridSearchCV or RandomizedSearchCV.
Same 12-fold walk-forward protocol as 08_definitive_classifier.py.

Outputs saved to results/regime_classifier_v2/definitive/:
  feature_extension_results.csv
  feature_extension_model_comparison.png
  feature_extension_set_comparison.png
  feature_extension_precision_recall_f1.png
  feature_extension_summary.csv
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & top-level constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "regime_classifier_v2" / "definitive"
DEFINITIVE_SCRIPT = (
    PROJECT_ROOT / "experiments" / "regime_classifier_v2" / "08_definitive_classifier.py"
)

LABEL_VARIANT   = "any_shift"
MIN_PERSISTENCE = 1
TARGET_HORIZONS = [5, 7]
MODEL_ORDER     = ["LR", "XGB", "RF"]

FEAT_F    = ["VIX_dist_nearest", "crossing_pressure", "threshold_instability"]
NEW_FEATS = [
    "time_since_last_threshold_touch",
    "recent_threshold_cross_attempts",
    "directional_consistency",
]

FEATURE_SETS: dict[str, list[str]] = {
    "F":          FEAT_F,
    "G":          FEAT_F + NEW_FEATS,
    "H_new_only": NEW_FEATS,
    "I_F+T1":     FEAT_F + ["time_since_last_threshold_touch"],
    "I_F+T2":     FEAT_F + ["recent_threshold_cross_attempts"],
    "I_F+T3":     FEAT_F + ["directional_consistency"],
}

# Fixed representative parameters -- same as used in 10_plot_f1_comparison_by_horizon_light.py
FIXED_MODEL_CONFIGS: dict[str, dict] = {
    "LR": {
        "display_name": "Logistic Regression",
        "params": {
            "select__k":          "all",
            "model__C":           0.01,
            "model__class_weight": "balanced",
            "model__penalty":     "l1",
        },
    },
    "XGB": {
        "display_name": "XGBoost",
        "params": {
            "n_estimators":    200,
            "max_depth":       3,
            "learning_rate":   0.05,
            "scale_pos_weight": 2.0,
            "min_child_weight": 5,
            "reg_alpha":       0.1,
            "reg_lambda":      5.0,
            "subsample":       1.0,
            "colsample_bytree": 0.8,
        },
    },
    "RF": {
        "display_name": "Random Forest",
        "params": {
            "n_estimators":    300,
            "max_depth":       None,
            "class_weight":    {0: 1, 1: 5},
            "min_samples_leaf": 5,
        },
    },
}

COLOR_MAP = {
    "Trivial Baseline":   "#8c8c8c",
    "Heuristic Baseline": "#d95f02",
    "ML Model":           "#1f78b4",
}
METRIC_COLORS = {
    "precision": "#66a61e",
    "recall":    "#e6ab02",
    "f1":        "#1f78b4",
}
FEAT_SET_COLORS = {
    "F":          "#1f78b4",
    "G":          "#33a02c",
    "H_new_only": "#e31a1c",
    "I_F+T1":     "#ff7f00",
    "I_F+T2":     "#6a3d9a",
    "I_F+T3":     "#b15928",
}
MODEL_COLORS = {
    "LR":  "#1f78b4",
    "XGB": "#33a02c",
    "RF":  "#e31a1c",
}


# ===========================================================================
# Section 1: New feature computation
# ===========================================================================

def compute_new_features(vix: pd.Series, thresholds: list) -> pd.DataFrame:
    """
    Compute the 3 new transition-specific features.

    All features use only information available up to and including day t.
    No look-ahead bias.

    1. time_since_last_threshold_touch
       Number of trading days since VIX last came within 1.0 point of any
       regime threshold.  High value = VIX has been comfortably in mid-regime
       for a long time.  Low value = recent threshold proximity.

    2. recent_threshold_cross_attempts
       Count of days in the past 10 trading days on which VIX was within
       1.0 point of any threshold.  Captures clustering of threshold
       approaches, which often precedes an actual crossing.

    3. directional_consistency
       In the last 5 trading days, the fraction of daily VIX moves that went
       up, mapped to a 0-1 consistency score via |2*frac_up - 1|.
       0 = choppy alternating moves; 1 = all moves in the same direction.
       Directional consistency near a threshold suggests an impending cross.
    """
    vix_arr        = vix.values
    n              = len(vix_arr)
    touch_distance = 1.0

    # ---- 1. time_since_last_threshold_touch --------------------------------
    near_threshold = np.array(
        [min(abs(v - t) for t in thresholds) < touch_distance for v in vix_arr],
        dtype=bool,
    )
    time_since = np.full(n, np.nan)
    last_touch = -1
    for i in range(n):
        if near_threshold[i]:
            last_touch = i
        if last_touch >= 0:
            time_since[i] = float(i - last_touch)

    # ---- 2. recent_threshold_cross_attempts --------------------------------
    near_series     = pd.Series(near_threshold.astype(float), index=vix.index)
    recent_attempts = near_series.rolling(10, min_periods=1).sum()

    # ---- 3. directional_consistency ----------------------------------------
    daily_change = vix.diff()
    up_day       = (daily_change > 0).astype(float)
    frac_up      = up_day.rolling(5, min_periods=1).mean()
    dir_consist  = (2.0 * frac_up - 1.0).abs()

    return pd.DataFrame(
        {
            "time_since_last_threshold_touch": pd.Series(time_since, index=vix.index),
            "recent_threshold_cross_attempts": recent_attempts,
            "directional_consistency":         dir_consist,
        },
        index=vix.index,
    )


# ===========================================================================
# Section 2: Load definitive module
# ===========================================================================

def load_definitive_module():
    """
    Dynamically import 08_definitive_classifier.py to reuse its walk-forward
    infrastructure without modifying or executing its main() function.
    A seaborn stub is injected so the import succeeds in environments where
    seaborn is unavailable.
    """
    if "seaborn" not in sys.modules:
        stub = types.ModuleType("seaborn")
        stub.heatmap = lambda *a, **kw: None  # type: ignore[attr-defined]
        sys.modules["seaborn"] = stub

    spec   = importlib.util.spec_from_file_location("regime_definitive", DEFINITIVE_SCRIPT)
    module = importlib.util.module_from_spec(spec)       # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(module)                      # type: ignore[union-attr]
    return module


# ===========================================================================
# Section 3: Build model
# ===========================================================================

def build_model(defmod, model_key: str, n_features: int):
    """Instantiate a fixed-parameter model.  Returns None if dependency missing."""
    params = dict(FIXED_MODEL_CONFIGS[model_key]["params"])

    if model_key == "LR":
        k_val = params.pop("select__k", "all")
        if (
            k_val != "all"
            and isinstance(k_val, (int, np.integer))
            and int(k_val) >= n_features
        ):
            k_val = "all"
        return defmod.Pipeline([
            ("select", defmod.SelectKBest(defmod.f_classif, k=k_val)),
            ("model", defmod.LogisticRegression(
                solver="saga",
                max_iter=2000,
                random_state=defmod.RANDOM_SEED,
                C=params["model__C"],
                class_weight=params["model__class_weight"],
                penalty=params["model__penalty"],
            )),
        ])

    if model_key == "XGB":
        if not defmod.HAS_XGB:
            return None
        return defmod.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=defmod.RANDOM_SEED,
            verbosity=0,
            **params,
        )

    if model_key == "RF":
        if not defmod.HAS_RF:
            return None
        return defmod.RandomForestClassifier(random_state=defmod.RANDOM_SEED, **params)

    raise ValueError(f"Unsupported model key: {model_key}")


# ===========================================================================
# Section 4: Walk-forward runner
# ===========================================================================

def run_walk_forward(defmod, dataset: pd.DataFrame, feature_cols: list,
                     model_key: str, horizon: int):
    """
    12-fold expanding-window walk-forward evaluation with fixed parameters.

    Per fold (training-data-only):
      - correlation pruning (|r| > 0.90)
      - StandardScaler fit
      - optimal threshold search on training labels

    Returns (records, y_all_true, y_all_prob).
    records is an empty list if the model dependency is missing.
    """
    records    = []
    y_all_true = []
    y_all_prob = []

    for fold_i, test_year in enumerate(defmod.TEST_YEARS):
        dates      = dataset.index
        train_mask = dates.year <= (test_year - 1)
        test_mask  = dates.year == test_year

        df_tr = dataset[train_mask]
        df_te = dataset[test_mask]
        if len(df_tr) == 0 or len(df_te) == 0:
            continue

        X_tr_raw = df_tr[feature_cols].values.astype(float)
        X_te_raw = df_te[feature_cols].values.astype(float)
        y_tr     = df_tr["shift_label"].values.astype(int)
        y_te     = df_te["shift_label"].values.astype(int)

        # Drop correlated features (training data only)
        kept_idx, kept_names, dropped = defmod.drop_correlated_features(
            X_tr_raw, feature_cols, y_tr
        )
        X_tr_raw = X_tr_raw[:, kept_idx]
        X_te_raw = X_te_raw[:, kept_idx]

        # Scale (training data only)
        scaler   = defmod.StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr_raw)
        X_te_s   = scaler.transform(X_te_raw)

        model = build_model(defmod, model_key, len(kept_names))
        if model is None:
            return [], np.array([]), np.array([])

        model.fit(X_tr_s, y_tr)

        y_tr_prob   = model.predict_proba(X_tr_s)[:, 1]
        train_m     = defmod.compute_metrics(y_tr, (y_tr_prob >= 0.5).astype(int))
        train_m["pr_auc"] = defmod.compute_pr_auc(y_tr, y_tr_prob)

        opt_threshold = defmod.find_optimal_threshold(y_tr, y_tr_prob)

        y_te_prob = model.predict_proba(X_te_s)[:, 1]
        y_te_pred = (y_te_prob >= 0.5).astype(int)
        test_m    = defmod.compute_metrics(y_te, y_te_pred)
        test_m["pr_auc"] = defmod.compute_pr_auc(y_te, y_te_prob)

        y_all_true.extend(y_te.tolist())
        y_all_prob.extend(y_te_prob.tolist())

        rec = {
            "fold":              fold_i + 1,
            "test_year":         test_year,
            "n_train":           len(y_tr),
            "n_test":            len(y_te),
            "shift_pct_train":   float(y_tr.mean() * 100),
            "shift_pct_test":    float(y_te.mean() * 100),
            "opt_threshold":     opt_threshold,
            "train_f1":          train_m["f1"],
            "train_recall":      train_m["recall"],
            "train_precision":   train_m["precision"],
            "train_pr_auc":      train_m["pr_auc"],
            "test_f1":           test_m["f1"],
            "test_recall":       test_m["recall"],
            "test_precision":    test_m["precision"],
            "test_pr_auc":       test_m["pr_auc"],
            "test_accuracy":     test_m["accuracy"],
            "test_far":          test_m["far"],
            "gap_f1":            train_m["f1"] - test_m["f1"],
            "dropped_features":  ",".join(dropped) if dropped else "",
        }
        rec.update(defmod.compute_all_baselines(df_tr, df_te, y_tr, y_te))
        records.append(rec)

        print(
            f"      fold {fold_i + 1:2d} ({test_year})"
            f"  f1={test_m['f1']:.3f}"
            f"  prec={test_m['precision']:.3f}"
            f"  rec={test_m['recall']:.3f}"
            f"  pr_auc={test_m['pr_auc']:.3f}"
        )

    return records, np.array(y_all_true), np.array(y_all_prob)


# ===========================================================================
# Section 5: Aggregation helpers
# ===========================================================================

def _numeric_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    return float(vals.mean()) if vals.notna().any() else float("nan")


def extract_model_summary(df_records: pd.DataFrame, model_key: str,
                          feat_set_name: str, horizon: int) -> dict:
    cfg = FIXED_MODEL_CONFIGS[model_key]
    return {
        "horizon":        horizon,
        "feature_set":    feat_set_name,
        "n_features":     len(FEATURE_SETS[feat_set_name]),
        "model":          cfg["display_name"],
        "model_key":      model_key,
        "family":         "ML Model",
        "f1":             _numeric_mean(df_records["test_f1"]),
        "precision":      _numeric_mean(df_records["test_precision"]),
        "recall":         _numeric_mean(df_records["test_recall"]),
        "far":            _numeric_mean(df_records["test_far"]),
        "pr_auc":         _numeric_mean(df_records["test_pr_auc"]),
        "accuracy":       _numeric_mean(df_records["test_accuracy"]),
        "gap_f1":         _numeric_mean(df_records["gap_f1"]),
        "train_f1":       _numeric_mean(df_records["train_f1"]),
        "fixed_params":   json.dumps(cfg["params"], sort_keys=True, default=str),
    }


def extract_baseline_summaries(df_records: pd.DataFrame, horizon: int) -> list[dict]:
    baseline_defs = [
        ("Always predict 0",  "Trivial Baseline",   "b1_always0"),
        ("Always predict 1",  "Trivial Baseline",   "b2_always1"),
        ("Stratified random", "Trivial Baseline",   "b3_stratified"),
        ("Yesterday's shift", "Heuristic Baseline", "b4_yesterday"),
        ("Threshold proximity","Heuristic Baseline","b5_proximity"),
        ("Regime tenure",     "Heuristic Baseline", "b6_tenure"),
    ]
    rows = []
    for method, family, prefix in baseline_defs:
        rows.append({
            "horizon":      horizon,
            "feature_set":  "baseline",
            "n_features":   0,
            "model":        method,
            "model_key":    prefix,
            "family":       family,
            "f1":           _numeric_mean(df_records[f"{prefix}_f1"]),
            "precision":    _numeric_mean(df_records[f"{prefix}_precision"]),
            "recall":       _numeric_mean(df_records[f"{prefix}_recall"]),
            "far":          float("nan"),
            "pr_auc":       float("nan"),
            "accuracy":     float("nan"),
            "gap_f1":       float("nan"),
            "train_f1":     float("nan"),
            "fixed_params": "",
        })
    return rows


def _best_individual_set(ml_df: pd.DataFrame, model_key: str, horizon: int) -> str:
    """Return the Set I variant with the highest F1 for a given model and horizon."""
    i_sets   = ["I_F+T1", "I_F+T2", "I_F+T3"]
    best_f1  = -1.0
    best_name = "n/a"
    for fs in i_sets:
        row = ml_df[
            (ml_df["model_key"] == model_key)
            & (ml_df["feature_set"] == fs)
            & (ml_df["horizon"] == horizon)
        ]
        if len(row) > 0:
            f1 = float(row["f1"].iloc[0])
            if not np.isnan(f1) and f1 > best_f1:
                best_f1  = f1
                best_name = fs
    return best_name


# ===========================================================================
# Section 6: Plots
# ===========================================================================

def _annotate_bars(ax, bars, values, fontsize=7):
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=fontsize, rotation=90,
            )


def make_model_comparison_chart(summary_df: pd.DataFrame, horizon: int) -> None:
    """
    Grouped bar chart: for each model (x-axis), show F1 by feature set (color groups).
    Saved as feature_extension_model_comparison.png.
    """
    ml_df = summary_df[
        (summary_df["horizon"] == horizon) & (summary_df["family"] == "ML Model")
    ].copy()

    feat_sets = list(FEATURE_SETS.keys())
    n_sets    = len(feat_sets)
    bar_w     = 0.12
    offsets   = np.linspace(
        -(n_sets - 1) / 2 * bar_w,
         (n_sets - 1) / 2 * bar_w,
        n_sets,
    )
    x = np.arange(len(MODEL_ORDER))

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, fs_name in enumerate(feat_sets):
        vals = []
        for mk in MODEL_ORDER:
            row = ml_df[(ml_df["model_key"] == mk) & (ml_df["feature_set"] == fs_name)]
            v = float(row["f1"].iloc[0]) if len(row) > 0 and not np.isnan(row["f1"].iloc[0]) else 0.0
            vals.append(v)
        bars = ax.bar(
            x + offsets[i], vals, bar_w * 0.9,
            label=f"Set {fs_name}",
            color=FEAT_SET_COLORS.get(fs_name, "#aaaaaa"),
            edgecolor="black", linewidth=0.5,
        )
        _annotate_bars(ax, bars, vals)

    ax.set_xticks(x)
    ax.set_xticklabels([FIXED_MODEL_CONFIGS[mk]["display_name"] for mk in MODEL_ORDER])
    ax.set_ylabel("Mean Test F1 (12 folds)")
    ax.set_title(
        f"Feature Extension: F1 by Model, grouped by Feature Set\n"
        f"H={horizon} | label={LABEL_VARIANT} | min_persistence={MIN_PERSISTENCE}"
    )
    ymax = float(ml_df["f1"].max()) if len(ml_df) > 0 else 0.5
    ax.set_ylim(0, min(1.0, ymax + 0.18))
    ax.legend(loc="upper right", frameon=True, fontsize=9, title="Feature set")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "feature_extension_model_comparison.png", dpi=180)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'feature_extension_model_comparison.png'}")


def make_set_comparison_chart(summary_df: pd.DataFrame, horizon: int) -> None:
    """
    Grouped bar chart: for each feature set (x-axis), show F1 by model (color groups).
    Saved as feature_extension_set_comparison.png.
    """
    ml_df = summary_df[
        (summary_df["horizon"] == horizon) & (summary_df["family"] == "ML Model")
    ].copy()

    feat_sets = list(FEATURE_SETS.keys())
    n_models  = len(MODEL_ORDER)
    bar_w     = 0.22
    offsets   = np.linspace(
        -(n_models - 1) / 2 * bar_w,
         (n_models - 1) / 2 * bar_w,
        n_models,
    )
    x = np.arange(len(feat_sets))

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, mk in enumerate(MODEL_ORDER):
        vals = []
        for fs_name in feat_sets:
            row = ml_df[(ml_df["model_key"] == mk) & (ml_df["feature_set"] == fs_name)]
            v = float(row["f1"].iloc[0]) if len(row) > 0 and not np.isnan(row["f1"].iloc[0]) else 0.0
            vals.append(v)
        bars = ax.bar(
            x + offsets[i], vals, bar_w * 0.9,
            label=FIXED_MODEL_CONFIGS[mk]["display_name"],
            color=MODEL_COLORS.get(mk, "#aaaaaa"),
            edgecolor="black", linewidth=0.5,
        )
        _annotate_bars(ax, bars, vals)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Set {fs}" for fs in feat_sets], rotation=15, ha="right")
    ax.set_ylabel("Mean Test F1 (12 folds)")
    ax.set_title(
        f"Feature Extension: F1 by Feature Set, grouped by Model\n"
        f"H={horizon} | label={LABEL_VARIANT} | min_persistence={MIN_PERSISTENCE}"
    )
    ymax = float(ml_df["f1"].max()) if len(ml_df) > 0 else 0.5
    ax.set_ylim(0, min(1.0, ymax + 0.18))
    ax.legend(loc="upper right", frameon=True, fontsize=9, title="Model")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "feature_extension_set_comparison.png", dpi=180)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'feature_extension_set_comparison.png'}")


def make_precision_recall_f1_chart(summary_df: pd.DataFrame, horizon: int) -> None:
    """
    Grouped bar chart: Set F vs Set G, precision/recall/F1 for each model.
    Shows whether the new features shift the precision-recall trade-off.
    Saved as feature_extension_precision_recall_f1.png.
    """
    ml_df = summary_df[
        (summary_df["horizon"] == horizon)
        & (summary_df["family"] == "ML Model")
        & (summary_df["feature_set"].isin(["F", "G"]))
    ].copy()

    metrics       = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1"]
    combo_labels  = []
    for mk in MODEL_ORDER:
        for fs in ["F", "G"]:
            combo_labels.append(f"{FIXED_MODEL_CONFIGS[mk]['display_name']}\nSet {fs}")
    n_combos = len(combo_labels)
    x        = np.arange(n_combos)
    bar_w    = 0.22
    offsets  = np.linspace(
        -(len(metrics) - 1) / 2 * bar_w,
         (len(metrics) - 1) / 2 * bar_w,
        len(metrics),
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    for m_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        vals = []
        for mk in MODEL_ORDER:
            for fs in ["F", "G"]:
                row = ml_df[(ml_df["model_key"] == mk) & (ml_df["feature_set"] == fs)]
                v = (
                    float(row[metric].iloc[0])
                    if len(row) > 0 and not np.isnan(row[metric].iloc[0])
                    else 0.0
                )
                vals.append(v)
        bars = ax.bar(
            x + offsets[m_idx], vals, bar_w * 0.9,
            label=label,
            color=METRIC_COLORS[metric],
            edgecolor="black", linewidth=0.5,
        )
        _annotate_bars(ax, bars, vals)

    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(
        f"Feature Extension: Precision, Recall, F1 -- Set F vs Set G\n"
        f"H={horizon} | label={LABEL_VARIANT} | min_persistence={MIN_PERSISTENCE}"
    )
    all_vals = [
        float(ml_df[m].max()) for m in metrics if len(ml_df) > 0 and not ml_df[m].isna().all()
    ]
    ymax = max(all_vals) if all_vals else 0.5
    ax.set_ylim(0, min(1.0, ymax + 0.18))
    ax.legend(loc="upper right", frameon=True, title="Metric")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "feature_extension_precision_recall_f1.png", dpi=180)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'feature_extension_precision_recall_f1.png'}")


# ===========================================================================
# Section 7: Main
# ===========================================================================

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    defmod = load_definitive_module()

    print("=" * 72)
    print("Feature Extension Experiment  (11_feature_extension_experiment.py)")
    print(f"Output directory : {OUT_DIR}")
    print(f"Setup            : label={LABEL_VARIANT}, min_persistence={MIN_PERSISTENCE}")
    print(f"Horizons         : {TARGET_HORIZONS}")
    print(f"Feature sets     : {list(FEATURE_SETS.keys())}")
    print(f"Models           : {MODEL_ORDER}")
    print("=" * 72)

    # ---- Load raw data and compute features --------------------------------
    prices, vix  = defmod.load_raw_data()
    regime       = defmod.compute_regime_labels(vix)
    tenure       = defmod.compute_regime_tenure(regime)
    features_df  = defmod.compute_all_features(prices, vix)

    # Compute the 3 new features and merge into the feature matrix
    new_feats_df    = compute_new_features(vix, list(defmod.THRESHOLDS))
    features_ext_df = features_df.join(new_feats_df, how="left")

    print(
        f"\nNew features computed: {NEW_FEATS}\n"
        f"Any NaN in new features: "
        f"{new_feats_df.isna().any().to_dict()}"
    )

    # ---- Walk-forward evaluation -------------------------------------------
    all_rows: list[dict] = []

    for horizon in TARGET_HORIZONS:
        print(f"\n{'=' * 72}")
        print(f"Horizon H={horizon}")
        print(f"{'=' * 72}")

        dataset         = defmod.build_dataset(
            features_ext_df, regime, tenure, vix, horizon, MIN_PERSISTENCE
        )
        baselines_added = False
        first_df        = None

        for feat_set_name, feat_cols in FEATURE_SETS.items():
            available = [c for c in feat_cols if c in dataset.columns]
            missing   = [c for c in feat_cols if c not in dataset.columns]
            if not available:
                print(f"  WARNING: Set {feat_set_name} has no available columns -- skipped.")
                continue
            if missing:
                print(f"  NOTE: Set {feat_set_name} missing {missing}; using {available}.")

            for model_key in MODEL_ORDER:
                if model_key == "XGB" and not defmod.HAS_XGB:
                    continue
                if model_key == "RF" and not defmod.HAS_RF:
                    continue

                print(
                    f"\n  >> Set {feat_set_name} | {FIXED_MODEL_CONFIGS[model_key]['display_name']}"
                    f" | H={horizon}"
                )
                records, _, _ = run_walk_forward(
                    defmod, dataset, available, model_key, horizon
                )
                if not records:
                    print(f"     No results (dependency missing or no folds).")
                    continue

                df_rec = pd.DataFrame(records)
                row    = extract_model_summary(df_rec, model_key, feat_set_name, horizon)
                all_rows.append(row)
                print(
                    f"  => mean test F1={row['f1']:.4f}  prec={row['precision']:.4f}"
                    f"  recall={row['recall']:.4f}  pr_auc={row['pr_auc']:.4f}"
                    f"  gap_f1={row['gap_f1']:+.4f}"
                )

                # Collect baselines once per horizon (they don't depend on feature set)
                if not baselines_added:
                    first_df        = df_rec
                    baselines_added = True

        # Add baseline rows once per horizon
        if first_df is not None:
            bl_rows = extract_baseline_summaries(first_df, horizon)
            all_rows.extend(bl_rows)
            print(f"\n  Baselines (H={horizon}):")
            for br in bl_rows:
                print(f"    {br['model']:22s}  F1={br['f1']:.4f}")

    if not all_rows:
        print("ERROR: No results generated.  Check dependencies and data paths.")
        return

    summary_df = pd.DataFrame(all_rows)

    # ---- Save detailed results CSV -----------------------------------------
    results_path = OUT_DIR / "feature_extension_results.csv"
    summary_df.to_csv(results_path, index=False)

    # ---- Generate plots (primary horizon = H=5) ----------------------------
    ph = TARGET_HORIZONS[0]
    print(f"\nGenerating plots for primary horizon H={ph} ...")
    make_model_comparison_chart(summary_df, ph)
    make_set_comparison_chart(summary_df, ph)
    make_precision_recall_f1_chart(summary_df, ph)

    # ---- Build improvement summary (Set G vs Set F) ------------------------
    ml_df          = summary_df[summary_df["family"] == "ML Model"].copy()
    improvement_rows: list[dict] = []
    for mk in MODEL_ORDER:
        for h in TARGET_HORIZONS:
            row_f = ml_df[
                (ml_df["model_key"] == mk)
                & (ml_df["feature_set"] == "F")
                & (ml_df["horizon"] == h)
            ]
            row_g = ml_df[
                (ml_df["model_key"] == mk)
                & (ml_df["feature_set"] == "G")
                & (ml_df["horizon"] == h)
            ]
            if len(row_f) == 0 or len(row_g) == 0:
                continue
            f1_f = float(row_f["f1"].iloc[0])
            f1_g = float(row_g["f1"].iloc[0])
            improvement_rows.append({
                "model":              FIXED_MODEL_CONFIGS[mk]["display_name"],
                "model_key":          mk,
                "horizon":            h,
                "f1_set_F":           f1_f,
                "f1_set_G":           f1_g,
                "improvement_G_F":    f1_g - f1_f,
                "best_individual_I":  _best_individual_set(ml_df, mk, h),
                "precision_set_F":    float(row_f["precision"].iloc[0]),
                "precision_set_G":    float(row_g["precision"].iloc[0]),
                "recall_set_F":       float(row_f["recall"].iloc[0]),
                "recall_set_G":       float(row_g["recall"].iloc[0]),
            })

    summary_path = OUT_DIR / "feature_extension_summary.csv"
    if improvement_rows:
        pd.DataFrame(improvement_rows).to_csv(summary_path, index=False)

    # ---- Final console summary ---------------------------------------------
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    for h in TARGET_HORIZONS:
        h_ml = ml_df[ml_df["horizon"] == h].sort_values("f1", ascending=False)
        if len(h_ml) == 0:
            continue
        best = h_ml.iloc[0]
        print(f"\n  H={h}:")
        print(f"    Best feature set : Set {best['feature_set']}")
        print(f"    Best model       : {best['model']}")
        print(f"    Best F1          : {best['f1']:.4f}")

    if improvement_rows:
        imp_df = pd.DataFrame(improvement_rows)
        for h in TARGET_HORIZONS:
            h_imp = imp_df[imp_df["horizon"] == h]
            if len(h_imp) == 0:
                continue
            print(f"\n  Set G vs Set F improvement (H={h}):")
            for _, ir in h_imp.iterrows():
                sign = "+" if ir["improvement_G_F"] >= 0 else ""
                print(
                    f"    {ir['model']:22s}"
                    f"  F1(F)={ir['f1_set_F']:.4f}"
                    f"  F1(G)={ir['f1_set_G']:.4f}"
                    f"  delta={sign}{ir['improvement_G_F']:.4f}"
                    f"  best_I={ir['best_individual_I']}"
                )

    bl_df = summary_df[summary_df["family"] != "ML Model"]
    for h in TARGET_HORIZONS:
        h_bl = bl_df[bl_df["horizon"] == h].sort_values("f1", ascending=False)
        if len(h_bl) > 0:
            b = h_bl.iloc[0]
            print(f"\n  Best baseline (H={h}): {b['model']}  F1={b['f1']:.4f}")

    # ---- Git status --------------------------------------------------------
    print("\nGit status:")
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--short"],
            capture_output=True, text=True, timeout=15,
        )
        print(result.stdout if result.stdout else "  (clean)")
        if result.stderr:
            print(result.stderr)
    except Exception as exc:
        print(f"  (could not run git status: {exc})")

    # ---- Output paths ------------------------------------------------------
    output_files = [
        OUT_DIR / "feature_extension_results.csv",
        OUT_DIR / "feature_extension_model_comparison.png",
        OUT_DIR / "feature_extension_set_comparison.png",
        OUT_DIR / "feature_extension_precision_recall_f1.png",
        OUT_DIR / "feature_extension_summary.csv",
    ]
    print("\nOutput files:")
    for f in output_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
