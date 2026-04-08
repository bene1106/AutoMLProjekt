"""
F1 is the primary metric here because regime-shift detection is imbalanced:
the "shift" class is materially rarer than "no shift", so accuracy can look
strong even when a method misses many shifts by mostly predicting 0. F1 is
more informative because it forces a joint precision/recall assessment for the
positive class we actually care about. Comparing H=1, H=2, and H=3 is useful
because it shows how predictive difficulty changes as the forecasting horizon
gets shorter and reveals whether gains come from genuinely earlier warning
signals or from easier near-term shift anticipation.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "regime_classifier_v2" / "definitive"
DEFINITIVE_SCRIPT = PROJECT_ROOT / "experiments" / "regime_classifier_v2" / "08_definitive_classifier.py"
PHASE4_FILE = OUT_DIR / "phase4_final_model.csv"

TARGET_HORIZONS = [1, 2, 3]
MODELS = ["LR", "XGB", "RF"]

COLOR_MAP = {
    "Trivial Baseline": "#8c8c8c",
    "Heuristic Baseline": "#d95f02",
    "ML Model": "#1f78b4",
}
METRIC_COLORS = {
    "precision": "#66a61e",
    "recall": "#e6ab02",
    "f1": "#1f78b4",
}

FAMILY_ORDER = {
    "ML Model": 0,
    "Heuristic Baseline": 1,
    "Trivial Baseline": 2,
}
METHOD_NAME_MAP = {
    "LR": "Logistic Regression",
    "XGB": "XGBoost",
    "RF": "Random Forest",
}


def load_definitive_module():
    if "seaborn" not in sys.modules:
        seaborn_stub = types.ModuleType("seaborn")

        def _heatmap(*args, **kwargs):
            raise RuntimeError("seaborn heatmap stub should not be used in this script.")

        seaborn_stub.heatmap = _heatmap
        sys.modules["seaborn"] = seaborn_stub
    spec = importlib.util.spec_from_file_location("regime_definitive", DEFINITIVE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    orig_grid = module.GridSearchCV
    orig_random = module.RandomizedSearchCV

    def grid_search_cv(*args, **kwargs):
        kwargs["n_jobs"] = 1
        return orig_grid(*args, **kwargs)

    def randomized_search_cv(*args, **kwargs):
        kwargs["n_jobs"] = 1
        return orig_random(*args, **kwargs)

    module.GridSearchCV = grid_search_cv
    module.RandomizedSearchCV = randomized_search_cv
    return module


def load_context():
    if not PHASE4_FILE.exists():
        raise FileNotFoundError(f"Missing definitive summary file: {PHASE4_FILE}")
    row = pd.read_csv(PHASE4_FILE).iloc[0]
    return {
        "label_variant": str(row["label_variant"]),
        "min_persistence": int(row["min_persistence"]),
        "feature_set": str(row["feature_set"]),
        "n_features": int(row["n_features"]),
        "features": str(row["features"]),
        "final_model": str(row["model"]),
        "reference_horizon": int(row["H"]),
    }


def build_model_factory(defmod, model_name, fixed_params=None):
    fixed_params = dict(fixed_params or {})

    if model_name == "LR":
        def get_model(n_neg, n_pos):
            return defmod.Pipeline([
                ("select", defmod.SelectKBest(defmod.f_classif)),
                ("model", defmod.LogisticRegression(
                    solver="saga",
                    max_iter=2000,
                    random_state=defmod.RANDOM_SEED,
                )),
            ])

        def get_grid(n_neg, n_pos, n_feats):
            if fixed_params:
                params = dict(fixed_params)
                k = params.get("select__k", "all")
                if k != "all" and isinstance(k, (int, np.integer)) and k >= n_feats:
                    params["select__k"] = "all"
                return {k2: [v2] for k2, v2 in params.items()}
            valid_k = sorted(set([k for k in [3, 5] if k < n_feats] + ["all"]))
            return {
                "select__k": valid_k,
                "model__C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [
                    "balanced",
                    {0: 1, 1: 1},
                    {0: 1, 1: 2},
                    {0: 1, 1: 3},
                    {0: 1, 1: 5},
                    {0: 1, 1: 10},
                    {0: 1, 1: 15},
                    {0: 1, 1: 20},
                ],
                "model__penalty": ["l1", "l2"],
            }

        return get_model, get_grid, False, 50

    if model_name == "XGB":
        if not defmod.HAS_XGB:
            return None

        def get_model(n_neg, n_pos):
            return defmod.XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=defmod.RANDOM_SEED,
                verbosity=0,
            )

        def get_grid(n_neg, n_pos, n_feats):
            if fixed_params:
                return {k: [v] for k, v in fixed_params.items()}
            base_ratio = n_neg / max(n_pos, 1)
            spw_options = [base_ratio * m for m in [0.5, 1.0, 2.0, 3.0, 5.0]]
            return {
                "n_estimators": [100, 200, 500],
                "max_depth": [3, 4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
                "scale_pos_weight": spw_options,
                "min_child_weight": [1, 3, 5],
                "reg_alpha": [0, 0.1, 1.0],
                "reg_lambda": [1.0, 5.0, 10.0],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            }

        n_iter = defmod.XGB_N_ITER_REDUCED if defmod.runtime_exceeded() else defmod.XGB_N_ITER_FULL
        return get_model, get_grid, True, n_iter

    if model_name == "RF":
        if not defmod.HAS_RF:
            return None

        def get_model(n_neg, n_pos):
            return defmod.RandomForestClassifier(random_state=defmod.RANDOM_SEED)

        def get_grid(n_neg, n_pos, n_feats):
            if fixed_params:
                return {k: [v] for k, v in fixed_params.items()}
            return {
                "n_estimators": [100, 300],
                "max_depth": [5, 10, None],
                "class_weight": [
                    "balanced",
                    {0: 1, 1: 5},
                    {0: 1, 1: 10},
                ],
                "min_samples_leaf": [1, 5],
            }

        return get_model, get_grid, False, 50

    raise ValueError(f"Unsupported model: {model_name}")


def numeric_mean(series):
    vals = pd.to_numeric(series, errors="coerce")
    return float(vals.mean()) if vals.notna().any() else np.nan


def baseline_rows_from_records(df_records, context, horizon):
    defs = [
        ("Always predict 0", "Trivial Baseline", "b1_always0"),
        ("Always predict 1", "Trivial Baseline", "b2_always1"),
        ("Stratified random", "Trivial Baseline", "b3_stratified"),
        ("Yesterday's shift", "Heuristic Baseline", "b4_yesterday"),
        ("Threshold proximity", "Heuristic Baseline", "b5_proximity"),
        ("Regime tenure", "Heuristic Baseline", "b6_tenure"),
    ]
    rows = []
    for method, family, prefix in defs:
        rows.append({
            "method": method,
            "family": family,
            "horizon": horizon,
            "f1": numeric_mean(df_records[f"{prefix}_f1"]),
            "precision": numeric_mean(df_records[f"{prefix}_precision"]),
            "recall": numeric_mean(df_records[f"{prefix}_recall"]),
            "far": np.nan,
            "pr_auc": np.nan,
            "accuracy": np.nan,
            "config_label": (
                f"Plan 11 extended horizon comparison | H={horizon} | "
                f"label={context['label_variant']} (min_persistence={context['min_persistence']}) | "
                f"Set {context['feature_set']}"
            ),
        })
    return rows


def model_row_from_records(df_records, context, model_name, horizon):
    return {
        "method": METHOD_NAME_MAP.get(model_name, model_name),
        "family": "ML Model",
        "horizon": horizon,
        "f1": numeric_mean(df_records["test_f1"]),
        "precision": numeric_mean(df_records["test_precision"]),
        "recall": numeric_mean(df_records["test_recall"]),
        "far": numeric_mean(df_records["test_far"]),
        "pr_auc": numeric_mean(df_records["test_pr_auc"]),
        "accuracy": numeric_mean(df_records["test_accuracy"]),
        "config_label": (
            f"Plan 11 extended horizon comparison | H={horizon} | "
            f"label={context['label_variant']} (min_persistence={context['min_persistence']}) | "
            f"Set {context['feature_set']}"
        ),
    }


def comparison_rule_text(context, horizon):
    return (
        "Fair comparison uses the definitive Plan 11 label definition and feature-set style, "
        f"extended to H={horizon}: label_variant={context['label_variant']}, "
        f"min_persistence={context['min_persistence']}, feature_set={context['feature_set']} "
        f"({context['n_features']} features from the definitive pipeline). "
        "ML models are evaluated with the same 12-fold expanding walk-forward protocol, "
        "training-only scaling/selection, threshold optimization, and baseline computation logic "
        "from 08_definitive_classifier.py. Baselines are averaged from the exact same fold splits."
    )


def add_chart_ranks(df):
    df = df.copy()
    df["chart_rank"] = np.nan
    plot_df = df.sort_values(["f1", "method"], ascending=[False, True]).reset_index(drop=True)
    for idx, method in enumerate(plot_df["method"], start=1):
        df.loc[df["method"] == method, "chart_rank"] = idx
    df["chart_rank"] = pd.to_numeric(df["chart_rank"], errors="coerce")
    return df, plot_df


def annotate_barh(ax, bars, values, pad=0.01):
    for bar, value in zip(bars, values):
        if pd.isna(value):
            continue
        x = min(float(value) + pad, 0.985)
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{float(value):.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )


def make_f1_chart(plot_df, horizon, context):
    df = plot_df.sort_values(["f1", "method"], ascending=[True, True]).copy()
    colors = [COLOR_MAP[family] for family in df["family"]]

    fig_height = max(5.5, 0.62 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    bars = ax.barh(df["method"], df["f1"], color=colors, edgecolor="black", linewidth=0.6)
    annotate_barh(ax, bars, df["f1"], pad=0.01)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_MAP[family], edgecolor="black", label=family)
        for family in ["Trivial Baseline", "Heuristic Baseline", "ML Model"]
        if family in set(df["family"])
    ]

    ax.set_xlim(0, min(1.0, max(float(df["f1"].max()) + 0.12, 0.75)))
    ax.set_xlabel("F1 Score")
    ax.set_ylabel("Method")
    ax.set_title(
        "Regime-Shift Prediction: F1-Based Comparison\n"
        f"Definitive Plan 11 Common Setup Extended to H={horizon} (Set {context['feature_set']})"
    )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"regime_classifier_f1_comparison_H{horizon}.png", dpi=150)
    plt.close(fig)


def make_precision_recall_f1_chart(plot_df, horizon, context):
    df = plot_df.sort_values(["f1", "method"], ascending=[True, True]).copy()
    metrics = ["precision", "recall", "f1"]
    labels = ["Precision", "Recall", "F1"]

    y = np.arange(len(df))
    bar_h = 0.22

    fig_height = max(5.5, 0.62 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        vals = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)
        bars = ax.barh(
            y + (idx - 1) * bar_h,
            vals,
            height=bar_h,
            color=METRIC_COLORS[metric],
            edgecolor="black",
            linewidth=0.5,
            label=label,
        )
        annotate_barh(ax, bars, vals, pad=0.008)

    ax.set_yticks(y)
    ax.set_yticklabels(df["method"])
    ax.set_xlim(0, min(1.0, max(float(df[metrics].max().max()) + 0.12, 0.75)))
    ax.set_xlabel("Score")
    ax.set_ylabel("Method")
    ax.set_title(
        "Regime-Shift Prediction: Precision, Recall, and F1\n"
        f"Definitive Plan 11 Common Setup Extended to H={horizon} (Set {context['feature_set']})"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"regime_classifier_precision_recall_f1_comparison_H{horizon}.png", dpi=150)
    plt.close(fig)


def save_horizon_csv(df, horizon):
    cols = [
        "chart_rank",
        "method",
        "family",
        "horizon",
        "f1",
        "precision",
        "recall",
        "far",
        "pr_auc",
        "accuracy",
        "config_label",
        "comparison_rule",
    ]
    df[cols].to_csv(OUT_DIR / f"regime_classifier_f1_comparison_H{horizon}.csv", index=False)


def summarize_horizon(plot_df, horizon):
    sorted_df = plot_df.sort_values(["f1", "method"], ascending=[False, True]).reset_index(drop=True)
    best_method = sorted_df.iloc[0]
    heuristic_df = sorted_df[sorted_df["family"] == "Heuristic Baseline"].copy()
    ml_df = sorted_df[sorted_df["family"] == "ML Model"].copy()
    best_heuristic = heuristic_df.iloc[0] if not heuristic_df.empty else None
    best_ml = ml_df.iloc[0] if not ml_df.empty else None
    ml_margin = np.nan
    if best_ml is not None and best_heuristic is not None:
        ml_margin = float(best_ml["f1"] - best_heuristic["f1"])
    return {
        "horizon": horizon,
        "best_method": str(best_method["method"]),
        "best_method_f1": float(best_method["f1"]),
        "best_heuristic": str(best_heuristic["method"]) if best_heuristic is not None else "n/a",
        "best_heuristic_f1": float(best_heuristic["f1"]) if best_heuristic is not None else np.nan,
        "best_ml": str(best_ml["method"]) if best_ml is not None else "n/a",
        "best_ml_f1": float(best_ml["f1"]) if best_ml is not None else np.nan,
        "ml_margin_vs_best_heuristic": ml_margin,
    }


def horizon_trend_note(summary_rows):
    if len(summary_rows) < 2:
        return "Not enough horizons to assess trend."
    ordered = sorted(summary_rows, key=lambda row: row["horizon"], reverse=True)
    ordered_f1 = [row["best_ml_f1"] for row in ordered if not pd.isna(row["best_ml_f1"])]
    if len(ordered_f1) < 2:
        return "Not enough ML results to assess trend."
    deltas = np.diff(ordered_f1)
    if np.all(deltas > 0):
        return "Best-ML F1 improves consistently as H decreases."
    if np.all(deltas < 0):
        return "Best-ML F1 worsens consistently as H decreases."
    return "Best-ML F1 changes non-monotonically as H decreases."


def run_horizon(defmod, context, features_df, regime, tenure, vix, horizon):
    print(f"\n{'=' * 70}")
    print(f"Horizon comparison for H={horizon}")
    print(f"{'=' * 70}")

    feature_cols = defmod.get_feature_sets()[context["feature_set"]]
    ds = defmod.build_dataset(
        features_df,
        regime,
        tenure,
        vix,
        horizon,
        context["min_persistence"],
    )
    available = [col for col in feature_cols if col in ds.columns]
    if not available:
        raise RuntimeError(f"No features available for H={horizon} in feature set {context['feature_set']}.")

    comparison_rule = comparison_rule_text(context, horizon)
    horizon_rows = []
    baseline_added = False

    for model_name in MODELS:
        factory = build_model_factory(defmod, model_name)
        if factory is None:
            print(f"Skipping {model_name} at H={horizon}: dependency not available.")
            continue
        get_model, get_grid, use_randomized, n_iter = factory
        records, _, _, _, _, _ = defmod.run_walk_forward_cv(
            ds,
            available,
            get_model,
            get_grid,
            use_randomized=use_randomized,
            n_iter=n_iter,
            phase_label=f"{model_name}/H{horizon}",
        )
        if not records:
            continue

        df_records = pd.DataFrame(records)
        horizon_rows.append(model_row_from_records(df_records, context, model_name, horizon))
        if not baseline_added:
            horizon_rows.extend(baseline_rows_from_records(df_records, context, horizon))
            baseline_added = True

    if not horizon_rows:
        raise RuntimeError(f"No comparison rows generated for H={horizon}.")

    df = pd.DataFrame(horizon_rows)
    df["comparison_rule"] = comparison_rule
    df["family_order"] = df["family"].map(FAMILY_ORDER).fillna(99)
    df = df.sort_values(["f1", "family_order", "method"], ascending=[False, True, True]).drop(columns=["family_order"])
    df, plot_df = add_chart_ranks(df)
    save_horizon_csv(df, horizon)
    make_f1_chart(plot_df, horizon, context)
    make_precision_recall_f1_chart(plot_df, horizon, context)

    summary = summarize_horizon(plot_df, horizon)
    print(
        f"Best method by F1: {summary['best_method']} ({summary['best_method_f1']:.3f})"
    )
    print(
        f"Best heuristic baseline: {summary['best_heuristic']} "
        f"({summary['best_heuristic_f1']:.3f})"
    )
    if not pd.isna(summary["ml_margin_vs_best_heuristic"]):
        print(
            f"Margin of best ML model over best heuristic: "
            f"{summary['ml_margin_vs_best_heuristic']:+.3f}"
        )

    return df, summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    defmod = load_definitive_module()
    context = load_context()

    print("=" * 70)
    print("Plan 11 extended horizon comparison")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 70)
    print(
        f"Using definitive setup: label={context['label_variant']} "
        f"(min_persistence={context['min_persistence']}), "
        f"feature_set={context['feature_set']}, reference_H={context['reference_horizon']}"
    )

    prices, vix = defmod.load_raw_data()
    regime = defmod.compute_regime_labels(vix)
    tenure = defmod.compute_regime_tenure(regime)
    features_df = defmod.compute_all_features(prices, vix)

    all_rows = []
    horizon_summaries = []
    for horizon in TARGET_HORIZONS:
        df_h, summary = run_horizon(defmod, context, features_df, regime, tenure, vix, horizon)
        all_rows.append(df_h)
        horizon_summaries.append(summary)

    combined = pd.concat(all_rows, ignore_index=True)
    combined_cols = [
        "method",
        "family",
        "horizon",
        "f1",
        "precision",
        "recall",
        "far",
        "pr_auc",
        "accuracy",
        "config_label",
        "comparison_rule",
    ]
    combined[combined_cols].to_csv(
        OUT_DIR / "regime_classifier_f1_comparison_all_horizons.csv",
        index=False,
    )

    trend_note = horizon_trend_note(horizon_summaries)
    print(f"\nTrend note: {trend_note}")
    print("\nSaved outputs:")
    for horizon in TARGET_HORIZONS:
        print(f"  {OUT_DIR / f'regime_classifier_f1_comparison_H{horizon}.csv'}")
        print(f"  {OUT_DIR / f'regime_classifier_f1_comparison_H{horizon}.png'}")
        print(f"  {OUT_DIR / f'regime_classifier_precision_recall_f1_comparison_H{horizon}.png'}")
    print(f"  {OUT_DIR / 'regime_classifier_f1_comparison_all_horizons.csv'}")


if __name__ == "__main__":
    main()
