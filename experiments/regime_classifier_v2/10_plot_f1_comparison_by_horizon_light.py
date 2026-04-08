"""
F1 is the primary metric for regime-shift detection because the positive
"shift" class is materially less common than "no shift". Accuracy can therefore
look deceptively strong even for methods that mostly predict 0 and miss many of
the shifts we care about. H=1, H=2, and H=3 are compared here because these
very short horizons were not part of the earlier definitive evaluation and need
to be recomputed on the same walk-forward protocol to see how near-term shift
predictability changes as the forecasting window shrinks. To keep this a fair
and lightweight horizon comparison, this script uses fixed representative model
parameters taken from earlier strong definitive settings instead of running a
fresh hyperparameter re-optimization for each horizon.
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

LABEL_VARIANT = "any_shift"
MIN_PERSISTENCE = 1
FEATURE_SET = "F"
TARGET_HORIZONS = [1, 2, 3]
MODEL_ORDER = ["LR", "XGB", "RF"]

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

FIXED_MODEL_CONFIGS = {
    "LR": {
        "display_name": "Logistic Regression",
        "params": {
            "select__k": "all",
            "model__C": 0.01,
            "model__class_weight": "balanced",
            "model__penalty": "l1",
        },
        "source_note": (
            "Fixed representative parameters from earlier definitive H=5 LR runs; "
            "most frequent strong Phase-2 configuration."
        ),
    },
    "XGB": {
        "display_name": "XGBoost",
        "params": {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "scale_pos_weight": 2.0,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 5.0,
            "subsample": 1.0,
            "colsample_bytree": 0.8,
        },
        "source_note": (
            "Fixed representative parameters from earlier strong definitive XGBoost "
            "settings; held constant across horizons for a lightweight comparison."
        ),
    },
    "RF": {
        "display_name": "Random Forest",
        "params": {
            "n_estimators": 300,
            "max_depth": None,
            "class_weight": {0: 1, 1: 5},
            "min_samples_leaf": 5,
        },
        "source_note": (
            "Fixed representative parameters from the definitive final-model RF setup; "
            "not re-optimized for H=1,2,3."
        ),
    },
}


def load_definitive_module():
    if "seaborn" not in sys.modules:
        seaborn_stub = types.ModuleType("seaborn")

        def _heatmap(*args, **kwargs):
            raise RuntimeError("seaborn heatmap stub should not be used in this script.")

        seaborn_stub.heatmap = _heatmap
        sys.modules["seaborn"] = seaborn_stub

    spec = importlib.util.spec_from_file_location("regime_definitive_light", DEFINITIVE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def numeric_mean(series):
    vals = pd.to_numeric(series, errors="coerce")
    return float(vals.mean()) if vals.notna().any() else np.nan


def json_compact(value):
    return json.dumps(value, sort_keys=True, default=str)


def config_label_for(model_key, horizon):
    config = FIXED_MODEL_CONFIGS[model_key]
    return (
        f"Fixed-parameter horizon comparison | H={horizon} | label={LABEL_VARIANT} "
        f"(min_persistence={MIN_PERSISTENCE}) | Set {FEATURE_SET} | "
        f"{config['display_name']} | params={json_compact(config['params'])}"
    )


def comparison_rule_text(horizon):
    return (
        f"Recomputed lightweight horizon comparison for H={horizon} using the definitive "
        f"walk-forward protocol with label_variant={LABEL_VARIANT}, "
        f"min_persistence={MIN_PERSISTENCE}, feature_set={FEATURE_SET}, daily time-step "
        "folds, training-only scaling, training-only correlation pruning, training-only "
        "threshold optimization, and the same six baselines. ML models use fixed "
        "representative parameters only; no new hyperparameter search is performed."
    )


def build_model(defmod, model_key, n_features):
    params = dict(FIXED_MODEL_CONFIGS[model_key]["params"])

    if model_key == "LR":
        k_val = params.pop("select__k", "all")
        if k_val != "all" and isinstance(k_val, (int, np.integer)) and int(k_val) >= n_features:
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
        return defmod.RandomForestClassifier(
            random_state=defmod.RANDOM_SEED,
            **params,
        )

    raise ValueError(f"Unsupported model key: {model_key}")


def run_walk_forward_fixed(defmod, dataset, feature_cols, model_key, horizon):
    records = []
    y_all_true = []
    y_all_prob = []
    cm_agg = np.zeros((2, 2), dtype=int)

    for fold_i, test_year in enumerate(defmod.TEST_YEARS):
        dates = dataset.index
        train_mask = dates.year <= (test_year - 1)
        test_mask = dates.year == test_year

        df_tr = dataset[train_mask]
        df_te = dataset[test_mask]
        if len(df_tr) == 0 or len(df_te) == 0:
            continue

        X_tr_raw = df_tr[feature_cols].values.astype(float)
        X_te_raw = df_te[feature_cols].values.astype(float)
        y_tr = df_tr["shift_label"].values.astype(int)
        y_te = df_te["shift_label"].values.astype(int)

        kept_idx, kept_names, dropped = defmod.drop_correlated_features(X_tr_raw, feature_cols, y_tr)
        X_tr_raw = X_tr_raw[:, kept_idx]
        X_te_raw = X_te_raw[:, kept_idx]

        scaler = defmod.StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_raw)
        X_te_s = scaler.transform(X_te_raw)

        model = build_model(defmod, model_key, len(kept_names))
        if model is None:
            return [], np.array([]), np.array([]), np.zeros((2, 2), dtype=int)

        model.fit(X_tr_s, y_tr)

        y_tr_prob = model.predict_proba(X_tr_s)[:, 1]
        y_tr_pred = (y_tr_prob >= 0.5).astype(int)
        train_m = defmod.compute_metrics(y_tr, y_tr_pred)
        train_m["pr_auc"] = defmod.compute_pr_auc(y_tr, y_tr_prob)

        opt_threshold = defmod.find_optimal_threshold(y_tr, y_tr_prob)

        y_te_prob = model.predict_proba(X_te_s)[:, 1]
        y_te_pred = (y_te_prob >= 0.5).astype(int)
        y_te_pred_opt = (y_te_prob >= opt_threshold).astype(int)

        test_m = defmod.compute_metrics(y_te, y_te_pred)
        test_m["pr_auc"] = defmod.compute_pr_auc(y_te, y_te_prob)
        test_m_opt = defmod.compute_metrics(y_te, y_te_pred_opt)

        y_all_true.extend(y_te.tolist())
        y_all_prob.extend(y_te_prob.tolist())
        cm_agg += defmod.confusion_matrix(y_te, y_te_pred, labels=[0, 1])

        rec = {
            "fold": fold_i + 1,
            "test_year": test_year,
            "n_train": len(y_tr),
            "n_test": len(y_te),
            "shift_pct_train": float(y_tr.mean() * 100),
            "shift_pct_test": float(y_te.mean() * 100),
            "best_params": json_compact(FIXED_MODEL_CONFIGS[model_key]["params"]),
            "parameter_strategy": "fixed_representative_parameters_no_search",
            "opt_threshold": opt_threshold,
            "train_recall": train_m["recall"],
            "train_precision": train_m["precision"],
            "train_f1": train_m["f1"],
            "train_pr_auc": train_m["pr_auc"],
            "test_recall": test_m["recall"],
            "test_precision": test_m["precision"],
            "test_f1": test_m["f1"],
            "test_pr_auc": test_m["pr_auc"],
            "test_accuracy": test_m["accuracy"],
            "test_far": test_m["far"],
            "gap_f1": train_m["f1"] - test_m["f1"],
            "test_f1_opt": test_m_opt["f1"],
            "test_recall_opt": test_m_opt["recall"],
            "test_prec_opt": test_m_opt["precision"],
            "dropped_features": ",".join(dropped) if dropped else "",
        }
        rec.update(defmod.compute_all_baselines(df_tr, df_te, y_tr, y_te))
        records.append(rec)

        print(
            f"    {model_key}/H{horizon} fold {fold_i + 1:2d} ({test_year})"
            f"  recall={test_m['recall']:.3f}"
            f"  prec={test_m['precision']:.3f}"
            f"  f1={test_m['f1']:.3f}"
            f"  far={test_m['far']:.3f}"
            f"  pr_auc={test_m['pr_auc']:.3f}"
            f"  opt_thr={opt_threshold:.3f}"
        )

    return records, np.array(y_all_true), np.array(y_all_prob), cm_agg


def baseline_rows_from_records(df_records, horizon):
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
            "parameter_strategy": "baseline_rule_no_ml_search",
            "fixed_parameters": "",
            "config_label": (
                f"Baseline | H={horizon} | label={LABEL_VARIANT} "
                f"(min_persistence={MIN_PERSISTENCE}) | Set {FEATURE_SET}"
            ),
            "source_note": "Same baseline rules as definitive classifier; recomputed on identical fold splits.",
        })
    return rows


def model_row_from_records(df_records, model_key, horizon):
    config = FIXED_MODEL_CONFIGS[model_key]
    return {
        "method": METHOD_NAME_MAP[model_key],
        "family": "ML Model",
        "horizon": horizon,
        "f1": numeric_mean(df_records["test_f1"]),
        "precision": numeric_mean(df_records["test_precision"]),
        "recall": numeric_mean(df_records["test_recall"]),
        "far": numeric_mean(df_records["test_far"]),
        "pr_auc": numeric_mean(df_records["test_pr_auc"]),
        "accuracy": numeric_mean(df_records["test_accuracy"]),
        "parameter_strategy": "fixed_representative_parameters_no_search",
        "fixed_parameters": json_compact(config["params"]),
        "config_label": config_label_for(model_key, horizon),
        "source_note": config["source_note"],
    }


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


def apply_family_label_colors(ax, df):
    for tick, family in zip(ax.get_yticklabels(), df["family"]):
        tick.set_color(COLOR_MAP.get(family, "black"))
        tick.set_fontweight("semibold")


def make_f1_chart(plot_df, horizon):
    df = plot_df.sort_values(["f1", "method"], ascending=[True, True]).copy()
    colors = [COLOR_MAP[family] for family in df["family"]]

    fig_height = max(5.8, 0.64 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    bars = ax.barh(df["method"], df["f1"], color=colors, edgecolor="black", linewidth=0.6)
    annotate_barh(ax, bars, df["f1"], pad=0.01)
    apply_family_label_colors(ax, df)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_MAP[family], edgecolor="black", label=family)
        for family in ["Trivial Baseline", "Heuristic Baseline", "ML Model"]
        if family in set(df["family"])
    ]

    xmax = min(1.0, max(float(df["f1"].max()) + 0.12, 0.75))
    ax.set_xlim(0, xmax)
    ax.set_xlabel("F1 Score")
    ax.set_ylabel("Method")
    ax.set_title(
        "Regime-Shift Prediction: Lightweight Fixed-Parameter F1 Comparison\n"
        f"H={horizon} | label=any_shift | min_persistence=1 | Set F"
    )
    ax.grid(axis="x", alpha=0.25)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"regime_classifier_f1_comparison_H{horizon}.png", dpi=180)
    plt.close(fig)


def make_precision_recall_f1_chart(plot_df, horizon):
    df = plot_df.sort_values(["f1", "method"], ascending=[True, True]).copy()
    metrics = ["precision", "recall", "f1"]
    labels = ["Precision", "Recall", "F1"]
    y = np.arange(len(df))
    bar_h = 0.22

    fig_height = max(5.8, 0.64 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(12.4, fig_height))
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
    apply_family_label_colors(ax, df)
    xmax = min(1.0, max(float(df[metrics].max().max()) + 0.12, 0.75))
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Score")
    ax.set_ylabel("Method")
    ax.set_title(
        "Regime-Shift Prediction: Precision, Recall, and F1\n"
        f"Lightweight Fixed-Parameter Comparison for H={horizon}"
    )
    ax.grid(axis="x", alpha=0.25)

    family_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_MAP[family], edgecolor="black", label=family)
        for family in ["Trivial Baseline", "Heuristic Baseline", "ML Model"]
    ]
    metric_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=METRIC_COLORS[metric], edgecolor="black", label=label)
        for metric, label in zip(metrics, labels)
    ]
    legend_1 = ax.legend(handles=metric_handles, loc="lower right", frameon=True, title="Metrics")
    ax.add_artist(legend_1)
    ax.legend(handles=family_handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=True, title="Method Family")

    plt.tight_layout()
    fig.savefig(OUT_DIR / f"regime_classifier_precision_recall_f1_comparison_H{horizon}.png", dpi=180)
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
        "parameter_strategy",
        "fixed_parameters",
        "config_label",
        "source_note",
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
    ordered = sorted(summary_rows, key=lambda row: row["horizon"], reverse=True)
    ml_scores = [row["best_ml_f1"] for row in ordered if not pd.isna(row["best_ml_f1"])]
    if len(ml_scores) < 2:
        return "Not enough ML horizon results to assess the H=3 -> H=1 trend."
    deltas = np.diff(ml_scores)
    if np.all(deltas > 0):
        return "Best-ML F1 improves consistently as H decreases from 3 to 1."
    if np.all(deltas < 0):
        return "Best-ML F1 worsens consistently as H decreases from 3 to 1."
    return "Best-ML F1 changes non-monotonically as H decreases from 3 to 1."


def run_horizon(defmod, features_df, regime, tenure, vix, horizon):
    print(f"\n{'=' * 72}")
    print(f"Lightweight fixed-parameter horizon comparison for H={horizon}")
    print(f"{'=' * 72}")

    ds = defmod.build_dataset(features_df, regime, tenure, vix, horizon, MIN_PERSISTENCE)
    feature_cols = [col for col in defmod.get_feature_sets()[FEATURE_SET] if col in ds.columns]
    if not feature_cols:
        raise RuntimeError(f"No Set {FEATURE_SET} features available for H={horizon}.")

    horizon_rows = []
    baseline_added = False
    comparison_rule = comparison_rule_text(horizon)

    for model_key in MODEL_ORDER:
        if model_key == "XGB" and not defmod.HAS_XGB:
            print("Skipping XGB: xgboost dependency not available.")
            continue
        if model_key == "RF" and not defmod.HAS_RF:
            print("Skipping RF: sklearn RandomForest dependency not available.")
            continue

        records, _, _, _ = run_walk_forward_fixed(defmod, ds, feature_cols, model_key, horizon)
        if not records:
            continue

        df_records = pd.DataFrame(records)
        horizon_rows.append(model_row_from_records(df_records, model_key, horizon))
        if not baseline_added:
            horizon_rows.extend(baseline_rows_from_records(df_records, horizon))
            baseline_added = True

    if not horizon_rows:
        raise RuntimeError(f"No rows were generated for H={horizon}.")

    df = pd.DataFrame(horizon_rows)
    df["comparison_rule"] = comparison_rule
    df["family_order"] = df["family"].map(FAMILY_ORDER).fillna(99)
    df = df.sort_values(["f1", "family_order", "method"], ascending=[False, True, True]).drop(columns=["family_order"])
    df, plot_df = add_chart_ranks(df)

    save_horizon_csv(df, horizon)
    make_f1_chart(plot_df, horizon)
    make_precision_recall_f1_chart(plot_df, horizon)

    summary = summarize_horizon(plot_df, horizon)
    print(f"Best method by F1: {summary['best_method']} ({summary['best_method_f1']:.3f})")
    print(f"Best heuristic baseline: {summary['best_heuristic']} ({summary['best_heuristic_f1']:.3f})")
    if not pd.isna(summary["ml_margin_vs_best_heuristic"]):
        print(f"Margin of best ML model over best heuristic: {summary['ml_margin_vs_best_heuristic']:+.3f}")

    return df, summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    defmod = load_definitive_module()

    print("=" * 72)
    print("Lightweight H=1/H=2/H=3 regime-classifier comparison")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 72)
    print(
        "Common setup: "
        f"label={LABEL_VARIANT}, min_persistence={MIN_PERSISTENCE}, "
        f"feature_set={FEATURE_SET}, walk-forward years={defmod.TEST_YEARS[0]}-{defmod.TEST_YEARS[-1]}"
    )
    for model_key in MODEL_ORDER:
        if model_key == "XGB" and not defmod.HAS_XGB:
            continue
        if model_key == "RF" and not defmod.HAS_RF:
            continue
        print(f"  {METHOD_NAME_MAP[model_key]} fixed params: {json_compact(FIXED_MODEL_CONFIGS[model_key]['params'])}")

    prices, vix = defmod.load_raw_data()
    regime = defmod.compute_regime_labels(vix)
    tenure = defmod.compute_regime_tenure(regime)
    features_df = defmod.compute_all_features(prices, vix)

    all_rows = []
    summaries = []
    for horizon in TARGET_HORIZONS:
        df_h, summary = run_horizon(defmod, features_df, regime, tenure, vix, horizon)
        all_rows.append(df_h)
        summaries.append(summary)

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
        "parameter_strategy",
        "fixed_parameters",
        "config_label",
        "source_note",
        "comparison_rule",
    ]
    combined[combined_cols].to_csv(OUT_DIR / "regime_classifier_f1_comparison_all_horizons.csv", index=False)

    trend_note = horizon_trend_note(summaries)
    print("\nSummary by horizon:")
    for summary in summaries:
        print(
            f"  H={summary['horizon']}: best={summary['best_method']} ({summary['best_method_f1']:.3f}), "
            f"best heuristic={summary['best_heuristic']} ({summary['best_heuristic_f1']:.3f}), "
            f"best ML margin={summary['ml_margin_vs_best_heuristic']:+.3f}"
        )
    print(f"Trend note: {trend_note}")

    print("\nSaved outputs:")
    for horizon in TARGET_HORIZONS:
        print(f"  {OUT_DIR / f'regime_classifier_f1_comparison_H{horizon}.csv'}")
        print(f"  {OUT_DIR / f'regime_classifier_f1_comparison_H{horizon}.png'}")
        print(f"  {OUT_DIR / f'regime_classifier_precision_recall_f1_comparison_H{horizon}.png'}")
    print(f"  {OUT_DIR / 'regime_classifier_f1_comparison_all_horizons.csv'}")


if __name__ == "__main__":
    main()
