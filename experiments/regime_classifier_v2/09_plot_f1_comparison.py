"""
F1 is more appropriate than accuracy for regime-shift detection because the
"shift" class is relatively rare. Accuracy can look strong by mostly predicting
"no shift", while F1 better reflects the precision/recall trade-off for the
minority class we actually care about catching.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "regime_classifier_v2" / "definitive"

PHASE4_FILE = OUT_DIR / "phase4_final_model.csv"
FULL_SUMMARY_FILE = OUT_DIR / "full_summary.csv"
PHASE3_FILE = OUT_DIR / "phase3_horizon_sweep.csv"
PHASE2_FILE = OUT_DIR / "phase2_model_results.csv"
MAX_ACC_FILE = PROJECT_ROOT / "results" / "regime_classifier_v2" / "max_accuracy" / "step3_model_comparison.csv"
OPT_FINAL_FILE = PROJECT_ROOT / "results" / "regime_classifier_v2" / "optimization" / "final_model_results.csv"

CSV_OUT = OUT_DIR / "regime_classifier_f1_comparison.csv"
FIG_F1_OUT = OUT_DIR / "regime_classifier_f1_comparison.png"
FIG_PRF_OUT = OUT_DIR / "regime_classifier_precision_recall_f1_comparison.png"

COLOR_MAP = {
    "Trivial Baseline": "#8c8c8c",
    "Heuristic Baseline": "#d95f02",
    "ML Model": "#1f78b4",
    "Legacy Reference": "#7570b3",
}


def load_csv(path):
    if not path.exists():
        print(f"WARNING: Missing input file: {path}")
        return None
    return pd.read_csv(path)


def numeric_mean(series):
    vals = pd.to_numeric(series, errors="coerce")
    return float(vals.mean()) if vals.notna().any() else np.nan


def build_context_row():
    summary = load_csv(PHASE4_FILE)
    if summary is None or summary.empty:
        summary = load_csv(FULL_SUMMARY_FILE)
    if summary is None or summary.empty:
        raise FileNotFoundError("Could not find Phase 4 / definitive summary CSV.")
    row = summary.iloc[0]
    return {
        "label_variant": str(row["label_variant"]),
        "min_persistence": int(row["min_persistence"]),
        "H": int(row["H"]),
        "feature_set": str(row["feature_set"]),
        "n_features": int(row["n_features"]),
        "features": str(row["features"]),
        "final_model": str(row["model"]),
    }


def build_phase2_model_rows(context):
    df = load_csv(PHASE2_FILE)
    if df is None or df.empty:
        return []

    h_suffix = f"_H{context['H']}"
    df = df[df["model_key"].astype(str).str.endswith(h_suffix)].copy()
    if df.empty:
        return []

    rows = []
    model_name_map = {
        "LR": "Logistic Regression",
        "XGB": "XGBoost",
        "RF": "Random Forest",
    }

    for model_key, grp in df.groupby("model_key"):
        short_name = str(model_key).split("_")[0]
        rows.append({
            "method": model_name_map.get(short_name, short_name),
            "family": "ML Model",
            "f1": numeric_mean(grp["test_f1"]),
            "precision": numeric_mean(grp["test_precision"]),
            "recall": numeric_mean(grp["test_recall"]),
            "far": numeric_mean(grp["test_far"]),
            "pr_auc": numeric_mean(grp["test_pr_auc"]),
            "accuracy": numeric_mean(grp["test_accuracy"]),
            "config_label": (
                f"Plan 11 common setup | H={context['H']} | "
                f"label={context['label_variant']} | Set {context['feature_set']}"
            ),
            "source_file": str(PHASE2_FILE.relative_to(PROJECT_ROOT)),
            "comparison_scope": "Plan11_common_setup",
            "directly_comparable": True,
            "included_in_chart": True,
        })
    return rows


def build_baseline_rows(context):
    df = load_csv(PHASE3_FILE)
    if df is None or df.empty:
        return []

    match = df[pd.to_numeric(df["H"], errors="coerce") == context["H"]]
    if match.empty:
        return []
    row = match.iloc[0]

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
            "f1": pd.to_numeric(row.get(f"{prefix}_f1"), errors="coerce"),
            "precision": pd.to_numeric(row.get(f"{prefix}_precision"), errors="coerce"),
            "recall": pd.to_numeric(row.get(f"{prefix}_recall"), errors="coerce"),
            "far": np.nan,
            "pr_auc": np.nan,
            "accuracy": np.nan,
            "config_label": (
                f"Plan 11 baseline | H={context['H']} | "
                f"label={context['label_variant']} | Set {context['feature_set']}"
            ),
            "source_file": str(PHASE3_FILE.relative_to(PROJECT_ROOT)),
            "comparison_scope": "Plan11_common_setup",
            "directly_comparable": True,
            "included_in_chart": True,
        })
    return rows


def build_legacy_rows():
    rows = []

    max_acc = load_csv(MAX_ACC_FILE)
    if max_acc is not None and not max_acc.empty:
        best_row = max_acc.sort_values("f1", ascending=False).iloc[0]
        rows.append({
            "method": f"Legacy {best_row['model']} (max-accuracy plan)",
            "family": "Legacy Reference",
            "f1": pd.to_numeric(best_row.get("f1"), errors="coerce"),
            "precision": pd.to_numeric(best_row.get("precision"), errors="coerce"),
            "recall": pd.to_numeric(best_row.get("recall"), errors="coerce"),
            "far": pd.to_numeric(best_row.get("far"), errors="coerce"),
            "pr_auc": np.nan,
            "accuracy": pd.to_numeric(best_row.get("accuracy"), errors="coerce"),
            "config_label": f"Legacy reference | source={best_row.get('source', 'n/a')}",
            "source_file": str(MAX_ACC_FILE.relative_to(PROJECT_ROOT)),
            "comparison_scope": "Legacy_reference_only",
            "directly_comparable": False,
            "included_in_chart": False,
        })

    opt_final = load_csv(OPT_FINAL_FILE)
    if opt_final is not None and not opt_final.empty:
        best_row = opt_final.sort_values("avg_f1", ascending=False).iloc[0]
        rows.append({
            "method": f"Legacy {best_row['model']} (optimization plan)",
            "family": "Legacy Reference",
            "f1": pd.to_numeric(best_row.get("avg_f1"), errors="coerce"),
            "precision": pd.to_numeric(best_row.get("avg_precision"), errors="coerce"),
            "recall": pd.to_numeric(best_row.get("avg_recall"), errors="coerce"),
            "far": np.nan,
            "pr_auc": np.nan,
            "accuracy": pd.to_numeric(best_row.get("avg_accuracy"), errors="coerce"),
            "config_label": f"Legacy reference | H={best_row.get('H', 'n/a')}",
            "source_file": str(OPT_FINAL_FILE.relative_to(PROJECT_ROOT)),
            "comparison_scope": "Legacy_reference_only",
            "directly_comparable": False,
            "included_in_chart": False,
        })

    return rows


def build_comparison_table():
    context = build_context_row()
    comparison_rule = (
        "Primary comparison uses the definitive Plan 11 common setup at "
        f"H={context['H']} with label={context['label_variant']} "
        f"(min_persistence={context['min_persistence']}) and feature set "
        f"{context['feature_set']}. ML rows come from Phase 2 aggregated fold "
        "results at that shared horizon; trivial and heuristic baselines come "
        "from the definitive horizon-sweep row for the same H. Legacy rows are "
        "kept in the CSV as references only and are excluded from the charts."
    )

    rows = []
    rows.extend(build_baseline_rows(context))
    rows.extend(build_phase2_model_rows(context))
    rows.extend(build_legacy_rows())

    if not rows:
        raise RuntimeError("No comparison rows could be built from available CSV files.")

    df = pd.DataFrame(rows)
    df["comparison_rule"] = comparison_rule
    df["chart_rank"] = np.nan

    plot_df = df[df["included_in_chart"]].copy()
    plot_df = plot_df.sort_values("f1", ascending=False).reset_index(drop=True)
    for idx, method in enumerate(plot_df["method"], start=1):
        df.loc[df["method"] == method, "chart_rank"] = idx

    df["chart_rank"] = pd.to_numeric(df["chart_rank"], errors="coerce")
    df["sort_group"] = df["included_in_chart"].astype(int)
    df = df.sort_values(["sort_group", "chart_rank", "f1"], ascending=[False, True, False]).drop(columns=["sort_group"])
    return df, plot_df, comparison_rule


def make_f1_chart(plot_df, context):
    df = plot_df.sort_values("f1", ascending=True).copy()
    colors = [COLOR_MAP[fam] for fam in df["family"]]

    fig_height = max(5.5, 0.6 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    bars = ax.barh(df["method"], df["f1"], color=colors, edgecolor="black", linewidth=0.6)

    for bar, value in zip(bars, df["f1"]):
        ax.text(
            min(value + 0.01, 0.99),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=10,
        )

    legend_handles = []
    for family in ["Trivial Baseline", "Heuristic Baseline", "ML Model"]:
        if family in set(df["family"]):
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_MAP[family], edgecolor="black", label=family)
            )

    ax.set_xlim(0, min(1.0, max(df["f1"].max() + 0.10, 0.75)))
    ax.set_xlabel("F1 Score")
    ax.set_ylabel("Method")
    ax.set_title(
        "Regime-Shift Prediction: F1-Based Comparison\n"
        f"Definitive Plan 11 Common Setup (H={context['H']}, Set {context['feature_set']})"
    )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(FIG_F1_OUT, dpi=150)
    plt.close(fig)


def make_precision_recall_f1_chart(plot_df, context):
    df = plot_df.sort_values("f1", ascending=True).copy()
    metrics = ["precision", "recall", "f1"]
    labels = ["Precision", "Recall", "F1"]
    colors = ["#66a61e", "#e6ab02", "#1f78b4"]

    y = np.arange(len(df))
    bar_h = 0.22

    fig_height = max(5.5, 0.6 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    for idx, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)
        ax.barh(y + (idx - 1) * bar_h, vals, height=bar_h, color=color, edgecolor="black", linewidth=0.5, label=label)

    ax.set_yticks(y)
    ax.set_yticklabels(df["method"])
    ax.set_xlim(0, min(1.0, max(df[metrics].max().max() + 0.10, 0.75)))
    ax.set_xlabel("Score")
    ax.set_ylabel("Method")
    ax.set_title(
        "Regime-Shift Prediction: Precision, Recall, and F1\n"
        f"Definitive Plan 11 Common Setup (H={context['H']}, Set {context['feature_set']})"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(FIG_PRF_OUT, dpi=150)
    plt.close(fig)


def print_summary(plot_df, comparison_rule):
    best_row = plot_df.sort_values("f1", ascending=False).iloc[0]

    heuristic_df = plot_df[plot_df["family"] == "Heuristic Baseline"].copy()
    trivial_df = plot_df[plot_df["family"] == "Trivial Baseline"].copy()

    best_heuristic = heuristic_df.sort_values("f1", ascending=False).iloc[0] if not heuristic_df.empty else None
    best_trivial = trivial_df.sort_values("f1", ascending=False).iloc[0] if not trivial_df.empty else None

    print("\n=== Regime Classifier F1 Comparison ===")
    print(f"Comparison setup: {comparison_rule}")
    print(f"Highest F1: {best_row['method']} ({best_row['f1']:.3f})")

    if best_heuristic is not None:
        diff_heur = best_row["f1"] - best_heuristic["f1"]
        print(
            f"Improvement vs best heuristic baseline "
            f"({best_heuristic['method']}): {diff_heur:+.3f}"
        )

    if best_trivial is not None:
        diff_triv = best_row["f1"] - best_trivial["f1"]
        print(
            f"Improvement vs best trivial baseline "
            f"({best_trivial['method']}): {diff_triv:+.3f}"
        )

    for trivial_name in ["Always predict 0", "Always predict 1", "Stratified random"]:
        match = trivial_df[trivial_df["method"] == trivial_name]
        if not match.empty:
            diff = best_row["f1"] - float(match.iloc[0]["f1"])
            print(f"Improvement vs {trivial_name}: {diff:+.3f}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    context = build_context_row()
    df, plot_df, comparison_rule = build_comparison_table()

    cols = [
        "chart_rank", "method", "family", "f1", "precision", "recall",
        "far", "pr_auc", "accuracy", "config_label", "comparison_scope",
        "directly_comparable", "included_in_chart", "source_file",
        "comparison_rule",
    ]
    df[cols].to_csv(CSV_OUT, index=False)

    make_f1_chart(plot_df, context)
    make_precision_recall_f1_chart(plot_df, context)
    print_summary(plot_df, comparison_rule)

    print(f"\nSaved CSV: {CSV_OUT}")
    print(f"Saved chart: {FIG_F1_OUT}")
    print(f"Saved chart: {FIG_PRF_OUT}")


if __name__ == "__main__":
    main()
