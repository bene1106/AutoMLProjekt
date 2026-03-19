"""
03_analyze_results.py -- Regime Classifier v2: Plots & Metrics

Loads saved CSV outputs from results/regime_classifier_v2/ and produces:

  Plot 1 -- Confusion Matrix (aggregated across 12 folds)
            One 2x2 matrix per model.
            Saved as: confusion_matrix_<model>.png

  Plot 2 -- Shift Recall per Fold
            Line plot: Recall on label=1 across 12 folds, one line per model.
            Horizontal line at 0.0 (naive baseline "always predict 0").
            Saved as: shift_recall_per_fold.png

  Plot 3 -- Feature Importance (RF + XGBoost)
            Bar plot: importance averaged across 12 folds, all 10 features.
            Saved as: feature_importance.png

Run after 02_train_evaluate.py:
  python experiments/regime_classifier_v2/03_analyze_results.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend: write PNG without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "regime_classifier_v2")

# Consistent color palette for the three models
MODEL_COLORS  = {
    "LogisticRegression": "#1f77b4",
    "RandomForest":       "#ff7f0e",
    "XGBoost":            "#2ca02c",
}
MODEL_MARKERS = {
    "LogisticRegression": "o",
    "RandomForest":       "s",
    "XGBoost":            "^",
}


# ---------------------------------------------------------------------------
# Plot 1: Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(model_name: str) -> None:
    """
    Plot a 2x2 confusion matrix for one model, aggregated across all 12 folds.
    Reads predictions_<model>.csv which contains y_true and y_pred per fold row.
    """
    safe_name = model_name.replace(" ", "_").lower()
    pred_path = os.path.join(RESULTS_DIR, f"predictions_{safe_name}.csv")

    if not os.path.exists(pred_path):
        print(f"  [SKIP] {model_name}: predictions file not found ({pred_path})")
        return

    pred_df = pd.read_csv(pred_path)
    y_true  = pred_df["y_true"].values.astype(int)
    y_pred  = pred_df["y_pred"].values.astype(int)

    # Build 2x2 confusion matrix manually
    cm = np.zeros((2, 2), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[yt, yp] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_labels = ["No Shift (0)", "Shift (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_yticklabels(tick_labels, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label",      fontsize=10)
    ax.set_title(f"Confusion Matrix: {model_name}\n"
                 f"(Aggregated across 12 Walk-Forward Folds)", fontsize=10)

    # Annotate each cell with its count
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{safe_name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: confusion_matrix_{safe_name}.png")


# ---------------------------------------------------------------------------
# Plot 2: Shift Recall per Fold
# ---------------------------------------------------------------------------

def plot_shift_recall_per_fold(fold_df: pd.DataFrame) -> None:
    """
    Line plot showing Recall on label=1 (shift detection) for each test year,
    one line per model.  Also draws the naive baseline at 0.0.
    """
    models    = [m for m in MODEL_COLORS if m in fold_df["model"].unique()]
    fig, ax   = plt.subplots(figsize=(11, 5))

    for model_name in models:
        sub  = fold_df[fold_df["model"] == model_name].sort_values("test_year")
        ax.plot(
            sub["test_year"], sub["recall_shift"],
            label=model_name,
            color=MODEL_COLORS[model_name],
            marker=MODEL_MARKERS[model_name],
            linewidth=2,
            markersize=6,
        )

    # Naive baselines
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5,
               label="Naive: always predict 0 (recall = 0)")
    ax.axhline(1.0, color="gray",  linestyle=":",  linewidth=1.0,
               label="Perfect recall = 1.0")

    years = sorted(fold_df["test_year"].unique())
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("Test Year", fontsize=11)
    ax.set_ylabel("Recall on Shift (label=1)", fontsize=11)
    ax.set_title("Regime Shift Detection: Recall per Fold\n"
                 "(Walk-Forward 2013-2024, H=10 days, any-formulation)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "shift_recall_per_fold.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: shift_recall_per_fold.png")


# ---------------------------------------------------------------------------
# Plot 3: Feature Importance
# ---------------------------------------------------------------------------

def plot_feature_importance(imp_df: pd.DataFrame) -> None:
    """
    Bar plot of all 10 features, importance averaged across RF and XGBoost
    and across all 12 walk-forward folds.
    Features are sorted descending by mean importance.
    """
    if imp_df.empty:
        print("  [SKIP] Feature importance data not available.")
        return

    # Average importance across both models and all folds
    imp_avg = (
        imp_df.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
    )

    # Per-model averages (for secondary annotation)
    imp_per_model = (
        imp_df.groupby(["model", "feature"])["importance"]
        .mean()
        .unstack("model")
        .reindex(imp_avg.index)  # same sort order as imp_avg
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    x      = np.arange(len(imp_avg))
    mean_v = imp_avg.mean()
    colors = [
        "#1f77b4" if v >= mean_v else "#aec7e8"
        for v in imp_avg.values
    ]

    bars = ax.bar(x, imp_avg.values, color=colors, edgecolor="gray",
                  linewidth=0.6, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(imp_avg.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Average Feature Importance", fontsize=11)
    ax.set_title("Feature Importance (RF + XGBoost, Avg Across 12 Folds)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, zorder=1)
    ax.axhline(mean_v, color="red", linestyle="--", linewidth=1.0,
               label=f"Mean importance = {mean_v:.4f}", zorder=3)
    ax.legend(fontsize=9)

    # Annotate each bar with its value
    for bar, val in zip(bars, imp_avg.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=7.5,
        )

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: feature_importance.png")


# ---------------------------------------------------------------------------
# Supplementary: F1 per fold (useful for thesis)
# ---------------------------------------------------------------------------

def plot_f1_per_fold(fold_df: pd.DataFrame) -> None:
    """
    Supplementary line plot: F1 on label=1 across folds (one line per model).
    """
    models  = [m for m in MODEL_COLORS if m in fold_df["model"].unique()]
    fig, ax = plt.subplots(figsize=(11, 5))

    for model_name in models:
        sub = fold_df[fold_df["model"] == model_name].sort_values("test_year")
        ax.plot(
            sub["test_year"], sub["f1_shift"],
            label=model_name,
            color=MODEL_COLORS[model_name],
            marker=MODEL_MARKERS[model_name],
            linewidth=2, markersize=6,
        )

    years = sorted(fold_df["test_year"].unique())
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Test Year", fontsize=11)
    ax.set_ylabel("F1 on Shift (label=1)", fontsize=11)
    ax.set_title("Regime Shift Detection: F1 per Fold (Walk-Forward 2013-2024)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "shift_f1_per_fold.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: shift_f1_per_fold.png  (supplementary)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Generating plots from saved results in:\n  {RESULTS_DIR}\n")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load fold-level results
    fold_path = os.path.join(RESULTS_DIR, "fold_results.csv")
    if not os.path.exists(fold_path):
        print(f"ERROR: {fold_path} not found.  Run 02_train_evaluate.py first.")
        sys.exit(1)

    fold_df = pd.read_csv(fold_path)
    print(f"Loaded fold_results.csv  : {len(fold_df)} rows | "
          f"models: {fold_df['model'].unique().tolist()}\n")

    # Load feature importances
    imp_path = os.path.join(RESULTS_DIR, "feature_importances.csv")
    imp_df   = pd.read_csv(imp_path) if os.path.exists(imp_path) else pd.DataFrame()

    # --- Plot 1: Confusion matrices ---
    print("Plot 1 -- Confusion matrices ...")
    for model_name in fold_df["model"].unique():
        plot_confusion_matrix(model_name)

    # --- Plot 2: Shift recall per fold ---
    print("\nPlot 2 -- Shift recall per fold ...")
    plot_shift_recall_per_fold(fold_df)

    # --- Plot 3: Feature importance ---
    print("\nPlot 3 -- Feature importance ...")
    plot_feature_importance(imp_df)

    # --- Supplementary: F1 per fold ---
    print("\nSupplementary -- F1 per fold ...")
    plot_f1_per_fold(fold_df)

    # --- Print aggregated metrics table ---
    print("\n" + "=" * 60)
    print("Aggregated Metrics (mean +/- std across 12 folds):")
    print(f"{'Model':<22} {'Recall':>8} {'Precision':>10} {'F1':>8} {'Accuracy':>10}")
    print("-" * 60)
    for model_name in fold_df["model"].unique():
        sub = fold_df[fold_df["model"] == model_name]
        print(f"{model_name:<22} "
              f"{sub['recall_shift'].mean():>6.3f}+/-{sub['recall_shift'].std():.3f}  "
              f"{sub['precision_shift'].mean():>6.3f}+/-{sub['precision_shift'].std():.3f}  "
              f"{sub['f1_shift'].mean():>5.3f}+/-{sub['f1_shift'].std():.3f}  "
              f"{sub['accuracy'].mean():>7.3f}+/-{sub['accuracy'].std():.3f}")

    if not imp_df.empty:
        print("\nTop 5 Features (avg importance, RF + XGBoost):")
        imp_avg = (imp_df.groupby("feature")["importance"].mean()
                   .sort_values(ascending=False))
        for rank, (feat, imp) in enumerate(imp_avg.head(5).items(), 1):
            print(f"  {rank}. {feat:<24} {imp:.4f}")
    print("=" * 60)

    print(f"\nAll plots saved to: {RESULTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
