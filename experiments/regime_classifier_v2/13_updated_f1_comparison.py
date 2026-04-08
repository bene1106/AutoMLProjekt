"""
13_updated_f1_comparison.py

Reads F1 values directly from feature_extension_results.csv and produces an
updated horizontal bar chart in the style of regime_classifier_f1_comparison.png.

Source data:
  results/regime_classifier_v2/definitive/feature_extension_results.csv

Filter:
  ML models  -- horizon=7, feature_set=I_F+T1
  Baselines  -- horizon=7, feature_set=baseline

Chart style (matches existing regime_classifier_f1_comparison.png):
  - Horizontal bars, sorted descending by F1
  - Grey   (#8c8c8c) = Trivial Baseline
  - Orange (#d95f02) = Heuristic Baseline
  - Blue   (#1f78b4) = ML Model
  - F1 value annotated at the right end of each bar
  - Legend by method family

Output:
  results/regime_classifier_v2/definitive/updated_f1_comparison.png
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR  = PROJECT_ROOT / "results" / "regime_classifier_v2" / "definitive"
CSV_PATH     = RESULTS_DIR / "feature_extension_results.csv"
OUT_PATH     = RESULTS_DIR / "updated_f1_comparison.png"

# ---------------------------------------------------------------------------
# Style constants  (match existing regime_classifier_f1_comparison.png)
# ---------------------------------------------------------------------------
COLOR_MAP = {
    "Trivial Baseline":   "#8c8c8c",
    "Heuristic Baseline": "#d95f02",
    "ML Model":           "#1f78b4",
}
FAMILY_ORDER = {
    "ML Model":           0,
    "Heuristic Baseline": 1,
    "Trivial Baseline":   2,
}

# Best-config metadata embedded in the title
BEST_MODEL      = "XGBoost"
BEST_FEAT_SET   = "I_F+T1"
BEST_HORIZON    = 7


def load_plot_data(csv_path: Path) -> pd.DataFrame:
    """
    Read the CSV and return the rows needed for the chart:
      - H=7, feature_set=I_F+T1, family=ML Model  (3 rows)
      - H=7, feature_set=baseline                  (6 rows)
    Columns retained: method, family, f1.
    """
    df = pd.read_csv(csv_path)

    ml_mask = (
        (df["horizon"] == BEST_HORIZON)
        & (df["feature_set"] == BEST_FEAT_SET)
        & (df["family"] == "ML Model")
    )
    bl_mask = (
        (df["horizon"] == BEST_HORIZON)
        & (df["feature_set"] == "baseline")
    )

    ml_rows = df.loc[ml_mask, ["model", "family", "f1"]].rename(columns={"model": "method"})
    bl_rows = df.loc[bl_mask, ["model", "family", "f1"]].rename(columns={"model": "method"})

    combined = pd.concat([ml_rows, bl_rows], ignore_index=True)
    combined["f1"] = pd.to_numeric(combined["f1"], errors="coerce").fillna(0.0)

    # Sort: primary = f1 descending, secondary = method name ascending (stable tie-break)
    combined = combined.sort_values(
        ["f1", "method"], ascending=[False, True]
    ).reset_index(drop=True)

    return combined


def annotate_barh(ax, bars, values: pd.Series, pad: float = 0.008) -> None:
    """Place F1 value labels to the right of each bar."""
    for bar, value in zip(bars, values):
        x = min(float(value) + pad, 0.985)
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{float(value):.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )


def apply_family_label_colors(ax, families: pd.Series) -> None:
    """Color y-axis tick labels by method family (matches existing chart style)."""
    for tick, family in zip(ax.get_yticklabels(), families):
        tick.set_color(COLOR_MAP.get(family, "black"))
        tick.set_fontweight("semibold")


def make_chart(plot_df: pd.DataFrame) -> None:
    """
    Draw the horizontal F1 bar chart and save to OUT_PATH.

    Layout:
      - Rows ordered bottom-to-top by descending F1 (barh ascending = visual descending)
      - Bar color = family color
      - Annotated F1 values
      - Legend for the three method families
    """
    # barh plots bottom-to-top, so reverse the sorted df so rank-1 appears at top
    df = plot_df.sort_values(["f1", "method"], ascending=[True, True]).copy()

    colors    = [COLOR_MAP[fam] for fam in df["family"]]
    fig_height = max(5.8, 0.64 * len(df) + 1.5)

    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    bars = ax.barh(df["method"], df["f1"], color=colors, edgecolor="black", linewidth=0.6)
    annotate_barh(ax, bars, df["f1"])
    apply_family_label_colors(ax, df["family"])

    # Legend (one patch per family present)
    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=COLOR_MAP[fam], edgecolor="black",
            label=fam,
        )
        for fam in ["Trivial Baseline", "Heuristic Baseline", "ML Model"]
        if fam in df["family"].values
    ]

    xmax = min(1.0, max(float(df["f1"].max()) + 0.12, 0.75))
    ax.set_xlim(0, xmax)
    ax.set_xlabel("F1 Score", fontsize=11)
    ax.set_ylabel("Method", fontsize=11)
    ax.set_title(
        "Regime-Shift Prediction: F1-Based Comparison\n"
        f"Best Configuration: {BEST_MODEL} | Set {BEST_FEAT_SET} | H={BEST_HORIZON}",
        fontsize=12,
        pad=10,
    )
    ax.grid(axis="x", alpha=0.25)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", frameon=True)

    plt.tight_layout()
    os.makedirs(OUT_PATH.parent, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180)
    plt.close(fig)


def main() -> None:
    print(f"Reading : {CSV_PATH}")
    plot_df = load_plot_data(CSV_PATH)

    print("\nData used for chart (H=7 | Set I_F+T1 for ML models | baselines):")
    print(f"  {'Method':<26} {'Family':<22} F1")
    print(f"  {'-'*26} {'-'*22} --------")
    for _, row in plot_df.iterrows():
        print(f"  {row['method']:<26} {row['family']:<22} {row['f1']:.4f}")

    make_chart(plot_df)
    print(f"\nSaved : {OUT_PATH}")


if __name__ == "__main__":
    main()
