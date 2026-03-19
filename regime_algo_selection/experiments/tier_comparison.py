# experiments/tier_comparison.py -- Step 4: Reflex Agent Tier 1 vs Tier 1+2 (Plan 4)
#
# Compares:
#   Pipeline A: Reflex Agent with K=48 (Tier 1 only)   -- uses stored Plan 3 results
#   Pipeline B: Reflex Agent with K=81 (Tier 1 + Tier 2) -- new walk-forward run
#
# Produces:
#   results/tier_comparison.csv
#   results/25_tier_comparison.png
#
# Usage (standalone):
#   cd Implementierung1
#   python -m regime_algo_selection.experiments.tier_comparison

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from regime_algo_selection.config import RESULTS_DIR, REGIME_NAMES, N_REGIMES, KAPPA
from regime_algo_selection.evaluation.walk_forward import WalkForwardValidator

os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Build comparison tables from two WalkForwardResult objects
# ---------------------------------------------------------------------------

def build_tier_comparison_table(
    wf_t1,         # WalkForwardResult: Tier 1 only (K=48)
    wf_t12,        # WalkForwardResult: Tier 1+2 (K=81)
) -> pd.DataFrame:
    """
    Build fold-by-fold head-to-head comparison table.
    """
    rows = []
    t12_folds = {fr.fold_id: fr for fr in wf_t12.folds}

    for fr_t1 in wf_t1.folds:
        fold_id   = fr_t1.fold_id
        test_year = fr_t1.fold_spec["test_start"][:4]
        fr_t12    = t12_folds.get(fold_id)

        ew_sharpe     = fr_t1.metrics_ew["sharpe_ratio"]
        t1_sharpe     = fr_t1.metrics_reflex["sharpe_ratio"]
        t12_sharpe    = fr_t12.metrics_reflex["sharpe_ratio"] if fr_t12 else np.nan
        t1_oracle     = fr_t1.metrics_oracle["sharpe_ratio"]
        t12_oracle    = fr_t12.metrics_oracle["sharpe_ratio"] if fr_t12 else np.nan

        # Did the Tier 1+2 agent pick any Tier 2 algo?
        t2_selected = False
        if fr_t12:
            for algo_name in fr_t12.reflex_mapping.values():
                if any(algo_name.startswith(f) for f in ["Ridge_", "Lasso_", "ElasticNet_"]):
                    t2_selected = True
                    break

        rows.append({
            "fold":          fold_id,
            "test_year":     test_year,
            "ew_sharpe":     ew_sharpe,
            "t1_sharpe":     t1_sharpe,
            "t12_sharpe":    t12_sharpe,
            "t1_oracle":     t1_oracle,
            "t12_oracle":    t12_oracle,
            "t1_vs_ew":      t1_sharpe - ew_sharpe,
            "t12_vs_ew":     t12_sharpe - ew_sharpe if not np.isnan(t12_sharpe) else np.nan,
            "t12_vs_t1":     t12_sharpe - t1_sharpe if not np.isnan(t12_sharpe) else np.nan,
            "t2_selected":   t2_selected,
            # Reflex mappings T1
            "t1_map_calm":   fr_t1.reflex_mapping.get(1, "N/A"),
            "t1_map_normal": fr_t1.reflex_mapping.get(2, "N/A"),
            "t1_map_tense":  fr_t1.reflex_mapping.get(3, "N/A"),
            "t1_map_crisis": fr_t1.reflex_mapping.get(4, "N/A"),
            # Reflex mappings T1+2
            "t12_map_calm":   fr_t12.reflex_mapping.get(1, "N/A") if fr_t12 else "N/A",
            "t12_map_normal": fr_t12.reflex_mapping.get(2, "N/A") if fr_t12 else "N/A",
            "t12_map_tense":  fr_t12.reflex_mapping.get(3, "N/A") if fr_t12 else "N/A",
            "t12_map_crisis": fr_t12.reflex_mapping.get(4, "N/A") if fr_t12 else "N/A",
        })

    df = pd.DataFrame(rows).set_index("fold")
    return df


def print_tier_comparison_table(df: pd.DataFrame) -> None:
    """Pretty-print the head-to-head comparison table."""
    print("\n" + "=" * 90)
    print("HEAD-TO-HEAD: Reflex Agent -- Tier 1 (K=48) vs Tier 1+2 (K=81)")
    print("=" * 90)
    header = (
        f"{'Fold':>5}  {'Year':>5}  {'EW':>8}  {'T1 Reflex':>10}  "
        f"{'T1+2 Reflex':>12}  {'T2 Sel?':>8}  {'T1+2 vs T1':>11}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for fold_id, row in df.iterrows():
        t12_s = f"{row['t12_sharpe']:+.4f}" if not pd.isna(row["t12_sharpe"]) else "  N/A  "
        diff  = f"{row['t12_vs_t1']:+.4f}"  if not pd.isna(row["t12_vs_t1"])  else "  N/A  "
        print(
            f"{fold_id:>5}  {row['test_year']:>5}  "
            f"{row['ew_sharpe']:>8.4f}  {row['t1_sharpe']:>10.4f}  "
            f"{t12_s:>12}  {'Yes' if row['t2_selected'] else 'No':>8}  "
            f"{diff:>11}"
        )
    print(sep)
    avgs = df[["ew_sharpe", "t1_sharpe", "t12_sharpe", "t12_vs_t1"]].mean()
    print(
        f"{'AVG':>5}  {'':>5}  "
        f"{avgs['ew_sharpe']:>8.4f}  {avgs['t1_sharpe']:>10.4f}  "
        f"{avgs['t12_sharpe']:>12.4f}  {'':>8}  "
        f"{avgs['t12_vs_t1']:>11.4f}"
    )

    # How many folds T1+2 wins
    t12_wins = (df["t12_vs_t1"] > 0).sum()
    t1_wins  = (df["t12_vs_t1"] < 0).sum()
    print(f"\nT1+2 beats T1: {t12_wins}/{len(df)} folds | T1 beats T1+2: {t1_wins}/{len(df)} folds")

    # Regime mapping comparison (most common per regime)
    print("\nReflex Mapping Comparison (mode over folds):")
    print(f"  {'Regime':<10} {'Tier1 Pick':<35} {'Tier1+2 Pick':<35} {'Changed?':>8}")
    print("  " + "-" * 90)
    for regime_name, col_t1, col_t12 in [
        ("Calm",   "t1_map_calm",   "t12_map_calm"),
        ("Normal", "t1_map_normal", "t12_map_normal"),
        ("Tense",  "t1_map_tense",  "t12_map_tense"),
        ("Crisis", "t1_map_crisis", "t12_map_crisis"),
    ]:
        mode_t1  = df[col_t1].mode()[0]  if len(df[col_t1].mode())  > 0 else "N/A"
        mode_t12 = df[col_t12].mode()[0] if len(df[col_t12].mode()) > 0 else "N/A"
        changed  = "YES" if mode_t1 != mode_t12 else "no"
        print(f"  {regime_name:<10} {mode_t1:<35} {mode_t12:<35} {changed:>8}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_tier_comparison(df: pd.DataFrame) -> None:
    """Plot Sharpe comparison across folds for T1 vs T1+2."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Reflex Agent: Tier 1 Only vs Tier 1+2 (Sharpe Ratio per Fold)",
                 fontsize=12, fontweight="bold")

    years = df["test_year"].values
    folds = df.index.values
    x = np.arange(len(years))
    w = 0.25

    ax = axes[0]
    ax.bar(x - w, df["ew_sharpe"].values,  w, label="EW",           color="#9E9E9E", alpha=0.85)
    ax.bar(x,     df["t1_sharpe"].values,  w, label="Reflex T1",    color="#4477AA", alpha=0.85)
    ax.bar(x + w, df["t12_sharpe"].values, w, label="Reflex T1+T2", color="#EE6677", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Net Sharpe Ratio", fontsize=10)
    ax.set_title("Sharpe by Fold", fontsize=11)
    ax.legend(fontsize=9)

    # Mark folds where T2 algo was selected
    for i, (fold_id, row) in enumerate(df.iterrows()):
        if row["t2_selected"]:
            ax.annotate("T2", (x[i] + w, row["t12_sharpe"]),
                        ha="center", va="bottom", fontsize=7,
                        color="#EE6677", fontweight="bold")

    ax2 = axes[1]
    diff = df["t12_vs_t1"].values
    colors = ["#228833" if d > 0 else "#AA3333" for d in diff]
    ax2.bar(x, diff, color=colors, alpha=0.85, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(years, rotation=45, fontsize=8)
    ax2.axhline(0, color="black", linewidth=1.0)
    ax2.set_ylabel("Sharpe Difference (T1+2 minus T1)", fontsize=10)
    ax2.set_title("T1+2 vs T1: Per-Fold Improvement", fontsize=11)

    avg_diff = np.nanmean(diff)
    ax2.axhline(avg_diff, color="#AA3333" if avg_diff < 0 else "#228833",
                linestyle="--", linewidth=1.5,
                label=f"Avg: {avg_diff:+.3f}")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "25_tier_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_tier_comparison(wf_t1, wf_t12) -> pd.DataFrame:
    """
    Build, print, and save all tier comparison outputs.

    Parameters
    ----------
    wf_t1  : WalkForwardResult  (Tier 1 only, K=48)
    wf_t12 : WalkForwardResult  (Tier 1+2, K=81)

    Returns
    -------
    pd.DataFrame comparison table
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 4: TIER 1 vs TIER 1+2 REFLEX AGENT COMPARISON")
    print("=" * 70)

    df = build_tier_comparison_table(wf_t1, wf_t12)
    print_tier_comparison_table(df)

    csv_path = os.path.join(RESULTS_DIR, "tier_comparison.csv")
    df.to_csv(csv_path)
    print(f"\n  Saved: {csv_path}")

    plot_tier_comparison(df)

    return df
