# experiments/replot_analyses.py -- Fast re-run of analyses & plots using saved CSVs
#
# Usage:
#   python -m regime_algo_selection.experiments.replot_analyses

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

from regime_algo_selection.config import RESULTS_DIR

# --------------------------------------------------------------------------
# Load the summary CSV that was produced by walk_forward_analysis.py
# --------------------------------------------------------------------------

CSV_PATH = os.path.join(RESULTS_DIR, "walk_forward_performance.csv")

PALETTE = {
    "EW":     "#4477AA",
    "Reflex": "#EE6677",
    "Oracle": "#228833",
}
REGIME_COLORS = {
    "Calm":   "#2196F3",
    "Normal": "#4CAF50",
    "Tense":  "#FF9800",
    "Crisis": "#F44336",
}
FAMILY_COLORS = {
    "EqualWeight":       "#9E9E9E",
    "MinimumVariance":   "#2196F3",
    "RiskParity":        "#4CAF50",
    "MaxDiversification":"#FF9800",
    "Momentum":          "#9C27B0",
    "TrendFollowing":    "#F44336",
    "MeanVariance":      "#795548",
}
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def load_summary():
    df = pd.read_csv(CSV_PATH, index_col=0)
    return df


# --------------------------------------------------------------------------
# Plot 17: Sharpe comparison
# --------------------------------------------------------------------------

def plot_17(df):
    years = df["test_year"].astype(str).tolist()
    x = np.arange(len(years))
    w = 0.25

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.bar(x - w, df["ew_sharpe"],     w, label="Equal Weight", color=PALETTE["EW"])
    ax1.bar(x,     df["reflex_sharpe"], w, label="Reflex Agent",  color=PALETTE["Reflex"])
    ax1.bar(x + w, df["oracle_sharpe"], w, label="Oracle Agent",  color=PALETTE["Oracle"])
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Annualised Sharpe Ratio")
    ax1.set_title("Plot 17: Walk-Forward Sharpe Comparison (2013-2024)")
    ax1.legend()
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)

    # Colour tick labels by dominant regime
    fig.canvas.draw()
    for lbl, row in zip(ax1.get_xticklabels(), df.itertuples()):
        lbl.set_color(REGIME_COLORS.get(row.dominant_regime, "black"))

    colors = [PALETTE["Reflex"] if v > 0 else PALETTE["EW"]
              for v in df["reflex_vs_ew"]]
    ax2.bar(x, df["reflex_vs_ew"], color=colors)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Reflex - EW")
    ax2.set_xticks(x)
    ax2.set_xticklabels(years)

    patches = [mpatches.Patch(color=c, label=r) for r, c in REGIME_COLORS.items()]
    ax2.legend(handles=patches, title="Dom. Regime", fontsize=8, loc="upper right", ncol=2)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "17_walk_forward_sharpe_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# --------------------------------------------------------------------------
# Plot 18: Regime-conditional
# --------------------------------------------------------------------------

def plot_18(df):
    grouped = df.groupby("dominant_regime")[["ew_sharpe", "reflex_sharpe"]].agg(["mean", "count"])

    rows_out = []
    for regime in ["Calm", "Normal", "Tense", "Crisis"]:
        if regime not in grouped.index:
            continue
        n = int(grouped.loc[regime, ("ew_sharpe", "count")])
        avg_ew  = grouped.loc[regime, ("ew_sharpe",     "mean")]
        avg_ref = grouped.loc[regime, ("reflex_sharpe", "mean")]
        rows_out.append({"regime": regime, "n_folds": n,
                         "avg_ew": avg_ew, "avg_reflex": avg_ref})
    rows_df = pd.DataFrame(rows_out).set_index("regime")

    x = np.arange(len(rows_df))
    w = 0.3
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, rows_df["avg_ew"],     w, label="Equal Weight", color=PALETTE["EW"])
    ax.bar(x + w/2, rows_df["avg_reflex"], w, label="Reflex Agent",  color=PALETTE["Reflex"])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    for i, (r, row) in enumerate(rows_df.iterrows()):
        ax.text(i, max(row["avg_ew"], row["avg_reflex"]) + 0.02,
                f"n={int(row['n_folds'])}", ha="center", fontsize=8, color="grey")

    ax.set_xticks(x)
    ax.set_xticklabels(rows_df.index)
    ax.set_ylabel("Avg Annualised Sharpe")
    ax.set_title("Plot 18: Regime-Conditional Performance (avg across folds by dominant regime)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "18_regime_conditional_performance.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    print(f"\nRegime-Conditional Performance:")
    print(f"{'Regime':>10}  {'N':>4}  {'Avg EW':>8}  {'Avg Reflex':>11}  {'Wins?':>6}")
    for r, row in rows_df.iterrows():
        wins = "YES" if row["avg_reflex"] > row["avg_ew"] else "NO"
        print(f"{r:>10}  {int(row['n_folds']):>4}  {row['avg_ew']:>8.4f}  "
              f"{row['avg_reflex']:>11.4f}  {wins:>6}")


# --------------------------------------------------------------------------
# Plot 19: Algorithm stability heatmap (from summary CSV)
# --------------------------------------------------------------------------

def _algo_family(name):
    if name == "EqualWeight":          return "EqualWeight"
    if name.startswith("MinVar"):      return "MinimumVariance"
    if name.startswith("RiskParity"):  return "RiskParity"
    if name.startswith("MaxDiv"):      return "MaxDiversification"
    if name.startswith("Momentum"):    return "Momentum"
    if name.startswith("Trend"):       return "TrendFollowing"
    if name.startswith("MeanVar"):     return "MeanVariance"
    return "Other"


def plot_19(df):
    regime_cols = {0: "map_calm", 1: "map_normal", 2: "map_tense", 3: "map_crisis"}
    regime_names = ["Calm", "Normal", "Tense", "Crisis"]
    family_list  = list(FAMILY_COLORS.keys())
    family_to_idx = {f: i for i, f in enumerate(family_list)}

    n_folds   = len(df)
    n_regimes = 4
    matrix    = np.full((n_regimes, n_folds), 0.0)
    label_mat = [[""] * n_folds for _ in range(n_regimes)]
    test_years = df["test_year"].astype(str).tolist()

    for fi, (_, row) in enumerate(df.iterrows()):
        for ri, (rname, col) in enumerate(zip(regime_names, regime_cols.values())):
            algo_name = str(row.get(col, ""))
            fam = _algo_family(algo_name)
            matrix[ri, fi] = family_to_idx.get(fam, 0)
            label_mat[ri][fi] = algo_name[:12]

    from collections import Counter
    print(f"\nAlgorithm Stability:")
    print(f"{'Regime':>8}  {'Most Common':>28}  {'Stability':>10}  {'N Unique':>8}")
    for ri, rname in enumerate(regime_names):
        algos = [label_mat[ri][fi] for fi in range(n_folds)]
        full_algos = [str(df.iloc[fi][list(regime_cols.values())[ri]]) for fi in range(n_folds)]
        c = Counter(full_algos)
        mc, mc_n = c.most_common(1)[0]
        stab = mc_n / n_folds * 100
        print(f"{rname:>8}  {mc:>28}  {stab:>9.1f}%  {len(c):>8}")

    fig, ax = plt.subplots(figsize=(14, 4))
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(family_list))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                   vmin=0, vmax=len(family_list) - 1)
    ax.set_xticks(range(n_folds))
    ax.set_xticklabels(test_years, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_regimes))
    ax.set_yticklabels(regime_names)

    for ri in range(n_regimes):
        for fi in range(n_folds):
            ax.text(fi, ri, label_mat[ri][fi],
                    ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_ticks(range(len(family_list)))
    cbar.set_ticklabels(family_list, fontsize=8)
    ax.set_title("Plot 19: Algorithm Stability (Reflex Mapping per Regime per Fold)")
    ax.set_xlabel("Test Year")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "19_algorithm_stability.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# --------------------------------------------------------------------------
# Plot 20: Algorithm rank stability (from rank CSV)
# --------------------------------------------------------------------------

def plot_20():
    rank_path = os.path.join(RESULTS_DIR, "algorithm_rank_stability.csv")
    if not os.path.exists(rank_path):
        print(f"Rank CSV not found: {rank_path} -- skipping Plot 20")
        return

    rank_df = pd.read_csv(rank_path, index_col=0)
    # Identify fold columns (numeric integer cols)
    fold_cols = [c for c in rank_df.columns if str(c).isdigit()]
    if not fold_cols:
        # Try integer-named columns
        fold_cols = [c for c in rank_df.columns if isinstance(c, int)
                     or (isinstance(c, str) and c.replace('.','').isdigit())]
    n_folds = len(fold_cols)

    if "avg_rank" not in rank_df.columns:
        rank_df["avg_rank"] = rank_df[fold_cols].mean(axis=1)
    rank_df = rank_df.sort_values("avg_rank")

    top_n = min(20, len(rank_df))
    top_algos = rank_df.head(top_n).index.tolist()
    plot_data = [rank_df.loc[algo, fold_cols].values.astype(float) for algo in top_algos]
    colors = [FAMILY_COLORS.get(_algo_family(a), "#9E9E9E") for a in top_algos]

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(plot_data, patch_artist=True, vert=True, notch=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xticks(range(1, len(top_algos) + 1))
    ax.set_xticklabels(top_algos, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Rank (lower = better)")
    ax.set_title(f"Plot 20: Algorithm Rank Stability (top {top_n} by avg rank, {n_folds} folds)")
    ax.invert_yaxis()

    seen = {}
    patches = []
    for algo in top_algos:
        fam = _algo_family(algo)
        if fam not in seen:
            seen[fam] = True
            patches.append(mpatches.Patch(
                color=FAMILY_COLORS.get(fam, "#9E9E9E"), label=fam, alpha=0.7))
    ax.legend(handles=patches, fontsize=8, loc="lower right")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "20_algorithm_rank_stability.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# --------------------------------------------------------------------------
# Plot 21: Oracle Gap over time
# --------------------------------------------------------------------------

def plot_21(df):
    x = np.arange(len(df))
    years = df["test_year"].astype(str).tolist()

    print(f"\nOracle Gap Analysis:")
    print(f"{'Year':>5}  {'OracleGap':>10}  {'Dom.Regime':>12}  {'RegimeAcc':>10}")
    print("-" * 44)
    for _, row in df.iterrows():
        print(f"{row['test_year']:>5}  {row['oracle_gap']:>10.4f}  "
              f"{row['dominant_regime']:>12}  {row['regime_accuracy']:>10.4f}")
    print(f"\nMean Oracle Gap: {df['oracle_gap'].mean():+.4f}")
    print(f"Std Oracle Gap : {df['oracle_gap'].std():.4f}")
    corr = df["oracle_gap"].corr(df["regime_accuracy"])
    print(f"Corr(OracleGap, RegimeAcc) = {corr:.4f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    bar_colors = [REGIME_COLORS.get(r, "grey") for r in df["dominant_regime"]]
    ax1.bar(x, df["oracle_gap"], color=bar_colors, alpha=0.8)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Oracle Gap (Oracle - Reflex Sharpe)")
    ax1.set_title("Plot 21: Oracle Gap Over Time")

    for i, (gap, acc) in enumerate(zip(df["oracle_gap"], df["regime_accuracy"])):
        offset = 0.01 if gap >= 0 else -0.04
        ax1.text(i, gap + offset, f"{acc:.2f}", ha="center", fontsize=7.5, color="black")

    ax2.plot(x, df["regime_accuracy"], marker="o", color="#555555", linewidth=1.5)
    ax2.axhline(df["regime_accuracy"].mean(), color="grey", linestyle="--", linewidth=1)
    ax2.set_ylabel("Regime Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels(years)
    ax2.set_xlabel("Test Year")

    patches = [mpatches.Patch(color=c, label=r) for r, c in REGIME_COLORS.items()]
    ax1.legend(handles=patches, title="Dom. Regime", fontsize=8, ncol=2)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "21_oracle_gap_over_time.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# --------------------------------------------------------------------------
# Plot 22: Plan 2 vs Plan 3
# --------------------------------------------------------------------------

def plot_22(df):
    plan2 = {
        "EW Sharpe":        0.37,
        "Reflex Sharpe":   -0.29,
        "Oracle Sharpe":   -0.30,
        "Oracle Gap":      -0.009,
        "Reflex vs EW Gap":-0.66,
    }
    plan3 = {
        "EW Sharpe":        df["ew_sharpe"].mean(),
        "Reflex Sharpe":    df["reflex_sharpe"].mean(),
        "Oracle Sharpe":    df["oracle_sharpe"].mean(),
        "Oracle Gap":       df["oracle_gap"].mean(),
        "Reflex vs EW Gap": df["reflex_vs_ew"].mean(),
    }
    plan3_std = {
        "EW Sharpe":        df["ew_sharpe"].std(),
        "Reflex Sharpe":    df["reflex_sharpe"].std(),
        "Oracle Sharpe":    df["oracle_sharpe"].std(),
        "Oracle Gap":       df["oracle_gap"].std(),
        "Reflex vs EW Gap": df["reflex_vs_ew"].std(),
    }

    print(f"\nPlan 2 vs Plan 3 Comparison:")
    print(f"{'Metric':>22}  {'Plan2 (single)':>16}  {'Plan3 (WF avg)':>16}  {'Plan3 STD':>10}")
    print("-" * 72)
    for k in plan2:
        print(f"{k:>22}  {plan2[k]:>16.4f}  {plan3[k]:>16.4f}  {plan3_std[k]:>10.4f}")

    wins = (df["reflex_sharpe"] > df["ew_sharpe"]).sum()
    n = len(df)
    print(f"\nReflex beats EW: {wins}/{n} folds")

    labels = list(plan2.keys())
    p2_vals = [plan2[k] for k in labels]
    p3_vals = [plan3[k] for k in labels]
    p3_errs = [plan3_std[k] for k in labels]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, p2_vals, w, label="Plan 2 (single 2021-2024)", color="#5A7AB5", alpha=0.85)
    ax.bar(x + w/2, p3_vals, w, label="Plan 3 (walk-forward avg)", color="#D45B5B", alpha=0.85,
           yerr=p3_errs, capsize=4, error_kw={"linewidth": 1.2})
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Plot 22: Plan 2 vs Plan 3 -- Single Window vs Walk-Forward Average")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "22_plan2_vs_plan3.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    comp_df = pd.DataFrame({"Plan2_single": plan2, "Plan3_WF_avg": plan3,
                             "Plan3_WF_std": plan3_std})
    comp_df.to_csv(os.path.join(RESULTS_DIR, "plan2_vs_plan3_comparison.csv"))
    print(f"Saved: {os.path.join(RESULTS_DIR, 'plan2_vs_plan3_comparison.csv')}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PLAN 3: Re-plotting analyses from saved CSVs")
    print("=" * 70)

    df = load_summary()
    print(f"Loaded summary with {len(df)} folds from {CSV_PATH}")

    print("\n--- Plot 17: Sharpe Comparison ---")
    plot_17(df)

    print("\n--- Analysis 2 / Plot 18: Regime-Conditional ---")
    plot_18(df)

    print("\n--- Analysis 3 / Plot 19: Algorithm Stability ---")
    plot_19(df)

    print("\n--- Analysis 4 / Plot 20: Algorithm Rank Stability ---")
    plot_20()

    print("\n--- Analysis 5 / Plot 21: Oracle Gap ---")
    plot_21(df)

    print("\n--- Step 5 / Plot 22: Plan 2 vs Plan 3 ---")
    plot_22(df)

    print("\n" + "=" * 70)
    print("RE-PLOT COMPLETE -- checking output files:")
    print("=" * 70)
    for fname in [
        "17_walk_forward_sharpe_comparison.png",
        "18_regime_conditional_performance.png",
        "19_algorithm_stability.png",
        "20_algorithm_rank_stability.png",
        "21_oracle_gap_over_time.png",
        "22_plan2_vs_plan3.png",
        "plan2_vs_plan3_comparison.csv",
    ]:
        full = os.path.join(RESULTS_DIR, fname)
        status = "OK" if os.path.exists(full) else "MISSING"
        print(f"  [{status}] {fname}")


if __name__ == "__main__":
    main()
