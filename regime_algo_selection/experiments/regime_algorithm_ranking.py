# experiments/regime_algorithm_ranking.py -- Per-Regime Algorithm Rankings (Plan 4, Step 3)
#
# Uses training-period regime-conditional Sharpe scores from agent.all_scores
# (stored in FoldResult.all_scores). No re-evaluation on test data needed.
#
# Analyses:
#   3.1 Per-regime ranking aggregated over walk-forward folds (training scores)
#   3.2 Per-regime top-5 summary table + plot
#   3.3 Tier 1 vs Tier 2 per-regime comparison + plot
#
# Also provides optional test-period regime-conditional ranking (slower).

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from regime_algo_selection.config import RESULTS_DIR, REGIME_NAMES, N_REGIMES
from regime_algo_selection.algorithms.base import TrainablePortfolioAlgorithm

os.makedirs(RESULTS_DIR, exist_ok=True)

REGIME_COLORS = {
    "Calm":   "#2196F3",
    "Normal": "#4CAF50",
    "Tense":  "#FF9800",
    "Crisis": "#F44336",
}


# ---------------------------------------------------------------------------
# Analysis 3.1: Per-Regime Ranking using training-period scores from all_scores
# ---------------------------------------------------------------------------

def compute_regime_rankings(
    wf_result,
    algorithms: list,
    **kwargs,   # accepts prices/returns/regime_labels for backward compat (ignored)
) -> dict:
    """
    Aggregate per-regime algorithm Sharpe scores across walk-forward folds.

    Uses FoldResult.all_scores which contains training-period regime-conditional
    net Sharpe for every algorithm (computed during agent.fit()).

    Returns
    -------
    dict: {regime_int: pd.DataFrame}
      index=algo_name, columns include mean_sharpe, std_sharpe, n_folds
      plus per-fold scores (fold_1, fold_2, ...)
    """
    # Collect scores: {regime_int: {algo_name: {fold_id: score}}}
    regime_fold_scores = {r: {} for r in range(1, N_REGIMES + 1)}

    for fr in wf_result.folds:
        fold_id = fr.fold_id
        if fr.all_scores is None:
            print(f"  WARNING: Fold {fold_id} has no all_scores (old format). Skipping.")
            continue

        for regime_int in range(1, N_REGIMES + 1):
            scores_for_regime = fr.all_scores.get(regime_int, {})
            for algo in algorithms:
                aname = algo.name
                if aname not in regime_fold_scores[regime_int]:
                    regime_fold_scores[regime_int][aname] = {}
                score = scores_for_regime.get(aname, np.nan)
                # Filter out sentinel values
                if score == -999.0 or score < -900:
                    score = np.nan
                regime_fold_scores[regime_int][aname][fold_id] = score

    # Build summary DataFrames
    regime_ranking_dfs = {}
    for regime_int in range(1, N_REGIMES + 1):
        scores_dict = regime_fold_scores[regime_int]
        df = pd.DataFrame(scores_dict).T   # index=algo_name, cols=fold_ids
        df.columns = [f"fold_{c}" for c in df.columns]
        df["mean_sharpe"] = df.mean(axis=1, skipna=True)
        df["std_sharpe"]  = df.std(axis=1,  skipna=True)
        df["n_folds"]     = df.filter(like="fold_").notna().sum(axis=1)
        df = df.sort_values("mean_sharpe", ascending=False)
        df.index.name = "algo_name"
        regime_ranking_dfs[regime_int] = df

    return regime_ranking_dfs


def add_tier_and_family(df: pd.DataFrame, algorithms: list) -> pd.DataFrame:
    """Add 'family' and 'tier' columns to a ranking DataFrame."""
    algo_meta = {}
    for algo in algorithms:
        tier = 2 if isinstance(algo, TrainablePortfolioAlgorithm) else 1
        algo_meta[algo.name] = {"family": algo.family, "tier": tier}

    df = df.copy()
    df["family"] = df.index.map(lambda n: algo_meta.get(n, {}).get("family", "Unknown"))
    df["tier"]   = df.index.map(lambda n: algo_meta.get(n, {}).get("tier", 1))
    return df


def save_regime_rankings(regime_ranking_dfs: dict, algorithms: list) -> None:
    """Save one CSV per regime with full rankings."""
    name_map = {1: "calm", 2: "normal", 3: "tense", 4: "crisis"}

    for regime_int, df in regime_ranking_dfs.items():
        df_out = add_tier_and_family(df, algorithms)
        df_out["rank"] = range(1, len(df_out) + 1)

        fold_cols = [c for c in df_out.columns if c.startswith("fold_")]
        n_folds_with_data = df_out["n_folds"].clip(lower=1)
        n_top10 = (df_out[fold_cols].rank(ascending=False) <= 10).sum(axis=1)
        df_out["consistent_top10"] = (n_top10 / n_folds_with_data > 0.5)

        path = os.path.join(RESULTS_DIR, f"regime_ranking_{name_map[regime_int]}.csv")
        df_out.to_csv(path)
        print(f"  Saved: {path}", flush=True)


# ---------------------------------------------------------------------------
# Analysis 3.2: Top-5 summary plot
# ---------------------------------------------------------------------------

def plot_regime_top5(regime_ranking_dfs: dict, algorithms: list) -> None:
    """Plot top-5 algorithms per regime as horizontal bar chart."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 7), sharey=False)
    fig.suptitle(
        "Top-5 Algorithms per Market Regime\n"
        "(avg net Sharpe, 12-fold walk-forward training scores)",
        fontsize=13, fontweight="bold",
    )

    summary_rows = []

    for ax, (regime_int, regime_name) in zip(axes, REGIME_NAMES.items()):
        df = add_tier_and_family(regime_ranking_dfs[regime_int], algorithms)
        top5 = df.head(5)

        if top5.empty or top5["mean_sharpe"].isna().all():
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title(f"Regime: {regime_name}", color=REGIME_COLORS[regime_name],
                         fontweight="bold", fontsize=11)
            continue

        colors = ["#EE6677" if t == 2 else "#4477AA" for t in top5["tier"]]
        n = len(top5)
        ax.barh(
            range(n), top5["mean_sharpe"].values,
            xerr=top5["std_sharpe"].fillna(0).values,
            color=colors, edgecolor="white", linewidth=0.5,
            error_kw={"elinewidth": 1, "capsize": 3},
        )
        ax.set_yticks(range(n))
        ax.set_yticklabels(
            [f"{i+1}. {name}" for i, name in enumerate(top5.index)],
            fontsize=8,
        )
        ax.invert_yaxis()
        ax.set_xlabel("Avg Net Sharpe (train)", fontsize=9)
        ax.set_title(f"Regime: {regime_name}", color=REGIME_COLORS[regime_name],
                     fontweight="bold", fontsize=11)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

        for rank, (name, row) in enumerate(top5.iterrows(), 1):
            summary_rows.append({
                "regime": regime_name,
                "rank": rank,
                "algo_name": name,
                "family": row["family"],
                "tier": row["tier"],
                "mean_sharpe": row["mean_sharpe"],
                "std_sharpe": row["std_sharpe"],
            })

    t1_patch = mpatches.Patch(color="#4477AA", label="Tier 1 (Heuristic)")
    t2_patch = mpatches.Patch(color="#EE6677", label="Tier 2 (Linear ML)")
    fig.legend(handles=[t1_patch, t2_patch], loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(RESULTS_DIR, "23_regime_top5_summary.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(RESULTS_DIR, "regime_top5_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}", flush=True)

    print("\nTop-5 per Regime (training-period regime-conditional net Sharpe):", flush=True)
    for rank in range(1, 6):
        row_parts = []
        for regime_name in REGIME_NAMES.values():
            sub = summary_df[(summary_df["regime"] == regime_name) & (summary_df["rank"] == rank)]
            if not sub.empty:
                aname = sub.iloc[0]["algo_name"]
                sh    = sub.iloc[0]["mean_sharpe"]
                row_parts.append(f"{regime_name}: {aname} ({sh:+.2f})")
        print(f"  Rank {rank}: " + " | ".join(row_parts), flush=True)


# ---------------------------------------------------------------------------
# Analysis 3.3: Tier 1 vs Tier 2 per regime
# ---------------------------------------------------------------------------

def plot_tier_comparison_per_regime(regime_ranking_dfs: dict, algorithms: list) -> pd.DataFrame:
    """Compare average rank of Tier 1 vs Tier 2 algorithms per regime."""
    rows = []

    for regime_int, regime_name in REGIME_NAMES.items():
        df = add_tier_and_family(regime_ranking_dfs[regime_int], algorithms)
        df = df.reset_index()
        df["rank"] = range(1, len(df) + 1)

        t1 = df[df["tier"] == 1]
        t2 = df[df["tier"] == 2]

        avg_rank_t1    = t1["rank"].mean() if len(t1) > 0 else np.nan
        avg_rank_t2    = t2["rank"].mean() if len(t2) > 0 else np.nan
        best_t2_name   = t2.iloc[0]["algo_name"]  if len(t2) > 0 else "N/A"
        best_t2_sharpe = t2.iloc[0]["mean_sharpe"] if len(t2) > 0 else np.nan
        t2_in_top10    = (t2["rank"] <= 10).any()  if len(t2) > 0 else False

        rows.append({
            "regime":         regime_name,
            "avg_rank_tier1": avg_rank_t1,
            "avg_rank_tier2": avg_rank_t2,
            "tier2_in_top10": t2_in_top10,
            "best_tier2":     best_t2_name,
            "best_tier2_sharpe": best_t2_sharpe,
            "n_tier1": len(t1),
            "n_tier2": len(t2),
        })

    result_df = pd.DataFrame(rows)

    print("\nTier 1 vs Tier 2 per Regime:", flush=True)
    print(f"  {'Regime':<10} {'AvgRank T1':>12} {'AvgRank T2':>12} "
          f"{'T2 in Top10':>13} {'Best T2':<30} {'T2 Sharpe':>10}", flush=True)
    print("  " + "-" * 85, flush=True)
    for _, row in result_df.iterrows():
        print(
            f"  {row['regime']:<10} {row['avg_rank_tier1']:>12.1f} "
            f"{row['avg_rank_tier2']:>12.1f} "
            f"{'Yes' if row['tier2_in_top10'] else 'No':>13} "
            f"{row['best_tier2']:<30} {row['best_tier2_sharpe']:>10.3f}",
            flush=True,
        )

    csv_path = os.path.join(RESULTS_DIR, "tier1_vs_tier2_per_regime.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}", flush=True)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Tier 1 vs Tier 2: Average Rank per Regime", fontsize=12, fontweight="bold")

    regimes = result_df["regime"].tolist()
    x = np.arange(len(regimes))
    w = 0.35

    ax = axes[0]
    ax.bar(x - w/2, result_df["avg_rank_tier1"], w,
           label="Tier 1 (Heuristics)", color="#4477AA", alpha=0.85)
    ax.bar(x + w/2, result_df["avg_rank_tier2"], w,
           label="Tier 2 (Linear ML)", color="#EE6677", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=10)
    ax.set_ylabel("Average Rank (lower is better)", fontsize=10)
    ax.set_title("Average Algorithm Rank by Tier", fontsize=11)
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    ax2 = axes[1]
    t2_sharpes = result_df["best_tier2_sharpe"].values
    colors = [REGIME_COLORS[r] for r in regimes]
    ax2.bar(x, t2_sharpes, color=colors, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(regimes, fontsize=10)
    ax2.set_ylabel("Best Tier 2 Avg Net Sharpe", fontsize=10)
    ax2.set_title("Best Tier 2 Algorithm Sharpe per Regime", fontsize=11)
    ax2.axhline(0, color="black", linewidth=0.8)
    for xi, (r, v) in enumerate(zip(result_df["best_tier2"], t2_sharpes)):
        ax2.text(xi, v + 0.02 * np.sign(v) if np.isfinite(v) else 0,
                 r, ha="center", va="bottom", fontsize=7, rotation=15)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "24_tier1_vs_tier2_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}", flush=True)

    return result_df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_regime_algorithm_ranking(
    wf_result,
    algorithms: list,
    prices=None,       # kept for backward compat, not used
    returns=None,      # kept for backward compat, not used
    regime_labels=None, # kept for backward compat, not used
    kappa=0.001,       # kept for backward compat, not used
) -> dict:
    """
    Run all regime-algorithm ranking analyses (3.1, 3.2, 3.3).
    Uses training-period regime-conditional Sharpe from FoldResult.all_scores.

    Returns regime_ranking_dfs dict for downstream use.
    """
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 3: PER-REGIME ALGORITHM RANKINGS", flush=True)
    print("=" * 70, flush=True)

    print("\n[3.1] Building per-regime rankings from training scores ...", flush=True)
    regime_ranking_dfs = compute_regime_rankings(wf_result, algorithms)

    print("\n[3.1] Saving per-regime CSV rankings ...", flush=True)
    save_regime_rankings(regime_ranking_dfs, algorithms)

    print("\n[3.2] Plotting top-5 per regime ...", flush=True)
    plot_regime_top5(regime_ranking_dfs, algorithms)

    print("\n[3.3] Tier 1 vs Tier 2 per regime ...", flush=True)
    plot_tier_comparison_per_regime(regime_ranking_dfs, algorithms)

    return regime_ranking_dfs
