# experiments/compute_rank_stability.py -- Compute algorithm rank stability (Plot 20)
#
# Loads extended data + fold specs from the saved CSV, evaluates all 48
# algorithms on each fold's test period (no refitting needed), saves
# algorithm_rank_stability.csv and produces plot 20.
#
# Expected runtime: ~2-4 minutes (48 algos x 12 folds x ~250 test days)
#
# Usage:
#   python -m regime_algo_selection.experiments.compute_rank_stability

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from regime_algo_selection.config import RESULTS_DIR, KAPPA
from regime_algo_selection.data.loader_extended import load_data_extended
from regime_algo_selection.data.features import compute_returns
from regime_algo_selection.algorithms.tier1_heuristics import build_tier1_algorithm_space
from regime_algo_selection.evaluation.walk_forward import (
    WalkForwardValidator, _eval_algo_test, _MAX_LOOKBACK
)

FAMILY_COLORS = {
    "EqualWeight":       "#9E9E9E",
    "MinimumVariance":   "#2196F3",
    "RiskParity":        "#4CAF50",
    "MaxDiversification":"#FF9800",
    "Momentum":          "#9C27B0",
    "TrendFollowing":    "#F44336",
    "MeanVariance":      "#795548",
}
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


def _algo_family(name):
    if name == "EqualWeight":          return "EqualWeight"
    if name.startswith("MinVar"):      return "MinimumVariance"
    if name.startswith("RiskParity"):  return "RiskParity"
    if name.startswith("MaxDiv"):      return "MaxDiversification"
    if name.startswith("Momentum"):    return "Momentum"
    if name.startswith("Trend"):       return "TrendFollowing"
    if name.startswith("MeanVar"):     return "MeanVariance"
    return "Other"


def main():
    print("=" * 70)
    print("Computing Algorithm Rank Stability (Analysis 4 / Plot 20)")
    print("=" * 70)

    # ---- Load data --------------------------------------------------------
    print("\nLoading extended data...")
    data = load_data_extended(force_download=False)
    prices  = data["prices"]
    returns = compute_returns(prices)
    print(f"  prices : {prices.shape}  ({prices.index[0].date()} to {prices.index[-1].date()})")
    print(f"  returns: {returns.shape}")

    # ---- Load fold specs from summary CSV ---------------------------------
    summary_path = os.path.join(RESULTS_DIR, "walk_forward_performance.csv")
    summary_df   = pd.read_csv(summary_path, index_col=0)
    print(f"\nLoaded {len(summary_df)} folds from {summary_path}")

    # Reconstruct fold specs
    wfv = WalkForwardValidator(train_years=8, test_years=1, step_years=1,
                               min_test_start="2013-01-01")
    folds_spec = wfv.generate_folds(data_end="2024-12-31")
    assert len(folds_spec) == len(summary_df), "Fold count mismatch!"

    # ---- Build algorithms -------------------------------------------------
    algorithms = build_tier1_algorithm_space()
    algo_names = [a.name for a in algorithms]
    n_folds    = len(folds_spec)

    print(f"\nEvaluating {len(algorithms)} algorithms across {n_folds} folds...")
    print("(test-period net Sharpe, no refitting required)")

    # ---- Evaluate each algo on each fold's test period -------------------
    # rank_matrix[algo_name][fold_id] = net Sharpe score
    score_matrix = {a.name: {} for a in algorithms}

    for fs in folds_spec:
        fold_id     = fs["fold"]
        test_start  = fs["test_start"]
        test_end    = fs["test_end"]

        test_mask  = (prices.index >= test_start) & (prices.index <= test_end)
        test_dates = prices.index[test_mask]
        print(f"  Fold {fold_id:>2}: test {test_start[:4]} ({len(test_dates)} days) ...", end="", flush=True)

        for algo in algorithms:
            score = _eval_algo_test(algo, prices, returns, test_dates, kappa=KAPPA)
            score_matrix[algo.name][fold_id] = score

        # Show best algo this fold
        fold_scores = {a: score_matrix[a][fold_id] for a in algo_names}
        best_name   = max(fold_scores, key=fold_scores.get)
        print(f"  best={best_name} ({fold_scores[best_name]:+.2f})")

    # ---- Build rank DataFrame -------------------------------------------
    fold_ids = [fs["fold"] for fs in folds_spec]
    rank_df  = pd.DataFrame(index=algo_names, columns=fold_ids, dtype=float)

    for fold_id in fold_ids:
        scores = {a: score_matrix[a][fold_id] for a in algo_names}
        sorted_names = sorted(scores, key=scores.get, reverse=True)
        for rank, name in enumerate(sorted_names, start=1):
            rank_df.loc[name, fold_id] = rank

    # Summary stats
    rank_df["avg_rank"]   = rank_df[fold_ids].mean(axis=1)
    rank_df["std_rank"]   = rank_df[fold_ids].std(axis=1)
    rank_df["top5_count"] = (rank_df[fold_ids] <= 5).sum(axis=1)
    ew_ranks = rank_df.loc["EqualWeight", fold_ids]
    rank_df["beats_ew"]   = (rank_df[fold_ids].lt(ew_ranks.values)).sum(axis=1)
    rank_df = rank_df.sort_values("avg_rank")

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "algorithm_rank_stability.csv")
    rank_df.to_csv(csv_path)
    print(f"\nSaved: {csv_path}")

    # Print top-20
    top_n = 20
    print(f"\nTop {top_n} algorithms by average test-period rank:")
    print(f"{'Algorithm':>30}  {'AvgRank':>8}  {'StdRank':>8}  {'Top5':>6}  {'BeatsEW':>8}")
    print("-" * 68)
    for name, row in rank_df.head(top_n).iterrows():
        print(f"{name:>30}  {row['avg_rank']:>8.1f}  {row['std_rank']:>8.1f}  "
              f"{int(row['top5_count']):>6}  {int(row['beats_ew']):>8}")

    # ---- Plot 20 ---------------------------------------------------------
    top_algos = rank_df.head(top_n).index.tolist()
    plot_data = [rank_df.loc[algo, fold_ids].values.astype(float) for algo in top_algos]
    colors    = [FAMILY_COLORS.get(_algo_family(a), "#9E9E9E") for a in top_algos]

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
    ax.set_title(f"Plot 20: Algorithm Rank Stability (top {top_n}, {n_folds} folds, 2013-2024)")
    ax.invert_yaxis()

    seen = {}
    patches = []
    for a in top_algos:
        fam = _algo_family(a)
        if fam not in seen:
            seen[fam] = True
            patches.append(mpatches.Patch(color=FAMILY_COLORS.get(fam, "#9E9E9E"),
                                          label=fam, alpha=0.7))
    ax.legend(handles=patches, fontsize=8, loc="lower right")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "20_algorithm_rank_stability.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ---- Also save the algorithm_stability CSV properly ------------------
    regime_cols = {"Calm": "map_calm", "Normal": "map_normal",
                   "Tense": "map_tense", "Crisis": "map_crisis"}
    from collections import Counter
    stab_rows = []
    for rname, col in regime_cols.items():
        if col not in summary_df.columns:
            continue
        algos = summary_df[col].tolist()
        c = Counter(algos)
        mc, mc_n = c.most_common(1)[0]
        row = {"regime": rname, "most_common_algo": mc,
               "stability_%": round(mc_n / len(algos) * 100, 1),
               "n_unique": len(c)}
        for fi, a in enumerate(algos):
            row[f"fold_{fi+1}"] = a
        stab_rows.append(row)
    stab_df = pd.DataFrame(stab_rows).set_index("regime")
    stab_csv = os.path.join(RESULTS_DIR, "algorithm_stability.csv")
    stab_df.to_csv(stab_csv)
    print(f"Saved: {stab_csv}")

    print("\n" + "=" * 70)
    print("DONE")
    for fname in ["20_algorithm_rank_stability.png",
                  "algorithm_rank_stability.csv",
                  "algorithm_stability.csv"]:
        full = os.path.join(RESULTS_DIR, fname)
        status = "OK" if os.path.exists(full) else "MISSING"
        print(f"  [{status}] {fname}")


if __name__ == "__main__":
    main()
