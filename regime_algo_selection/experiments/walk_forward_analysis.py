# experiments/walk_forward_analysis.py -- Plan 3: Walk-Forward Analysis
#
# Steps executed:
#   1. Load extended data (2000-2024)
#   2. Verify all 48 algorithms work for N=2 (SPY+TLT subset)
#   3. Run 12-fold walk-forward validation
#   4. Analyses 1-5 with plots 17-22
#   5. Plan 2 vs Plan 3 comparison table
#
# Usage:
#   cd Implementierung1
#   python -m regime_algo_selection.experiments.walk_forward_analysis

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ---- project imports -------------------------------------------------------
from regime_algo_selection.config import (
    RESULTS_DIR, KAPPA, REGIME_NAMES, N_REGIMES,
)
from regime_algo_selection.data.loader_extended import load_data_extended
from regime_algo_selection.data.features import (
    compute_returns, compute_vix_features,
)
from regime_algo_selection.regimes.ground_truth import compute_regime_labels
from regime_algo_selection.algorithms.tier1_heuristics import (
    build_tier1_algorithm_space,
)
from regime_algo_selection.evaluation.walk_forward import (
    WalkForwardValidator, WalkForwardResult,
)

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})
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


# ===========================================================================
# Step 1: Load extended data
# ===========================================================================

def load_all_data():
    print("\n" + "=" * 70)
    print("STEP 1: Loading extended data")
    print("=" * 70)

    data = load_data_extended(force_download=False)
    prices = data["prices"]
    vix    = data["vix"]

    # Forward returns (5-asset)
    returns = compute_returns(prices)

    # VIX features from the full extended VIX series
    vix_features = compute_vix_features(vix)

    # Regime labels from the full extended VIX series
    regime_labels = compute_regime_labels(vix)

    # Align vix_features and regime_labels to same index
    common_vix = vix_features.index.intersection(regime_labels.index)
    vix_features  = vix_features.loc[common_vix]
    regime_labels = regime_labels.loc[common_vix]

    print(f"\nData summary:")
    print(f"  prices       : {prices.shape}  ({prices.index[0].date()} to {prices.index[-1].date()})")
    print(f"  returns      : {returns.shape}")
    print(f"  vix_features : {vix_features.shape}  ({vix_features.index[0].date()} to ...)")
    print(f"  regime_labels: {regime_labels.shape}  dist={dict(regime_labels.value_counts().sort_index())}")

    return prices, vix, returns, vix_features, regime_labels


# ===========================================================================
# Step 2: Verify algorithms work with N=2
# ===========================================================================

def verify_n2(algorithms, prices):
    print("\n" + "=" * 70)
    print("STEP 2: Verifying all algorithms work for N=2 (SPY+TLT)")
    print("=" * 70)

    # Use SPY+TLT from a 2-year window
    spy_tlt = prices[["SPY", "TLT"]].copy()
    # Pick a stable window in the middle of the data
    mask = (spy_tlt.index >= "2010-01-01") & (spy_tlt.index <= "2011-12-31")
    prices_2a = spy_tlt.loc[mask]

    if len(prices_2a) < 300:
        print(f"  WARNING: Only {len(prices_2a)} days for N=2 test -- results may be noisy")

    fail_count = 0
    for algo in algorithms:
        try:
            w = algo.compute_weights(prices_2a)
            assert len(w) == 2, f"Expected 2 weights, got {len(w)}"
            assert abs(w.sum() - 1.0) < 1e-6, f"Weights do not sum to 1: {w.sum()}"
            assert all(w >= -1e-9), f"Negative weights: {w}"
        except Exception as e:
            print(f"  FAIL {algo.name}: {e}")
            fail_count += 1

    if fail_count == 0:
        print(f"  All {len(algorithms)} algorithms produce valid 2-asset weight vectors.")
    else:
        print(f"  {fail_count}/{len(algorithms)} algorithms FAILED N=2 test.")

    return fail_count == 0


# ===========================================================================
# Step 3: Walk-Forward Validation
# ===========================================================================

def run_walk_forward(prices, vix, returns, vix_features, regime_labels, algorithms):
    print("\n" + "=" * 70)
    print("STEP 3: Walk-Forward Validation (12 folds)")
    print("=" * 70)

    wfv = WalkForwardValidator(
        train_years=8,
        test_years=1,
        step_years=1,
        min_test_start="2013-01-01",
    )

    # Preview folds
    folds_spec = wfv.generate_folds(data_end="2024-12-31")
    print(f"\nGenerated {len(folds_spec)} folds:")
    for fs in folds_spec:
        print(f"  Fold {fs['fold']:>2}: train {fs['train_start'][:4]}-{fs['train_end'][:4]}, "
              f"test {fs['test_start'][:4]}")

    result = wfv.run_all(
        prices=prices,
        vix=vix,
        returns=returns,
        vix_features=vix_features,
        regime_labels=regime_labels,
        algorithms=algorithms,
        kappa=KAPPA,
        data_end="2024-12-31",
    )

    # Save summary CSV
    csv_path = os.path.join(RESULTS_DIR, "walk_forward_performance.csv")
    result.summary_df.to_csv(csv_path)
    print(f"\nSaved: {csv_path}")

    return result


# ===========================================================================
# Analysis helpers
# ===========================================================================

def _algo_family(name: str) -> str:
    if name == "EqualWeight":                return "EqualWeight"
    if name.startswith("MinVar"):            return "MinimumVariance"
    if name.startswith("RiskParity"):        return "RiskParity"
    if name.startswith("MaxDiv"):            return "MaxDiversification"
    if name.startswith("Momentum"):          return "Momentum"
    if name.startswith("Trend"):             return "TrendFollowing"
    if name.startswith("MeanVar"):           return "MeanVariance"
    return "Other"


# ===========================================================================
# Analysis 1: Per-Fold Performance Comparison (Plot 17)
# ===========================================================================

def analysis_1_performance_comparison(wf_result: WalkForwardResult):
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Per-Fold Performance Comparison")
    print("=" * 70)

    df = wf_result.summary_df.copy()

    # Print table
    print(f"\n{'Fold':>4}  {'Year':>5}  {'Dom.Reg':>8}  "
          f"{'EW':>8}  {'Reflex':>8}  {'Oracle':>8}  "
          f"{'OracGap':>8}  {'Ref-EW':>8}  {'BestAlgo':>22}")
    print("-" * 100)

    for fold_id, row in df.iterrows():
        # Best algo in test period
        fold_r = wf_result.folds[fold_id - 1]
        best_name, best_score = max(fold_r.algo_scores.items(), key=lambda x: x[1])
        print(
            f"{fold_id:>4}  {row['test_year']:>5}  {row['dominant_regime']:>8}  "
            f"{row['ew_sharpe']:>8.4f}  {row['reflex_sharpe']:>8.4f}  "
            f"{row['oracle_sharpe']:>8.4f}  {row['oracle_gap']:>8.4f}  "
            f"{row['reflex_vs_ew']:>8.4f}  {best_name:>22} ({best_score:.2f})"
        )

    print("-" * 100)
    cols = ["ew_sharpe", "reflex_sharpe", "oracle_sharpe", "oracle_gap", "reflex_vs_ew"]
    avgs = df[cols].mean()
    stds = df[cols].std()
    print(f"{'AVG':>4}  {'':>5}  {'':>8}  "
          f"{avgs['ew_sharpe']:>8.4f}  {avgs['reflex_sharpe']:>8.4f}  "
          f"{avgs['oracle_sharpe']:>8.4f}  {avgs['oracle_gap']:>8.4f}  "
          f"{avgs['reflex_vs_ew']:>8.4f}")
    print(f"{'STD':>4}  {'':>5}  {'':>8}  "
          f"{stds['ew_sharpe']:>8.4f}  {stds['reflex_sharpe']:>8.4f}  "
          f"{stds['oracle_sharpe']:>8.4f}  {stds['oracle_gap']:>8.4f}  "
          f"{stds['reflex_vs_ew']:>8.4f}")

    wins = (df["reflex_sharpe"] > df["ew_sharpe"]).sum()
    print(f"\nReflex beats EW in {wins}/{len(df)} folds")

    # ---- Plot 17 ----------------------------------------------------------
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

    # Colour x-labels by dominant regime
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    # Colour tick labels by dominant regime (draw first so labels exist)
    fig.canvas.draw()
    for lbl, row in zip(ax1.get_xticklabels(), df.itertuples()):
        lbl.set_color(REGIME_COLORS.get(row.dominant_regime, "black"))

    # Bottom panel: reflex-vs-EW
    colors = [PALETTE["Reflex"] if v > 0 else PALETTE["EW"]
              for v in df["reflex_vs_ew"]]
    ax2.bar(x, df["reflex_vs_ew"], color=colors)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Reflex - EW")
    ax2.set_xticks(x)
    ax2.set_xticklabels(years)

    # Regime legend
    patches = [mpatches.Patch(color=c, label=r) for r, c in REGIME_COLORS.items()]
    ax2.legend(handles=patches, title="Dom. Regime", fontsize=8,
               loc="upper right", ncol=2)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "17_walk_forward_sharpe_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ===========================================================================
# Analysis 2: Regime-Conditional Performance (Plot 18)
# ===========================================================================

def analysis_2_regime_conditional(wf_result: WalkForwardResult):
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Regime-Conditional Performance")
    print("=" * 70)

    df = wf_result.summary_df.copy()

    grouped = df.groupby("dominant_regime")[
        ["ew_sharpe", "reflex_sharpe", "oracle_sharpe"]
    ].agg(["mean", "count"])

    print(f"\n{'Dominant Regime':>16}  {'#Folds':>7}  {'Avg EW':>8}  "
          f"{'Avg Reflex':>11}  {'Reflex Wins?':>12}")
    print("-" * 65)

    rows_out = []
    for regime in ["Calm", "Normal", "Tense", "Crisis"]:
        if regime not in grouped.index:
            continue
        n       = int(grouped.loc[regime, ("ew_sharpe", "count")])
        avg_ew  = grouped.loc[regime, ("ew_sharpe",     "mean")]
        avg_ref = grouped.loc[regime, ("reflex_sharpe", "mean")]
        wins    = "YES" if avg_ref > avg_ew else "NO"
        print(f"{regime:>16}  {n:>7}  {avg_ew:>8.4f}  {avg_ref:>11.4f}  {wins:>12}")
        rows_out.append({"regime": regime, "n_folds": n,
                         "avg_ew": avg_ew, "avg_reflex": avg_ref})

    rows_df = pd.DataFrame(rows_out).set_index("regime")

    # ---- Plot 18 ----------------------------------------------------------
    regimes_present = rows_df.index.tolist()
    x  = np.arange(len(regimes_present))
    w  = 0.3

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, rows_df["avg_ew"],     w, label="Equal Weight", color=PALETTE["EW"])
    ax.bar(x + w/2, rows_df["avg_reflex"], w, label="Reflex Agent",  color=PALETTE["Reflex"])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    for i, (r, row) in enumerate(rows_df.iterrows()):
        label = f"n={int(row['n_folds'])}"
        ax.text(i, max(row["avg_ew"], row["avg_reflex"]) + 0.02, label,
                ha="center", fontsize=8, color="grey")

    ax.set_xticks(x)
    ax.set_xticklabels(regimes_present)
    ax.set_ylabel("Avg Annualised Sharpe")
    ax.set_title("Plot 18: Regime-Conditional Performance (avg across folds by dominant regime)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "18_regime_conditional_performance.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ===========================================================================
# Analysis 3: Algorithm Stability (Plot 19)
# ===========================================================================

def analysis_3_algorithm_stability(wf_result: WalkForwardResult, algorithms):
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Algorithm Stability Across Folds")
    print("=" * 70)

    df = wf_result.summary_df

    regime_cols = {"Calm": "map_calm", "Normal": "map_normal",
                   "Tense": "map_tense", "Crisis": "map_crisis"}
    regime_ids  = {1: "Calm", 2: "Normal", 3: "Tense", 4: "Crisis"}

    # Build stability matrix
    stability_rows = []
    print(f"\n{'Regime':>8}  {'Most Common Algo':>28}  {'Stability':>10}  {'Unique Algos':>12}")
    print("-" * 66)

    for rid, rname in regime_ids.items():
        col = regime_cols.get(rname)
        if col not in df.columns:
            continue
        algos_per_fold = df[col].tolist()
        from collections import Counter
        counts = Counter(algos_per_fold)
        most_common, mc_count = counts.most_common(1)[0]
        stability = mc_count / len(algos_per_fold) * 100
        n_unique  = len(counts)
        print(f"{rname:>8}  {most_common:>28}  {stability:>9.1f}%  {n_unique:>12}")
        stability_rows.append({
            "regime": rname, "most_common": most_common,
            "stability_%": stability, "n_unique": n_unique,
            "algos": algos_per_fold,
        })

    # Build heatmap matrix: rows=regimes, cols=folds, value=family index
    family_list = list(FAMILY_COLORS.keys())
    family_to_idx = {f: i for i, f in enumerate(family_list)}

    n_folds   = len(wf_result.folds)
    n_regimes = 4
    matrix    = np.full((n_regimes, n_folds), np.nan)
    label_mat = [[""] * n_folds for _ in range(n_regimes)]
    test_years = df["test_year"].tolist()

    for fi, fold in enumerate(wf_result.folds):
        for ri, (rid, rname) in enumerate(regime_ids.items()):
            algo_name = fold.reflex_mapping.get(rid, "")
            fam = _algo_family(algo_name)
            matrix[ri, fi] = family_to_idx.get(fam, -1)
            # Shorten label
            short = algo_name.replace("MinimumVariance", "MinVar")
            label_mat[ri][fi] = algo_name[:12]

    # ---- Plot 19 ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 4))
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(family_list))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                   vmin=0, vmax=len(family_list) - 1)

    ax.set_xticks(range(n_folds))
    ax.set_xticklabels(test_years, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_regimes))
    ax.set_yticklabels([regime_ids[i + 1] for i in range(n_regimes)])

    # Add algo name as text
    for ri in range(n_regimes):
        for fi in range(n_folds):
            ax.text(fi, ri, label_mat[ri][fi],
                    ha="center", va="center", fontsize=6.5, color="white",
                    fontweight="bold")

    # Colorbar legend
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

    # Save stability CSV
    stab_rows = []
    for row in stability_rows:
        r = {"regime": row["regime"], "most_common_algo": row["most_common"],
             "stability_%": row["stability_%"], "n_unique": row["n_unique"]}
        for fi, a in enumerate(row["algos"]):
            r[f"fold_{fi+1}"] = a
        stab_rows.append(r)
    stab_df = pd.DataFrame(stab_rows).set_index("regime")
    path_csv = os.path.join(RESULTS_DIR, "algorithm_stability.csv")
    stab_df.to_csv(path_csv)
    print(f"Saved: {path_csv}")


# ===========================================================================
# Analysis 4: Algorithm Rank Stability (Plot 20)
# ===========================================================================

def analysis_4_rank_stability(wf_result: WalkForwardResult, top_n: int = 20):
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Algorithm Rank Stability")
    print("=" * 70)

    n_folds = len(wf_result.folds)

    # Build rank matrix: algo x fold
    all_algo_names = sorted(wf_result.folds[0].algo_scores.keys())
    rank_df = pd.DataFrame(index=all_algo_names, columns=range(1, n_folds + 1), dtype=float)

    ew_sharpes = []
    for fold in wf_result.folds:
        scores = fold.algo_scores
        sorted_names = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        for rank, name in enumerate(sorted_names, start=1):
            rank_df.loc[name, fold.fold_id] = rank
        ew_sharpes.append(scores.get("EqualWeight", -999))

    # Summary stats
    rank_df["avg_rank"]    = rank_df.mean(axis=1)
    rank_df["std_rank"]    = rank_df.std(axis=1)
    rank_df["top5_count"]  = (rank_df.iloc[:, :n_folds] <= 5).sum(axis=1)
    rank_df["beats_ew"]    = sum(
        (rank_df[fold.fold_id] < rank_df.loc["EqualWeight", fold.fold_id]).astype(int)
        for fold in wf_result.folds
    )

    rank_df = rank_df.sort_values("avg_rank")

    print(f"\n{'Algorithm':>30}  {'AvgRank':>8}  {'StdRank':>8}  "
          f"{'Top5':>6}  {'BeatsEW':>8}")
    print("-" * 68)
    for name, row in rank_df.head(top_n).iterrows():
        print(f"{name:>30}  {row['avg_rank']:>8.1f}  {row['std_rank']:>8.1f}  "
              f"{int(row['top5_count']):>6}  {int(row['beats_ew']):>8}")

    # Save CSV
    rank_df.to_csv(os.path.join(RESULTS_DIR, "algorithm_rank_stability.csv"))
    print(f"Saved: {os.path.join(RESULTS_DIR, 'algorithm_rank_stability.csv')}")

    # ---- Plot 20 ----------------------------------------------------------
    # Box plot of ranks for top-N algorithms
    top_algos = rank_df.head(top_n).index.tolist()
    fold_cols  = list(range(1, n_folds + 1))
    plot_data  = [rank_df.loc[algo, fold_cols].values.astype(float) for algo in top_algos]
    families   = [_algo_family(a) for a in top_algos]
    colors     = [FAMILY_COLORS.get(f, "#9E9E9E") for f in families]

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
    ax.set_title(f"Plot 20: Algorithm Rank Stability (top {top_n} by avg rank, across {n_folds} folds)")
    ax.invert_yaxis()

    # Legend
    seen = {}
    patches = []
    for algo, fam in zip(top_algos, families):
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

    return rank_df


# ===========================================================================
# Analysis 5: Oracle Gap Over Time (Plot 21)
# ===========================================================================

def analysis_5_oracle_gap(wf_result: WalkForwardResult):
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Oracle Gap Over Time")
    print("=" * 70)

    df = wf_result.summary_df.copy()

    print(f"\n{'Year':>5}  {'OracleGap':>10}  {'Dom.Regime':>12}  {'RegimeAcc':>10}")
    print("-" * 44)
    for _, row in df.iterrows():
        print(f"{row['test_year']:>5}  {row['oracle_gap']:>10.4f}  "
              f"{row['dominant_regime']:>12}  {row['regime_accuracy']:>10.4f}")

    print(f"\nMean Oracle Gap : {df['oracle_gap'].mean():+.4f}")
    print(f"Std Oracle Gap  : {df['oracle_gap'].std():.4f}")

    # Correlation oracle gap vs regime accuracy
    corr = df["oracle_gap"].corr(df["regime_accuracy"])
    print(f"Corr(OracleGap, RegimeAcc) = {corr:.4f}")

    # ---- Plot 21 ----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    x = np.arange(len(df))
    years = df["test_year"].tolist()

    # Top: Oracle Gap bar chart coloured by dominant regime
    bar_colors = [REGIME_COLORS.get(r, "grey") for r in df["dominant_regime"]]
    bars = ax1.bar(x, df["oracle_gap"], color=bar_colors, alpha=0.8)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Oracle Gap (Oracle - Reflex Sharpe)")
    ax1.set_title("Plot 21: Oracle Gap Over Time")

    # Annotate regime accuracy
    for i, (gap, acc) in enumerate(zip(df["oracle_gap"], df["regime_accuracy"])):
        ax1.text(i, gap + (0.01 if gap >= 0 else -0.04),
                 f"{acc:.2f}", ha="center", fontsize=7.5, color="black")

    # Bottom: Regime accuracy
    ax2.plot(x, df["regime_accuracy"], marker="o", color="#555555", linewidth=1.5)
    ax2.axhline(df["regime_accuracy"].mean(), color="grey", linestyle="--", linewidth=1)
    ax2.set_ylabel("Regime Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels(years)
    ax2.set_xlabel("Test Year")

    # Regime legend
    patches = [mpatches.Patch(color=c, label=r) for r, c in REGIME_COLORS.items()]
    ax1.legend(handles=patches, title="Dom. Regime", fontsize=8, ncol=2)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "21_oracle_gap_over_time.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ===========================================================================
# Step 5: Plan 2 vs Plan 3 comparison (Plot 22)
# ===========================================================================

def analysis_plan2_vs_plan3(wf_result: WalkForwardResult):
    print("\n" + "=" * 70)
    print("STEP 5: Plan 2 vs Plan 3 Comparison")
    print("=" * 70)

    df = wf_result.summary_df.copy()

    # Plan 2 hard-coded results (single 2021-2024 test window, net-Sharpe fit)
    plan2 = {
        "Regime Classifier Accuracy": 0.8455,
        "EW Sharpe":                  0.37,
        "Reflex Sharpe":             -0.29,
        "Oracle Sharpe":             -0.30,
        "Oracle Gap":                -0.009,
        "Reflex vs EW Gap":          -0.66,
    }

    # Plan 3 aggregated over all walk-forward folds
    plan3 = {
        "Regime Classifier Accuracy": df["regime_accuracy"].mean(),
        "EW Sharpe":                  df["ew_sharpe"].mean(),
        "Reflex Sharpe":              df["reflex_sharpe"].mean(),
        "Oracle Sharpe":              df["oracle_sharpe"].mean(),
        "Oracle Gap":                 df["oracle_gap"].mean(),
        "Reflex vs EW Gap":           df["reflex_vs_ew"].mean(),
    }

    print(f"\n{'Metric':>35}  {'Plan2 (single)':>16}  {'Plan3 (WF avg)':>16}")
    print("-" * 72)
    for k in plan2:
        p2 = plan2[k]
        p3 = plan3[k]
        print(f"{k:>35}  {p2:>16.4f}  {p3:>16.4f}")

    # Also report std across folds for Plan 3
    plan3_std = {
        "Regime Classifier Accuracy": df["regime_accuracy"].std(),
        "EW Sharpe":                  df["ew_sharpe"].std(),
        "Reflex Sharpe":              df["reflex_sharpe"].std(),
        "Oracle Sharpe":              df["oracle_sharpe"].std(),
        "Oracle Gap":                 df["oracle_gap"].std(),
        "Reflex vs EW Gap":           df["reflex_vs_ew"].std(),
    }
    print(f"\nPlan 3 standard deviations across folds:")
    for k, v in plan3_std.items():
        print(f"  {k:>35}: {v:.4f}")

    # Win/loss stats
    n_folds = len(df)
    wins = (df["reflex_sharpe"] > df["ew_sharpe"]).sum()
    print(f"\nReflex beats EW: {wins}/{n_folds} folds")
    print(f"Reflex beats Oracle: {(df['reflex_sharpe'] > df['oracle_sharpe']).sum()}/{n_folds} folds")

    # Save CSV
    comp_df = pd.DataFrame({"Plan2_single": plan2, "Plan3_WF_avg": plan3,
                             "Plan3_WF_std": plan3_std})
    path_csv = os.path.join(RESULTS_DIR, "plan2_vs_plan3_comparison.csv")
    comp_df.to_csv(path_csv)
    print(f"Saved: {path_csv}")

    # ---- Plot 22 ----------------------------------------------------------
    metrics_short = {
        "EW Sharpe":        ("EW Sharpe", "ew_sharpe"),
        "Reflex Sharpe":    ("Reflex Sharpe", "reflex_sharpe"),
        "Oracle Sharpe":    ("Oracle Sharpe", "oracle_sharpe"),
        "Oracle Gap":       ("Oracle Gap", "oracle_gap"),
        "Reflex vs EW Gap": ("Ref-EW Gap", "reflex_vs_ew"),
    }
    labels_short = [v[0] for v in metrics_short.values()]
    p2_vals = [plan2[k] for k in metrics_short]
    p3_vals = [plan3[k] for k in metrics_short]
    p3_errs = [plan3_std[k] for k in metrics_short]

    x = np.arange(len(labels_short))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, p2_vals, w, label="Plan 2 (single 2021-2024)", color="#5A7AB5", alpha=0.85)
    ax.bar(x + w / 2, p3_vals, w, label="Plan 3 (walk-forward avg)", color="#D45B5B", alpha=0.85,
           yerr=p3_errs, capsize=4, error_kw={"linewidth": 1.2})
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, rotation=20, ha="right")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Plot 22: Plan 2 vs Plan 3 -- Single Window vs Walk-Forward Average")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "22_plan2_vs_plan3.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("PLAN 3: Extended Data & Walk-Forward Validation")
    print("=" * 70)

    # ---- Load data --------------------------------------------------------
    prices, vix, returns, vix_features, regime_labels = load_all_data()

    # ---- Build algorithm space --------------------------------------------
    algorithms = build_tier1_algorithm_space()

    # ---- Step 2: N=2 verification -----------------------------------------
    ok = verify_n2(algorithms, prices)
    if not ok:
        print("WARNING: Some algorithms failed N=2 test -- continuing anyway")

    # ---- Step 3: Walk-forward validation ----------------------------------
    wf_result = run_walk_forward(
        prices, vix, returns, vix_features, regime_labels, algorithms
    )

    # ---- Steps 4-5: Analyses ----------------------------------------------
    analysis_1_performance_comparison(wf_result)
    analysis_2_regime_conditional(wf_result)
    analysis_3_algorithm_stability(wf_result, algorithms)
    rank_df = analysis_4_rank_stability(wf_result)
    analysis_5_oracle_gap(wf_result)
    analysis_plan2_vs_plan3(wf_result)

    print("\n" + "=" * 70)
    print("PLAN 3 COMPLETE")
    print("=" * 70)
    print(f"All plots and CSVs saved to: {RESULTS_DIR}")
    print("\nFiles produced:")
    for fname in [
        "walk_forward_performance.csv",
        "algorithm_stability.csv",
        "algorithm_rank_stability.csv",
        "plan2_vs_plan3_comparison.csv",
        "17_walk_forward_sharpe_comparison.png",
        "18_regime_conditional_performance.png",
        "19_algorithm_stability.png",
        "20_algorithm_rank_stability.png",
        "21_oracle_gap_over_time.png",
        "22_plan2_vs_plan3.png",
    ]:
        full = os.path.join(RESULTS_DIR, fname)
        exists = os.path.exists(full)
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {fname}")


if __name__ == "__main__":
    main()
