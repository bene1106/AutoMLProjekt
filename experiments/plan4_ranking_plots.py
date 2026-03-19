"""
experiments/plan4_ranking_plots.py
====================================
Plan 4 -- Algorithm Ranking & Regime-Specific Visualisations (K=81, Tier 1+2)

Produces:
  results/algorithm_ranking_walkforward.csv
  results/26_algorithm_ranking_walkforward.png
  results/27_algorithm_ranking_by_family.png
  results/28_regime_ranking_calm.png
  results/29_regime_ranking_normal.png
  results/30_regime_ranking_tense.png
  results/31_regime_ranking_crisis.png
  results/32_regime_top5_table.png

Run from Implementierung1/ as:
    python -m experiments.plan4_ranking_plots
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

from regime_algo_selection.config import RESULTS_DIR

# ── Colours ────────────────────────────────────────────────────────────────────

FAMILY_COLORS = {
    "EqualWeight"        : "#607D8B",
    "MinimumVariance"    : "#1565C0",
    "RiskParity"         : "#2E7D32",
    "MaxDiversification" : "#6A1B9A",
    "Momentum"           : "#E65100",
    "TrendFollowing"     : "#F9A825",
    "MeanVariance"       : "#C62828",
    "RidgePortfolio"     : "#00838F",
    "LassoPortfolio"     : "#AD1457",
    "ElasticNetPortfolio": "#4E342E",
}

REGIME_COLORS = {
    "Calm"  : "#4CAF50",
    "Normal": "#2196F3",
    "Tense" : "#FF9800",
    "Crisis": "#F44336",
}

REGIME_FILES = {
    "Calm"  : "regime_ranking_calm.csv",
    "Normal": "regime_ranking_normal.csv",
    "Tense" : "regime_ranking_tense.csv",
    "Crisis": "regime_ranking_crisis.csv",
}

PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor"  : "#F8F9FA",
    "axes.grid"       : True,
    "grid.color"      : "#E0E0E0",
    "grid.linestyle"  : "-",
    "grid.linewidth"  : 0.5,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "font.family"     : "DejaVu Sans",
}

plt.rcParams.update(PLOT_STYLE)

FOLD_COLS = [f"fold_{i}" for i in range(1, 13)]


# ── Helper: pretty algo label ──────────────────────────────────────────────────

def _short_label(name: str) -> str:
    """Return a shorter display label for long algorithm names."""
    return name


# ── 1. Load pickle ─────────────────────────────────────────────────────────────

def load_fold_results(pkl_path: str):
    """Load the walk-forward pickle; return a list of FoldResult objects."""
    with open(pkl_path, "rb") as fh:
        obj = pickle.load(fh)
    # obj may be WalkForwardResult or list[FoldResult]
    if hasattr(obj, "folds"):
        return obj.folds
    return list(obj)


# ── 2. Build overall algorithm ranking from regime ranking CSVs ────────────────

def build_overall_ranking(fold_results) -> pd.DataFrame:
    """
    Build overall algorithm ranking by aggregating test-period Sharpe data
    from regime_ranking_*.csv files across all four regimes and 12 folds.

    For each algorithm:
      - fold_i score = unweighted mean of the four regime-specific fold_i values
        (NaNs ignored -- regime may not occur in every fold)
      - avg_sharpe  = mean of the 12 fold means
      - std_sharpe  = std dev of the 12 fold means
      - avg_rank    = mean rank within each fold (across all algos)

    Falls back to training-period scores from the pickle if regime CSVs miss an algo.
    """
    # ── Load regime CSVs ─────────────────────────────────────────────────────
    regime_dfs = {}
    for regime, fname in REGIME_FILES.items():
        path = os.path.join(RESULTS_DIR, fname)
        regime_dfs[regime] = pd.read_csv(path).set_index("algo_name")

    # Collect union of all algo names
    all_algos = set()
    for rdf in regime_dfs.values():
        all_algos.update(rdf.index.tolist())

    # For metadata (family, tier) use any one regime DF
    ref_df = list(regime_dfs.values())[0]

    rows = []
    for algo_name in sorted(all_algos):
        row = {"algo_name": algo_name}

        fold_means = []  # one value per fold (avg across regimes)
        for i in range(1, 13):
            col = f"fold_{i}"
            vals = []
            for rdf in regime_dfs.values():
                if algo_name in rdf.index and col in rdf.columns:
                    v = rdf.loc[algo_name, col]
                    try:
                        v = float(v)
                        if np.isfinite(v):
                            vals.append(v)
                    except (ValueError, TypeError):
                        pass
            fold_mean_i = float(np.mean(vals)) if vals else np.nan
            row[f"fold_{i}"] = fold_mean_i
            if not np.isnan(fold_mean_i):
                fold_means.append(fold_mean_i)

        row["avg_sharpe"] = float(np.mean(fold_means))   if fold_means            else np.nan
        row["std_sharpe"] = float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else 0.0
        row["n_folds"]    = len(fold_means)

        # Metadata
        if algo_name in ref_df.index:
            row["family"] = ref_df.loc[algo_name, "family"] if "family" in ref_df.columns else _infer_family(algo_name)
            row["tier"]   = int(ref_df.loc[algo_name, "tier"]) if "tier" in ref_df.columns else _infer_tier(algo_name)
        else:
            row["family"] = _infer_family(algo_name)
            row["tier"]   = _infer_tier(algo_name)

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("avg_sharpe", ascending=False).reset_index(drop=True)

    # ── Per-fold ranks (higher avg_sharpe in that fold -> rank 1) ───────────
    fold_cols_present = [f"fold_{i}" for i in range(1, 13) if f"fold_{i}" in df.columns]
    rank_rows = []
    for col in fold_cols_present:
        ranked = df[col].rank(ascending=False, method="min", na_option="bottom")
        rank_rows.append(ranked.rename(f"rank_{col}"))

    rank_df = pd.concat(rank_rows, axis=1)
    df["avg_rank"] = rank_df.mean(axis=1)
    df["std_rank"] = rank_df.std(axis=1, ddof=1)

    df["overall_rank"] = df["avg_sharpe"].rank(ascending=False, method="min").astype(int)

    return df


def _infer_family(name: str) -> str:
    n = name.lower()
    if "equalweight" in n or n == "ew":
        return "EqualWeight"
    if "minvar" in n:
        return "MinimumVariance"
    if "riskparity" in n:
        return "RiskParity"
    if "maxdiv" in n:
        return "MaxDiversification"
    if "momentum" in n:
        return "Momentum"
    if "trend" in n:
        return "TrendFollowing"
    if "meanvar" in n:
        return "MeanVariance"
    if "ridge" in n:
        return "RidgePortfolio"
    if "lasso" in n and "elastic" not in n:
        return "LassoPortfolio"
    if "elastic" in n:
        return "ElasticNetPortfolio"
    return "Other"


def _infer_tier(name: str) -> int:
    n = name.lower()
    if any(x in n for x in ["ridge", "lasso", "elastic"]):
        return 2
    return 1


# ── 3. Plot 26 -- Overall algorithm ranking (horizontal bar) ───────────────────

def plot_26_overall_ranking(df: pd.DataFrame, out_path: str):
    df_sorted = df.sort_values("avg_sharpe", ascending=True).copy()
    n = len(df_sorted)
    fig_h = max(16, n * 0.22)
    fig, ax = plt.subplots(figsize=(14, fig_h), facecolor="white")

    colors  = [FAMILY_COLORS.get(f, "#9E9E9E") for f in df_sorted["family"]]
    y_pos   = np.arange(n)
    bars    = ax.barh(y_pos, df_sorted["avg_sharpe"], color=colors, alpha=0.85,
                      height=0.72, zorder=3)

    # Error bars
    ax.barh(y_pos, df_sorted["std_sharpe"], left=df_sorted["avg_sharpe"],
            color="none", xerr=None, zorder=2)
    ax.errorbar(df_sorted["avg_sharpe"], y_pos,
                xerr=df_sorted["std_sharpe"],
                fmt="none", color="#333333", linewidth=0.7, capsize=2, zorder=4)

    # Reference line at 0
    ax.axvline(0, color="#555555", linewidth=1.0, linestyle="--", zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["algo_name"], fontsize=7.5)
    ax.set_xlabel("Average Net Sharpe Ratio (12 walk-forward folds)", fontsize=12)
    ax.set_title(
        "Algorithm Ranking -- Average Net Sharpe Across 12 Walk-Forward Folds\n"
        "K=81 algorithms (Tier 1 heuristics + Tier 2 linear), error bars = 1 std dev",
        fontsize=13, fontweight="bold", pad=14,
    )

    # Legend for families
    seen = []
    handles = []
    for fam in df_sorted["family"].unique():
        if fam not in seen:
            seen.append(fam)
            handles.append(
                mpatches.Patch(color=FAMILY_COLORS.get(fam, "#9E9E9E"),
                               label=fam, alpha=0.85)
            )
    ax.legend(handles=handles, title="Algorithm Family",
              loc="lower right", fontsize=9, title_fontsize=10,
              framealpha=0.9, edgecolor="#CCCCCC")

    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.6)
    ax.tick_params(axis="y", labelsize=7.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── 4. Plot 27 -- Sharpe distribution by family (box plot) ────────────────────

def plot_27_family_boxplot(df: pd.DataFrame, out_path: str):
    # Melt fold columns into long form
    fold_cols = [c for c in df.columns if c.startswith("fold_")]
    long_rows = []
    for _, row in df.iterrows():
        for fc in fold_cols:
            v = row[fc]
            if not pd.isna(v) and v != -999.0:
                long_rows.append({"family": row["family"], "sharpe": v})
    long_df = pd.DataFrame(long_rows)

    family_order = (
        long_df.groupby("family")["sharpe"].median()
        .sort_values(ascending=False).index.tolist()
    )

    fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")

    bp_data   = [long_df[long_df["family"] == fam]["sharpe"].values
                 for fam in family_order]
    bplot     = ax.boxplot(bp_data, vert=True, patch_artist=True,
                           widths=0.55, showfliers=True,
                           flierprops=dict(marker="o", markersize=3,
                                           alpha=0.4, linestyle="none"),
                           medianprops=dict(color="white", linewidth=2.5),
                           whiskerprops=dict(linewidth=1.2),
                           capprops=dict(linewidth=1.2))

    for patch, fam in zip(bplot["boxes"], family_order):
        c = FAMILY_COLORS.get(fam, "#9E9E9E")
        patch.set_facecolor(c)
        patch.set_alpha(0.80)

    ax.set_xticks(range(1, len(family_order) + 1))
    ax.set_xticklabels(family_order, rotation=25, ha="right", fontsize=10)
    ax.axhline(0, color="#555555", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Net Sharpe Ratio (individual fold observations)", fontsize=12)
    ax.set_title(
        "Algorithm Family Performance Distribution Across Walk-Forward Folds\n"
        "K=81 algorithms, 12 folds x 81 algorithms = up to 972 observations",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.6)

    # Annotation: count per family
    for i, fam in enumerate(family_order, 1):
        vals = long_df[long_df["family"] == fam]["sharpe"]
        ax.text(i, ax.get_ylim()[0] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"n={len(vals)}", ha="center", va="top", fontsize=8, color="#555555")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── 5. Plots 28-31 -- Per-regime horizontal bar charts ────────────────────────

def _plot_regime_ranking(regime_name: str, csv_path: str, plot_num: int,
                         out_path: str):
    df = pd.read_csv(csv_path)

    # Sort by mean_sharpe ascending (so best is at top after barh)
    df_sorted = df.sort_values("mean_sharpe", ascending=True).reset_index(drop=True)
    n = len(df_sorted)
    fig_h = max(16, n * 0.22)

    fig, ax = plt.subplots(figsize=(14, fig_h), facecolor="white")
    colors  = [FAMILY_COLORS.get(f, "#9E9E9E") for f in df_sorted["family"]]
    y_pos   = np.arange(n)

    ax.barh(y_pos, df_sorted["mean_sharpe"], color=colors, alpha=0.85,
            height=0.72, zorder=3)
    ax.errorbar(df_sorted["mean_sharpe"], y_pos,
                xerr=df_sorted["std_sharpe"],
                fmt="none", color="#333333", linewidth=0.7, capsize=2, zorder=4)
    ax.axvline(0, color="#555555", linewidth=1.0, linestyle="--", zorder=5)

    # Highlight top 5
    top5_idx = df_sorted.nlargest(5, "mean_sharpe").index
    for idx in top5_idx:
        ax.get_children()   # force render
        y = int(idx)
        ax.barh(y, df_sorted.loc[y, "mean_sharpe"],
                color=FAMILY_COLORS.get(df_sorted.loc[y, "family"], "#9E9E9E"),
                alpha=1.0, height=0.72, zorder=3,
                edgecolor="black", linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["algo_name"], fontsize=7.5)
    ax.set_xlabel("Average Net Sharpe Ratio (regime-filtered folds)", fontsize=12)

    regime_color = REGIME_COLORS.get(regime_name, "#777777")
    ax.set_title(
        f"Algorithm Ranking -- {regime_name} Regime  [Plot {plot_num}]\n"
        f"Average net Sharpe on days classified as {regime_name}, "
        f"across walk-forward folds  |  error bars = 1 std dev",
        fontsize=13, fontweight="bold", pad=14,
        color="black",
    )

    # Coloured regime strip on title background
    fig.patches.append(mpatches.FancyBboxPatch(
        (0.0, 0.965), 1.0, 0.035,
        transform=fig.transFigure, clip_on=False,
        facecolor=regime_color, alpha=0.18, zorder=0,
    ))

    # Legend
    seen = []
    handles = []
    for fam in df_sorted["family"].unique():
        if fam not in seen:
            seen.append(fam)
            handles.append(
                mpatches.Patch(color=FAMILY_COLORS.get(fam, "#9E9E9E"),
                               label=fam, alpha=0.85)
            )
    ax.legend(handles=handles, title="Algorithm Family",
              loc="lower right", fontsize=9, title_fontsize=10,
              framealpha=0.9, edgecolor="#CCCCCC")

    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── 6. Plot 32 -- Top-5 per regime table image ────────────────────────────────

def plot_32_top5_table(out_path: str):
    regimes     = ["Calm", "Normal", "Tense", "Crisis"]
    top5_by_reg = {}

    for regime in regimes:
        csv_path = os.path.join(RESULTS_DIR, REGIME_FILES[regime])
        df = pd.read_csv(csv_path)
        top5 = df.nlargest(5, "mean_sharpe")[["algo_name", "mean_sharpe", "family"]].copy()
        top5 = top5.reset_index(drop=True)
        top5_by_reg[regime] = top5

    # Build table data
    col_labels  = ["Rank"] + [f"{r} Regime" for r in regimes]
    table_data  = []
    for rank in range(5):
        row = [f"#{rank+1}"]
        for regime in regimes:
            r_df = top5_by_reg[regime]
            if rank < len(r_df):
                name   = r_df.iloc[rank]["algo_name"]
                sharpe = r_df.iloc[rank]["mean_sharpe"]
                row.append(f"{name}\n({sharpe:+.3f})")
            else:
                row.append("--")
        table_data.append(row)

    fig, ax = plt.subplots(figsize=(16, 5.5), facecolor="white")
    ax.axis("off")

    # Column widths
    col_widths = [0.06] + [0.235] * 4

    tbl = ax.table(
        cellText   = table_data,
        colLabels  = col_labels,
        loc        = "center",
        cellLoc    = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 3.2)

    # Style header row
    for j, regime in enumerate([""] + regimes):
        cell = tbl[0, j]
        cell.set_text_props(fontweight="bold", color="white", fontsize=11)
        if j == 0:
            cell.set_facecolor("#37474F")
        else:
            cell.set_facecolor(REGIME_COLORS[regime])

    # Style data rows
    for i in range(1, 6):
        # Rank column
        tbl[i, 0].set_facecolor("#ECEFF1")
        tbl[i, 0].set_text_props(fontweight="bold", fontsize=11)

        for j, regime in enumerate(regimes, 1):
            cell = tbl[i, j]
            # Gradient: top 1 darker, rest lighter
            alpha = 0.45 - (i - 1) * 0.07
            base  = to_rgba(REGIME_COLORS[regime], alpha=alpha)
            cell.set_facecolor(base)
            cell.set_text_props(fontsize=9)

            # Bold top-1
            if i == 1:
                cell.set_text_props(fontweight="bold", fontsize=9)

    # Alternating subtle shading for rank col
    for i in range(1, 6):
        tbl[i, 0].set_facecolor("#E8EAF6" if i % 2 == 0 else "#ECEFF1")

    fig.suptitle(
        "Top 5 Algorithms per Market Regime  --  Plan 4 Walk-Forward (K=81, Tier 1+2)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # Footnote
    fig.text(0.5, 0.01,
             "Values = average net Sharpe ratio on regime-filtered trading days across 12 walk-forward folds  "
             "|  KAPPA=0.001 switching cost",
             ha="center", va="bottom", fontsize=8.5, color="#555555")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Load pickle for reference (training-period all_scores per fold)
    pkl_path = os.path.join(RESULTS_DIR, "walk_forward_t12_all_scores.pkl")
    print(f"Loading fold results from: {pkl_path}")
    fold_results = load_fold_results(pkl_path)
    print(f"  Loaded {len(fold_results)} fold dicts "
          f"(keys per fold: {list(fold_results[0].keys())})")

    # ── Build & save overall ranking from regime CSVs (test-period data) ─────
    print("\nBuilding overall algorithm ranking from regime ranking CSVs ...")
    ranking_df = build_overall_ranking(fold_results)

    csv_out = os.path.join(RESULTS_DIR, "algorithm_ranking_walkforward.csv")
    cols_out = (
        ["overall_rank", "algo_name", "family", "tier",
         "avg_sharpe", "std_sharpe", "avg_rank", "std_rank", "n_folds"]
        + FOLD_COLS
    )
    cols_out = [c for c in cols_out if c in ranking_df.columns]
    ranking_df[cols_out].to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}  ({len(ranking_df)} algorithms)")

    # ── Plot 26 ──────────────────────────────────────────────────────────────
    print("\nPlot 26: Overall algorithm ranking ...")
    plot_26_overall_ranking(
        ranking_df,
        os.path.join(RESULTS_DIR, "26_algorithm_ranking_walkforward.png"),
    )

    # ── Plot 27 ──────────────────────────────────────────────────────────────
    print("Plot 27: Algorithm ranking by family (box plot) ...")
    plot_27_family_boxplot(
        ranking_df,
        os.path.join(RESULTS_DIR, "27_algorithm_ranking_by_family.png"),
    )

    # ── Plots 28-31 ──────────────────────────────────────────────────────────
    regime_specs = [
        ("Calm",   "regime_ranking_calm.csv",   28, "28_regime_ranking_calm.png"),
        ("Normal", "regime_ranking_normal.csv",  29, "29_regime_ranking_normal.png"),
        ("Tense",  "regime_ranking_tense.csv",   30, "30_regime_ranking_tense.png"),
        ("Crisis", "regime_ranking_crisis.csv",  31, "31_regime_ranking_crisis.png"),
    ]
    for regime_name, csv_name, plot_num, png_name in regime_specs:
        print(f"Plot {plot_num}: {regime_name} regime ranking ...")
        _plot_regime_ranking(
            regime_name,
            os.path.join(RESULTS_DIR, csv_name),
            plot_num,
            os.path.join(RESULTS_DIR, png_name),
        )

    # ── Plot 32 ──────────────────────────────────────────────────────────────
    print("Plot 32: Top-5 per regime table ...")
    plot_32_top5_table(
        os.path.join(RESULTS_DIR, "32_regime_top5_table.png"),
    )

    print("\nAll done.")

    # Quick summary to terminal
    print("\n--- Overall Ranking (Top 15) ---")
    top15 = ranking_df.head(15)[["overall_rank", "algo_name", "family",
                                  "avg_sharpe", "std_sharpe", "avg_rank"]]
    print(top15.to_string(index=False))


if __name__ == "__main__":
    main()
