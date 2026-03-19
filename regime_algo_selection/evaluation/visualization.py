# evaluation/visualization.py — Plots and tables for the regime-aware system

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on any machine
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from regime_algo_selection.config import REGIME_NAMES, RESULTS_DIR

# Colour palette
REGIME_COLORS = {1: "#4CAF50", 2: "#2196F3", 3: "#FF9800", 4: "#F44336"}
STRATEGY_COLORS = {
    "Equal Weight" : "#888888",
    "Reflex Agent" : "#1565C0",
    "Oracle Agent" : "#6A1B9A",
    "Buy & Hold"   : "#FF8F00",
}


def _shade_regimes(ax, regime_series: pd.Series) -> None:
    """Add background shading by regime to an axes."""
    dates = regime_series.index
    prev_r = None
    start  = dates[0]
    for i, (d, r) in enumerate(regime_series.items()):
        if r != prev_r:
            if prev_r is not None:
                ax.axvspan(start, d, alpha=0.12, color=REGIME_COLORS.get(prev_r, "#CCCCCC"),
                           linewidth=0)
            start  = d
            prev_r = r
    if prev_r is not None:
        ax.axvspan(start, dates[-1], alpha=0.12, color=REGIME_COLORS.get(prev_r, "#CCCCCC"),
                   linewidth=0)


# ── Plot 1: Cumulative Wealth ──────────────────────────────────────────────────

def plot_cumulative_wealth(results_dict: dict, regime_series: pd.Series) -> str:
    """
    Plot cumulative wealth curves for all strategies.

    Parameters
    ----------
    results_dict : {label: BacktestResult}
    regime_series : pd.Series — true regime labels for background shading
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    for label, result in results_dict.items():
        r = result.net_returns.dropna()
        wealth = (1 + r).cumprod()
        color  = STRATEGY_COLORS.get(label, None)
        lw     = 2.5 if "Agent" in label else 1.5
        ls     = "-" if label != "Oracle Agent" else "--"
        ax.plot(wealth, label=label, color=color, linewidth=lw, linestyle=ls)

    _shade_regimes(ax, regime_series)

    # Legend for regime shading
    patches = [mpatches.Patch(color=c, alpha=0.4, label=REGIME_NAMES[r])
               for r, c in REGIME_COLORS.items()]
    legend1 = ax.legend(handles=patches, loc="upper left", fontsize=8, title="Regime")
    ax.add_artist(legend1)
    ax.legend(loc="upper center", ncol=len(results_dict), fontsize=9)

    ax.set_title("Cumulative Wealth (Test Period)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($1 start)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "01_cumulative_wealth.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Plot 2: Regime Classification Over Time ────────────────────────────────────

def plot_regime_classification(
    vix: pd.Series,
    regime_true: pd.Series,
    regime_pred: pd.Series,
    start_date: str = None,
    end_date:   str = None,
) -> str:
    """
    Two-panel plot: VIX with thresholds | true vs predicted regime.
    """
    if start_date:
        vix          = vix.loc[start_date:]
        regime_true  = regime_true.loc[start_date:]
        regime_pred  = regime_pred.loc[start_date:]
    if end_date:
        vix          = vix.loc[:end_date]
        regime_true  = regime_true.loc[:end_date]
        regime_pred  = regime_pred.loc[:end_date]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    # Top: VIX
    ax = axes[0]
    ax.plot(vix, color="#333333", linewidth=0.8, label="VIX")
    for thr, col in zip([15, 20, 30], ["#4CAF50", "#FF9800", "#F44336"]):
        ax.axhline(thr, color=col, linewidth=1.2, linestyle="--", alpha=0.7,
                   label=f"VIX={thr}")
    ax.set_ylabel("VIX Level")
    ax.set_title("VIX Over Time with Regime Thresholds", fontweight="bold")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.2)

    # Bottom: True vs predicted regime
    ax2 = axes[1]
    common = regime_true.index.intersection(regime_pred.index)
    for r in [1, 2, 3, 4]:
        mask_true = regime_true.loc[common] == r
        ax2.bar(common[mask_true], np.ones(mask_true.sum()),
                color=REGIME_COLORS[r], alpha=0.7, width=1.5, label=f"True: {REGIME_NAMES[r]}")
        mask_pred = regime_pred.loc[common] == r
        ax2.bar(common[mask_pred], -np.ones(mask_pred.sum()),
                color=REGIME_COLORS[r], alpha=0.3, width=1.5)

    ax2.set_ylabel("Regime (+ true / - pred)")
    ax2.set_title("True Regime (top) vs Predicted Regime (bottom)", fontweight="bold")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.legend(fontsize=7, ncol=4)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "02_regime_classification.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Plot 3: Algorithm Selection Over Time ──────────────────────────────────────

def plot_algorithm_selection(result, top_n: int = 10) -> str:
    """
    Stacked area plot of which algorithm was selected each day.
    Only shows the top_n most-used algorithms for readability.
    """
    selections = result.algorithm_selections

    # One-hot encode
    dummies = pd.get_dummies(selections)
    top_algos = dummies.sum().nlargest(top_n).index.tolist()
    dummies = dummies[top_algos]

    # Rolling 20-day mean for smoothed view
    smooth = dummies.rolling(20, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(13, 5))
    smooth.plot.area(ax=ax, alpha=0.75, linewidth=0)
    ax.set_title("Algorithm Selection Frequency (Reflex Agent, 20-day rolling avg)",
                 fontweight="bold")
    ax.set_ylabel("Selection Frequency")
    ax.set_xlabel("Date")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "03_algorithm_selection.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Plot 4: Per-Regime Performance Table ──────────────────────────────────────

def plot_regime_table(regime_df: pd.DataFrame) -> str:
    """Render per-regime metrics as a matplotlib table image."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    regime_df_display = regime_df.copy()
    regime_df_display.index = [
        f"Regime {r} ({REGIME_NAMES[r]})" for r in regime_df_display.index
    ]

    tbl = ax.table(
        cellText   = regime_df_display.values,
        colLabels  = regime_df_display.columns.tolist(),
        rowLabels  = regime_df_display.index.tolist(),
        cellLoc    = "center",
        loc        = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.4, 1.6)

    # Header shading
    for j in range(len(regime_df_display.columns)):
        tbl[0, j].set_facecolor("#CFD8DC")

    # Row shading by regime
    for i, r in enumerate(regime_df.index, start=1):
        tbl[i, -1].set_facecolor(REGIME_COLORS.get(r, "#FFFFFF") + "44")

    ax.set_title("Per-Regime Performance (Test Period)", fontweight="bold", pad=20)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "04_per_regime_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Plot 5: Confusion Matrix ───────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray) -> str:
    """Plot regime classifier confusion matrix as a seaborn heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = [f"{r}\n({REGIME_NAMES[r]})" for r in [1, 2, 3, 4]]
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=labels, yticklabels=labels,
        cmap="Blues", ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Regime", fontsize=10)
    ax.set_ylabel("True Regime",      fontsize=10)
    ax.set_title("Regime Classifier Confusion Matrix (Test Set)", fontweight="bold")
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "05_confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Plot 6: Metrics Summary Table ─────────────────────────────────────────────

def plot_metrics_summary(metrics_dict: dict) -> str:
    """Render strategy comparison metrics as a matplotlib table image."""
    strategies = list(metrics_dict.keys())
    row_labels = [
        "Cum. Return (%)", "Ann. Return (%)", "Ann. Volatility (%)",
        "Sharpe Ratio", "Sortino Ratio", "Max Drawdown (%)",
        "Total Turnover", "Switch Cost (bps)", "Regime Accuracy", "# Days",
    ]
    keys = [
        "cumulative_return", "ann_return", "ann_volatility",
        "sharpe_ratio", "sortino_ratio", "max_drawdown",
        "total_turnover", "total_switching_cost", "regime_accuracy", "n_days",
    ]

    cell_data = []
    for key in keys:
        row = []
        for s in strategies:
            val = metrics_dict[s].get(key, "N/A")
            if isinstance(val, float) and not np.isnan(val):
                row.append(f"{val:.4f}")
            elif isinstance(val, int):
                row.append(str(val))
            else:
                row.append("N/A")
        cell_data.append(row)

    fig, ax = plt.subplots(figsize=(max(8, 3 * len(strategies)), 5))
    ax.axis("off")
    tbl = ax.table(
        cellText  = cell_data,
        rowLabels = row_labels,
        colLabels = strategies,
        cellLoc   = "center",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.3, 1.5)

    for j in range(len(strategies)):
        tbl[0, j].set_facecolor("#B0BEC5")

    ax.set_title("Strategy Performance Summary (Test Period)", fontweight="bold", pad=20)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "06_metrics_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
