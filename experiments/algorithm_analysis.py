"""
experiments/algorithm_analysis.py
====================================
Task 2.1 -- Per-Algorithm Analysis
Task 2.2 -- Reflex Agent with Net-Sharpe Fitting

Run from Implementierung1/ as:
    python -m experiments.algorithm_analysis
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regime_algo_selection.data.loader   import load_data
from regime_algo_selection.data.features import compute_returns, compute_vix_features
from regime_algo_selection.regimes.ground_truth import compute_regime_labels
from regime_algo_selection.regimes.classifier   import RegimeClassifier
from regime_algo_selection.algorithms.tier1_heuristics import (
    build_tier1_algorithm_space, EqualWeight,
)
from regime_algo_selection.agents.reflex_agent import ReflexAgent, OracleAgent
from regime_algo_selection.evaluation.backtest import Backtester
from regime_algo_selection.evaluation.metrics  import compute_all_metrics
from regime_algo_selection.config import (
    TRAIN_END, VAL_END, KAPPA, REGIME_NAMES, RESULTS_DIR, RANDOM_SEED,
)

FAMILY_COLORS = {
    "EqualWeight"       : "#607D8B",
    "MinimumVariance"   : "#1565C0",
    "RiskParity"        : "#2E7D32",
    "MaxDiversification": "#6A1B9A",
    "Momentum"          : "#E65100",
    "TrendFollowing"    : "#F9A825",
    "MeanVariance"      : "#C62828",
}


# ── Single-algo agent helper ───────────────────────────────────────────────────

class SingleAlgoAgent:
    """Always selects the same algorithm, regardless of regime."""
    def __init__(self, algo):
        self._algo = algo
    def select(self, regime):
        return self._algo


# ── Task 2.1 ──────────────────────────────────────────────────────────────────

def task_2_1_algorithm_ranking(
    algorithms, backtester, test_start, test_end
):
    print("\n" + "="*60)
    print("TASK 2.1: Per-Algorithm Ranking on Test Period (2021-2024)")
    print("="*60)

    rows = []
    n    = len(algorithms)

    for i, algo in enumerate(algorithms, 1):
        agent  = SingleAlgoAgent(algo)
        result = backtester.run(
            agent       = agent,
            start_date  = test_start,
            end_date    = test_end,
            run_label   = algo.name,
        )
        m = compute_all_metrics(result)
        rows.append({
            "Algorithm"    : algo.name,
            "Family"       : algo.family,
            "Sharpe"       : m["sharpe_ratio"],
            "Cum Return %" : m["cumulative_return"],
            "Max DD %"     : m["max_drawdown"],
            "Turnover"     : m["total_turnover"],
            "Switch Cost"  : m["total_switching_cost"],
        })
        if i % 10 == 0:
            print(f"  Evaluated {i}/{n} algorithms...")

    # Equal Weight (reference)
    ew_agent  = SingleAlgoAgent(EqualWeight())
    ew_result = backtester.run(
        agent=ew_agent, start_date=test_start, end_date=test_end,
        run_label="EqualWeight_ref",
    )
    ew_m = compute_all_metrics(ew_result)
    rows.append({
        "Algorithm"    : "** EqualWeight (ref) **",
        "Family"       : "EqualWeight",
        "Sharpe"       : ew_m["sharpe_ratio"],
        "Cum Return %" : ew_m["cumulative_return"],
        "Max DD %"     : ew_m["max_drawdown"],
        "Turnover"     : ew_m["total_turnover"],
        "Switch Cost"  : ew_m["total_switching_cost"],
    })

    df = pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"

    # Save CSV
    df.to_csv(os.path.join(RESULTS_DIR, "algorithm_ranking.csv"))

    # Print top/bottom
    print("\n  Top 10 algorithms by Sharpe:")
    print(df.head(10).to_string())
    print("\n  Bottom 5 algorithms by Sharpe:")
    print(df.tail(5).to_string())

    ew_rank = df[df["Algorithm"].str.contains("EqualWeight")].index[0]
    ew_sharpe = ew_m["sharpe_ratio"]
    n_beat_ew = (df["Sharpe"] > ew_sharpe).sum() - 1  # exclude EW itself
    print(f"\n  Equal Weight rank: {ew_rank} / {len(df)-1}  (Sharpe={ew_sharpe:.4f})")
    print(f"  Algorithms beating Equal Weight: {n_beat_ew}")

    _plot_algorithm_ranking(df)
    _plot_family_boxplot(df)

    return df


def _plot_algorithm_ranking(df):
    # Exclude the EW reference row for cleaner plot
    plot_df = df[~df["Algorithm"].str.contains("ref")].copy()
    plot_df = plot_df.sort_values("Sharpe")

    fig, ax = plt.subplots(figsize=(10, 14))
    colors  = [FAMILY_COLORS.get(f, "#888888") for f in plot_df["Family"]]
    bars    = ax.barh(
        plot_df["Algorithm"], plot_df["Sharpe"],
        color=colors, edgecolor="white", height=0.75,
    )
    ax.axvline(0,           color="black", linewidth=0.8)
    ax.axvline(0.3712,      color="gray",  linewidth=1.5, linestyle="--",
               label="Equal Weight (0.3712)")
    ax.set_xlabel("Sharpe Ratio (Test 2021-2024)")
    ax.set_title("All 48 Tier-1 Algorithms: Sharpe Ratio Ranking\n(2021-2024 Test Period)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.2)

    # Family legend
    handles = [
        plt.Rectangle((0,0), 1, 1, color=c, label=f)
        for f, c in FAMILY_COLORS.items()
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, title="Family")

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "12_algorithm_ranking.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_family_boxplot(df):
    plot_df = df[~df["Algorithm"].str.contains("ref")].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    families = list(FAMILY_COLORS.keys())
    data     = [plot_df.loc[plot_df["Family"] == f, "Sharpe"].values for f in families]
    bp = ax.boxplot(data, vert=True, patch_artist=True, labels=families)
    for patch, family in zip(bp["boxes"], families):
        patch.set_facecolor(FAMILY_COLORS.get(family, "#888888"))
        patch.set_alpha(0.8)
    ax.axhline(0,      color="black", linewidth=0.8)
    ax.axhline(0.3712, color="gray",  linewidth=1.5, linestyle="--",
               label="Equal Weight (0.3712)")
    ax.set_ylabel("Sharpe Ratio (Test 2021-2024)")
    ax.set_title("Sharpe Distribution by Algorithm Family", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "13_algorithm_families.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Task 2.2 ──────────────────────────────────────────────────────────────────

def task_2_2_net_sharpe_agent(
    algorithms, backtester, classifier,
    returns, regime_labels, prices,
    train_dates, test_start, test_end,
):
    print("\n" + "="*60)
    print("TASK 2.2: Reflex Agent -- Net-Sharpe vs Gross-Sharpe Fitting")
    print("="*60)

    train_labels = regime_labels.loc[regime_labels.index.isin(train_dates)]
    train_returns = returns.loc[train_dates]

    # Gross-Sharpe agent (original behaviour)
    print("\n--- Gross-Sharpe Agent ---")
    gross_agent = ReflexAgent()
    gross_agent.fit(
        algorithms=algorithms, returns=train_returns,
        regime_labels=train_labels, prices=prices,
        metric="gross",
    )

    # Net-Sharpe agent
    print("\n--- Net-Sharpe Agent ---")
    net_agent = ReflexAgent()
    net_agent.fit(
        algorithms=algorithms, returns=train_returns,
        regime_labels=train_labels, prices=prices,
        metric="net", kappa=KAPPA,
    )

    # EW baseline
    class _EWAgent:
        def __init__(self): self._algo = EqualWeight()
        def select(self, r): return self._algo

    # Backtest all three
    res_ew    = backtester.run(_EWAgent(), test_start, test_end, "Equal Weight")
    res_gross = backtester.run(gross_agent, test_start, test_end, "Reflex (gross)")
    res_net   = backtester.run(net_agent,   test_start, test_end, "Reflex (net)")

    from regime_algo_selection.evaluation.metrics import compute_all_metrics
    m_ew    = compute_all_metrics(res_ew)
    m_gross = compute_all_metrics(res_gross)
    m_net   = compute_all_metrics(res_net)

    # Mapping comparison table
    print("\n--- Regime Mapping Comparison ---")
    print(f"  {'Regime':<20} {'Gross-Sharpe Pick':<25} {'Net-Sharpe Pick':<25}")
    print("  " + "-"*70)
    for r in range(1, 5):
        gname = gross_agent.mapping[r].name
        nname = net_agent.mapping[r].name
        flag  = " <-- CHANGED" if gname != nname else ""
        print(f"  Regime {r} ({REGIME_NAMES[r]:<6}) {gname:<25} {nname:<25}{flag}")

    # Performance comparison
    print("\n--- Performance Comparison ---")
    print(f"  {'Strategy':<25} {'Sharpe':>10} {'Cum Ret%':>12} {'Turnover':>12}")
    print("  " + "-"*60)
    for label, m in [("Equal Weight", m_ew), ("Reflex (gross fit)", m_gross),
                      ("Reflex (net fit)", m_net)]:
        print(f"  {label:<25} {m['sharpe_ratio']:>10.4f} "
              f"{m['cumulative_return']:>12.2f} {m['total_turnover']:>12.2f}")

    return {
        "gross": (gross_agent, res_gross, m_gross),
        "net"  : (net_agent,   res_net,   m_net),
        "ew"   : (_EWAgent(), res_ew, m_ew),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Part 2: Algorithm Analysis & Agent Diagnostics")
    print("="*60)

    print("\nLoading data...")
    data   = load_data()
    prices = data["prices"]
    vix    = data["vix"]

    returns      = compute_returns(prices)
    vix_features = compute_vix_features(vix)
    regime_labels= compute_regime_labels(vix)

    common_dates = returns.index
    train_mask   = common_dates <= TRAIN_END
    test_mask    = common_dates > VAL_END
    train_dates  = common_dates[train_mask]
    test_dates   = common_dates[test_mask]
    test_start   = str(test_dates[0].date())
    test_end     = str(test_dates[-1].date())

    # Train classifier (needed for backtester)
    vix_train = vix_features.loc[vix_features.index.isin(train_dates)].dropna()
    y_train   = regime_labels.loc[vix_train.index]
    classifier = RegimeClassifier("logistic_regression")
    classifier.fit(vix_train, y_train)

    algorithms = build_tier1_algorithm_space()

    backtester = Backtester(
        algorithms        = algorithms,
        regime_classifier = classifier,
        returns           = returns,
        prices            = prices,
        vix_features      = vix_features,
        regime_labels     = regime_labels,
        kappa             = KAPPA,
    )

    # Task 2.1
    ranking_df = task_2_1_algorithm_ranking(algorithms, backtester, test_start, test_end)

    # Task 2.2
    agent_results = task_2_2_net_sharpe_agent(
        algorithms, backtester, classifier,
        returns, regime_labels, prices,
        train_dates, test_start, test_end,
    )

    print("\n" + "="*60)
    print("  Tasks 2.1 & 2.2 complete. All outputs saved to results/")
    print("="*60)


if __name__ == "__main__":
    main()
