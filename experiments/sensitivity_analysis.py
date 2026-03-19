"""
experiments/sensitivity_analysis.py
=====================================
Task 2.3 -- Switching Cost Sensitivity Analysis

Strategy: precompute (gross_returns, daily_turnovers) for every (algo, regime) pair
ONCE on the training data, then derive net-Sharpe for any kappa in O(1).
This makes 7 kappa values fast after one initial precomputation.

Run from Implementierung1/ as:
    python -m experiments.sensitivity_analysis
"""
import os
import sys
import time
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
from regime_algo_selection.agents.reflex_agent import ReflexAgent
from regime_algo_selection.evaluation.backtest import Backtester
from regime_algo_selection.evaluation.metrics  import compute_all_metrics
from regime_algo_selection.config import (
    TRAIN_END, VAL_END, N_ASSETS, REGIME_NAMES, RESULTS_DIR, RANDOM_SEED,
)

KAPPA_VALUES = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]


# ── Precompute per-algo-regime statistics ──────────────────────────────────────

def precompute_algo_regime_stats(algorithms, prices, returns, regime_labels,
                                  train_dates):
    """
    For every (algo, regime) pair evaluate on training days.
    Returns: dict[(algo.name, regime)] -> (gross_rets array, turnovers array)
    so that net_ret_t = gross_ret_t - kappa * turnover_t,
    and net_sharpe = sharpe(net_rets) for any kappa without re-running weights.
    """
    train_labels  = regime_labels.loc[regime_labels.index.isin(train_dates)]
    train_returns = returns.loc[train_dates]
    common_dates  = train_returns.index.intersection(train_labels.index)

    stats = {}   # (algo_name, regime) -> {"gross": array, "turn": array}
    n_algos = len(algorithms)

    for ai, algo in enumerate(algorithms, 1):
        for regime in range(1, 5):
            regime_dates = common_dates[train_labels.loc[common_dates] == regime]

            gross_list = []
            turn_list  = []
            prev_w     = np.ones(N_ASSETS) / N_ASSETS

            for t in regime_dates:
                ph = prices.loc[prices.index < t]
                if len(ph) < 22:
                    continue
                try:
                    w = algo.compute_weights(ph)
                except Exception:
                    w = np.ones(N_ASSETS) / N_ASSETS

                w = np.where(np.isfinite(w), w, 0.0)
                w = np.clip(w, 0, None)
                s = w.sum()
                w = w / s if s > 1e-12 else np.ones(N_ASSETS) / N_ASSETS

                if t not in train_returns.index:
                    prev_w = w
                    continue

                r = train_returns.loc[t].fillna(0).values
                gross_list.append(float(w @ r))
                turn_list.append(float(np.abs(w - prev_w).sum()))
                prev_w = w

            stats[(algo.name, regime)] = {
                "gross": np.array(gross_list),
                "turn" : np.array(turn_list),
            }

        if ai % 10 == 0:
            print(f"    Precomputed {ai}/{n_algos} algorithms...")

    return stats


def best_algo_for_regime(algorithms, stats, regime, kappa):
    """Find the algorithm with the highest net-Sharpe for a given kappa."""
    best_sharpe = -np.inf
    best_algo   = algorithms[0]

    for algo in algorithms:
        entry = stats.get((algo.name, regime))
        if entry is None:
            continue
        gross = entry["gross"]
        turn  = entry["turn"]
        if len(gross) < 10:
            continue

        net = gross - kappa * turn
        std = net.std()
        if std < 1e-12:
            sr = net.mean() * np.sqrt(252)
        else:
            sr = (net.mean() / std) * np.sqrt(252)

        if sr > best_sharpe:
            best_sharpe = sr
            best_algo   = algo

    return best_algo, best_sharpe


class _EWAgent:
    def __init__(self): self._algo = EqualWeight()
    def select(self, r): return self._algo


# ── Task 2.3 ──────────────────────────────────────────────────────────────────

def task_2_3_sensitivity(algorithms, stats, backtester_factory,
                          regime_labels, test_start, test_end):
    print("\n" + "="*60)
    print("TASK 2.3: Switching Cost Sensitivity Analysis")
    print("="*60)

    rows        = []
    mapping_log = {}   # kappa -> {regime: algo_name}

    for kappa in KAPPA_VALUES:
        # Build mapping: for each regime pick best net-Sharpe algo
        mapping = {}
        for regime in range(1, 5):
            algo, sr = best_algo_for_regime(algorithms, stats, regime, kappa)
            mapping[regime] = algo

        mapping_log[kappa] = {r: a.name for r, a in mapping.items()}

        # Simple lookup agent
        class _KappaAgent:
            def __init__(self, m):
                self._m = m
            def select(self, r):
                return self._m.get(int(r), list(self._m.values())[0])

        agent = _KappaAgent(mapping)
        bt    = backtester_factory(kappa)
        result = bt.run(agent, test_start, test_end, run_label=f"kappa={kappa}")
        m = compute_all_metrics(result)

        rows.append({
            "kappa"          : kappa,
            "Sharpe"         : m["sharpe_ratio"],
            "Cum Return %"   : m["cumulative_return"],
            "Total Cost bps" : m["total_switching_cost"],
            "Turnover"       : m["total_turnover"],
            "Calm Algo"      : mapping[1].name,
            "Normal Algo"    : mapping[2].name,
            "Tense Algo"     : mapping[3].name,
            "Crisis Algo"    : mapping[4].name,
        })
        print(f"  kappa={kappa:.4f}  Sharpe={m['sharpe_ratio']:.4f}  "
              f"Cum={m['cumulative_return']:.2f}%  "
              f"Turn={m['total_turnover']:.1f}  "
              f"Cost={m['total_switching_cost']:.4f}bps  "
              f"Mapping: C={mapping[1].name[:12]} N={mapping[2].name[:12]} "
              f"T={mapping[3].name[:12]} Cr={mapping[4].name[:12]}")

    # EW reference
    bt_base   = backtester_factory(0.001)
    res_ew    = bt_base.run(_EWAgent(), test_start, test_end, "EW")
    ew_sharpe = compute_all_metrics(res_ew)["sharpe_ratio"]

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "kappa_sensitivity.csv"), index=False)

    print("\n--- Full Sensitivity Table ---")
    print(df[["kappa","Sharpe","Cum Return %","Total Cost bps","Turnover",
              "Calm Algo","Normal Algo"]].to_string(index=False))

    _plot_sensitivity(df, ew_sharpe, mapping_log)
    return df


def _plot_sensitivity(df, ew_sharpe, mapping_log):
    # Plot 14: Sharpe vs kappa
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["kappa"], df["Sharpe"], marker="o", color="#1565C0",
            linewidth=2, markersize=7, label="Reflex Agent (net-fit)")
    ax.axhline(ew_sharpe, color="gray", linewidth=1.8, linestyle="--",
               label=f"Equal Weight ({ew_sharpe:.4f})")
    ax.axhline(0, color="black", linewidth=0.7)
    for _, row in df.iterrows():
        ax.annotate(f"{row['Sharpe']:.3f}",
                    xy=(row["kappa"], row["Sharpe"]),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_xscale("symlog", linthresh=0.0001)
    ax.set_xlabel("kappa (switching cost coefficient)")
    ax.set_ylabel("Sharpe Ratio (Test 2021-2024)")
    ax.set_title("Sharpe Ratio vs Switching Cost (kappa)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "14_kappa_sensitivity_sharpe.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # Plot 15: Turnover vs kappa
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["kappa"], df["Turnover"], marker="s", color="#E65100",
            linewidth=2, markersize=7, label="Portfolio Turnover")
    ax2 = ax.twinx()
    ax2.plot(df["kappa"], df["Total Cost bps"], marker="^", color="#6A1B9A",
             linewidth=2, markersize=7, linestyle="--",
             label="Total Switch Cost (bps)")
    ax.set_xscale("symlog", linthresh=0.0001)
    ax.set_xlabel("kappa (switching cost coefficient)")
    ax.set_ylabel("Total Turnover", color="#E65100")
    ax2.set_ylabel("Total Switching Cost (bps)", color="#6A1B9A")
    ax.set_title("Turnover and Switching Cost vs kappa", fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "15_kappa_sensitivity_turnover.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # Plot 16: Regime mapping heatmap
    regimes = [1, 2, 3, 4]
    kappas  = list(mapping_log.keys())
    all_names   = sorted({n for km in mapping_log.values() for n in km.values()})
    name_to_int = {n: i for i, n in enumerate(all_names)}

    matrix = np.array([[name_to_int[mapping_log[k][r]] for k in kappas]
                        for r in regimes])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(matrix, aspect="auto", cmap="tab20",
              vmin=0, vmax=max(len(all_names)-1, 1))
    ax.set_xticks(range(len(kappas)))
    ax.set_xticklabels([str(k) for k in kappas], fontsize=9)
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels([f"R{r} ({REGIME_NAMES[r]})" for r in regimes], fontsize=9)
    for i, r in enumerate(regimes):
        for j, k in enumerate(kappas):
            name  = mapping_log[k][r]
            short = name[:13]
            ax.text(j, i, short, ha="center", va="center", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7))
    ax.set_xlabel("kappa")
    ax.set_title("Regime Mapping: Selected Algorithm per Regime per kappa",
                 fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "16_kappa_regime_mapping.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Task 2.3: Switching Cost Sensitivity Analysis")
    print("="*60)

    print("\nLoading data...")
    data   = load_data()
    prices = data["prices"]
    vix    = data["vix"]

    returns       = compute_returns(prices)
    vix_features  = compute_vix_features(vix)
    regime_labels = compute_regime_labels(vix)

    common_dates = returns.index
    train_mask   = common_dates <= TRAIN_END
    test_mask    = common_dates > VAL_END
    train_dates  = common_dates[train_mask]
    test_dates   = common_dates[test_mask]
    test_start   = str(test_dates[0].date())
    test_end     = str(test_dates[-1].date())

    # Train classifier
    vix_train  = vix_features.loc[vix_features.index.isin(train_dates)].dropna()
    y_train    = regime_labels.loc[vix_train.index]
    classifier = RegimeClassifier("logistic_regression")
    classifier.fit(vix_train, y_train)

    algorithms = build_tier1_algorithm_space()

    # Precompute algo-regime stats ONCE (the expensive step)
    print("\nPrecomputing algo-regime statistics on training data (runs once)...")
    t0    = time.time()
    stats = precompute_algo_regime_stats(algorithms, prices, returns,
                                          regime_labels, train_dates)
    print(f"  Precomputation done in {time.time()-t0:.1f}s")

    # Backtester factory: different kappa per run
    def backtester_factory(kappa):
        return Backtester(
            algorithms        = algorithms,
            regime_classifier = classifier,
            returns           = returns,
            prices            = prices,
            vix_features      = vix_features,
            regime_labels     = regime_labels,
            kappa             = kappa,
        )

    df = task_2_3_sensitivity(
        algorithms, stats, backtester_factory,
        regime_labels, test_start, test_end,
    )

    print("\n" + "="*60)
    print("  Task 2.3 complete. All outputs saved to results/")
    print("="*60)


if __name__ == "__main__":
    main()
