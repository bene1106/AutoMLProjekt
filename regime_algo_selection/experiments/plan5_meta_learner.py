# experiments/plan5_meta_learner.py -- Plan 12: Meta-Learner with Oracle Regime
#
# Full walk-forward experiment for the Meta-Learner Stage 2.
#
# Pipeline per fold:
#   Stage 0 : Pre-train Tier 2 algorithms on training block
#   Stage 1 : Precompute all 81 algorithm outputs for train + test
#   Stage 2 : Train MetaLearnerNetwork (sequential reward maximisation)
#   Eval    : Evaluate meta-learner + baselines on test year
#
# Baselines per fold:
#   - Equal Weight (EW)
#   - Best Individual Algorithm (oracle: highest in-sample net Sharpe)
#   - Reflex Agent with Oracle Regime (per-regime best algo from training)
#   - Random Meta-Learner (alpha_t ~ Uniform)   [sanity check]
#
# Usage:
#   cd Implementierung1
#   python -u -m regime_algo_selection.experiments.plan5_meta_learner

import os
import sys
import time
import copy
import warnings
warnings.filterwarnings("ignore")

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---- project imports -------------------------------------------------------
from regime_algo_selection.config import (
    RESULTS_DIR, KAPPA, REGIME_NAMES, N_REGIMES, ASSETS, N_ASSETS, RANDOM_SEED,
)
from regime_algo_selection.data.loader_extended import load_data_extended
from regime_algo_selection.data.features import (
    compute_returns, compute_asset_features,
)
from regime_algo_selection.regimes.ground_truth import compute_regime_labels
from regime_algo_selection.algorithms.tier1_heuristics import build_algorithm_space
from regime_algo_selection.algorithms.base import TrainablePortfolioAlgorithm
from regime_algo_selection.algorithms.stage0 import pretrain_tier2_algorithms
from regime_algo_selection.evaluation.walk_forward import WalkForwardValidator

from regime_algo_selection.meta_learner.dataset import MetaLearnerDataset
from regime_algo_selection.meta_learner.network import MetaLearnerNetwork
from regime_algo_selection.meta_learner.trainer import MetaLearnerTrainer
from regime_algo_selection.meta_learner.inference import MetaLearnerAgent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # Meta-learner architecture
    "input_dim":    29,          # 25 asset features + 4 regime one-hot
    "hidden_dims":  [128, 64],
    "dropout":      0.1,

    # Training
    "n_epochs":     100,
    "lr":           0.01,
    "weight_decay": 1e-4,
    "grad_clip":    1.0,

    # Costs (same as Plan 4 for comparability)
    "kappa":        KAPPA,       # 0.001
    "kappa_a":      0.0,         # algorithm switching cost — off for first run
    "lambda_entropy": 0.1,       # entropy regularisation coefficient

    # Walk-forward
    "train_years":  8,
    "test_years":   1,
    "step_years":   1,
    "min_test_start": "2013-01-01",
    "data_end":     "2024-12-31",

    # Random seed
    "seed":         RANDOM_SEED,
}

PLAN5_RESULTS_DIR = os.path.join(RESULTS_DIR, "plan5_meta_learner")
os.makedirs(PLAN5_RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_metrics(daily_net_returns: np.ndarray) -> dict:
    """Compute standard portfolio metrics from an array of daily net returns."""
    r = daily_net_returns[np.isfinite(daily_net_returns)]
    T = len(r)
    if T < 10:
        return {k: np.nan for k in [
            "sharpe", "sortino", "ann_return", "ann_vol", "max_drawdown",
            "total_turnover", "n_days",
        ]}

    ann = 252
    cumw = np.cumprod(1 + r)
    cum_ret = cumw[-1] - 1
    ann_ret = (1 + cum_ret) ** (ann / T) - 1
    ann_vol = r.std() * np.sqrt(ann)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else np.nan

    downside = r[r < 0]
    dd_std = downside.std() * np.sqrt(ann) if len(downside) > 1 else np.nan
    sortino = ann_ret / dd_std if (dd_std and dd_std > 1e-12) else np.nan

    running_max = np.maximum.accumulate(cumw)
    drawdown = (cumw - running_max) / running_max
    max_dd = float(drawdown.min())

    return {
        "sharpe":      round(sharpe,   4),
        "sortino":     round(sortino,  4),
        "ann_return":  round(ann_ret * 100, 2),
        "ann_vol":     round(ann_vol  * 100, 2),
        "max_drawdown": round(max_dd  * 100, 2),
        "n_days":      T,
    }


def _portfolio_sharpe_from_weights(
    weight_matrix: np.ndarray,   # shape (T, N)
    returns_matrix: np.ndarray,  # shape (T, N)
    kappa: float = KAPPA,
) -> tuple:
    """
    Compute net daily returns and turnover from a weight time series.

    Returns: (net_returns array (T,), turnover array (T,))
    """
    T = weight_matrix.shape[0]
    net_rets = np.zeros(T)
    turnovers = np.zeros(T)
    prev_w = np.ones(returns_matrix.shape[1]) / returns_matrix.shape[1]

    for t in range(T):
        w = weight_matrix[t]
        r = returns_matrix[t]
        gross = float(w @ r)
        cost = kappa * float(np.abs(w - prev_w).sum())
        net_rets[t] = gross - cost
        turnovers[t] = float(np.abs(w - prev_w).sum())
        prev_w = w

    return net_rets, turnovers


# ---------------------------------------------------------------------------
# Baseline: per-regime best algorithm from precomputed outputs
# ---------------------------------------------------------------------------

def _fit_reflex_oracle(
    dataset: MetaLearnerDataset,
    train_indices: np.ndarray,
    regime_labels: pd.Series,
    kappa: float,
) -> dict:
    """
    Fit a Reflex Agent mapping {regime -> best_algo_idx} using precomputed
    algorithm outputs on the training data.

    Returns
    -------
    dict: {regime_int: k_best} where k_best is the index into algorithms list
    """
    K = dataset.K
    N = dataset.N
    mapping = {}

    for regime in range(1, N_REGIMES + 1):
        # Collect training indices where oracle regime == regime
        regime_idx = [
            idx for idx in train_indices
            if dataset.get_regime(idx) == regime
        ]
        if not regime_idx:
            # Fallback: use the globally best algo (computed below)
            mapping[regime] = None
            continue

        best_k = 0
        best_sharpe = -np.inf

        for k in range(K):
            rets = []
            prev_w = np.ones(N) / N
            for idx in regime_idx:
                w = dataset.get_algorithm_outputs(idx)[k]
                r = dataset.get_returns(idx)
                gross = float(w @ r)
                cost = kappa * float(np.abs(w - prev_w).sum())
                rets.append(gross - cost)
                prev_w = w

            if len(rets) < 10:
                continue
            arr = np.array(rets)
            std = arr.std()
            sr = (arr.mean() / std) * np.sqrt(252) if std > 1e-12 else arr.mean() * np.sqrt(252)
            if sr > best_sharpe:
                best_sharpe = sr
                best_k = k

        mapping[regime] = best_k

    # Fill in None entries with global best
    global_best_k = _best_algo_overall(dataset, train_indices, kappa)
    for regime in range(1, N_REGIMES + 1):
        if mapping.get(regime) is None:
            mapping[regime] = global_best_k

    return mapping


def _best_algo_overall(
    dataset: MetaLearnerDataset,
    train_indices: np.ndarray,
    kappa: float,
) -> int:
    """Return the index of the algorithm with the highest net Sharpe over train_indices."""
    K = dataset.K
    N = dataset.N
    best_k = 0
    best_sharpe = -np.inf

    for k in range(K):
        rets = []
        prev_w = np.ones(N) / N
        for idx in train_indices:
            w = dataset.get_algorithm_outputs(idx)[k]
            r = dataset.get_returns(idx)
            gross = float(w @ r)
            cost = kappa * float(np.abs(w - prev_w).sum())
            rets.append(gross - cost)
            prev_w = w

        if len(rets) < 10:
            continue
        arr = np.array(rets)
        std = arr.std()
        sr = (arr.mean() / std) * np.sqrt(252) if std > 1e-12 else -np.inf
        if sr > best_sharpe:
            best_sharpe = sr
            best_k = k

    return best_k


# ---------------------------------------------------------------------------
# Evaluation: run strategies on test set using precomputed outputs
# ---------------------------------------------------------------------------

def _eval_strategy_on_test(
    dataset: MetaLearnerDataset,
    test_indices: np.ndarray,
    weight_fn,            # callable: idx -> w (shape N)
    kappa: float = KAPPA,
) -> dict:
    """
    Evaluate any strategy on the test set by computing its daily net returns.

    Parameters
    ----------
    weight_fn : callable(idx) -> np.ndarray of shape (N,)
                Returns the portfolio weights at time step idx.
    """
    N = dataset.N
    net_rets = []
    gross_rets = []
    turnovers = []
    prev_w = np.ones(N) / N

    for idx in test_indices:
        w = weight_fn(idx)
        # Safety
        w = np.where(np.isfinite(w), w, 0.0)
        w = np.clip(w, 0.0, None)
        s = w.sum()
        w = w / s if s > 1e-12 else np.ones(N) / N

        r = dataset.get_returns(idx)
        gross = float(w @ r)
        cost = kappa * float(np.abs(w - prev_w).sum())

        net_rets.append(gross - cost)
        gross_rets.append(gross)
        turnovers.append(float(np.abs(w - prev_w).sum()))
        prev_w = w

    arr = np.array(net_rets)
    metrics = _compute_metrics(arr)
    metrics["avg_daily_turnover"] = round(float(np.mean(turnovers)), 6) if turnovers else np.nan
    return metrics, arr


# ---------------------------------------------------------------------------
# Per-fold runner
# ---------------------------------------------------------------------------

def _build_fresh_algorithms():
    """Build a fresh set of 81 algorithm instances (required per fold for Tier 2)."""
    return build_algorithm_space(tiers=[1, 2])


def run_fold(
    fold_id: int,
    fold_spec: dict,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime_labels: pd.Series,
    full_asset_features: pd.DataFrame,
    config: dict,
) -> dict:
    """Run one walk-forward fold and return a results dictionary."""
    sep = "=" * 70
    print(f"\n{sep}", flush=True)
    print(f"FOLD {fold_id}  |  Train {fold_spec['train_start'][:4]}–{fold_spec['train_end'][:4]}"
          f"  |  Test {fold_spec['test_start'][:4]}", flush=True)
    print(sep, flush=True)

    train_start = fold_spec["train_start"]
    train_end   = fold_spec["train_end"]
    test_start  = fold_spec["test_start"]
    test_end    = fold_spec["test_end"]
    kappa       = config["kappa"]

    # ------------------------------------------------------------------ #
    # STAGE 0 — build + pre-train algorithms                              #
    # ------------------------------------------------------------------ #
    t0 = time.time()
    algorithms = _build_fresh_algorithms()
    K = len(algorithms)
    print(f"  Built K={K} algorithms", flush=True)

    has_tier2 = any(isinstance(a, TrainablePortfolioAlgorithm) for a in algorithms)
    if has_tier2:
        pretrain_tier2_algorithms(
            algorithms, full_asset_features, returns, train_start, train_end
        )

    # ------------------------------------------------------------------ #
    # DATASET — assemble and precompute algo outputs                      #
    # ------------------------------------------------------------------ #
    print("  Building MetaLearnerDataset ...", flush=True)
    dataset = MetaLearnerDataset(
        prices=prices,
        all_asset_features=full_asset_features,
        returns=returns,
        regime_labels=regime_labels,
        algorithms=algorithms,
    )

    # Fit StandardScaler on training features only
    dataset.fit_scaler(train_start, train_end)

    # Precompute algorithm outputs for ALL valid dates (train + test)
    print("  Precomputing algorithm outputs (this is the slow step) ...", flush=True)
    dataset.precompute_algo_outputs()

    # Integer index sets
    train_idx = dataset.get_indices_for_period(train_start, train_end)
    test_idx  = dataset.get_indices_for_period(test_start,  test_end)
    print(f"  Train: {len(train_idx)} days | Test: {len(test_idx)} days",
          flush=True)

    if len(train_idx) == 0 or len(test_idx) == 0:
        print("  WARNING: empty train or test set — skipping fold.", flush=True)
        return None

    t1 = time.time()
    print(f"  Stage 0 + precompute done in {t1 - t0:.1f}s", flush=True)

    # ------------------------------------------------------------------ #
    # BASELINES — computed from precomputed algo outputs                  #
    # ------------------------------------------------------------------ #
    print("  Fitting Reflex Oracle baseline ...", flush=True)
    reflex_mapping = _fit_reflex_oracle(dataset, train_idx, regime_labels, kappa)
    best_k_overall = _best_algo_overall(dataset, train_idx, kappa)

    # Equal Weight: algo index 0 = EqualWeight (always first in build_algorithm_space)
    ew_k = next(
        (k for k, a in enumerate(algorithms) if a.name == "EqualWeight"), 0
    )

    def _ew_fn(idx):
        return dataset.get_algorithm_outputs(idx)[ew_k]

    def _reflex_oracle_fn(idx):
        regime = dataset.get_regime(idx)
        k = reflex_mapping.get(regime, ew_k)
        return dataset.get_algorithm_outputs(idx)[k]

    def _best_algo_fn(idx):
        return dataset.get_algorithm_outputs(idx)[best_k_overall]

    def _random_ml_fn(idx):
        # Sanity check: random uniform alpha
        alpha = np.random.dirichlet(np.ones(K))
        W = dataset.get_algorithm_outputs(idx)  # (K, N)
        w = alpha @ W
        return w

    m_ew,     rets_ew     = _eval_strategy_on_test(dataset, test_idx, _ew_fn,     kappa)
    m_reflex, rets_reflex = _eval_strategy_on_test(dataset, test_idx, _reflex_oracle_fn, kappa)
    m_best,   rets_best   = _eval_strategy_on_test(dataset, test_idx, _best_algo_fn, kappa)
    m_random, rets_random = _eval_strategy_on_test(dataset, test_idx, _random_ml_fn, kappa)

    print(f"  EW Sharpe          : {m_ew['sharpe']:+.4f}", flush=True)
    print(f"  Reflex Oracle Sharpe: {m_reflex['sharpe']:+.4f}", flush=True)
    print(f"  Best Algo Sharpe   : {m_best['sharpe']:+.4f}", flush=True)
    print(f"  Random ML Sharpe   : {m_random['sharpe']:+.4f}", flush=True)

    # ------------------------------------------------------------------ #
    # STAGE 2 — train meta-learner                                        #
    # ------------------------------------------------------------------ #
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    network = MetaLearnerNetwork(
        input_dim=config["input_dim"],
        n_algorithms=K,
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
    )
    trainer = MetaLearnerTrainer(
        network=network,
        kappa=config["kappa"],
        kappa_a=config["kappa_a"],
        lambda_entropy=config["lambda_entropy"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        n_epochs=config["n_epochs"],
        grad_clip=config["grad_clip"],
    )

    print(f"  Training MetaLearner ({config['n_epochs']} epochs) ...", flush=True)
    t2 = time.time()
    training_history = trainer.train_fold(dataset, train_idx)
    t3 = time.time()
    print(f"  Training done in {t3 - t2:.1f}s", flush=True)

    # ------------------------------------------------------------------ #
    # EVALUATE meta-learner on test set                                   #
    # ------------------------------------------------------------------ #
    agent = MetaLearnerAgent(network=network, algorithms=algorithms)

    # Collect alpha_t and w_t for each test day
    test_dates   = dataset.dates[test_idx]
    alpha_matrix = np.zeros((len(test_idx), K), dtype=np.float32)
    w_matrix     = np.zeros((len(test_idx), N_ASSETS), dtype=np.float32)

    for i, idx in enumerate(test_idx):
        x_t   = dataset.get_input(idx)
        W_t   = dataset.get_algorithm_outputs(idx)
        w_t, alpha_t = agent.select(x_t, W_t)
        alpha_matrix[i] = alpha_t
        w_matrix[i]     = w_t

    # Compute metrics from the weight matrix
    returns_matrix = np.array([dataset.get_returns(idx) for idx in test_idx])
    net_rets_ml, turnovers_ml = _portfolio_sharpe_from_weights(
        w_matrix, returns_matrix, kappa
    )
    m_ml = _compute_metrics(net_rets_ml)
    m_ml["avg_daily_turnover"] = round(float(turnovers_ml.mean()), 6)

    # Entropy per day
    entropy_series = np.array([
        MetaLearnerAgent.entropy(alpha_matrix[i]) for i in range(len(test_idx))
    ])

    print(f"  MetaLearner Sharpe : {m_ml['sharpe']:+.4f}", flush=True)
    print(f"  MetaLearner AvgH   : {entropy_series.mean():.4f}", flush=True)

    # ------------------------------------------------------------------ #
    # SANITY CHECKS                                                       #
    # ------------------------------------------------------------------ #
    # Check 4: composite portfolio validity
    w_sum = w_matrix.sum(axis=1)
    if not np.allclose(w_sum, 1.0, atol=1e-4):
        print(f"  WARNING: portfolio weights don't sum to 1 (mean={w_sum.mean():.4f})")

    # Check 3: are weights near EW?
    w_mean = w_matrix.mean(axis=0)
    if np.allclose(w_mean, 1 / N_ASSETS, atol=0.01):
        print("  NOTE: MetaLearner converged near Equal-Weight (may indicate weak signal)")

    return {
        "fold_id":          fold_id,
        "fold_spec":        fold_spec,
        # Metrics
        "metrics_ew":       m_ew,
        "metrics_reflex":   m_reflex,
        "metrics_best":     m_best,
        "metrics_random":   m_random,
        "metrics_ml":       m_ml,
        # Time series (for plots)
        "test_dates":       test_dates,
        "alpha_matrix":     alpha_matrix,   # (T_test, K)
        "w_matrix":         w_matrix,       # (T_test, N)
        "net_rets_ml":      net_rets_ml,    # (T_test,)
        "entropy_series":   entropy_series, # (T_test,)
        "reflex_mapping":   {r: algorithms[k].name for r, k in reflex_mapping.items()},
        "best_algo_name":   algorithms[best_k_overall].name,
        "training_history": training_history,
        # Algorithm metadata
        "algo_names":       [a.name for a in algorithms],
        "algo_families":    [a.family for a in algorithms],
    }


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _build_summary_df(fold_results: list) -> pd.DataFrame:
    rows = []
    for fr in fold_results:
        if fr is None:
            continue
        rows.append({
            "fold":         fr["fold_id"],
            "test_year":    fr["fold_spec"]["test_start"][:4],
            "ew_sharpe":    fr["metrics_ew"]["sharpe"],
            "reflex_sharpe": fr["metrics_reflex"]["sharpe"],
            "best_sharpe":  fr["metrics_best"]["sharpe"],
            "ml_sharpe":    fr["metrics_ml"]["sharpe"],
            "ml_sortino":   fr["metrics_ml"]["sortino"],
            "ml_maxdd":     fr["metrics_ml"]["max_drawdown"],
            "ml_turnover":  fr["metrics_ml"].get("avg_daily_turnover", np.nan),
            "ml_entropy":   float(fr["entropy_series"].mean()),
            "ml_vs_ew":     fr["metrics_ml"]["sharpe"] - fr["metrics_ew"]["sharpe"],
            "ml_vs_reflex": fr["metrics_ml"]["sharpe"] - fr["metrics_reflex"]["sharpe"],
            "best_algo":    fr["best_algo_name"],
        })
    return pd.DataFrame(rows).set_index("fold")


def _print_summary(df: pd.DataFrame) -> None:
    header = (
        f"{'Fold':>5}  {'Year':>5}  {'EW':>8}  {'Reflex':>8}  "
        f"{'Best':>8}  {'ML':>8}  {'ML-EW':>7}  {'ML-Ref':>7}  {'H(a)':>7}"
    )
    sep = "-" * len(header)
    print("\n" + "=" * 70)
    print("PLAN 5: META-LEARNER RESULTS SUMMARY")
    print("=" * 70)
    print(header)
    print(sep)
    for fold_id, row in df.iterrows():
        print(
            f"{fold_id:>5}  {row['test_year']:>5}  "
            f"{row['ew_sharpe']:>8.4f}  {row['reflex_sharpe']:>8.4f}  "
            f"{row['best_sharpe']:>8.4f}  {row['ml_sharpe']:>8.4f}  "
            f"{row['ml_vs_ew']:>7.4f}  {row['ml_vs_reflex']:>7.4f}  "
            f"{row['ml_entropy']:>7.4f}"
        )
    print(sep)
    nums = df[["ew_sharpe", "reflex_sharpe", "best_sharpe", "ml_sharpe",
               "ml_vs_ew", "ml_vs_reflex", "ml_entropy"]]
    avgs = nums.mean()
    print(
        f"{'AVG':>5}  {'':>5}  "
        f"{avgs['ew_sharpe']:>8.4f}  {avgs['reflex_sharpe']:>8.4f}  "
        f"{avgs['best_sharpe']:>8.4f}  {avgs['ml_sharpe']:>8.4f}  "
        f"{avgs['ml_vs_ew']:>7.4f}  {avgs['ml_vs_reflex']:>7.4f}  "
        f"{avgs['ml_entropy']:>7.4f}"
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_training_convergence(fold_results: list, save_dir: str) -> None:
    """Plot 1: Training reward curves per fold."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(fold_results)))

    for fr, col in zip(fold_results, colors):
        if fr is None:
            continue
        hist = fr["training_history"]["epoch_reward"]
        label = fr["fold_spec"]["test_start"][:4]
        ax.plot(range(1, len(hist) + 1), hist, color=col, alpha=0.8, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Reward per Day")
    ax.set_title("Plan 12: Meta-Learner Training Convergence (all folds)")
    ax.legend(title="Test Year", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "plot1_training_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_sharpe_comparison(summary_df: pd.DataFrame, save_dir: str) -> None:
    """Plot 2: Sharpe ratio comparison bar chart per fold."""
    strategies = ["ew_sharpe", "reflex_sharpe", "best_sharpe", "ml_sharpe"]
    labels = ["Equal Weight", "Reflex (Oracle)", "Best Algo (Oracle)", "MetaLearner (Oracle)"]
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728"]

    x = np.arange(len(summary_df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (col, label, color) in enumerate(zip(strategies, labels, colors)):
        vals = summary_df[col].values
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["test_year"].values, rotation=45)
    ax.set_xlabel("Test Year")
    ax.set_ylabel("Net Sharpe Ratio")
    ax.set_title("Plan 12: Sharpe Ratio Comparison per Walk-Forward Fold")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "plot2_sharpe_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_mixing_weights(fold_results: list, save_dir: str, n_folds_to_show: int = 2) -> None:
    """Plot 3: Stacked area chart of alpha_t for selected test years."""
    # Pick the first n_folds_to_show non-None results
    selected = [fr for fr in fold_results if fr is not None][:n_folds_to_show]

    for fr in selected:
        test_year = fr["fold_spec"]["test_start"][:4]
        alpha_mat = fr["alpha_matrix"]   # (T, K)
        dates     = fr["test_dates"]
        families  = fr["algo_families"]

        # Group alpha by family
        unique_families = list(dict.fromkeys(families))  # order-preserving unique
        family_alpha = np.zeros((len(dates), len(unique_families)), dtype=np.float32)
        for fi, fam in enumerate(unique_families):
            idxs = [k for k, f in enumerate(families) if f == fam]
            family_alpha[:, fi] = alpha_mat[:, idxs].sum(axis=1)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 1]})

        # Stacked area
        cmap = plt.cm.tab10
        ax = axes[0]
        cum = np.zeros(len(dates))
        for fi, fam in enumerate(unique_families):
            color = cmap(fi % 10)
            ax.fill_between(dates, cum, cum + family_alpha[:, fi],
                            alpha=0.75, label=fam, color=color)
            cum = cum + family_alpha[:, fi]
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mixing Weight α_t")
        ax.set_title(f"Plan 12: Meta-Learner Mixing Weights — Test {test_year}")
        ax.legend(title="Algorithm Family", bbox_to_anchor=(1.01, 1),
                  loc="upper left", fontsize=8)

        # Entropy
        ax2 = axes[1]
        ax2.plot(dates, fr["entropy_series"], color="purple", linewidth=0.8)
        ax2.set_ylabel("H(α_t)")
        ax2.set_title("Algorithm Entropy")
        ax2.grid(True, alpha=0.3)

        # Regime labels
        ax3 = axes[2]
        regime_colors = {1: "#2ecc71", 2: "#3498db", 3: "#e67e22", 4: "#e74c3c"}
        regime_arr = np.array([fr["test_dates"]])  # placeholder
        # Re-extract regime for each date
        # (stored in dataset but we need them here — extract from alpha entropy context)
        # We'll use the VIX regime: not stored here, skip the colour map
        # Just indicate the date range label
        ax3.set_ylabel("(regime overlay\nnot stored in fold)")
        ax3.set_visible(False)

        plt.tight_layout()
        path = os.path.join(save_dir, f"plot3_mixing_weights_{test_year}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def _plot_entropy_over_time(fold_results: list, save_dir: str) -> None:
    """Plot 4: Algorithm entropy H(alpha_t) over all test periods."""
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = plt.cm.tab20(np.linspace(0, 1, len(fold_results)))

    for fr, col in zip(fold_results, colors):
        if fr is None:
            continue
        label = fr["fold_spec"]["test_start"][:4]
        ax.plot(fr["test_dates"], fr["entropy_series"], color=col, alpha=0.7,
                linewidth=0.8, label=label)

    # Reference: max entropy = log(K)
    if fold_results and fold_results[0] is not None:
        K = fold_results[0]["alpha_matrix"].shape[1]
        ax.axhline(np.log(K), color="black", linestyle="--", linewidth=0.8,
                   label=f"Max entropy log(K)={np.log(K):.2f}")

    ax.set_xlabel("Date")
    ax.set_ylabel("H(α_t)")
    ax.set_title("Plan 12: Algorithm Selection Entropy over Time")
    ax.legend(title="Test Year", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "plot4_entropy_over_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_regime_heatmap(fold_results: list, save_dir: str) -> None:
    """Plot 5: Heatmap — average alpha by regime and algorithm family."""
    # Aggregate across all folds
    regime_family_sums = {r: {} for r in range(1, 5)}
    regime_family_counts = {r: 0 for r in range(1, 5)}

    for fr in fold_results:
        if fr is None:
            continue
        alpha_mat = fr["alpha_matrix"]
        families  = fr["algo_families"]

        # We don't have per-day regime stored in fold_result — we need to
        # look up from regime_labels. As a fallback, aggregate without regime
        # (just plot overall family distribution)

    # --- Simplified version: overall average alpha per family across all folds ---
    family_to_mean = {}
    for fr in fold_results:
        if fr is None:
            continue
        alpha_mat = fr["alpha_matrix"]  # (T, K)
        families  = fr["algo_families"]
        unique_fams = list(dict.fromkeys(families))
        for fam in unique_fams:
            idxs = [k for k, f in enumerate(families) if f == fam]
            mean_alpha = alpha_mat[:, idxs].sum(axis=1).mean()
            if fam not in family_to_mean:
                family_to_mean[fam] = []
            family_to_mean[fam].append(mean_alpha)

    fam_names = sorted(family_to_mean.keys())
    fam_means = [np.mean(family_to_mean[f]) for f in fam_names]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(fam_names, fam_means, color=plt.cm.tab10(np.arange(len(fam_names)) % 10))
    ax.set_xlabel("Algorithm Family")
    ax.set_ylabel("Mean Mixing Weight (all folds + dates)")
    ax.set_title("Plan 12: Average Algorithm Family Selection by Meta-Learner")
    ax.set_xticklabels(fam_names, rotation=45, ha="right")
    ax.axhline(1.0 / len(fam_names), color="red", linestyle="--",
               linewidth=0.8, label="Uniform baseline")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "plot5_family_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_regime_conditional_heatmap(
    fold_results: list,
    regime_labels: pd.Series,
    save_dir: str,
) -> None:
    """Plot 5b: Heatmap alpha by regime x family (requires regime_labels)."""
    # Accumulate per regime
    regime_family_alpha = {r: {} for r in range(1, 5)}

    for fr in fold_results:
        if fr is None:
            continue
        alpha_mat = fr["alpha_matrix"]   # (T_test, K)
        families  = fr["algo_families"]
        test_dates = fr["test_dates"]
        unique_fams = list(dict.fromkeys(families))

        for ti, t in enumerate(test_dates):
            if t not in regime_labels.index:
                continue
            regime = int(regime_labels.loc[t])
            for fam in unique_fams:
                idxs = [k for k, f in enumerate(families) if f == fam]
                val = float(alpha_mat[ti, idxs].sum())
                if fam not in regime_family_alpha[regime]:
                    regime_family_alpha[regime][fam] = []
                regime_family_alpha[regime][fam].append(val)

    # Build matrix
    all_fams = sorted({
        f for rd in regime_family_alpha.values() for f in rd.keys()
    })
    n_regimes = 4
    mat = np.zeros((n_regimes, len(all_fams)))
    for ri, regime in enumerate(range(1, 5)):
        for fi, fam in enumerate(all_fams):
            vals = regime_family_alpha[regime].get(fam, [0.0])
            mat[ri, fi] = np.mean(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(all_fams)))
    ax.set_xticklabels(all_fams, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels([REGIME_NAMES[r] for r in range(1, 5)])
    ax.set_title("Plan 12: Regime-Conditional Average Alpha by Algorithm Family")
    plt.colorbar(im, ax=ax, label="Mean α")

    # Annotate cells
    for ri in range(n_regimes):
        for fi in range(len(all_fams)):
            ax.text(fi, ri, f"{mat[ri, fi]:.3f}", ha="center", va="center",
                    fontsize=7, color="black")

    plt.tight_layout()
    path = os.path.join(save_dir, "plot5b_regime_family_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(config: dict = CONFIG) -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for Plan 12. Install with: pip install torch"
        )

    print("\n" + "=" * 70)
    print("PLAN 12: META-LEARNER WITH ORACLE REGIME")
    print("=" * 70)
    print(f"Config: {config}", flush=True)

    # ------------------------------------------------------------------ #
    # Load data                                                           #
    # ------------------------------------------------------------------ #
    print("\nLoading data ...", flush=True)
    data = load_data_extended()
    prices = data["prices"]
    vix    = data["vix"]
    print(f"  Prices: {prices.shape}  ({prices.index[0].date()} – {prices.index[-1].date()})")
    print(f"  VIX:    {len(vix)} rows")

    # Forward returns and features
    returns             = compute_returns(prices)
    full_asset_features = compute_asset_features(prices)   # MultiIndex (asset, feature)
    regime_labels       = compute_regime_labels(vix)       # Series {1,2,3,4}

    # Align all series to common dates
    common = prices.index.intersection(returns.index).intersection(regime_labels.index)
    prices              = prices.loc[common]
    returns             = returns.loc[common]
    regime_labels       = regime_labels.loc[common]
    full_asset_features = full_asset_features.loc[full_asset_features.index.intersection(common)]

    print(f"  Common dates: {len(common)}")
    print(f"  Regime dist: { {REGIME_NAMES[r]: int((regime_labels == r).sum()) for r in range(1, 5)} }")

    # ------------------------------------------------------------------ #
    # Walk-forward folds                                                  #
    # ------------------------------------------------------------------ #
    wfv = WalkForwardValidator(
        train_years=config["train_years"],
        test_years=config["test_years"],
        step_years=config["step_years"],
        min_test_start=config["min_test_start"],
    )
    folds = wfv.generate_folds(data_end=config["data_end"])
    print(f"\nGenerated {len(folds)} walk-forward folds")

    # ------------------------------------------------------------------ #
    # Run each fold                                                       #
    # ------------------------------------------------------------------ #
    fold_results = []
    for fold_spec in folds:
        fold_id = fold_spec["fold"]
        t_start = time.time()
        result = run_fold(
            fold_id=fold_id,
            fold_spec=fold_spec,
            prices=prices,
            returns=returns,
            regime_labels=regime_labels,
            full_asset_features=full_asset_features,
            config=config,
        )
        t_end = time.time()
        print(f"  Fold {fold_id} done in {(t_end - t_start) / 60:.1f} min", flush=True)
        fold_results.append(result)

    # ------------------------------------------------------------------ #
    # Aggregate and display results                                       #
    # ------------------------------------------------------------------ #
    valid_folds = [fr for fr in fold_results if fr is not None]
    if not valid_folds:
        print("ERROR: all folds failed.")
        return

    summary_df = _build_summary_df(valid_folds)
    _print_summary(summary_df)

    # Save CSV
    csv_path = os.path.join(PLAN5_RESULTS_DIR, "summary_metrics.csv")
    summary_df.to_csv(csv_path)
    print(f"\n  Saved summary CSV: {csv_path}")

    # ------------------------------------------------------------------ #
    # Plots                                                               #
    # ------------------------------------------------------------------ #
    print("\nGenerating plots ...", flush=True)
    _plot_training_convergence(valid_folds, PLAN5_RESULTS_DIR)
    _plot_sharpe_comparison(summary_df, PLAN5_RESULTS_DIR)
    _plot_mixing_weights(valid_folds, PLAN5_RESULTS_DIR, n_folds_to_show=2)
    _plot_entropy_over_time(valid_folds, PLAN5_RESULTS_DIR)
    _plot_regime_heatmap(valid_folds, PLAN5_RESULTS_DIR)
    _plot_regime_conditional_heatmap(valid_folds, regime_labels, PLAN5_RESULTS_DIR)

    print(f"\nAll plots saved to: {PLAN5_RESULTS_DIR}")
    print("\nPLAN 12 COMPLETE.")


if __name__ == "__main__":
    run_experiment()
