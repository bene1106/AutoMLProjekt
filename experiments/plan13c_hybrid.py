# experiments/plan13c_hybrid.py -- Plan 13c: Hybrid Meta-Learner + BO Within-Tier
#
# Architecture:
#   Level 1: TierSelector NN (from Plan 13a) — β_t ∈ Δ_3 (tier mixing weights)
#   Level 2: BO exhaustive search within each tier per regime
#            → best_algo[regime][tier]: lookup table (no Specialist NNs)
#   Combination: w_t = Σ_f β_{t,f} · w^(best_algo[regime][f])_t
#
# Reuses:
#   - results/plan13a_hierarchical/cache/fold_XX_algo_outputs.npy
#   - meta_learner/hierarchical_network.py (TierSelector)
#   - meta_learner/dataset.py (MetaLearnerDataset)
#   - Same walk-forward folds as Plans 13a/13b
#
# Three 13c strategies:
#   A) Hybrid: ML TierSelector + BO within-tier
#   B) BO-Only Per-Tier: tier weights ∝ in-sample Sharpe (no NN)
#   C) Best-Tier-Only: always pick tier with highest in-sample Sharpe per regime
#
# Prints final comparison table across all Plans (13a, 13b, 13c) + EW.
#
# Usage:
#   cd Implementierung1
#   python -u -m experiments.plan13c_hybrid

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. Hybrid strategy (A) will be skipped.", flush=True)

from regime_algo_selection.config import (
    RESULTS_DIR, KAPPA, REGIME_NAMES, N_ASSETS, RANDOM_SEED,
)
from regime_algo_selection.data.loader_extended import load_data_extended
from regime_algo_selection.data.features import compute_returns, compute_asset_features
from regime_algo_selection.regimes.ground_truth import compute_regime_labels
from regime_algo_selection.algorithms.tier1_heuristics import build_algorithm_space
from regime_algo_selection.algorithms.base import TrainablePortfolioAlgorithm
from regime_algo_selection.algorithms.stage0 import pretrain_algorithms
from regime_algo_selection.evaluation.walk_forward import WalkForwardValidator
from regime_algo_selection.meta_learner.dataset import MetaLearnerDataset

if TORCH_AVAILABLE:
    from regime_algo_selection.meta_learner.hierarchical_network import TierSelector

try:
    from regime_algo_selection.algorithms.tier3_nonlinear import Tier3Algorithm
    HAS_TIER3 = True
except ImportError:
    HAS_TIER3 = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # TierSelector (ML component — identical to Plan 13a Phase B)
    "input_dim": 29,            # 25 asset features + 4 regime one-hot
    "selector_hidden": [64, 32],
    "dropout": 0.1,
    "selector_lr": 0.005,
    "selector_epochs": 50,
    "lambda_tier": 0.05,
    "weight_decay": 1e-4,

    # BO (within-tier exhaustive search)
    "kappa": KAPPA,             # 0.001

    # Walk-forward (identical to Plans 13a/13b)
    "train_years": 8,
    "test_years": 1,
    "step_years": 1,
    "min_test_start": "2013-01-01",
    "data_end": "2024-12-31",

    # Algorithm space
    "tiers": [1, 2, 3],

    # Cache (reuse Plan 13a outputs)
    "cache_dir": os.path.join(RESULTS_DIR, "plan13a_hierarchical", "cache"),

    # Prior results for comparison table
    "plan13a_summary": os.path.join(RESULTS_DIR, "plan13a_hierarchical", "summary_metrics.csv"),
    "plan13b_summary": os.path.join(RESULTS_DIR, "plan13b_bayesian_opt", "summary_metrics.csv"),

    # Output
    "output_dir": os.path.join(RESULTS_DIR, "plan13c_hybrid"),

    # Random seed
    "seed": RANDOM_SEED,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


# ---------------------------------------------------------------------------
# Metric helpers (identical to Plans 13a/13b)
# ---------------------------------------------------------------------------

def _compute_metrics(daily_net_returns: np.ndarray) -> dict:
    r = daily_net_returns[np.isfinite(daily_net_returns)]
    T = len(r)
    if T < 10:
        return {k: np.nan for k in [
            "sharpe", "sortino", "ann_return", "ann_vol", "max_drawdown", "n_days",
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
        "sharpe":       round(float(sharpe),   4),
        "sortino":      round(float(sortino),  4),
        "ann_return":   round(ann_ret * 100,   2),
        "ann_vol":      round(ann_vol * 100,   2),
        "max_drawdown": round(max_dd  * 100,   2),
        "n_days":       T,
    }


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-10, 1.0)
    return float(-np.sum(p * np.log(p)))


def _eval_strategy(
    weight_fn,
    returns_arr: np.ndarray,
    test_indices: np.ndarray,
    kappa: float,
    N: int,
) -> tuple:
    prev_w = np.ones(N) / N
    net_rets = []
    turnovers = []
    for i in test_indices:
        w = weight_fn(i)
        w = np.where(np.isfinite(w), w, 0.0)
        w = np.clip(w, 0.0, None)
        s = w.sum()
        w = w / s if s > 1e-12 else np.ones(N) / N
        r = returns_arr[i]
        gross = float(w @ r)
        cost = kappa * float(np.abs(w - prev_w).sum())
        net_rets.append(gross - cost)
        turnovers.append(float(np.abs(w - prev_w).sum()))
        prev_w = w
    arr = np.array(net_rets)
    m = _compute_metrics(arr)
    m["avg_daily_turnover"] = round(float(np.mean(turnovers)), 6) if turnovers else np.nan
    return m, arr


# ---------------------------------------------------------------------------
# BO within-tier: per-regime, per-tier exhaustive selection
# ---------------------------------------------------------------------------

def bo_within_tier(
    algo_outputs: np.ndarray,
    returns_arr: np.ndarray,
    regime_arr: np.ndarray,
    train_mask: np.ndarray,
    tier_algorithm_indices: list,
    kappa: float,
) -> tuple:
    """
    For each regime × tier combination, evaluate all algorithms in that tier
    on training data and select the one with the best net Sharpe.

    Returns
    -------
    best_algo_per_regime_tier : dict {regime_id: {tier_id: global_algo_index}}
    tier_sharpe_per_regime    : dict {regime_id: {tier_id: float}}
    """
    best = {}
    tier_sharpe = {}

    for regime_id in [1, 2, 3, 4]:
        regime_mask = train_mask & (regime_arr == regime_id)
        regime_days = np.where(regime_mask)[0]
        T_reg = len(regime_days)

        best[regime_id] = {}
        tier_sharpe[regime_id] = {}

        for tier_id, t_indices in enumerate(tier_algorithm_indices):
            if T_reg < 10:
                best[regime_id][tier_id] = t_indices[0]
                tier_sharpe[regime_id][tier_id] = 0.0
                continue

            # Vectorised across all algorithms in this tier
            w_reg = algo_outputs[regime_days][:, t_indices, :]   # (T_reg, K_f, N)
            r_reg = returns_arr[regime_days]                      # (T_reg, N)
            port_ret = np.einsum("tkn,tn->tk", w_reg, r_reg)     # (T_reg, K_f)

            if T_reg > 1:
                turnover = np.sum(np.abs(np.diff(w_reg, axis=0)), axis=2)  # (T_reg-1, K_f)
                costs = kappa * turnover
                net_ret = port_ret[1:] - costs                   # (T_reg-1, K_f)
            else:
                net_ret = port_ret

            if net_ret.shape[0] < 5:
                best[regime_id][tier_id] = t_indices[0]
                tier_sharpe[regime_id][tier_id] = 0.0
                continue

            mean_r = net_ret.mean(axis=0)   # (K_f,)
            std_r  = net_ret.std(axis=0)    # (K_f,)
            valid  = std_r > 1e-10
            sharpes = np.where(valid, mean_r / std_r * np.sqrt(252), 0.0)

            best_local = int(np.argmax(sharpes))
            best[regime_id][tier_id] = t_indices[best_local]
            tier_sharpe[regime_id][tier_id] = float(sharpes[best_local])

    return best, tier_sharpe


# ---------------------------------------------------------------------------
# Strategy B: BO-Only Per-Tier (Sharpe-weighted tier blend, no NN)
# ---------------------------------------------------------------------------

def evaluate_bo_per_tier(
    best_algo_per_regime_tier: dict,
    tier_sharpe_per_regime: dict,
    algo_outputs: np.ndarray,
    returns_arr: np.ndarray,
    regime_arr: np.ndarray,
    test_indices: np.ndarray,
    kappa: float,
    N: int,
) -> tuple:
    """
    Blend 3 tier-level portfolios, weighted proportional to max(0, sharpe_f).
    No neural network — purely Sharpe-weighted tier combination.
    """
    def _fn(i):
        regime = int(regime_arr[i])
        tier_info = tier_sharpe_per_regime.get(regime, {})
        raw_weights = np.array([
            max(0.0, tier_info.get(f, 0.0)) for f in range(3)
        ], dtype=np.float64)
        total = raw_weights.sum()
        if total < 1e-12:
            # All negative Sharpes: fall back to equal tier weights
            raw_weights = np.ones(3) / 3.0
        else:
            raw_weights /= total

        w_blend = np.zeros(N, dtype=np.float64)
        for tier_id in range(3):
            k = best_algo_per_regime_tier[regime][tier_id]
            w_blend += raw_weights[tier_id] * algo_outputs[i, k, :]
        return w_blend

    return _eval_strategy(_fn, returns_arr, test_indices, kappa, N)


# ---------------------------------------------------------------------------
# Strategy C: Best-Tier-Only (hard tier + hard within-tier)
# ---------------------------------------------------------------------------

def evaluate_best_tier_only(
    best_algo_per_regime_tier: dict,
    tier_sharpe_per_regime: dict,
    algo_outputs: np.ndarray,
    returns_arr: np.ndarray,
    regime_arr: np.ndarray,
    test_indices: np.ndarray,
    kappa: float,
    N: int,
) -> tuple:
    """
    For each regime, always use the single tier with the best in-sample Sharpe,
    and the single best algo within that tier.
    Hard tier selection + hard within-tier selection = one algorithm per regime.
    """
    # Pre-compute best tier per regime
    best_tier_per_regime = {}
    for regime_id in [1, 2, 3, 4]:
        tier_info = tier_sharpe_per_regime.get(regime_id, {})
        best_t = max(range(3), key=lambda f: tier_info.get(f, 0.0))
        best_tier_per_regime[regime_id] = best_t

    def _fn(i):
        regime = int(regime_arr[i])
        best_tier = best_tier_per_regime.get(regime, 0)
        k = best_algo_per_regime_tier[regime][best_tier]
        return algo_outputs[i, k, :]

    return _eval_strategy(_fn, returns_arr, test_indices, kappa, N)


# ---------------------------------------------------------------------------
# Strategy A: Hybrid ML TierSelector + BO within-tier
# ---------------------------------------------------------------------------

def train_tier_selector(
    tier_selector: "TierSelector",
    best_algo_per_regime_tier: dict,
    algo_outputs: np.ndarray,
    returns_arr: np.ndarray,
    regime_arr: np.ndarray,
    dataset: MetaLearnerDataset,
    train_indices: np.ndarray,
    config: dict,
) -> "TierSelector":
    """
    Train TierSelector NN to blend 3 BO-selected tier portfolios.

    Unlike Plan 13a Phase B, there are no Specialist NNs: each tier contributes
    exactly one concrete portfolio (the BO-best for the current regime and tier).
    This gives cleaner gradient signal.

    Loss per step: -(portfolio_return × 252 + lambda_tier × H(β_t))
    """
    kappa = config["kappa"]
    lambda_tier = config["lambda_tier"]

    optimizer = torch.optim.Adam(
        tier_selector.parameters(),
        lr=config["selector_lr"],
        weight_decay=config["weight_decay"],
    )

    tier_selector.train()
    N = algo_outputs.shape[2]

    for epoch in range(config["selector_epochs"]):
        w_prev = torch.ones(N, dtype=torch.float32) / N
        epoch_reward = 0.0

        for idx in train_indices:
            regime = int(regime_arr[idx])

            # Get each tier's best algo portfolio for this regime
            tier_portfolios = []
            for tier_id in range(3):
                k = best_algo_per_regime_tier[regime][tier_id]
                w_tier = torch.tensor(algo_outputs[idx, k, :], dtype=torch.float32)
                tier_portfolios.append(w_tier)
            tier_portfolios = torch.stack(tier_portfolios)   # (3, N)

            # TierSelector → β_t
            X_t = torch.tensor(dataset.get_input(idx), dtype=torch.float32)
            beta_t = tier_selector(X_t)                      # (3,)

            # Blended portfolio
            w_t = torch.matmul(beta_t, tier_portfolios)      # (N,)

            # Reward with switching cost
            r_next = torch.tensor(returns_arr[idx], dtype=torch.float32)
            portfolio_ret = torch.dot(w_t, r_next)
            port_cost = kappa * torch.sum(torch.abs(w_t - w_prev))
            reward = portfolio_ret - port_cost

            # Entropy regularisation on β
            entropy = -torch.sum(beta_t * torch.log(beta_t + 1e-10))
            loss = -(reward * 252.0 + lambda_tier * entropy)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tier_selector.parameters(), 1.0)
            optimizer.step()

            w_prev = w_t.detach()
            epoch_reward += reward.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"    Epoch {epoch+1:3d}/{config['selector_epochs']}: "
                f"avg_reward={epoch_reward / max(len(train_indices), 1):.6f}",
                flush=True,
            )

    tier_selector.eval()
    return tier_selector


def evaluate_hybrid(
    tier_selector: "TierSelector",
    best_algo_per_regime_tier: dict,
    algo_outputs: np.ndarray,
    returns_arr: np.ndarray,
    regime_arr: np.ndarray,
    dataset: MetaLearnerDataset,
    test_indices: np.ndarray,
    kappa: float,
    N: int,
) -> tuple:
    """
    At each test day:
    1. Look up regime → get best algo per tier (from BO lookup)
    2. Run TierSelector NN → β_t (tier weights)
    3. Blend: w_t = Σ_f β_{t,f} · w^(best_algo[regime][f])_t
    """
    beta_records = []

    def _fn(i):
        regime = int(regime_arr[i])

        tier_portfolios = np.array([
            algo_outputs[i, best_algo_per_regime_tier[regime][f], :]
            for f in range(3)
        ], dtype=np.float32)                                  # (3, N)

        X_t = torch.tensor(dataset.get_input(i), dtype=torch.float32)
        with torch.no_grad():
            beta_t = tier_selector(X_t).numpy()              # (3,)
        beta_records.append((i, regime, beta_t.copy()))

        w_blend = beta_t @ tier_portfolios                    # (N,)
        return w_blend

    m, arr = _eval_strategy(_fn, returns_arr, test_indices, kappa, N)
    return m, arr, beta_records


# ---------------------------------------------------------------------------
# Baselines reused from Plan 13b logic
# ---------------------------------------------------------------------------

def evaluate_ew(algo_outputs, returns_arr, test_indices, kappa, N, ew_k):
    def _fn(i):
        return algo_outputs[i, ew_k, :]
    return _eval_strategy(_fn, returns_arr, test_indices, kappa, N)


def evaluate_bo_hard_selection(
    algo_outputs, returns_arr, regime_arr, train_mask, test_indices, kappa, N, algo_names
):
    """Plan 13b Hard Selection: best algo per regime (across all tiers)."""
    K = algo_outputs.shape[1]
    best_algo = {}
    for regime_id in [1, 2, 3, 4]:
        rm = train_mask & (regime_arr == regime_id)
        regime_days = np.where(rm)[0]
        if len(regime_days) < 10:
            best_algo[regime_id] = 0
            continue
        w_reg = algo_outputs[regime_days]
        r_reg = returns_arr[regime_days]
        port_ret = np.einsum("tkn,tn->tk", w_reg, r_reg)
        if len(regime_days) > 1:
            turnover = np.sum(np.abs(np.diff(w_reg, axis=0)), axis=2)
            net_ret = port_ret[1:] - kappa * turnover
        else:
            net_ret = port_ret
        mean_r = net_ret.mean(axis=0)
        std_r = net_ret.std(axis=0)
        sharpes = np.where(std_r > 1e-10, mean_r / std_r * np.sqrt(252), 0.0)
        best_algo[regime_id] = int(np.argmax(sharpes))

    def _fn(i):
        return algo_outputs[i, best_algo.get(int(regime_arr[i]), 0), :]
    return _eval_strategy(_fn, returns_arr, test_indices, kappa, N)


def evaluate_bo_top3_blend(
    algo_outputs, returns_arr, regime_arr, train_mask, test_indices, kappa, N
):
    """Plan 13b Top-3 Blend: equal-weight of top-3 algos per regime."""
    K = algo_outputs.shape[1]
    top3 = {}
    for regime_id in [1, 2, 3, 4]:
        rm = train_mask & (regime_arr == regime_id)
        regime_days = np.where(rm)[0]
        if len(regime_days) < 10:
            top3[regime_id] = [0, 0, 0]
            continue
        w_reg = algo_outputs[regime_days]
        r_reg = returns_arr[regime_days]
        port_ret = np.einsum("tkn,tn->tk", w_reg, r_reg)
        if len(regime_days) > 1:
            turnover = np.sum(np.abs(np.diff(w_reg, axis=0)), axis=2)
            net_ret = port_ret[1:] - kappa * turnover
        else:
            net_ret = port_ret
        mean_r = net_ret.mean(axis=0)
        std_r = net_ret.std(axis=0)
        sharpes = np.where(std_r > 1e-10, mean_r / std_r * np.sqrt(252), 0.0)
        top3[regime_id] = np.argsort(sharpes)[-3:][::-1].tolist()

    def _fn(i):
        algos = top3.get(int(regime_arr[i]), [0])
        return np.mean([algo_outputs[i, k, :] for k in algos], axis=0)
    return _eval_strategy(_fn, returns_arr, test_indices, kappa, N)


def evaluate_oracle_per_regime(
    algo_outputs, returns_arr, regime_arr, test_indices, kappa, N
):
    """Oracle: best algo per regime on test data (hindsight upper bound)."""
    K = algo_outputs.shape[1]
    oracle_best = {}
    for regime_id in [1, 2, 3, 4]:
        r_days = test_indices[regime_arr[test_indices] == regime_id]
        if len(r_days) < 5:
            oracle_best[regime_id] = 0
            continue
        w_r = algo_outputs[r_days]
        ret_r = returns_arr[r_days]
        port_ret = np.einsum("tkn,tn->tk", w_r, ret_r)
        if len(r_days) > 1:
            turnover = np.sum(np.abs(np.diff(w_r, axis=0)), axis=2)
            net_ret = port_ret[1:] - kappa * turnover
        else:
            net_ret = port_ret
        mean_r = net_ret.mean(axis=0)
        std_r = net_ret.std(axis=0)
        sharpes = np.where(std_r > 1e-10, mean_r / std_r * np.sqrt(252), 0.0)
        oracle_best[regime_id] = int(np.argmax(sharpes))

    def _fn(i):
        return algo_outputs[i, oracle_best.get(int(regime_arr[i]), 0), :]
    return _eval_strategy(_fn, returns_arr, test_indices, kappa, N)


# ---------------------------------------------------------------------------
# Per-fold runner
# ---------------------------------------------------------------------------

def run_fold(
    fold_id: int,
    fold_spec: dict,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime_labels: pd.Series,
    full_asset_features: pd.DataFrame,
    config: dict,
) -> dict:
    sep = "=" * 70
    print(f"\n{sep}", flush=True)
    print(
        f"FOLD {fold_id}  |  Train {fold_spec['train_start'][:4]}–{fold_spec['train_end'][:4]}"
        f"  |  Test {fold_spec['test_start'][:4]}",
        flush=True,
    )
    print(sep, flush=True)

    train_start = fold_spec["train_start"]
    train_end   = fold_spec["train_end"]
    test_start  = fold_spec["test_start"]
    test_end    = fold_spec["test_end"]
    kappa       = config["kappa"]

    # ------------------------------------------------------------------
    # Build algorithm metadata (no training needed if cache exists)
    # ------------------------------------------------------------------
    t0 = time.time()
    algorithms = build_algorithm_space(tiers=config["tiers"])
    K = len(algorithms)
    algo_names = [a.name for a in algorithms]

    n_tier1 = sum(1 for a in algorithms if not isinstance(a, TrainablePortfolioAlgorithm))
    n_trainable = K - n_tier1
    n_tier3 = sum(1 for a in algorithms if HAS_TIER3 and isinstance(a, Tier3Algorithm))
    n_tier2 = n_trainable - n_tier3
    tier_sizes = [n_tier1, n_tier2, n_tier3]

    tier1_idx = list(range(n_tier1))
    tier2_idx = list(range(n_tier1, n_tier1 + n_tier2))
    tier3_idx = list(range(n_tier1 + n_tier2, K))
    tier_algorithm_indices = [tier1_idx, tier2_idx, tier3_idx]

    ew_k = next((k for k, a in enumerate(algorithms) if a.name == "EqualWeight"), 0)
    print(f"  K={K}: {n_tier1} Tier1, {n_tier2} Tier2, {n_tier3} Tier3", flush=True)

    # ------------------------------------------------------------------
    # Load or compute algo outputs
    # ------------------------------------------------------------------
    cache_path = os.path.join(config["cache_dir"], f"fold_{fold_id:02d}_algo_outputs.npy")

    if os.path.exists(cache_path):
        print(f"  Loading cached algo outputs: {cache_path}", flush=True)
        dataset = MetaLearnerDataset(
            prices=prices,
            all_asset_features=full_asset_features,
            returns=returns,
            regime_labels=regime_labels,
            algorithms=algorithms,
        )
        dataset.fit_scaler(train_start, train_end)
        dataset._algo_outputs = np.load(cache_path)
        print(f"  Cache shape: {dataset._algo_outputs.shape}", flush=True)
    else:
        print("  Cache not found — precomputing algorithm outputs ...", flush=True)
        has_trainable = any(isinstance(a, TrainablePortfolioAlgorithm) for a in algorithms)
        if has_trainable:
            pretrain_algorithms(algorithms, full_asset_features, returns, train_start, train_end)
        dataset = MetaLearnerDataset(
            prices=prices,
            all_asset_features=full_asset_features,
            returns=returns,
            regime_labels=regime_labels,
            algorithms=algorithms,
        )
        dataset.fit_scaler(train_start, train_end)
        dataset.batch_precompute_algo_outputs()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, dataset._algo_outputs)
        print(f"  Saved to cache: {cache_path}", flush=True)

    t1 = time.time()
    print(f"  Data ready in {t1 - t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Build index arrays
    # ------------------------------------------------------------------
    train_idx = dataset.get_indices_for_period(train_start, train_end)
    test_idx  = dataset.get_indices_for_period(test_start,  test_end)

    if len(train_idx) == 0 or len(test_idx) == 0:
        print("  WARNING: empty train or test set — skipping fold.", flush=True)
        return None

    print(f"  Train: {len(train_idx)} days | Test: {len(test_idx)} days", flush=True)

    algo_outputs = dataset._algo_outputs                                    # (T, K, N)
    returns_arr  = np.array([dataset.get_returns(i) for i in range(len(dataset.dates))])
    regime_arr   = np.array([dataset.get_regime(i)  for i in range(len(dataset.dates))])
    N = algo_outputs.shape[2]

    train_mask = np.zeros(len(dataset.dates), dtype=bool)
    train_mask[train_idx] = True

    # ------------------------------------------------------------------
    # BO within-tier (per regime × tier exhaustive search)
    # ------------------------------------------------------------------
    print("  Running BO within-tier (per regime × tier) ...", flush=True)
    t2 = time.time()
    best_algo_per_regime_tier, tier_sharpe_per_regime = bo_within_tier(
        algo_outputs=algo_outputs,
        returns_arr=returns_arr,
        regime_arr=regime_arr,
        train_mask=train_mask,
        tier_algorithm_indices=tier_algorithm_indices,
        kappa=kappa,
    )
    t3 = time.time()
    print(f"  BO within-tier done in {t3 - t2:.2f}s", flush=True)

    # Print BO selection summary
    for regime_id in [1, 2, 3, 4]:
        parts = []
        for tier_id in range(3):
            k = best_algo_per_regime_tier[regime_id][tier_id]
            s = tier_sharpe_per_regime[regime_id][tier_id]
            parts.append(
                f"T{tier_id+1}=#{k}({algo_names[k][:18]},S={s:+.3f})"
            )
        print(
            f"    Regime {regime_id} ({REGIME_NAMES[regime_id]:7s}): " + "  ".join(parts),
            flush=True,
        )

    # ------------------------------------------------------------------
    # Strategy A: Hybrid ML TierSelector + BO
    # ------------------------------------------------------------------
    if TORCH_AVAILABLE:
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])

        tier_selector = TierSelector(
            input_dim=config["input_dim"],
            hidden_dims=config["selector_hidden"],
            dropout=config["dropout"],
        )

        print(
            f"  Training TierSelector ({config['selector_epochs']} epochs, "
            f"lr={config['selector_lr']}, λ_tier={config['lambda_tier']}) ...",
            flush=True,
        )
        t4 = time.time()
        tier_selector = train_tier_selector(
            tier_selector=tier_selector,
            best_algo_per_regime_tier=best_algo_per_regime_tier,
            algo_outputs=algo_outputs,
            returns_arr=returns_arr,
            regime_arr=regime_arr,
            dataset=dataset,
            train_indices=train_idx,
            config=config,
        )
        t5 = time.time()
        print(f"  TierSelector training done in {t5 - t4:.1f}s", flush=True)

        m_hybrid, net_hybrid, beta_records = evaluate_hybrid(
            tier_selector=tier_selector,
            best_algo_per_regime_tier=best_algo_per_regime_tier,
            algo_outputs=algo_outputs,
            returns_arr=returns_arr,
            regime_arr=regime_arr,
            dataset=dataset,
            test_indices=test_idx,
            kappa=kappa,
            N=N,
        )

        # Entropy diagnostics for TierSelector
        betas = np.array([b[2] for b in beta_records])
        H_beta = np.mean([_entropy(betas[i]) for i in range(len(betas))])

        regime_beta_mean = {}
        for i, idx in enumerate(test_idx):
            regime = int(regime_arr[idx])
            if regime not in regime_beta_mean:
                regime_beta_mean[regime] = []
            regime_beta_mean[regime].append(betas[i])
        regime_beta_mean = {
            r: np.mean(v, axis=0) for r, v in regime_beta_mean.items()
        }

        print(f"  Strategy A (Hybrid):  Sharpe={m_hybrid['sharpe']:+.4f}  H(β)={H_beta:.4f}",
              flush=True)
        print("  Mean β per regime:", flush=True)
        for r_id in sorted(regime_beta_mean.keys()):
            b = regime_beta_mean[r_id]
            print(f"    {REGIME_NAMES.get(r_id,'?'):7s}: T1={b[0]:.3f}  T2={b[1]:.3f}  T3={b[2]:.3f}",
                  flush=True)
    else:
        m_hybrid = {k: np.nan for k in ["sharpe", "sortino", "ann_return", "ann_vol",
                                         "max_drawdown", "n_days", "avg_daily_turnover"]}
        net_hybrid = np.array([])
        H_beta = np.nan
        regime_beta_mean = {}
        beta_records = []

    # ------------------------------------------------------------------
    # Strategy B: BO-Only Per-Tier (Sharpe-weighted tiers, no NN)
    # ------------------------------------------------------------------
    m_bo_tier, net_bo_tier = evaluate_bo_per_tier(
        best_algo_per_regime_tier, tier_sharpe_per_regime,
        algo_outputs, returns_arr, regime_arr, test_idx, kappa, N
    )
    print(f"  Strategy B (BO-PerTier): Sharpe={m_bo_tier['sharpe']:+.4f}", flush=True)

    # ------------------------------------------------------------------
    # Strategy C: Best-Tier-Only (hard selection)
    # ------------------------------------------------------------------
    m_best_tier, net_best_tier = evaluate_best_tier_only(
        best_algo_per_regime_tier, tier_sharpe_per_regime,
        algo_outputs, returns_arr, regime_arr, test_idx, kappa, N
    )
    print(f"  Strategy C (BestTier):  Sharpe={m_best_tier['sharpe']:+.4f}", flush=True)

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    m_ew, net_ew = evaluate_ew(algo_outputs, returns_arr, test_idx, kappa, N, ew_k)
    m_hard, net_hard = evaluate_bo_hard_selection(
        algo_outputs, returns_arr, regime_arr, train_mask, test_idx, kappa, N, algo_names
    )
    m_blend, net_blend = evaluate_bo_top3_blend(
        algo_outputs, returns_arr, regime_arr, train_mask, test_idx, kappa, N
    )
    m_oracle, net_oracle = evaluate_oracle_per_regime(
        algo_outputs, returns_arr, regime_arr, test_idx, kappa, N
    )

    print(
        f"  Baselines:"
        f"\n    EW:               Sharpe={m_ew['sharpe']:+.4f}"
        f"\n    13b Hard Sel:     Sharpe={m_hard['sharpe']:+.4f}"
        f"\n    13b Top-3 Blend:  Sharpe={m_blend['sharpe']:+.4f}"
        f"\n    Oracle:           Sharpe={m_oracle['sharpe']:+.4f}",
        flush=True,
    )

    return {
        "fold_id":      fold_id,
        "fold_spec":    fold_spec,
        "tier_sizes":   tier_sizes,
        "algo_names":   algo_names,
        "ew_k":         ew_k,
        # 13c strategy metrics
        "metrics_hybrid":    m_hybrid,
        "metrics_bo_tier":   m_bo_tier,
        "metrics_best_tier": m_best_tier,
        # Baseline metrics
        "metrics_ew":        m_ew,
        "metrics_hard":      m_hard,
        "metrics_blend":     m_blend,
        "metrics_oracle":    m_oracle,
        # Net return series
        "net_hybrid":        net_hybrid,
        "net_bo_tier":       net_bo_tier,
        "net_best_tier":     net_best_tier,
        "net_ew":            net_ew,
        "net_hard":          net_hard,
        "net_blend":         net_blend,
        "net_oracle":        net_oracle,
        # Test dates
        "test_dates":        dataset.dates[test_idx],
        # BO selection results
        "best_algo_per_regime_tier":  best_algo_per_regime_tier,
        "tier_sharpe_per_regime":     tier_sharpe_per_regime,
        # TierSelector diagnostics
        "H_beta":            H_beta,
        "regime_beta_mean":  regime_beta_mean,
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(fold_results: list, config: dict) -> None:
    valid = [fr for fr in fold_results if fr is not None]
    if not valid:
        print("No valid folds.", flush=True)
        return

    # Load Plan 13a and 13b prior results
    plan13a = {}
    if os.path.exists(config["plan13a_summary"]):
        df13a = pd.read_csv(config["plan13a_summary"])
        plan13a = {int(r["fold"]): float(r["hier_sharpe"]) for _, r in df13a.iterrows()}

    plan13b_hard = {}
    plan13b_blend = {}
    if os.path.exists(config["plan13b_summary"]):
        df13b = pd.read_csv(config["plan13b_summary"])
        plan13b_hard  = {int(r["fold"]): float(r["hard_sharpe"])  for _, r in df13b.iterrows()}
        plan13b_blend = {int(r["fold"]): float(r["blend_sharpe"]) for _, r in df13b.iterrows()}

    # Per-fold table
    print("\n" + "=" * 90, flush=True)
    print("PLAN 13c HYBRID — PER-FOLD RESULTS", flush=True)
    print("=" * 90, flush=True)
    hdr = (
        f"{'Fold':>4}  {'Year':>5}  {'EW':>7}  {'Hybrid':>7}  "
        f"{'BOTier':>7}  {'BestT':>7}  {'13b Hard':>8}  {'13b Blnd':>8}  "
        f"{'Oracle':>7}  {'H(β)':>6}"
    )
    print(hdr, flush=True)
    print("-" * 90, flush=True)

    for fr in valid:
        fold_id = fr["fold_id"]
        year    = fr["fold_spec"]["test_start"][:4]
        s_ew    = fr["metrics_ew"]["sharpe"]
        s_hyb   = fr["metrics_hybrid"]["sharpe"]
        s_bo    = fr["metrics_bo_tier"]["sharpe"]
        s_bt    = fr["metrics_best_tier"]["sharpe"]
        s_hard  = fr["metrics_hard"]["sharpe"]
        s_blnd  = fr["metrics_blend"]["sharpe"]
        s_orc   = fr["metrics_oracle"]["sharpe"]
        h_b     = fr["H_beta"]

        def _fmt(v):
            return f"{v:+.4f}" if not np.isnan(v) else "   nan"

        h_b_str = f"{h_b:>6.3f}" if not np.isnan(h_b) else "   nan"
        print(
            f"{fold_id:>4}  {year:>5}  "
            f"{_fmt(s_ew)}  {_fmt(s_hyb)}  {_fmt(s_bo)}  {_fmt(s_bt)}  "
            f"{_fmt(s_hard):>8}  {_fmt(s_blnd):>8}  "
            f"{_fmt(s_orc)}  {h_b_str:>6}",
            flush=True,
        )

    print("-" * 90, flush=True)

    # Aggregate stats per strategy
    def _agg(sharpes_list):
        arr = [s for s in sharpes_list if not np.isnan(s)]
        if not arr:
            return np.nan, 0
        return np.mean(arr), arr

    strategies = {
        "EW":                  [fr["metrics_ew"]["sharpe"]         for fr in valid],
        "13c Hybrid ML+BO":    [fr["metrics_hybrid"]["sharpe"]     for fr in valid],
        "13c BO-Only Per-Tier":[fr["metrics_bo_tier"]["sharpe"]    for fr in valid],
        "13c Best-Tier-Only":  [fr["metrics_best_tier"]["sharpe"]  for fr in valid],
        "13b Hard Selection":  [fr["metrics_hard"]["sharpe"]       for fr in valid],
        "13b Top-3 Blend":     [fr["metrics_blend"]["sharpe"]      for fr in valid],
        "Oracle Per-Regime":   [fr["metrics_oracle"]["sharpe"]     for fr in valid],
    }

    # Add 13a from file
    if plan13a:
        strategies["13a Hierarchical ML"] = [
            plan13a.get(fr["fold_id"], np.nan) for fr in valid
        ]

    ew_sharpes = strategies["EW"]

    # Final comparison table
    print("\n" + "=" * 70, flush=True)
    print("FINAL COMPARISON: All Three Options (12-fold Walk-Forward, 2013-2024)", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Strategy':<28}  {'Avg Sharpe':>11}  {'Beats EW':>9}  {'Folds Won':>10}", flush=True)
    print("-" * 70, flush=True)

    for name, sharpes in strategies.items():
        valid_s = [s for s in sharpes if not np.isnan(s)]
        if not valid_s:
            avg_s = np.nan
            beats = "n/a"
            won   = "n/a"
        else:
            avg_s = np.mean(valid_s)
            beats_list = [
                s > ew_sharpes[i]
                for i, s in enumerate(sharpes)
                if not np.isnan(s) and not np.isnan(ew_sharpes[i])
            ]
            beats = f"{sum(beats_list)}/{len(beats_list)}" if beats_list else "n/a"

            # "Folds won" = beats EW + better than 13b hard (the best known non-oracle)
            won_n = sum(
                s > ew_sharpes[i]
                for i, s in enumerate(sharpes)
                if not np.isnan(s) and not np.isnan(ew_sharpes[i])
            )
            won = f"{won_n}/{len(valid_s)}"

        if not np.isnan(avg_s if avg_s == avg_s else float("nan")):
            print(f"{name:<28}  {avg_s:>+11.4f}  {beats:>9}  {won:>10}", flush=True)
        else:
            print(f"{name:<28}  {'   nan':>11}  {beats:>9}  {won:>10}", flush=True)

    print("-" * 70, flush=True)
    print("(Beats EW = folds where strategy > EW Sharpe)", flush=True)
    print("(Folds Won = same as Beats EW — column kept for readability)", flush=True)
    print("=" * 70, flush=True)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(fold_results: list, config: dict) -> None:
    valid = [fr for fr in fold_results if fr is not None]
    if not valid:
        return

    out_dir = config["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Load prior results
    plan13a = {}
    if os.path.exists(config["plan13a_summary"]):
        df13a = pd.read_csv(config["plan13a_summary"])
        plan13a = {int(r["fold"]): float(r["hier_sharpe"]) for _, r in df13a.iterrows()}

    plan13b_hard = {}
    plan13b_blend = {}
    if os.path.exists(config["plan13b_summary"]):
        df13b = pd.read_csv(config["plan13b_summary"])
        plan13b_hard  = {int(r["fold"]): float(r["hard_sharpe"])  for _, r in df13b.iterrows()}
        plan13b_blend = {int(r["fold"]): float(r["blend_sharpe"]) for _, r in df13b.iterrows()}

    # 1. summary_metrics.csv
    rows = []
    for fr in valid:
        fold_id = fr["fold_id"]
        year    = fr["fold_spec"]["test_start"][:4]
        s_ew    = fr["metrics_ew"]["sharpe"]
        rows.append({
            "fold":                fold_id,
            "test_year":           year,
            "hybrid_sharpe":       fr["metrics_hybrid"]["sharpe"],
            "bo_per_tier_sharpe":  fr["metrics_bo_tier"]["sharpe"],
            "best_tier_sharpe":    fr["metrics_best_tier"]["sharpe"],
            "ew_sharpe":           s_ew,
            "hard_sharpe":         fr["metrics_hard"]["sharpe"],
            "blend_sharpe":        fr["metrics_blend"]["sharpe"],
            "oracle_sharpe":       fr["metrics_oracle"]["sharpe"],
            "plan13a_sharpe":      plan13a.get(fold_id, np.nan),
            "H_beta":              fr["H_beta"],
            "hybrid_vs_ew":        fr["metrics_hybrid"]["sharpe"] - s_ew
                                   if not np.isnan(fr["metrics_hybrid"]["sharpe"]) else np.nan,
            "bo_tier_vs_ew":       fr["metrics_bo_tier"]["sharpe"] - s_ew,
            "best_tier_vs_ew":     fr["metrics_best_tier"]["sharpe"] - s_ew,
            "hard_vs_ew":          fr["metrics_hard"]["sharpe"] - s_ew,
            "hybrid_ann_return":   fr["metrics_hybrid"]["ann_return"],
            "bo_tier_ann_return":  fr["metrics_bo_tier"]["ann_return"],
            "best_tier_ann_return":fr["metrics_best_tier"]["ann_return"],
            "ew_ann_return":       fr["metrics_ew"]["ann_return"],
            "hybrid_maxdd":        fr["metrics_hybrid"]["max_drawdown"],
            "bo_tier_maxdd":       fr["metrics_bo_tier"]["max_drawdown"],
            "ew_maxdd":            fr["metrics_ew"]["max_drawdown"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)
    print(f"\nSaved: {os.path.join(out_dir, 'summary_metrics.csv')}", flush=True)

    # 2. bo_selected_algos.csv
    bo_rows = []
    for fr in valid:
        fold_id = fr["fold_id"]
        algo_names = fr["algo_names"]
        for regime_id in [1, 2, 3, 4]:
            for tier_id in range(3):
                k = fr["best_algo_per_regime_tier"][regime_id][tier_id]
                s = fr["tier_sharpe_per_regime"][regime_id][tier_id]
                bo_rows.append({
                    "fold":        fold_id,
                    "test_year":   fr["fold_spec"]["test_start"][:4],
                    "regime_id":   regime_id,
                    "regime_name": REGIME_NAMES[regime_id],
                    "tier_id":     tier_id + 1,
                    "best_algo_k": k,
                    "best_algo_name": algo_names[k] if k < len(algo_names) else "?",
                    "train_sharpe":  s,
                })
    pd.DataFrame(bo_rows).to_csv(os.path.join(out_dir, "bo_selected_algos.csv"), index=False)
    print(f"Saved: {os.path.join(out_dir, 'bo_selected_algos.csv')}", flush=True)

    # 3. tier_selection_by_regime.csv (mean β per regime per fold)
    tier_rows = []
    for fr in valid:
        fold_id = fr["fold_id"]
        for regime_id, b_mean in fr["regime_beta_mean"].items():
            tier_rows.append({
                "fold":         fold_id,
                "test_year":    fr["fold_spec"]["test_start"][:4],
                "regime_id":    regime_id,
                "regime_name":  REGIME_NAMES.get(regime_id, "?"),
                "beta_tier1":   float(b_mean[0]),
                "beta_tier2":   float(b_mean[1]),
                "beta_tier3":   float(b_mean[2]),
                "H_beta":       _entropy(b_mean),
            })
    if tier_rows:
        pd.DataFrame(tier_rows).to_csv(
            os.path.join(out_dir, "tier_selection_by_regime.csv"), index=False
        )
        print(f"Saved: {os.path.join(out_dir, 'tier_selection_by_regime.csv')}", flush=True)

    # 4. comparison_all_options.csv (full cross-plan table)
    cmp_rows = []
    ew_sharpes = {fr["fold_id"]: fr["metrics_ew"]["sharpe"] for fr in valid}

    for fr in valid:
        fold_id = fr["fold_id"]
        row = {
            "fold":               fold_id,
            "test_year":          fr["fold_spec"]["test_start"][:4],
            "ew":                 fr["metrics_ew"]["sharpe"],
            "plan13a_hier":       plan13a.get(fold_id, np.nan),
            "plan13b_hard":       fr["metrics_hard"]["sharpe"],
            "plan13b_blend":      fr["metrics_blend"]["sharpe"],
            "plan13c_hybrid":     fr["metrics_hybrid"]["sharpe"],
            "plan13c_bo_tier":    fr["metrics_bo_tier"]["sharpe"],
            "plan13c_best_tier":  fr["metrics_best_tier"]["sharpe"],
            "oracle":             fr["metrics_oracle"]["sharpe"],
        }
        cmp_rows.append(row)

    df_cmp = pd.DataFrame(cmp_rows)
    df_cmp.to_csv(os.path.join(out_dir, "comparison_all_options.csv"), index=False)
    print(f"Saved: {os.path.join(out_dir, 'comparison_all_options.csv')}", flush=True)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(config: dict = CONFIG) -> list:
    if TORCH_AVAILABLE:
        print("PyTorch available — Strategy A (Hybrid) will run.", flush=True)
    else:
        print("PyTorch NOT available — Strategy A (Hybrid) will be skipped.", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("PLAN 13c: HYBRID META-LEARNER + BO WITHIN TIER — FULL 12-FOLD RUN", flush=True)
    print("=" * 70, flush=True)
    print(f"  selector_epochs={config['selector_epochs']}  "
          f"selector_lr={config['selector_lr']}  "
          f"lambda_tier={config['lambda_tier']}", flush=True)
    print(f"  kappa={config['kappa']}  seed={config['seed']}", flush=True)

    print("\nLoading data ...", flush=True)
    data = load_data_extended()
    prices = data["prices"]
    vix    = data["vix"]
    print(f"  Prices: {prices.shape}  ({prices.index[0].date()} – {prices.index[-1].date()})",
          flush=True)

    returns             = compute_returns(prices)
    full_asset_features = compute_asset_features(prices)
    regime_labels       = compute_regime_labels(vix)

    common = (
        prices.index
        .intersection(returns.index)
        .intersection(regime_labels.index)
    )
    prices              = prices.loc[common]
    returns             = returns.loc[common]
    regime_labels       = regime_labels.loc[common]
    full_asset_features = full_asset_features.loc[
        full_asset_features.index.intersection(common)
    ]
    print(f"  Common dates: {len(common)}", flush=True)

    wfv = WalkForwardValidator(
        train_years=config["train_years"],
        test_years=config["test_years"],
        step_years=config["step_years"],
        min_test_start=config["min_test_start"],
    )
    all_folds = wfv.generate_folds(data_end=config["data_end"])
    print(f"Generated {len(all_folds)} walk-forward folds", flush=True)

    fold_results = []
    for fold_spec in all_folds:
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

    _print_summary(fold_results, config)
    save_results(fold_results, config)

    print("\nPLAN 13c COMPLETE.", flush=True)
    return fold_results


if __name__ == "__main__":
    run_experiment()
