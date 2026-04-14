# experiments/plan13b_bayesian_optimization.py -- Plan 13b: Pure BO / Exhaustive Selection
#
# Replaces the hierarchical meta-learner with direct per-regime exhaustive search.
# For each of the 4 regimes, evaluates all 117 algorithms on training days in that
# regime, then selects the best one (Hard Selection) or top-3 (Top-3 Blend).
#
# Reuses cached algorithm outputs from Plan 13a:
#   results/plan13a_hierarchical/cache/fold_XX_algo_outputs.npy
#
# Usage:
#   cd Implementierung1
#   python -u -m regime_algo_selection.experiments.plan13b_bayesian_optimization

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

# ---- project imports -------------------------------------------------------
from regime_algo_selection.config import (
    RESULTS_DIR, KAPPA, REGIME_NAMES, N_REGIMES, N_ASSETS, RANDOM_SEED,
)
from regime_algo_selection.data.loader_extended import load_data_extended
from regime_algo_selection.data.features import compute_returns, compute_asset_features
from regime_algo_selection.regimes.ground_truth import compute_regime_labels
from regime_algo_selection.algorithms.tier1_heuristics import build_algorithm_space
from regime_algo_selection.algorithms.base import TrainablePortfolioAlgorithm
from regime_algo_selection.algorithms.stage0 import pretrain_algorithms
from regime_algo_selection.evaluation.walk_forward import WalkForwardValidator
from regime_algo_selection.meta_learner.dataset import MetaLearnerDataset

try:
    from regime_algo_selection.algorithms.tier3_nonlinear import Tier3Algorithm
    HAS_TIER3 = True
except ImportError:
    HAS_TIER3 = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # BO / exhaustive search
    "top_n": 3,                 # blend top-N algorithms per regime
    "kappa": KAPPA,             # 0.001 switching cost

    # Walk-forward (same as Plan 13a)
    "train_years": 8,
    "test_years": 1,
    "step_years": 1,
    "min_test_start": "2013-01-01",
    "data_end": "2024-12-31",

    # Algorithm space
    "tiers": [1, 2, 3],

    # Cache (reuse Plan 13a)
    "cache_dir": os.path.join(RESULTS_DIR, "plan13a_hierarchical", "cache"),
    "plan13a_summary": os.path.join(RESULTS_DIR, "plan13a_hierarchical", "summary_metrics.csv"),

    # Output
    "output_dir": os.path.join(RESULTS_DIR, "plan13b_bayesian_opt"),

    # Random seed
    "seed": RANDOM_SEED,
}

# ---------------------------------------------------------------------------
# Metric helpers (identical to Plan 13a)
# ---------------------------------------------------------------------------

def _compute_metrics(daily_net_returns: np.ndarray) -> dict:
    """Compute standard portfolio metrics from an array of daily net returns."""
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


# ---------------------------------------------------------------------------
# Per-regime exhaustive search
# ---------------------------------------------------------------------------

def _sharpe_for_algo(weights: np.ndarray, rets: np.ndarray, kappa: float) -> float:
    """
    Compute Sharpe ratio for a single algorithm on a set of days.

    weights : (T_reg, N)
    rets    : (T_reg, N)
    """
    port_ret = np.sum(weights * rets, axis=1)  # (T_reg,)

    if len(weights) > 1:
        turnover = np.sum(np.abs(np.diff(weights, axis=0)), axis=1)  # (T_reg-1,)
        costs = kappa * turnover
        net_ret = port_ret[1:] - costs
    else:
        net_ret = port_ret

    if len(net_ret) < 5 or np.std(net_ret) < 1e-10:
        return 0.0
    return float(np.mean(net_ret) / np.std(net_ret) * np.sqrt(252))


def run_exhaustive_per_regime(
    algo_outputs: np.ndarray,
    returns_arr: np.ndarray,
    regime_arr: np.ndarray,
    train_mask: np.ndarray,
    config: dict,
    algo_names: list,
) -> tuple:
    """
    Exhaustively evaluate all K algorithms per regime on training data.

    Parameters
    ----------
    algo_outputs : (T, K, N)
    returns_arr  : (T, N)
    regime_arr   : (T,)  — regime labels {1,2,3,4}
    train_mask   : (T,)  — boolean, True for training days
    config       : CONFIG dict
    algo_names   : list of K algorithm name strings

    Returns
    -------
    best_algo    : {regime_id: int}
    top_n_algos  : {regime_id: list[int]}
    sharpes_per_regime : {regime_id: np.ndarray shape (K,)}
    regime_day_counts  : {regime_id: int}
    """
    K = algo_outputs.shape[1]
    kappa = config["kappa"]
    top_n = config["top_n"]

    best_algo = {}
    top_n_algos = {}
    sharpes_per_regime = {}
    regime_day_counts = {}

    for regime_id in [1, 2, 3, 4]:
        regime_train_mask = train_mask & (regime_arr == regime_id)
        regime_days = np.where(regime_train_mask)[0]
        regime_day_counts[regime_id] = len(regime_days)

        if len(regime_days) < 10:
            print(f"    Regime {regime_id} ({REGIME_NAMES[regime_id]:7s}): "
                  f"too few days ({len(regime_days)}), using EW fallback", flush=True)
            best_algo[regime_id] = 0
            top_n_algos[regime_id] = [0] * min(top_n, K)
            sharpes_per_regime[regime_id] = np.zeros(K)
            continue

        # Vectorised: compute Sharpe for all K algorithms at once
        # algo_outputs[regime_days] : (T_reg, K, N)
        # returns_arr[regime_days]  : (T_reg, N)
        T_reg = len(regime_days)
        w_reg = algo_outputs[regime_days]   # (T_reg, K, N)
        r_reg = returns_arr[regime_days]    # (T_reg, N)

        # port_ret: (T_reg, K)
        port_ret = np.einsum("tkn,tn->tk", w_reg, r_reg)

        if T_reg > 1:
            # turnover per algo: diff over time, sum over assets  → (T_reg-1, K)
            turnover = np.sum(np.abs(np.diff(w_reg, axis=0)), axis=2)  # (T_reg-1, K)
            costs = kappa * turnover
            net_ret = port_ret[1:] - costs    # (T_reg-1, K)
        else:
            net_ret = port_ret                # (T_reg, K)

        mean_r = net_ret.mean(axis=0)         # (K,)
        std_r = net_ret.std(axis=0)           # (K,)

        # Avoid division by zero; treat near-zero std as 0 Sharpe
        valid = (std_r > 1e-10) & (net_ret.shape[0] >= 5)
        sharpes = np.where(valid, mean_r / std_r * np.sqrt(252), 0.0)
        sharpes_per_regime[regime_id] = sharpes

        best_idx = int(np.argmax(sharpes))
        best_algo[regime_id] = best_idx

        effective_top_n = min(top_n, K)
        top_n_idx = np.argsort(sharpes)[-effective_top_n:][::-1].tolist()
        top_n_algos[regime_id] = top_n_idx

        print(
            f"    Regime {regime_id} ({REGIME_NAMES[regime_id]:7s}): "
            f"{T_reg:4d} days  |  "
            f"best=#{best_idx:3d} {algo_names[best_idx][:30]:<30} "
            f"Sharpe={sharpes[best_idx]:+.4f}  |  "
            f"top-{effective_top_n}: {top_n_idx}",
            flush=True,
        )

    return best_algo, top_n_algos, sharpes_per_regime, regime_day_counts


# ---------------------------------------------------------------------------
# Test-time evaluation
# ---------------------------------------------------------------------------

def _eval_strategy(
    weight_fn,
    returns_arr: np.ndarray,
    test_indices: np.ndarray,
    kappa: float,
    N: int,
) -> tuple:
    """
    Evaluate any weight function on test indices.

    weight_fn(i) → np.ndarray shape (N,)   (i = integer index into full array)
    Returns (metrics dict, net_returns array)
    """
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


def evaluate_hard_selection(
    best_algo: dict,
    algo_outputs: np.ndarray,
    returns_arr: np.ndarray,
    regime_arr: np.ndarray,
    test_indices: np.ndarray,
    kappa: float,
    N: int,
) -> tuple:
    """Hard Selection: at each test day, use best_algo[regime]."""
    def _fn(i):
        regime = int(regime_arr[i])
        k = best_algo.get(regime, 0)
        return algo_outputs[i, k, :]

    return _eval_strategy(_fn, returns_arr, test_indices, kappa, N)


def evaluate_top_n_blend(
    top_n_algos: dict,
    algo_outputs: np.ndarray,
    returns_arr: np.ndarray,
    regime_arr: np.ndarray,
    test_indices: np.ndarray,
    kappa: float,
    N: int,
) -> tuple:
    """Top-N Blend: at each test day, equal-weight blend of top-N algos for that regime."""
    def _fn(i):
        regime = int(regime_arr[i])
        algos = top_n_algos.get(regime, [0])
        w = np.mean([algo_outputs[i, k, :] for k in algos], axis=0)
        return w

    return _eval_strategy(_fn, returns_arr, test_indices, kappa, N)


def evaluate_ew(
    algo_outputs: np.ndarray,
    returns_arr: np.ndarray,
    test_indices: np.ndarray,
    kappa: float,
    N: int,
    ew_k: int = 0,
) -> tuple:
    """Equal Weight: always use algorithm index ew_k (should be EqualWeight)."""
    def _fn(i):
        return algo_outputs[i, ew_k, :]

    return _eval_strategy(_fn, returns_arr, test_indices, kappa, N)


def evaluate_best_single_global(
    algo_outputs: np.ndarray,
    returns_arr: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    kappa: float,
    N: int,
) -> tuple:
    """
    Best Single Algorithm (global): find globally best algo on training data,
    ignoring regimes. Use it for all test days.
    """
    K = algo_outputs.shape[1]

    # Compute training Sharpe for each algo
    w_train = algo_outputs[train_indices]   # (T_train, K, N)
    r_train = returns_arr[train_indices]    # (T_train, N)

    port_ret = np.einsum("tkn,tn->tk", w_train, r_train)
    if len(train_indices) > 1:
        turnover = np.sum(np.abs(np.diff(w_train, axis=0)), axis=2)
        costs = kappa * turnover
        net_ret = port_ret[1:] - costs
    else:
        net_ret = port_ret

    mean_r = net_ret.mean(axis=0)
    std_r = net_ret.std(axis=0)
    valid = std_r > 1e-10
    sharpes = np.where(valid, mean_r / std_r * np.sqrt(252), 0.0)

    best_k = int(np.argmax(sharpes))

    def _fn(i):
        return algo_outputs[i, best_k, :]

    m, arr = _eval_strategy(_fn, returns_arr, test_indices, kappa, N)
    m["best_k"] = best_k
    return m, arr, best_k


def evaluate_oracle_best_per_regime(
    algo_outputs: np.ndarray,
    returns_arr: np.ndarray,
    regime_arr: np.ndarray,
    test_indices: np.ndarray,
    kappa: float,
    N: int,
) -> tuple:
    """
    Oracle Best Per Regime: find best algo per regime on TEST data (hindsight).
    This is the upper bound for any regime-based selection strategy.
    """
    K = algo_outputs.shape[1]
    oracle_best = {}

    for regime_id in [1, 2, 3, 4]:
        regime_test_mask = np.array([regime_arr[i] == regime_id for i in test_indices])
        r_days = test_indices[regime_test_mask]

        if len(r_days) < 5:
            oracle_best[regime_id] = 0
            continue

        w_r = algo_outputs[r_days]
        ret_r = returns_arr[r_days]
        port_ret = np.einsum("tkn,tn->tk", w_r, ret_r)

        if len(r_days) > 1:
            turnover = np.sum(np.abs(np.diff(w_r, axis=0)), axis=2)
            costs = kappa * turnover
            net_ret = port_ret[1:] - costs
        else:
            net_ret = port_ret

        mean_r = net_ret.mean(axis=0)
        std_r = net_ret.std(axis=0)
        valid = std_r > 1e-10
        sharpes = np.where(valid, mean_r / std_r * np.sqrt(252), 0.0)
        oracle_best[regime_id] = int(np.argmax(sharpes))

    def _fn(i):
        regime = int(regime_arr[i])
        k = oracle_best.get(regime, 0)
        return algo_outputs[i, k, :]

    return _eval_strategy(_fn, returns_arr, test_indices, kappa, N)


# ---------------------------------------------------------------------------
# Tier analysis helper
# ---------------------------------------------------------------------------

def _get_tier_for_algo(k: int, n_tier1: int, n_tier2: int) -> int:
    """Return tier (1, 2, or 3) for algorithm index k."""
    if k < n_tier1:
        return 1
    elif k < n_tier1 + n_tier2:
        return 2
    return 3


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
    # Build algorithms (needed for metadata + fallback precompute)
    # ------------------------------------------------------------------
    t0 = time.time()
    algorithms = build_algorithm_space(tiers=config["tiers"])
    K = len(algorithms)
    algo_names = [a.name for a in algorithms]

    # Determine tier boundaries
    n_tier1 = sum(1 for a in algorithms if not isinstance(a, TrainablePortfolioAlgorithm))
    n_trainable = K - n_tier1
    if HAS_TIER3:
        n_tier3 = sum(1 for a in algorithms if isinstance(a, Tier3Algorithm))
    else:
        n_tier3 = 0
    n_tier2 = n_trainable - n_tier3
    tier_sizes = [n_tier1, n_tier2, n_tier3]

    print(f"  K={K} algorithms: {n_tier1} Tier1, {n_tier2} Tier2, {n_tier3} Tier3", flush=True)

    # Find EW algorithm index
    ew_k = next((k for k, a in enumerate(algorithms) if a.name == "EqualWeight"), 0)

    # ------------------------------------------------------------------
    # Load or compute algo outputs
    # ------------------------------------------------------------------
    cache_path = os.path.join(config["cache_dir"], f"fold_{fold_id:02d}_algo_outputs.npy")

    if os.path.exists(cache_path):
        print(f"  Loading cached algo outputs: {cache_path}", flush=True)
        # No pretrain needed — we load outputs from cache, not from compute_weights()
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

    # Convert dataset indices → full-array slices
    algo_outputs = dataset._algo_outputs                     # (T, K, N)
    returns_arr = np.array([dataset.get_returns(i) for i in range(len(dataset.dates))])
    regime_arr  = np.array([dataset.get_regime(i) for i in range(len(dataset.dates))])
    N = algo_outputs.shape[2]

    train_mask = np.zeros(len(dataset.dates), dtype=bool)
    train_mask[train_idx] = True

    # ------------------------------------------------------------------
    # Per-regime exhaustive search on training data
    # ------------------------------------------------------------------
    print("  Running per-regime exhaustive search ...", flush=True)
    t2 = time.time()
    best_algo, top_n_algos, sharpes_per_regime, regime_day_counts = run_exhaustive_per_regime(
        algo_outputs=algo_outputs,
        returns_arr=returns_arr,
        regime_arr=regime_arr,
        train_mask=train_mask,
        config=config,
        algo_names=algo_names,
    )
    t3 = time.time()
    print(f"  Exhaustive search done in {t3 - t2:.2f}s", flush=True)

    # ------------------------------------------------------------------
    # Evaluate strategies on test data
    # ------------------------------------------------------------------
    print("  Evaluating strategies on test data ...", flush=True)

    # Strategy A: Hard Selection
    m_hard, net_hard = evaluate_hard_selection(
        best_algo, algo_outputs, returns_arr, regime_arr, test_idx, kappa, N
    )

    # Strategy B: Top-3 Blend
    m_blend, net_blend = evaluate_top_n_blend(
        top_n_algos, algo_outputs, returns_arr, regime_arr, test_idx, kappa, N
    )

    # Baseline: Equal Weight
    m_ew, net_ew = evaluate_ew(algo_outputs, returns_arr, test_idx, kappa, N, ew_k)

    # Baseline: Best Single Algorithm (global, train-set selected)
    m_global, net_global, best_global_k = evaluate_best_single_global(
        algo_outputs, returns_arr, train_idx, test_idx, kappa, N
    )

    # Oracle: Best Per Regime on test data (hindsight upper bound)
    m_oracle, net_oracle = evaluate_oracle_best_per_regime(
        algo_outputs, returns_arr, regime_arr, test_idx, kappa, N
    )

    print(
        f"  Results:"
        f"\n    Hard Selection : Sharpe={m_hard['sharpe']:+.4f}"
        f"\n    Top-3 Blend    : Sharpe={m_blend['sharpe']:+.4f}"
        f"\n    Equal Weight   : Sharpe={m_ew['sharpe']:+.4f}"
        f"\n    Best Single    : Sharpe={m_global['sharpe']:+.4f}  (algo #{best_global_k} {algo_names[best_global_k]})"
        f"\n    Oracle Per-Reg : Sharpe={m_oracle['sharpe']:+.4f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Tier analysis for selected algorithms
    # ------------------------------------------------------------------
    tier_selection = {}
    for regime_id in [1, 2, 3, 4]:
        k = best_algo[regime_id]
        tier_selection[regime_id] = _get_tier_for_algo(k, n_tier1, n_tier2)

    # Top-10 tier distribution per regime
    tier_distribution = {}
    for regime_id in [1, 2, 3, 4]:
        sharpes_r = sharpes_per_regime.get(regime_id, np.zeros(K))
        top10_idx = np.argsort(sharpes_r)[-10:][::-1]
        tier_counts = {1: 0, 2: 0, 3: 0}
        for k in top10_idx:
            t = _get_tier_for_algo(int(k), n_tier1, n_tier2)
            tier_counts[t] += 1
        tier_distribution[regime_id] = tier_counts

    return {
        "fold_id":           fold_id,
        "fold_spec":         fold_spec,
        "tier_sizes":        tier_sizes,
        "algo_names":        algo_names,
        "ew_k":              ew_k,
        # Selection results
        "best_algo":         best_algo,
        "top_n_algos":       top_n_algos,
        "sharpes_per_regime": {r: s.tolist() for r, s in sharpes_per_regime.items()},
        "regime_day_counts": regime_day_counts,
        # Strategy metrics
        "metrics_hard":      m_hard,
        "metrics_blend":     m_blend,
        "metrics_ew":        m_ew,
        "metrics_global":    m_global,
        "metrics_oracle":    m_oracle,
        # Net return series
        "net_hard":          net_hard,
        "net_blend":         net_blend,
        "net_ew":            net_ew,
        "net_global":        net_global,
        "net_oracle":        net_oracle,
        # Test dates
        "test_dates":        dataset.dates[test_idx],
        "best_global_k":     best_global_k,
        # Diagnostics
        "tier_selection":    tier_selection,        # {regime_id: tier}
        "tier_distribution": tier_distribution,     # {regime_id: {tier: count in top-10}}
    }


# ---------------------------------------------------------------------------
# Summary and saving
# ---------------------------------------------------------------------------

def _load_plan13a_results(path: str) -> dict:
    """Load Plan 13a summary metrics if available."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    # fold → hier_sharpe
    return {int(row["fold"]): float(row["hier_sharpe"]) for _, row in df.iterrows()}


def save_results(fold_results: list, config: dict) -> None:
    """Save all diagnostic outputs to results/plan13b_bayesian_opt/."""
    out_dir = config["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "algo_sharpe_per_regime"), exist_ok=True)

    plan13a = _load_plan13a_results(config["plan13a_summary"])
    valid = [fr for fr in fold_results if fr is not None]

    # ── 1. summary_metrics.csv ────────────────────────────────────────────────
    rows = []
    for fr in valid:
        fold_id  = fr["fold_id"]
        year     = fr["fold_spec"]["test_start"][:4]
        p13a_s   = plan13a.get(fold_id, np.nan)
        rows.append({
            "fold":              fold_id,
            "test_year":         year,
            "hard_sharpe":       fr["metrics_hard"]["sharpe"],
            "blend_sharpe":      fr["metrics_blend"]["sharpe"],
            "ew_sharpe":         fr["metrics_ew"]["sharpe"],
            "global_sharpe":     fr["metrics_global"]["sharpe"],
            "oracle_sharpe":     fr["metrics_oracle"]["sharpe"],
            "plan13a_sharpe":    p13a_s,
            "hard_vs_ew":        fr["metrics_hard"]["sharpe"] - fr["metrics_ew"]["sharpe"],
            "blend_vs_ew":       fr["metrics_blend"]["sharpe"] - fr["metrics_ew"]["sharpe"],
            "hard_vs_plan13a":   (fr["metrics_hard"]["sharpe"] - p13a_s)
                                  if not np.isnan(p13a_s) else np.nan,
            "oracle_gap":        fr["metrics_oracle"]["sharpe"] - fr["metrics_hard"]["sharpe"],
            "hard_ann_return":   fr["metrics_hard"]["ann_return"],
            "blend_ann_return":  fr["metrics_blend"]["ann_return"],
            "ew_ann_return":     fr["metrics_ew"]["ann_return"],
            "hard_maxdd":        fr["metrics_hard"]["max_drawdown"],
            "blend_maxdd":       fr["metrics_blend"]["max_drawdown"],
            "ew_maxdd":          fr["metrics_ew"]["max_drawdown"],
            "hard_turnover":     fr["metrics_hard"].get("avg_daily_turnover", np.nan),
            "blend_turnover":    fr["metrics_blend"].get("avg_daily_turnover", np.nan),
        })
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)

    # ── 2. best_algo_per_regime.csv ───────────────────────────────────────────
    rows2 = []
    for fr in valid:
        fold_id = fr["fold_id"]
        algo_names = fr["algo_names"]
        n_tier1, n_tier2, _ = fr["tier_sizes"]
        for regime_id in [1, 2, 3, 4]:
            k = fr["best_algo"][regime_id]
            rows2.append({
                "fold":        fold_id,
                "test_year":   fr["fold_spec"]["test_start"][:4],
                "regime_id":   regime_id,
                "regime_name": REGIME_NAMES[regime_id],
                "best_algo_k": k,
                "best_algo_name": algo_names[k] if k < len(algo_names) else "?",
                "best_algo_tier": _get_tier_for_algo(k, n_tier1, n_tier2),
                "train_days":  fr["regime_day_counts"].get(regime_id, 0),
                "top3_algos":  str(fr["top_n_algos"].get(regime_id, [])),
            })
    pd.DataFrame(rows2).to_csv(
        os.path.join(out_dir, "best_algo_per_regime.csv"), index=False
    )

    # ── 3. algo_sharpe_per_regime/fold_XX_regime_sharpes.csv ─────────────────
    for fr in valid:
        fold_id    = fr["fold_id"]
        algo_names = fr["algo_names"]
        n_tier1, n_tier2, _ = fr["tier_sizes"]
        all_rows = []
        for regime_id in [1, 2, 3, 4]:
            sharpes = fr["sharpes_per_regime"].get(regime_id, [])
            for k, s in enumerate(sharpes):
                all_rows.append({
                    "algo_k":    k,
                    "algo_name": algo_names[k] if k < len(algo_names) else "?",
                    "tier":      _get_tier_for_algo(k, n_tier1, n_tier2),
                    "regime_id": regime_id,
                    "regime_name": REGIME_NAMES[regime_id],
                    "train_sharpe": s,
                })
        pd.DataFrame(all_rows).to_csv(
            os.path.join(out_dir, "algo_sharpe_per_regime",
                         f"fold_{fold_id:02d}_regime_sharpes.csv"),
            index=False,
        )

    # ── 4. regime_selection_stability.csv ────────────────────────────────────
    stability_rows = []
    for regime_id in [1, 2, 3, 4]:
        algo_selections = [fr["best_algo"][regime_id] for fr in valid
                           if regime_id in fr["best_algo"]]
        if not algo_selections:
            continue
        from collections import Counter
        cnt = Counter(algo_selections)
        most_common_k, most_common_count = cnt.most_common(1)[0]
        stability_rows.append({
            "regime_id":         regime_id,
            "regime_name":       REGIME_NAMES[regime_id],
            "n_folds":           len(algo_selections),
            "unique_algos":      len(cnt),
            "most_common_k":     most_common_k,
            "most_common_name":  valid[0]["algo_names"][most_common_k]
                                  if valid else "?",
            "most_common_count": most_common_count,
            "stability_pct":     round(most_common_count / len(algo_selections) * 100, 1),
        })
    pd.DataFrame(stability_rows).to_csv(
        os.path.join(out_dir, "regime_selection_stability.csv"), index=False
    )

    # ── 5. tier_distribution_per_regime.csv ──────────────────────────────────
    tier_dist_rows = []
    for fr in valid:
        fold_id = fr["fold_id"]
        for regime_id in [1, 2, 3, 4]:
            dist = fr["tier_distribution"].get(regime_id, {1: 0, 2: 0, 3: 0})
            tier_dist_rows.append({
                "fold":        fold_id,
                "test_year":   fr["fold_spec"]["test_start"][:4],
                "regime_id":   regime_id,
                "regime_name": REGIME_NAMES[regime_id],
                "tier1_in_top10": dist.get(1, 0),
                "tier2_in_top10": dist.get(2, 0),
                "tier3_in_top10": dist.get(3, 0),
                "best_tier":   fr["tier_selection"].get(regime_id, 1),
            })
    pd.DataFrame(tier_dist_rows).to_csv(
        os.path.join(out_dir, "tier_distribution_per_regime.csv"), index=False
    )

    print(f"\nAll results saved to: {out_dir}", flush=True)


def _print_summary(fold_results: list, config: dict) -> None:
    valid = [fr for fr in fold_results if fr is not None]
    if not valid:
        print("No valid folds.")
        return

    plan13a = _load_plan13a_results(config["plan13a_summary"])

    header = (
        f"{'Fold':>5}  {'Year':>5}  "
        f"{'Hard':>8}  {'Blend':>8}  {'EW':>8}  "
        f"{'Global':>8}  {'Oracle':>8}  {'13a':>8}  "
        f"{'Hrd-EW':>7}  {'Hrd-13a':>8}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * 75, flush=True)
    print("PLAN 13b: BAYESIAN OPTIMIZATION / EXHAUSTIVE SELECTION — RESULTS", flush=True)
    print("=" * 75, flush=True)
    print(header, flush=True)
    print(sep, flush=True)

    for fr in valid:
        fold_id  = fr["fold_id"]
        year     = fr["fold_spec"]["test_start"][:4]
        hs       = fr["metrics_hard"]["sharpe"]
        bs       = fr["metrics_blend"]["sharpe"]
        es       = fr["metrics_ew"]["sharpe"]
        gs       = fr["metrics_global"]["sharpe"]
        os_      = fr["metrics_oracle"]["sharpe"]
        p13a_s   = plan13a.get(fold_id, float("nan"))

        h_ew  = hs - es if not (np.isnan(hs) or np.isnan(es)) else float("nan")
        h_13a = hs - p13a_s if not np.isnan(p13a_s) else float("nan")

        def _fmt(v):
            return f"{v:+8.4f}" if not np.isnan(v) else "     nan"

        print(
            f"{fold_id:>5}  {year:>5}  "
            f"{_fmt(hs)}  {_fmt(bs)}  {_fmt(es)}  "
            f"{_fmt(gs)}  {_fmt(os_)}  {_fmt(p13a_s)}  "
            f"{_fmt(h_ew)}  {_fmt(h_13a)}",
            flush=True,
        )

    print(sep, flush=True)

    # Averages
    keys = ["metrics_hard", "metrics_blend", "metrics_ew", "metrics_global", "metrics_oracle"]
    avgs = {}
    for k in keys:
        vals = [fr[k]["sharpe"] for fr in valid if not np.isnan(fr[k]["sharpe"])]
        avgs[k] = float(np.mean(vals)) if vals else float("nan")

    p13a_vals = [plan13a[fr["fold_id"]] for fr in valid
                 if fr["fold_id"] in plan13a and not np.isnan(plan13a[fr["fold_id"]])]
    avg_13a = float(np.mean(p13a_vals)) if p13a_vals else float("nan")

    h_ew_avg  = avgs["metrics_hard"] - avgs["metrics_ew"]
    h_13a_avg = avgs["metrics_hard"] - avg_13a

    def _fmt(v):
        return f"{v:+8.4f}" if not np.isnan(v) else "     nan"

    print(
        f"{'AVG':>5}  {'':>5}  "
        f"{_fmt(avgs['metrics_hard'])}  {_fmt(avgs['metrics_blend'])}  {_fmt(avgs['metrics_ew'])}  "
        f"{_fmt(avgs['metrics_global'])}  {_fmt(avgs['metrics_oracle'])}  {_fmt(avg_13a)}  "
        f"{_fmt(h_ew_avg)}  {_fmt(h_13a_avg)}",
        flush=True,
    )
    print("=" * 75, flush=True)

    # Hard vs Blend wins
    hard_wins = sum(1 for fr in valid
                    if fr["metrics_hard"]["sharpe"] >= fr["metrics_blend"]["sharpe"])
    hard_beats_ew = sum(1 for fr in valid
                        if fr["metrics_hard"]["sharpe"] >= fr["metrics_ew"]["sharpe"])
    hard_beats_13a = sum(1 for fr in valid
                         if fr["fold_id"] in plan13a
                         and not np.isnan(plan13a[fr["fold_id"]])
                         and fr["metrics_hard"]["sharpe"] >= plan13a[fr["fold_id"]])

    print(f"\nDiagnostic Counts ({len(valid)} folds):", flush=True)
    print(f"  Hard >= Blend    : {hard_wins}/{len(valid)}", flush=True)
    print(f"  Hard >= EW       : {hard_beats_ew}/{len(valid)}", flush=True)
    print(f"  Hard >= Plan 13a : {hard_beats_13a}/{len(valid)} (where 13a available)", flush=True)

    # Regime selection stability
    print(f"\nRegime Selection Stability:", flush=True)
    from collections import Counter
    for regime_id in [1, 2, 3, 4]:
        sels = [fr["best_algo"][regime_id] for fr in valid
                if regime_id in fr["best_algo"]]
        if not sels:
            continue
        cnt = Counter(sels)
        mc_k, mc_c = cnt.most_common(1)[0]
        names = valid[0]["algo_names"] if valid else []
        mc_name = names[mc_k] if mc_k < len(names) else "?"
        print(
            f"  Regime {regime_id} ({REGIME_NAMES[regime_id]:7s}): "
            f"{len(cnt)} unique algos, most common=#{mc_k} ({mc_name}, "
            f"{mc_c}/{len(sels)} folds = {mc_c/len(sels)*100:.0f}%)",
            flush=True,
        )

    # Tier distribution (averaged across folds)
    print(f"\nTier of Best Algorithm Per Regime (across folds):", flush=True)
    for regime_id in [1, 2, 3, 4]:
        tiers = [fr["tier_selection"][regime_id] for fr in valid
                 if regime_id in fr["tier_selection"]]
        if not tiers:
            continue
        from collections import Counter
        tc = Counter(tiers)
        print(
            f"  Regime {regime_id} ({REGIME_NAMES[regime_id]:7s}): "
            f"Tier1={tc.get(1,0)}, Tier2={tc.get(2,0)}, Tier3={tc.get(3,0)}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(config: dict = CONFIG) -> list:
    np.random.seed(config["seed"])

    print("\n" + "=" * 70, flush=True)
    print("PLAN 13b: BAYESIAN OPTIMIZATION / EXHAUSTIVE SELECTION — FULL RUN", flush=True)
    print("=" * 70, flush=True)
    print(f"  kappa={config['kappa']}, top_n={config['top_n']}", flush=True)
    print(f"  Cache dir: {config['cache_dir']}", flush=True)
    print(f"  Output dir: {config['output_dir']}", flush=True)

    # ------------------------------------------------------------------ #
    # Load data                                                           #
    # ------------------------------------------------------------------ #
    print("\nLoading data ...", flush=True)
    data = load_data_extended()
    prices = data["prices"]
    vix    = data["vix"]
    print(f"  Prices: {prices.shape}  ({prices.index[0].date()} – {prices.index[-1].date()})")
    print(f"  VIX:    {len(vix)} rows")

    returns             = compute_returns(prices)
    full_asset_features = compute_asset_features(prices)
    regime_labels       = compute_regime_labels(vix)

    # Align to common dates
    common = prices.index.intersection(returns.index).intersection(regime_labels.index)
    prices              = prices.loc[common]
    returns             = returns.loc[common]
    regime_labels       = regime_labels.loc[common]
    full_asset_features = full_asset_features.loc[full_asset_features.index.intersection(common)]
    print(f"  Common dates: {len(common)}", flush=True)

    # ------------------------------------------------------------------ #
    # Generate folds                                                      #
    # ------------------------------------------------------------------ #
    wfv = WalkForwardValidator(
        train_years=config["train_years"],
        test_years=config["test_years"],
        step_years=config["step_years"],
        min_test_start=config["min_test_start"],
    )
    all_folds = wfv.generate_folds(data_end=config["data_end"])
    print(f"Generated {len(all_folds)} walk-forward folds", flush=True)

    # ------------------------------------------------------------------ #
    # Run all folds                                                       #
    # ------------------------------------------------------------------ #
    fold_results = []
    t_exp_start = time.time()

    for fold_spec in all_folds:
        fold_id = fold_spec["fold"]
        t_fold_start = time.time()
        result = run_fold(
            fold_id=fold_id,
            fold_spec=fold_spec,
            prices=prices,
            returns=returns,
            regime_labels=regime_labels,
            full_asset_features=full_asset_features,
            config=config,
        )
        t_fold_end = time.time()
        print(f"  Fold {fold_id} done in {t_fold_end - t_fold_start:.1f}s", flush=True)
        fold_results.append(result)

    t_exp_end = time.time()
    print(f"\nTotal experiment time: {(t_exp_end - t_exp_start) / 60:.1f} min", flush=True)

    # ------------------------------------------------------------------ #
    # Print summary and save                                              #
    # ------------------------------------------------------------------ #
    _print_summary(fold_results, config)
    save_results(fold_results, config)

    print("\nPLAN 13b COMPLETE.", flush=True)
    return fold_results


if __name__ == "__main__":
    run_experiment()
