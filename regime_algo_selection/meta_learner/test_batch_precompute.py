# meta_learner/test_batch_precompute.py
#
# Correctness verification: batch_precompute_algo_outputs() must produce
# outputs numerically identical to precompute_algo_outputs() when applied
# to the same algorithm instances.
#
# Also measures the speedup on a larger window (all algorithms, 200 days).
#
# Usage:
#   cd Implementierung1
#   python -u -m regime_algo_selection.meta_learner.test_batch_precompute

import sys
import time
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

import numpy as np
import pandas as pd

from regime_algo_selection.data.loader_extended import load_data_extended
from regime_algo_selection.data.features import compute_returns, compute_asset_features
from regime_algo_selection.regimes.ground_truth import compute_regime_labels
from regime_algo_selection.algorithms.tier1_heuristics import build_algorithm_space
from regime_algo_selection.algorithms.stage0 import pretrain_algorithms
from regime_algo_selection.meta_learner.dataset import MetaLearnerDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_START  = "2005-01-01"
TRAIN_END    = "2011-12-31"
ABS_TOL      = 1e-5   # max acceptable element-wise absolute difference (float32)

# Test 1: small correctness check (fast, 5 algos, 100 days)
SMALL_N_DAYS  = 100
SMALL_N_ALGOS = 5

# Test 2: full-scale timing benchmark (all 117 algos, 200 days)
LARGE_N_DAYS  = 200
LARGE_TIERS   = [1, 2, 3]

# ---------------------------------------------------------------------------
# Helper: pick a diverse algorithm sample
# ---------------------------------------------------------------------------

def _pick_sample(algos, n):
    """Pick n algorithms covering different families."""
    from regime_algo_selection.algorithms.tier1_heuristics import (
        EqualWeight, MinimumVariance, RiskParity, TrendFollowing, Momentum,
    )
    from regime_algo_selection.algorithms.tier2_linear import RidgePortfolio
    from regime_algo_selection.algorithms.tier3_nonlinear import Tier3Algorithm

    want_types = [
        EqualWeight, MinimumVariance, RiskParity, TrendFollowing, RidgePortfolio,
        Momentum, Tier3Algorithm,
    ]
    selected = []
    used = set()
    for wt in want_types:
        for a in algos:
            if isinstance(a, wt) and id(a) not in used:
                selected.append(a)
                used.add(id(a))
                break
    # Pad with remaining
    for a in algos:
        if len(selected) >= n:
            break
        if id(a) not in used:
            selected.append(a)
            used.add(id(a))
    return selected[:n]


# ---------------------------------------------------------------------------
# Helper: run one method on a restricted date window
# ---------------------------------------------------------------------------

def _run_on_subset(dataset, subset_dates, use_batch: bool):
    """
    Temporarily restrict dataset.dates to subset_dates, run precompute,
    return (output_array, elapsed_seconds).
    """
    orig_dates = dataset.dates
    orig_map   = dataset._date_to_idx

    dataset.dates        = subset_dates
    dataset._date_to_idx = {t: i for i, t in enumerate(subset_dates)}

    t0 = time.time()
    if use_batch:
        dataset.batch_precompute_algo_outputs()
    else:
        dataset.precompute_algo_outputs()
    elapsed = time.time() - t0

    result = dataset._algo_outputs.copy()

    # Restore
    dataset.dates         = orig_dates
    dataset._date_to_idx  = orig_map
    dataset._algo_outputs = None
    return result, elapsed


# ---------------------------------------------------------------------------
# Test 1: correctness on 5 algorithms × 100 days
# ---------------------------------------------------------------------------

def test_correctness(prices, returns, full_asset_features, regime_labels):
    sep = "-" * 65
    print(f"\n{sep}", flush=True)
    print("TEST 1 — Correctness (5 algos × 100 days)", flush=True)
    print(sep, flush=True)

    # Build ONE set of algorithm instances, pre-train once
    algos = _pick_sample(build_algorithm_space(tiers=[1, 2, 3]), n=SMALL_N_ALGOS)
    print(f"  Algorithms: {[a.name for a in algos]}", flush=True)
    pretrain_algorithms(algos, full_asset_features, returns, TRAIN_START, TRAIN_END)

    dataset = MetaLearnerDataset(
        prices, full_asset_features, returns, regime_labels, algos
    )
    dataset.fit_scaler(TRAIN_START, TRAIN_END)

    train_idx = dataset.get_indices_for_period(TRAIN_START, TRAIN_END)
    idx_subset = train_idx[60: 60 + SMALL_N_DAYS]
    subset_dates = dataset.dates[idx_subset]
    print(f"  Window: {len(subset_dates)} days "
          f"({subset_dates[0].date()} – {subset_dates[-1].date()})", flush=True)

    # Run original on the SAME dataset instance
    print("\n  Running original ...", flush=True)
    out_orig, t_orig = _run_on_subset(dataset, subset_dates, use_batch=False)

    # Run batch on the SAME dataset instance (same fitted algorithms)
    print("  Running batch   ...", flush=True)
    out_batch, t_batch = _run_on_subset(dataset, subset_dates, use_batch=True)

    # Compare
    K = len(algos)
    print(f"\n  {'k':>3}  {'Algorithm':<45}  {'MaxAbsDiff':>12}  OK?", flush=True)
    print("  " + "-" * 70, flush=True)
    all_ok = True
    for k in range(K):
        diff = np.abs(out_orig[:, k, :] - out_batch[:, k, :])
        max_diff = float(diff.max())
        ok = max_diff < ABS_TOL
        if not ok:
            all_ok = False
        mark = "OK" if ok else "FAIL <<"
        print(f"  {k:>3}  {algos[k].name:<45}  {max_diff:>12.2e}  {mark}", flush=True)

    global_max = float(np.abs(out_orig - out_batch).max())
    print(f"\n  Global max absolute diff : {global_max:.2e}  (tolerance {ABS_TOL:.0e})", flush=True)
    print(f"  Original: {t_orig:.2f}s  |  Batch: {t_batch:.2f}s  "
          f"|  Speed-up: {t_orig/max(t_batch,0.001):.2f}x", flush=True)

    # Weights-sum-to-1 sanity
    for label, arr in [("original", out_orig), ("batch", out_batch)]:
        sums = arr.sum(axis=2)
        print(f"  Weight sums ({label}): "
              f"[{sums.min():.6f}, {sums.max():.6f}]", flush=True)

    if all_ok:
        print(f"\n  TEST 1 PASSED", flush=True)
    else:
        print(f"\n  TEST 1 FAILED", flush=True)
    return all_ok


# ---------------------------------------------------------------------------
# Test 2: timing benchmark on all 117 algorithms × 200 days
# ---------------------------------------------------------------------------

def test_timing(prices, returns, full_asset_features, regime_labels):
    sep = "-" * 65
    print(f"\n{sep}", flush=True)
    print(f"TEST 2 — Timing benchmark (all 117 algos × {LARGE_N_DAYS} days)", flush=True)
    print(sep, flush=True)

    algos = build_algorithm_space(tiers=LARGE_TIERS)
    print(f"  Pre-training {sum(1 for a in algos if hasattr(a,'_is_fitted'))} trainable algorithms ...", flush=True)
    pretrain_algorithms(algos, full_asset_features, returns, TRAIN_START, TRAIN_END)

    dataset = MetaLearnerDataset(
        prices, full_asset_features, returns, regime_labels, algos
    )
    dataset.fit_scaler(TRAIN_START, TRAIN_END)

    train_idx = dataset.get_indices_for_period(TRAIN_START, TRAIN_END)
    idx_subset = train_idx[60: 60 + LARGE_N_DAYS]
    subset_dates = dataset.dates[idx_subset]
    print(f"  Window: {len(subset_dates)} days "
          f"({subset_dates[0].date()} – {subset_dates[-1].date()})", flush=True)

    K = len(algos)
    N_days = len(subset_dates)
    print(f"  Total algo-day calls: {K * N_days:,}", flush=True)

    print("\n  Running original ...", flush=True)
    out_orig, t_orig = _run_on_subset(dataset, subset_dates, use_batch=False)

    print("  Running batch   ...", flush=True)
    out_batch, t_batch = _run_on_subset(dataset, subset_dates, use_batch=True)

    # Quick correctness check (global)
    global_max = float(np.abs(out_orig - out_batch).max())
    ok = global_max < ABS_TOL

    print(f"\n  Original time : {t_orig:.1f}s  ({t_orig/60:.2f} min)", flush=True)
    print(f"  Batch time    : {t_batch:.1f}s  ({t_batch/60:.2f} min)", flush=True)
    print(f"  Speed-up      : {t_orig/max(t_batch,0.001):.2f}x", flush=True)
    print(f"  Global max diff: {global_max:.2e}  ({'OK' if ok else 'FAIL'})", flush=True)

    # Per-tier breakdown
    from regime_algo_selection.algorithms.base import TrainablePortfolioAlgorithm
    from regime_algo_selection.algorithms.tier3_nonlinear import Tier3Algorithm
    tier1_ks = [k for k, a in enumerate(algos) if not isinstance(a, TrainablePortfolioAlgorithm)]
    tier23_ks = [k for k, a in enumerate(algos) if isinstance(a, TrainablePortfolioAlgorithm)]

    if tier1_ks and tier23_ks:
        d1  = np.abs(out_orig[:, tier1_ks, :] - out_batch[:, tier1_ks, :]).max()
        d23 = np.abs(out_orig[:, tier23_ks, :] - out_batch[:, tier23_ks, :]).max()
        print(f"\n  Max diff Tier 1  : {d1:.2e}", flush=True)
        print(f"  Max diff Tier 2+3: {d23:.2e}", flush=True)

    projected_orig  = t_orig  / N_days * 2000 / 3600
    projected_batch = t_batch / N_days * 2000 / 3600
    print(f"\n  Projected per-fold time @ 2000 days:", flush=True)
    print(f"    Original : {projected_orig:.2f} h", flush=True)
    print(f"    Batch    : {projected_batch:.2f} h", flush=True)
    print(f"    Saved    : {projected_orig - projected_batch:.2f} h", flush=True)

    if ok:
        print(f"\n  TEST 2 PASSED", flush=True)
    else:
        print(f"\n  TEST 2 FAILED (diff exceeds tolerance)", flush=True)
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sep = "=" * 65
    print(f"\n{sep}", flush=True)
    print("BATCH PRECOMPUTE TEST SUITE", flush=True)
    print(sep, flush=True)

    print("\nLoading data ...", flush=True)
    data   = load_data_extended()
    prices = data["prices"]
    vix    = data["vix"]

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

    ok1 = test_correctness(prices, returns, full_asset_features, regime_labels)
    ok2 = test_timing(prices, returns, full_asset_features, regime_labels)

    print(f"\n{'=' * 65}", flush=True)
    if ok1 and ok2:
        print("ALL TESTS PASSED", flush=True)
    else:
        print("SOME TESTS FAILED", flush=True)
    print("=" * 65, flush=True)

    return ok1 and ok2


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
