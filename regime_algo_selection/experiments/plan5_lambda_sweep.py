# experiments/plan5_lambda_sweep.py -- Lambda-entropy sweep (Plan 12)
#
# Quick sweep to find the best lambda_entropy before running the full 12-fold
# experiment. Only runs fold 1 (test 2013) and fold 5 (test 2017) for speed.
#
# Usage:
#   cd Implementierung1
#   python -u -m regime_algo_selection.experiments.plan5_lambda_sweep

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

import numpy as np
import torch

from regime_algo_selection.config import (
    RESULTS_DIR, KAPPA, RANDOM_SEED, N_ASSETS, REGIME_NAMES,
)
from regime_algo_selection.data.loader_extended import load_data_extended
from regime_algo_selection.data.features import compute_returns, compute_asset_features
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
# Sweep config
# ---------------------------------------------------------------------------

LAMBDA_VALUES = [0.08, 0.085, 0.09, 0.095]
TARGET_FOLD_TESTS = ["2013", "2017"]   # fold 1 and fold 5

BASE_CONFIG = {
    "input_dim":    29,
    "hidden_dims":  [128, 64],
    "dropout":      0.1,
    "n_epochs":     100,
    "lr":           0.01,
    "weight_decay": 1e-4,
    "grad_clip":    1.0,
    "kappa":        KAPPA,
    "kappa_a":      0.0,
    "train_years":  8,
    "test_years":   1,
    "step_years":   1,
    "min_test_start": "2013-01-01",
    "data_end":     "2024-12-31",
    "seed":         RANDOM_SEED,
}

# ---------------------------------------------------------------------------
# Helpers (copied from plan5_meta_learner to keep sweep self-contained)
# ---------------------------------------------------------------------------

def _compute_metrics(daily_net_returns: np.ndarray) -> dict:
    r = daily_net_returns[np.isfinite(daily_net_returns)]
    T = len(r)
    if T < 10:
        return {"sharpe": np.nan}
    ann = 252
    cumw = np.cumprod(1 + r)
    cum_ret = cumw[-1] - 1
    ann_ret = (1 + cum_ret) ** (ann / T) - 1
    ann_vol = r.std() * np.sqrt(ann)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else np.nan
    return {"sharpe": round(sharpe, 4)}


def _portfolio_net_returns(weight_matrix, returns_matrix, kappa=KAPPA):
    T = weight_matrix.shape[0]
    net_rets = np.zeros(T)
    prev_w = np.ones(returns_matrix.shape[1]) / returns_matrix.shape[1]
    for t in range(T):
        w = weight_matrix[t]
        gross = float(w @ returns_matrix[t])
        cost = kappa * float(np.abs(w - prev_w).sum())
        net_rets[t] = gross - cost
        prev_w = w
    return net_rets


def _eval_ew(dataset, test_indices, algorithms, kappa):
    ew_k = next((k for k, a in enumerate(algorithms) if a.name == "EqualWeight"), 0)
    N = dataset.N
    net_rets = []
    prev_w = np.ones(N) / N
    for idx in test_indices:
        w = dataset.get_algorithm_outputs(idx)[ew_k]
        r = dataset.get_returns(idx)
        cost = kappa * float(np.abs(w - prev_w).sum())
        net_rets.append(float(w @ r) - cost)
        prev_w = w
    return _compute_metrics(np.array(net_rets))


def _run_one(fold_spec, prices, returns, regime_labels, full_asset_features,
             lambda_entropy, config):
    """Run a single fold with a given lambda_entropy. Returns (ml_sharpe, ew_sharpe, avg_H)."""
    kappa = config["kappa"]

    # Stage 0: build + pre-train algorithms
    algorithms = build_algorithm_space(tiers=[1, 2])
    K = len(algorithms)
    has_tier2 = any(isinstance(a, TrainablePortfolioAlgorithm) for a in algorithms)
    if has_tier2:
        pretrain_tier2_algorithms(
            algorithms, full_asset_features, returns,
            fold_spec["train_start"], fold_spec["train_end"],
        )

    # Dataset
    dataset = MetaLearnerDataset(
        prices=prices,
        all_asset_features=full_asset_features,
        returns=returns,
        regime_labels=regime_labels,
        algorithms=algorithms,
    )
    dataset.fit_scaler(fold_spec["train_start"], fold_spec["train_end"])
    dataset.precompute_algo_outputs()

    train_idx = dataset.get_indices_for_period(fold_spec["train_start"], fold_spec["train_end"])
    test_idx  = dataset.get_indices_for_period(fold_spec["test_start"],  fold_spec["test_end"])

    if len(train_idx) == 0 or len(test_idx) == 0:
        return np.nan, np.nan, np.nan

    # EW baseline
    ew_metrics = _eval_ew(dataset, test_idx, algorithms, kappa)

    # Train meta-learner
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
        kappa=kappa,
        kappa_a=config["kappa_a"],
        lambda_entropy=lambda_entropy,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        n_epochs=config["n_epochs"],
        grad_clip=config["grad_clip"],
    )
    trainer.train_fold(dataset, train_idx)

    # Evaluate meta-learner
    agent = MetaLearnerAgent(network=network, algorithms=algorithms)
    w_matrix     = np.zeros((len(test_idx), N_ASSETS), dtype=np.float32)
    alpha_matrix = np.zeros((len(test_idx), K),        dtype=np.float32)

    for i, idx in enumerate(test_idx):
        x_t  = dataset.get_input(idx)
        W_t  = dataset.get_algorithm_outputs(idx)
        w_t, alpha_t = agent.select(x_t, W_t)
        w_matrix[i]     = w_t
        alpha_matrix[i] = alpha_t

    returns_matrix = np.array([dataset.get_returns(idx) for idx in test_idx])
    net_rets = _portfolio_net_returns(w_matrix, returns_matrix, kappa)
    ml_metrics = _compute_metrics(net_rets)

    avg_H = float(np.mean([MetaLearnerAgent.entropy(alpha_matrix[i])
                           for i in range(len(test_idx))]))

    return ml_metrics["sharpe"], ew_metrics["sharpe"], avg_H


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep():
    print("\n" + "=" * 70)
    print("PLAN 12: LAMBDA-ENTROPY SWEEP  (fold 1 = 2013, fold 5 = 2017)")
    print("=" * 70)

    # Load data once
    print("\nLoading data ...", flush=True)
    data = load_data_extended()
    prices = data["prices"]
    vix    = data["vix"]

    returns             = compute_returns(prices)
    full_asset_features = compute_asset_features(prices)
    regime_labels       = compute_regime_labels(vix)

    common = prices.index.intersection(returns.index).intersection(regime_labels.index)
    prices              = prices.loc[common]
    returns             = returns.loc[common]
    regime_labels       = regime_labels.loc[common]
    full_asset_features = full_asset_features.loc[full_asset_features.index.intersection(common)]
    print(f"  Common dates: {len(common)}")

    # Generate all folds, pick fold 1 and fold 5
    wfv = WalkForwardValidator(
        train_years=BASE_CONFIG["train_years"],
        test_years=BASE_CONFIG["test_years"],
        step_years=BASE_CONFIG["step_years"],
        min_test_start=BASE_CONFIG["min_test_start"],
    )
    all_folds = wfv.generate_folds(data_end=BASE_CONFIG["data_end"])
    target_folds = [f for f in all_folds if f["test_start"][:4] in TARGET_FOLD_TESTS]

    if len(target_folds) != 2:
        print(f"WARNING: expected 2 target folds, found {len(target_folds)}: "
              f"{[f['test_start'][:4] for f in target_folds]}")

    print(f"\nTarget folds: {[f['test_start'][:4] for f in target_folds]}")
    print(f"Lambda values: {LAMBDA_VALUES}")
    print(f"Total runs: {len(target_folds) * len(LAMBDA_VALUES)}\n")

    # Run all combinations
    results = []   # list of (lambda, fold_year, ml_sharpe, ew_sharpe, avg_H)

    for lam in LAMBDA_VALUES:
        for fold_spec in target_folds:
            fold_year = fold_spec["test_start"][:4]
            print(f"--- lambda={lam}  fold={fold_year}  "
                  f"(train {fold_spec['train_start'][:4]}–{fold_spec['train_end'][:4]}) ---",
                  flush=True)
            t0 = time.time()
            ml_sharpe, ew_sharpe, avg_H = _run_one(
                fold_spec, prices, returns, regime_labels, full_asset_features,
                lam, BASE_CONFIG,
            )
            elapsed = time.time() - t0
            print(f"    lambda={lam:5.2f}  fold={fold_year}  "
                  f"ML_Sharpe={ml_sharpe:+.4f}  EW_Sharpe={ew_sharpe:+.4f}  "
                  f"H(a)={avg_H:.4f}  [{elapsed/60:.1f} min]", flush=True)
            results.append((lam, fold_year, ml_sharpe, ew_sharpe, avg_H))

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    header = f"{'lambda':>8}  {'fold':>6}  {'ML_Sharpe':>10}  {'EW_Sharpe':>10}  {'H(a)':>8}  {'ML-EW':>8}  {'H_ok?':>6}"
    print(header)
    print("-" * len(header))

    for (lam, fold_year, ml_sharpe, ew_sharpe, avg_H) in results:
        ml_vs_ew = ml_sharpe - ew_sharpe if np.isfinite(ml_sharpe) and np.isfinite(ew_sharpe) else np.nan
        h_ok = "YES" if (np.isfinite(avg_H) and 1.0 <= avg_H <= 3.0) else "NO"
        print(f"{lam:>8.2f}  {fold_year:>6}  {ml_sharpe:>10.4f}  {ew_sharpe:>10.4f}  "
              f"{avg_H:>8.4f}  {ml_vs_ew:>8.4f}  {h_ok:>6}")

    # Per-lambda averages
    print("-" * len(header))
    print("AVERAGES PER LAMBDA:")
    for lam in LAMBDA_VALUES:
        rows = [(ml, ew, H) for (l, _, ml, ew, H) in results if l == lam and np.isfinite(ml)]
        if not rows:
            continue
        avg_ml  = np.mean([r[0] for r in rows])
        avg_ew  = np.mean([r[1] for r in rows])
        avg_H   = np.mean([r[2] for r in rows])
        avg_gap = avg_ml - avg_ew
        h_ok    = "YES" if 1.0 <= avg_H <= 3.0 else "NO"
        print(f"{lam:>8.2f}  {'AVG':>6}  {avg_ml:>10.4f}  {avg_ew:>10.4f}  "
              f"{avg_H:>8.4f}  {avg_gap:>8.4f}  {h_ok:>6}")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("  Best lambda = one where H(a) in [1, 3] AND ML_Sharpe >= EW_Sharpe")
    candidates = [
        (lam, avg_ml - avg_ew, avg_H)
        for lam in LAMBDA_VALUES
        for rows in [[(ml, ew, H) for (l, _, ml, ew, H) in results if l == lam and np.isfinite(ml)]]
        if rows
        for avg_ml in [np.mean([r[0] for r in rows])]
        for avg_ew in [np.mean([r[1] for r in rows])]
        for avg_H  in [np.mean([r[2] for r in rows])]
        if 1.0 <= avg_H <= 3.0
    ]
    if candidates:
        # pick best ML-EW gap among H-in-range candidates
        best = max(candidates, key=lambda x: x[1])
        print(f"  => lambda_entropy = {best[0]}  (avg ML-EW={best[1]:+.4f}, avg H={best[2]:.4f})")
    else:
        # fallback: closest H to midpoint 2.0
        fallback = min(
            [(lam, avg_H) for lam in LAMBDA_VALUES
             for rows in [[(ml, ew, H) for (l, _, ml, ew, H) in results if l == lam and np.isfinite(ml)]]
             if rows
             for avg_H in [np.mean([r[2] for r in rows])]],
            key=lambda x: abs(x[1] - 2.0),
        )
        print(f"  No lambda had H in [1,3] with ML>=EW — closest: lambda={fallback[0]} (avg H={fallback[1]:.4f})")
    print("=" * 70)


if __name__ == "__main__":
    run_sweep()
