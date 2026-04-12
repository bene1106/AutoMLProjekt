# experiments/plan13a_lambda_sweep.py -- Plan 13a: lambda_spec sweep on Fold 1
#
# Motivation: Fold 1 sanity check showed H(g1/g2/g3) near maximum entropy,
# meaning specialists are nearly uniform (lambda_spec=0.05 is too high).
# The tier selector worked well (H(beta)=0.779).
#
# This script precomputes algorithm outputs ONCE for Fold 1, then tests
# lambda_spec in {0.005, 0.01, 0.02, 0.03} with lambda_tier fixed at 0.05.
# Each sweep run re-initializes the network with the same random seed.
#
# Usage:
#   cd Implementierung1
#   python -u -m regime_algo_selection.experiments.plan13a_lambda_sweep

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
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from regime_algo_selection.config import (
    RESULTS_DIR, KAPPA, REGIME_NAMES, RANDOM_SEED,
)
from regime_algo_selection.data.loader_extended import load_data_extended
from regime_algo_selection.data.features import compute_returns, compute_asset_features
from regime_algo_selection.regimes.ground_truth import compute_regime_labels
from regime_algo_selection.algorithms.tier1_heuristics import build_algorithm_space
from regime_algo_selection.algorithms.base import TrainablePortfolioAlgorithm
from regime_algo_selection.algorithms.stage0 import pretrain_algorithms
from regime_algo_selection.evaluation.walk_forward import WalkForwardValidator

from regime_algo_selection.meta_learner.dataset import MetaLearnerDataset
from regime_algo_selection.meta_learner.hierarchical_network import HierarchicalMetaLearner
from regime_algo_selection.meta_learner.hierarchical_trainer import HierarchicalTrainer

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

LAMBDA_SPEC_VALUES = [0.005, 0.01, 0.02, 0.03]
LAMBDA_TIER_FIXED  = 0.05

SWEEP_CONFIG = {
    "input_dim":          29,
    "selector_hidden":    [64, 32],
    "specialist_hidden":  [64, 32],
    "dropout":            0.1,
    "specialist_lr":      0.005,
    "specialist_epochs":  30,
    "selector_lr":        0.005,
    "selector_epochs":    20,
    "lambda_tier":        LAMBDA_TIER_FIXED,
    "kappa":              KAPPA,
    "kappa_a":            0.0,
    "tiers":              [1, 2, 3],
    "seed":               RANDOM_SEED,
    # Walk-forward (only Fold 1, test 2013)
    "train_years":        8,
    "test_years":         1,
    "step_years":         1,
    "min_test_start":     "2013-01-01",
    "data_end":           "2024-12-31",
}

SWEEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "plan13a_hierarchical")
os.makedirs(SWEEP_RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_sharpe(daily_net_returns: np.ndarray) -> float:
    r = daily_net_returns[np.isfinite(daily_net_returns)]
    if len(r) < 10:
        return np.nan
    ann = 252
    cumw = np.cumprod(1 + r)
    ann_ret = (cumw[-1]) ** (ann / len(r)) - 1
    ann_vol = r.std() * np.sqrt(ann)
    return float(ann_ret / ann_vol) if ann_vol > 1e-12 else np.nan


def _eval_ew(dataset, test_idx, kappa):
    ew_k = 0   # EqualWeight is always first
    prev_w = np.ones(dataset.N) / dataset.N
    rets = []
    for idx in test_idx:
        w = dataset.get_algorithm_outputs(idx)[ew_k]
        r = dataset.get_returns(idx)
        gross = float(w @ r)
        cost = kappa * float(np.abs(w - prev_w).sum())
        rets.append(gross - cost)
        prev_w = w
    return _compute_sharpe(np.array(rets))


def _eval_hierarchical(model, dataset, test_idx, tier_indices, kappa):
    """Evaluate the trained hierarchical model on test set."""
    model.eval()
    N = dataset.N
    prev_w = np.ones(N) / N
    rets = []

    H_beta_list  = []
    H_gamma_list = [[] for _ in range(3)]

    with torch.no_grad():
        for idx in test_idx:
            X_t = torch.tensor(dataset.get_input(idx), dtype=torch.float32)
            W_all = dataset.get_algorithm_outputs(idx)

            alpha_t, beta_t, gammas = model(X_t)

            beta_np   = beta_t.numpy()
            gammas_np = [g.numpy() for g in gammas]

            # Composite portfolio
            W_tensor = torch.tensor(W_all, dtype=torch.float32)
            w_t = torch.matmul(alpha_t, W_tensor).numpy()
            w_t = np.clip(w_t, 0.0, None)
            s = w_t.sum()
            w_t = w_t / s if s > 1e-12 else np.ones(N) / N

            r = dataset.get_returns(idx)
            gross = float(w_t @ r)
            cost = kappa * float(np.abs(w_t - prev_w).sum())
            rets.append(gross - cost)
            prev_w = w_t

            # Entropies
            def H(p):
                p = np.clip(p, 1e-10, 1.0)
                return float(-np.sum(p * np.log(p)))

            H_beta_list.append(H(beta_np))
            for f in range(3):
                H_gamma_list[f].append(H(gammas_np[f]))

    return {
        "sharpe":  _compute_sharpe(np.array(rets)),
        "H_beta":  float(np.mean(H_beta_list)),
        "H_gamma": [float(np.mean(H_gamma_list[f])) for f in range(3)],
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_lambda_sweep() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. Install with: pip install torch")

    print("\n" + "=" * 70, flush=True)
    print("PLAN 13a: lambda_spec SWEEP — Fold 1 (test 2013)", flush=True)
    print(f"  lambda_spec values: {LAMBDA_SPEC_VALUES}", flush=True)
    print(f"  lambda_tier fixed:  {LAMBDA_TIER_FIXED}", flush=True)
    print(f"  specialist_epochs:  {SWEEP_CONFIG['specialist_epochs']}", flush=True)
    print(f"  selector_epochs:    {SWEEP_CONFIG['selector_epochs']}", flush=True)
    print("=" * 70, flush=True)

    # ------------------------------------------------------------------ #
    # 1. Load data                                                        #
    # ------------------------------------------------------------------ #
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
    full_asset_features = full_asset_features.loc[
        full_asset_features.index.intersection(common)
    ]
    print(f"  Common dates: {len(common)}", flush=True)

    # ------------------------------------------------------------------ #
    # 2. Get Fold 1 spec                                                  #
    # ------------------------------------------------------------------ #
    wfv = WalkForwardValidator(
        train_years=SWEEP_CONFIG["train_years"],
        test_years=SWEEP_CONFIG["test_years"],
        step_years=SWEEP_CONFIG["step_years"],
        min_test_start=SWEEP_CONFIG["min_test_start"],
    )
    all_folds = wfv.generate_folds(data_end=SWEEP_CONFIG["data_end"])
    fold1 = next(f for f in all_folds if f["fold"] == 1)
    train_start = fold1["train_start"]
    train_end   = fold1["train_end"]
    test_start  = fold1["test_start"]
    test_end    = fold1["test_end"]
    print(f"  Fold 1: train {train_start[:4]}-{train_end[:4]}, test {test_start[:4]}",
          flush=True)

    # ------------------------------------------------------------------ #
    # 3. Stage 0: build and pre-train algorithms ONCE                     #
    # ------------------------------------------------------------------ #
    print("\nStage 0: building and pre-training algorithms ...", flush=True)
    t0 = time.time()
    algorithms = build_algorithm_space(tiers=SWEEP_CONFIG["tiers"])
    K = len(algorithms)

    from regime_algo_selection.algorithms.tier3_nonlinear import Tier3Algorithm
    n_tier1 = sum(1 for a in algorithms if not isinstance(a, TrainablePortfolioAlgorithm))
    n_tier3 = sum(1 for a in algorithms if isinstance(a, Tier3Algorithm))
    n_tier2 = sum(1 for a in algorithms if isinstance(a, TrainablePortfolioAlgorithm)) - n_tier3
    tier_sizes = [n_tier1, n_tier2, n_tier3]
    tier_algorithm_indices = [
        list(range(n_tier1)),
        list(range(n_tier1, n_tier1 + n_tier2)),
        list(range(n_tier1 + n_tier2, K)),
    ]
    print(f"  K={K}: {n_tier1} Tier1, {n_tier2} Tier2, {n_tier3} Tier3", flush=True)

    pretrain_algorithms(algorithms, full_asset_features, returns, train_start, train_end)

    # ------------------------------------------------------------------ #
    # 4. Build dataset and precompute outputs ONCE                        #
    # ------------------------------------------------------------------ #
    print("\nBuilding MetaLearnerDataset and precomputing outputs (ONCE) ...",
          flush=True)
    dataset = MetaLearnerDataset(
        prices=prices,
        all_asset_features=full_asset_features,
        returns=returns,
        regime_labels=regime_labels,
        algorithms=algorithms,
    )
    dataset.fit_scaler(train_start, train_end)
    dataset.precompute_algo_outputs()

    train_idx = dataset.get_indices_for_period(train_start, train_end)
    test_idx  = dataset.get_indices_for_period(test_start,  test_end)
    print(f"  Train: {len(train_idx)} days | Test: {len(test_idx)} days", flush=True)

    t1 = time.time()
    print(f"  Stage 0 + precompute done in {t1 - t0:.1f}s", flush=True)

    # EW baseline (computed once)
    ew_sharpe = _eval_ew(dataset, test_idx, SWEEP_CONFIG["kappa"])
    print(f"  EW Sharpe (baseline): {ew_sharpe:+.4f}", flush=True)

    # ------------------------------------------------------------------ #
    # 5. Sweep over lambda_spec values                                    #
    # ------------------------------------------------------------------ #
    results = []

    for lam_spec in LAMBDA_SPEC_VALUES:
        print(f"\n{'─' * 60}", flush=True)
        print(f"  lambda_spec = {lam_spec}  (lambda_tier = {LAMBDA_TIER_FIXED})",
              flush=True)

        # Fresh model with fixed seed
        torch.manual_seed(SWEEP_CONFIG["seed"])
        np.random.seed(SWEEP_CONFIG["seed"])

        model = HierarchicalMetaLearner(
            input_dim=SWEEP_CONFIG["input_dim"],
            tier_sizes=tier_sizes,
            selector_hidden=SWEEP_CONFIG["selector_hidden"],
            specialist_hidden=SWEEP_CONFIG["specialist_hidden"],
            dropout=SWEEP_CONFIG["dropout"],
        )

        trainer = HierarchicalTrainer(
            model=model,
            tier_algorithm_indices=tier_algorithm_indices,
            kappa=SWEEP_CONFIG["kappa"],
            kappa_a=SWEEP_CONFIG["kappa_a"],
            specialist_lr=SWEEP_CONFIG["specialist_lr"],
            selector_lr=SWEEP_CONFIG["selector_lr"],
            specialist_epochs=SWEEP_CONFIG["specialist_epochs"],
            selector_epochs=SWEEP_CONFIG["selector_epochs"],
            lambda_spec=lam_spec,
            lambda_tier=SWEEP_CONFIG["lambda_tier"],
        )

        t_train = time.time()
        trainer.train_fold(dataset, train_idx)
        t_done = time.time()
        print(f"  Training done in {t_done - t_train:.1f}s", flush=True)

        metrics = _eval_hierarchical(
            model, dataset, test_idx, tier_algorithm_indices, SWEEP_CONFIG["kappa"]
        )

        print(f"  Sharpe  = {metrics['sharpe']:+.4f}  (EW = {ew_sharpe:+.4f})", flush=True)
        print(f"  H(beta) = {metrics['H_beta']:.4f}  |  "
              f"H(g1) = {metrics['H_gamma'][0]:.4f}  "
              f"H(g2) = {metrics['H_gamma'][1]:.4f}  "
              f"H(g3) = {metrics['H_gamma'][2]:.4f}",
              flush=True)

        results.append({
            "lambda_spec": lam_spec,
            "lambda_tier": LAMBDA_TIER_FIXED,
            "sharpe":      metrics["sharpe"],
            "ew_sharpe":   ew_sharpe,
            "H_beta":      metrics["H_beta"],
            "H_g1":        metrics["H_gamma"][0],
            "H_g2":        metrics["H_gamma"][1],
            "H_g3":        metrics["H_gamma"][2],
        })

    # ------------------------------------------------------------------ #
    # 6. Summary table                                                    #
    # ------------------------------------------------------------------ #
    H_max = [float(np.log(s)) for s in tier_sizes]

    print("\n" + "=" * 70, flush=True)
    print("LAMBDA_spec SWEEP — SUMMARY (Fold 1, test 2013)", flush=True)
    print("=" * 70, flush=True)
    header = (
        f"{'lam_spec':>10}  {'H(beta)':>7}  {'H(g1)':>7}  "
        f"{'H(g2)':>7}  {'H(g3)':>7}  {'Sharpe':>8}  {'EW_Sharpe':>9}"
    )
    sep = "-" * len(header)
    print(header, flush=True)
    print(sep, flush=True)

    for r in results:
        print(
            f"{r['lambda_spec']:>10.4f}  "
            f"{r['H_beta']:>7.4f}  {r['H_g1']:>7.4f}  "
            f"{r['H_g2']:>7.4f}  {r['H_g3']:>7.4f}  "
            f"{r['sharpe']:>8.4f}  {r['ew_sharpe']:>9.4f}",
            flush=True,
        )

    print(sep, flush=True)
    print(f"  H_max: T1={H_max[0]:.3f}  T2={H_max[1]:.3f}  T3={H_max[2]:.3f}", flush=True)
    print(f"  Target H(beta) in [0.1, 1.0]", flush=True)
    print(f"  Target H(g_f) in [0.3, H_max_f - 0.3]", flush=True)

    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(SWEEP_RESULTS_DIR, "lambda_sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved sweep results to: {csv_path}", flush=True)
    print("\nLAMBDA SWEEP COMPLETE.", flush=True)


if __name__ == "__main__":
    run_lambda_sweep()
