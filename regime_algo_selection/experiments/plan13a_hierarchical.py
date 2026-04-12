# experiments/plan13a_hierarchical.py -- Plan 13a: Hierarchical Meta-Learner
#
# Implements a two-level hierarchical meta-learner over a 117-algorithm space
# (48 Tier 1 + 33 Tier 2 + 36 Tier 3) using oracle regime labels.
#
# Pipeline per fold:
#   Stage 0 : Pre-train Tier 2+3 algorithms on training block
#   Stage 1 : Precompute all 117 algorithm outputs for train + test
#   Phase A  : Train each TierSpecialist independently on its tier's algorithms
#   Phase B  : Freeze specialists, train TierSelector on blended portfolio
#   Eval     : Evaluate hierarchical model + baselines on test year
#
# Sanity check (THIS RUN): Fold 1 (test 2013) and Fold 6 (test 2018) only.
#
# Usage:
#   cd Implementierung1
#   python -u -m regime_algo_selection.experiments.plan13a_hierarchical

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

# ---- project imports -------------------------------------------------------
from regime_algo_selection.config import (
    RESULTS_DIR, KAPPA, REGIME_NAMES, N_REGIMES, ASSETS, N_ASSETS, RANDOM_SEED,
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
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # Network architecture
    "input_dim": 29,           # 25 asset features + 4 regime one-hot
    "selector_hidden": [64, 32],
    "specialist_hidden": [64, 32],
    "dropout": 0.1,

    # Training Phase A (Specialists)
    "specialist_lr": 0.005,
    "specialist_epochs": 80,
    "lambda_spec": 0.05,

    # Training Phase B (Tier Selector)
    "selector_lr": 0.005,
    "selector_epochs": 50,
    "lambda_tier": 0.05,

    # Costs (same as Plan 12 for comparability)
    "kappa": KAPPA,            # 0.001
    "kappa_a": 0.0,

    # Walk-forward
    "train_years": 8,
    "test_years": 1,
    "step_years": 1,
    "min_test_start": "2013-01-01",
    "data_end": "2024-12-31",

    # Algorithm space
    "tiers": [1, 2, 3],

    # Random seed
    "seed": RANDOM_SEED,
}

# Sanity check config: only 2 folds, reduced epochs
SANITY_CONFIG = {
    **CONFIG,
    "specialist_epochs": 20,
    "selector_epochs": 15,
}

PLAN13A_RESULTS_DIR = os.path.join(RESULTS_DIR, "plan13a_hierarchical")
os.makedirs(PLAN13A_RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Metric helpers  (same as Plan 12)
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
        "sharpe":       round(sharpe,   4),
        "sortino":      round(sortino,  4),
        "ann_return":   round(ann_ret * 100, 2),
        "ann_vol":      round(ann_vol  * 100, 2),
        "max_drawdown": round(max_dd   * 100, 2),
        "n_days":       T,
    }


def _eval_strategy_on_test(
    dataset: MetaLearnerDataset,
    test_indices: np.ndarray,
    weight_fn,
    kappa: float = KAPPA,
) -> tuple:
    """Evaluate any strategy on the test set by computing daily net returns."""
    N = dataset.N
    net_rets = []
    turnovers = []
    prev_w = np.ones(N) / N

    for idx in test_indices:
        w = weight_fn(idx)
        w = np.where(np.isfinite(w), w, 0.0)
        w = np.clip(w, 0.0, None)
        s = w.sum()
        w = w / s if s > 1e-12 else np.ones(N) / N

        r = dataset.get_returns(idx)
        gross = float(w @ r)
        cost = kappa * float(np.abs(w - prev_w).sum())
        net_rets.append(gross - cost)
        turnovers.append(float(np.abs(w - prev_w).sum()))
        prev_w = w

    arr = np.array(net_rets)
    metrics = _compute_metrics(arr)
    metrics["avg_daily_turnover"] = round(float(np.mean(turnovers)), 6) if turnovers else np.nan
    return metrics, arr


# ---------------------------------------------------------------------------
# Entropy helper
# ---------------------------------------------------------------------------

def _entropy(p: np.ndarray) -> float:
    """Shannon entropy H(p) = -Σ p_i log(p_i)."""
    p = np.clip(p, 1e-10, 1.0)
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# Per-fold runner
# ---------------------------------------------------------------------------

def _build_fresh_algorithms(tiers):
    """Build a fresh set of algorithm instances (required per fold for Tier 2+3)."""
    return build_algorithm_space(tiers=tiers)


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

    # ------------------------------------------------------------------ #
    # STAGE 0 — build + pre-train algorithms                              #
    # ------------------------------------------------------------------ #
    t0 = time.time()
    algorithms = _build_fresh_algorithms(config["tiers"])
    K = len(algorithms)

    # Compute tier boundaries for the hierarchical model
    n_tier1 = sum(1 for a in algorithms if a.family in {
        "EqualWeight", "MinimumVariance", "RiskParity", "MaxDiversification",
        "Momentum", "TrendFollowing", "MeanVariance",
    })
    # Simpler: count by class type
    n_tier1 = sum(1 for a in algorithms if not isinstance(a, TrainablePortfolioAlgorithm))
    n_trainable = sum(1 for a in algorithms if isinstance(a, TrainablePortfolioAlgorithm))

    # Tier 1 algorithms come first, then Tier 2, then Tier 3
    from regime_algo_selection.algorithms.tier3_nonlinear import Tier3Algorithm
    n_tier3 = sum(1 for a in algorithms if isinstance(a, Tier3Algorithm))
    n_tier2 = n_trainable - n_tier3
    tier_sizes = [n_tier1, n_tier2, n_tier3]

    # Global algorithm indices per tier
    tier1_idx = list(range(n_tier1))
    tier2_idx = list(range(n_tier1, n_tier1 + n_tier2))
    tier3_idx = list(range(n_tier1 + n_tier2, K))
    tier_algorithm_indices = [tier1_idx, tier2_idx, tier3_idx]

    print(f"  Built K={K} algorithms: {n_tier1} Tier1, {n_tier2} Tier2, {n_tier3} Tier3",
          flush=True)

    has_trainable = any(isinstance(a, TrainablePortfolioAlgorithm) for a in algorithms)
    if has_trainable:
        pretrain_algorithms(
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
    dataset.fit_scaler(train_start, train_end)

    print("  Precomputing algorithm outputs (slow step) ...", flush=True)
    dataset.precompute_algo_outputs()

    train_idx = dataset.get_indices_for_period(train_start, train_end)
    test_idx  = dataset.get_indices_for_period(test_start,  test_end)
    print(f"  Train: {len(train_idx)} days | Test: {len(test_idx)} days", flush=True)

    if len(train_idx) == 0 or len(test_idx) == 0:
        print("  WARNING: empty train or test set — skipping fold.", flush=True)
        return None

    t1 = time.time()
    print(f"  Stage 0 + precompute done in {t1 - t0:.1f}s", flush=True)

    # ------------------------------------------------------------------ #
    # BASELINES                                                           #
    # ------------------------------------------------------------------ #
    # Equal Weight: first algorithm in space is EqualWeight
    ew_k = next(
        (k for k, a in enumerate(algorithms) if a.name == "EqualWeight"), 0
    )

    def _ew_fn(idx):
        return dataset.get_algorithm_outputs(idx)[ew_k]

    m_ew, _ = _eval_strategy_on_test(dataset, test_idx, _ew_fn, kappa)
    print(f"  EW Sharpe: {m_ew['sharpe']:+.4f}", flush=True)

    # ------------------------------------------------------------------ #
    # INITIALIZE HIERARCHICAL META-LEARNER                                #
    # ------------------------------------------------------------------ #
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    model = HierarchicalMetaLearner(
        input_dim=config["input_dim"],
        tier_sizes=tier_sizes,
        selector_hidden=config["selector_hidden"],
        specialist_hidden=config["specialist_hidden"],
        dropout=config["dropout"],
    )

    trainer = HierarchicalTrainer(
        model=model,
        tier_algorithm_indices=tier_algorithm_indices,
        kappa=config["kappa"],
        kappa_a=config["kappa_a"],
        specialist_lr=config["specialist_lr"],
        selector_lr=config["selector_lr"],
        specialist_epochs=config["specialist_epochs"],
        selector_epochs=config["selector_epochs"],
        lambda_spec=config["lambda_spec"],
        lambda_tier=config["lambda_tier"],
    )

    # ------------------------------------------------------------------ #
    # TRAIN (Phase A + Phase B)                                           #
    # ------------------------------------------------------------------ #
    print(
        f"  Training: Phase A ({config['specialist_epochs']} ep × 3 specialists) + "
        f"Phase B ({config['selector_epochs']} ep) ...",
        flush=True,
    )
    t2 = time.time()
    training_history = trainer.train_fold(dataset, train_idx)
    t3 = time.time()
    print(f"  Training done in {t3 - t2:.1f}s", flush=True)

    # ------------------------------------------------------------------ #
    # EVALUATE on test set                                                #
    # ------------------------------------------------------------------ #
    model.eval()
    test_dates = dataset.dates[test_idx]
    T_test = len(test_idx)

    alpha_matrix  = np.zeros((T_test, K),       dtype=np.float32)
    beta_matrix   = np.zeros((T_test, 3),       dtype=np.float32)
    gamma_matrices = [
        np.zeros((T_test, tier_sizes[f]), dtype=np.float32)
        for f in range(3)
    ]
    w_matrix = np.zeros((T_test, N_ASSETS),     dtype=np.float32)

    with torch.no_grad():
        for i, idx in enumerate(test_idx):
            X_t = torch.tensor(dataset.get_input(idx), dtype=torch.float32)
            W_all = dataset.get_algorithm_outputs(idx)  # (K, N)

            alpha_t, beta_t, gammas = model(X_t)

            alpha_np = alpha_t.numpy()
            beta_np  = beta_t.numpy()
            gammas_np = [g.numpy() for g in gammas]

            alpha_matrix[i]  = alpha_np
            beta_matrix[i]   = beta_np
            for f in range(3):
                gamma_matrices[f][i] = gammas_np[f]

            # Composite portfolio: w_t = sum_k alpha_{t,k} w^(k)_t
            W_tensor = torch.tensor(W_all, dtype=torch.float32)
            w_t = torch.matmul(alpha_t, W_tensor).numpy()
            w_t = np.clip(w_t, 0.0, None)
            s = w_t.sum()
            w_matrix[i] = w_t / s if s > 1e-12 else np.ones(N_ASSETS) / N_ASSETS

    # Compute returns and metrics
    returns_matrix = np.array([dataset.get_returns(idx) for idx in test_idx])
    prev_w = np.ones(N_ASSETS) / N_ASSETS
    net_rets = []
    turnovers = []
    for t in range(T_test):
        w = w_matrix[t]
        r = returns_matrix[t]
        gross = float(w @ r)
        cost = kappa * float(np.abs(w - prev_w).sum())
        net_rets.append(gross - cost)
        turnovers.append(float(np.abs(w - prev_w).sum()))
        prev_w = w

    net_rets_arr = np.array(net_rets)
    m_hier = _compute_metrics(net_rets_arr)
    m_hier["avg_daily_turnover"] = round(float(np.mean(turnovers)), 6)

    # Entropy diagnostics
    H_alpha  = np.mean([_entropy(alpha_matrix[i])        for i in range(T_test)])
    H_beta   = np.mean([_entropy(beta_matrix[i])         for i in range(T_test)])
    H_gamma  = [np.mean([_entropy(gamma_matrices[f][i])  for i in range(T_test)])
                for f in range(3)]

    print(f"  Hierarchical Sharpe : {m_hier['sharpe']:+.4f}", flush=True)
    print(f"  H(beta) = {H_beta:.4f}  |  "
          f"H(g1) = {H_gamma[0]:.4f}  H(g2) = {H_gamma[1]:.4f}  H(g3) = {H_gamma[2]:.4f}  |  "
          f"H(alpha) = {H_alpha:.4f}",
          flush=True)

    # Regime-conditional beta averages
    regime_beta = {}
    for t, idx in enumerate(test_idx):
        r = dataset.get_regime(idx)
        if r not in regime_beta:
            regime_beta[r] = []
        regime_beta[r].append(beta_matrix[t])

    regime_beta_mean = {
        r: np.mean(vals, axis=0).tolist()
        for r, vals in regime_beta.items()
    }
    print("  Mean beta per regime:", flush=True)
    for r_id in sorted(regime_beta_mean.keys()):
        b = regime_beta_mean[r_id]
        print(f"    {REGIME_NAMES.get(r_id,'?'):7s}: "
              f"T1={b[0]:.3f}  T2={b[1]:.3f}  T3={b[2]:.3f}",
              flush=True)

    return {
        "fold_id":             fold_id,
        "fold_spec":           fold_spec,
        "tier_sizes":          tier_sizes,
        # Metrics
        "metrics_ew":          m_ew,
        "metrics_hierarchical": m_hier,
        # Entropy
        "H_alpha":             float(H_alpha),
        "H_beta":              float(H_beta),
        "H_gamma":             [float(h) for h in H_gamma],
        # Time series
        "test_dates":          test_dates,
        "alpha_matrix":        alpha_matrix,
        "beta_matrix":         beta_matrix,
        "gamma_matrices":      gamma_matrices,
        "w_matrix":            w_matrix,
        "net_rets":            net_rets_arr,
        "regime_beta_mean":    regime_beta_mean,
        # Training curves
        "training_history":    training_history,
        # Algorithm metadata
        "algo_names":          [a.name for a in algorithms],
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(fold_results: list) -> None:
    valid = [fr for fr in fold_results if fr is not None]
    if not valid:
        print("No valid folds.")
        return

    header = (
        f"{'Fold':>5}  {'Year':>5}  {'EW':>8}  {'Hier':>8}  "
        f"{'Hier-EW':>8}  {'H(b)':>6}  {'H(g1)':>6}  {'H(g2)':>6}  {'H(g3)':>6}  {'H(a)':>6}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * 70, flush=True)
    print("PLAN 13a: HIERARCHICAL META-LEARNER — SANITY CHECK RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(header, flush=True)
    print(sep, flush=True)

    for fr in valid:
        fold_id = fr["fold_id"]
        year = fr["fold_spec"]["test_start"][:4]
        ew_s = fr["metrics_ew"]["sharpe"]
        hi_s = fr["metrics_hierarchical"]["sharpe"]
        diff = hi_s - ew_s if (not np.isnan(hi_s) and not np.isnan(ew_s)) else np.nan
        print(
            f"{fold_id:>5}  {year:>5}  "
            f"{ew_s:>8.4f}  {hi_s:>8.4f}  {diff:>8.4f}  "
            f"{fr['H_beta']:>6.3f}  "
            f"{fr['H_gamma'][0]:>6.3f}  {fr['H_gamma'][1]:>6.3f}  {fr['H_gamma'][2]:>6.3f}  "
            f"{fr['H_alpha']:>6.3f}",
            flush=True,
        )

    print(sep, flush=True)

    # Sanity check assertions
    print("\nSANITY CHECKS:", flush=True)
    for fr in valid:
        name = fr["fold_spec"]["test_start"][:4]
        ts = fr["tier_sizes"]
        H_beta = fr["H_beta"]
        H_gamma = fr["H_gamma"]
        H_max_tier = [np.log(ts[0]), np.log(ts[1]), np.log(ts[2])]

        beta_ok = 0.1 <= H_beta <= 1.0
        g1_ok = 0.3 <= H_gamma[0] <= (H_max_tier[0] - 0.3)
        g2_ok = 0.3 <= H_gamma[1] <= (H_max_tier[1] - 0.3)
        g3_ok = 0.3 <= H_gamma[2] <= (H_max_tier[2] - 0.3)

        print(f"  Fold {fr['fold_id']} (test {name}):", flush=True)
        print(f"    H(beta)={H_beta:.3f}  target [0.1, 1.0]   {'OK' if beta_ok else 'FAIL'}", flush=True)
        print(f"    H(g1)={H_gamma[0]:.3f}  target [0.3, {H_max_tier[0]-0.3:.2f}]  {'OK' if g1_ok else 'FAIL'}", flush=True)
        print(f"    H(g2)={H_gamma[1]:.3f}  target [0.3, {H_max_tier[1]-0.3:.2f}]  {'OK' if g2_ok else 'FAIL'}", flush=True)
        print(f"    H(g3)={H_gamma[2]:.3f}  target [0.3, {H_max_tier[2]-0.3:.2f}]  {'OK' if g3_ok else 'FAIL'}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sanity_check(config: dict = SANITY_CONFIG) -> list:
    """Run sanity check: Fold 1 (test 2013) and Fold 6 (test 2018) only."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for Plan 13a. Install with: pip install torch"
        )

    print("\n" + "=" * 70, flush=True)
    print("PLAN 13a: HIERARCHICAL META-LEARNER — SANITY CHECK (2 folds)", flush=True)
    print("=" * 70, flush=True)
    print(f"  specialist_epochs={config['specialist_epochs']}, "
          f"selector_epochs={config['selector_epochs']}", flush=True)
    print(f"  lambda_spec={config['lambda_spec']}, "
          f"lambda_tier={config['lambda_tier']}", flush=True)

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
    print(f"  Common dates: {len(common)}")

    # ------------------------------------------------------------------ #
    # Generate folds, pick fold 1 and fold 6                             #
    # ------------------------------------------------------------------ #
    wfv = WalkForwardValidator(
        train_years=config["train_years"],
        test_years=config["test_years"],
        step_years=config["step_years"],
        min_test_start=config["min_test_start"],
    )
    all_folds = wfv.generate_folds(data_end=config["data_end"])
    print(f"Generated {len(all_folds)} walk-forward folds total")

    # Sanity check: Fold 1 (test 2013) and Fold 6 (test 2018)
    sanity_fold_ids = {1, 6}
    sanity_folds = [f for f in all_folds if f["fold"] in sanity_fold_ids]
    if not sanity_folds:
        # Fallback: use first 2 folds
        sanity_folds = all_folds[:2]
    print(f"Running {len(sanity_folds)} sanity folds: "
          f"{[f['test_start'][:4] for f in sanity_folds]}", flush=True)

    # ------------------------------------------------------------------ #
    # Run selected folds                                                  #
    # ------------------------------------------------------------------ #
    fold_results = []
    for fold_spec in sanity_folds:
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
    # Print summary                                                       #
    # ------------------------------------------------------------------ #
    _print_summary(fold_results)

    # Save results
    valid = [fr for fr in fold_results if fr is not None]
    if valid:
        rows = []
        for fr in valid:
            rows.append({
                "fold":        fr["fold_id"],
                "test_year":   fr["fold_spec"]["test_start"][:4],
                "ew_sharpe":   fr["metrics_ew"]["sharpe"],
                "hier_sharpe": fr["metrics_hierarchical"]["sharpe"],
                "H_beta":      fr["H_beta"],
                "H_gamma1":    fr["H_gamma"][0],
                "H_gamma2":    fr["H_gamma"][1],
                "H_gamma3":    fr["H_gamma"][2],
                "H_alpha":     fr["H_alpha"],
            })
        csv_path = os.path.join(PLAN13A_RESULTS_DIR, "sanity_check_results.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\nSaved sanity check results to: {csv_path}", flush=True)

    print("\nPLAN 13a SANITY CHECK COMPLETE.", flush=True)
    return fold_results


def run_experiment(config: dict = CONFIG) -> list:
    """Run full 12-fold experiment (call AFTER sanity check passes)."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for Plan 13a. Install with: pip install torch"
        )

    print("\n" + "=" * 70, flush=True)
    print("PLAN 13a: HIERARCHICAL META-LEARNER — FULL 12-FOLD RUN", flush=True)
    print("=" * 70, flush=True)

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

    _print_summary(fold_results)

    valid = [fr for fr in fold_results if fr is not None]
    if valid:
        rows = []
        for fr in valid:
            rows.append({
                "fold":        fr["fold_id"],
                "test_year":   fr["fold_spec"]["test_start"][:4],
                "ew_sharpe":   fr["metrics_ew"]["sharpe"],
                "hier_sharpe": fr["metrics_hierarchical"]["sharpe"],
                "H_beta":      fr["H_beta"],
                "H_gamma1":    fr["H_gamma"][0],
                "H_gamma2":    fr["H_gamma"][1],
                "H_gamma3":    fr["H_gamma"][2],
                "H_alpha":     fr["H_alpha"],
            })
        csv_path = os.path.join(PLAN13A_RESULTS_DIR, "summary_metrics.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\nSaved summary to: {csv_path}", flush=True)

    print("\nPLAN 13a COMPLETE.", flush=True)
    return fold_results


if __name__ == "__main__":
    # SANITY CHECK: 2 folds only
    run_sanity_check()
