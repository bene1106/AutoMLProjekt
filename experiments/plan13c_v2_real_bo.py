# experiments/plan13c_v2_real_bo.py -- Plan 13c-v2: Real BO with Warm Start
#
# Architecture:
#   Meta-Learner (pre-trained from Plan 13a) → frozen tier-selector β ∈ Δ3
#   Optuna TPE per (regime, tier) → algorithm + HP selection within each tier
#   Warm-started with Plan 13a's default configs per (regime, tier)
#
# Key differences from Plan 13b-v3:
#   - 12 BO studies per fold (4 regimes × 3 tiers) instead of 4
#   - Search restricted to per-tier algorithm families (smaller space)
#   - TierSelector β from Plan 13a blends tier outputs at test time
#   - Val-split identical to 13b-v3 (chronological 80/20)
#
# TierSelector note:
#   Plan 13a's full run showed β ≈ [0.99, 0.005, 0.005] for all regimes
#   (H_beta=0.042-0.616, log confirmed Calm T1=0.992, Normal T1=0.998,
#   Tense T1=0.999). We use regime-dependent static β from these observations.
#   This is the "frozen pre-trained TierSelector" from Plan 13a.
#
# Usage:
#   cd Implementierung1
#   python -u -m experiments.plan13c_v2_real_bo --fold 1 --smoke
#   python -u -m experiments.plan13c_v2_real_bo --fold 1
#   python -u -m experiments.plan13c_v2_real_bo --fold 1 --skip-existing

import os
import sys
import time
import gc
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

try:
    import psutil as _psutil
    def _rss_mb() -> float:
        return _psutil.Process().memory_info().rss / 1024**2
except ImportError:
    _psutil = None
    def _rss_mb() -> float:
        return float("nan")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---- project imports -------------------------------------------------------
from regime_algo_selection.config import (
    RESULTS_DIR, KAPPA, REGIME_NAMES, N_ASSETS, RANDOM_SEED,
)
from regime_algo_selection.data.loader_extended import load_data_extended
from regime_algo_selection.data.features import compute_returns, compute_asset_features
from regime_algo_selection.regimes.ground_truth import compute_regime_labels
from regime_algo_selection.algorithms.base import TrainablePortfolioAlgorithm
from regime_algo_selection.algorithms.stage0 import build_training_matrix
from regime_algo_selection.algorithms.tier1_heuristics import (
    EqualWeight, MinimumVariance, RiskParity, MaxDiversification,
    Momentum, TrendFollowing, MeanVariance,
)
from regime_algo_selection.algorithms.tier2_linear import (
    RidgePortfolio, LassoPortfolio, ElasticNetPortfolio,
)
from regime_algo_selection.algorithms.tier3_nonlinear import Tier3Algorithm
from regime_algo_selection.evaluation.walk_forward import WalkForwardValidator

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # BO settings per tier (full run)
    "n_trials_tier1": 30,
    "n_trials_tier2": 50,
    "n_trials_tier3": 80,

    # Smoke test overrides
    "n_trials_smoke": 10,

    # Val split
    "seed": RANDOM_SEED,
    "val_frac": 0.20,               # inner 80/20 chronological split
    "min_regime_days": 20,           # below this: EW fallback (relaxed vs 13b-v3's 100)

    # Costs
    "kappa": KAPPA,

    # Walk-forward (identical to Plans 13a/13b-v3)
    "train_years": 8,
    "test_years": 1,
    "step_years": 1,
    "min_test_start": "2013-01-01",
    "data_end": "2024-12-31",

    # Output
    "output_dir": os.path.join(RESULTS_DIR, "plan13c_v2"),
}

OUT_DIR       = CONFIG["output_dir"]
STUDIES_DIR   = os.path.join(OUT_DIR, "optuna_studies")
ANALYSIS_DIR  = os.path.join(OUT_DIR, "analysis")

os.makedirs(OUT_DIR,      exist_ok=True)
os.makedirs(STUDIES_DIR,  exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

N = N_ASSETS  # 5

# ---------------------------------------------------------------------------
# Plan 13a TierSelector: frozen static β (derived from Plan 13a full run)
# ---------------------------------------------------------------------------
# Plan 13a showed that TierSelector collapses to Tier 1 for all regimes:
#   Calm:   T1=0.992  T2=0.004  T3=0.004  (Fold 1 log)
#   Normal: T1=0.998  T2=0.001  T3=0.001
#   Tense:  T1=0.999  T2=0.000  T3=0.000
#   Crisis: assumed similar (no data in Fold 1 log)
# These are used as static β for test-time blending.
PLAN13A_STATIC_BETA = {
    1: np.array([0.992, 0.004, 0.004]),   # Calm
    2: np.array([0.998, 0.001, 0.001]),   # Normal
    3: np.array([0.999, 0.0005, 0.0005]), # Tense
    4: np.array([0.990, 0.005, 0.005]),   # Crisis (assumed similar)
}
# Normalize to sum to 1
for k in PLAN13A_STATIC_BETA:
    PLAN13A_STATIC_BETA[k] = PLAN13A_STATIC_BETA[k] / PLAN13A_STATIC_BETA[k].sum()

# ---------------------------------------------------------------------------
# Warm-start seeds per tier (default fallback since no Plan 13a trace file)
# ---------------------------------------------------------------------------
DEFAULT_WARM_SEEDS = {
    "tier1": [
        {"t1_family": "MinVariance",      "minvar_lookback": 60},
        {"t1_family": "MaxDiversification","maxdiv_lookback": 90},
        {"t1_family": "EqualWeight"},
    ],
    "tier2": [
        {"t2_family": "Lasso",     "lookback": 60,  "lasso_alpha": 0.01},
        {"t2_family": "Ridge",     "lookback": 120, "ridge_alpha": 0.10},
        {"t2_family": "ElasticNet","lookback": 60,  "enet_alpha": 0.05, "enet_l1_ratio": 0.5},
    ],
    "tier3": [
        {"t3_family": "RandomForest",    "lookback": 60,
         "rf_n_estimators": 100, "rf_max_depth": 5, "rf_min_leaf": 5},
        {"t3_family": "GradientBoosting","lookback": 60,
         "gbm_n_estimators": 100, "gbm_max_depth": 5,
         "gbm_lr": 0.05, "gbm_subsample": 0.8},
        {"t3_family": "MLP",            "lookback": 60,
         "mlp_hidden1": 64, "mlp_hidden2": 32, "mlp_alpha": 0.001},
    ],
}

# ---------------------------------------------------------------------------
# Metric helpers (identical to Plan 13b-v3)
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


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.where(np.isfinite(x), x, 0.0)
    x = x - x.max()
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 1e-12 else np.ones(len(x)) / len(x)


def _compute_sharpe(weights: np.ndarray, rets: np.ndarray, kappa: float) -> float:
    port_ret = np.sum(weights * rets, axis=1)
    if len(weights) > 1:
        turnover = np.sum(np.abs(np.diff(weights, axis=0)), axis=1)
        costs = kappa * turnover
        net_ret = port_ret[1:] - costs
    else:
        net_ret = port_ret
    if len(net_ret) < 5 or np.std(net_ret) < 1e-10:
        return float('-inf')
    return float(np.mean(net_ret) / np.std(net_ret) * np.sqrt(252))

# ---------------------------------------------------------------------------
# Per-tier search space samplers
# ---------------------------------------------------------------------------

def sample_tier1(trial: optuna.Trial) -> dict:
    """Tier 1 (Heuristics): 7 families, 0-2 HPs each."""
    family = trial.suggest_categorical("t1_family", [
        "EqualWeight", "MinVariance", "RiskParity", "MaxDiversification",
        "MeanVariance", "Momentum", "TrendFollowing",
    ])
    if family == "EqualWeight":
        return {"tier": 1, "family": family}
    elif family == "MinVariance":
        return {"tier": 1, "family": family,
                "minvar_lookback": trial.suggest_int("minvar_lookback", 5, 252)}
    elif family == "RiskParity":
        return {"tier": 1, "family": family,
                "riskparity_lookback": trial.suggest_int("riskparity_lookback", 5, 252)}
    elif family == "MaxDiversification":
        return {"tier": 1, "family": family,
                "maxdiv_lookback": trial.suggest_int("maxdiv_lookback", 5, 252)}
    elif family == "MeanVariance":
        return {"tier": 1, "family": family,
                "mv_lookback": trial.suggest_int("mv_lookback", 5, 252),
                "mv_gamma": trial.suggest_float("mv_gamma", 0.1, 10.0, log=True)}
    elif family == "Momentum":
        return {"tier": 1, "family": family,
                "mom_lookback": trial.suggest_int("mom_lookback", 5, 252),
                "mom_type": trial.suggest_categorical("mom_type", ["linear", "exp"])}
    else:  # TrendFollowing
        return {"tier": 1, "family": family,
                "trend_lookback": trial.suggest_int("trend_lookback", 5, 252),
                "trend_beta": trial.suggest_int("trend_beta", 1, 5)}


def sample_tier2(trial: optuna.Trial) -> dict:
    """Tier 2 (Linear ML): 3 families, lookback + alpha (+ l1_ratio for ElasticNet)."""
    family = trial.suggest_categorical("t2_family", ["Ridge", "Lasso", "ElasticNet"])
    lookback = trial.suggest_int("lookback", 20, 252)
    if family == "Ridge":
        return {"tier": 2, "family": family, "lookback": lookback,
                "ridge_alpha": trial.suggest_float("ridge_alpha", 1e-4, 100.0, log=True)}
    elif family == "Lasso":
        return {"tier": 2, "family": family, "lookback": lookback,
                "lasso_alpha": trial.suggest_float("lasso_alpha", 1e-4, 10.0, log=True)}
    else:  # ElasticNet
        return {"tier": 2, "family": family, "lookback": lookback,
                "enet_alpha": trial.suggest_float("enet_alpha", 1e-4, 10.0, log=True),
                "enet_l1_ratio": trial.suggest_float("enet_l1_ratio", 0.1, 0.9)}


def sample_tier3(trial: optuna.Trial) -> dict:
    """Tier 3 (Non-Linear ML): RF / GBM / MLP with model-specific HPs."""
    family = trial.suggest_categorical("t3_family", [
        "RandomForest", "GradientBoosting", "MLP",
    ])
    lookback = trial.suggest_int("lookback", 20, 252)
    if family == "RandomForest":
        return {"tier": 3, "family": family, "lookback": lookback,
                "rf_n_estimators": trial.suggest_int("rf_n_estimators", 50, 300),
                "rf_max_depth":    trial.suggest_int("rf_max_depth", 3, 20),
                "rf_min_leaf":     trial.suggest_int("rf_min_leaf", 1, 20)}
    elif family == "GradientBoosting":
        return {"tier": 3, "family": family, "lookback": lookback,
                "gbm_n_estimators": trial.suggest_int("gbm_n_estimators", 50, 300),
                "gbm_max_depth":    trial.suggest_int("gbm_max_depth", 2, 10),
                "gbm_lr":           trial.suggest_float("gbm_lr", 0.001, 0.5, log=True),
                "gbm_subsample":    trial.suggest_float("gbm_subsample", 0.5, 1.0)}
    else:  # MLP
        h1 = trial.suggest_int("mlp_hidden1", 16, 256)
        h2 = trial.suggest_int("mlp_hidden2", 0, 128)
        return {"tier": 3, "family": family, "lookback": lookback,
                "mlp_hidden1": h1, "mlp_hidden2": h2,
                "mlp_alpha": trial.suggest_float("mlp_alpha", 1e-5, 0.1, log=True)}


TIER_SAMPLERS = {1: sample_tier1, 2: sample_tier2, 3: sample_tier3}

# ---------------------------------------------------------------------------
# Algorithm factory (reused from 13b-v3 pattern)
# ---------------------------------------------------------------------------

def create_algorithm_from_config(config: dict):
    """Build a portfolio algorithm instance from a sampled HP config dict."""
    family = config["family"]

    # ── Tier 1 ──────────────────────────────────────────────────────────────
    if family == "EqualWeight":
        return EqualWeight()
    elif family == "MinVariance":
        return MinimumVariance(lookback=config["minvar_lookback"])
    elif family == "RiskParity":
        return RiskParity(lookback=config["riskparity_lookback"])
    elif family == "MaxDiversification":
        return MaxDiversification(lookback=config["maxdiv_lookback"])
    elif family == "MeanVariance":
        return MeanVariance(lookback=config["mv_lookback"],
                            risk_aversion=config["mv_gamma"])
    elif family == "Momentum":
        return Momentum(lookback=config["mom_lookback"],
                        weighting=config["mom_type"])
    elif family == "TrendFollowing":
        return TrendFollowing(lookback=config["trend_lookback"],
                              beta=config.get("trend_beta", 1))

    # ── Tier 2 ──────────────────────────────────────────────────────────────
    elif family == "Ridge":
        return RidgePortfolio(lambda_ridge=config["ridge_alpha"],
                              lookback=config["lookback"])
    elif family == "Lasso":
        return LassoPortfolio(lambda_lasso=config["lasso_alpha"],
                              lookback=config["lookback"])
    elif family == "ElasticNet":
        return ElasticNetPortfolio(lambda_en=config["enet_alpha"],
                                   l1_ratio=config["enet_l1_ratio"],
                                   lookback=config["lookback"])

    # ── Tier 3 ──────────────────────────────────────────────────────────────
    elif family == "RandomForest":
        return Tier3Algorithm(
            family="RandomForest", model_class=RandomForestRegressor,
            model_params={"n_estimators": config["rf_n_estimators"],
                          "max_depth": config["rf_max_depth"],
                          "min_samples_leaf": config["rf_min_leaf"],
                          "random_state": 42, "n_jobs": -1},
            lookback=config["lookback"],
            name=f"RF_n{config['rf_n_estimators']}_d{config['rf_max_depth']}",
        )
    elif family == "GradientBoosting":
        return Tier3Algorithm(
            family="GradientBoosting", model_class=GradientBoostingRegressor,
            model_params={"n_estimators": config["gbm_n_estimators"],
                          "max_depth": config["gbm_max_depth"],
                          "learning_rate": config["gbm_lr"],
                          "subsample": config["gbm_subsample"],
                          "random_state": 42},
            lookback=config["lookback"],
            name=f"GBM_n{config['gbm_n_estimators']}_d{config['gbm_max_depth']}",
        )
    elif family == "MLP":
        h1 = config["mlp_hidden1"]
        h2 = config.get("mlp_hidden2", 0)
        hidden = (h1,) if h2 == 0 else (h1, h2)
        return Tier3Algorithm(
            family="MLP", model_class=MLPRegressor,
            model_params={"hidden_layer_sizes": hidden, "alpha": config["mlp_alpha"],
                          "max_iter": 300, "random_state": 42,
                          "early_stopping": True, "validation_fraction": 0.1},
            lookback=config["lookback"],
            name=f"MLP_h{h1}x{h2}",
        )

    raise ValueError(f"Unknown algorithm family: {family}")


# ---------------------------------------------------------------------------
# Warm-start seed extraction
# ---------------------------------------------------------------------------

def extract_warm_start_seeds(regime_id: int, tier: int) -> list:
    """
    Return warm-start seed dicts compatible with study.enqueue_trial().

    Plan 13a does not save a selection_trace.csv (it stores algo_outputs.npy
    which are pre-computed portfolio weights, not per-day HP logs).
    We use sensible defaults derived from Plan 13a's algorithm library.

    Returns list of dicts (Optuna param dicts for enqueue_trial).
    """
    tier_key = f"tier{tier}"
    seeds = DEFAULT_WARM_SEEDS.get(tier_key, [])

    # Map from DEFAULT_WARM_SEEDS format to algo family
    optuna_seeds = []
    for s in seeds:
        seed_params = {}
        for k, v in s.items():
            seed_params[k] = v
        optuna_seeds.append(seed_params)

    return optuna_seeds


# ---------------------------------------------------------------------------
# Core evaluation: fit on inner_train, evaluate on val
# ---------------------------------------------------------------------------

def _softmax_weights_from_mu(mu_all: np.ndarray) -> np.ndarray:
    return np.array([_softmax(mu_all[i]) for i in range(len(mu_all))])


def _eval_config_on_block(
    config: dict,
    X_fit: np.ndarray,
    Y_fit: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    eval_dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
    N: int,
    kappa: float,
) -> float:
    """
    Fit config on (X_fit, Y_fit), evaluate on (X_eval, eval_dates).
    Returns Sharpe on eval block (net of kappa), or float('-inf') on failure.

    CRITICAL NaN handling (Lesson 4 from 13b-v3):
        X_eval may contain NaN rows from rolling feature warmup.
        We apply finite_mask at predict time. Return float('-inf'), NOT 0.0.
    """
    ew = np.ones(N) / N
    T_eval = len(eval_dates)
    if T_eval == 0:
        return float('-inf')

    try:
        algo = create_algorithm_from_config(config)
    except Exception:
        return float('-inf')

    if isinstance(algo, TrainablePortfolioAlgorithm):
        try:
            algo.fit(X_fit, Y_fit)
        except Exception:
            return float('-inf')
        if not algo._is_fitted or algo._scaler is None:
            return float('-inf')

        # CRITICAL: filter NaN rows before predict (Lesson 4)
        eval_finite_mask = np.isfinite(X_eval).all(axis=1)
        X_eval_clean = X_eval[eval_finite_mask]
        y_eval_clean = y_eval[eval_finite_mask]

        if len(X_eval_clean) < 5:
            return float('-inf')

        try:
            X_scaled = algo._scaler.transform(X_eval_clean)
        except Exception:
            return float('-inf')
        try:
            if hasattr(algo, "_models") and isinstance(algo._models, dict) and algo._models:
                mu_all = np.column_stack([
                    algo._models[j].predict(X_scaled) for j in range(N)
                ])
            else:
                mu_all = algo._model.predict(X_scaled)
        except Exception:
            return float('-inf')
        weights = _softmax_weights_from_mu(mu_all)
        return _compute_sharpe(weights, y_eval_clean, kappa)

    else:
        # Tier 1: compute weights day-by-day using price history
        weights = []
        for d in eval_dates:
            try:
                ph = prices.loc[:d]
                w = algo.compute_weights(ph)
            except Exception:
                w = ew.copy()
            w = np.where(np.isfinite(w), w, 0.0)
            w = np.clip(w, 0.0, None)
            s = w.sum()
            w = w / s if s > 1e-12 else ew.copy()
            weights.append(w)
        weights = np.array(weights)
        return _compute_sharpe(weights, y_eval, kappa)


# ---------------------------------------------------------------------------
# BO objective factory with val-split
# ---------------------------------------------------------------------------

def make_objective_for_tier(
    tier: int,
    X_train_inner: np.ndarray,
    Y_train_inner: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    val_dates: pd.DatetimeIndex,
    train_inner_dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
    N: int,
    kappa: float,
    trial_log: list,
    regime_id: int,
):
    """
    Create Optuna objective for a specific tier.
    Objective = val-Sharpe (NOT train-Sharpe — Lesson from 13b-v2).
    Also records train_sharpe for diagnostic gap analysis.
    """
    sampler_fn = TIER_SAMPLERS[tier]

    def objective(trial: optuna.Trial) -> float:
        config = sampler_fn(trial)
        try:
            trial.set_user_attr("config", config)
        except Exception:
            pass

        # Val Sharpe (objective)
        val_sharpe = _eval_config_on_block(
            config=config,
            X_fit=X_train_inner, Y_fit=Y_train_inner,
            X_eval=X_val, y_eval=Y_val,
            eval_dates=val_dates,
            prices=prices, N=N, kappa=kappa,
        )

        # Train Sharpe (diagnostic only, NOT objective)
        train_sharpe = _eval_config_on_block(
            config=config,
            X_fit=X_train_inner, Y_fit=Y_train_inner,
            X_eval=X_train_inner, y_eval=Y_train_inner,
            eval_dates=train_inner_dates,
            prices=prices, N=N, kappa=kappa,
        )

        gap = train_sharpe - val_sharpe
        try:
            trial.set_user_attr("train_sharpe", float(train_sharpe))
            trial.set_user_attr("val_sharpe",   float(val_sharpe))
            trial.set_user_attr("train_val_gap", float(gap))
        except Exception:
            pass

        trial_log.append({
            "regime_id":    regime_id,
            "tier":         tier,
            "trial_num":    trial.number,
            "family":       config.get("family", "?"),
            "train_sharpe": round(float(train_sharpe), 6),
            "val_sharpe":   round(float(val_sharpe),   6),
            "gap":          round(float(gap),          6),
        })

        return val_sharpe  # CRITICAL: val, not train

    return objective


# ---------------------------------------------------------------------------
# BO study runner per (regime, tier)
# ---------------------------------------------------------------------------

def run_bo_for_regime_tier(
    fold_id: int,
    regime_id: int,
    tier: int,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_regime: np.ndarray,
    Y_regime: np.ndarray,
    regime_dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
    N: int,
    kappa: float,
    n_trials: int,
    seed: int,
    val_frac: float,
    trial_log: list,
) -> optuna.Study | None:
    """
    Run one Optuna TPE study for a (regime, tier) combination.

    Inner 80/20 chronological split:
      - BO objective = val Sharpe (last 20% of regime days)
      - train Sharpe recorded for gap diagnostic
    """
    # Inner val split
    n_total = len(regime_dates)
    split_idx = int(n_total * (1 - val_frac))

    train_inner_dates = regime_dates[:split_idx]
    val_dates         = regime_dates[split_idx:]

    X_inner_train = X_regime[:split_idx]
    Y_inner_train = Y_regime[:split_idx]
    X_inner_val   = X_regime[split_idx:]
    Y_inner_val   = Y_regime[split_idx:]

    print(
        f"    inner-train: {len(train_inner_dates)} days"
        f"  | inner-val: {len(val_dates)} days",
        end="  ", flush=True,
    )

    if len(val_dates) < 5:
        print("-> val block too small, using EW fallback", flush=True)
        return None

    # Create study
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study   = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"fold{fold_id}_r{regime_id}_tier{tier}",
    )

    # Warm start: enqueue Plan 13a's top configs
    seeds = extract_warm_start_seeds(regime_id, tier)
    n_enqueued = 0
    for seed_params in seeds:
        try:
            study.enqueue_trial(seed_params)
            n_enqueued += 1
        except Exception as e:
            print(f"    [warm-start seed rejected: {e}]", flush=True)
    if n_enqueued > 0:
        print(f"(warm-start: {n_enqueued} seeds enqueued)", end="  ", flush=True)

    # Build objective
    objective = make_objective_for_tier(
        tier=tier,
        X_train_inner=X_inner_train,
        Y_train_inner=Y_inner_train,
        X_val=X_inner_val,
        Y_val=Y_inner_val,
        val_dates=val_dates,
        train_inner_dates=train_inner_dates,
        prices=prices,
        N=N,
        kappa=kappa,
        trial_log=trial_log,
        regime_id=regime_id,
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study


# ---------------------------------------------------------------------------
# Compute weights from a config on test dates (refit on full training data)
# ---------------------------------------------------------------------------

def compute_weights_for_config(
    config: dict,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    asset_features: pd.DataFrame,
    test_dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
    N: int,
) -> np.ndarray:
    """
    Build and fit the algorithm on full training data, compute weights for
    each test date.  Returns (T_test, N) weight array.
    """
    T_test = len(test_dates)
    ew = np.ones(N) / N

    try:
        algo = create_algorithm_from_config(config)
    except Exception:
        return np.tile(ew, (T_test, 1))

    if isinstance(algo, TrainablePortfolioAlgorithm):
        try:
            algo.fit(X_train, Y_train)
        except Exception:
            return np.tile(ew, (T_test, 1))
        if not algo._is_fitted or algo._scaler is None:
            return np.tile(ew, (T_test, 1))
        try:
            algo.attach_full_features(asset_features)
        except Exception:
            pass

    weights = []
    for d in test_dates:
        try:
            ph = prices.loc[:d]
            w = algo.compute_weights(ph)
        except Exception:
            w = ew.copy()
        w = np.where(np.isfinite(w), w, 0.0)
        w = np.clip(w, 0.0, None)
        s = w.sum()
        w = w / s if s > 1e-12 else ew.copy()
        weights.append(w)

    return np.array(weights)


# ---------------------------------------------------------------------------
# Test-time evaluation helpers
# ---------------------------------------------------------------------------

def _eval_strategy(
    weight_fn,
    returns_arr: np.ndarray,
    test_indices: np.ndarray,
    kappa: float,
    N: int,
) -> tuple:
    """Evaluate a weight function on test indices. Returns (metrics, net_returns)."""
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


def evaluate_hierarchical_blend(
    best_weights_per_regime_tier: dict,
    returns_arr: np.ndarray,
    regime_arr: np.ndarray,
    test_indices: np.ndarray,
    kappa: float,
    N: int,
) -> tuple:
    """
    Hierarchical blending using Plan 13a's TierSelector β.

    For each test day t in regime s_t:
        w_t = β[s_t][0] * w^(tier1)[t] + β[s_t][1] * w^(tier2)[t] + β[s_t][2] * w^(tier3)[t]

    best_weights_per_regime_tier: {regime_id: {tier: (T_test_regime, N) array}}
    """
    # Build per-test-day position maps
    regime_test_positions = {}
    for regime_id in [1, 2, 3, 4]:
        regime_test_positions[regime_id] = [
            (pos, i) for pos, i in enumerate(test_indices)
            if regime_arr[i] == regime_id
        ]

    weights_all = np.zeros((len(test_indices), N))

    for regime_id in [1, 2, 3, 4]:
        positions = regime_test_positions[regime_id]
        if not positions:
            continue

        beta = PLAN13A_STATIC_BETA[regime_id]  # [β_t1, β_t2, β_t3]

        for pos, i in positions:
            blended = np.zeros(N)
            for tier_idx, tier in enumerate([1, 2, 3]):
                w_tier = best_weights_per_regime_tier.get(regime_id, {}).get(tier)
                if w_tier is None or len(w_tier) == 0:
                    w_tier_day = np.ones(N) / N
                else:
                    # Find the row in w_tier for this test day
                    n_regime_days = regime_test_positions[regime_id]
                    local_pos = [p for p, _ in n_regime_days].index(pos)
                    if local_pos < len(w_tier):
                        w_tier_day = w_tier[local_pos]
                    else:
                        w_tier_day = np.ones(N) / N
                blended += beta[tier_idx] * w_tier_day

            # Renormalize
            blended = np.where(np.isfinite(blended), blended, 0.0)
            blended = np.clip(blended, 0.0, None)
            s = blended.sum()
            weights_all[pos] = blended / s if s > 1e-12 else np.ones(N) / N

    def _fn(i):
        j = np.searchsorted(test_indices, i)
        if j < len(test_indices) and test_indices[j] == i:
            return weights_all[j]
        return np.ones(N) / N

    return _eval_strategy(_fn, returns_arr, test_indices, kappa, N)


def evaluate_tier1_hard(
    best_weights_per_regime_tier: dict,
    returns_arr: np.ndarray,
    regime_arr: np.ndarray,
    test_indices: np.ndarray,
    kappa: float,
    N: int,
) -> tuple:
    """
    Tier 1 hard selection (useful as ablation: what if we just use Tier 1?).
    Since β_tier1 ≈ 0.99 from Plan 13a, this is nearly identical to blend.
    """
    regime_test_positions = {r: [] for r in [1, 2, 3, 4]}
    for pos, i in enumerate(test_indices):
        regime_test_positions[regime_arr[i]].append((pos, i))

    weights_all = np.zeros((len(test_indices), N))

    for regime_id in [1, 2, 3, 4]:
        positions = regime_test_positions[regime_id]
        if not positions:
            continue

        w_tier1 = best_weights_per_regime_tier.get(regime_id, {}).get(1)
        if w_tier1 is None or len(w_tier1) == 0:
            for pos, i in positions:
                weights_all[pos] = np.ones(N) / N
        else:
            for local_pos, (pos, i) in enumerate(positions):
                if local_pos < len(w_tier1):
                    weights_all[pos] = w_tier1[local_pos]
                else:
                    weights_all[pos] = np.ones(N) / N

    def _fn(i):
        j = np.searchsorted(test_indices, i)
        if j < len(test_indices) and test_indices[j] == i:
            return weights_all[j]
        return np.ones(N) / N

    return _eval_strategy(_fn, returns_arr, test_indices, kappa, N)


def evaluate_ew(returns_arr: np.ndarray, test_indices: np.ndarray,
                kappa: float, N: int) -> tuple:
    return _eval_strategy(lambda i: np.ones(N) / N, returns_arr, test_indices, kappa, N)


# ---------------------------------------------------------------------------
# Per-fold persistence helpers
# ---------------------------------------------------------------------------

def is_fold_complete(fold_id: int) -> bool:
    csv_path = os.path.join(OUT_DIR, f"fold_{fold_id:02d}_result.csv")
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        return len(df) >= 3
    except Exception:
        return False


def _fsync_csv(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)
    with open(path, "rb+") as f:
        f.flush()
        os.fsync(f.fileno())


def _fsync_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())


def save_fold_result(fold_result: dict) -> None:
    """
    Persist per-fold artifacts immediately after completion.
    Writes: fold_NN_result.csv, fold_NN_best_configs.json, fold_NN_trial_log.csv
    """
    fold_id   = fold_result["fold_id"]
    fold_spec = fold_result["fold_spec"]
    test_year = fold_spec["test_start"][:4]

    # fold_NN_result.csv — 3 rows: hierarchical blend, tier1 hard, ew
    strategies = [
        ("hier_blend", fold_result["metrics_blend"]),
        ("tier1_hard",  fold_result["metrics_tier1"]),
        ("ew",          fold_result["metrics_ew"]),
    ]
    rows = []
    for strat, m in strategies:
        rows.append({
            "fold":      fold_id,
            "test_year": test_year,
            "strategy":  strat,
            "sharpe":    m.get("sharpe",             float("nan")),
            "sortino":   m.get("sortino",            float("nan")),
            "max_dd":    m.get("max_drawdown",       float("nan")),
            "turnover":  m.get("avg_daily_turnover", float("nan")),
        })

    csv_path = os.path.join(OUT_DIR, f"fold_{fold_id:02d}_result.csv")
    _fsync_csv(csv_path, pd.DataFrame(rows))
    print(f"  [persist] Saved: {csv_path}", flush=True)

    # fold_NN_best_configs.json
    json_path = os.path.join(OUT_DIR, f"fold_{fold_id:02d}_best_configs.json")
    _fsync_json(json_path, fold_result.get("regime_tier_stats", {}))
    print(f"  [persist] Saved: {json_path}", flush=True)

    # fold_NN_trial_log.csv
    trial_log = fold_result.get("trial_log", [])
    if trial_log:
        log_path = os.path.join(OUT_DIR, f"fold_{fold_id:02d}_trial_log.csv")
        _fsync_csv(log_path, pd.DataFrame(trial_log))
        print(f"  [persist] Saved: {log_path}  ({len(trial_log)} trial rows)", flush=True)


# ---------------------------------------------------------------------------
# Per-fold runner
# ---------------------------------------------------------------------------

def run_fold(
    fold_id: int,
    fold_spec: dict,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime_labels: pd.Series,
    asset_features: pd.DataFrame,
    returns_arr: np.ndarray,
    regime_arr: np.ndarray,
    all_dates: pd.DatetimeIndex,
    config: dict,
) -> dict | None:
    mem_start = _rss_mb()
    sep = "=" * 72
    print(f"\n{sep}", flush=True)
    print(
        f"FOLD {fold_id}  |  "
        f"Train {fold_spec['train_start'][:4]}-{fold_spec['train_end'][:4]}"
        f"  |  Test {fold_spec['test_start'][:4]}",
        flush=True,
    )
    print(sep, flush=True)
    print(f"  [mem] Fold {fold_id} start: {mem_start:.0f} MB", flush=True)

    train_start     = fold_spec["train_start"]
    train_end       = fold_spec["train_end"]
    test_start      = fold_spec["test_start"]
    test_end        = fold_spec["test_end"]
    kappa           = config["kappa"]
    val_frac        = config["val_frac"]
    min_regime_days = config["min_regime_days"]

    smoke = config.get("smoke", False)
    n_trials_t1 = config["n_trials_smoke"] if smoke else config["n_trials_tier1"]
    n_trials_t2 = config["n_trials_smoke"] if smoke else config["n_trials_tier2"]
    n_trials_t3 = config["n_trials_smoke"] if smoke else config["n_trials_tier3"]
    trial_budgets = {1: n_trials_t1, 2: n_trials_t2, 3: n_trials_t3}

    # ── Build training matrices ────────────────────────────────────────────────
    t0 = time.time()
    X_train, Y_train = build_training_matrix(
        asset_features, returns, train_start, train_end
    )
    if len(X_train) == 0:
        print("  WARNING: empty X_train — skipping fold.", flush=True)
        return None

    feat_mask = (asset_features.index >= train_start) & (asset_features.index <= train_end)
    ret_mask  = (returns.index >= train_start) & (returns.index <= train_end)
    common_train = (
        asset_features.loc[feat_mask].index
        .intersection(returns.loc[ret_mask].index)
    )
    regime_train   = regime_labels.reindex(common_train).fillna(2).astype(int)
    ret_train_df   = returns.reindex(common_train).fillna(0.0)

    print(
        f"  X_train: {X_train.shape}  |  "
        f"Train dates: {len(common_train)}"
        f"  ({common_train[0].date()} - {common_train[-1].date()})",
        flush=True,
    )

    # ── Test indices ───────────────────────────────────────────────────────────
    test_mask    = (all_dates >= test_start) & (all_dates <= test_end)
    test_indices = np.where(test_mask)[0]
    test_dates_arr = all_dates[test_indices]

    print(
        f"  Test idx: {len(test_indices)} days"
        f"  ({all_dates[test_indices[0]].date() if len(test_indices) else '?'}"
        f" - {all_dates[test_indices[-1]].date() if len(test_indices) else '?'})",
        flush=True,
    )

    if len(test_indices) == 0:
        print("  WARNING: empty test set — skipping fold.", flush=True)
        return None

    t1 = time.time()
    print(f"  Data preparation: {t1 - t0:.1f}s", flush=True)

    # ── Run BO per (regime, tier) — 12 studies ─────────────────────────────────
    best_configs    = {r: {} for r in [1, 2, 3, 4]}  # {regime_id: {tier: config}}
    regime_tier_stats = {}
    trial_log_all   = []

    print(f"\n  TierSelector β (from Plan 13a, frozen static prior):", flush=True)
    for r in [1, 2, 3, 4]:
        b = PLAN13A_STATIC_BETA[r]
        print(f"    Regime {r} ({REGIME_NAMES[r]:7s}): "
              f"T1={b[0]:.3f}  T2={b[1]:.3f}  T3={b[2]:.3f}", flush=True)

    for regime_id in [1, 2, 3, 4]:
        regime_name = REGIME_NAMES[regime_id]
        regime_mask_train = (regime_train.values == regime_id)
        regime_dates      = common_train[regime_mask_train]
        X_regime          = X_train[regime_mask_train]
        Y_regime          = ret_train_df.values[regime_mask_train]
        n_regime_days     = int(regime_mask_train.sum())

        print(
            f"\n  Regime {regime_id} ({regime_name:7s}): "
            f"{n_regime_days:4d} training days",
            flush=True,
        )

        for tier in [1, 2, 3]:
            t_study_start = time.time()
            n_trials = trial_budgets[tier]

            print(
                f"    [Tier {tier} | {n_trials} trials]  ",
                end="", flush=True,
            )

            # Min-regime-size fallback
            if n_regime_days < min_regime_days:
                print(
                    f"-> only {n_regime_days} days < {min_regime_days} — EW fallback",
                    flush=True,
                )
                ew_cfg = {"tier": tier, "family": "EqualWeight"}
                best_configs[regime_id][tier] = ew_cfg
                regime_tier_stats[f"r{regime_id}_t{tier}"] = {
                    "regime": regime_name, "tier": tier,
                    "family": "EqualWeight", "val_sharpe": float("nan"),
                    "train_sharpe": float("nan"), "gap": float("nan"),
                    "fallback": True, "n_regime_days": n_regime_days,
                }
                continue

            # Run BO study
            trial_log_study = []
            study = run_bo_for_regime_tier(
                fold_id=fold_id,
                regime_id=regime_id,
                tier=tier,
                X_train=X_train,
                Y_train=Y_train,
                X_regime=X_regime,
                Y_regime=Y_regime,
                regime_dates=regime_dates,
                prices=prices,
                N=N,
                kappa=kappa,
                n_trials=n_trials,
                seed=config["seed"] + fold_id * 100 + regime_id * 10 + tier,
                val_frac=val_frac,
                trial_log=trial_log_study,
            )

            trial_log_all.extend(trial_log_study)

            if study is None:
                # val block too small
                ew_cfg = {"tier": tier, "family": "EqualWeight"}
                best_configs[regime_id][tier] = ew_cfg
                regime_tier_stats[f"r{regime_id}_t{tier}"] = {
                    "regime": regime_name, "tier": tier,
                    "family": "EqualWeight", "val_sharpe": float("nan"),
                    "train_sharpe": float("nan"), "gap": float("nan"),
                    "fallback": True, "n_regime_days": n_regime_days,
                }
            else:
                best_trial  = study.best_trial
                best_val    = study.best_value
                best_config = best_trial.user_attrs.get("config",
                                                        best_trial.params.copy())
                best_train_sh = best_trial.user_attrs.get("train_sharpe", float("nan"))
                best_val_sh   = best_trial.user_attrs.get("val_sharpe",   best_val)
                best_gap      = best_trial.user_attrs.get("train_val_gap",
                                                          best_train_sh - best_val_sh)

                completed = [t for t in study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE
                             and t.value is not None]

                best_configs[regime_id][tier] = best_config
                regime_tier_stats[f"r{regime_id}_t{tier}"] = {
                    "regime": regime_name, "tier": tier,
                    "family": best_config.get("family", "?"),
                    "config": best_config,
                    "val_sharpe":   round(float(best_val_sh),  4),
                    "train_sharpe": round(float(best_train_sh), 4),
                    "gap":          round(float(best_gap),      4),
                    "fallback": False,
                    "n_regime_days": n_regime_days,
                    "n_completed_trials": len(completed),
                    "n_pruned": sum(1 for t in study.trials
                                   if t.state == optuna.trial.TrialState.PRUNED),
                }

                t_study_end = time.time()
                print(
                    f"\n    -> val_sharpe={best_val_sh:+.4f}  "
                    f"train_sharpe={best_train_sh:+.4f}  "
                    f"gap={best_gap:+.4f}  "
                    f"family={best_config.get('family', '?')}  "
                    f"({len(completed)}/{n_trials} trials in "
                    f"{t_study_end - t_study_start:.0f}s)",
                    flush=True,
                )

                # Memory cleanup (Lesson 3 from 13b-v2)
                del study
                gc.collect()

    # ── Generate test-time weights (refit on full training data) ───────────────
    print("\n  Computing test weights (refit on full training data) ...", flush=True)
    t2 = time.time()

    best_weights_per_regime_tier = {}

    for regime_id in [1, 2, 3, 4]:
        best_weights_per_regime_tier[regime_id] = {}

        regime_test_mask  = (regime_arr[test_indices] == regime_id)
        regime_test_dates = test_dates_arr[regime_test_mask]

        if len(regime_test_dates) == 0:
            for tier in [1, 2, 3]:
                best_weights_per_regime_tier[regime_id][tier] = np.zeros((0, N))
            continue

        for tier in [1, 2, 3]:
            best_cfg = best_configs[regime_id].get(tier,
                                                   {"tier": 1, "family": "EqualWeight"})
            w_tier = compute_weights_for_config(
                config=best_cfg,
                X_train=X_train, Y_train=Y_train,
                asset_features=asset_features,
                test_dates=regime_test_dates,
                prices=prices, N=N,
            )
            best_weights_per_regime_tier[regime_id][tier] = w_tier

    t3 = time.time()
    print(f"  Test weights computed in {t3 - t2:.1f}s", flush=True)

    # ── Evaluate strategies ────────────────────────────────────────────────────
    m_blend, net_blend = evaluate_hierarchical_blend(
        best_weights_per_regime_tier, returns_arr, regime_arr, test_indices, kappa, N
    )
    m_tier1, net_tier1 = evaluate_tier1_hard(
        best_weights_per_regime_tier, returns_arr, regime_arr, test_indices, kappa, N
    )
    m_ew,    net_ew    = evaluate_ew(returns_arr, test_indices, kappa, N)

    print(
        f"\n  Results:"
        f"\n    Hier Blend  : Sharpe={m_blend['sharpe']:+.4f}"
        f"\n    Tier1 Hard  : Sharpe={m_tier1['sharpe']:+.4f}"
        f"\n    EW          : Sharpe={m_ew['sharpe']:+.4f}",
        flush=True,
    )

    mem_end = _rss_mb()
    print(
        f"  [mem] Fold {fold_id} end: {mem_end:.0f} MB"
        f"  (delta: {mem_end - mem_start:+.0f} MB)",
        flush=True,
    )

    return {
        "fold_id":             fold_id,
        "fold_spec":           fold_spec,
        "metrics_blend":       m_blend,
        "metrics_tier1":       m_tier1,
        "metrics_ew":          m_ew,
        "net_blend":           net_blend,
        "net_tier1":           net_tier1,
        "net_ew":              net_ew,
        "best_configs":        best_configs,
        "regime_tier_stats":   regime_tier_stats,
        "trial_log":           trial_log_all,
        "test_dates":          test_dates_arr,
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    config: dict = CONFIG,
    target_fold: int | None = None,
    skip_existing: bool = False,
    smoke: bool = False,
) -> list:
    np.random.seed(config["seed"])

    run_config = dict(config)
    if smoke:
        run_config["smoke"] = True
        print("\n  [SMOKE TEST] n_trials overridden to", config["n_trials_smoke"],
              "per tier", flush=True)
    else:
        run_config["smoke"] = False

    if target_fold is not None:
        print("\n" + "=" * 72, flush=True)
        print(f"PLAN 13c-v2: REAL BO WITH WARM START — SINGLE FOLD {target_fold}", flush=True)
    else:
        print("\n" + "=" * 72, flush=True)
        print("PLAN 13c-v2: REAL BO WITH WARM START — FULL 12-FOLD RUN", flush=True)
    print("=" * 72, flush=True)

    n1 = run_config["n_trials_smoke"] if smoke else run_config["n_trials_tier1"]
    n2 = run_config["n_trials_smoke"] if smoke else run_config["n_trials_tier2"]
    n3 = run_config["n_trials_smoke"] if smoke else run_config["n_trials_tier3"]
    print(
        f"  Trials per tier: T1={n1}, T2={n2}, T3={n3}"
        f"  val_frac={run_config['val_frac']}, kappa={run_config['kappa']}",
        flush=True,
    )
    print(f"  min_regime_days={run_config['min_regime_days']}", flush=True)
    print(f"  Output: {OUT_DIR}", flush=True)
    print(f"  TierSelector β: static from Plan 13a (T1≈0.99 for all regimes)", flush=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading data ...", flush=True)
    data = load_data_extended()
    prices = data["prices"]
    vix    = data["vix"]
    print(f"  Prices: {prices.shape}  ({prices.index[0].date()} - {prices.index[-1].date()})",
          flush=True)

    returns_raw    = compute_returns(prices)
    asset_features = compute_asset_features(prices)
    regime_labels  = compute_regime_labels(vix)

    common = (prices.index
              .intersection(returns_raw.index)
              .intersection(regime_labels.index))
    prices         = prices.loc[common]
    returns_raw    = returns_raw.loc[common]
    regime_labels  = regime_labels.loc[common]
    asset_features = asset_features.loc[asset_features.index.intersection(common)]
    print(f"  Common dates: {len(common)}", flush=True)

    all_dates   = common
    returns_arr = returns_raw.values
    regime_arr  = regime_labels.reindex(all_dates).fillna(2).astype(int).values

    # ── Generate folds ─────────────────────────────────────────────────────────
    wfv = WalkForwardValidator(
        train_years=run_config["train_years"],
        test_years=run_config["test_years"],
        step_years=run_config["step_years"],
        min_test_start=run_config["min_test_start"],
    )
    all_folds = wfv.generate_folds(data_end=run_config["data_end"])
    print(f"  Generated {len(all_folds)} walk-forward folds", flush=True)

    # ── Print warm-start info ──────────────────────────────────────────────────
    print("\n  Warm-start seed validation:", flush=True)
    for tier in [1, 2, 3]:
        seeds = extract_warm_start_seeds(1, tier)
        families = [s.get(f"t{tier}_family", "?") for s in seeds]
        print(f"    Tier {tier}: {len(seeds)} seeds — families: {families}", flush=True)

    # ── Run folds ──────────────────────────────────────────────────────────────
    fold_results = []
    t_exp_start  = time.time()

    for fold_spec in all_folds:
        fold_id = fold_spec["fold"]

        if target_fold is not None and fold_id != target_fold:
            continue

        if skip_existing and is_fold_complete(fold_id):
            print(
                f"\n  Fold {fold_id} already complete — skipping.",
                flush=True,
            )
            continue

        t_fold_start = time.time()

        result = run_fold(
            fold_id=fold_id,
            fold_spec=fold_spec,
            prices=prices,
            returns=returns_raw,
            regime_labels=regime_labels,
            asset_features=asset_features,
            returns_arr=returns_arr,
            regime_arr=regime_arr,
            all_dates=all_dates,
            config=run_config,
        )

        t_fold_end = time.time()
        fold_min = (t_fold_end - t_fold_start) / 60
        print(f"  Fold {fold_id} completed in {fold_min:.1f} min", flush=True)

        if result is not None:
            save_fold_result(result)

        fold_results.append(result)
        gc.collect()

    t_exp_end = time.time()
    total_min = (t_exp_end - t_exp_start) / 60
    print(f"\nTotal experiment time: {total_min:.1f} min ({total_min/60:.1f} h)", flush=True)

    # ── Summary table ──────────────────────────────────────────────────────────
    valid = [fr for fr in fold_results if fr is not None]
    if valid:
        print("\n" + "=" * 60, flush=True)
        print("PLAN 13c-v2 RESULTS SUMMARY", flush=True)
        print("=" * 60, flush=True)
        print(f"{'Fold':>5}  {'Year':>5}  {'Blend':>8}  {'T1Hard':>8}  {'EW':>8}", flush=True)
        print("-" * 45, flush=True)
        all_blends, all_t1, all_ews = [], [], []
        for fr in valid:
            fid  = fr["fold_id"]
            year = fr["fold_spec"]["test_start"][:4]
            b    = fr["metrics_blend"]["sharpe"]
            t    = fr["metrics_tier1"]["sharpe"]
            e    = fr["metrics_ew"]["sharpe"]
            all_blends.append(b); all_t1.append(t); all_ews.append(e)
            def _f(v):
                try:
                    return f"{float(v):+8.4f}" if not np.isnan(float(v)) else "     nan"
                except Exception:
                    return "     nan"
            print(f"{fid:>5}  {year:>5}  {_f(b)}  {_f(t)}  {_f(e)}", flush=True)
        print("-" * 45, flush=True)
        def _avg(lst):
            fs = [x for x in lst if not np.isnan(float(x))]
            return float(np.mean(fs)) if fs else float("nan")
        print(
            f"{'AVG':>5}  {'':>5}  {_f(_avg(all_blends))}  {_f(_avg(all_t1))}  {_f(_avg(all_ews))}",
            flush=True,
        )
        print("=" * 60, flush=True)
        n = len(valid)
        beats_ew = sum(1 for b in all_blends
                       if not np.isnan(float(b)) and b > 1.02)
        print(f"  Hier Blend > EW (+1.02): {beats_ew}/{n}", flush=True)
        print("PLAN 13c-v2 COMPLETE.", flush=True)

    return fold_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plan 13c-v2: Real BO with warm start from Plan 13a"
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Run only this specific fold (1-12). Omit to run all folds.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=False,
        help="Skip folds whose fold_NN_result.csv already exists.",
    )
    parser.add_argument(
        "--smoke", action="store_true", default=False,
        help="Smoke-test mode: 10 trials per tier per regime (instead of 30/50/80).",
    )
    args = parser.parse_args()

    run_experiment(
        target_fold=args.fold,
        skip_existing=args.skip_existing,
        smoke=args.smoke,
    )
