# experiments/plan13b_v2_true_bo.py -- Plan 13b-v2: True Bayesian Optimization
#
# Replaces the exhaustive search over 117 fixed algorithm configurations (Plan 13b)
# with TRUE Bayesian Optimization over the continuous hyperparameter space of each
# algorithm family. Optuna TPE discovers the optimal configuration per regime.
#
# Architecture:
#   - Continuous search space: 7 Tier 1 + 3 Tier 2 + 3 Tier 3 families
#   - 200 Optuna TPE trials per regime (4 regimes) per fold (12 folds)
#   - Warm start: seed studies with Plan 13b's best configs per regime
#   - Two test strategies: Hard BO (best) and Top-3 BO (blended)
#   - Vectorised Tier 2+3 evaluation: fit once, batch-predict all regime days
#
# Usage:
#   cd Implementierung1
#   python -u -m experiments.plan13b_v2_true_bo

import os
import sys
import time
import pickle
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
    # BO settings
    "n_trials": 200,            # trials per regime per fold
    "seed": RANDOM_SEED,
    # Cap regime days used in the objective to keep trial cost manageable.
    # All trials within one study see the same cap (fair comparison).
    # Reduces Tier 1 slow-algo (MinVar/MaxDiv) time from ~30s to ~10s per trial.
    "max_obj_days": 200,

    # Costs
    "kappa": KAPPA,

    # Walk-forward (identical to Plans 13a/13b/13c)
    "train_years": 8,
    "test_years": 1,
    "step_years": 1,
    "min_test_start": "2013-01-01",
    "data_end": "2024-12-31",

    # Tiers
    "tiers": [1, 2, 3],

    # Top-N for blend strategy
    "top_n": 3,

    # Warm start from Plan 13b
    "warm_start_from_13b": True,
    "plan13b_best_algo_csv": os.path.join(
        RESULTS_DIR, "plan13b_bayesian_opt", "best_algo_per_regime.csv"
    ),

    # Prior results for comparison table
    "plan13a_summary": os.path.join(RESULTS_DIR, "plan13a_hierarchical", "summary_metrics.csv"),
    "plan13b_summary": os.path.join(RESULTS_DIR, "plan13b_bayesian_opt", "summary_metrics.csv"),
    "plan13c_summary": os.path.join(RESULTS_DIR, "plan13c_hybrid", "summary_metrics.csv"),

    # Output
    "output_dir": os.path.join(RESULTS_DIR, "plan13b_v2_true_bo"),
}

OUT_DIR        = CONFIG["output_dir"]
STUDIES_DIR    = os.path.join(OUT_DIR, "optuna_studies")
ANALYSIS_DIR   = os.path.join(OUT_DIR, "search_analysis")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(STUDIES_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

N = N_ASSETS  # 5


# ---------------------------------------------------------------------------
# Metric helpers
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
    """
    Compute annualised Sharpe ratio (net of switching costs).

    weights : (T_reg, N)
    rets    : (T_reg, N)
    """
    port_ret = np.sum(weights * rets, axis=1)
    if len(weights) > 1:
        turnover = np.sum(np.abs(np.diff(weights, axis=0)), axis=1)
        costs = kappa * turnover
        net_ret = port_ret[1:] - costs
    else:
        net_ret = port_ret
    if len(net_ret) < 5 or np.std(net_ret) < 1e-10:
        return 0.0
    return float(np.mean(net_ret) / np.std(net_ret) * np.sqrt(252))


# ---------------------------------------------------------------------------
# Continuous search space (TPE-compatible)
# ---------------------------------------------------------------------------

def sample_tier1(trial: optuna.Trial) -> dict:
    # Use "t1_family" to avoid name collision with t2/t3 family params
    family = trial.suggest_categorical("t1_family", [
        "EqualWeight", "MinVariance", "RiskParity", "MaxDiversification",
        "MeanVariance", "Momentum", "TrendFollowing",
    ])
    if family == "EqualWeight":
        return {"tier": 1, "family": "EqualWeight"}
    elif family == "MinVariance":
        return {"tier": 1, "family": "MinVariance",
                "minvar_lookback": trial.suggest_int("minvar_lookback", 10, 252)}
    elif family == "RiskParity":
        return {"tier": 1, "family": "RiskParity",
                "riskparity_lookback": trial.suggest_int("riskparity_lookback", 10, 252)}
    elif family == "MaxDiversification":
        return {"tier": 1, "family": "MaxDiversification",
                "maxdiv_lookback": trial.suggest_int("maxdiv_lookback", 10, 252)}
    elif family == "MeanVariance":
        return {"tier": 1, "family": "MeanVariance",
                "mv_lookback": trial.suggest_int("mv_lookback", 10, 252),
                "mv_gamma": trial.suggest_float("mv_gamma", 0.1, 10.0, log=True)}
    elif family == "Momentum":
        return {"tier": 1, "family": "Momentum",
                "mom_lookback": trial.suggest_int("mom_lookback", 5, 252),
                "mom_type": trial.suggest_categorical("mom_type", ["linear", "exp"])}
    else:  # TrendFollowing
        return {"tier": 1, "family": "TrendFollowing",
                "trend_lookback": trial.suggest_int("trend_lookback", 5, 252),
                "trend_beta": trial.suggest_int("trend_beta", 1, 5)}


def sample_tier2(trial: optuna.Trial) -> dict:
    # Use "t2_family" to avoid name collision
    family = trial.suggest_categorical("t2_family", ["Ridge", "Lasso", "ElasticNet"])
    lookback = trial.suggest_int("lookback", 20, 252)
    if family == "Ridge":
        return {"tier": 2, "family": "Ridge", "lookback": lookback,
                "ridge_alpha": trial.suggest_float("ridge_alpha", 1e-4, 100.0, log=True)}
    elif family == "Lasso":
        return {"tier": 2, "family": "Lasso", "lookback": lookback,
                "lasso_alpha": trial.suggest_float("lasso_alpha", 1e-4, 10.0, log=True)}
    else:  # ElasticNet
        return {"tier": 2, "family": "ElasticNet", "lookback": lookback,
                "enet_alpha": trial.suggest_float("enet_alpha", 1e-4, 10.0, log=True),
                "enet_l1_ratio": trial.suggest_float("enet_l1_ratio", 0.1, 0.9)}


def sample_tier3(trial: optuna.Trial) -> dict:
    # Use "t3_family" to avoid name collision
    family = trial.suggest_categorical("t3_family", [
        "RandomForest", "GradientBoosting", "MLP",
    ])
    lookback = trial.suggest_int("lookback", 20, 252)
    if family == "RandomForest":
        return {"tier": 3, "family": "RandomForest", "lookback": lookback,
                "rf_n_estimators": trial.suggest_int("rf_n_estimators", 50, 300),
                "rf_max_depth": trial.suggest_int("rf_max_depth", 3, 20),
                "rf_min_leaf": trial.suggest_int("rf_min_leaf", 1, 20)}
    elif family == "GradientBoosting":
        return {"tier": 3, "family": "GradientBoosting", "lookback": lookback,
                "gbm_n_estimators": trial.suggest_int("gbm_n_estimators", 50, 300),
                "gbm_max_depth": trial.suggest_int("gbm_max_depth", 2, 10),
                "gbm_lr": trial.suggest_float("gbm_lr", 0.001, 0.5, log=True),
                "gbm_subsample": trial.suggest_float("gbm_subsample", 0.5, 1.0)}
    else:  # MLP
        hidden1 = trial.suggest_int("mlp_hidden1", 16, 256)
        hidden2 = trial.suggest_int("mlp_hidden2", 0, 128)
        return {"tier": 3, "family": "MLP", "lookback": lookback,
                "mlp_hidden1": hidden1, "mlp_hidden2": hidden2,
                "mlp_alpha": trial.suggest_float("mlp_alpha", 1e-5, 0.1, log=True)}


def sample_algorithm(trial: optuna.Trial) -> dict:
    """
    Sample one algorithm config from the joint continuous search space.

    Uses separate parameter names (t1_family, t2_family, t3_family) to avoid
    Optuna's restriction on dynamic value spaces for the same parameter name.
    """
    tier = trial.suggest_categorical("tier", [1, 2, 3])
    if tier == 1:
        return sample_tier1(trial)
    elif tier == 2:
        return sample_tier2(trial)
    else:
        return sample_tier3(trial)


# ---------------------------------------------------------------------------
# Algorithm factory
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
        return MeanVariance(
            lookback=config["mv_lookback"],
            risk_aversion=config["mv_gamma"],
        )
    elif family == "Momentum":
        return Momentum(
            lookback=config["mom_lookback"],
            weighting=config["mom_type"],
        )
    elif family == "TrendFollowing":
        return TrendFollowing(
            lookback=config["trend_lookback"],
            beta=config.get("trend_beta", 1),
        )

    # ── Tier 2 ──────────────────────────────────────────────────────────────
    elif family == "Ridge":
        return RidgePortfolio(
            lambda_ridge=config["ridge_alpha"],
            lookback=config["lookback"],
        )
    elif family == "Lasso":
        return LassoPortfolio(
            lambda_lasso=config["lasso_alpha"],
            lookback=config["lookback"],
        )
    elif family == "ElasticNet":
        return ElasticNetPortfolio(
            lambda_en=config["enet_alpha"],
            l1_ratio=config["enet_l1_ratio"],
            lookback=config["lookback"],
        )

    # ── Tier 3 ──────────────────────────────────────────────────────────────
    elif family == "RandomForest":
        return Tier3Algorithm(
            family="RandomForest",
            model_class=RandomForestRegressor,
            model_params={
                "n_estimators": config["rf_n_estimators"],
                "max_depth": config["rf_max_depth"],
                "min_samples_leaf": config["rf_min_leaf"],
                "random_state": 42,
                "n_jobs": -1,
            },
            lookback=config["lookback"],
            name=f"RF_BO_n{config['rf_n_estimators']}_d{config['rf_max_depth']}",
        )
    elif family == "GradientBoosting":
        return Tier3Algorithm(
            family="GradientBoosting",
            model_class=GradientBoostingRegressor,
            model_params={
                "n_estimators": config["gbm_n_estimators"],
                "max_depth": config["gbm_max_depth"],
                "learning_rate": config["gbm_lr"],
                "subsample": config["gbm_subsample"],
                "random_state": 42,
            },
            lookback=config["lookback"],
            name=f"GBM_BO_n{config['gbm_n_estimators']}_d{config['gbm_max_depth']}",
        )
    elif family == "MLP":
        h1 = config["mlp_hidden1"]
        h2 = config.get("mlp_hidden2", 0)
        hidden = (h1,) if h2 == 0 else (h1, h2)
        return Tier3Algorithm(
            family="MLP",
            model_class=MLPRegressor,
            model_params={
                "hidden_layer_sizes": hidden,
                "alpha": config["mlp_alpha"],
                "max_iter": 300,
                "random_state": 42,
                "early_stopping": True,
                "validation_fraction": 0.1,
            },
            lookback=config["lookback"],
            name=f"MLP_BO_h{h1}x{h2}",
        )

    raise ValueError(f"Unknown algorithm family: {family}")


# ---------------------------------------------------------------------------
# Warm start: parse Plan 13b best configs
# ---------------------------------------------------------------------------

def _algo_name_to_optuna_params(algo_name: str) -> dict | None:
    """
    Convert a Plan 13b algorithm name to Optuna trial params for warm start.
    Returns None if the name cannot be parsed.
    """
    # The warm start params must match the Optuna parameter names exactly.
    # Tier 1 uses "t1_family", Tier 2 uses "t2_family", Tier 3 uses "t3_family".
    try:
        if algo_name == "EqualWeight":
            return {"tier": 1, "t1_family": "EqualWeight"}

        if algo_name.startswith("MinVar_L"):
            L = int(algo_name.split("_L")[1])
            return {"tier": 1, "t1_family": "MinVariance", "minvar_lookback": L}

        if algo_name.startswith("RiskParity_L"):
            L = int(algo_name.split("_L")[1])
            return {"tier": 1, "t1_family": "RiskParity", "riskparity_lookback": L}

        if algo_name.startswith("MaxDiv_L"):
            L = int(algo_name.split("_L")[1])
            return {"tier": 1, "t1_family": "MaxDiversification", "maxdiv_lookback": L}

        if algo_name.startswith("MeanVar_L"):
            # MeanVar_L60_G1.0
            parts = algo_name.split("_")
            L = int(parts[1][1:])
            G = float(parts[2][1:])
            return {"tier": 1, "t1_family": "MeanVariance", "mv_lookback": L, "mv_gamma": G}

        if algo_name.startswith("Momentum_L"):
            # Momentum_L60_linear or Momentum_L60_exp
            parts = algo_name.split("_")
            L = int(parts[1][1:])
            w = parts[2]  # "linear" or "exp"
            return {"tier": 1, "t1_family": "Momentum", "mom_lookback": L, "mom_type": w}

        if algo_name.startswith("Trend_L"):
            # Trend_L60_B1
            parts = algo_name.split("_")
            L = int(parts[1][1:])
            beta = int(parts[2][1:]) if len(parts) > 2 else 1
            beta = min(max(beta, 1), 5)  # clamp to search space
            return {"tier": 1, "t1_family": "TrendFollowing",
                    "trend_lookback": L, "trend_beta": beta}

        if algo_name.startswith("Ridge_L"):
            # Ridge_L60_a0.01
            parts = algo_name.split("_")
            L = int(parts[1][1:])
            a = float(parts[2][1:])
            return {"tier": 2, "t2_family": "Ridge", "lookback": L, "ridge_alpha": a}

        if algo_name.startswith("Lasso_L"):
            parts = algo_name.split("_")
            L = int(parts[1][1:])
            a = float(parts[2][1:])
            return {"tier": 2, "t2_family": "Lasso", "lookback": L, "lasso_alpha": a}

        if algo_name.startswith("ElasticNet_L"):
            # ElasticNet_L60_a0.01_r0.25
            parts = algo_name.split("_")
            L = int(parts[1][1:])
            a = float(parts[2][1:])
            r = float(parts[3][1:])
            return {"tier": 2, "t2_family": "ElasticNet",
                    "lookback": L, "enet_alpha": a, "enet_l1_ratio": r}

        if algo_name.startswith("RF_n"):
            # RF_n300_d5_L60 or RF_n300_dNone_L60
            parts = algo_name.split("_")
            n = int(parts[1][1:])
            d_str = parts[2][1:]
            d = int(d_str) if d_str != "None" else 15
            d = min(max(d, 3), 20)   # clamp to search space [3, 20]
            n = min(max(n, 50), 300)  # clamp to search space [50, 300]
            L = int(parts[3][1:])
            return {"tier": 3, "t3_family": "RandomForest", "lookback": L,
                    "rf_n_estimators": n, "rf_max_depth": d, "rf_min_leaf": 1}

        if algo_name.startswith("GBM_n"):
            # GBM_n300_d5_lr0.1_L60
            parts = algo_name.split("_")
            n = int(parts[1][1:])
            d = int(parts[2][1:])
            lr = float(parts[3][2:])
            L = int(parts[4][1:])
            n = min(max(n, 50), 300)  # clamp to search space [50, 300]
            d = min(max(d, 2), 10)    # clamp to search space [2, 10]
            return {"tier": 3, "t3_family": "GradientBoosting", "lookback": L,
                    "gbm_n_estimators": n, "gbm_max_depth": d,
                    "gbm_lr": lr, "gbm_subsample": 0.8}

        if algo_name.startswith("MLP_h"):
            # MLP_h64_a0.0001_L60 or MLP_h64x32_a0.0001_L60
            parts = algo_name.split("_")
            h_str = parts[1][1:]
            h_parts = h_str.split("x")
            h1 = int(h_parts[0])
            h2 = int(h_parts[1]) if len(h_parts) > 1 else 0
            a = float(parts[2][1:])
            L = int(parts[3][1:])
            return {"tier": 3, "t3_family": "MLP", "lookback": L,
                    "mlp_hidden1": h1, "mlp_hidden2": h2, "mlp_alpha": a}

    except Exception:
        pass
    return None


def load_warmstart_configs(csv_path: str) -> dict:
    """
    Load Plan 13b's best algo per regime per fold.

    Returns {(fold_id, regime_id): optuna_params_dict}
    """
    if not os.path.exists(csv_path):
        print(f"  [warm start] CSV not found: {csv_path}", flush=True)
        return {}

    df = pd.read_csv(csv_path)
    configs = {}
    for _, row in df.iterrows():
        fold_id   = int(row["fold"])
        regime_id = int(row["regime_id"])
        algo_name = str(row["best_algo_name"])
        params = _algo_name_to_optuna_params(algo_name)
        if params is not None:
            configs[(fold_id, regime_id)] = params
    print(f"  [warm start] Loaded {len(configs)} best configs from Plan 13b", flush=True)
    return configs


# ---------------------------------------------------------------------------
# BO Objective function (per regime)
# ---------------------------------------------------------------------------

def make_objective(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_regime: np.ndarray,
    regime_returns: np.ndarray,
    regime_dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
    N: int,
    kappa: float,
):
    """
    Create the Optuna objective function for one regime.

    X_train       : (T_train, n_feat)  — full training period features
    Y_train       : (T_train, N)       — full training period returns
    X_regime      : (T_reg, n_feat)    — features for regime days (subset of X_train)
    regime_returns: (T_reg, N)         — returns for regime days
    regime_dates  : DatetimeIndex      — dates for regime days (for Tier 1 price slices)
    prices        : full price DataFrame (for Tier 1 compute_weights)
    """

    def objective(trial: optuna.Trial) -> float:
        config = sample_algorithm(trial)
        # Store the full config so we can retrieve it from best_trial.user_attrs
        try:
            trial.set_user_attr("config", config)
        except Exception:
            pass
        try:
            algo = create_algorithm_from_config(config)
        except Exception:
            return 0.0

        if isinstance(algo, TrainablePortfolioAlgorithm):
            # ── Tier 2+3: fit once, batch-predict all regime days ──────────────
            try:
                algo.fit(X_train, Y_train)
            except Exception:
                return 0.0

            if not algo._is_fitted or algo._scaler is None:
                return 0.0

            try:
                X_scaled = algo._scaler.transform(X_regime)  # (T_reg, n_feat)
            except Exception:
                return 0.0

            # Batch predict: Tier 3 has _models dict, Tier 2 has _model
            try:
                if (hasattr(algo, "_models") and isinstance(algo._models, dict)
                        and algo._models):
                    mu_all = np.column_stack([
                        algo._models[j].predict(X_scaled) for j in range(N)
                    ])  # (T_reg, N)
                else:
                    mu_all = algo._model.predict(X_scaled)  # (T_reg, N)
            except Exception:
                return 0.0

            weights = np.array([_softmax(mu_all[i]) for i in range(len(regime_dates))])

        else:
            # ── Tier 1: compute weights day by day ────────────────────────────
            weights = []
            for d in regime_dates:
                try:
                    ph = prices.loc[:d]
                    w = algo.compute_weights(ph)
                except Exception:
                    w = np.ones(N) / N
                w = np.where(np.isfinite(w), w, 0.0)
                w = np.clip(w, 0.0, None)
                s = w.sum()
                w = w / s if s > 1e-12 else np.ones(N) / N
                weights.append(w)
            weights = np.array(weights)

        return _compute_sharpe(weights, regime_returns, kappa)

    return objective


# ---------------------------------------------------------------------------
# BO study runner (per fold × regime)
# ---------------------------------------------------------------------------

def run_bo_for_regime(
    fold_id: int,
    regime_id: int,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_regime: np.ndarray,
    regime_returns: np.ndarray,
    regime_dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
    N: int,
    kappa: float,
    n_trials: int,
    seed: int,
    warmstart_params: dict | None,
    max_obj_days: int = 200,
) -> optuna.Study:
    """
    Run one Optuna TPE study for a single regime in a single fold.

    max_obj_days : Cap regime days used in the objective to keep trial cost
                   manageable. All trials within one study see the same cap,
                   ensuring fair comparison across trials. The cap primarily
                   speeds up Tier 1 slow algorithms (MinVar/MaxDiv/MeanVar).
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Apply max_obj_days cap: use first max_obj_days regime days
    # (same subsample for all trials → fair comparison within the study)
    T_reg = len(regime_dates)
    if T_reg > max_obj_days:
        rng = np.random.RandomState(seed)
        subset_idx = np.sort(rng.choice(T_reg, max_obj_days, replace=False))
        X_regime_obj       = X_regime[subset_idx]
        regime_returns_obj = regime_returns[subset_idx]
        regime_dates_obj   = regime_dates[subset_idx]
    else:
        X_regime_obj       = X_regime
        regime_returns_obj = regime_returns
        regime_dates_obj   = regime_dates

    # Warm start: enqueue the Plan 13b best config as the first trial
    if warmstart_params is not None:
        try:
            study.enqueue_trial(warmstart_params)
        except Exception as e:
            print(f"    [warm start] Failed to enqueue: {e}", flush=True)

    objective = make_objective(
        X_train=X_train,
        Y_train=Y_train,
        X_regime=X_regime_obj,
        regime_returns=regime_returns_obj,
        regime_dates=regime_dates_obj,
        prices=prices,
        N=N,
        kappa=kappa,
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study


# ---------------------------------------------------------------------------
# Compute weights for a given config (for test evaluation)
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
    Given a HP config, build and fit the algorithm, then compute weights
    for each test date.

    Returns weights: (T_test, N)
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

        # Attach full features for fast lookup during compute_weights
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


def evaluate_hard_bo(
    best_weights_per_regime: dict,
    returns_arr: np.ndarray,
    regime_arr: np.ndarray,
    test_indices: np.ndarray,
    kappa: float,
    N: int,
) -> tuple:
    """Hard BO: at each test day, use the best BO config for that regime."""
    # Build a lookup: test_index → weight row
    # best_weights_per_regime[regime_id] = (T_test, N)
    # test_indices[j] = global index for the j-th test day
    # regime_arr[test_indices[j]] = regime for j-th test day

    # Build per-regime index map
    regime_test_idx = {}
    for regime_id in [1, 2, 3, 4]:
        regime_test_idx[regime_id] = [j for j, i in enumerate(test_indices)
                                       if regime_arr[i] == regime_id]

    # Assemble weights for all test days (in order)
    weights_all = np.zeros((len(test_indices), N))
    for regime_id in [1, 2, 3, 4]:
        idxs = regime_test_idx[regime_id]
        if not idxs or regime_id not in best_weights_per_regime:
            for j in idxs:
                weights_all[j] = np.ones(N) / N
        else:
            w_reg = best_weights_per_regime[regime_id]  # (T_test_reg, N)
            for pos, j in enumerate(idxs):
                if pos < len(w_reg):
                    weights_all[j] = w_reg[pos]
                else:
                    weights_all[j] = np.ones(N) / N

    def _fn(i):
        j = np.searchsorted(test_indices, i)
        if j < len(test_indices) and test_indices[j] == i:
            return weights_all[j]
        return np.ones(N) / N

    return _eval_strategy(_fn, returns_arr, test_indices, kappa, N)


def evaluate_top3_blend(
    top3_weights_per_regime: dict,
    returns_arr: np.ndarray,
    regime_arr: np.ndarray,
    test_indices: np.ndarray,
    kappa: float,
    N: int,
) -> tuple:
    """Top-3 BO: blend weights from top-3 BO trials per regime."""
    # top3_weights_per_regime[regime_id] = list of (T_test_reg, N) arrays

    regime_test_idx = {}
    for regime_id in [1, 2, 3, 4]:
        regime_test_idx[regime_id] = [j for j, i in enumerate(test_indices)
                                       if regime_arr[i] == regime_id]

    weights_all = np.zeros((len(test_indices), N))
    for regime_id in [1, 2, 3, 4]:
        idxs = regime_test_idx[regime_id]
        if not idxs or regime_id not in top3_weights_per_regime:
            for j in idxs:
                weights_all[j] = np.ones(N) / N
        else:
            w_list = top3_weights_per_regime[regime_id]  # list of (T_test_reg, N)
            if not w_list:
                for j in idxs:
                    weights_all[j] = np.ones(N) / N
            else:
                # Stack and mean: (n_top, T_test_reg, N) → (T_test_reg, N)
                blended = np.mean(np.stack(w_list, axis=0), axis=0)
                for pos, j in enumerate(idxs):
                    if pos < len(blended):
                        weights_all[j] = blended[pos]
                    else:
                        weights_all[j] = np.ones(N) / N

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
    warmstart_all: dict,
) -> dict:
    mem_start = _rss_mb()
    sep = "=" * 72
    print(f"\n{sep}", flush=True)
    print(
        f"FOLD {fold_id}  |  "
        f"Train {fold_spec['train_start'][:4]}–{fold_spec['train_end'][:4]}"
        f"  |  Test {fold_spec['test_start'][:4]}",
        flush=True,
    )
    print(sep, flush=True)
    print(f"  [mem] Fold {fold_id} start: {mem_start:.0f} MB", flush=True)

    train_start = fold_spec["train_start"]
    train_end   = fold_spec["train_end"]
    test_start  = fold_spec["test_start"]
    test_end    = fold_spec["test_end"]
    kappa       = config["kappa"]
    n_trials    = config["n_trials"]

    # ── Build training matrices ───────────────────────────────────────────────
    t0 = time.time()
    X_train, Y_train = build_training_matrix(
        asset_features, returns, train_start, train_end
    )
    if len(X_train) == 0:
        print("  WARNING: empty X_train — skipping fold.", flush=True)
        return None

    # Get common training dates (same logic as build_training_matrix)
    feat_mask = (asset_features.index >= train_start) & (asset_features.index <= train_end)
    ret_mask  = (returns.index >= train_start) & (returns.index <= train_end)
    common_train = (
        asset_features.loc[feat_mask].index
        .intersection(returns.loc[ret_mask].index)
    )
    regime_train = regime_labels.reindex(common_train).fillna(2).astype(int)
    ret_train_df = returns.reindex(common_train).fillna(0.0)

    print(
        f"  X_train: {X_train.shape}  |  "
        f"Train dates: {len(common_train)}"
        f"  ({common_train[0].date()} – {common_train[-1].date()})",
        flush=True,
    )

    # ── Test indices (positions in all_dates) ─────────────────────────────────
    test_mask = (all_dates >= test_start) & (all_dates <= test_end)
    test_indices = np.where(test_mask)[0]
    train_mask_all = (all_dates >= train_start) & (all_dates <= train_end)
    train_indices = np.where(train_mask_all)[0]

    test_dates_arr = all_dates[test_indices]

    print(
        f"  Train idx: {len(train_indices)} days"
        f"  |  Test idx: {len(test_indices)} days"
        f"  ({all_dates[test_indices[0]].date() if len(test_indices) else '?'}"
        f" – {all_dates[test_indices[-1]].date() if len(test_indices) else '?'})",
        flush=True,
    )

    if len(test_indices) == 0:
        print("  WARNING: empty test set — skipping fold.", flush=True)
        return None

    t1 = time.time()
    print(f"  Data preparation: {t1 - t0:.1f}s", flush=True)

    # ── Run BO per regime ─────────────────────────────────────────────────────
    bo_studies = {}
    best_configs = {}       # regime_id → best HP config dict
    top3_configs = {}       # regime_id → list of top-3 HP config dicts
    regime_train_days = {}  # regime_id → n_training_days

    for regime_id in [1, 2, 3, 4]:
        regime_name = REGIME_NAMES[regime_id]
        t_reg_start = time.time()

        # Regime training days
        regime_mask_train = (regime_train.values == regime_id)
        regime_dates = common_train[regime_mask_train]
        X_regime = X_train[regime_mask_train]
        regime_returns = ret_train_df.values[regime_mask_train]  # (T_reg, N)

        n_regime_days = int(regime_mask_train.sum())
        regime_train_days[regime_id] = n_regime_days
        print(
            f"  Regime {regime_id} ({regime_name:7s}): "
            f"{n_regime_days:4d} training days",
            end="  ", flush=True,
        )

        if n_regime_days < 10:
            print("→ too few days, using EW fallback", flush=True)
            best_configs[regime_id] = {"tier": 1, "family": "EqualWeight"}
            top3_configs[regime_id] = [{"tier": 1, "family": "EqualWeight"}] * 3
            bo_studies[regime_id] = None
            continue

        # Warm start
        warmstart_params = warmstart_all.get((fold_id, regime_id)) if config.get(
            "warm_start_from_13b"
        ) else None

        # Run BO
        study = run_bo_for_regime(
            fold_id=fold_id,
            regime_id=regime_id,
            X_train=X_train,
            Y_train=Y_train,
            X_regime=X_regime,
            regime_returns=regime_returns,
            regime_dates=regime_dates,
            prices=prices,
            N=N,
            kappa=kappa,
            n_trials=n_trials,
            seed=config["seed"] + fold_id * 10 + regime_id,
            warmstart_params=warmstart_params,
            max_obj_days=config.get("max_obj_days", 200),
        )
        bo_studies[regime_id] = study

        best_trial = study.best_trial
        best_sharpe = study.best_value

        # Retrieve the full config stored as user_attr (avoids t1/t2/t3_family translation)
        best_config = best_trial.user_attrs.get("config", best_trial.params.copy())

        # Extract top-3 completed trials by objective value
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE
                     and t.value is not None]
        completed_sorted = sorted(completed, key=lambda t: t.value, reverse=True)
        top3 = completed_sorted[:min(config["top_n"], len(completed_sorted))]
        top3_cfgs = [t.user_attrs.get("config", t.params.copy()) for t in top3]

        best_configs[regime_id] = best_config
        top3_configs[regime_id] = top3_cfgs

        t_reg_end = time.time()
        print(
            f"→ best Sharpe={best_sharpe:+.4f}  "
            f"family={best_config.get('family', '?')}  "
            f"({n_trials} trials in {t_reg_end - t_reg_start:.0f}s)",
            flush=True,
        )

    # ── Save Optuna studies ───────────────────────────────────────────────────
    for regime_id, study in bo_studies.items():
        if study is not None:
            pkl_path = os.path.join(
                STUDIES_DIR, f"fold_{fold_id:02d}_regime_{regime_id}.pkl"
            )
            with open(pkl_path, "wb") as f:
                pickle.dump(study, f)

    # Free study objects — already on disk, no need to hold in RAM
    del bo_studies
    gc.collect()

    # ── Generate test-time weights ────────────────────────────────────────────
    print("  Computing test weights ...", flush=True)
    t2 = time.time()

    # Hard BO: best config per regime → compute weights for all test days in that regime
    best_weights_per_regime = {}
    top3_weights_per_regime = {}

    for regime_id in [1, 2, 3, 4]:
        # Test days belonging to this regime
        regime_test_mask = (regime_arr[test_indices] == regime_id)
        regime_test_dates = test_dates_arr[regime_test_mask]

        if len(regime_test_dates) == 0:
            best_weights_per_regime[regime_id] = np.zeros((0, N))
            top3_weights_per_regime[regime_id] = []
            continue

        # Hard BO weights
        best_cfg = best_configs[regime_id]
        w_hard = compute_weights_for_config(
            config=best_cfg,
            X_train=X_train,
            Y_train=Y_train,
            asset_features=asset_features,
            test_dates=regime_test_dates,
            prices=prices,
            N=N,
        )
        best_weights_per_regime[regime_id] = w_hard

        # Top-3 weights: compute for each of the top-3 configs
        w_top3_list = []
        for cfg in top3_configs[regime_id]:
            w_k = compute_weights_for_config(
                config=cfg,
                X_train=X_train,
                Y_train=Y_train,
                asset_features=asset_features,
                test_dates=regime_test_dates,
                prices=prices,
                N=N,
            )
            w_top3_list.append(w_k)
        top3_weights_per_regime[regime_id] = w_top3_list

    t3 = time.time()
    print(f"  Test weights computed in {t3 - t2:.1f}s", flush=True)

    # ── Evaluate strategies ───────────────────────────────────────────────────
    m_hard, net_hard = evaluate_hard_bo(
        best_weights_per_regime, returns_arr, regime_arr, test_indices, kappa, N
    )
    m_blend, net_blend = evaluate_top3_blend(
        top3_weights_per_regime, returns_arr, regime_arr, test_indices, kappa, N
    )
    m_ew, net_ew = evaluate_ew(returns_arr, test_indices, kappa, N)

    print(
        f"  Results:"
        f"\n    Hard BO  : Sharpe={m_hard['sharpe']:+.4f}"
        f"\n    Top-3 BO : Sharpe={m_blend['sharpe']:+.4f}"
        f"\n    EW       : Sharpe={m_ew['sharpe']:+.4f}",
        flush=True,
    )

    mem_end = _rss_mb()
    print(
        f"  [mem] Fold {fold_id} end: {mem_end:.0f} MB"
        f"  (delta: {mem_end - mem_start:+.0f} MB)",
        flush=True,
    )

    return {
        "fold_id":               fold_id,
        "fold_spec":             fold_spec,
        "metrics_hard":          m_hard,
        "metrics_blend":         m_blend,
        "metrics_ew":            m_ew,
        "net_hard":              net_hard,
        "net_blend":             net_blend,
        "net_ew":                net_ew,
        "best_configs":          best_configs,
        "top3_configs":          top3_configs,
        "test_dates":            test_dates_arr,
        "train_days_per_regime": regime_train_days,
        # bo_studies NOT included — freed after pkl save to reduce peak RSS
    }


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(fold_results: list, config: dict) -> None:
    valid = [fr for fr in fold_results if fr is not None]

    # ── 1. summary_metrics.csv ────────────────────────────────────────────────
    rows = []
    for fr in valid:
        fold_id = fr["fold_id"]
        year    = fr["fold_spec"]["test_start"][:4]
        rows.append({
            "fold":             fold_id,
            "test_year":        year,
            "hard_sharpe":      fr["metrics_hard"]["sharpe"],
            "blend_sharpe":     fr["metrics_blend"]["sharpe"],
            "ew_sharpe":        fr["metrics_ew"]["sharpe"],
            "hard_vs_ew":       fr["metrics_hard"]["sharpe"] - fr["metrics_ew"]["sharpe"],
            "blend_vs_ew":      fr["metrics_blend"]["sharpe"] - fr["metrics_ew"]["sharpe"],
            "hard_ann_return":  fr["metrics_hard"]["ann_return"],
            "blend_ann_return": fr["metrics_blend"]["ann_return"],
            "ew_ann_return":    fr["metrics_ew"]["ann_return"],
            "hard_maxdd":       fr["metrics_hard"]["max_drawdown"],
            "blend_maxdd":      fr["metrics_blend"]["max_drawdown"],
            "ew_maxdd":         fr["metrics_ew"]["max_drawdown"],
            "hard_turnover":    fr["metrics_hard"].get("avg_daily_turnover", np.nan),
            "blend_turnover":   fr["metrics_blend"].get("avg_daily_turnover", np.nan),
        })
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(os.path.join(OUT_DIR, "summary_metrics.csv"), index=False)
    print(f"  Saved: summary_metrics.csv", flush=True)

    # ── 2. best_config_per_regime.json ───────────────────────────────────────
    best_cfg_all = {}
    for fr in valid:
        fold_id = fr["fold_id"]
        best_cfg_all[fold_id] = {}
        for regime_id in [1, 2, 3, 4]:
            cfg = fr["best_configs"].get(regime_id, {})
            best_cfg_all[fold_id][regime_id] = cfg
    with open(os.path.join(OUT_DIR, "best_config_per_regime.json"), "w") as f:
        json.dump(best_cfg_all, f, indent=2)
    print(f"  Saved: best_config_per_regime.json", flush=True)

    # ── 3. search_analysis/best_family_per_regime.csv ─────────────────────────
    fam_rows = []
    for fr in valid:
        fold_id = fr["fold_id"]
        year    = fr["fold_spec"]["test_start"][:4]
        for regime_id in [1, 2, 3, 4]:
            cfg = fr["best_configs"].get(regime_id, {})
            fam_rows.append({
                "fold":        fold_id,
                "test_year":   year,
                "regime_id":   regime_id,
                "regime_name": REGIME_NAMES[regime_id],
                "tier":        cfg.get("tier", "?"),
                "family":      cfg.get("family", "?"),
                "bo_sharpe":   fr["fold_spec"].get("bo_train_sharpe", np.nan),
            })
    pd.DataFrame(fam_rows).to_csv(
        os.path.join(ANALYSIS_DIR, "best_family_per_regime.csv"), index=False
    )

    # ── 4. HP importance analysis ─────────────────────────────────────────────
    try:
        imp_rows = []
        for fr in valid:
            fold_id = fr["fold_id"]
            year    = fr["fold_spec"]["test_start"][:4]
            for regime_id, study in fr.get("bo_studies", {}).items():
                if study is None:
                    continue
                try:
                    importances = optuna.importance.get_param_importances(study)
                    for param, imp in importances.items():
                        imp_rows.append({
                            "fold": fold_id, "test_year": year,
                            "regime_id": regime_id,
                            "regime_name": REGIME_NAMES[regime_id],
                            "param": param, "importance": imp,
                        })
                except Exception:
                    pass
        if imp_rows:
            pd.DataFrame(imp_rows).to_csv(
                os.path.join(ANALYSIS_DIR, "hp_importance.csv"), index=False
            )
            print(f"  Saved: search_analysis/hp_importance.csv", flush=True)
    except Exception as e:
        print(f"  [HP importance] skipped: {e}", flush=True)

    print(f"\nAll results saved to: {OUT_DIR}", flush=True)


# ---------------------------------------------------------------------------
# Per-fold persistence helpers
# ---------------------------------------------------------------------------

def is_fold_complete(fold_id: int) -> bool:
    """Return True if fold_NN_result.csv exists and has 3 strategy rows."""
    csv_path = os.path.join(OUT_DIR, f"fold_{fold_id:02d}_result.csv")
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        return len(df) >= 3
    except Exception:
        return False


def save_fold_result(fold_result: dict, config: dict) -> None:
    """
    Persist per-fold test metrics immediately after a fold completes.

    Writes two files to OUT_DIR:
      fold_NN_result.csv       — 3 rows (hard_bo, top3_bo, ew) with key metrics
      fold_NN_best_configs.json — best HP config per regime
    Both are flushed + fsync'd so a subsequent crash can't corrupt them.
    """
    fold_id   = fold_result["fold_id"]
    fold_spec = fold_result["fold_spec"]
    test_year = fold_spec["test_start"][:4]

    train_days_json = json.dumps(
        {str(k): v for k, v in fold_result.get("train_days_per_regime", {}).items()}
    )

    strategies = [
        ("hard_bo",  fold_result["metrics_hard"]),
        ("top3_bo",  fold_result["metrics_blend"]),
        ("ew",       fold_result["metrics_ew"]),
    ]
    rows = []
    for strat, m in strategies:
        rows.append({
            "fold":                  fold_id,
            "test_year":             test_year,
            "strategy":              strat,
            "sharpe":                m.get("sharpe",              float("nan")),
            "sortino":               m.get("sortino",             float("nan")),
            "max_dd":                m.get("max_drawdown",        float("nan")),
            "turnover":              m.get("avg_daily_turnover",  float("nan")),
            "train_days_per_regime": train_days_json,
        })

    csv_path = os.path.join(OUT_DIR, f"fold_{fold_id:02d}_result.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(csv_path, "rb+") as f:
        f.flush()
        os.fsync(f.fileno())
    print(f"  [persist] Saved: {csv_path}", flush=True)

    json_path = os.path.join(OUT_DIR, f"fold_{fold_id:02d}_best_configs.json")
    payload = {str(k): v for k, v in fold_result["best_configs"].items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"  [persist] Saved: {json_path}", flush=True)


# ---------------------------------------------------------------------------
# Comparison table: 13b-v2 vs 13b vs 13a vs 13c vs EW
# ---------------------------------------------------------------------------

def _load_sharpes(path: str, col: str) -> dict:
    """Load {fold: sharpe} from a summary CSV."""
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        if "fold" not in df.columns or col not in df.columns:
            return {}
        return {int(r["fold"]): float(r[col]) for _, r in df.iterrows()}
    except Exception:
        return {}


def print_comparison_table(fold_results: list, config: dict) -> None:
    valid = [fr for fr in fold_results if fr is not None]
    if not valid:
        return

    p13a   = _load_sharpes(config["plan13a_summary"],  "hier_sharpe")
    p13b   = _load_sharpes(config["plan13b_summary"],  "hard_sharpe")
    p13c   = _load_sharpes(config["plan13c_summary"],  "hybrid_sharpe")

    header = (
        f"{'Fold':>5}  {'Year':>5}  "
        f"{'13b-v2':>8}  {'Top3-v2':>8}  "
        f"{'13b':>8}  {'13c':>8}  {'13a':>8}  {'EW':>8}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * 80, flush=True)
    print("COMPARISON TABLE: PLAN 13b-v2 vs PRIOR PLANS", flush=True)
    print("=" * 80, flush=True)
    print(header, flush=True)
    print(sep, flush=True)

    def _fmt(v):
        return f"{v:+8.4f}" if (v is not None and not np.isnan(float(v))) else "      na"

    all_v2, all_blend, all_13a, all_13b, all_13c, all_ew = [], [], [], [], [], []

    for fr in valid:
        fid  = fr["fold_id"]
        year = fr["fold_spec"]["test_start"][:4]
        v2   = fr["metrics_hard"]["sharpe"]
        bl   = fr["metrics_blend"]["sharpe"]
        ew   = fr["metrics_ew"]["sharpe"]
        a    = p13a.get(fid, float("nan"))
        b    = p13b.get(fid, float("nan"))
        c    = p13c.get(fid, float("nan"))

        all_v2.append(v2);  all_blend.append(bl)
        all_ew.append(ew);  all_13a.append(a)
        all_13b.append(b);  all_13c.append(c)

        print(
            f"{fid:>5}  {year:>5}  "
            f"{_fmt(v2)}  {_fmt(bl)}  "
            f"{_fmt(b)}  {_fmt(c)}  {_fmt(a)}  {_fmt(ew)}",
            flush=True,
        )

    print(sep, flush=True)

    def _avg(lst):
        fs = [x for x in lst if x is not None and not np.isnan(float(x))]
        return float(np.mean(fs)) if fs else float("nan")

    print(
        f"{'AVG':>5}  {'':>5}  "
        f"{_fmt(_avg(all_v2))}  {_fmt(_avg(all_blend))}  "
        f"{_fmt(_avg(all_13b))}  {_fmt(_avg(all_13c))}  "
        f"{_fmt(_avg(all_13a))}  {_fmt(_avg(all_ew))}",
        flush=True,
    )
    print("=" * 80, flush=True)

    # Win counts
    v2_beats_ew   = sum(1 for s in all_v2 if not np.isnan(s) and s > 0)
    v2_beats_13b  = sum(1 for s, b in zip(all_v2, all_13b)
                        if not np.isnan(s) and not np.isnan(b) and s > b)
    n = len(valid)
    print(f"\nDiagnostics ({n} folds):", flush=True)
    print(f"  Hard BO > EW (Sharpe > 0) : {v2_beats_ew}/{n}", flush=True)
    print(f"  Hard BO > 13b (grid)       : {v2_beats_13b}/{n}", flush=True)
    print("=" * 80, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(config: dict = CONFIG, target_fold: int | None = None) -> list:
    np.random.seed(config["seed"])

    if target_fold is not None:
        print("\n" + "=" * 72, flush=True)
        print(f"PLAN 13b-v2: TRUE BAYESIAN OPTIMIZATION — SINGLE FOLD {target_fold}", flush=True)
    else:
        print("\n" + "=" * 72, flush=True)
        print("PLAN 13b-v2: TRUE BAYESIAN OPTIMIZATION — FULL 12-FOLD RUN", flush=True)
    print("=" * 72, flush=True)
    print(f"  n_trials={config['n_trials']}, kappa={config['kappa']}", flush=True)
    print(f"  warm_start={config.get('warm_start_from_13b', False)}", flush=True)
    print(f"  Output: {OUT_DIR}", flush=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading data ...", flush=True)
    data = load_data_extended()
    prices = data["prices"]
    vix    = data["vix"]
    print(f"  Prices: {prices.shape}  ({prices.index[0].date()} – {prices.index[-1].date()})",
          flush=True)

    returns_raw     = compute_returns(prices)
    asset_features  = compute_asset_features(prices)
    regime_labels   = compute_regime_labels(vix)

    # Align to common dates
    common = (prices.index
              .intersection(returns_raw.index)
              .intersection(regime_labels.index))
    prices          = prices.loc[common]
    returns_raw     = returns_raw.loc[common]
    regime_labels   = regime_labels.loc[common]
    asset_features  = asset_features.loc[asset_features.index.intersection(common)]
    print(f"  Common dates: {len(common)}", flush=True)

    all_dates    = common                                                  # DatetimeIndex
    returns_arr  = returns_raw.values                                      # (T_all, N)
    regime_arr   = regime_labels.reindex(all_dates).fillna(2).astype(int).values  # (T_all,)

    # ── Warm start configs ─────────────────────────────────────────────────────
    warmstart_all = {}
    if config.get("warm_start_from_13b"):
        warmstart_all = load_warmstart_configs(config["plan13b_best_algo_csv"])

    # ── Generate folds ─────────────────────────────────────────────────────────
    wfv = WalkForwardValidator(
        train_years=config["train_years"],
        test_years=config["test_years"],
        step_years=config["step_years"],
        min_test_start=config["min_test_start"],
    )
    all_folds = wfv.generate_folds(data_end=config["data_end"])
    print(f"  Generated {len(all_folds)} walk-forward folds", flush=True)

    # ── Run all folds ──────────────────────────────────────────────────────────
    fold_results = []
    t_exp_start = time.time()

    for fold_spec in all_folds:
        fold_id = fold_spec["fold"]

        # Single-fold mode: skip everything except the requested fold
        if target_fold is not None and fold_id != target_fold:
            continue

        # Skip-if-exists: don't re-run a fold whose results are already on disk
        if is_fold_complete(fold_id):
            print(
                f"\n  Fold {fold_id} already complete (fold_{fold_id:02d}_result.csv"
                f" exists) — skipping.",
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
            config=config,
            warmstart_all=warmstart_all,
        )
        t_fold_end = time.time()
        fold_min = (t_fold_end - t_fold_start) / 60
        print(f"  Fold {fold_id} completed in {fold_min:.1f} min", flush=True)

        if result is not None:
            save_fold_result(result, config)

        fold_results.append(result)

    t_exp_end = time.time()
    total_min = (t_exp_end - t_exp_start) / 60
    print(f"\nTotal experiment time: {total_min:.1f} min ({total_min/60:.1f} h)", flush=True)

    # ── Summary and save ───────────────────────────────────────────────────────
    print_comparison_table(fold_results, config)
    save_results(fold_results, config)

    print("\nPLAN 13b-v2 COMPLETE.", flush=True)
    return fold_results


# ---------------------------------------------------------------------------
# Recover fold results from saved Optuna .pkl studies
# ---------------------------------------------------------------------------

def recover_fold_from_pkl(fold_id: int, config: dict = CONFIG) -> None:
    """
    Reconstruct per-fold test results from existing Optuna .pkl studies.

    Use this when a fold's BO has already completed and studies are saved,
    but the test-weight computation crashed before writing fold_NN_result.csv.

    Example:
        python -m experiments.plan13b_v2_true_bo --recover-fold 1
    """
    print(f"\n{'=' * 72}", flush=True)
    print(f"RECOVER FOLD {fold_id}: loading studies from {STUDIES_DIR}", flush=True)
    print(f"{'=' * 72}", flush=True)

    # Verify all 4 .pkl files are present
    for regime_id in [1, 2, 3, 4]:
        pkl_path = os.path.join(STUDIES_DIR, f"fold_{fold_id:02d}_regime_{regime_id}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"Cannot recover fold {fold_id}: missing {pkl_path}"
            )
    print("  All 4 .pkl study files found.", flush=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("  Loading data ...", flush=True)
    data           = load_data_extended()
    prices         = data["prices"]
    vix            = data["vix"]
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

    all_dates   = common
    returns_arr = returns_raw.values
    regime_arr  = regime_labels.reindex(all_dates).fillna(2).astype(int).values

    # ── Fold spec ──────────────────────────────────────────────────────────────
    wfv = WalkForwardValidator(
        train_years=config["train_years"],
        test_years=config["test_years"],
        step_years=config["step_years"],
        min_test_start=config["min_test_start"],
    )
    fold_specs = {f["fold"]: f for f in wfv.generate_folds(data_end=config["data_end"])}
    if fold_id not in fold_specs:
        raise ValueError(f"Fold {fold_id} not found in walk-forward folds")
    fold_spec = fold_specs[fold_id]

    train_start = fold_spec["train_start"]
    train_end   = fold_spec["train_end"]
    test_start  = fold_spec["test_start"]
    test_end    = fold_spec["test_end"]
    kappa       = config["kappa"]

    # ── Training matrices ──────────────────────────────────────────────────────
    X_train, Y_train = build_training_matrix(
        asset_features, returns_raw, train_start, train_end
    )
    feat_mask    = (asset_features.index >= train_start) & (asset_features.index <= train_end)
    ret_mask     = (returns_raw.index >= train_start) & (returns_raw.index <= train_end)
    common_train = (
        asset_features.loc[feat_mask].index
        .intersection(returns_raw.loc[ret_mask].index)
    )
    regime_train = regime_labels.reindex(common_train).fillna(2).astype(int)

    # ── Test indices ───────────────────────────────────────────────────────────
    test_mask      = (all_dates >= test_start) & (all_dates <= test_end)
    test_indices   = np.where(test_mask)[0]
    test_dates_arr = all_dates[test_indices]
    print(
        f"  Test: {len(test_indices)} days"
        f" ({all_dates[test_indices[0]].date()} – {all_dates[test_indices[-1]].date()})",
        flush=True,
    )

    # ── Extract configs from .pkl studies ─────────────────────────────────────
    best_configs       = {}
    top3_configs       = {}
    regime_train_days  = {}

    for regime_id in [1, 2, 3, 4]:
        pkl_path = os.path.join(STUDIES_DIR, f"fold_{fold_id:02d}_regime_{regime_id}.pkl")
        with open(pkl_path, "rb") as f:
            study = pickle.load(f)

        # Record training days for this regime
        regime_mask_train = (regime_train.values == regime_id)
        regime_train_days[regime_id] = int(regime_mask_train.sum())

        best_trial  = study.best_trial
        best_config = best_trial.user_attrs.get("config", best_trial.params.copy())

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        completed_sorted = sorted(completed, key=lambda t: t.value, reverse=True)
        top3 = completed_sorted[:min(config["top_n"], len(completed_sorted))]
        top3_cfgs = [t.user_attrs.get("config", t.params.copy()) for t in top3]

        best_configs[regime_id] = best_config
        top3_configs[regime_id] = top3_cfgs

        print(
            f"  Regime {regime_id}: best Sharpe={study.best_value:+.4f}"
            f"  family={best_config.get('family', '?')}  "
            f"({len(completed)} completed trials)",
            flush=True,
        )
        del study

    gc.collect()

    # ── Compute test weights ───────────────────────────────────────────────────
    print("  Computing test weights ...", flush=True)
    t0 = time.time()

    best_weights_per_regime = {}
    top3_weights_per_regime = {}

    for regime_id in [1, 2, 3, 4]:
        regime_test_mask  = (regime_arr[test_indices] == regime_id)
        regime_test_dates = test_dates_arr[regime_test_mask]

        if len(regime_test_dates) == 0:
            best_weights_per_regime[regime_id] = np.zeros((0, N))
            top3_weights_per_regime[regime_id] = []
            continue

        w_hard = compute_weights_for_config(
            config=best_configs[regime_id],
            X_train=X_train, Y_train=Y_train,
            asset_features=asset_features,
            test_dates=regime_test_dates,
            prices=prices, N=N,
        )
        best_weights_per_regime[regime_id] = w_hard

        w_top3_list = []
        for cfg in top3_configs[regime_id]:
            w_k = compute_weights_for_config(
                config=cfg,
                X_train=X_train, Y_train=Y_train,
                asset_features=asset_features,
                test_dates=regime_test_dates,
                prices=prices, N=N,
            )
            w_top3_list.append(w_k)
        top3_weights_per_regime[regime_id] = w_top3_list

    t1 = time.time()
    print(f"  Test weights computed in {t1 - t0:.1f}s", flush=True)

    # ── Evaluate strategies ────────────────────────────────────────────────────
    m_hard,  _ = evaluate_hard_bo(
        best_weights_per_regime, returns_arr, regime_arr, test_indices, kappa, N
    )
    m_blend, _ = evaluate_top3_blend(
        top3_weights_per_regime, returns_arr, regime_arr, test_indices, kappa, N
    )
    m_ew,    _ = evaluate_ew(returns_arr, test_indices, kappa, N)

    print(
        f"\n  Results:"
        f"\n    Hard BO  : Sharpe={m_hard['sharpe']:+.4f}"
        f"\n    Top-3 BO : Sharpe={m_blend['sharpe']:+.4f}"
        f"\n    EW       : Sharpe={m_ew['sharpe']:+.4f}",
        flush=True,
    )

    # ── Persist ────────────────────────────────────────────────────────────────
    fold_result = {
        "fold_id":               fold_id,
        "fold_spec":             fold_spec,
        "metrics_hard":          m_hard,
        "metrics_blend":         m_blend,
        "metrics_ew":            m_ew,
        "best_configs":          best_configs,
        "train_days_per_regime": regime_train_days,
    }
    save_fold_result(fold_result, config)
    print(f"\nFold {fold_id} recovery complete.", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plan 13b-v2: True Bayesian Optimization"
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Run only this specific fold (1–12). Omit to run all folds.",
    )
    parser.add_argument(
        "--recover-fold", type=int, default=None, dest="recover_fold",
        help="Recover test results from saved .pkl studies for this fold (no BO).",
    )
    args = parser.parse_args()

    if args.recover_fold is not None:
        recover_fold_from_pkl(args.recover_fold)
    else:
        run_experiment(target_fold=args.fold)
