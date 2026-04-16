# experiments/plan13b_v3_bo_val_split.py -- Plan 13b-v3: BO with Inner Val-Split
#
# Addresses Plan 13b-v2's severe train-test divergence (train +1.55, test -0.23)
# by introducing a chronological 80/20 inner train/validation split per regime.
# Optuna TPE now optimises on VAL Sharpe (not train Sharpe), preventing the
# in-sample exploitation that caused v2's overfitting.
#
# Key changes vs. Plan 13b-v2:
#   - Inner chronological 80/20 split: BO objective = VAL Sharpe
#   - Refit best config on full regime training data after BO
#   - Trials: 200 -> 100 + MedianPruner
#   - Min-regime-size fallback (<100 days -> EqualWeight)
#   - Per-fold trial_log.csv with train_sharpe, val_sharpe, train_val_gap
#
# Usage:
#   cd Implementierung1
#   python -u -m experiments.plan13b_v3_bo_val_split --fold 1
#   python -u -m experiments.plan13b_v3_bo_val_split --fold 1 --smoke
#   python -u -m experiments.plan13b_v3_bo_val_split --skip-existing
#   python -u -m experiments.plan13b_v3_bo_val_split            # all 12 folds

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
    # BO settings
    "n_trials": 100,             # reduced from 200 (v2); TPE converges in 50-100
    "seed": RANDOM_SEED,
    "val_frac": 0.2,             # inner validation fraction (chronological)
    "min_regime_days": 100,      # below this: skip BO, use EW fallback

    # Costs
    "kappa": KAPPA,

    # Walk-forward (identical to Plans 13a/13b/13b-v2/13c)
    "train_years": 8,
    "test_years": 1,
    "step_years": 1,
    "min_test_start": "2013-01-01",
    "data_end": "2024-12-31",

    # Tiers
    "tiers": [1, 2, 3],

    # Top-N for blend strategy (top by val Sharpe)
    "top_n": 3,

    # Warm start from Plan 13b
    "warm_start_from_13b": True,
    "plan13b_best_algo_csv": os.path.join(
        RESULTS_DIR, "plan13b_bayesian_opt", "best_algo_per_regime.csv"
    ),

    # Prior results for comparison table
    "plan13b_summary": os.path.join(RESULTS_DIR, "plan13b_bayesian_opt", "summary_metrics.csv"),
    "plan13b_v2_summary": os.path.join(RESULTS_DIR, "plan13b_v2_true_bo", "summary_metrics.csv"),

    # Output
    "output_dir": os.path.join(RESULTS_DIR, "plan13b_v3"),
}

OUT_DIR     = CONFIG["output_dir"]
STUDIES_DIR = os.path.join(OUT_DIR, "optuna_studies")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(STUDIES_DIR, exist_ok=True)

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

    Returns float('-inf') for degenerate inputs (too few days, zero variance)
    so that Optuna TPE treats these as genuinely bad trials rather than
    break-even results.
    """
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
# Inner validation split
# ---------------------------------------------------------------------------

def split_regime_chronologically(X_regime, y_regime, val_frac=0.2):
    """Split regime-filtered data into inner train (first 80%) and val (last 20%).

    CRITICAL: split chronologically, NOT randomly, to prevent temporal leakage.
    """
    n = len(X_regime)
    split_idx = int(n * (1 - val_frac))
    X_train = X_regime[:split_idx]
    y_train = y_regime[:split_idx]
    X_val   = X_regime[split_idx:]
    y_val   = y_regime[split_idx:]
    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Continuous search space (TPE-compatible) — identical to v2
# ---------------------------------------------------------------------------

def sample_tier1(trial: optuna.Trial) -> dict:
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
# Algorithm factory — identical to v2
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
# Warm start: parse Plan 13b best configs — identical to v2
# ---------------------------------------------------------------------------

def _algo_name_to_optuna_params(algo_name: str):
    """Convert a Plan 13b algorithm name to Optuna trial params for warm start."""
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
            parts = algo_name.split("_")
            L = int(parts[1][1:])
            G = float(parts[2][1:])
            return {"tier": 1, "t1_family": "MeanVariance", "mv_lookback": L, "mv_gamma": G}
        if algo_name.startswith("Momentum_L"):
            parts = algo_name.split("_")
            L = int(parts[1][1:])
            w = parts[2]
            return {"tier": 1, "t1_family": "Momentum", "mom_lookback": L, "mom_type": w}
        if algo_name.startswith("Trend_L"):
            parts = algo_name.split("_")
            L = int(parts[1][1:])
            beta = int(parts[2][1:]) if len(parts) > 2 else 1
            beta = min(max(beta, 1), 5)
            return {"tier": 1, "t1_family": "TrendFollowing",
                    "trend_lookback": L, "trend_beta": beta}
        if algo_name.startswith("Ridge_L"):
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
            parts = algo_name.split("_")
            L = int(parts[1][1:])
            a = float(parts[2][1:])
            r = float(parts[3][1:])
            return {"tier": 2, "t2_family": "ElasticNet",
                    "lookback": L, "enet_alpha": a, "enet_l1_ratio": r}
        if algo_name.startswith("RF_n"):
            parts = algo_name.split("_")
            n = int(parts[1][1:])
            d_str = parts[2][1:]
            d = int(d_str) if d_str != "None" else 15
            d = min(max(d, 3), 20)
            n = min(max(n, 50), 300)
            L = int(parts[3][1:])
            return {"tier": 3, "t3_family": "RandomForest", "lookback": L,
                    "rf_n_estimators": n, "rf_max_depth": d, "rf_min_leaf": 1}
        if algo_name.startswith("GBM_n"):
            parts = algo_name.split("_")
            n = int(parts[1][1:])
            d = int(parts[2][1:])
            lr = float(parts[3][2:])
            L = int(parts[4][1:])
            n = min(max(n, 50), 300)
            d = min(max(d, 2), 10)
            return {"tier": 3, "t3_family": "GradientBoosting", "lookback": L,
                    "gbm_n_estimators": n, "gbm_max_depth": d,
                    "gbm_lr": lr, "gbm_subsample": 0.8}
        if algo_name.startswith("MLP_h"):
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
    """Load Plan 13b's best algo per regime per fold. Returns {(fold_id, regime_id): optuna_params}"""
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


def _softmax_weights_from_mu(mu_all: np.ndarray) -> np.ndarray:
    """Convert (T, N) expected-return matrix to (T, N) softmax weights."""
    return np.array([_softmax(mu_all[i]) for i in range(len(mu_all))])


# ---------------------------------------------------------------------------
# Core: compute Sharpe for a config on a given (X, y, dates, prices) block
# ---------------------------------------------------------------------------

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
    Fit config on (X_fit, Y_fit), evaluate weights on (X_eval, eval_dates).
    Returns Sharpe on the eval block (net of kappa).

    X_eval rows correspond to eval_dates 1:1.

    NaN handling for Tier 2/3:
        X_eval may contain NaN rows at the start of a training window (lookback
        warm-up period). GBM, MLP, Ridge, Lasso, ElasticNet all raise ValueError
        on NaN input to predict(), whereas RandomForest silently tolerates it.
        We filter out NaN rows from X_eval and y_eval before prediction so that
        all families are treated consistently. The Sharpe is then computed on the
        clean subset (days with valid features).
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

        # Filter NaN rows from eval data before transforming/predicting.
        # NaN rows arise at the start of training windows (lookback warm-up).
        # Without this filter: GBM/MLP/Ridge raise ValueError → silently return 0.0.
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
            if (hasattr(algo, "_models") and isinstance(algo._models, dict) and algo._models):
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
        # Tier 1: compute weights day by day using price history
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
# Val-based Optuna objective (KEY CHANGE vs v2)
# ---------------------------------------------------------------------------

def make_objective_with_val(
    X_train_inner: np.ndarray,
    Y_train_inner: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    val_dates: pd.DatetimeIndex,
    # For train-sharpe diagnostic (uses inner train dates)
    X_train_full: np.ndarray,          # full regime (inner train + val)
    Y_train_full: np.ndarray,
    train_inner_dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
    N: int,
    kappa: float,
    trial_log: list,
    regime_id: int,
):
    """
    Create the Optuna objective that optimises on VAL Sharpe (not train Sharpe).

    Procedure per trial:
      1. Sample config
      2. Fit on X_train_inner / Y_train_inner
      3. Evaluate on X_val / Y_val  → val_sharpe (objective)
      4. Evaluate on X_train_inner  → train_sharpe (diagnostic only)
      5. Record both in trial.user_attrs and trial_log
    """

    def objective(trial: optuna.Trial) -> float:
        config = sample_algorithm(trial)
        try:
            trial.set_user_attr("config", config)
        except Exception:
            pass

        # ── Val Sharpe (objective) ────────────────────────────────────────
        val_sharpe = _eval_config_on_block(
            config=config,
            X_fit=X_train_inner,
            Y_fit=Y_train_inner,
            X_eval=X_val,
            y_eval=Y_val,
            eval_dates=val_dates,
            prices=prices,
            N=N,
            kappa=kappa,
        )

        # ── Train Sharpe (diagnostic, NOT the objective) ──────────────────
        train_sharpe = _eval_config_on_block(
            config=config,
            X_fit=X_train_inner,
            Y_fit=Y_train_inner,
            X_eval=X_train_inner,
            y_eval=Y_train_inner,
            eval_dates=train_inner_dates,
            prices=prices,
            N=N,
            kappa=kappa,
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
            "trial_num":    trial.number,
            "family":       config.get("family", "?"),
            "tier":         config.get("tier", "?"),
            "train_sharpe": round(float(train_sharpe), 6),
            "val_sharpe":   round(float(val_sharpe),   6),
            "gap":          round(float(gap),          6),
            "pruned":       False,
        })

        return val_sharpe  # <-- KEY: val, not train

    return objective


# ---------------------------------------------------------------------------
# BO study runner (per fold × regime) — v3 variant with val-split
# ---------------------------------------------------------------------------

def run_bo_for_regime_v3(
    fold_id: int,
    regime_id: int,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_regime: np.ndarray,          # full regime training block
    Y_regime: np.ndarray,
    regime_dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
    N: int,
    kappa: float,
    n_trials: int,
    seed: int,
    val_frac: float,
    warmstart_params: dict | None,
    trial_log: list,
) -> optuna.Study:
    """
    Run one Optuna TPE study for a single regime in a single fold.

    Inner 80/20 chronological split:
      - BO objective = val Sharpe (last 20% of regime days)
      - train Sharpe (first 80%) recorded as diagnostic only
    """
    # ── Inner val split ────────────────────────────────────────────────────
    X_inner_train, Y_inner_train, X_inner_val, Y_inner_val = (
        split_regime_chronologically(X_regime, Y_regime, val_frac=val_frac)
    )

    n_total = len(regime_dates)
    split_idx = int(n_total * (1 - val_frac))
    train_inner_dates = regime_dates[:split_idx]
    val_dates         = regime_dates[split_idx:]

    print(
        f"    inner-train: {len(train_inner_dates)} days  "
        f"| inner-val: {len(val_dates)} days",
        end="  ", flush=True,
    )

    if len(val_dates) < 5:
        print("→ val block too small, using EW", flush=True)
        return None  # caller will fall back to EW

    # ── Optuna study ───────────────────────────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0)
    study   = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    # Warm start: enqueue the Plan 13b best config as the first trial
    if warmstart_params is not None:
        try:
            study.enqueue_trial(warmstart_params)
        except Exception as e:
            print(f"    [warm start] Failed to enqueue: {e}", flush=True)

    objective = make_objective_with_val(
        X_train_inner=X_inner_train,
        Y_train_inner=Y_inner_train,
        X_val=X_inner_val,
        Y_val=Y_inner_val,
        val_dates=val_dates,
        X_train_full=X_regime,
        Y_train_full=Y_regime,
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
# Compute weights for a config on test dates (using full training data for refit)
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
    Given a HP config, build and fit the algorithm on full training data,
    then compute weights for each test date.

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
# Test-time evaluation helpers — identical to v2
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
    regime_test_idx = {}
    for regime_id in [1, 2, 3, 4]:
        regime_test_idx[regime_id] = [j for j, i in enumerate(test_indices)
                                       if regime_arr[i] == regime_id]

    weights_all = np.zeros((len(test_indices), N))
    for regime_id in [1, 2, 3, 4]:
        idxs = regime_test_idx[regime_id]
        if not idxs or regime_id not in best_weights_per_regime:
            for j in idxs:
                weights_all[j] = np.ones(N) / N
        else:
            w_reg = best_weights_per_regime[regime_id]
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
    """Top-3 BO: blend weights from top-3 trials by val Sharpe per regime."""
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
            w_list = top3_weights_per_regime[regime_id]
            if not w_list:
                for j in idxs:
                    weights_all[j] = np.ones(N) / N
            else:
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
    n_trials        = config["n_trials"]
    val_frac        = config["val_frac"]
    min_regime_days = config["min_regime_days"]

    # ── Build training matrices ───────────────────────────────────────────────
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
    regime_train = regime_labels.reindex(common_train).fillna(2).astype(int)
    ret_train_df = returns.reindex(common_train).fillna(0.0)

    print(
        f"  X_train: {X_train.shape}  |  "
        f"Train dates: {len(common_train)}"
        f"  ({common_train[0].date()} - {common_train[-1].date()})",
        flush=True,
    )

    # ── Test indices ──────────────────────────────────────────────────────────
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

    # ── Run BO per regime ─────────────────────────────────────────────────────
    bo_studies     = {}
    best_configs   = {}
    top3_configs   = {}
    regime_stats   = {}  # for JSON: {regime_id: {family, val_sharpe, train_sharpe, gap, fallback}}
    trial_log_all  = []  # rows for fold_NN_trial_log.csv

    for regime_id in [1, 2, 3, 4]:
        regime_name = REGIME_NAMES[regime_id]
        t_reg_start = time.time()

        regime_mask_train = (regime_train.values == regime_id)
        regime_dates      = common_train[regime_mask_train]
        X_regime          = X_train[regime_mask_train]
        Y_regime          = ret_train_df.values[regime_mask_train]

        n_regime_days = int(regime_mask_train.sum())
        print(
            f"\n  Regime {regime_id} ({regime_name:7s}): "
            f"{n_regime_days:4d} training days",
            flush=True,
        )

        # ── Min-regime-size fallback ──────────────────────────────────────
        if n_regime_days < min_regime_days:
            print(
                f"    -> only {n_regime_days} days < {min_regime_days} "
                f"(min_regime_days) — using EqualWeight fallback",
                flush=True,
            )
            ew_cfg = {"tier": 1, "family": "EqualWeight"}
            best_configs[regime_id] = ew_cfg
            top3_configs[regime_id] = [ew_cfg] * 3
            bo_studies[regime_id]   = None
            regime_stats[regime_id] = {
                "family": "EqualWeight", "tier": 1,
                "val_sharpe": float("nan"), "train_sharpe": float("nan"),
                "train_val_gap": float("nan"), "fallback": True,
                "n_regime_days": n_regime_days,
            }
            continue

        # ── Warm start ─────────────────────────────────────────────────────
        warmstart_params = None
        if config.get("warm_start_from_13b"):
            warmstart_params = warmstart_all.get((fold_id, regime_id))

        # ── Run BO with val-split ──────────────────────────────────────────
        trial_log_regime = []
        study = run_bo_for_regime_v3(
            fold_id=fold_id,
            regime_id=regime_id,
            X_train=X_train,
            Y_train=Y_train,
            X_regime=X_regime,
            Y_regime=Y_regime,
            regime_dates=regime_dates,
            prices=prices,
            N=N,
            kappa=kappa,
            n_trials=n_trials,
            seed=config["seed"] + fold_id * 10 + regime_id,
            val_frac=val_frac,
            warmstart_params=warmstart_params,
            trial_log=trial_log_regime,
        )

        trial_log_all.extend(trial_log_regime)

        if study is None:
            # val block too small fallback
            ew_cfg = {"tier": 1, "family": "EqualWeight"}
            best_configs[regime_id] = ew_cfg
            top3_configs[regime_id] = [ew_cfg] * 3
            bo_studies[regime_id]   = None
            regime_stats[regime_id] = {
                "family": "EqualWeight", "tier": 1,
                "val_sharpe": float("nan"), "train_sharpe": float("nan"),
                "train_val_gap": float("nan"), "fallback": True,
                "n_regime_days": n_regime_days,
            }
            continue

        bo_studies[regime_id] = study

        best_trial  = study.best_trial
        best_val    = study.best_value
        best_config = best_trial.user_attrs.get("config", best_trial.params.copy())

        # Train/val gap for best trial
        best_train_sh = best_trial.user_attrs.get("train_sharpe", float("nan"))
        best_val_sh   = best_trial.user_attrs.get("val_sharpe",   best_val)
        best_gap      = best_trial.user_attrs.get("train_val_gap", best_train_sh - best_val_sh)

        # Top-3 by val Sharpe
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE
                     and t.value is not None]
        completed_sorted = sorted(completed, key=lambda t: t.value, reverse=True)
        top3 = completed_sorted[:min(config["top_n"], len(completed_sorted))]
        top3_cfgs = [t.user_attrs.get("config", t.params.copy()) for t in top3]

        best_configs[regime_id] = best_config
        top3_configs[regime_id] = top3_cfgs

        regime_stats[regime_id] = {
            "family":        best_config.get("family", "?"),
            "tier":          best_config.get("tier", "?"),
            "config":        best_config,
            "val_sharpe":    round(float(best_val_sh),  4),
            "train_sharpe":  round(float(best_train_sh), 4),
            "train_val_gap": round(float(best_gap),     4),
            "fallback":      False,
            "n_regime_days": n_regime_days,
            "n_completed_trials": len(completed),
        }

        t_reg_end = time.time()
        print(
            f"    -> best val_sharpe={best_val:+.4f}  "
            f"train_sharpe={best_train_sh:+.4f}  "
            f"gap={best_gap:+.4f}  "
            f"family={best_config.get('family', '?')}  "
            f"({len(completed)} trials in {t_reg_end - t_reg_start:.0f}s)",
            flush=True,
        )

    # ── Memory cleanup ────────────────────────────────────────────────────────
    del bo_studies
    gc.collect()

    # ── Generate test-time weights (refit on full regime training data) ───────
    print("\n  Computing test weights (refit on full training data) ...", flush=True)
    t2 = time.time()

    best_weights_per_regime = {}
    top3_weights_per_regime = {}

    for regime_id in [1, 2, 3, 4]:
        regime_test_mask  = (regime_arr[test_indices] == regime_id)
        regime_test_dates = test_dates_arr[regime_test_mask]

        if len(regime_test_dates) == 0:
            best_weights_per_regime[regime_id] = np.zeros((0, N))
            top3_weights_per_regime[regime_id] = []
            continue

        # Hard BO: refit best config on full X_train (the whole fold training set)
        # This is the standard practice: refit on all available data after BO selects config
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

        # Top-3 weights
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
    m_hard,  net_hard  = evaluate_hard_bo(
        best_weights_per_regime, returns_arr, regime_arr, test_indices, kappa, N
    )
    m_blend, net_blend = evaluate_top3_blend(
        top3_weights_per_regime, returns_arr, regime_arr, test_indices, kappa, N
    )
    m_ew,    net_ew    = evaluate_ew(returns_arr, test_indices, kappa, N)

    print(
        f"\n  Results:"
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
        "regime_stats":          regime_stats,
        "trial_log":             trial_log_all,
        "test_dates":            test_dates_arr,
    }


# ---------------------------------------------------------------------------
# Per-fold persistence
# ---------------------------------------------------------------------------

def is_fold_complete(fold_id: int) -> bool:
    """Return True if fold_NN_result.csv exists and has >= 3 strategy rows."""
    csv_path = os.path.join(OUT_DIR, f"fold_{fold_id:02d}_result.csv")
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        return len(df) >= 3
    except Exception:
        return False


def _fsync_csv(path: str, df: pd.DataFrame) -> None:
    """Write DataFrame to CSV and fsync."""
    df.to_csv(path, index=False)
    with open(path, "rb+") as f:
        f.flush()
        os.fsync(f.fileno())


def _fsync_json(path: str, payload: dict) -> None:
    """Write JSON and fsync."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def save_fold_result(fold_result: dict, config: dict) -> None:
    """
    Persist per-fold artifacts immediately after a fold completes.

    Writes:
      fold_NN_result.csv         — 3 rows (hard_bo, top3_bo, ew)
      fold_NN_best_configs.json  — best config per regime with val/train Sharpes
      fold_NN_trial_log.csv      — all trials: regime, trial_num, family,
                                   train_sharpe, val_sharpe, gap, pruned
    """
    fold_id   = fold_result["fold_id"]
    fold_spec = fold_result["fold_spec"]
    test_year = fold_spec["test_start"][:4]

    # ── fold_NN_result.csv ────────────────────────────────────────────────────
    strategies = [
        ("hard_bo",  fold_result["metrics_hard"]),
        ("top3_bo",  fold_result["metrics_blend"]),
        ("ew",       fold_result["metrics_ew"]),
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

    # ── fold_NN_best_configs.json ─────────────────────────────────────────────
    json_path = os.path.join(OUT_DIR, f"fold_{fold_id:02d}_best_configs.json")
    _fsync_json(json_path, {
        str(k): v for k, v in fold_result.get("regime_stats", {}).items()
    })
    print(f"  [persist] Saved: {json_path}", flush=True)

    # ── fold_NN_trial_log.csv ─────────────────────────────────────────────────
    trial_log = fold_result.get("trial_log", [])
    if trial_log:
        log_path = os.path.join(OUT_DIR, f"fold_{fold_id:02d}_trial_log.csv")
        _fsync_csv(log_path, pd.DataFrame(trial_log))
        print(f"  [persist] Saved: {log_path}  ({len(trial_log)} trial rows)", flush=True)


# ---------------------------------------------------------------------------
# Comparison table
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

    p13b   = _load_sharpes(config["plan13b_summary"],    "hard_sharpe")
    p13bv2 = _load_sharpes(config["plan13b_v2_summary"], "hard_sharpe")

    header = (
        f"{'Fold':>5}  {'Year':>5}  "
        f"{'Hard-v3':>9}  {'Top3-v3':>9}  "
        f"{'EW':>8}  {'13b':>8}  {'13b-v2':>9}  "
        f"{'AvgGap':>8}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * 80, flush=True)
    print("COMPARISON TABLE: PLAN 13b-v3 vs PRIOR PLANS", flush=True)
    print("=" * 80, flush=True)
    print(header, flush=True)
    print(sep, flush=True)

    def _fmt(v):
        try:
            return f"{float(v):+9.4f}" if not np.isnan(float(v)) else "       na"
        except Exception:
            return "       na"

    all_v3, all_blend, all_ew, all_13b, all_v2, all_gap = [], [], [], [], [], []

    for fr in valid:
        fid  = fr["fold_id"]
        year = fr["fold_spec"]["test_start"][:4]
        v3   = fr["metrics_hard"]["sharpe"]
        bl   = fr["metrics_blend"]["sharpe"]
        ew   = fr["metrics_ew"]["sharpe"]
        b    = p13b.get(fid, float("nan"))
        v2   = p13bv2.get(fid, float("nan"))

        # Average train-val gap across regimes for this fold
        regime_stats = fr.get("regime_stats", {})
        gaps = [s["train_val_gap"] for s in regime_stats.values()
                if not s.get("fallback", False) and not np.isnan(float(s.get("train_val_gap", float("nan"))))]
        avg_gap = float(np.mean(gaps)) if gaps else float("nan")

        all_v3.append(v3);  all_blend.append(bl)
        all_ew.append(ew);  all_13b.append(b)
        all_v2.append(v2);  all_gap.append(avg_gap)

        print(
            f"{fid:>5}  {year:>5}  "
            f"{_fmt(v3)}  {_fmt(bl)}  "
            f"{_fmt(ew)}  {_fmt(b)}  {_fmt(v2)}  "
            f"{_fmt(avg_gap)}",
            flush=True,
        )

    print(sep, flush=True)

    def _avg(lst):
        fs = [x for x in lst if x is not None and not np.isnan(float(x))]
        return float(np.mean(fs)) if fs else float("nan")

    print(
        f"{'AVG':>5}  {'':>5}  "
        f"{_fmt(_avg(all_v3))}  {_fmt(_avg(all_blend))}  "
        f"{_fmt(_avg(all_ew))}  {_fmt(_avg(all_13b))}  "
        f"{_fmt(_avg(all_v2))}  "
        f"{_fmt(_avg(all_gap))}",
        flush=True,
    )
    print("=" * 80, flush=True)

    n = len(valid)
    v3_beats_ew  = sum(1 for s in all_v3 if not np.isnan(s) and s > 0)
    v3_beats_13b = sum(1 for s, b in zip(all_v3, all_13b)
                       if not np.isnan(s) and not np.isnan(b) and s > b)
    print(f"\nDiagnostics ({n} folds):", flush=True)
    print(f"  Hard v3 > EW (Sharpe > 0) : {v3_beats_ew}/{n}", flush=True)
    print(f"  Hard v3 > 13b (grid)       : {v3_beats_13b}/{n}", flush=True)
    print("=" * 80, flush=True)


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

    if smoke:
        config = dict(config)
        config["n_trials"] = 10
        print("\n  [SMOKE TEST] n_trials overridden to 10", flush=True)

    if target_fold is not None:
        print("\n" + "=" * 72, flush=True)
        print(f"PLAN 13b-v3: BO WITH VAL-SPLIT — SINGLE FOLD {target_fold}", flush=True)
    else:
        print("\n" + "=" * 72, flush=True)
        print("PLAN 13b-v3: BO WITH VAL-SPLIT — FULL 12-FOLD RUN", flush=True)
    print("=" * 72, flush=True)
    print(
        f"  n_trials={config['n_trials']}, val_frac={config['val_frac']}, "
        f"kappa={config['kappa']}",
        flush=True,
    )
    print(f"  min_regime_days={config['min_regime_days']}", flush=True)
    print(f"  warm_start={config.get('warm_start_from_13b', False)}", flush=True)
    print(f"  Output: {OUT_DIR}", flush=True)

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

    # Align to common dates
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

    # ── Run folds ──────────────────────────────────────────────────────────────
    fold_results = []
    t_exp_start  = time.time()

    for fold_spec in all_folds:
        fold_id = fold_spec["fold"]

        if target_fold is not None and fold_id != target_fold:
            continue

        if skip_existing and is_fold_complete(fold_id):
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

    print_comparison_table(fold_results, config)
    print("\nPLAN 13b-v3 COMPLETE.", flush=True)
    return fold_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plan 13b-v3: BO with inner chronological val-split"
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
        help="Smoke-test mode: 10 trials per regime (overrides n_trials).",
    )
    args = parser.parse_args()

    run_experiment(
        target_fold=args.fold,
        skip_existing=args.skip_existing,
        smoke=args.smoke,
    )
