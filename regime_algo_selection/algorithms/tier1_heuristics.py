# algorithms/tier1_heuristics.py — All Tier 1 heuristic portfolio algorithms

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from regime_algo_selection.algorithms.base import PortfolioAlgorithm
from regime_algo_selection.config import (
    N_ASSETS, LOOKBACKS_COV, LOOKBACKS_MOM, LOOKBACKS_TREND,
    TREND_BETAS, RISK_AVERSIONS,
)

# Ridge regularisation added to covariance matrices to avoid singularity
_RIDGE = 1e-6


def _returns_from_prices(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Return the last `lookback` daily returns from a price history."""
    daily_r = prices.pct_change(fill_method=None).iloc[1:]  # drop first NaN row
    return daily_r.iloc[-lookback:]


def _cov_matrix(prices: pd.DataFrame, lookback: int) -> np.ndarray:
    """Compute regularised sample covariance matrix."""
    r = _returns_from_prices(prices, lookback)
    Sigma = r.cov().values
    Sigma += _RIDGE * np.eye(Sigma.shape[0])
    return Sigma


def _vol_vector(prices: pd.DataFrame, lookback: int) -> np.ndarray:
    """Annualised volatility per asset."""
    r = _returns_from_prices(prices, lookback)
    return r.std().values * np.sqrt(252)


# ── F1: Equal Weight ───────────────────────────────────────────────────────────

class EqualWeight(PortfolioAlgorithm):
    def __init__(self):
        super().__init__("EqualWeight", "EqualWeight", {})

    def compute_weights(self, prices_history: pd.DataFrame, features: dict = None) -> np.ndarray:
        n = prices_history.shape[1]
        return self._equal_weight(n)


# ── F2: Minimum Variance ───────────────────────────────────────────────────────

class MinimumVariance(PortfolioAlgorithm):
    def __init__(self, lookback: int):
        super().__init__(f"MinVar_L{lookback}", "MinimumVariance", {"lookback": lookback})
        self.lookback = lookback

    def compute_weights(self, prices_history: pd.DataFrame, features: dict = None) -> np.ndarray:
        n = prices_history.shape[1]
        if len(prices_history) < self.lookback + 1:
            return self._equal_weight(n)

        Sigma = _cov_matrix(prices_history, self.lookback)

        w0 = self._equal_weight(n)
        result = minimize(
            fun=lambda w: w @ Sigma @ w,
            x0=w0,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if result.success:
            return self._safe_normalize(result.x)
        return self._equal_weight(n)


# ── F3: Risk Parity (Inverse Volatility) ──────────────────────────────────────

class RiskParity(PortfolioAlgorithm):
    def __init__(self, lookback: int):
        super().__init__(f"RiskParity_L{lookback}", "RiskParity", {"lookback": lookback})
        self.lookback = lookback

    def compute_weights(self, prices_history: pd.DataFrame, features: dict = None) -> np.ndarray:
        n = prices_history.shape[1]
        if len(prices_history) < self.lookback + 1:
            return self._equal_weight(n)

        vols = _vol_vector(prices_history, self.lookback)
        vols = np.where(vols < 1e-10, 1e-10, vols)  # avoid div-by-zero
        w = 1.0 / vols
        return self._safe_normalize(w)


# ── F4: Maximum Diversification ────────────────────────────────────────────────

class MaxDiversification(PortfolioAlgorithm):
    def __init__(self, lookback: int):
        super().__init__(f"MaxDiv_L{lookback}", "MaxDiversification", {"lookback": lookback})
        self.lookback = lookback

    def compute_weights(self, prices_history: pd.DataFrame, features: dict = None) -> np.ndarray:
        n = prices_history.shape[1]
        if len(prices_history) < self.lookback + 1:
            return self._equal_weight(n)

        Sigma = _cov_matrix(prices_history, self.lookback)
        vols = np.sqrt(np.diag(Sigma))
        vols = np.where(vols < 1e-10, 1e-10, vols)

        def neg_diversification_ratio(w):
            port_vol = np.sqrt(w @ Sigma @ w)
            if port_vol < 1e-12:
                return 0.0
            return -(w @ vols) / port_vol

        w0 = self._equal_weight(n)
        result = minimize(
            fun=neg_diversification_ratio,
            x0=w0,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if result.success:
            return self._safe_normalize(result.x)
        return self._equal_weight(n)


# ── F5: Momentum ───────────────────────────────────────────────────────────────

class Momentum(PortfolioAlgorithm):
    def __init__(self, lookback: int, weighting: str = "linear"):
        super().__init__(
            f"Momentum_L{lookback}_{weighting}",
            "Momentum",
            {"lookback": lookback, "weighting": weighting},
        )
        self.lookback = lookback
        self.weighting = weighting

    def compute_weights(self, prices_history: pd.DataFrame, features: dict = None) -> np.ndarray:
        n = prices_history.shape[1]
        if len(prices_history) < self.lookback + 1:
            return self._equal_weight(n)

        if self.weighting == "linear":
            # Cumulative return over lookback
            scores = (
                prices_history.iloc[-1].values / prices_history.iloc[-self.lookback - 1].values - 1
            )
        else:  # exp / ewm
            daily_r = prices_history.pct_change(fill_method=None).iloc[1:]
            ewm = daily_r.ewm(span=self.lookback).mean()
            scores = ewm.iloc[-1].values

        # Softmax for non-negative weights (numerically stable)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        scores -= scores.max()
        w = np.exp(scores)
        return self._safe_normalize(w)


# ── F6: Trend Following ────────────────────────────────────────────────────────

class TrendFollowing(PortfolioAlgorithm):
    def __init__(self, lookback: int, beta: int):
        super().__init__(
            f"Trend_L{lookback}_B{beta}",
            "TrendFollowing",
            {"lookback": lookback, "beta": beta},
        )
        self.lookback = lookback
        self.beta = beta

    def compute_weights(self, prices_history: pd.DataFrame, features: dict = None) -> np.ndarray:
        n = prices_history.shape[1]
        if len(prices_history) < self.lookback + 1:
            return self._equal_weight(n)

        # Trend signal: price > SMA(lookback)
        sma = prices_history.iloc[-self.lookback:].mean().values
        current = prices_history.iloc[-1].values
        signals = (current > sma).astype(float)  # binary 0/1

        if signals.sum() == 0:
            return self._equal_weight(n)

        raw_w = signals ** self.beta
        return self._safe_normalize(raw_w)


# ── F7: Mean-Variance ─────────────────────────────────────────────────────────

class MeanVariance(PortfolioAlgorithm):
    def __init__(self, lookback: int, risk_aversion: float):
        super().__init__(
            f"MeanVar_L{lookback}_G{risk_aversion}",
            "MeanVariance",
            {"lookback": lookback, "risk_aversion": risk_aversion},
        )
        self.lookback = lookback
        self.risk_aversion = risk_aversion

    def compute_weights(self, prices_history: pd.DataFrame, features: dict = None) -> np.ndarray:
        n = prices_history.shape[1]
        if len(prices_history) < self.lookback + 1:
            return self._equal_weight(n)

        r = _returns_from_prices(prices_history, self.lookback)
        mu = r.mean().values * 252      # annualised mean
        Sigma = r.cov().values * 252    # annualised cov
        Sigma += _RIDGE * np.eye(n)
        gamma = self.risk_aversion

        def neg_utility(w):
            return -(w @ mu - (gamma / 2) * (w @ Sigma @ w))

        w0 = self._equal_weight(n)
        result = minimize(
            fun=neg_utility,
            x0=w0,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if result.success:
            return self._safe_normalize(result.x)
        return self._equal_weight(n)


# ── Algorithm Registry ─────────────────────────────────────────────────────────

def _build_tier1_list() -> list:
    """Build the 48 Tier 1 heuristic algorithms."""
    algos = []
    algos.append(EqualWeight())
    for L in LOOKBACKS_COV:
        algos.append(MinimumVariance(L))
    for L in [20, 60, 120, 252]:
        algos.append(RiskParity(L))
    for L in [20, 60, 120, 252]:
        algos.append(MaxDiversification(L))
    for L in LOOKBACKS_MOM:
        for w in ["linear", "exp"]:
            algos.append(Momentum(L, w))
    for L in LOOKBACKS_TREND:
        for beta in TREND_BETAS:
            algos.append(TrendFollowing(L, beta))
    for L in [20, 60, 120]:
        for gamma in RISK_AVERSIONS:
            algos.append(MeanVariance(L, gamma))
    return algos


def build_tier1_algorithm_space() -> list:
    """
    Instantiate ALL Tier 1 algorithm configurations.

    Expected total: ~48 algorithms
        F1: 1
        F2: 5  (MinVar  x 5 lookbacks)
        F3: 4  (RiskParity x 4 lookbacks)
        F4: 4  (MaxDiv  x 4 lookbacks)
        F5: 10 (Momentum x 5 lookbacks x 2 weighting schemes)
        F6: 12 (Trend x 4 lookbacks x 3 betas)
        F7: 12 (MeanVar x 3 lookbacks x 4 risk aversions)
    """
    algos = _build_tier1_list()
    print(
        f"Built K={len(algos)} Tier 1 algorithms across 7 families "
        f"(F1=1, F2=5, F3=4, F4=4, F5=10, F6=12, F7=12)"
    )
    return algos


def build_algorithm_space(tiers: list = None) -> list:
    """
    Build the combined algorithm space for the given tiers.

    Parameters
    ----------
    tiers : list of int, default [1]
        Which tiers to include. Use [1] for Tier 1 only (K=48),
        [1, 2] for Tier 1 + Tier 2 (K=81),
        [1, 2, 3] for all tiers (K=117).

    Returns
    -------
    list of PortfolioAlgorithm instances.
    """
    if tiers is None:
        tiers = [1]

    from regime_algo_selection.algorithms.tier2_linear import build_tier2_algorithm_space
    from regime_algo_selection.algorithms.tier3_nonlinear import build_tier3_algorithms

    tier1_algos = _build_tier1_list() if 1 in tiers else []
    tier2_algos = build_tier2_algorithm_space() if 2 in tiers else []
    tier3_algos = build_tier3_algorithms() if 3 in tiers else []

    algos = tier1_algos + tier2_algos + tier3_algos
    n1, n2, n3 = len(tier1_algos), len(tier2_algos), len(tier3_algos)
    print(f"Built K={len(algos)} algorithms: {n1} Tier 1, {n2} Tier 2, {n3} Tier 3")
    return algos
