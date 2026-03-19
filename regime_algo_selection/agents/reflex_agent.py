# agents/reflex_agent.py — Reflex Agent and Oracle Agent

import numpy as np
import pandas as pd

from regime_algo_selection.config import N_REGIMES, REGIME_NAMES, N_ASSETS, KAPPA


def _evaluate_algo_in_regime(
    algo,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime_dates: pd.DatetimeIndex,
    kappa: float = 0.0,
) -> float:
    """
    Evaluate algorithm `algo` on the subset of dates belonging to a single regime.

    Parameters
    ----------
    kappa : float
        If > 0, subtracts switching costs from daily returns before computing Sharpe.
        This enables net-Sharpe fitting.

    Returns
    -------
    float : annualised Sharpe ratio (gross if kappa=0, net otherwise)
    """
    daily_rets = []
    prev_w = np.ones(N_ASSETS) / N_ASSETS

    for t in regime_dates:
        prices_up_to_t = prices.loc[prices.index < t]
        if len(prices_up_to_t) < 22:
            continue
        try:
            w = algo.compute_weights(prices_up_to_t)
        except Exception:
            w = np.ones(N_ASSETS) / N_ASSETS

        w = np.where(np.isfinite(w), w, 0.0)
        w = np.clip(w, 0, None)
        s = w.sum()
        if s < 1e-12:
            w = np.ones(N_ASSETS) / N_ASSETS
        else:
            w /= s

        if t not in returns.index:
            prev_w = w
            continue

        r = returns.loc[t].fillna(0).values
        gross_ret = float(w @ r)
        switch_cost = kappa * float(np.abs(w - prev_w).sum())
        daily_rets.append(gross_ret - switch_cost)
        prev_w = w

    if len(daily_rets) < 10:
        return -999.0

    arr = np.array(daily_rets)
    mean = arr.mean()
    std  = arr.std()
    if std < 1e-12:
        return mean * np.sqrt(252)
    return (mean / std) * np.sqrt(252)


class ReflexAgent:
    """
    Regime -> fixed algorithm lookup table.

    fit() evaluates every algorithm on each regime's training dates and
    picks the one with the highest Sharpe ratio (gross or net).
    """

    def __init__(self):
        self.mapping: dict = {}         # {regime_int: PortfolioAlgorithm}
        self.fit_metric: str = "gross"  # "gross" or "net"
        self.fit_kappa: float = 0.0
        self.all_scores: dict = {}      # {regime_int: {algo_name: score}} -- set by fit()

    # ------------------------------------------------------------------
    def fit(
        self,
        algorithms: list,
        returns: pd.DataFrame,
        regime_labels: pd.Series,
        prices: pd.DataFrame,
        metric: str = "gross",
        kappa: float = KAPPA,
    ) -> "ReflexAgent":
        """
        For each regime r, find the algorithm with the highest in-regime Sharpe.

        Parameters
        ----------
        algorithms    : list of PortfolioAlgorithm
        returns       : forward returns DataFrame (training period)
        regime_labels : true regime series (training period)
        prices        : full price history (used to build look-back windows)
        metric        : "gross" (current default) or "net" (penalises high turnover)
        kappa         : switching cost coefficient, only used when metric="net"
        """
        self.fit_metric = metric
        self.fit_kappa  = kappa if metric == "net" else 0.0

        label = "net-Sharpe" if metric == "net" else "gross-Sharpe"
        print(f"Fitting ReflexAgent ({label}) -- evaluating algorithms per regime...")

        common_dates = returns.index.intersection(regime_labels.index)

        for regime in range(1, N_REGIMES + 1):
            regime_dates = common_dates[regime_labels.loc[common_dates] == regime]
            print(
                f"  Regime {regime} ({REGIME_NAMES[regime]}): "
                f"{len(regime_dates)} training days"
            )

            best_score = -np.inf
            best_algo  = algorithms[0]
            regime_scores = {}

            for algo in algorithms:
                score = _evaluate_algo_in_regime(
                    algo, prices, returns, regime_dates,
                    kappa=self.fit_kappa,
                )
                regime_scores[algo.name] = score
                if score > best_score:
                    best_score = score
                    best_algo  = algo

            self.mapping[regime] = best_algo
            self.all_scores[regime] = regime_scores
            print(f"    Best: {best_algo.name}  ({label}={best_score:.4f})")

        print(f"\nReflexAgent mapping: { {r: a.name for r, a in self.mapping.items()} }")
        return self

    # ------------------------------------------------------------------
    def select(self, regime_estimate: int):
        """Return the pre-assigned algorithm for this regime."""
        regime_estimate = int(regime_estimate)
        if regime_estimate not in self.mapping:
            regime_estimate = 2   # fallback: Normal
        return self.mapping[regime_estimate]


class OracleAgent(ReflexAgent):
    """
    Same lookup table as ReflexAgent, but at decision time it receives
    the TRUE regime s*_t (upper bound on achievable performance).
    """
    pass
