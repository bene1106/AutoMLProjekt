# evaluation/backtest.py — Core backtesting engine

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from regime_algo_selection.config import KAPPA, N_ASSETS, ASSETS


@dataclass
class BacktestResult:
    """Container for all time series produced by a backtest run."""
    portfolio_returns: pd.Series       # gross daily returns
    net_returns: pd.Series             # after switching costs
    weights_history: pd.DataFrame      # N assets x T dates
    algorithm_selections: pd.Series    # algorithm name selected each day
    regime_predictions: pd.Series      # s_hat_t (classifier or oracle)
    regime_true: pd.Series             # s*_t
    switching_costs: pd.Series         # C_t = kappa * ||w_t - w_{t-1}||_1
    algorithm_name: str = "unnamed"    # label for this backtest run


class Backtester:
    """
    Walk-forward simulation loop.

    At each day t in [start_date, end_date]:
      1. Classifier predicts regime s_hat_t from lagged VIX features z_t.
      2. Agent selects algorithm A_k based on s_hat_t (or s*_t for oracle).
      3. A_k computes weights w_t from prices up to t-1.
      4. Portfolio earns gross return R_t = w_t . r_{t->t+1}.
      5. Switching cost C_t = kappa * ||w_t - w_{t-1}||_1.
      6. Net return = R_t - C_t.
    """

    def __init__(
        self,
        algorithms: list,
        regime_classifier,        # RegimeClassifier
        returns: pd.DataFrame,    # forward returns (index=date, r_t = ret from t to t+1)
        prices: pd.DataFrame,     # raw prices (for weight computation)
        vix_features: pd.DataFrame,
        regime_labels: pd.Series,
        kappa: float = KAPPA,
    ):
        self.algorithms = algorithms
        self.classifier = regime_classifier
        self.returns = returns
        self.prices = prices
        self.vix_features = vix_features
        self.regime_labels = regime_labels
        self.kappa = kappa

        # Build lookup: name -> algorithm object
        self._algo_map = {a.name: a for a in algorithms}

    # ------------------------------------------------------------------
    def run(
        self,
        agent,
        start_date: str,
        end_date: str,
        run_label: str = "backtest",
        use_true_regime: bool = False,
    ) -> BacktestResult:
        """
        Run the backtest for a given agent over [start_date, end_date].

        Parameters
        ----------
        agent : ReflexAgent or OracleAgent
            Must implement .select(regime) -> PortfolioAlgorithm.
        start_date, end_date : str
            Date range for the backtest.
        run_label : str
            Label for the BacktestResult.
        use_true_regime : bool
            If True, feed agent the TRUE regime (oracle mode).
        """
        # Restrict to date range
        mask = (self.returns.index >= start_date) & (self.returns.index <= end_date)
        dates = self.returns.index[mask]

        if len(dates) == 0:
            raise ValueError(f"No dates in [{start_date}, {end_date}]")

        # Storage
        port_returns    = []
        net_rets        = []
        algo_selections = []
        regime_preds    = []
        switch_costs    = []
        weights_list    = []

        prev_w = np.ones(N_ASSETS) / N_ASSETS  # start equal-weighted

        # Pre-compute regime predictions for all dates at once (efficiency)
        vix_feats_test = self.vix_features.loc[dates]
        if use_true_regime:
            regime_pred_series = self.regime_labels.loc[dates]
        else:
            regime_pred_series = self.classifier.predict(vix_feats_test)

        for t in dates:
            # --- 1. Regime estimate ---
            s_hat = regime_pred_series.loc[t]
            s_hat = int(s_hat) if not np.isnan(s_hat) else 2  # fallback Normal

            # --- 2. Agent selects algorithm ---
            algo = agent.select(s_hat)

            # --- 3. Compute weights from prices up to t-1 ---
            prices_up_to_t = self.prices.loc[self.prices.index < t]
            if len(prices_up_to_t) == 0:
                w_t = np.ones(N_ASSETS) / N_ASSETS
            else:
                try:
                    w_t = algo.compute_weights(prices_up_to_t)
                except Exception:
                    w_t = np.ones(N_ASSETS) / N_ASSETS

            # Safety: ensure valid weights
            w_t = np.where(np.isfinite(w_t), w_t, 0.0)
            w_t = np.clip(w_t, 0, None)
            total = w_t.sum()
            if total < 1e-12:
                w_t = np.ones(N_ASSETS) / N_ASSETS
            else:
                w_t = w_t / total

            # --- 4. Gross portfolio return ---
            if t in self.returns.index:
                r_t = self.returns.loc[t].values
                r_t = np.where(np.isfinite(r_t), r_t, 0.0)
            else:
                r_t = np.zeros(N_ASSETS)
            R_t = float(w_t @ r_t)

            # --- 5. Switching cost ---
            C_t = self.kappa * float(np.abs(w_t - prev_w).sum())

            # --- 6. Net return ---
            R_net_t = R_t - C_t

            # Store
            port_returns.append(R_t)
            net_rets.append(R_net_t)
            algo_selections.append(algo.name)
            regime_preds.append(s_hat)
            switch_costs.append(C_t)
            weights_list.append(w_t)

            prev_w = w_t

        # Assemble DataFrames
        weights_df = pd.DataFrame(
            np.array(weights_list), index=dates, columns=ASSETS
        )

        return BacktestResult(
            portfolio_returns   = pd.Series(port_returns,    index=dates, name="gross_return"),
            net_returns         = pd.Series(net_rets,        index=dates, name="net_return"),
            weights_history     = weights_df,
            algorithm_selections= pd.Series(algo_selections, index=dates, name="algorithm"),
            regime_predictions  = pd.Series(regime_preds,    index=dates, name="regime_pred"),
            regime_true         = self.regime_labels.loc[dates],
            switching_costs     = pd.Series(switch_costs,    index=dates, name="switching_cost"),
            algorithm_name      = run_label,
        )
