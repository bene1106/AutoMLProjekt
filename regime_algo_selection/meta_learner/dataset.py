# meta_learner/dataset.py -- Dataset assembly for the Meta-Learner (Plan 12)
#
# Assembles X_t = [asset_features (25), regime_onehot (4)] for each time step.
# Precomputes algorithm outputs W_t^(k) for all K algorithms to avoid
# recomputing them on every training epoch.
#
# LEAKAGE PREVENTION:
#   - All features are already lagged in compute_asset_features() (uses data up to t-1)
#   - StandardScaler fitted on training block only
#   - Algorithm outputs use prices up to t-1 (prices_hist = prices.iloc[start:pos])
#   - Regime labels are ground truth from VIX (no look-ahead)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# 5 features per asset, matching the plan's initial feature set
SELECTED_FEATURES = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "mom_60d"]
# Longest lookback in the selected features is mom_60d (60 days)
WARMUP_DAYS = 60
# Minimum price history required by any algorithm
MIN_ALGO_HISTORY = 22
# Price window used when calling algo.compute_weights()
MAX_LOOKBACK = 310


class MetaLearnerDataset:
    """
    Assembles inputs and precomputes quantities needed for meta-learner training.

    For each valid time step t (integer index into self.dates):
        X_t          : input vector (29,) = asset_features (25) + regime_onehot (4)
        W_t[k]       : portfolio weights from algorithm k at time t, shape (N,)
        r_{t->t+1}   : next-period returns, shape (N,)

    All features are lagged: they only use information available at close of day t-1.

    Usage
    -----
    dataset = MetaLearnerDataset(prices, all_asset_features, returns, regime_labels, algorithms)
    dataset.precompute_algo_outputs()          # one-time, expensive
    dataset.fit_scaler(train_start, train_end) # fit scaler on training block only
    train_idx = dataset.get_indices_for_period(train_start, train_end)
    test_idx  = dataset.get_indices_for_period(test_start,  test_end)
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        all_asset_features: pd.DataFrame,
        returns: pd.DataFrame,
        regime_labels: pd.Series,
        algorithms: list,
    ):
        """
        Parameters
        ----------
        prices             : price DataFrame, shape (T_all, N)
        all_asset_features : MultiIndex columns (asset, feature), 9 features per asset
                             from compute_asset_features(). This function selects a subset.
        returns            : forward returns DataFrame, shape (T_all, N)
                             returns.loc[t] = return from t to t+1
        regime_labels      : Series {1,2,3,4}, oracle regime s*_t
        algorithms         : list of K PortfolioAlgorithm instances (pre-trained Tier 2)
        """
        self.prices = prices
        self.algorithms = algorithms
        self.K = len(algorithms)
        self.N = prices.shape[1]
        self.returns = returns
        self.regime_labels = regime_labels

        # Select the 5-feature subset: shape (T_all, N*5) = (T_all, 25)
        # Columns must exist in all_asset_features
        sel_cols = []
        for asset in prices.columns:
            for feat in SELECTED_FEATURES:
                if (asset, feat) in all_asset_features.columns:
                    sel_cols.append((asset, feat))
        if not sel_cols:
            raise ValueError(
                "None of the SELECTED_FEATURES found in all_asset_features. "
                f"Expected columns like {[(prices.columns[0], f) for f in SELECTED_FEATURES]}"
            )
        self.asset_features = all_asset_features[sel_cols]  # shape (T_all, 25)

        # Valid dates: features non-NaN, returns available, regime label available
        feat_valid_dates = self.asset_features.dropna(how="any").index
        self.dates = (
            feat_valid_dates
            .intersection(returns.index)
            .intersection(regime_labels.index)
        )
        # Sorted ascending (already the case, but be explicit)
        self.dates = self.dates.sort_values()

        # Integer lookup map: date -> position in self.dates
        self._date_to_idx = {t: i for i, t in enumerate(self.dates)}

        # Precomputed quantities (filled by fit_scaler / precompute_algo_outputs)
        self._algo_outputs: np.ndarray = None   # shape (T, K, N), float32
        self._scaled_features: np.ndarray = None  # shape (T, 25), float32
        self._scaler: StandardScaler = None

    # ------------------------------------------------------------------
    # Scaler
    # ------------------------------------------------------------------

    def fit_scaler(self, train_start: str, train_end: str) -> None:
        """
        Fit StandardScaler on training block, transform all dates.

        Must be called AFTER precompute_algo_outputs() is not needed —
        can be called any time before get_input() is used.

        Parameters
        ----------
        train_start, train_end : str or Timestamp
            Boundaries of the training block (inclusive).
        """
        train_dates = self.dates[
            (self.dates >= pd.Timestamp(train_start)) &
            (self.dates <= pd.Timestamp(train_end))
        ]
        if len(train_dates) == 0:
            raise ValueError(f"No valid dates in training period {train_start}–{train_end}")

        train_feat = self.asset_features.loc[train_dates].values.astype(np.float32)
        self._scaler = StandardScaler()
        self._scaler.fit(train_feat)

        all_feat = self.asset_features.loc[self.dates].values.astype(np.float32)
        self._scaled_features = self._scaler.transform(all_feat)

    # ------------------------------------------------------------------
    # Algorithm output precomputation
    # ------------------------------------------------------------------

    def precompute_algo_outputs(self, max_lookback: int = MAX_LOOKBACK) -> None:
        """
        Precompute W_t^(k) for all K algorithms and all valid dates.

        This is the most expensive operation (~O(T * K) algorithm evaluations).
        Computed ONCE per fold; results cached in self._algo_outputs.

        Uses a sliding price window (size <= max_lookback) for efficiency.
        Falls back to equal-weight when price history is insufficient.
        """
        price_idx = self.prices.index
        T = len(self.dates)
        ew = np.ones(self.N, dtype=np.float32) / self.N

        outputs = np.zeros((T, self.K, self.N), dtype=np.float32)

        for i, t in enumerate(self.dates):
            if i % 250 == 0:
                print(f"\r    Precomputing algo outputs: {i}/{T}", end="", flush=True)

            # Price history up to t-1: prices.iloc[start:pos] excludes t
            try:
                pos = price_idx.get_loc(t)
            except KeyError:
                outputs[i] = ew[np.newaxis, :]
                continue

            start = max(0, pos - max_lookback)
            prices_hist = self.prices.iloc[start:pos]

            for k, algo in enumerate(self.algorithms):
                if len(prices_hist) < MIN_ALGO_HISTORY:
                    outputs[i, k] = ew
                    continue
                try:
                    w = algo.compute_weights(prices_hist)
                    w = np.where(np.isfinite(w), w, 0.0)
                    w = np.clip(w, 0.0, None)
                    total = w.sum()
                    outputs[i, k] = w / total if total > 1e-12 else ew
                except Exception:
                    outputs[i, k] = ew

        print(f"\r    Precomputed algo outputs: {T}/{T} done.        ")
        self._algo_outputs = outputs

    # ------------------------------------------------------------------
    # Per-step getters (used by trainer and experiment)
    # ------------------------------------------------------------------

    def get_input(self, idx: int) -> np.ndarray:
        """
        Return X_t for time step idx.

        Returns
        -------
        np.ndarray of shape (29,) = [scaled_asset_features (25), regime_onehot (4)]
        """
        if self._scaled_features is not None:
            asset_feat = self._scaled_features[idx]
        else:
            asset_feat = self.asset_features.iloc[idx].values.astype(np.float32)

        t = self.dates[idx]
        regime = int(self.regime_labels.loc[t])
        regime_oh = np.zeros(4, dtype=np.float32)
        if 1 <= regime <= 4:
            regime_oh[regime - 1] = 1.0

        return np.concatenate([asset_feat, regime_oh])  # shape (29,)

    def get_algorithm_outputs(self, idx: int) -> np.ndarray:
        """
        Return W_t = [w_t^(1), ..., w_t^(K)], shape (K, N).

        Requires precompute_algo_outputs() to have been called.
        """
        if self._algo_outputs is None:
            raise RuntimeError("Call precompute_algo_outputs() before get_algorithm_outputs().")
        return self._algo_outputs[idx]

    def get_returns(self, idx: int) -> np.ndarray:
        """Return r_{t->t+1} for time step idx. Shape (N,)."""
        t = self.dates[idx]
        return self.returns.loc[t].fillna(0.0).values.astype(np.float32)

    def get_regime(self, idx: int) -> int:
        """Return oracle regime s*_t for time step idx."""
        return int(self.regime_labels.loc[self.dates[idx]])

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def get_indices_for_period(self, start: str, end: str) -> np.ndarray:
        """Return integer indices (into self.dates) that fall within [start, end]."""
        mask = (self.dates >= pd.Timestamp(start)) & (self.dates <= pd.Timestamp(end))
        return np.where(mask)[0]
