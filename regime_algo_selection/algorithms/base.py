# algorithms/base.py — Abstract base class for all portfolio algorithms

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class PortfolioAlgorithm(ABC):
    """
    Base class for all portfolio construction algorithms.

    Every concrete algorithm must implement compute_weights().
    """

    def __init__(self, name: str, family: str, hyperparams: dict):
        self.name = name              # e.g., "MinVar_L60"
        self.family = family          # e.g., "MinimumVariance"
        self.hyperparams = hyperparams

    @abstractmethod
    def compute_weights(
        self,
        prices_history: pd.DataFrame,
        features: dict = None,
    ) -> np.ndarray:
        """
        Given historical prices up to (and including) the most recent close,
        return portfolio weights w_t in the simplex Delta_N.

        Parameters
        ----------
        prices_history : pd.DataFrame
            Price DataFrame up to and including day t-1.
            Index = dates (sorted ascending), columns = asset tickers.
        features : dict, optional
            Any additional info needed by the algorithm.

        Returns
        -------
        np.ndarray of shape (N,) — non-negative weights that sum to 1.
        """

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _equal_weight(n: int) -> np.ndarray:
        return np.ones(n) / n

    @staticmethod
    def _safe_normalize(w: np.ndarray) -> np.ndarray:
        """Clip negatives, normalize. Fall back to equal weight if all zero."""
        w = np.clip(w, 0, None)
        total = w.sum()
        if total < 1e-12:
            return np.ones(len(w)) / len(w)
        return w / total

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class TrainablePortfolioAlgorithm(PortfolioAlgorithm):
    """
    Extension of PortfolioAlgorithm for ML-based Tier 2 algorithms.

    These algorithms must be fitted ONCE on training data (Stage 0)
    before they can be used for inference. After fit(), the model is frozen.
    """

    def __init__(self, name: str, family: str, hyperparams: dict):
        super().__init__(name, family, hyperparams)
        self._is_fitted: bool = False
        # Pre-computed asset features for O(log n) lookup during inference.
        # Set by pretrain_tier2_algorithms() after calling fit().
        self._af_values: np.ndarray = None   # shape (T_all, n_feat), float64
        self._af_index:  np.ndarray = None   # shape (T_all,), int64 epoch-ns

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def attach_full_features(self, asset_features: pd.DataFrame) -> None:
        """
        Store the full asset features DataFrame for fast O(log n) lookup.

        asset_features has MultiIndex columns (asset, feature) and index=date.
        At inference date t, we look up the features one row AHEAD of
        prices_history.index[-1] = t-1 (i.e., features at t = info up to t-1).
        """
        self._af_values = asset_features.values.astype(float)
        self._af_index  = asset_features.index.astype(np.int64).values

    def _get_features_fast(self, last_price_date) -> np.ndarray:
        """
        O(log n) lookup of pre-computed features for the day AFTER last_price_date.
        Falls back to None if lookup fails (caller will use _compute_feature_row).
        """
        if self._af_index is None:
            return None
        ts = np.int64(pd.Timestamp(last_price_date).value)
        pos = np.searchsorted(self._af_index, ts, side="right")
        if pos < len(self._af_index):
            x = self._af_values[pos]
            return x if np.isfinite(x).all() else None
        return None

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "TrainablePortfolioAlgorithm":
        """
        Train the model on historical data. Called once during Stage 0.

        Parameters
        ----------
        X_train : np.ndarray, shape (T, n_features)
            Pre-computed lagged per-asset features for the training period.
        Y_train : np.ndarray, shape (T, N_assets)
            Next-period returns (targets) for the training period.

        Returns
        -------
        self (for chaining)
        """
        raise NotImplementedError("Subclasses must implement fit()")

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax -> simplex weights."""
        x = np.where(np.isfinite(x), x, 0.0)
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    # Maximum lookback needed: 120 days (mom_120d) + 1 for pct_change = 122;
    # add a small buffer -> use last 130 rows only.
    _FEAT_LOOKBACK: int = 130

    @staticmethod
    def _compute_feature_row(prices_hist: pd.DataFrame) -> np.ndarray:
        """
        Compute per-asset features (9 per asset) from a price history window.
        Matches the feature set used in data/features.py:
          ret_1d, ret_5d, ret_20d, ret_60d, vol_20d, vol_60d,
          mom_20d, mom_60d, mom_120d
        The last row of prices_hist corresponds to t-1, so features are lagged.

        Only uses the last _FEAT_LOOKBACK=130 rows for speed -- this is sufficient
        for all features whose maximum lookback is 120 days.
        """
        # Slice to at most last 130 rows (max lookback needed = 120 + 1)
        tail = prices_hist.iloc[-TrainablePortfolioAlgorithm._FEAT_LOOKBACK:]
        daily_r = tail.pct_change(fill_method=None)
        features = []

        for col in tail.columns:
            r = daily_r[col]
            p = tail[col]

            def _safe(series):
                v = series.iloc[-1] if len(series) > 0 else np.nan
                return float(v) if np.isfinite(v) else 0.0

            ret_1d   = _safe(r)
            ret_5d   = _safe(p.pct_change(5))   if len(p) > 5   else 0.0
            ret_20d  = _safe(p.pct_change(20))  if len(p) > 20  else 0.0
            ret_60d  = _safe(p.pct_change(60))  if len(p) > 60  else 0.0
            vol_20d  = _safe(r.rolling(20).std()) * np.sqrt(252) if len(r) >= 20 else 0.0
            vol_60d  = _safe(r.rolling(60).std()) * np.sqrt(252) if len(r) >= 60 else 0.0
            mom_20d  = ret_20d
            mom_60d  = ret_60d
            mom_120d = _safe(p.pct_change(120)) if len(p) > 120 else 0.0

            features.extend([ret_1d, ret_5d, ret_20d, ret_60d,
                             vol_20d, vol_60d, mom_20d, mom_60d, mom_120d])

        return np.array(features, dtype=float)
