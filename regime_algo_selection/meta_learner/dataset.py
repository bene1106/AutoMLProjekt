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

from regime_algo_selection.algorithms.base import TrainablePortfolioAlgorithm


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
    # Batch precomputation (optimized)
    # ------------------------------------------------------------------

    def batch_precompute_algo_outputs(self, max_lookback: int = MAX_LOOKBACK) -> None:
        """
        Optimized batch version of precompute_algo_outputs().

        Speedups compared to the original:

        Tier 2+3 (TrainablePortfolioAlgorithm):
          Builds the full (T, n_feat) feature matrix for all days at once using
          the pre-attached _af_values / _af_index arrays, scales in one call, then
          batch-predicts all T days in a single sklearn call per algorithm.
          Reduces ~138 K individual 1-sample sklearn calls to 69 batch calls.

        Tier 1 simple heuristics (EW, RiskParity, Momentum/linear, TrendFollowing):
          Precomputes rolling vol / pct_change / SMA on the full price history once
          per unique lookback, then indexes into the pre-built arrays per day.

        Tier 1 complex heuristics (MinVar, MaxDiv, MeanVar) and Momentum/exp:
          Falls back to the original day-by-day loop (scipy.optimize.minimize is
          inherently per-day and not parallelisable here).

        Results are stored in self._algo_outputs (same shape / dtype as the
        original method) so all downstream code is unaffected.
        """
        from regime_algo_selection.algorithms.tier1_heuristics import (
            EqualWeight, RiskParity, Momentum, TrendFollowing,
        )

        price_idx = self.prices.index
        T = len(self.dates)
        ew = np.ones(self.N, dtype=np.float32) / self.N
        outputs = np.zeros((T, self.K, self.N), dtype=np.float32)

        # ── Step 1: vectorised date → price-index position map ──────────────────
        print("    [batch] computing date positions ...", flush=True)
        date_pos = np.full(T, -1, dtype=np.int64)
        for i, t in enumerate(self.dates):
            try:
                date_pos[i] = price_idx.get_loc(t)
            except KeyError:
                pass
        # has_price[i]: date i is in price_idx at position > 0
        # (position 0 → no history before it → prices_hist would be empty)
        has_price = date_pos > 0

        # ── Step 2: precompute rolling stats for simple Tier 1 algorithms ───────
        print("    [batch] precomputing rolling stats for simple Tier 1 ...", flush=True)
        daily_r = self.prices.pct_change(fill_method=None)

        rolling_vol: dict = {}  # lookback → rolling(L).std() DataFrame
        rolling_mom: dict = {}  # lookback → pct_change(L) DataFrame
        rolling_sma: dict = {}  # lookback → rolling(L).mean() DataFrame

        for algo in self.algorithms:
            if isinstance(algo, RiskParity):
                L = algo.lookback
                if L not in rolling_vol:
                    rolling_vol[L] = daily_r.rolling(L).std()
            elif isinstance(algo, Momentum) and algo.weighting == "linear":
                L = algo.lookback
                if L not in rolling_mom:
                    rolling_mom[L] = self.prices.pct_change(L)
            elif isinstance(algo, TrendFollowing):
                L = algo.lookback
                if L not in rolling_sma:
                    rolling_sma[L] = self.prices.rolling(L).mean()

        # ── Step 3: per-algorithm output computation ─────────────────────────────
        for k, algo in enumerate(self.algorithms):
            label = f"{algo.name[:45]}"
            print(
                f"\r    [batch] algo {k + 1:3d}/{self.K}: {label:<45}",
                end="", flush=True,
            )

            # ── Tier 2+3: fully vectorised batch prediction ──────────────────────
            if (
                isinstance(algo, TrainablePortfolioAlgorithm)
                and algo._is_fitted
                and algo._af_index is not None
                and algo._scaler is not None
            ):
                # For each dataset date t_i, the original code does:
                #   prices_hist = prices.iloc[start : date_pos[i]]
                #   algo._get_features_fast(prices_hist.index[-1])
                #     = searchsorted(af_index, price_idx[date_pos[i]-1].value, "right")
                # Reproduce this for all i at once.
                last_price_ns = np.full(T, -1, dtype=np.int64)
                for i in range(T):
                    if has_price[i]:
                        last_price_ns[i] = price_idx[int(date_pos[i]) - 1].value

                af_pos = np.searchsorted(algo._af_index, last_price_ns, side="right")

                in_range = (last_price_ns >= 0) & (af_pos < len(algo._af_index))
                af_pos_safe = np.clip(af_pos, 0, len(algo._af_index) - 1)

                X_raw = algo._af_values[af_pos_safe]            # (T, n_feat)
                finite_rows = np.isfinite(X_raw).all(axis=1)   # (T,)
                valid_af = in_range & finite_rows

                # Rows with invalid features → zero (masked out after softmax)
                X_all = np.where(valid_af[:, np.newaxis], X_raw, 0.0)
                X_scaled = algo._scaler.transform(X_all)        # (T, n_feat) scaled

                # Batch predict: Tier 2 has _model (multi-output), Tier 3 has _models (per-asset)
                if (
                    hasattr(algo, "_models")
                    and isinstance(algo._models, dict)
                    and algo._models
                ):
                    # Tier 3: one sklearn model per asset
                    mu_all = np.column_stack([
                        algo._models[j].predict(X_scaled)
                        for j in range(self.N)
                    ])                                          # (T, N)
                elif hasattr(algo, "_model") and algo._model is not None:
                    # Tier 2: single multi-output model
                    mu_all = algo._model.predict(X_scaled)     # (T, N)
                else:
                    outputs[:, k, :] = ew
                    continue

                # Row-wise numerically stable softmax
                mu_all = np.where(np.isfinite(mu_all), mu_all, 0.0)
                mu_all -= mu_all.max(axis=1, keepdims=True)
                exp_mu = np.exp(mu_all)
                denom = exp_mu.sum(axis=1, keepdims=True)
                denom = np.where(denom > 1e-12, denom, 1.0)
                w_all = (exp_mu / denom).astype(np.float32)    # (T, N)

                # Apply: valid days → predicted weights, invalid → equal weight
                outputs[:, k, :] = np.where(valid_af[:, np.newaxis], w_all, ew)
                continue

            # ── Tier 1: EqualWeight (trivially constant) ─────────────────────────
            if isinstance(algo, EqualWeight):
                outputs[:, k, :] = ew
                continue

            # ── Tier 1: RiskParity (vectorised rolling vol) ──────────────────────
            if isinstance(algo, RiskParity):
                L = algo.lookback
                rvol = rolling_vol[L]
                for i in range(T):
                    if not has_price[i]:
                        outputs[i, k] = ew
                        continue
                    t_prev = price_idx[int(date_pos[i]) - 1]
                    if t_prev not in rvol.index:
                        outputs[i, k] = ew
                        continue
                    row = rvol.loc[t_prev].values
                    if not np.any(np.isfinite(row)):
                        outputs[i, k] = ew
                        continue
                    vols = np.where(np.isfinite(row) & (row > 1e-10), row, 1e-10)
                    w = 1.0 / vols
                    s = w.sum()
                    outputs[i, k] = (w / s).astype(np.float32)
                continue

            # ── Tier 1: Momentum linear (vectorised pct_change) ──────────────────
            if isinstance(algo, Momentum) and algo.weighting == "linear":
                L = algo.lookback
                rmom = rolling_mom[L]
                for i in range(T):
                    if not has_price[i]:
                        outputs[i, k] = ew
                        continue
                    t_prev = price_idx[int(date_pos[i]) - 1]
                    if t_prev not in rmom.index:
                        outputs[i, k] = ew
                        continue
                    scores = rmom.loc[t_prev].values
                    scores = np.where(np.isfinite(scores), scores, 0.0)
                    scores -= scores.max()
                    w = np.exp(scores)
                    s = w.sum()
                    outputs[i, k] = (
                        (w / s).astype(np.float32) if s > 1e-12 else ew
                    )
                continue

            # ── Tier 1: TrendFollowing (vectorised SMA) ───────────────────────────
            if isinstance(algo, TrendFollowing):
                L = algo.lookback
                beta = algo.beta
                rsma = rolling_sma[L]
                for i in range(T):
                    if not has_price[i]:
                        outputs[i, k] = ew
                        continue
                    t_prev = price_idx[int(date_pos[i]) - 1]
                    if t_prev not in rsma.index:
                        outputs[i, k] = ew
                        continue
                    current = self.prices.loc[t_prev].values
                    sma_vals = rsma.loc[t_prev].values
                    signals = (current > sma_vals).astype(float)
                    if signals.sum() == 0:
                        outputs[i, k] = ew
                        continue
                    raw_w = signals ** beta
                    s = raw_w.sum()
                    outputs[i, k] = (raw_w / s).astype(np.float32)
                continue

            # ── Fallback: original day-by-day loop ────────────────────────────────
            # Covers: MinVar, MaxDiv, MeanVar (scipy.optimize), Momentum/exp
            for i, t in enumerate(self.dates):
                if not has_price[i]:
                    outputs[i, k] = ew
                    continue
                p = int(date_pos[i])
                start = max(0, p - max_lookback)
                prices_hist = self.prices.iloc[start:p]
                if len(prices_hist) < MIN_ALGO_HISTORY:
                    outputs[i, k] = ew
                    continue
                try:
                    w = algo.compute_weights(prices_hist)
                    w = np.where(np.isfinite(w), w, 0.0)
                    w = np.clip(w, 0.0, None)
                    total = w.sum()
                    outputs[i, k] = (
                        (w / total).astype(np.float32) if total > 1e-12 else ew
                    )
                except Exception:
                    outputs[i, k] = ew

        print(f"\r    [batch] Precomputed algo outputs: {self.K}/{self.K} done.        ")
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
