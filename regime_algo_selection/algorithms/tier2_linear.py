# algorithms/tier2_linear.py -- Tier 2 Linear ML Portfolio Algorithms
#
# Three families of regularized linear regression models:
#   F8: Ridge Portfolio    (12 configurations)
#   F9: Lasso Portfolio    ( 9 configurations)
#   F10: Elastic Net       (12 configurations)
#
# These algorithms follow the "Stage 0" pre-training pattern:
#   1. fit(X_train, Y_train) is called ONCE on training data (frozen after)
#   2. compute_weights(prices_hist) predicts returns -> softmax -> weights
#
# Total Tier 2: 12 + 9 + 12 = 33 algorithms

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from regime_algo_selection.algorithms.base import TrainablePortfolioAlgorithm


# ---------------------------------------------------------------------------
# F8: Ridge Portfolio
# Hyperparams: lambda_ridge in {0.01, 0.1, 1, 10} x L in {60, 120, 252}
# Total: 4 x 3 = 12
# ---------------------------------------------------------------------------

class RidgePortfolio(TrainablePortfolioAlgorithm):
    """
    Ridge-regularised linear regression portfolio.

    Fits Ridge(alpha=lambda_ridge) mapping asset features -> next-day returns.
    At inference: predict returns -> softmax -> portfolio weights.
    """

    def __init__(self, lambda_ridge: float, lookback: int):
        super().__init__(
            name=f"Ridge_L{lookback}_a{lambda_ridge}",
            family="RidgePortfolio",
            hyperparams={"lambda_ridge": lambda_ridge, "lookback": lookback},
        )
        self.lambda_ridge = lambda_ridge
        self.lookback = lookback
        self._model = None
        self._scaler = None
        self._n_assets = None

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "RidgePortfolio":
        """Train Ridge regression on (features, next-day returns)."""
        # Filter out NaN rows
        mask = np.isfinite(X_train).all(axis=1) & np.isfinite(Y_train).all(axis=1)
        X, Y = X_train[mask], Y_train[mask]

        if len(X) < 30:
            # Not enough data: fall back to equal-weight (is_fitted stays False)
            self._is_fitted = False
            return self

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Ridge natively supports multi-output
        self._model = Ridge(alpha=self.lambda_ridge, fit_intercept=True)
        self._model.fit(X_scaled, Y)
        self._n_assets = Y.shape[1]
        self._is_fitted = True
        return self

    def compute_weights(self, prices_history: pd.DataFrame, features: dict = None) -> np.ndarray:
        n = prices_history.shape[1]
        if not self._is_fitted or len(prices_history) < 22:
            return self._equal_weight(n)

        x = self._get_features_fast(prices_history.index[-1])
        if x is None:
            x = self._compute_feature_row(prices_history)
        x_scaled = self._scaler.transform(x.reshape(1, -1))
        mu_hat = self._model.predict(x_scaled).flatten()
        return self._softmax(mu_hat)


# ---------------------------------------------------------------------------
# F9: Lasso Portfolio
# Hyperparams: lambda_lasso in {0.001, 0.01, 0.1} x L in {60, 120, 252}
# Total: 3 x 3 = 9
# ---------------------------------------------------------------------------

class LassoPortfolio(TrainablePortfolioAlgorithm):
    """
    Lasso-regularised linear regression portfolio.

    Uses MultiOutputRegressor(Lasso) since sklearn Lasso doesn't support
    multi-output natively.
    """

    def __init__(self, lambda_lasso: float, lookback: int):
        super().__init__(
            name=f"Lasso_L{lookback}_a{lambda_lasso}",
            family="LassoPortfolio",
            hyperparams={"lambda_lasso": lambda_lasso, "lookback": lookback},
        )
        self.lambda_lasso = lambda_lasso
        self.lookback = lookback
        self._model = None
        self._scaler = None
        self._n_assets = None

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "LassoPortfolio":
        mask = np.isfinite(X_train).all(axis=1) & np.isfinite(Y_train).all(axis=1)
        X, Y = X_train[mask], Y_train[mask]

        if len(X) < 30:
            self._is_fitted = False
            return self

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        base_lasso = Lasso(alpha=self.lambda_lasso, fit_intercept=True, max_iter=5000)
        self._model = MultiOutputRegressor(base_lasso, n_jobs=1)
        self._model.fit(X_scaled, Y)
        self._n_assets = Y.shape[1]
        self._is_fitted = True
        return self

    def compute_weights(self, prices_history: pd.DataFrame, features: dict = None) -> np.ndarray:
        n = prices_history.shape[1]
        if not self._is_fitted or len(prices_history) < 22:
            return self._equal_weight(n)

        x = self._get_features_fast(prices_history.index[-1])
        if x is None:
            x = self._compute_feature_row(prices_history)
        x_scaled = self._scaler.transform(x.reshape(1, -1))
        mu_hat = self._model.predict(x_scaled).flatten()
        return self._softmax(mu_hat)


# ---------------------------------------------------------------------------
# F10: Elastic Net Portfolio
# Hyperparams: lambda in {0.01, 0.1} x rho in {0.25, 0.5, 0.75} x L in {60, 120}
# Total: 2 x 3 x 2 = 12
# ---------------------------------------------------------------------------

class ElasticNetPortfolio(TrainablePortfolioAlgorithm):
    """
    Elastic Net regularised linear regression portfolio.

    Combines L1 + L2 penalty: alpha controls overall strength,
    l1_ratio (rho) controls the L1/L2 mix.
    """

    def __init__(self, lambda_en: float, l1_ratio: float, lookback: int):
        super().__init__(
            name=f"ElasticNet_L{lookback}_a{lambda_en}_r{l1_ratio}",
            family="ElasticNetPortfolio",
            hyperparams={"lambda_en": lambda_en, "l1_ratio": l1_ratio, "lookback": lookback},
        )
        self.lambda_en = lambda_en
        self.l1_ratio = l1_ratio
        self.lookback = lookback
        self._model = None
        self._scaler = None
        self._n_assets = None

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "ElasticNetPortfolio":
        mask = np.isfinite(X_train).all(axis=1) & np.isfinite(Y_train).all(axis=1)
        X, Y = X_train[mask], Y_train[mask]

        if len(X) < 30:
            self._is_fitted = False
            return self

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        base_en = ElasticNet(
            alpha=self.lambda_en, l1_ratio=self.l1_ratio,
            fit_intercept=True, max_iter=5000,
        )
        self._model = MultiOutputRegressor(base_en, n_jobs=1)
        self._model.fit(X_scaled, Y)
        self._n_assets = Y.shape[1]
        self._is_fitted = True
        return self

    def compute_weights(self, prices_history: pd.DataFrame, features: dict = None) -> np.ndarray:
        n = prices_history.shape[1]
        if not self._is_fitted or len(prices_history) < 22:
            return self._equal_weight(n)

        x = self._get_features_fast(prices_history.index[-1])
        if x is None:
            x = self._compute_feature_row(prices_history)
        x_scaled = self._scaler.transform(x.reshape(1, -1))
        mu_hat = self._model.predict(x_scaled).flatten()
        return self._softmax(mu_hat)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_tier2_algorithm_space() -> list:
    """
    Instantiate all Tier 2 algorithm configurations.

    F8: Ridge     4 alphas x 3 lookbacks = 12
    F9: Lasso     3 alphas x 3 lookbacks =  9
    F10: ElasticNet  2 alphas x 3 l1_ratios x 2 lookbacks = 12
    Total: 33
    """
    algos = []

    # F8: Ridge
    for lam in [0.01, 0.1, 1.0, 10.0]:
        for L in [60, 120, 252]:
            algos.append(RidgePortfolio(lambda_ridge=lam, lookback=L))

    # F9: Lasso
    for lam in [0.001, 0.01, 0.1]:
        for L in [60, 120, 252]:
            algos.append(LassoPortfolio(lambda_lasso=lam, lookback=L))

    # F10: ElasticNet
    for lam in [0.01, 0.1]:
        for rho in [0.25, 0.5, 0.75]:
            for L in [60, 120]:
                algos.append(ElasticNetPortfolio(lambda_en=lam, l1_ratio=rho, lookback=L))

    n_f8  = 4 * 3
    n_f9  = 3 * 3
    n_f10 = 2 * 3 * 2
    print(
        f"Built K={len(algos)} Tier 2 algorithms "
        f"(F8=Ridge:{n_f8}, F9=Lasso:{n_f9}, F10=ElasticNet:{n_f10})"
    )
    return algos
