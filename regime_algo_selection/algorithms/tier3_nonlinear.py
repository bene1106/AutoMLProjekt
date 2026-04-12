# algorithms/tier3_nonlinear.py -- Tier 3 Non-Linear ML Portfolio Algorithms
#
# Three families of non-linear ML regression models:
#   F11: Random Forest       (12 configurations)
#   F12: Gradient Boosting   (16 configurations)
#   F13: MLP                 ( 8 configurations)
#
# Same interface as Tier 2: fit(X_train, Y_train) then compute_weights(prices_hist).
# Trains ONE sklearn model per asset to map lagged features -> next-period return.
#
# Total Tier 3: 12 + 16 + 8 = 36 algorithms

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from regime_algo_selection.algorithms.base import TrainablePortfolioAlgorithm


class Tier3Algorithm(TrainablePortfolioAlgorithm):
    """
    Non-linear ML algorithm for return prediction -> portfolio weights.

    Same interface as Tier 2: fit() on training data, compute_weights() at
    decision time. Trains one sklearn model per asset.

    Pipeline:
    1. fit(): Train one model per asset on (features, next_period_return) pairs
    2. compute_weights(): Predict returns for next period -> softmax -> weights
    """

    def __init__(
        self,
        family: str,
        model_class,
        model_params: dict,
        lookback: int,
        name: str,
    ):
        """
        Args:
            family: "RandomForest", "GradientBoosting", or "MLP"
            model_class: sklearn estimator class
            model_params: dict of hyperparameters
            lookback: L, number of days of history used as features
            name: human-readable name, e.g. "RF_n100_d5_L60"
        """
        super().__init__(
            name=name,
            family=family,
            hyperparams={**model_params, "lookback": lookback},
        )
        self.family_name = family
        self.model_class = model_class
        self.model_params = model_params
        self.lookback = lookback
        self._models = {}   # asset_idx -> fitted sklearn model
        self._scaler = None
        self._n_assets = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "Tier3Algorithm":
        """
        Train one model per asset on (features, next_period_return) pairs.

        Parameters
        ----------
        X_train : np.ndarray, shape (T, n_features)
            Flattened per-asset features for the training period.
        Y_train : np.ndarray, shape (T, N_assets)
            Next-period returns (targets) for the training period.
        """
        mask = np.isfinite(X_train).all(axis=1) & np.isfinite(Y_train).all(axis=1)
        X, Y = X_train[mask], Y_train[mask]

        if len(X) < 30:
            self._is_fitted = False
            return self

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._n_assets = Y.shape[1]
        self._models = {}

        for j in range(self._n_assets):
            model = self.model_class(**self.model_params)
            model.fit(X_scaled, Y[:, j])
            self._models[j] = model

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def compute_weights(
        self, prices_history: pd.DataFrame, features: dict = None
    ) -> np.ndarray:
        """
        Predict next-period returns -> softmax -> portfolio weights in Delta_N.
        """
        n = prices_history.shape[1]
        if not self._is_fitted or len(prices_history) < 22:
            return self._equal_weight(n)

        # Fast path: use pre-attached features if available
        x = self._get_features_fast(prices_history.index[-1])
        if x is None:
            x = self._compute_feature_row(prices_history)

        x_scaled = self._scaler.transform(x.reshape(1, -1))
        mu_hat = np.array(
            [self._models[j].predict(x_scaled)[0] for j in range(self._n_assets)],
            dtype=float,
        )
        return self._softmax(mu_hat)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_tier3_algorithms() -> list:
    """
    Instantiate all Tier 3 algorithm configurations.

    F11: RandomForest   n_est in {100,300} x max_depth in {5,10,None} x L in {60,120} = 12
    F12: GradBoost      n_est in {100,300} x depth in {3,5} x lr in {0.05,0.1} x L in {60,120} = 16
    F13: MLP            hidden in {(64,),(64,32)} x alpha in {0.0001,0.001} x L in {60,120} = 8
    Total: 36
    """
    algorithms = []

    # F11: Random Forest (12)
    for n_est in [100, 300]:
        for max_d in [5, 10, None]:
            for L in [60, 120]:
                depth_str = str(max_d) if max_d is not None else "None"
                name = f"RF_n{n_est}_d{depth_str}_L{L}"
                algorithms.append(Tier3Algorithm(
                    family="RandomForest",
                    model_class=RandomForestRegressor,
                    model_params={
                        "n_estimators": n_est,
                        "max_depth": max_d,
                        "random_state": 42,
                        "n_jobs": -1,
                    },
                    lookback=L,
                    name=name,
                ))

    # F12: Gradient Boosting (16)
    for n_est in [100, 300]:
        for max_d in [3, 5]:
            for lr in [0.05, 0.1]:
                for L in [60, 120]:
                    name = f"GBM_n{n_est}_d{max_d}_lr{lr}_L{L}"
                    algorithms.append(Tier3Algorithm(
                        family="GradientBoosting",
                        model_class=GradientBoostingRegressor,
                        model_params={
                            "n_estimators": n_est,
                            "max_depth": max_d,
                            "learning_rate": lr,
                            "random_state": 42,
                        },
                        lookback=L,
                        name=name,
                    ))

    # F13: MLP (8)
    for hidden in [(64,), (64, 32)]:
        for alpha in [0.0001, 0.001]:
            for L in [60, 120]:
                hidden_str = "x".join(str(h) for h in hidden)
                name = f"MLP_h{hidden_str}_a{alpha}_L{L}"
                algorithms.append(Tier3Algorithm(
                    family="MLP",
                    model_class=MLPRegressor,
                    model_params={
                        "hidden_layer_sizes": hidden,
                        "alpha": alpha,
                        "max_iter": 500,
                        "random_state": 42,
                        "early_stopping": True,
                        "validation_fraction": 0.1,
                    },
                    lookback=L,
                    name=name,
                ))

    n_rf  = 2 * 3 * 2
    n_gbm = 2 * 2 * 2 * 2
    n_mlp = 2 * 2 * 2
    print(
        f"Built K={len(algorithms)} Tier 3 algorithms "
        f"(F11=RF:{n_rf}, F12=GBM:{n_gbm}, F13=MLP:{n_mlp})"
    )
    return algorithms
