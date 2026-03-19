# regimes/classifier.py — Regime classifier (Logistic Regression / RF / XGBoost)

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from regime_algo_selection.config import N_REGIMES, RANDOM_SEED


class RegimeClassifier:
    """
    Classifies market regimes from lagged VIX features z_t.

    Parameters
    ----------
    model_type : str
        One of "logistic_regression", "random_forest".
    """

    def __init__(self, model_type: str = "logistic_regression"):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self._fitted = False

        if model_type == "logistic_regression":
            self.model = LogisticRegression(
                max_iter=1000,
                C=1.0,
                multi_class="multinomial",
                solver="lbfgs",
                random_state=RANDOM_SEED,
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

    # ------------------------------------------------------------------
    def _prepare(self, X: pd.DataFrame) -> np.ndarray:
        """Drop NaN rows and return (array, valid_index)."""
        valid = X.dropna()
        return valid, valid.index

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RegimeClassifier":
        """
        Train the classifier.

        Parameters
        ----------
        X_train : pd.DataFrame
            VIX features z_t (from compute_vix_features), aligned to training dates.
        y_train : pd.Series
            True regime labels s*_t for SAME day t (known at training time).
        """
        X_valid, idx = self._prepare(X_train)
        y_valid = y_train.loc[idx]

        # Align
        common = X_valid.index.intersection(y_valid.index)
        X_arr = self.scaler.fit_transform(X_valid.loc[common])
        y_arr = y_valid.loc[common].values

        self.model.fit(X_arr, y_arr)
        self._fitted = True
        self._classes = self.model.classes_
        print(
            f"RegimeClassifier ({self.model_type}) trained on {len(common)} samples. "
            f"Classes: {list(self._classes)}"
        )
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return p_hat_t: probability distribution over 4 regimes.

        For rows with NaN features, returns uniform probability (1/N_REGIMES).

        Returns
        -------
        pd.DataFrame: columns ["prob_regime_1", ..., "prob_regime_4"], index=X.index
        """
        assert self._fitted, "Call fit() first."

        col_names = [f"prob_regime_{r}" for r in range(1, N_REGIMES + 1)]
        result = pd.DataFrame(
            1.0 / N_REGIMES, index=X.index, columns=col_names
        )

        valid_mask = X.notna().all(axis=1)
        if valid_mask.any():
            X_arr = self.scaler.transform(X.loc[valid_mask])
            probs = self.model.predict_proba(X_arr)  # shape (n, n_classes)

            # Map model classes (may not start at 1) to columns
            for j, cls in enumerate(self._classes):
                col = f"prob_regime_{cls}"
                if col in result.columns:
                    result.loc[valid_mask, col] = probs[:, j]

        return result

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Return s_hat_t = argmax of predict_proba.

        Returns
        -------
        pd.Series: int regime labels, index=X.index
        """
        proba = self.predict_proba(X)
        # +1 because columns are named prob_regime_1..4 (1-indexed)
        pred = proba.values.argmax(axis=1) + 1
        return pd.Series(pred, index=X.index, name="regime_pred")

    # ------------------------------------------------------------------
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate classifier on test data.

        Returns
        -------
        dict with keys: accuracy, classification_report, confusion_matrix
        """
        y_pred = self.predict(X_test)

        # Align
        common = y_test.index.intersection(y_pred.index)
        y_true_arr = y_test.loc[common].values
        y_pred_arr = y_pred.loc[common].values

        # Only evaluate on rows where features were valid
        valid = X_test.loc[common].notna().all(axis=1)
        y_true_arr = y_true_arr[valid.values]
        y_pred_arr = y_pred_arr[valid.values]

        acc = accuracy_score(y_true_arr, y_pred_arr)
        report = classification_report(
            y_true_arr, y_pred_arr,
            labels=[1, 2, 3, 4],
            target_names=["Calm", "Normal", "Tense", "Crisis"],
            zero_division=0,
        )
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[1, 2, 3, 4])

        return {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm,
        }
