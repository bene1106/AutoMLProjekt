# algorithms/stage0.py -- Stage 0: Pre-training for Tier 2 Algorithms
#
# Stage 0 must happen INSIDE each walk-forward fold, using only that
# fold's training data. NEVER trained on test data.
#
# Usage:
#   from regime_algo_selection.algorithms.stage0 import pretrain_tier2_algorithms
#   algorithms = pretrain_tier2_algorithms(algorithms, asset_features, returns, train_mask)

import numpy as np
import pandas as pd

from regime_algo_selection.algorithms.base import TrainablePortfolioAlgorithm


def build_training_matrix(
    asset_features: pd.DataFrame,
    returns: pd.DataFrame,
    train_start: str,
    train_end: str,
) -> tuple:
    """
    Build the (X_train, Y_train) matrices for Tier 2 pre-training.

    Parameters
    ----------
    asset_features : pd.DataFrame
        MultiIndex columns (asset, feature). Lagged features computed by
        data/features.py:compute_asset_features().
    returns : pd.DataFrame
        Forward returns (columns = assets). Shape: (T_all, N).
    train_start, train_end : str
        Date range for training.

    Returns
    -------
    X_train : np.ndarray, shape (T_train, n_features)
        Flattened asset features (sorted by MultiIndex level order).
    Y_train : np.ndarray, shape (T_train, N_assets)
        Next-period returns for each asset.
    """
    # Filter to training period
    feat_mask = (asset_features.index >= train_start) & (asset_features.index <= train_end)
    ret_mask  = (returns.index >= train_start)        & (returns.index <= train_end)

    feat_train = asset_features.loc[feat_mask]
    ret_train  = returns.loc[ret_mask]

    # Align on common dates
    common = feat_train.index.intersection(ret_train.index)
    if len(common) == 0:
        return np.empty((0, asset_features.shape[1])), np.empty((0, returns.shape[1]))

    X = feat_train.loc[common].values.astype(float)  # (T, n_features)
    Y = ret_train.loc[common].values.astype(float)    # (T, N_assets)

    return X, Y


def pretrain_tier2_algorithms(
    algorithms: list,
    asset_features: pd.DataFrame,
    returns: pd.DataFrame,
    train_start: str,
    train_end: str,
) -> list:
    """
    Pre-train all Tier 2 (TrainablePortfolioAlgorithm) algorithms.

    Tier 1 algorithms are left untouched. Tier 2 algorithms have fit()
    called with the training-period features and returns.

    Parameters
    ----------
    algorithms : list of PortfolioAlgorithm
        Mixed list of Tier 1 and Tier 2 algorithms.
    asset_features : pd.DataFrame
        Full-period asset features (MultiIndex columns).
    returns : pd.DataFrame
        Full-period forward returns.
    train_start, train_end : str
        Training period boundaries (inclusive).

    Returns
    -------
    list : same algorithms, with Tier 2 algorithms now fitted.
    """
    tier2 = [a for a in algorithms if isinstance(a, TrainablePortfolioAlgorithm)]
    if not tier2:
        return algorithms

    print(f"  Stage 0: pre-training {len(tier2)} Tier 2 algorithms "
          f"({train_start[:4]}-{train_end[:4]}) ...")

    X_train, Y_train = build_training_matrix(
        asset_features, returns, train_start, train_end
    )

    if len(X_train) == 0:
        print("  Stage 0: WARNING -- no training data found, skipping.")
        return algorithms

    n_fitted = 0
    for algo in tier2:
        try:
            algo.fit(X_train, Y_train)
            if algo.is_fitted:
                # Attach the FULL asset_features for fast O(log n) lookup during inference.
                # This avoids recomputing features from raw prices at each decision step.
                algo.attach_full_features(asset_features)
                n_fitted += 1
        except Exception as e:
            print(f"  Stage 0: fit failed for {algo.name}: {e}")

    print(f"  Stage 0 complete: {n_fitted}/{len(tier2)} Tier 2 algorithms fitted "
          f"(training rows: {len(X_train)}, features: {X_train.shape[1]})")
    return algorithms
