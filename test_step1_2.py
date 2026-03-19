"""
Test script for Step 1 (config) and Step 2 (data pipeline).
Run from the Implementierung1 directory.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

# == Step 1: Config ==
print("=" * 60)
print("STEP 1: Testing config.py")
print("=" * 60)

from regime_algo_selection.config import (
    ASSETS, N_ASSETS, START_DATE, END_DATE, TRAIN_END, VAL_END,
    REGIME_THRESHOLDS, REGIME_NAMES, N_REGIMES, KAPPA,
    LOOKBACKS_COV, LOOKBACKS_MOM, LOOKBACKS_TREND, TREND_BETAS, RISK_AVERSIONS,
    DATA_DIR, RESULTS_DIR,
)

assert N_ASSETS == 5, f"Expected 5 assets, got {N_ASSETS}"
assert len(ASSETS) == 5
assert N_REGIMES == 4
assert KAPPA == 0.001
assert os.path.isdir(DATA_DIR), f"DATA_DIR not created: {DATA_DIR}"
assert os.path.isdir(RESULTS_DIR), f"RESULTS_DIR not created: {RESULTS_DIR}"
print(f"  Assets:      {ASSETS}")
print(f"  Date range:  {START_DATE} to {END_DATE}")
print(f"  Train/Val:   train<='{TRAIN_END}', val<='{VAL_END}'")
print(f"  Regimes:     {REGIME_NAMES}")
print(f"  KAPPA:       {KAPPA}")
print("  Config OK\n")

# == Step 2a: Loader ==
print("=" * 60)
print("STEP 2a: Testing data/loader.py")
print("=" * 60)

from regime_algo_selection.data.loader import load_data

data = load_data()
prices = data["prices"]
vix    = data["vix"]

# Basic shape / type checks
assert isinstance(prices, pd.DataFrame), "prices must be a DataFrame"
assert isinstance(vix, pd.Series), "vix must be a Series"
assert list(prices.columns) == ASSETS, f"Price columns mismatch: {prices.columns.tolist()}"
assert pd.api.types.is_datetime64_any_dtype(prices.index), "prices index must be DatetimeIndex"
assert prices.isnull().sum().sum() == 0, "prices contain NaNs after cleaning"
assert vix.isnull().sum() == 0, "vix contains NaNs after cleaning"
assert len(prices) > 1000, f"Too few rows: {len(prices)}"
assert len(prices) == len(vix), "prices and vix have different lengths"

print(f"  prices shape: {prices.shape}  ({prices.index[0].date()} to {prices.index[-1].date()})")
print(f"  vix   shape:  {vix.shape}")
print(f"  prices sample (last 3):\n{prices.tail(3).to_string()}")
print(f"  vix   sample (last 3): {vix.tail(3).values}")
print("  Loader OK\n")

# == Step 2b: Features ==
print("=" * 60)
print("STEP 2b: Testing data/features.py")
print("=" * 60)

from regime_algo_selection.data.features import (
    compute_returns, compute_asset_features, compute_vix_features
)

# --- compute_returns ---
returns = compute_returns(prices)
assert isinstance(returns, pd.DataFrame)
assert list(returns.columns) == ASSETS
assert len(returns) == len(prices) - 1, (
    f"Returns should have 1 fewer row than prices. Got {len(returns)} vs {len(prices)}"
)
valid_returns = returns.dropna()
assert len(valid_returns) > 1000

print(f"  returns shape: {returns.shape}")
print(f"  returns (last 3):\n{returns.tail(3).to_string()}")

# Forward-return sanity check
expected = (prices.iloc[1] - prices.iloc[0]) / prices.iloc[0]
actual   = returns.iloc[0]
assert np.allclose(actual.values, expected.values, rtol=1e-5), (
    f"Forward return mismatch:\n  expected={expected.values}\n  actual={actual.values}"
)
print("  compute_returns: forward-return sanity check passed")

# --- compute_vix_features ---
vix_feats = compute_vix_features(vix)
expected_cols = ["vix_prev", "vix_change_1d", "vix_change_5d",
                 "vix_ma5", "vix_ma20", "vix_std20", "vix_relative"]
assert list(vix_feats.columns) == expected_cols, (
    f"VIX feature columns: {vix_feats.columns.tolist()}"
)
assert len(vix_feats) == len(vix)

# All features must be lagged: vix_prev at date d == vix at d-1
dates = vix.index
for i in range(1, min(10, len(dates))):
    d      = dates[i]
    d_prev = dates[i - 1]
    assert abs(vix_feats.loc[d, "vix_prev"] - vix.loc[d_prev]) < 1e-8, (
        f"vix_prev not properly lagged at {d}"
    )
print("  compute_vix_features: lag check passed")

vix_feats_valid = vix_feats.dropna()
assert len(vix_feats_valid) > 1000, f"Too few valid VIX feature rows: {len(vix_feats_valid)}"
print(f"  vix_features shape: {vix_feats.shape}, valid rows: {len(vix_feats_valid)}")
print(f"  vix_features (last 3):\n{vix_feats.tail(3).to_string()}")

# --- compute_asset_features ---
asset_feats = compute_asset_features(prices)
assert isinstance(asset_feats.columns, pd.MultiIndex), "Expected MultiIndex columns"
assert set(asset_feats.columns.get_level_values(0)) == set(ASSETS)
expected_feat_names = {
    "ret_1d","ret_5d","ret_20d","ret_60d",
    "vol_20d","vol_60d","mom_20d","mom_60d","mom_120d"
}
actual_feat_names = set(asset_feats.columns.get_level_values(1))
assert actual_feat_names == expected_feat_names, (
    f"Unexpected feature names: {actual_feat_names}"
)

# Lag check for ret_1d
daily_r = prices.pct_change()
for i in range(1, min(5, len(dates))):
    d      = dates[i]
    d_prev = dates[i - 1]
    for asset in ASSETS:
        expected_r1 = daily_r.loc[d_prev, asset]
        actual_r1   = asset_feats.loc[d, (asset, "ret_1d")]
        if not np.isnan(expected_r1) and not np.isnan(actual_r1):
            assert abs(actual_r1 - expected_r1) < 1e-8, (
                f"ret_1d lag mismatch for {asset} at {d}"
            )
print("  compute_asset_features: lag check passed")
print(f"  asset_features shape: {asset_feats.shape}")
print(f"  asset_features columns (first 6): {asset_feats.columns.tolist()[:6]} ...")

# == Summary ==
print("\n" + "=" * 60)
print("ALL TESTS PASSED -- Steps 1 & 2 are correct.")
print("=" * 60)
