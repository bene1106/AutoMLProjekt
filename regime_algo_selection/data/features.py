# data/features.py — Feature engineering (asset features + VIX features)

import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple daily forward returns.

    r_{t+1,i} = (P_{t+1,i} - P_{t,i}) / P_{t,i}

    The return stored at date t is the return FROM t TO t+1.
    Returns have the same date index as prices; the last row is NaN.
    """
    returns = prices.pct_change(fill_method=None).shift(-1)
    # Drop the final row (no return known for the last price date)
    returns = returns.iloc[:-1]
    return returns


def compute_asset_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lagged asset-level features.

    All features are properly lagged so they only use information
    available at decision time t (i.e., prices up to t-1).

    Returns
    -------
    pd.DataFrame with MultiIndex columns (asset, feature) and index=date.
    """
    daily_returns = prices.pct_change(fill_method=None)

    feature_dict = {}

    for asset in prices.columns:
        r = daily_returns[asset]
        p = prices[asset]

        # Lagged returns (shift by 1 so value at t uses data up to t-1)
        feature_dict[(asset, "ret_1d")]  = r.shift(1)
        feature_dict[(asset, "ret_5d")]  = p.shift(1).pct_change(5)
        feature_dict[(asset, "ret_20d")] = p.shift(1).pct_change(20)
        feature_dict[(asset, "ret_60d")] = p.shift(1).pct_change(60)

        # Rolling volatility (annualized), lagged by 1
        feature_dict[(asset, "vol_20d")] = r.shift(1).rolling(20).std() * np.sqrt(252)
        feature_dict[(asset, "vol_60d")] = r.shift(1).rolling(60).std() * np.sqrt(252)

        # Momentum: cumulative return over window (using lagged prices)
        feature_dict[(asset, "mom_20d")]  = p.shift(1).pct_change(20)
        feature_dict[(asset, "mom_60d")]  = p.shift(1).pct_change(60)
        feature_dict[(asset, "mom_120d")] = p.shift(1).pct_change(120)

    df = pd.DataFrame(feature_dict)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["asset", "feature"])
    return df


def compute_vix_features(vix: pd.Series) -> pd.DataFrame:
    """
    Compute lagged VIX features z_t used for regime estimation.

    CRITICAL: All features are lagged — only use VIX information
    available at decision time t (data up to t-1).

    Returns
    -------
    pd.DataFrame with index=date, columns=feature names.
    """
    # v_{t-1}: yesterday's VIX close (primary regime signal)
    vix_prev       = vix.shift(1)

    # 1-day and 5-day VIX change (lagged)
    vix_change_1d  = vix.shift(1) - vix.shift(2)
    vix_change_5d  = vix.shift(1) - vix.shift(6)

    # Moving averages of VIX (lagged by 1)
    vix_ma5        = vix.shift(1).rolling(5).mean()
    vix_ma20       = vix.shift(1).rolling(20).mean()

    # Rolling std of VIX (lagged by 1)
    vix_std20      = vix.shift(1).rolling(20).std()

    # VIX relative to its 20-day moving average (lagged)
    vix_relative   = vix.shift(1) / vix_ma20

    features = pd.DataFrame({
        "vix_prev":      vix_prev,
        "vix_change_1d": vix_change_1d,
        "vix_change_5d": vix_change_5d,
        "vix_ma5":       vix_ma5,
        "vix_ma20":      vix_ma20,
        "vix_std20":     vix_std20,
        "vix_relative":  vix_relative,
    })

    return features


def compute_cross_asset_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lagged cross-asset stress indicators for regime estimation (Feature Set F).

    CRITICAL: All features use only t-1 information (lagged by 1 day).

    Features
    --------
    spy_tlt_corr_20d     : 20-day rolling correlation of SPY and TLT daily returns
    spy_gld_corr_20d     : 20-day rolling correlation of SPY and GLD daily returns
    cross_asset_disp     : cross-sectional std of daily returns across all 5 ETFs
    spy_drawdown_20d     : SPY price / SPY 20-day rolling max - 1
    tlt_spy_spread_1d    : TLT 1d return - SPY 1d return (flight-to-bonds signal)
    gld_spy_spread_1d    : GLD 1d return - SPY 1d return (flight-to-gold signal)
    avg_abs_return_1d    : mean of |1d return| across all 5 ETFs (turbulence proxy)

    Returns
    -------
    pd.DataFrame with index=date, columns=feature names.
    """
    daily_r = prices.pct_change(fill_method=None)

    # Rolling 20-day correlations (computed on lagged returns, then shift result by 1)
    spy_r = daily_r["SPY"]
    tlt_r = daily_r["TLT"]
    gld_r = daily_r["GLD"]

    spy_tlt_corr = spy_r.rolling(20).corr(tlt_r).shift(1)
    spy_gld_corr = spy_r.rolling(20).corr(gld_r).shift(1)

    # Cross-sectional dispersion: std of all-asset returns per day, lagged
    cross_disp = daily_r.std(axis=1).shift(1)

    # SPY drawdown from 20-day high (price-based, lagged)
    spy_max_20d      = prices["SPY"].rolling(20).max()
    spy_drawdown_20d = (prices["SPY"] / spy_max_20d - 1).shift(1)

    # 1-day return spreads (lagged)
    tlt_spy_spread = (daily_r["TLT"] - daily_r["SPY"]).shift(1)
    gld_spy_spread = (daily_r["GLD"] - daily_r["SPY"]).shift(1)

    # Average absolute return (market turbulence proxy), lagged
    avg_abs_ret = daily_r.abs().mean(axis=1).shift(1)

    features = pd.DataFrame({
        "spy_tlt_corr_20d" : spy_tlt_corr,
        "spy_gld_corr_20d" : spy_gld_corr,
        "cross_asset_disp" : cross_disp,
        "spy_drawdown_20d" : spy_drawdown_20d,
        "tlt_spy_spread_1d": tlt_spy_spread,
        "gld_spy_spread_1d": gld_spy_spread,
        "avg_abs_return_1d": avg_abs_ret,
    })

    return features
