# data/loader_extended.py -- Extended data loader for Plan 3
#
# Downloads VIX from 2000-01-01 and 5-asset prices from their individual
# launch dates.  Returns clean 5-asset prices (from 2004-12 onward, when all
# ETFs are available) plus extended VIX (from 2000).
#
# Usage (standalone):
#   python -m regime_algo_selection.data.loader_extended

import os
import pandas as pd
import yfinance as yf

from regime_algo_selection.config import DATA_DIR

# --------------------------------------------------------------------------
# Cache paths (separate from the Plan 1/2 cache so we don't overwrite it)
# --------------------------------------------------------------------------
EXT_PRICES_CACHE = os.path.join(DATA_DIR, "prices_extended.csv")
EXT_VIX_CACHE    = os.path.join(DATA_DIR, "vix_extended.csv")

EXT_START  = "2000-01-01"
EXT_END    = "2024-12-31"
ALL_ASSETS = ["SPY", "TLT", "GLD", "EFA", "VNQ"]


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _download_prices_extended() -> pd.DataFrame:
    """Download all 5 ETFs from EXT_START.  Assets not yet listed will be NaN."""
    print(f"Downloading extended prices ({ALL_ASSETS}) from {EXT_START} to {EXT_END}...")
    raw = yf.download(
        ALL_ASSETS,
        start=EXT_START,
        end=EXT_END,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][ALL_ASSETS]
    else:
        prices = raw[["Close"]]
        prices.columns = ALL_ASSETS
    return prices


def _download_vix_extended() -> pd.Series:
    """Download VIX from EXT_START."""
    print(f"Downloading extended VIX from {EXT_START} to {EXT_END}...")
    raw = yf.download(
        "^VIX",
        start=EXT_START,
        end=EXT_END,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        vix = raw["Close"].squeeze()
    else:
        vix = raw["Close"]
    vix.name = "VIX"
    return vix


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def load_data_extended(force_download: bool = False) -> dict:
    """
    Load (or download & cache) extended price/VIX data.

    Returns
    -------
    dict:
        "prices"      -- pd.DataFrame: 5-asset clean prices from ~2004-12 onward
        "prices_full" -- pd.DataFrame: 5-asset prices from 2000 (NaN pre-launch)
        "vix"         -- pd.Series:    VIX from 2000-01-01 to 2024-12-31
    """
    if (
        not force_download
        and os.path.exists(EXT_PRICES_CACHE)
        and os.path.exists(EXT_VIX_CACHE)
    ):
        print("Loading cached extended data...")
        prices_full = pd.read_csv(EXT_PRICES_CACHE, index_col=0, parse_dates=True)
        vix = pd.read_csv(EXT_VIX_CACHE, index_col=0, parse_dates=True).squeeze()
        vix.name = "VIX"
    else:
        prices_full = _download_prices_extended()
        vix = _download_vix_extended()

        # Align to common trading-day index
        common = prices_full.index.intersection(vix.index)
        prices_full = prices_full.loc[common]
        vix = vix.loc[common].ffill().dropna()

        os.makedirs(DATA_DIR, exist_ok=True)
        prices_full.to_csv(EXT_PRICES_CACHE)
        vix.to_frame().to_csv(EXT_VIX_CACHE)
        print(f"Extended data cached to {DATA_DIR}")

    # 5-asset clean prices: forward-fill, then drop rows with any remaining NaN
    prices = prices_full.ffill().dropna()

    # Align VIX to at least prices start
    vix = vix.loc[vix.index >= vix.index[0]]

    _validate(prices, prices_full, vix)
    return {"prices": prices, "prices_full": prices_full, "vix": vix}


def _validate(prices: pd.DataFrame, prices_full: pd.DataFrame, vix: pd.Series) -> None:
    print(
        f"Extended VIX   : {len(vix)} days "
        f"({vix.index[0].date()} to {vix.index[-1].date()})"
    )
    print(
        f"5-asset prices : {len(prices)} days "
        f"({prices.index[0].date()} to {prices.index[-1].date()})"
    )
    # Expected earliest date for clean 5-asset data
    assert prices.index[0].year in (2004, 2005), (
        f"Unexpected start year for 5-asset prices: {prices.index[0].date()}"
    )
    # Check no NaN in clean prices
    assert prices.isna().sum().sum() == 0, "NaN values in cleaned prices!"
    print("Validation passed.")


# --------------------------------------------------------------------------
# Standalone entry-point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    data = load_data_extended(force_download=False)
    print("\nData shapes:")
    print(f"  prices      : {data['prices'].shape}")
    print(f"  prices_full : {data['prices_full'].shape}")
    print(f"  vix         : {data['vix'].shape}")
    print("\nFirst 5-asset date :", data["prices"].index[0].date())
    print("Last date          :", data["prices"].index[-1].date())
    print("VIX first date     :", data["vix"].index[0].date())
