# data/loader.py — Download & cache price data + VIX

import os
import pandas as pd
import yfinance as yf

from regime_algo_selection.config import (
    ASSETS, START_DATE, END_DATE, DATA_DIR
)

PRICES_CACHE = os.path.join(DATA_DIR, "prices.csv")
VIX_CACHE    = os.path.join(DATA_DIR, "vix.csv")


def _download_prices() -> pd.DataFrame:
    """Download adjusted close prices for all assets via yfinance."""
    print(f"Downloading prices for {ASSETS} from {START_DATE} to {END_DATE}...")
    raw = yf.download(
        ASSETS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )
    # yfinance returns MultiIndex columns (field, ticker) when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = ASSETS
    prices = prices[ASSETS]  # ensure column order
    return prices


def _download_vix() -> pd.Series:
    """Download VIX daily close."""
    print(f"Downloading VIX from {START_DATE} to {END_DATE}...")
    raw = yf.download(
        "^VIX",
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        vix = raw["Close"].squeeze()
    else:
        vix = raw["Close"]
    vix.name = "VIX"
    return vix


def load_data(force_download: bool = False) -> dict:
    """
    Load (or download & cache) price data and VIX.

    Returns
    -------
    dict with keys:
        "prices": pd.DataFrame (index=date, columns=ASSETS)
        "vix":    pd.Series   (index=date, values=VIX close)
    """
    if not force_download and os.path.exists(PRICES_CACHE) and os.path.exists(VIX_CACHE):
        print("Loading cached data...")
        prices = pd.read_csv(PRICES_CACHE, index_col=0, parse_dates=True)
        vix    = pd.read_csv(VIX_CACHE,    index_col=0, parse_dates=True).squeeze()
        vix.name = "VIX"
    else:
        prices = _download_prices()
        vix    = _download_vix()

        # Align to common dates
        common_index = prices.index.intersection(vix.index)
        prices = prices.loc[common_index]
        vix    = vix.loc[common_index]

        # Forward-fill missing values, then drop remaining NaNs
        prices = prices.ffill().dropna()
        vix    = vix.ffill().dropna()

        # Restrict to common dates after cleaning
        common_index = prices.index.intersection(vix.index)
        prices = prices.loc[common_index]
        vix    = vix.loc[common_index]

        # Cache to disk
        os.makedirs(DATA_DIR, exist_ok=True)
        prices.to_csv(PRICES_CACHE)
        vix.to_frame().to_csv(VIX_CACHE)
        print(f"Data cached to {DATA_DIR}")

    print(
        f"Loaded data: {len(prices)} trading days "
        f"({prices.index[0].date()} to {prices.index[-1].date()})"
    )
    return {"prices": prices, "vix": vix}
