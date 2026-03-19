"""
01_build_dataset.py -- Regime Classifier v2: Build Feature Matrix + Binary Shift Labels

This script:
  1. Loads price and VIX data from cache.
  2. Computes the existing 4-class regime labels (deterministic VIX thresholds).
  3. Computes the binary shift label: "will regime change within next H=10 days?"
     Uses the any(...) formulation: label=1 if ANY day in [t+1, t+H] differs from regime_t.
  4. Computes 10 features (VIX-based + SPY-based, no look-ahead bias).
  5. Saves the cleaned dataset to results/regime_classifier_v2/dataset.csv.

Run from project root (Implementierung1/):
  python experiments/regime_classifier_v2/01_build_dataset.py
"""

import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup: allow running from project root or script directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from regime_algo_selection.config import REGIME_THRESHOLDS, DATA_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
H = 10   # Shift label horizon: check if any of the next H trading days differs
L = 20   # Primary lookback window for rolling statistics

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "regime_classifier_v2")
os.makedirs(RESULTS_DIR, exist_ok=True)

PRICES_CACHE = os.path.join(DATA_DIR, "prices.csv")
VIX_CACHE    = os.path.join(DATA_DIR, "vix.csv")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data() -> tuple:
    """Load prices (SPY, TLT, GLD, EFA, VNQ) and VIX from cache CSV files."""
    print("Loading cached price and VIX data...")
    prices = pd.read_csv(PRICES_CACHE, index_col=0, parse_dates=True)
    vix    = pd.read_csv(VIX_CACHE,    index_col=0, parse_dates=True).squeeze()
    vix.name = "VIX"
    print(f"  Prices : {len(prices)} days  "
          f"({prices.index[0].date()} to {prices.index[-1].date()})")
    print(f"  VIX    : {len(vix)} days")
    return prices, vix


# ---------------------------------------------------------------------------
# Regime labels (identical thresholds as the rest of the codebase)
# ---------------------------------------------------------------------------

def compute_regime_labels(vix: pd.Series) -> pd.Series:
    """
    Apply the deterministic mapping g(v_t):
        v_t <= 15       -> regime 1 (Calm)
        15 < v_t <= 20  -> regime 2 (Normal)
        20 < v_t <= 30  -> regime 3 (Tense)
        v_t > 30        -> regime 4 (Crisis)

    Thresholds imported from config.REGIME_THRESHOLDS = [15, 20, 30].
    Do NOT change these -- they must match the rest of the pipeline.
    """
    t1, t2, t3 = REGIME_THRESHOLDS  # 15, 20, 30
    labels = pd.cut(
        vix,
        bins=[-float("inf"), t1, t2, t3, float("inf")],
        labels=[1, 2, 3, 4],
    ).astype(int)
    labels.name = "regime"
    return labels


# ---------------------------------------------------------------------------
# Binary shift label
# ---------------------------------------------------------------------------

def compute_binary_shift_label(regime: pd.Series, h: int) -> pd.Series:
    """
    For each day t, assign:
        label_t = 1  if any(regime[t+1 : t+H+1] != regime[t])
        label_t = 0  otherwise
        label_t = NaN for the last h rows (no complete future window)

    The any(...) formulation catches transient shifts that revert within the window.
    Example: Calm -> Tense -> Calm within 10 days gives label=1, not 0.

    No look-ahead bias: the label uses future regime values, but labels are only
    used as training targets, never as input features.
    """
    regime_arr = regime.values
    n          = len(regime_arr)
    labels     = np.full(n, np.nan)

    for i in range(n - h):
        today_regime = regime_arr[i]
        future       = regime_arr[i + 1 : i + h + 1]   # next h days (exclusive today)
        labels[i]    = 1.0 if np.any(future != today_regime) else 0.0

    # The last h rows have no complete future window -> NaN (dropped later)
    return pd.Series(labels, index=regime.index, name="shift_label")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_features(prices: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """
    Compute the 10-feature matrix.  All features use data up to and including
    time t only.  No look-ahead bias.

    Features:
        VIX_MA20           - 20-day moving average of VIX (level proxy)
        z_VIX              - z-score: (VIX - VIX_MA20) / std_20  [outlier detection]
        delta_VIX          - daily VIX change: VIX_t - VIX_{t-1}
        VIX_slope_5        - 5-day VIX trend: VIX_t - VIX_{t-5}
        VIX_slope_20       - 20-day VIX trend: VIX_t - VIX_{t-20}
        VIX_rolling_std_10 - VIX instability: rolling 10-day std
        max_VIX_window     - 20-day rolling max of VIX
        min_VIX_window     - 20-day rolling min of VIX
        SPY_return_5       - cumulative 5-day SPY return (market momentum)
        vol_ratio          - rolling_vol_5 / rolling_vol_60  (short/long vol ratio)

    Note: VIX_rolling_std_20 (= std_VIX_window) is used only for z_VIX calculation
    and is NOT included as a separate feature to avoid redundancy.
    """
    spy         = prices["SPY"]
    spy_returns = spy.pct_change()

    # --- VIX level ---
    VIX_MA20           = vix.rolling(20).mean()
    VIX_rolling_std_20 = vix.rolling(20).std()

    # --- VIX outlier: z-score ---
    # z_VIX > 2 signals a potential VIX breakout (most important new feature)
    z_VIX = (vix - VIX_MA20) / VIX_rolling_std_20

    # --- VIX dynamics ---
    delta_VIX          = vix - vix.shift(1)           # daily change
    VIX_slope_5        = vix - vix.shift(5)            # 5-day trend
    VIX_slope_20       = vix - vix.shift(20)           # 20-day trend (longer context)
    VIX_rolling_std_10 = vix.rolling(10).std()         # VIX's own instability

    # --- VIX window summaries ---
    max_VIX_window = vix.rolling(20).max()
    min_VIX_window = vix.rolling(20).min()
    # std_VIX_window == VIX_rolling_std_20 -> include only once (already in z_VIX)

    # --- SPY return-based ---
    SPY_return_5   = spy / spy.shift(5) - 1            # cumulative 5-day return
    rolling_vol_5  = spy_returns.rolling(5).std()
    rolling_vol_60 = spy_returns.rolling(60).std()     # max warm-up: 60 days
    vol_ratio      = rolling_vol_5 / rolling_vol_60    # short vs long-term volatility

    features = pd.DataFrame({
        "VIX_MA20":            VIX_MA20,
        "z_VIX":               z_VIX,
        "delta_VIX":           delta_VIX,
        "VIX_slope_5":         VIX_slope_5,
        "VIX_slope_20":        VIX_slope_20,
        "VIX_rolling_std_10":  VIX_rolling_std_10,
        "max_VIX_window":      max_VIX_window,
        "min_VIX_window":      min_VIX_window,
        "SPY_return_5":        SPY_return_5,
        "vol_ratio":           vol_ratio,
    }, index=vix.index)

    return features  # 10 columns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load raw data
    prices, vix = load_raw_data()

    # 2. Compute 4-class regime labels
    regime = compute_regime_labels(vix)
    print("\nRegime distribution (4-class):")
    for r, cnt in regime.value_counts().sort_index().items():
        print(f"  Regime {r}: {cnt:5d} days  ({100 * cnt / len(regime):.1f}%)")

    # 3. Compute binary shift labels (horizon H=10)
    shift_label = compute_binary_shift_label(regime, H)
    valid_mask  = shift_label.notna()
    n_valid     = int(valid_mask.sum())
    n_shift     = int((shift_label[valid_mask] == 1).sum())
    n_noshift   = int((shift_label[valid_mask] == 0).sum())

    print(f"\nBinary shift label (H={H}, any-formulation):")
    print(f"  label=0 (no shift) : {n_noshift:5d} days  ({100 * n_noshift / n_valid:.1f}%)")
    print(f"  label=1 (shift)    : {n_shift:5d} days  ({100 * n_shift / n_valid:.1f}%)")
    print(f"  NaN (last {H} rows) : {len(shift_label) - n_valid} days  (dropped)")

    # 4. Compute features
    features = compute_features(prices, vix)
    print(f"\nFeature matrix shape (before NaN drop): {features.shape}")

    # 5. Assemble dataset and drop NaN rows
    #    Sources of NaN:
    #      - First ~60 rows: rolling window warm-up (rolling_vol_60 needs 60 days)
    #      - Last H rows: no complete future label window
    dataset              = features.copy()
    dataset["regime"]      = regime
    dataset["shift_label"] = shift_label

    n_before = len(dataset)
    dataset  = dataset.dropna()
    n_after  = len(dataset)

    print(f"\nDataset after NaN drop: {n_before} -> {n_after} rows "
          f"(dropped {n_before - n_after})")
    print(f"  Date range : {dataset.index[0].date()} to {dataset.index[-1].date()}")

    # Verify class distribution in cleaned dataset
    n_clean_shift   = int((dataset["shift_label"] == 1).sum())
    n_clean_noshift = int((dataset["shift_label"] == 0).sum())
    n_clean_total   = n_clean_shift + n_clean_noshift
    print(f"\nClass distribution in final dataset:")
    print(f"  label=0 : {n_clean_noshift:5d} days  ({100 * n_clean_noshift / n_clean_total:.1f}%)")
    print(f"  label=1 : {n_clean_shift:5d} days  ({100 * n_clean_shift / n_clean_total:.1f}%)")

    # 6. Save dataset
    out_path = os.path.join(RESULTS_DIR, "dataset.csv")
    dataset.to_csv(out_path)
    print(f"\nDataset saved to : {out_path}")

    # Save overall class distribution CSV
    dist_df = pd.DataFrame({
        "label":       [0, 1],
        "description": ["no_shift", "shift"],
        "count":       [n_clean_noshift, n_clean_shift],
        "pct":         [round(100 * n_clean_noshift / n_clean_total, 2),
                        round(100 * n_clean_shift   / n_clean_total, 2)],
    })
    dist_path = os.path.join(RESULTS_DIR, "class_distribution_overall.csv")
    dist_df.to_csv(dist_path, index=False)
    print(f"Class distribution saved to : {dist_path}")

    print("\nDone. Run 02_train_evaluate.py next.")


if __name__ == "__main__":
    main()
