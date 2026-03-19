# regimes/ground_truth.py — Deterministic VIX -> regime label mapping

import pandas as pd

from regime_algo_selection.config import REGIME_THRESHOLDS


def compute_regime_labels(vix: pd.Series) -> pd.Series:
    """
    Apply the deterministic mapping g(v_t):
        v_t <= 15       -> regime 1 (Calm)
        15 < v_t <= 20  -> regime 2 (Normal)
        20 < v_t <= 30  -> regime 3 (Tense)
        v_t > 30        -> regime 4 (Crisis)

    This is s*_t — the TRUE regime, known only after market close on day t.

    Returns
    -------
    pd.Series: index=date, values in {1, 2, 3, 4}
    """
    t1, t2, t3 = REGIME_THRESHOLDS  # 15, 20, 30

    labels = pd.cut(
        vix,
        bins=[-float("inf"), t1, t2, t3, float("inf")],
        labels=[1, 2, 3, 4],
    ).astype(int)

    labels.name = "regime"
    return labels


def compute_lagged_regime(regime_labels: pd.Series) -> pd.Series:
    """
    Return s*_{t-1}: yesterday's true regime.

    This is the naive persistence baseline for regime estimation —
    "assume today's regime equals yesterday's regime."
    """
    lagged = regime_labels.shift(1)
    lagged.name = "regime_lagged"
    return lagged
