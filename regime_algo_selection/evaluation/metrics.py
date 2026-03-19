# evaluation/metrics.py — All evaluation metrics

import numpy as np
import pandas as pd

from regime_algo_selection.evaluation.backtest import BacktestResult


def _cumulative_wealth(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def compute_all_metrics(result: BacktestResult) -> dict:
    """
    Compute a comprehensive set of portfolio performance metrics.

    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    dict of scalar metrics (suitable for display as a table row).
    """
    r = result.net_returns.dropna()
    r_gross = result.portfolio_returns.dropna()

    T = len(r)
    ann_factor = 252

    # Cumulative wealth
    wealth = _cumulative_wealth(r)
    cum_return = float(wealth.iloc[-1] - 1) if len(wealth) > 0 else np.nan

    # Annualised return (geometric)
    if T > 0:
        ann_return = float((1 + cum_return) ** (ann_factor / T) - 1)
    else:
        ann_return = np.nan

    # Annualised volatility
    ann_vol = float(r.std() * np.sqrt(ann_factor)) if T > 1 else np.nan

    # Sharpe ratio (rf = 0)
    sharpe = float(ann_return / ann_vol) if ann_vol and ann_vol > 1e-12 else np.nan

    # Sortino ratio
    downside = r[r < 0]
    downside_dev = float(downside.std() * np.sqrt(ann_factor)) if len(downside) > 1 else np.nan
    sortino = float(ann_return / downside_dev) if downside_dev and downside_dev > 1e-12 else np.nan

    # Maximum drawdown
    if len(wealth) > 0:
        rolling_max = wealth.cummax()
        drawdown = (wealth - rolling_max) / rolling_max
        max_dd = float(drawdown.min())
    else:
        max_dd = np.nan

    # Turnover
    w_hist = result.weights_history
    turnover = w_hist.diff().abs().sum(axis=1)
    total_turnover  = float(turnover.sum())
    avg_turnover    = float(turnover.mean())

    # Total switching cost
    total_switch_cost = float(result.switching_costs.sum())

    # Regime accuracy
    regime_acc = np.nan
    if len(result.regime_predictions) > 0 and len(result.regime_true) > 0:
        common = result.regime_predictions.index.intersection(result.regime_true.index)
        pred = result.regime_predictions.loc[common].values
        true = result.regime_true.loc[common].values
        valid = np.isfinite(pred) & np.isfinite(true)
        if valid.sum() > 0:
            regime_acc = float((pred[valid] == true[valid]).mean())

    return {
        "cumulative_return"   : round(cum_return  * 100, 2),   # %
        "ann_return"          : round(ann_return   * 100, 2),   # %
        "ann_volatility"      : round(ann_vol      * 100, 2),   # %
        "sharpe_ratio"        : round(sharpe,         4),
        "sortino_ratio"       : round(sortino,        4),
        "max_drawdown"        : round(max_dd       * 100, 2),   # %
        "total_turnover"      : round(total_turnover,  2),
        "avg_daily_turnover"  : round(avg_turnover,    4),
        "total_switching_cost": round(total_switch_cost * 100, 4),  # bps
        "regime_accuracy"     : round(regime_acc,      4) if not np.isnan(regime_acc) else np.nan,
        "n_days"              : T,
    }


def per_regime_metrics(result: BacktestResult) -> pd.DataFrame:
    """
    Compute per-regime performance statistics.

    Returns a DataFrame: rows = regimes, cols = metrics.
    """
    rows = []
    for regime in [1, 2, 3, 4]:
        mask = result.regime_true == regime
        r_reg = result.net_returns.loc[mask].dropna()
        n = len(r_reg)
        if n == 0:
            rows.append({"regime": regime, "n_days": 0})
            continue

        ann = float(r_reg.mean() * 252 * 100)
        vol = float(r_reg.std() * np.sqrt(252) * 100)
        sr  = float(r_reg.mean() / r_reg.std() * np.sqrt(252)) if r_reg.std() > 1e-12 else np.nan

        # Most selected algorithm in this regime
        algo_in_regime = result.algorithm_selections.loc[mask]
        best_algo = algo_in_regime.value_counts().index[0] if len(algo_in_regime) > 0 else "N/A"

        rows.append({
            "regime"       : regime,
            "n_days"       : n,
            "ann_return_%" : round(ann, 2),
            "ann_vol_%"    : round(vol, 2),
            "sharpe"       : round(sr,  4),
            "top_algo"     : best_algo,
        })

    return pd.DataFrame(rows).set_index("regime")


def print_metrics_table(metrics_dict: dict) -> None:
    """
    Print a side-by-side comparison of multiple strategies.

    Parameters
    ----------
    metrics_dict : dict  {label: metrics_dict}
    """
    strategies = list(metrics_dict.keys())
    metric_keys = [
        "cumulative_return", "ann_return", "ann_volatility",
        "sharpe_ratio", "sortino_ratio", "max_drawdown",
        "total_turnover", "total_switching_cost", "regime_accuracy", "n_days",
    ]
    labels = [
        "Cum. Return (%)", "Ann. Return (%)", "Ann. Volatility (%)",
        "Sharpe Ratio", "Sortino Ratio", "Max Drawdown (%)",
        "Total Turnover", "Switch Cost (bps)", "Regime Accuracy", "# Days",
    ]

    col_w = 22
    header = f"{'Metric':<28}" + "".join(f"{s:>{col_w}}" for s in strategies)
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for key, label in zip(metric_keys, labels):
        row = f"{label:<28}"
        for s in strategies:
            val = metrics_dict[s].get(key, "N/A")
            if isinstance(val, float) and not np.isnan(val):
                row += f"{val:>{col_w}.4f}"
            elif isinstance(val, int):
                row += f"{val:>{col_w}}"
            else:
                row += f"{'N/A':>{col_w}}"
        print(row)
    print(sep)
