# evaluation/walk_forward.py -- Walk-Forward Validation Framework (Plan 3)
#
# Generates time-series folds and runs the full fit-backtest pipeline
# for each fold.  All algorithm evaluation uses the 5-asset universe
# (prices available from ~2004-12 onward).

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from regime_algo_selection.config import KAPPA, REGIME_NAMES, N_REGIMES
from regime_algo_selection.regimes.classifier import RegimeClassifier
from regime_algo_selection.regimes.ground_truth import (
    compute_regime_labels,
    compute_lagged_regime,
)
from regime_algo_selection.agents.reflex_agent import ReflexAgent, OracleAgent
from regime_algo_selection.evaluation.backtest import Backtester
from regime_algo_selection.evaluation.metrics import compute_all_metrics
from regime_algo_selection.algorithms.stage0 import pretrain_tier2_algorithms
from regime_algo_selection.algorithms.base import TrainablePortfolioAlgorithm


# ---------------------------------------------------------------------------
# Simple agent wrappers
# ---------------------------------------------------------------------------

class _SingleAlgoAgent:
    """Agent that always selects the same algorithm regardless of regime."""
    def __init__(self, algo):
        self._algo = algo
        self.mapping = {r: algo for r in range(1, N_REGIMES + 1)}

    def select(self, regime: int):
        return self._algo


# ---------------------------------------------------------------------------
# Fast per-algorithm evaluator (test period)
# Uses a sliding price window to avoid O(n^2) DataFrame slicing.
# ---------------------------------------------------------------------------

_MAX_LOOKBACK = 310   # slightly more than max algo lookback (252)


def _eval_algo_test(
    algo,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    test_dates: pd.DatetimeIndex,
    kappa: float = KAPPA,
) -> float:
    """
    Evaluate algorithm on the test period using a sliding price window.

    Returns annualised net Sharpe ratio (or -999 if fewer than 10 returns).
    """
    n = prices.shape[1]
    daily_rets: List[float] = []
    prev_w = np.ones(n) / n

    # Pre-index prices for fast positional lookup
    price_arr = prices.values           # shape (T, n)
    price_idx = prices.index

    for t in test_dates:
        # Find position of t in the full price index
        try:
            pos = price_idx.get_loc(t)
        except KeyError:
            continue

        start = max(0, pos - _MAX_LOOKBACK)
        prices_hist = prices.iloc[start:pos]  # prices BEFORE t

        if len(prices_hist) < 22:
            prev_w = np.ones(n) / n
            continue

        try:
            w = algo.compute_weights(prices_hist)
        except Exception:
            w = np.ones(n) / n

        w = np.where(np.isfinite(w), w, 0.0)
        w = np.clip(w, 0, None)
        total = w.sum()
        if total < 1e-12:
            w = np.ones(n) / n
        else:
            w /= total

        if t in returns.index:
            r = returns.loc[t].fillna(0).values
            gross = float(w @ r)
            cost = kappa * float(np.abs(w - prev_w).sum())
            daily_rets.append(gross - cost)

        prev_w = w

    if len(daily_rets) < 10:
        return -999.0

    arr = np.array(daily_rets)
    std = arr.std()
    if std < 1e-12:
        return float(arr.mean() * np.sqrt(252))
    return float((arr.mean() / std) * np.sqrt(252))


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold_id: int
    fold_spec: dict
    metrics_reflex: dict
    metrics_oracle: dict
    metrics_ew: dict
    algo_scores: dict          # {algo_name: test_period_net_sharpe}
    regime_accuracy: float
    reflex_mapping: dict       # {regime_int: algo_name}
    dominant_regime: str       # most frequent regime in test period
    regime_dist: dict          # {regime_name: day_count}
    all_scores: dict = None    # {regime_int: {algo_name: training_sharpe}} from agent.all_scores


@dataclass
class WalkForwardResult:
    folds: List[FoldResult]
    summary_df: pd.DataFrame   # one row per fold


# ---------------------------------------------------------------------------
# Walk-Forward Validator
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """
    Generates walk-forward folds and runs the full pipeline per fold.

    Parameters
    ----------
    train_years  : length of training window in years
    test_years   : length of test window in years
    step_years   : number of years to advance each fold
    min_test_start : earliest allowed test start date (ensures 5 assets available)
    """

    def __init__(
        self,
        train_years: int = 8,
        test_years: int = 1,
        step_years: int = 1,
        min_test_start: str = "2013-01-01",
    ):
        self.train_years = train_years
        self.test_years = test_years
        self.step_years = step_years
        self.min_test_start = pd.Timestamp(min_test_start)

    # ------------------------------------------------------------------
    def generate_folds(
        self,
        data_end: str = "2024-12-31",
    ) -> List[dict]:
        """
        Generate fold specifications.

        Each fold dict has keys:
          fold, train_start, train_end, test_start, test_end
        """
        folds = []
        fold_id = 1
        data_end_ts = pd.Timestamp(data_end)

        test_start = self.min_test_start

        while True:
            test_end = test_start + pd.DateOffset(years=self.test_years) - pd.DateOffset(days=1)
            if test_end > data_end_ts:
                break

            train_end = test_start - pd.DateOffset(days=1)
            train_start = train_end - pd.DateOffset(years=self.train_years) + pd.DateOffset(days=1)

            folds.append({
                "fold": fold_id,
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end":   train_end.strftime("%Y-%m-%d"),
                "test_start":  test_start.strftime("%Y-%m-%d"),
                "test_end":    test_end.strftime("%Y-%m-%d"),
            })

            fold_id += 1
            test_start += pd.DateOffset(years=self.step_years)

        return folds

    # ------------------------------------------------------------------
    def run_fold(
        self,
        fold_spec: dict,
        prices: pd.DataFrame,
        vix: pd.Series,
        returns: pd.DataFrame,
        vix_features: pd.DataFrame,
        regime_labels: pd.Series,
        algorithms: list,
        kappa: float = KAPPA,
        asset_features: pd.DataFrame = None,
    ) -> FoldResult:
        """
        Run one fold: train classifier + reflex agent, then backtest.

        The regime classifier is trained on ALL available VIX data up to
        train_end (including any extended pre-2005 history).
        Algorithm evaluation uses the 5-asset prices/returns.
        """
        fold_id      = fold_spec["fold"]
        train_start  = fold_spec["train_start"]
        train_end    = fold_spec["train_end"]
        test_start   = fold_spec["test_start"]
        test_end     = fold_spec["test_end"]

        n_folds_total = "?"   # filled in by run_all; left as placeholder here

        sep = "-" * 60
        print(f"\n{sep}", flush=True)
        print(f"Fold {fold_id}: Train {train_start[:4]}-{train_end[:4]}, "
              f"Test {test_start[:4]}", flush=True)
        print(sep, flush=True)

        # ---- Split data --------------------------------------------------
        # Training returns (5-asset prices period)
        ret_mask  = (returns.index >= train_start) & (returns.index <= train_end)
        ret_train = returns.loc[ret_mask]

        # Training regime labels (5-asset price dates)
        rl_mask_5a = (regime_labels.index >= train_start) & (regime_labels.index <= train_end)
        rl_train_5a = regime_labels.loc[rl_mask_5a]

        # Classifier training: ALL VIX data up to train_end (extended history)
        clf_mask = vix_features.index <= train_end
        vx_clf   = vix_features.loc[clf_mask]
        rl_clf   = regime_labels.loc[regime_labels.index <= train_end]
        common_clf = vx_clf.index.intersection(rl_clf.index)
        vx_clf   = vx_clf.loc[common_clf]
        rl_clf   = rl_clf.loc[common_clf]

        # Test period
        test_mask    = (prices.index >= test_start) & (prices.index <= test_end)
        test_dates   = prices.index[test_mask]
        vx_test      = vix_features.loc[vix_features.index.intersection(test_dates)]
        rl_test      = regime_labels.loc[regime_labels.index.intersection(test_dates)]

        print(f"  Train returns   : {len(ret_train)} days")
        print(f"  Classifier data : {len(common_clf)} days "
              f"({common_clf[0].date()} to {common_clf[-1].date()})")
        print(f"  Test            : {len(test_dates)} days")

        # ---- 0. Stage 0: Pre-train Tier 2 algorithms (if any) -----------
        has_tier2 = any(isinstance(a, TrainablePortfolioAlgorithm) for a in algorithms)
        if has_tier2 and asset_features is not None:
            pretrain_tier2_algorithms(
                algorithms, asset_features, returns, train_start, train_end
            )
        elif has_tier2:
            print("  WARNING: Tier 2 algorithms present but asset_features not provided."
                  " Tier 2 will use equal-weight fallback.")

        # ---- 1. Regime classifier ----------------------------------------
        classifier = RegimeClassifier(model_type="logistic_regression")
        classifier.fit(vx_clf, rl_clf)

        eval_res = classifier.evaluate(vx_test, rl_test)
        regime_accuracy = eval_res["accuracy"]
        print(f"  Regime accuracy : {regime_accuracy:.4f}")

        # ---- 2. Reflex Agent ---------------------------------------------
        print("  Fitting ReflexAgent ...")
        agent = ReflexAgent()
        agent.fit(
            algorithms, ret_train, rl_train_5a, prices,
            metric="net", kappa=kappa,
        )

        # ---- 3. Backtest -------------------------------------------------
        backtester = Backtester(
            algorithms=algorithms,
            regime_classifier=classifier,
            returns=returns,
            prices=prices,
            vix_features=vix_features,
            regime_labels=regime_labels,
            kappa=kappa,
        )

        # Reflex
        res_reflex = backtester.run(agent, test_start, test_end, "Reflex")
        m_reflex   = compute_all_metrics(res_reflex)

        # Oracle (same mapping, true regimes)
        oracle = OracleAgent()
        oracle.mapping = agent.mapping.copy()
        res_oracle = backtester.run(oracle, test_start, test_end, "Oracle",
                                    use_true_regime=True)
        m_oracle = compute_all_metrics(res_oracle)

        # Equal Weight
        ew_algo  = next(a for a in algorithms if a.name == "EqualWeight")
        ew_agent = _SingleAlgoAgent(ew_algo)
        res_ew   = backtester.run(ew_agent, test_start, test_end, "EW")
        m_ew     = compute_all_metrics(res_ew)

        print(f"  EW Sharpe    : {m_ew['sharpe_ratio']:+.4f}", flush=True)
        print(f"  Reflex Sharpe: {m_reflex['sharpe_ratio']:+.4f}", flush=True)
        print(f"  Oracle Sharpe: {m_oracle['sharpe_ratio']:+.4f}", flush=True)

        # ---- 4. Per-algorithm test-period ranking -------------------------
        print(f"  Evaluating all {len(algorithms)} algorithms on test period ...", flush=True)
        algo_scores: dict = {}
        for algo in algorithms:
            algo_scores[algo.name] = _eval_algo_test(
                algo, prices, returns, test_dates, kappa=kappa
            )

        # ---- 5. Dominant regime in test period ---------------------------
        regime_counts = rl_test.value_counts()
        dominant_id   = int(regime_counts.index[0])
        dominant_name = REGIME_NAMES.get(dominant_id, str(dominant_id))
        regime_dist   = {
            REGIME_NAMES.get(int(r), str(r)): int(c)
            for r, c in regime_counts.items()
        }

        print(f"  Dominant regime: {dominant_name}  "
              f"(dist: {regime_dist})", flush=True)

        return FoldResult(
            fold_id=fold_id,
            fold_spec=fold_spec,
            metrics_reflex=m_reflex,
            metrics_oracle=m_oracle,
            metrics_ew=m_ew,
            algo_scores=algo_scores,
            regime_accuracy=regime_accuracy,
            reflex_mapping={r: a.name for r, a in agent.mapping.items()},
            dominant_regime=dominant_name,
            regime_dist=regime_dist,
            all_scores=dict(agent.all_scores),   # training-period regime scores
        )

    # ------------------------------------------------------------------
    def run_all(
        self,
        prices: pd.DataFrame,
        vix: pd.Series,
        returns: pd.DataFrame,
        vix_features: pd.DataFrame,
        regime_labels: pd.Series,
        algorithms: list,
        kappa: float = KAPPA,
        data_end: str = "2024-12-31",
        asset_features: pd.DataFrame = None,
    ) -> WalkForwardResult:
        """
        Run all folds and return aggregated results.

        Parameters
        ----------
        asset_features : pd.DataFrame, optional
            Pre-computed per-asset features (MultiIndex columns) for Stage 0.
            Required when algorithms contain Tier 2 TrainablePortfolioAlgorithm instances.
        """
        folds_spec = self.generate_folds(data_end=data_end)
        n_folds = len(folds_spec)
        print(f"\nWalk-Forward Validation: {n_folds} folds")
        print(f"  Train window : {self.train_years} years")
        print(f"  Test window  : {self.test_years} year(s)")
        print(f"  Step         : {self.step_years} year(s)")

        fold_results: List[FoldResult] = []
        for fs in folds_spec:
            fr = self.run_fold(
                fs, prices, vix, returns, vix_features, regime_labels,
                algorithms, kappa, asset_features=asset_features,
            )
            fold_results.append(fr)

        summary_df = _build_summary_df(fold_results)
        print("\n" + "=" * 70)
        print("WALK-FORWARD COMPLETE")
        print("=" * 70)
        _print_summary(summary_df)

        return WalkForwardResult(folds=fold_results, summary_df=summary_df)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _build_summary_df(folds: List[FoldResult]) -> pd.DataFrame:
    rows = []
    for fr in folds:
        rows.append({
            "fold":             fr.fold_id,
            "test_year":        fr.fold_spec["test_start"][:4],
            "train_period":     f"{fr.fold_spec['train_start'][:4]}-{fr.fold_spec['train_end'][:4]}",
            "dominant_regime":  fr.dominant_regime,
            "regime_accuracy":  fr.regime_accuracy,
            "ew_sharpe":        fr.metrics_ew["sharpe_ratio"],
            "reflex_sharpe":    fr.metrics_reflex["sharpe_ratio"],
            "oracle_sharpe":    fr.metrics_oracle["sharpe_ratio"],
            "oracle_gap":       fr.metrics_oracle["sharpe_ratio"] - fr.metrics_reflex["sharpe_ratio"],
            "reflex_vs_ew":     fr.metrics_reflex["sharpe_ratio"] - fr.metrics_ew["sharpe_ratio"],
            "ew_ann_ret":       fr.metrics_ew["ann_return"],
            "reflex_ann_ret":   fr.metrics_reflex["ann_return"],
            "oracle_ann_ret":   fr.metrics_oracle["ann_return"],
            "ew_maxdd":         fr.metrics_ew["max_drawdown"],
            "reflex_maxdd":     fr.metrics_reflex["max_drawdown"],
            # Reflex mapping per regime
            "map_calm":   fr.reflex_mapping.get(1, "N/A"),
            "map_normal": fr.reflex_mapping.get(2, "N/A"),
            "map_tense":  fr.reflex_mapping.get(3, "N/A"),
            "map_crisis": fr.reflex_mapping.get(4, "N/A"),
        })
    return pd.DataFrame(rows).set_index("fold")


def _print_summary(df: pd.DataFrame) -> None:
    header = (
        f"{'Fold':>5}  {'Year':>5}  {'Dom.Reg':>8}  "
        f"{'EW Sharpe':>10}  {'Reflex':>10}  {'Oracle':>10}  "
        f"{'OracleGap':>10}  {'Ref-EW':>8}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for fold_id, row in df.iterrows():
        print(
            f"{fold_id:>5}  {row['test_year']:>5}  {row['dominant_regime']:>8}  "
            f"{row['ew_sharpe']:>10.4f}  {row['reflex_sharpe']:>10.4f}  "
            f"{row['oracle_sharpe']:>10.4f}  {row['oracle_gap']:>10.4f}  "
            f"{row['reflex_vs_ew']:>8.4f}"
        )
    print(sep)
    # Averages
    nums = df[["ew_sharpe", "reflex_sharpe", "oracle_sharpe", "oracle_gap", "reflex_vs_ew"]]
    avgs = nums.mean()
    print(
        f"{'AVG':>5}  {'':>5}  {'':>8}  "
        f"{avgs['ew_sharpe']:>10.4f}  {avgs['reflex_sharpe']:>10.4f}  "
        f"{avgs['oracle_sharpe']:>10.4f}  {avgs['oracle_gap']:>10.4f}  "
        f"{avgs['reflex_vs_ew']:>8.4f}"
    )
    stds = nums.std()
    print(
        f"{'STD':>5}  {'':>5}  {'':>8}  "
        f"{stds['ew_sharpe']:>10.4f}  {stds['reflex_sharpe']:>10.4f}  "
        f"{stds['oracle_sharpe']:>10.4f}  {stds['oracle_gap']:>10.4f}  "
        f"{stds['reflex_vs_ew']:>8.4f}"
    )
