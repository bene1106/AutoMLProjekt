"""
main.py -- Regime-Aware Reflex Agent Baseline
=============================================
Runs the full pipeline:
  1. Load data
  2. Feature engineering
  3. Regime labels
  4. Build algorithm space (K ~48)
  5. Train regime classifier
  6. Fit reflex agent (best algo per regime on training data)
  7. Backtest reflex agent, oracle agent, equal-weight on TEST set
  8. Compute metrics
  9. Generate all visualisations
 10. Print analysis summary
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
from regime_algo_selection.config import (
    TRAIN_END, VAL_END, KAPPA, REGIME_NAMES, RESULTS_DIR,
)

# ── Data ───────────────────────────────────────────────────────────────────────
from regime_algo_selection.data.loader   import load_data
from regime_algo_selection.data.features import (
    compute_returns, compute_asset_features, compute_vix_features,
)

# ── Regimes ────────────────────────────────────────────────────────────────────
from regime_algo_selection.regimes.ground_truth import compute_regime_labels
from regime_algo_selection.regimes.classifier   import RegimeClassifier

# ── Algorithms ─────────────────────────────────────────────────────────────────
from regime_algo_selection.algorithms.tier1_heuristics import build_tier1_algorithm_space

# ── Agents ─────────────────────────────────────────────────────────────────────
from regime_algo_selection.agents.reflex_agent import ReflexAgent, OracleAgent

# ── Evaluation ─────────────────────────────────────────────────────────────────
from regime_algo_selection.evaluation.backtest       import Backtester
from regime_algo_selection.evaluation.metrics        import (
    compute_all_metrics, per_regime_metrics, print_metrics_table,
)
from regime_algo_selection.evaluation.visualization  import (
    plot_cumulative_wealth, plot_regime_classification,
    plot_algorithm_selection, plot_regime_table,
    plot_confusion_matrix, plot_metrics_summary,
)


# ==============================================================================
def main():
    print("\n" + "=" * 70)
    print("  Regime-Aware Reflex Agent Baseline")
    print("=" * 70 + "\n")

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print("[1/9] Loading data...")
    data   = load_data()
    prices = data["prices"]
    vix    = data["vix"]

    # ── 2. Feature engineering ─────────────────────────────────────────────────
    print("\n[2/9] Computing features...")
    returns      = compute_returns(prices)
    vix_features = compute_vix_features(vix)
    regime_labels = compute_regime_labels(vix)

    print(f"  Returns:      {returns.shape}")
    print(f"  VIX features: {vix_features.shape}")
    print(f"  Regime dist:  { {k: int((regime_labels == k).sum()) for k in [1,2,3,4]} }")

    # ── 3. Temporal split masks ────────────────────────────────────────────────
    # Use returns.index as the canonical date index (4779 rows = prices minus 1)
    # vix_features / regime_labels share prices.index (4780 rows).
    # We restrict all splits to dates present in returns.
    common_dates = returns.index

    train_mask = common_dates <= TRAIN_END
    test_mask  = common_dates > VAL_END

    train_dates = common_dates[train_mask]
    test_dates  = common_dates[test_mask]
    test_start  = str(test_dates[0].date())
    test_end    = str(test_dates[-1].date())

    print(f"\n  Train: {train_dates[0].date()} to {train_dates[-1].date()} "
          f"({len(train_dates)} days)")
    print(f"  Test:  {test_start} to {test_end} ({len(test_dates)} days)")

    # ── 4. Build algorithm space ───────────────────────────────────────────────
    print("\n[3/9] Building Tier 1 algorithm space...")
    algorithms = build_tier1_algorithm_space()

    # ── 5. Train regime classifier ─────────────────────────────────────────────
    print("\n[4/9] Training regime classifier...")

    # Align features and labels to training period (restrict to common_dates)
    vix_train_raw = vix_features.loc[vix_features.index.isin(train_dates)].dropna()
    y_train       = regime_labels.loc[vix_train_raw.index]
    vix_train     = vix_train_raw

    classifier = RegimeClassifier("logistic_regression")
    classifier.fit(vix_train, y_train)

    # Evaluate on test set (restrict to common_dates)
    vix_test = vix_features.loc[vix_features.index.isin(test_dates)]
    y_test   = regime_labels.loc[vix_test.index]
    clf_eval  = classifier.evaluate(vix_test, y_test)

    print(f"\n  Classifier accuracy (test): {clf_eval['accuracy']:.4f}")
    print("\n  Classification Report:")
    print(clf_eval["classification_report"])

    # ── 6. Fit agents ──────────────────────────────────────────────────────────
    print("[5/9] Fitting Reflex Agent (may take a few minutes)...")

    reflex = ReflexAgent()
    reflex.fit(
        algorithms      = algorithms,
        returns         = returns.loc[train_dates],
        regime_labels   = regime_labels.loc[regime_labels.index.isin(train_dates)],
        prices          = prices,
    )

    oracle = OracleAgent()
    oracle.fit(
        algorithms    = algorithms,
        returns       = returns.loc[train_dates],
        regime_labels = regime_labels.loc[regime_labels.index.isin(train_dates)],
        prices        = prices,
    )

    # ── 7. Backtesting ─────────────────────────────────────────────────────────
    print("\n[6/9] Running backtests on test set...")

    # Equal-Weight agent: always selects EqualWeight regardless of regime
    from regime_algo_selection.algorithms.tier1_heuristics import EqualWeight
    class EqualWeightAgent:
        def __init__(self):
            self._algo = EqualWeight()
        def select(self, regime):
            return self._algo

    ew_agent = EqualWeightAgent()

    backtester = Backtester(
        algorithms     = algorithms,
        regime_classifier = classifier,
        returns        = returns,
        prices         = prices,
        vix_features   = vix_features,
        regime_labels  = regime_labels,
        kappa          = KAPPA,
    )

    print("  Running Equal-Weight baseline...")
    result_ew = backtester.run(
        agent=ew_agent, start_date=test_start, end_date=test_end,
        run_label="Equal Weight",
    )

    print("  Running Reflex Agent...")
    result_reflex = backtester.run(
        agent=reflex, start_date=test_start, end_date=test_end,
        run_label="Reflex Agent",
    )

    print("  Running Oracle Agent...")
    result_oracle = backtester.run(
        agent=oracle, start_date=test_start, end_date=test_end,
        run_label="Oracle Agent", use_true_regime=True,
    )

    # ── 8. Compute metrics ─────────────────────────────────────────────────────
    print("\n[7/9] Computing metrics...")
    metrics = {
        "Equal Weight" : compute_all_metrics(result_ew),
        "Reflex Agent" : compute_all_metrics(result_reflex),
        "Oracle Agent" : compute_all_metrics(result_oracle),
    }

    regime_df = per_regime_metrics(result_reflex)

    # ── 9. Visualisations ──────────────────────────────────────────────────────
    print("\n[8/9] Generating plots...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    regime_test = regime_labels.loc[regime_labels.index.isin(test_dates)]

    plot_cumulative_wealth(
        {"Equal Weight": result_ew, "Reflex Agent": result_reflex, "Oracle Agent": result_oracle},
        regime_test,
    )
    plot_regime_classification(
        vix, regime_labels, result_reflex.regime_predictions,
        start_date=test_start, end_date=test_end,
    )
    plot_algorithm_selection(result_reflex)
    plot_regime_table(regime_df)
    plot_confusion_matrix(clf_eval["confusion_matrix"])
    plot_metrics_summary(metrics)

    # ── 10. Print analysis ─────────────────────────────────────────────────────
    print("\n[9/9] Analysis Summary")
    print("=" * 70)

    print("\n--- Strategy Comparison Table ---")
    print_metrics_table(metrics)

    print("\n--- Per-Regime Performance (Reflex Agent) ---")
    print(regime_df.to_string())

    print("\n--- Reflex Agent Regime Mapping ---")
    for r, algo in reflex.mapping.items():
        print(f"  Regime {r} ({REGIME_NAMES[r]:7s}) -> {algo.name}")

    sharpe_reflex = metrics["Reflex Agent"]["sharpe_ratio"]
    sharpe_oracle = metrics["Oracle Agent"]["sharpe_ratio"]
    sharpe_ew     = metrics["Equal Weight"]["sharpe_ratio"]

    print(f"\n--- Key Findings ---")
    print(f"  Oracle Gap  (Sharpe):      {sharpe_oracle - sharpe_reflex:+.4f}  "
          f"(performance lost to regime mis-prediction)")
    print(f"  Reflex vs EW (Sharpe):     {sharpe_reflex - sharpe_ew:+.4f}  "
          f"(gain from regime-aware selection)")
    print(f"  Regime accuracy (test):    {clf_eval['accuracy']:.4f}")

    print(f"\nAll plots saved to: {RESULTS_DIR}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
