# experiments/plan4_full.py -- Plan 4 Full Pipeline Entry Point
#
# Runs all Plan 4 steps in sequence:
#   Step 1: Build Tier 1+2 algorithm space (K=81)
#   Step 2: Walk-Forward with Tier 1+2 (Run B, new)
#           -- Run A (Tier 1 only) loaded from saved CSV if available
#   Step 3: Per-regime algorithm rankings (Analyses 3.1, 3.2, 3.3)
#   Step 4: Tier 1 vs Tier 1+2 comparison (Analysis 4)
#   Step 5: Final summary for professor
#
# Usage:
#   cd Implementierung1
#   python -u -m regime_algo_selection.experiments.plan4_full
#   (the -u flag disables output buffering for real-time progress)

import os
import sys
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# ---- project imports -------------------------------------------------------
from regime_algo_selection.config import RESULTS_DIR, KAPPA, REGIME_NAMES, N_REGIMES
from regime_algo_selection.data.loader_extended import load_data_extended
from regime_algo_selection.data.features import (
    compute_returns, compute_vix_features, compute_asset_features,
)
from regime_algo_selection.regimes.ground_truth import compute_regime_labels
from regime_algo_selection.algorithms.tier1_heuristics import (
    build_tier1_algorithm_space, build_algorithm_space,
)
from regime_algo_selection.evaluation.walk_forward import (
    WalkForwardValidator, WalkForwardResult, FoldResult,
)

from regime_algo_selection.experiments.regime_algorithm_ranking import (
    run_regime_algorithm_ranking,
)
from regime_algo_selection.experiments.tier_comparison import run_tier_comparison
from regime_algo_selection.experiments.final_summary import run_final_summary

os.makedirs(RESULTS_DIR, exist_ok=True)

_T1_CSV        = os.path.join(RESULTS_DIR, "walk_forward_t1_plan4.csv")
_T12_CSV       = os.path.join(RESULTS_DIR, "walk_forward_t12_plan4.csv")
_T12_SCORES_PKL = os.path.join(RESULTS_DIR, "walk_forward_t12_all_scores.pkl")


# ===========================================================================
# Lightweight WalkForwardResult reconstruction from summary CSV
# ===========================================================================

def _load_wf_result_from_csv(csv_path: str) -> "WalkForwardResult":
    """
    Reconstruct a minimal WalkForwardResult from a summary CSV.
    FoldResult.all_scores will be None (not saved in CSV).
    This is sufficient for tier_comparison and final_summary.
    """
    df = pd.read_csv(csv_path, index_col=0)

    def _metrics(sharpe_col, ann_ret_col=None, maxdd_col=None) -> dict:
        # Returns a function that builds a metrics dict for a given row
        pass

    folds = []
    for fold_id, row in df.iterrows():
        fold_spec = {
            "fold": int(fold_id),
            "train_start": row.get("train_period", "?-?").split("-")[0] + "-01-01",
            "train_end":   row.get("train_period", "?-?").split("-")[-1] + "-12-31",
            "test_start":  str(row["test_year"]) + "-01-01",
            "test_end":    str(row["test_year"]) + "-12-31",
        }
        m_ew = {
            "sharpe_ratio": row.get("ew_sharpe",     0.0),
            "ann_return":   row.get("ew_ann_ret",    0.0),
            "max_drawdown": row.get("ew_maxdd",      0.0),
        }
        m_reflex = {
            "sharpe_ratio": row.get("reflex_sharpe",    0.0),
            "ann_return":   row.get("reflex_ann_ret",   0.0),
            "max_drawdown": row.get("reflex_maxdd",     0.0),
        }
        m_oracle = {
            "sharpe_ratio": row.get("oracle_sharpe",  0.0),
            "ann_return":   row.get("oracle_ann_ret", 0.0),
            "max_drawdown": row.get("ew_maxdd",       0.0),
        }
        reflex_mapping = {
            1: row.get("map_calm",   "N/A"),
            2: row.get("map_normal", "N/A"),
            3: row.get("map_tense",  "N/A"),
            4: row.get("map_crisis", "N/A"),
        }
        fr = FoldResult(
            fold_id=int(fold_id),
            fold_spec=fold_spec,
            metrics_reflex=m_reflex,
            metrics_oracle=m_oracle,
            metrics_ew=m_ew,
            algo_scores={},
            regime_accuracy=row.get("regime_accuracy", 0.0),
            reflex_mapping=reflex_mapping,
            dominant_regime=row.get("dominant_regime", "?"),
            regime_dist={},
            all_scores=None,
        )
        folds.append(fr)

    wf_result = WalkForwardResult(folds=folds, summary_df=df)

    # Attach all_scores from pickle if available
    pkl_path = _T12_SCORES_PKL if "t12" in csv_path else None
    if pkl_path and os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            all_scores_list = pickle.load(f)   # list of dicts, one per fold
        for fr, all_sc in zip(wf_result.folds, all_scores_list):
            fr.all_scores = all_sc
        print(f"  Loaded all_scores from {pkl_path}", flush=True)

    return wf_result


# ===========================================================================
# Step 0: Load Data
# ===========================================================================

def load_all_data():
    print("\n" + "=" * 70, flush=True)
    print("STEP 0: LOAD DATA", flush=True)
    print("=" * 70, flush=True)

    raw = load_data_extended()
    prices_raw = raw["prices"]
    vix_raw    = raw["vix"]

    returns        = compute_returns(prices_raw)
    vix_features   = compute_vix_features(vix_raw)
    regime_labels  = compute_regime_labels(vix_raw)
    asset_features = compute_asset_features(prices_raw)

    prices = prices_raw.loc[returns.index[0]:]

    print(f"  Prices        : {len(prices)} days  ({prices.index[0].date()} to {prices.index[-1].date()})", flush=True)
    print(f"  Returns       : {len(returns)} days", flush=True)
    print(f"  Asset features: {asset_features.shape}", flush=True)

    return prices, vix_raw, returns, vix_features, regime_labels, asset_features


# ===========================================================================
# Step 1: Build Algorithm Spaces
# ===========================================================================

def build_algorithm_spaces():
    print("\n" + "=" * 70, flush=True)
    print("STEP 1: BUILD ALGORITHM SPACES", flush=True)
    print("=" * 70, flush=True)

    print("\n  [A] Tier 1 only (K=48):", flush=True)
    algos_t1 = build_algorithm_space(tiers=[1])

    print("\n  [B] Tier 1 + Tier 2 (K=81):", flush=True)
    algos_t12 = build_algorithm_space(tiers=[1, 2])

    return algos_t1, algos_t12


# ===========================================================================
# Step 2: Walk-Forward Runs (with CSV caching)
# ===========================================================================

def run_walk_forward_both(
    prices, vix, returns, vix_features, regime_labels, asset_features,
    algos_t1, algos_t12,
):
    print("\n" + "=" * 70, flush=True)
    print("STEP 2: WALK-FORWARD VALIDATION", flush=True)
    print("=" * 70, flush=True)

    wf = WalkForwardValidator(
        train_years=8, test_years=1, step_years=1,
        min_test_start="2013-01-01",
    )

    # --- Run A: Tier 1 only -------------------------------------------------
    if os.path.exists(_T1_CSV):
        print(f"\n[Run A] Loading Tier 1 results from saved CSV: {_T1_CSV}", flush=True)
        wf_t1 = _load_wf_result_from_csv(_T1_CSV)
        print(f"  Loaded {len(wf_t1.folds)} folds.", flush=True)
    else:
        print("\n[Run A] Tier 1 only (K=48) -- running walk-forward ...", flush=True)
        t0 = time.time()
        wf_t1 = wf.run_all(
            prices=prices, vix=vix, returns=returns,
            vix_features=vix_features, regime_labels=regime_labels,
            algorithms=algos_t1, kappa=KAPPA, data_end="2024-12-31",
            asset_features=None,
        )
        print(f"  Run A complete in {(time.time()-t0)/60:.1f} min", flush=True)
        wf_t1.summary_df.to_csv(_T1_CSV)
        print(f"  Saved: {_T1_CSV}", flush=True)

    # --- Run B: Tier 1 + Tier 2 ---------------------------------------------
    if os.path.exists(_T12_CSV):
        print(f"\n[Run B] Loading Tier 1+2 results from saved CSV: {_T12_CSV}", flush=True)
        # For regime ranking, we need all_scores -- cannot load from CSV.
        # Must re-run if all_scores needed. Check if regime_ranking CSVs exist.
        regime_csvs_exist = all(
            os.path.exists(os.path.join(RESULTS_DIR, f"regime_ranking_{r}.csv"))
            for r in ["calm", "normal", "tense", "crisis"]
        )
        if regime_csvs_exist:
            print("  Regime ranking CSVs already exist. Loading T1+2 from CSV.", flush=True)
            wf_t12 = _load_wf_result_from_csv(_T12_CSV)
            print(f"  Loaded {len(wf_t12.folds)} folds.", flush=True)
        else:
            print("  Regime ranking CSVs missing -- re-running Run B to get all_scores ...", flush=True)
            os.remove(_T12_CSV)
            wf_t12 = _run_b(wf, prices, vix, returns, vix_features, regime_labels,
                            asset_features, algos_t12)
    else:
        wf_t12 = _run_b(wf, prices, vix, returns, vix_features, regime_labels,
                        asset_features, algos_t12)

    return wf_t1, wf_t12


def _run_b(wf, prices, vix, returns, vix_features, regime_labels,
           asset_features, algos_t12):
    print("\n[Run B] Tier 1+2 (K=81) with Stage 0 pre-training ...", flush=True)
    t0 = time.time()
    wf_t12 = wf.run_all(
        prices=prices, vix=vix, returns=returns,
        vix_features=vix_features, regime_labels=regime_labels,
        algorithms=algos_t12, kappa=KAPPA, data_end="2024-12-31",
        asset_features=asset_features,
    )
    print(f"  Run B complete in {(time.time()-t0)/60:.1f} min", flush=True)
    wf_t12.summary_df.to_csv(_T12_CSV)
    print(f"  Saved: {_T12_CSV}", flush=True)
    # Save all_scores as pickle for cache re-use
    all_scores_list = [fr.all_scores for fr in wf_t12.folds]
    with open(_T12_SCORES_PKL, "wb") as f:
        pickle.dump(all_scores_list, f)
    print(f"  Saved: {_T12_SCORES_PKL}", flush=True)
    return wf_t12


# ===========================================================================
# Steps 3-5
# ===========================================================================

def _load_regime_rankings_from_csv(algorithms: list) -> dict:
    """Load pre-computed regime rankings from CSV files."""
    from regime_algo_selection.config import REGIME_NAMES
    from regime_algo_selection.algorithms.base import TrainablePortfolioAlgorithm
    name_map = {1: "calm", 2: "normal", 3: "tense", 4: "crisis"}
    algo_meta = {a.name: {"family": a.family, "tier": 2 if isinstance(a, TrainablePortfolioAlgorithm) else 1}
                 for a in algorithms}
    result = {}
    for regime_int, regime_name in REGIME_NAMES.items():
        path = os.path.join(RESULTS_DIR, f"regime_ranking_{name_map[regime_int]}.csv")
        df = pd.read_csv(path, index_col=0)
        df.index.name = "algo_name"
        result[regime_int] = df
        print(f"  Loaded {path} ({len(df)} algorithms)", flush=True)
    return result


def run_step3(wf_t12, algos_t12, prices=None, returns=None, regime_labels=None):
    # Check if all folds have no all_scores (loaded from CSV)
    has_all_scores = any(fr.all_scores is not None for fr in wf_t12.folds)
    regime_csvs_exist = all(
        os.path.exists(os.path.join(RESULTS_DIR, f"regime_ranking_{r}.csv"))
        for r in ["calm", "normal", "tense", "crisis"]
    )

    if not has_all_scores and regime_csvs_exist:
        print("\n" + "=" * 70, flush=True)
        print("ANALYSIS 3: PER-REGIME ALGORITHM RANKINGS", flush=True)
        print("=" * 70, flush=True)
        print("\n[3.1] Loading pre-computed rankings from CSV files ...", flush=True)
        regime_ranking_dfs = _load_regime_rankings_from_csv(algos_t12)
        # Re-run plots only (fast)
        from regime_algo_selection.experiments.regime_algorithm_ranking import (
            plot_regime_top5, plot_tier_comparison_per_regime,
        )
        print("\n[3.2] Plotting top-5 per regime ...", flush=True)
        plot_regime_top5(regime_ranking_dfs, algos_t12)
        print("\n[3.3] Tier 1 vs Tier 2 per regime ...", flush=True)
        plot_tier_comparison_per_regime(regime_ranking_dfs, algos_t12)
        return regime_ranking_dfs

    regime_ranking_dfs = run_regime_algorithm_ranking(
        wf_result=wf_t12,
        algorithms=algos_t12,
    )
    return regime_ranking_dfs


def run_step4(wf_t1, wf_t12):
    return run_tier_comparison(wf_t1, wf_t12)


def run_step5(wf_t1, wf_t12, tier_df, regime_ranking_dfs, algos_t1, algos_t12):
    run_final_summary(
        wf_t1=wf_t1,
        wf_t12=wf_t12,
        tier_comparison_df=tier_df,
        regime_ranking_dfs=regime_ranking_dfs,
        algorithms_t1=algos_t1,
        algorithms_t12=algos_t12,
    )


# ===========================================================================
# Main
# ===========================================================================

def main():
    total_start = time.time()
    print("\n" + "#" * 70, flush=True)
    print("# PLAN 4: TIER 2 ALGORITHMS + PER-REGIME RANKINGS", flush=True)
    print("#" * 70, flush=True)

    prices, vix, returns, vix_features, regime_labels, asset_features = load_all_data()
    algos_t1, algos_t12 = build_algorithm_spaces()
    wf_t1, wf_t12 = run_walk_forward_both(
        prices, vix, returns, vix_features, regime_labels, asset_features,
        algos_t1, algos_t12,
    )
    regime_ranking_dfs = run_step3(wf_t12, algos_t12)
    tier_df = run_step4(wf_t1, wf_t12)
    run_step5(wf_t1, wf_t12, tier_df, regime_ranking_dfs, algos_t1, algos_t12)

    total_min = (time.time() - total_start) / 60
    print(f"\n{'='*70}", flush=True)
    print(f"PLAN 4 COMPLETE -- total runtime: {total_min:.1f} minutes", flush=True)
    print(f"Outputs saved to: {RESULTS_DIR}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
