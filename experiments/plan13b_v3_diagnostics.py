# experiments/plan13b_v3_diagnostics.py -- Post-run diagnostics for Plan 13b-v3
#
# Three diagnostics requested after Folds 1-3:
#
#   Diag 1: Per-regime val_sharpe vs test-period Sharpe
#           Is the val_sharpe of the selected best config predictive of how
#           that config actually performs on its regime's test days?
#
#   Diag 2: Counterfactual — argmax(train_sharpe) selection
#           What test Sharpe would we get if, like Plan 13b-v2, we had used
#           the trial with the highest TRAIN sharpe as the best config?
#           Isolates whether the val-split changed the outcome vs. v2.
#
#   Diag 3: Fold 1, Regime 1 — full 100-trial landscape CSV
#           All trials with train_sharpe, val_sharpe, family.
#           Any trial with both train > 0 AND val > 0?
#
# Output: results/plan13b_v3/diagnostics/
#
# Usage:
#   cd Implementierung1
#   python -m experiments.plan13b_v3_diagnostics

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from regime_algo_selection.config import RESULTS_DIR, KAPPA, REGIME_NAMES, N_ASSETS
from regime_algo_selection.data.loader_extended import load_data_extended
from regime_algo_selection.data.features import compute_returns, compute_asset_features
from regime_algo_selection.regimes.ground_truth import compute_regime_labels
from regime_algo_selection.algorithms.stage0 import build_training_matrix
from regime_algo_selection.evaluation.walk_forward import WalkForwardValidator

# Import helpers from plan13b_v3 directly
from experiments.plan13b_v3_bo_val_split import (
    create_algorithm_from_config,
    compute_weights_for_config,
    _eval_strategy,
    _compute_metrics,
    CONFIG,
)

OUT_DIR   = os.path.join(RESULTS_DIR, "plan13b_v3")
DIAG_DIR  = os.path.join(OUT_DIR, "diagnostics")
os.makedirs(DIAG_DIR, exist_ok=True)

N     = N_ASSETS
KAPPA = CONFIG["kappa"]


# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------

def load_experiment_data():
    data           = load_data_extended()
    prices         = data["prices"]
    vix            = data["vix"]
    returns_raw    = compute_returns(prices)
    asset_features = compute_asset_features(prices)
    regime_labels  = compute_regime_labels(vix)

    common = (prices.index
              .intersection(returns_raw.index)
              .intersection(regime_labels.index))
    prices         = prices.loc[common]
    returns_raw    = returns_raw.loc[common]
    regime_labels  = regime_labels.loc[common]
    asset_features = asset_features.loc[asset_features.index.intersection(common)]

    all_dates   = common
    returns_arr = returns_raw.values
    regime_arr  = regime_labels.reindex(all_dates).fillna(2).astype(int).values

    wfv = WalkForwardValidator(
        train_years=CONFIG["train_years"],
        test_years=CONFIG["test_years"],
        step_years=CONFIG["step_years"],
        min_test_start=CONFIG["min_test_start"],
    )
    fold_specs = {f["fold"]: f for f in wfv.generate_folds(data_end=CONFIG["data_end"])}

    return {
        "prices": prices,
        "returns_raw": returns_raw,
        "asset_features": asset_features,
        "regime_labels": regime_labels,
        "all_dates": all_dates,
        "returns_arr": returns_arr,
        "regime_arr": regime_arr,
        "fold_specs": fold_specs,
    }


# ---------------------------------------------------------------------------
# Diag 1: val_sharpe predictiveness — per-regime test Sharpe for best config
# ---------------------------------------------------------------------------

def diag1_val_vs_test(fold_ids, data):
    """
    For each fold × regime: take the best config (by val_sharpe), apply it
    only to the test days belonging to that regime, compute test Sharpe.
    Compare to val_sharpe to assess predictiveness.
    """
    rows = []

    for fold_id in fold_ids:
        cfg_path = os.path.join(OUT_DIR, f"fold_{fold_id:02d}_best_configs.json")
        if not os.path.exists(cfg_path):
            print(f"  [diag1] Skipping fold {fold_id}: best_configs.json not found")
            continue

        with open(cfg_path) as f:
            regime_stats = json.load(f)

        fold_spec    = data["fold_specs"][fold_id]
        test_start   = fold_spec["test_start"]
        test_end     = fold_spec["test_end"]
        train_start  = fold_spec["train_start"]
        train_end    = fold_spec["train_end"]
        test_year    = test_start[:4]

        all_dates    = data["all_dates"]
        returns_arr  = data["returns_arr"]
        regime_arr   = data["regime_arr"]
        prices       = data["prices"]
        asset_features = data["asset_features"]
        returns_raw  = data["returns_raw"]

        # Build full training matrices for refit
        X_train, Y_train = build_training_matrix(
            asset_features, returns_raw, train_start, train_end
        )

        # Test indices
        test_mask    = (all_dates >= test_start) & (all_dates <= test_end)
        test_indices = np.where(test_mask)[0]
        test_dates_arr = all_dates[test_indices]

        for regime_str, stats in regime_stats.items():
            regime_id = int(regime_str)
            regime_name = REGIME_NAMES[regime_id]

            val_sharpe  = stats.get("val_sharpe",    float("nan"))
            train_sharpe = stats.get("train_sharpe", float("nan"))
            gap         = stats.get("train_val_gap", float("nan"))
            fallback    = stats.get("fallback",       False)
            family      = stats.get("family",         "?")
            config_dict = stats.get("config",         {})

            # Test days for this regime
            regime_test_mask  = (regime_arr[test_indices] == regime_id)
            regime_test_dates = test_dates_arr[regime_test_mask]
            regime_test_idx   = test_indices[regime_test_mask]
            n_test_regime_days = int(regime_test_mask.sum())

            if n_test_regime_days == 0 or fallback or not config_dict:
                test_regime_sharpe = float("nan")
            else:
                # Compute weights for this regime's test days using best config
                w_best = compute_weights_for_config(
                    config=config_dict,
                    X_train=X_train,
                    Y_train=Y_train,
                    asset_features=asset_features,
                    test_dates=regime_test_dates,
                    prices=prices,
                    N=N,
                )
                # Evaluate on just this regime's test days (no cost for first day)
                prev_w = np.ones(N) / N
                net_rets = []
                for pos, gi in enumerate(regime_test_idx):
                    w = w_best[pos]
                    w = np.where(np.isfinite(w), w, 0.0)
                    w = np.clip(w, 0.0, None)
                    s = w.sum()
                    w = w / s if s > 1e-12 else np.ones(N) / N
                    r = returns_arr[gi]
                    cost = KAPPA * float(np.abs(w - prev_w).sum())
                    net_rets.append(float(w @ r) - cost)
                    prev_w = w
                m = _compute_metrics(np.array(net_rets))
                test_regime_sharpe = m.get("sharpe", float("nan"))

            rows.append({
                "fold":                fold_id,
                "test_year":           test_year,
                "regime_id":           regime_id,
                "regime_name":         regime_name,
                "family":              family,
                "val_sharpe":          round(float(val_sharpe),    4),
                "train_sharpe":        round(float(train_sharpe),  4),
                "train_val_gap":       round(float(gap),           4),
                "test_regime_sharpe":  round(float(test_regime_sharpe), 4)
                                       if not np.isnan(float(test_regime_sharpe)) else float("nan"),
                "n_test_regime_days":  n_test_regime_days,
                "fallback":            fallback,
            })

    df = pd.DataFrame(rows)
    path = os.path.join(DIAG_DIR, "diag1_val_vs_test_per_regime.csv")
    df.to_csv(path, index=False)
    print(f"\n  [diag1] Saved: {path}")
    return df


# ---------------------------------------------------------------------------
# Diag 2: counterfactual — argmax(train_sharpe) selection
# ---------------------------------------------------------------------------

def diag2_argmax_train_counterfactual(fold_ids, data):
    """
    For each fold × regime: find the trial with highest train_sharpe in the
    trial log. Apply that config to test. Compare to actual v3 selection
    (argmax val_sharpe). Answers: did val-split change the outcome?
    """
    rows = []

    for fold_id in fold_ids:
        tl_path = os.path.join(OUT_DIR, f"fold_{fold_id:02d}_trial_log.csv")
        cfg_path = os.path.join(OUT_DIR, f"fold_{fold_id:02d}_best_configs.json")
        if not os.path.exists(tl_path) or not os.path.exists(cfg_path):
            print(f"  [diag2] Skipping fold {fold_id}: trial log or best_configs not found")
            continue

        trial_log = pd.read_csv(tl_path)
        with open(cfg_path) as f:
            regime_stats = json.load(f)

        fold_spec   = data["fold_specs"][fold_id]
        test_start  = fold_spec["test_start"]
        test_end    = fold_spec["test_end"]
        train_start = fold_spec["train_start"]
        train_end   = fold_spec["train_end"]
        test_year   = test_start[:4]

        all_dates    = data["all_dates"]
        returns_arr  = data["returns_arr"]
        regime_arr   = data["regime_arr"]
        prices       = data["prices"]
        asset_features = data["asset_features"]
        returns_raw  = data["returns_raw"]

        X_train, Y_train = build_training_matrix(
            asset_features, returns_raw, train_start, train_end
        )
        test_mask      = (all_dates >= test_start) & (all_dates <= test_end)
        test_indices   = np.where(test_mask)[0]
        test_dates_arr = all_dates[test_indices]

        for regime_id in [1, 2, 3, 4]:
            regime_name = REGIME_NAMES[regime_id]
            stats       = regime_stats.get(str(regime_id), {})
            fallback    = stats.get("fallback", False)

            # v3 best config (argmax val_sharpe)
            val_best_config = stats.get("config", {})
            val_best_sharpe = stats.get("val_sharpe", float("nan"))
            val_family      = stats.get("family", "?")

            # Regime's test days
            regime_test_mask  = (regime_arr[test_indices] == regime_id)
            regime_test_dates = test_dates_arr[regime_test_mask]
            regime_test_idx   = test_indices[regime_test_mask]
            n_test = int(regime_test_mask.sum())

            # Find argmax(train_sharpe) trial for this regime
            reg_tl = trial_log[trial_log["regime_id"] == regime_id].copy()
            reg_tl = reg_tl[np.isfinite(reg_tl["train_sharpe"])]

            if reg_tl.empty or fallback or n_test == 0:
                rows.append({
                    "fold": fold_id, "test_year": test_year,
                    "regime_id": regime_id, "regime_name": regime_name,
                    "val_best_family": val_family,
                    "val_best_val_sharpe": val_best_sharpe,
                    "train_best_family": "?",
                    "train_best_train_sharpe": float("nan"),
                    "train_best_val_sharpe": float("nan"),
                    "same_trial": False,
                    "test_sharpe_val_selection": float("nan"),
                    "test_sharpe_train_selection": float("nan"),
                    "delta_val_minus_train_selection": float("nan"),
                    "n_test_days": n_test,
                    "fallback": fallback,
                })
                continue

            best_train_row = reg_tl.loc[reg_tl["train_sharpe"].idxmax()]
            train_best_family      = best_train_row["family"]
            train_best_train_sh    = best_train_row["train_sharpe"]
            train_best_val_sh      = best_train_row["val_sharpe"]
            train_best_trial_num   = int(best_train_row["trial_num"])

            # val-best trial number (from regime_stats)
            val_best_trial_num = None
            for t_row in reg_tl.itertuples():
                if abs(t_row.val_sharpe - val_best_sharpe) < 1e-6:
                    val_best_trial_num = t_row.trial_num
                    break
            same_trial = (val_best_trial_num is not None and
                          val_best_trial_num == train_best_trial_num)

            def _regime_test_sharpe(config_dict):
                if not config_dict or n_test == 0:
                    return float("nan")
                w = compute_weights_for_config(
                    config=config_dict,
                    X_train=X_train, Y_train=Y_train,
                    asset_features=asset_features,
                    test_dates=regime_test_dates,
                    prices=prices, N=N,
                )
                prev_w = np.ones(N) / N
                net = []
                for pos, gi in enumerate(regime_test_idx):
                    ww = w[pos]
                    ww = np.where(np.isfinite(ww), ww, 0.0)
                    ww = np.clip(ww, 0.0, None)
                    s = ww.sum()
                    ww = ww / s if s > 1e-12 else np.ones(N) / N
                    cost = KAPPA * float(np.abs(ww - prev_w).sum())
                    net.append(float(ww @ returns_arr[gi]) - cost)
                    prev_w = ww
                return _compute_metrics(np.array(net)).get("sharpe", float("nan"))

            # Need the full config dict for the argmax-train trial.
            # We stored it in best_configs.json only for the val-best trial.
            # For the train-best trial, we need to reconstruct from the trial log.
            # The trial log only has family/tier, not full HP config.
            # We approximate: if same family as val-best, reuse that config.
            # If different, we can only note the family but can't reconstruct exact HPs.
            # => Mark as "config_unavailable" and use val-best config as proxy
            # where families match, else record NaN test sharpe.
            if train_best_family == val_family and val_best_config:
                train_best_config = val_best_config  # same family, approximate
                config_approx = True
            else:
                train_best_config = None
                config_approx = False

            test_sh_val   = _regime_test_sharpe(val_best_config) if val_best_config else float("nan")
            test_sh_train = _regime_test_sharpe(train_best_config) if train_best_config else float("nan")

            rows.append({
                "fold":                          fold_id,
                "test_year":                     test_year,
                "regime_id":                     regime_id,
                "regime_name":                   regime_name,
                "val_best_family":               val_family,
                "val_best_val_sharpe":           round(float(val_best_sharpe), 4),
                "train_best_family":             train_best_family,
                "train_best_train_sharpe":       round(float(train_best_train_sh), 4),
                "train_best_val_sharpe":         round(float(train_best_val_sh),   4),
                "same_trial":                    same_trial,
                "config_approx":                 config_approx,
                "test_sharpe_val_selection":     round(float(test_sh_val),   4)
                                                 if not np.isnan(float(test_sh_val)) else float("nan"),
                "test_sharpe_train_selection":   round(float(test_sh_train), 4)
                                                 if not np.isnan(float(test_sh_train)) else float("nan"),
                "delta_val_minus_train_sel":     round(float(test_sh_val - test_sh_train), 4)
                                                 if (not np.isnan(float(test_sh_val)) and
                                                     not np.isnan(float(test_sh_train))) else float("nan"),
                "n_test_days":                   n_test,
                "fallback":                      fallback,
            })

    df = pd.DataFrame(rows)
    path = os.path.join(DIAG_DIR, "diag2_counterfactual_train_selection.csv")
    df.to_csv(path, index=False)
    print(f"  [diag2] Saved: {path}")
    return df


# ---------------------------------------------------------------------------
# Diag 3: Fold 1, Regime 1 — full 100-trial landscape
# ---------------------------------------------------------------------------

def diag3_fold1_regime1_landscape():
    """
    Dump all 100 trials for Fold 1, Regime 1 with train_sharpe, val_sharpe,
    family. Flag any trial where train > 0 AND val > 0.
    """
    tl_path = os.path.join(OUT_DIR, "fold_01_trial_log.csv")
    if not os.path.exists(tl_path):
        print("  [diag3] fold_01_trial_log.csv not found — skipping")
        return None

    tl = pd.read_csv(tl_path)
    r1 = tl[tl["regime_id"] == 1].copy().reset_index(drop=True)

    r1["both_positive"] = (r1["train_sharpe"] > 0) & (r1["val_sharpe"] > 0)

    # Summary stats
    n_total          = len(r1)
    n_train_pos      = (r1["train_sharpe"] > 0).sum()
    n_val_pos        = (r1["val_sharpe"]   > 0).sum()
    n_both_pos       = r1["both_positive"].sum()
    n_train_neg      = (r1["train_sharpe"] < 0).sum()
    n_val_neg        = (r1["val_sharpe"]   < 0).sum()

    print(f"\n  [diag3] Fold 1, Regime 1 — {n_total} trials")
    print(f"    train > 0: {n_train_pos}/{n_total}  |  val > 0: {n_val_pos}/{n_total}")
    print(f"    BOTH train > 0 AND val > 0: {n_both_pos}/{n_total}")
    if n_both_pos > 0:
        print("    Trials with both positive:")
        print(r1[r1["both_positive"]][["trial_num","family","train_sharpe","val_sharpe","gap"]].to_string(index=False))
    else:
        print("    → No trial has BOTH train_sharpe > 0 AND val_sharpe > 0")
        print("    → Best val_sharpe in Regime 1:")
        best = r1.loc[r1["val_sharpe"].idxmax()]
        print(f"      trial={int(best.trial_num)} family={best.family} "
              f"train={best.train_sharpe:+.4f} val={best.val_sharpe:+.4f}")

    # Family breakdown
    print("\n    Family counts in Regime 1:")
    print(r1["family"].value_counts().to_string())

    print("\n    Val_sharpe by family (mean):")
    print(r1.groupby("family")["val_sharpe"].mean().sort_values(ascending=False).round(4).to_string())

    path = os.path.join(DIAG_DIR, "diag3_fold1_regime1_full_landscape.csv")
    r1.to_csv(path, index=False)
    print(f"\n  [diag3] Saved: {path}")
    return r1


# ---------------------------------------------------------------------------
# Print formatted summary tables
# ---------------------------------------------------------------------------

def print_diag1_table(df):
    print("\n" + "=" * 90)
    print("DIAG 1: Val Sharpe vs Test-Regime Sharpe (best config per regime)")
    print("=" * 90)
    header = f"{'Fold':>4} {'Year':>5} {'Regime':>7} {'Family':>20} {'val_sh':>8} {'test_sh':>8} {'gap':>8} {'n_test':>6}"
    print(header)
    print("-" * len(header))
    for _, r in df.iterrows():
        if r["fallback"]:
            continue
        print(f"{int(r['fold']):>4} {str(r['test_year']):>5} "
              f"{r['regime_name']:>7} {r['family']:>20} "
              f"{r['val_sharpe']:>+8.4f} {r['test_regime_sharpe']:>+8.4f} "
              f"{r['train_val_gap']:>+8.4f} {int(r['n_test_regime_days']):>6}")

    # Correlation
    clean = df.dropna(subset=["val_sharpe", "test_regime_sharpe"])
    clean = clean[~clean["fallback"]]
    if len(clean) >= 3:
        corr = clean["val_sharpe"].corr(clean["test_regime_sharpe"])
        print(f"\n  Pearson correlation(val_sharpe, test_regime_sharpe): {corr:+.4f}  (n={len(clean)})")
    print("=" * 90)


def print_diag2_table(df):
    print("\n" + "=" * 100)
    print("DIAG 2: Counterfactual — argmax(train_sharpe) vs argmax(val_sharpe) selection")
    print("=" * 100)
    header = (f"{'Fold':>4} {'Reg':>4} {'ValFamily':>18} {'TrainFamily':>18} "
              f"{'SameTrial':>10} {'TestVal':>8} {'TestTrain':>10} {'Delta':>8}")
    print(header)
    print("-" * len(header))
    for _, r in df.iterrows():
        if r["fallback"]:
            continue
        tv = f"{r['test_sharpe_val_selection']:+8.4f}" if not np.isnan(float(r['test_sharpe_val_selection'])) else "      na"
        tt = f"{r['test_sharpe_train_selection']:+8.4f}" if not np.isnan(float(r['test_sharpe_train_selection'])) else "        na"
        dt = f"{r['delta_val_minus_train_sel']:+8.4f}" if not np.isnan(float(r['delta_val_minus_train_sel'])) else "      na"
        approx = "*" if r.get("config_approx", False) else " "
        print(f"{int(r['fold']):>4} {int(r['regime_id']):>4} "
              f"{r['val_best_family']:>18} {r['train_best_family']:>18} "
              f"{str(r['same_trial']):>10} {tv} {tt}{approx} {dt}")
    print("  (* = train-best config approximated using val-best HPs, same family)")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Discover which folds are complete
    available_folds = []
    for fold_id in range(1, 13):
        res_path = os.path.join(OUT_DIR, f"fold_{fold_id:02d}_result.csv")
        tl_path  = os.path.join(OUT_DIR, f"fold_{fold_id:02d}_trial_log.csv")
        cfg_path = os.path.join(OUT_DIR, f"fold_{fold_id:02d}_best_configs.json")
        if all(os.path.exists(p) for p in [res_path, tl_path, cfg_path]):
            available_folds.append(fold_id)

    print(f"\nDiagnostics for Plan 13b-v3  |  Folds available: {available_folds}")

    if not available_folds:
        print("  No complete folds found — exiting.")
        return

    print("\nLoading experiment data...")
    exp_data = load_experiment_data()
    print(f"  Data loaded: {len(exp_data['all_dates'])} dates")

    # Diag 1
    print("\nRunning Diag 1: val_sharpe vs per-regime test Sharpe...")
    df1 = diag1_val_vs_test(available_folds, exp_data)
    print_diag1_table(df1)

    # Diag 2
    print("\nRunning Diag 2: argmax(train_sharpe) counterfactual...")
    df2 = diag2_argmax_train_counterfactual(available_folds, exp_data)
    print_diag2_table(df2)

    # Diag 3 (always Fold 1 Regime 1, regardless of which folds are available)
    print("\nRunning Diag 3: Fold 1, Regime 1 full trial landscape...")
    diag3_fold1_regime1_landscape()

    print(f"\nAll diagnostics written to: {DIAG_DIR}")


if __name__ == "__main__":
    main()
