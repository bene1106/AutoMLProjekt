# experiments/plan13c_v2_aggregate.py -- Plan 13c-v2: Aggregate Fold Results
#
# Usage:
#   cd Implementierung1
#   python -m experiments.plan13c_v2_aggregate

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "plan13c_v2"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ── Baselines for comparison ──────────────────────────────────────────────────
BASELINES = {
    "Oracle Per-Regime":         1.65,
    "Equal Weight (+EW)":        1.02,
    "Plan 13c (old, exhaustive)":0.88,
    "Plan 13b (blend)":          0.79,
    "Plan 13b (hard)":           0.75,
    "Plan 13a (hierarchical NN)":-0.65,
    "Plan 13b-v3 (BO+val-split)":-0.61,
}


def load_fold_results(folds: list[int]) -> pd.DataFrame:
    dfs = []
    for fold in folds:
        csv_path = RESULTS_DIR / f"fold_{fold:02d}_result.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["fold"] = fold
            dfs.append(df)
        else:
            print(f"  [MISSING] fold_{fold:02d}_result.csv")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def apply_decision_rule(avg_sharpe: float) -> str:
    if avg_sharpe < 0.00:
        return "STOP — Same pattern as 13b-v3. Pivot to new approach."
    elif avg_sharpe < 0.50:
        return "CAUTION — Continue with skepticism; prepare pivot narrative."
    elif avg_sharpe < 0.88:
        return "CONTINUE — Comparable to Plan 13c (old)."
    elif avg_sharpe < 1.02:
        return "CONTINUE — Exceeds Plan 13c (old)."
    else:
        return "EXCEPTIONAL — First method to beat Equal Weight (+1.02)."


def main():
    folds_to_check = list(range(1, 13))
    available_folds = [f for f in folds_to_check
                       if (RESULTS_DIR / f"fold_{f:02d}_result.csv").exists()]

    print(f"Available folds: {available_folds}")
    if not available_folds:
        print("No fold results found. Run experiments first.")
        return

    df = load_fold_results(available_folds)

    # Filter to hierarchical blend strategy (primary strategy)
    blend_df = df[df["strategy"] == "hier_blend"].copy()
    tier1_df = df[df["strategy"] == "tier1_hard"].copy()
    ew_df    = df[df["strategy"] == "ew"].copy()

    print("\n" + "=" * 70)
    print("PLAN 13c-v2 AGGREGATE RESULTS")
    print("=" * 70)
    print(f"\n{'Fold':>5}  {'Year':>5}  {'Blend':>8}  {'T1Hard':>8}  {'EW':>8}")
    print("-" * 45)

    for fold in sorted(blend_df["fold"].unique()):
        b_row = blend_df[blend_df["fold"] == fold]
        t_row = tier1_df[tier1_df["fold"] == fold]
        e_row = ew_df[ew_df["fold"] == fold]

        b = b_row["sharpe"].values[0] if len(b_row) else float("nan")
        t = t_row["sharpe"].values[0] if len(t_row) else float("nan")
        e = e_row["sharpe"].values[0] if len(e_row) else float("nan")
        year = b_row["test_year"].values[0] if len(b_row) else "?"

        def _f(v):
            try:
                return f"{float(v):+8.4f}" if not np.isnan(float(v)) else "     nan"
            except Exception:
                return "     nan"

        print(f"{fold:>5}  {year:>5}  {_f(b)}  {_f(t)}  {_f(e)}")

    print("-" * 45)

    avg_blend = blend_df["sharpe"].mean()
    avg_tier1 = tier1_df["sharpe"].mean()
    avg_ew    = ew_df["sharpe"].mean()
    def _f(v):
        try:
            return f"{float(v):+8.4f}" if not np.isnan(float(v)) else "     nan"
        except Exception:
            return "     nan"
    print(f"{'AVG':>5}  {'':>5}  {_f(avg_blend)}  {_f(avg_tier1)}  {_f(avg_ew)}")
    print("=" * 70)

    n = len(available_folds)
    beats_ew_blend = (blend_df["sharpe"] > 1.02).sum()
    beats_ew_t1    = (tier1_df["sharpe"] > 1.02).sum()

    print(f"\nDiagnostics ({n} folds):")
    print(f"  Hier Blend > EW (+1.02): {beats_ew_blend}/{n}")
    print(f"  Tier1 Hard > EW (+1.02): {beats_ew_t1}/{n}")

    # ── Decision rule (diagnostic folds 1-3) ────────────────────────────────
    diag_folds = [f for f in [1, 2, 3] if f in available_folds]
    if diag_folds:
        diag_df = blend_df[blend_df["fold"].isin(diag_folds)]
        diag_avg = diag_df["sharpe"].mean()
        print(f"\nDECISION RULE (Folds {diag_folds}):")
        print(f"  Avg Sharpe: {diag_avg:+.4f}")
        print(f"  Decision:   {apply_decision_rule(diag_avg)}")

    # ── Comparison with baselines ────────────────────────────────────────────
    print("\nComparison with baselines:")
    all_methods = {
        **BASELINES,
        "Plan 13c-v2 Hier Blend (this)": avg_blend,
        "Plan 13c-v2 Tier1 Hard":        avg_tier1,
    }
    for method, sharpe in sorted(all_methods.items(), key=lambda x: -float(x[1])):
        marker = " ◄ THIS" if "13c-v2" in method else ""
        try:
            print(f"  {sharpe:+7.4f}  {method}{marker}")
        except Exception:
            pass

    # ── Save summary ─────────────────────────────────────────────────────────
    summary_rows = []
    for fold in sorted(blend_df["fold"].unique()):
        b_row = blend_df[blend_df["fold"] == fold]
        t_row = tier1_df[tier1_df["fold"] == fold]
        e_row = ew_df[ew_df["fold"] == fold]
        summary_rows.append({
            "fold": fold,
            "test_year": b_row["test_year"].values[0] if len(b_row) else None,
            "blend_sharpe": b_row["sharpe"].values[0] if len(b_row) else None,
            "tier1_sharpe": t_row["sharpe"].values[0] if len(t_row) else None,
            "ew_sharpe":    e_row["sharpe"].values[0] if len(e_row) else None,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = ANALYSIS_DIR / "aggregate_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
