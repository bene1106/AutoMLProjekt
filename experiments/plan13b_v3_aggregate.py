# experiments/plan13b_v3_aggregate.py -- Plan 13b-v3: Aggregation Utility
#
# Reads per-fold result CSVs from results/plan13b_v3/,
# computes summary statistics, and writes plan13b_v3_summary.csv.
#
# Usage:
#   cd Implementierung1
#   python -m experiments.plan13b_v3_aggregate

import os
import numpy as np
import pandas as pd

from regime_algo_selection.config import RESULTS_DIR

OUT_DIR     = os.path.join(RESULTS_DIR, "plan13b_v3")
SUMMARY_CSV = os.path.join(OUT_DIR, "plan13b_v3_summary.csv")

# Prior plans for comparison
PRIOR = {
    "13b":    os.path.join(RESULTS_DIR, "plan13b_bayesian_opt",  "summary_metrics.csv"),
    "13b-v2": os.path.join(RESULTS_DIR, "plan13b_v2_true_bo",   "summary_metrics.csv"),
    "ew":     None,  # computed inline from plan13b_v3 fold results
}


def load_fold_results(out_dir: str) -> pd.DataFrame:
    """Read all fold_NN_result.csv files and concatenate."""
    frames = []
    for fname in sorted(os.listdir(out_dir)):
        if fname.startswith("fold_") and fname.endswith("_result.csv"):
            path = os.path.join(out_dir, fname)
            try:
                df = pd.read_csv(path)
                frames.append(df)
            except Exception as e:
                print(f"  Warning: could not read {path}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_prior(path: str, col: str) -> dict:
    """Load {fold: sharpe} from a prior plan summary CSV."""
    if path is None or not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        if "fold" not in df.columns or col not in df.columns:
            return {}
        return {int(r["fold"]): float(r[col]) for _, r in df.iterrows()}
    except Exception:
        return {}


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot and aggregate fold results.

    Input columns: fold, test_year, strategy, sharpe, sortino, max_dd, turnover
    Returns per-fold wide format with avg/median/std appended.
    """
    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot_table(
        index=["fold", "test_year"],
        columns="strategy",
        values=["sharpe", "sortino", "max_dd", "turnover"],
        aggfunc="first",
    )
    pivot.columns = ["_".join(c) for c in pivot.columns]
    pivot = pivot.reset_index().sort_values("fold")

    # Rename for readability
    rename = {}
    for col in pivot.columns:
        for s in ["hard_bo", "top3_bo", "ew"]:
            if col.endswith(f"_{s}"):
                prefix = col.replace(f"_{s}", "")
                rename[col] = f"{s}_{prefix}"
    pivot = pivot.rename(columns=rename)

    return pivot


def print_summary_table(pivot: pd.DataFrame, prior_13b: dict, prior_13bv2: dict) -> None:
    """Print a comparison table to stdout."""
    if pivot.empty:
        print("  No fold results found.", flush=True)
        return

    sharpe_col_hard  = "hard_bo_sharpe"
    sharpe_col_top3  = "top3_bo_sharpe"
    sharpe_col_ew    = "ew_sharpe"

    header = (
        f"{'Fold':>5}  {'Year':>5}  "
        f"{'Hard-v3':>9}  {'Top3-v3':>9}  {'EW':>8}  "
        f"{'13b':>8}  {'13b-v2':>9}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * 80)
    print("PLAN 13b-v3 SUMMARY", flush=True)
    print("=" * 80)
    print(header)
    print(sep)

    def _fmt(v):
        try:
            return f"{float(v):+9.4f}" if not np.isnan(float(v)) else "       na"
        except Exception:
            return "       na"

    all_hard, all_top3, all_ew, all_13b, all_v2 = [], [], [], [], []

    for _, row in pivot.iterrows():
        fid  = int(row["fold"])
        year = str(row.get("test_year", "?"))
        hard = row.get(sharpe_col_hard, float("nan"))
        top3 = row.get(sharpe_col_top3, float("nan"))
        ew   = row.get(sharpe_col_ew,   float("nan"))
        b    = prior_13b.get(fid,   float("nan"))
        v2   = prior_13bv2.get(fid, float("nan"))

        all_hard.append(hard); all_top3.append(top3)
        all_ew.append(ew);     all_13b.append(b)
        all_v2.append(v2)

        print(
            f"{fid:>5}  {year:>5}  "
            f"{_fmt(hard)}  {_fmt(top3)}  {_fmt(ew)}  "
            f"{_fmt(b)}  {_fmt(v2)}"
        )

    print(sep)

    def _avg(lst):
        fs = [x for x in lst if not np.isnan(float(x))]
        return float(np.mean(fs)) if fs else float("nan")

    def _med(lst):
        fs = [x for x in lst if not np.isnan(float(x))]
        return float(np.median(fs)) if fs else float("nan")

    def _std(lst):
        fs = [x for x in lst if not np.isnan(float(x))]
        return float(np.std(fs, ddof=1)) if len(fs) > 1 else float("nan")

    print(
        f"{'AVG':>5}  {'':>5}  "
        f"{_fmt(_avg(all_hard))}  {_fmt(_avg(all_top3))}  {_fmt(_avg(all_ew))}  "
        f"{_fmt(_avg(all_13b))}  {_fmt(_avg(all_v2))}"
    )
    print(
        f"{'MED':>5}  {'':>5}  "
        f"{_fmt(_med(all_hard))}  {_fmt(_med(all_top3))}  {_fmt(_med(all_ew))}  "
        f"{'':>8}  {'':>9}"
    )
    print(
        f"{'STD':>5}  {'':>5}  "
        f"{_fmt(_std(all_hard))}  {_fmt(_std(all_top3))}  {_fmt(_std(all_ew))}  "
        f"{'':>8}  {'':>9}"
    )
    print("=" * 80)

    n = len(pivot)
    v3_beats_ew  = sum(1 for s in all_hard if not np.isnan(s) and s > 0)
    v3_beats_13b = sum(1 for s, b in zip(all_hard, all_13b)
                       if not np.isnan(s) and not np.isnan(b) and s > b)
    print(f"\nDiagnostics ({n} folds):")
    print(f"  Hard v3 > EW (Sharpe > 0) : {v3_beats_ew}/{n}")
    print(f"  Hard v3 > 13b (grid)       : {v3_beats_13b}/{n}")
    print("=" * 80)


def main() -> None:
    print(f"\nAggregating Plan 13b-v3 results from: {OUT_DIR}", flush=True)

    df_raw = load_fold_results(OUT_DIR)
    if df_raw.empty:
        print("  No fold result files found — nothing to aggregate.", flush=True)
        return

    folds_found = sorted(df_raw["fold"].unique())
    print(f"  Folds found: {folds_found}", flush=True)

    pivot = summarise(df_raw)

    # Load prior plans
    prior_13b   = _load_prior(PRIOR["13b"],    "hard_sharpe")
    prior_13bv2 = _load_prior(PRIOR["13b-v2"], "hard_sharpe")

    print_summary_table(pivot, prior_13b, prior_13bv2)

    # Save summary
    pivot.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved summary to: {SUMMARY_CSV}", flush=True)

    # Also save a simplified version matching prior plan format
    simple_rows = []
    for _, row in pivot.iterrows():
        simple_rows.append({
            "fold":         int(row["fold"]),
            "test_year":    row.get("test_year", "?"),
            "hard_sharpe":  row.get("hard_bo_sharpe",  float("nan")),
            "blend_sharpe": row.get("top3_bo_sharpe",  float("nan")),
            "ew_sharpe":    row.get("ew_sharpe",        float("nan")),
        })
    simple_df = pd.DataFrame(simple_rows)
    simple_path = os.path.join(OUT_DIR, "summary_metrics.csv")
    simple_df.to_csv(simple_path, index=False)
    print(f"Saved summary_metrics.csv (simple format) to: {simple_path}", flush=True)


if __name__ == "__main__":
    main()
