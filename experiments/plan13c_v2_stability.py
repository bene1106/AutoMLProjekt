# experiments/plan13c_v2_stability.py -- Plan 13c-v2: HP Stability Diagnostic
#
# Computes Coefficient of Variation (CoV) of chosen HPs across folds.
# CoV > 30% = BO fitting noise, not signal (same threshold as 13b-v3).
#
# Usage:
#   cd Implementierung1
#   python -m experiments.plan13c_v2_stability

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR  = Path(__file__).parent.parent / "results" / "plan13c_v2"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

REGIMES = {1: "Calm", 2: "Normal", 3: "Tense", 4: "Crisis"}
TIERS   = [1, 2, 3]


def load_best_configs(folds: list[int]) -> dict:
    """Load {fold: best_configs_dict} from fold_NN_best_configs.json."""
    configs = {}
    for fold in folds:
        path = RESULTS_DIR / f"fold_{fold:02d}_best_configs.json"
        if path.exists():
            with open(path) as f:
                configs[fold] = json.load(f)
        else:
            print(f"  [MISSING] fold_{fold:02d}_best_configs.json")
    return configs


def extract_hp_values(configs: dict, regime_id: int, tier: int) -> dict:
    """
    For a given (regime, tier), extract HP values across all folds.
    Returns dict of {hp_name: [values across folds]}.
    """
    key = f"r{regime_id}_t{tier}"
    hp_lists = {}
    algo_list = []

    for fold, fold_configs in configs.items():
        entry = fold_configs.get(key, {})
        family = entry.get("family", "Unknown")
        algo_list.append(family)

        config = entry.get("config", {})
        for hp_name, hp_val in config.items():
            if hp_name in ("tier", "family"):
                continue
            if isinstance(hp_val, (int, float)):
                if hp_name not in hp_lists:
                    hp_lists[hp_name] = []
                hp_lists[hp_name].append(float(hp_val))

    return algo_list, hp_lists


def compute_cov(values: list) -> float | None:
    if len(values) < 2:
        return None
    arr = np.array(values)
    mean = np.mean(arr)
    if abs(mean) < 1e-10:
        return None
    return float(np.std(arr) / abs(mean))


def main():
    available_folds = [f for f in range(1, 13)
                       if (RESULTS_DIR / f"fold_{f:02d}_best_configs.json").exists()]

    print(f"Available folds for stability analysis: {available_folds}")
    if len(available_folds) < 2:
        print("Need at least 2 folds for stability analysis.")
        return

    configs = load_best_configs(available_folds)

    # ── Per (regime, tier) stability ──────────────────────────────────────────
    rows = []
    print("\n" + "=" * 70)
    print("HP STABILITY REPORT (CoV across folds)")
    print("=" * 70)
    print(f"  CoV < 20%  : Stable — BO finds consistent optima")
    print(f"  CoV 20-30% : Moderate — acceptable")
    print(f"  CoV > 30%  : RED FLAG — BO fitting noise (same as 13b-v3)")
    print("=" * 70)

    for regime_id, regime_name in REGIMES.items():
        for tier in TIERS:
            algo_list, hp_lists = extract_hp_values(configs, regime_id, tier)

            # Algorithm stability (how often same algo chosen)
            if algo_list:
                most_common = max(set(algo_list), key=algo_list.count)
                algo_stability = algo_list.count(most_common) / len(algo_list)
            else:
                most_common = "?"
                algo_stability = float("nan")

            # HP stability (CoV per HP)
            hp_covs = {}
            for hp_name, values in hp_lists.items():
                cov = compute_cov(values)
                if cov is not None:
                    hp_covs[hp_name] = cov

            # Find the primary HP for CoV reporting
            # For Tier 1: lookback variants; for Tier 2/3: lookback
            primary_hp = None
            primary_cov = None
            lookback_keys = [k for k in hp_covs if "lookback" in k]
            if lookback_keys:
                primary_hp  = lookback_keys[0]
                primary_cov = hp_covs[primary_hp]
            elif hp_covs:
                primary_hp  = list(hp_covs.keys())[0]
                primary_cov = hp_covs[primary_hp]

            # Flag
            flag = ""
            if primary_cov is not None:
                if primary_cov > 0.30:
                    flag = " *** RED FLAG"
                elif primary_cov > 0.20:
                    flag = " ** MODERATE"

            rows.append({
                "regime_id":       regime_id,
                "regime":          regime_name,
                "tier":            tier,
                "algo_mode":       most_common,
                "algo_stability":  round(float(algo_stability), 3),
                "algo_list":       str(algo_list),
                "primary_hp":      primary_hp,
                "primary_hp_cov":  round(float(primary_cov), 4) if primary_cov else None,
                **{f"hp_{k}_cov": round(v, 4) for k, v in hp_covs.items()},
            })

            print(
                f"  Regime {regime_id} ({regime_name:7s}) | Tier {tier}:"
                f"  algo={most_common:<20s}  stability={algo_stability:.2f}"
                f"  {primary_hp or '---'}CoV={primary_cov:.2f}" if primary_cov
                else f"  Regime {regime_id} ({regime_name:7s}) | Tier {tier}:"
                     f"  algo={most_common:<20s}  stability={algo_stability:.2f}",
                end=flag + "\n", flush=True,
            )

    print("=" * 70)

    # ── Save stability table ──────────────────────────────────────────────────
    stability_df = pd.DataFrame(rows)
    out_path = ANALYSIS_DIR / f"hp_stability_folds{'_'.join(str(f) for f in available_folds[:3])}.csv"
    stability_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # ── Summary statistics ────────────────────────────────────────────────────
    cov_values = [r["primary_hp_cov"] for r in rows
                  if r.get("primary_hp_cov") is not None]
    if cov_values:
        print(f"\nCoV Summary across all (regime, tier) combinations:")
        print(f"  Mean CoV:   {np.mean(cov_values):.3f}")
        print(f"  Max CoV:    {np.max(cov_values):.3f}")
        print(f"  Red flags (CoV > 0.30): "
              f"{sum(1 for v in cov_values if v > 0.30)}/{len(cov_values)}")

    # ── Also report algorithm selection frequency per tier ────────────────────
    print("\nAlgorithm selection frequency by tier:")
    for tier in TIERS:
        tier_rows = [r for r in rows if r["tier"] == tier]
        all_algos = []
        for r in tier_rows:
            try:
                algos = r["algo_list"].strip("[]").replace("'", "").split(", ")
                all_algos.extend(algos)
            except Exception:
                pass
        if all_algos:
            from collections import Counter
            counts = Counter(all_algos)
            total = sum(counts.values())
            print(f"  Tier {tier}:")
            for algo, cnt in counts.most_common():
                print(f"    {algo:<30s}: {cnt}/{total} ({cnt/total:.1%})")


if __name__ == "__main__":
    main()
