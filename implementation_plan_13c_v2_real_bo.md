# Implementation Plan 13c-v2: Real Bayesian Optimization with Warm Start

**Author:** Bene
**Date:** April 2026
**Supersedes:** Implementation Plan 13c (Hybrid with exhaustive within-tier search)
**Related:** Plan 13a (Hierarchical Meta-Learner), Plan 13b-v3 (BO with Val-Split, structurally failed)

---

## 1. Motivation

### 1.1 What Prof. Ler Specified

From the call notes (direct quotes):

> *"substitute this part with the bayesian optimization: so i dont have to build a metalearner: everything is determined by the bayesian optimization process"*
>
> *"try all 3 options:*
> *1. craft your own metalearner; dont combine them → independent!!! not using bayesian optimization; only the metalearner*
> *2. just do the 1st: metalearner only on modelfamily, then substitute the algoselection with bayesian network; then*
> *3. no metalearner at all; only bayesian network"*
>
> *"→ use the baesian optimizer, but not start from scratch — with warm start from option 1"*

**Option 2 (this plan) is specified as:**
- Meta-Learner on model family level (tier selection) — reuses pre-trained TierSelector from Plan 13a
- Bayesian Optimization substitutes algorithm selection within tier
- **Warm-started from Option 1's outputs** (this is the key differentiator)

### 1.2 Why This Is Different From Plan 13b-v3

| Aspect | Plan 13b-v3 | Plan 13c-v2 |
|:-:|:-:|:-:|
| BO search space | 13 families + all HPs | 3-7 algorithms + HPs per tier |
| BO studies per fold | 4 (1 per regime) | 12 (1 per regime × tier) |
| Meta-Learner role | None | Selects tier weights β ∈ Δ3 |
| Warm start | None (cold start) | **Seed trials from Plan 13a's tier selections** |
| Search space structure | Flat over all families | Hierarchical: tier → algo → HPs |

**The critical methodological novelty is warm-starting.** Plan 13b-v3 proved that cold BO over a large search space finds no signal on this 5-ETF problem. The hypothesis for 13c-v2 is that warm-starting with Meta-Learner priors + smaller per-tier search spaces provides enough structure for BO to find robust configurations.

### 1.3 Why This Is Different From Plan 13c (Old)

| Aspect | Plan 13c (old) | Plan 13c-v2 |
|:-:|:-:|:-:|
| Within-tier search | Exhaustive over discrete configs | **Optuna TPE (real BO)** |
| HP exploration | Fixed config grid | **Continuous HP space** |
| Warm start | N/A (no BO) | **Seed trials from Plan 13a** |
| Validation | Used train-fold evaluation | **Chronological 80/20 val-split** |

Plan 13c (old) called its within-tier search "BO" but it was actually exhaustive enumeration over ~81 pre-defined configs. Plan 13c-v2 implements what Prof. Ler actually specified.

---

## 2. Architecture

### 2.1 High-Level Flow

```
Training Phase (per Walk-Forward Fold):
┌─────────────────────────────────────────────────────┐
│ Phase 0 (pre-trained):                              │
│   TierSelector NN from Plan 13a                     │
│   Frozen, reused as-is                              │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│ Phase 1: BO per (Regime, Tier)                      │
│   12 Optuna Studies (4 regimes × 3 tiers)           │
│   Each study:                                       │
│     - Suggests algorithm within tier (categorical)  │
│     - Suggests HPs conditional on algorithm         │
│     - Evaluated on chronological 80/20 val-split    │
│     - Warm-started with seed trials from Plan 13a   │
└─────────────────────────────────────────────────────┘
                       ↓
              best_config[regime][tier]
              (a lookup table of 12 optimized configs)

Decision Phase (per test day t):
┌─────────────────────────────────────────────────────┐
│ 1. Regime Classifier → p_t ∈ [0,1]                  │
│    → map to regime s_t ∈ {Calm, Normal, Tense,     │
│      Crisis}                                         │
│                                                      │
│ 2. TierSelector NN (Plan 13a) → β_t ∈ Δ3            │
│    (tier mixing weights)                            │
│                                                      │
│ 3. Per tier k: lookup BO's best config:             │
│    A_k = best_config[s_t][tier_k]                   │
│    → w_t^(k) = A_k(context)                         │
│                                                      │
│ 4. Final portfolio:                                 │
│    w_t = Σ_k β_{t,k} × w_t^(k)                      │
└─────────────────────────────────────────────────────┘
```

### 2.2 Key Components

**TierSelector NN (from Plan 13a, frozen):**
- Architecture: FFNN [64, 32], input size 29 (25 asset features + 4 one-hot regime)
- Output: Softmax over 3 tiers → β ∈ Δ3
- Reused as-is; NOT retrained in this plan

**BO per (Regime, Tier) — 12 independent Optuna studies:**
- Sampler: TPE (Tree-structured Parzen Estimator, Bergstra et al. 2011)
- Pruner: MedianPruner (n_startup_trials=5)
- Objective: Sharpe ratio on chronological 80/20 val-split
- Warm start: 3-5 seed trials from Plan 13a tier-winners

---

## 3. BO Search Space per Tier

### 3.1 Tier 1 (Heuristics) — 30 Trials

Conditional search space:

```python
def objective_tier1(trial):
    algo = trial.suggest_categorical("algo", [
        "EqualWeight", "MinVariance", "MeanVariance",
        "MaxDiversification", "RiskParity", "Momentum", "TrendFollowing"
    ])

    if algo == "EqualWeight":
        pass  # no HPs
    elif algo == "MinVariance":
        lookback = trial.suggest_int("lookback", 5, 252)
    elif algo == "MeanVariance":
        lookback = trial.suggest_int("lookback", 5, 252)
        gamma = trial.suggest_float("gamma", 0.01, 10.0, log=True)
    elif algo == "MaxDiversification":
        lookback = trial.suggest_int("lookback", 5, 252)
    elif algo == "RiskParity":
        lookback = trial.suggest_int("lookback", 5, 252)
    elif algo == "Momentum":
        lookback = trial.suggest_int("lookback", 5, 252)
    elif algo == "TrendFollowing":
        lookback = trial.suggest_int("lookback", 5, 252)

    return evaluate_sharpe(algo, hps, val_split)
```

### 3.2 Tier 2 (Linear ML) — 50 Trials

```python
def objective_tier2(trial):
    algo = trial.suggest_categorical("algo", ["Lasso", "Ridge", "ElasticNet"])
    lookback = trial.suggest_int("lookback", 20, 252)

    if algo == "Lasso":
        alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
    elif algo == "Ridge":
        alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
    elif algo == "ElasticNet":
        alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

    return evaluate_sharpe(algo, hps, val_split)
```

### 3.3 Tier 3 (Non-Linear ML) — 80 Trials

```python
def objective_tier3(trial):
    algo = trial.suggest_categorical("algo", [
        "RandomForest", "GradientBoosting", "MLP"
    ])
    lookback = trial.suggest_int("lookback", 20, 252)

    if algo == "RandomForest":
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    elif algo == "GradientBoosting":
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.3, log=True)
    elif algo == "MLP":
        hidden_size = trial.suggest_int("hidden_size", 32, 256)
        n_layers = trial.suggest_int("n_layers", 1, 3)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    return evaluate_sharpe(algo, hps, val_split)
```

### 3.4 Total Trial Budget

| Tier | Trials per Study | Studies per Fold (4 regimes) | Total per Fold |
|:-:|:-:|:-:|:-:|
| Tier 1 | 30 | 4 | 120 |
| Tier 2 | 50 | 4 | 200 |
| Tier 3 | 80 | 4 | 320 |
| **Total per fold** | — | **12** | **640** |
| **Total 12 folds** | — | **144** | **7,680 trials** |

With MedianPruner pruning ~40% of trials, effective training time ~5 hours per fold.

---

## 4. Warm Start Mechanism

### 4.1 The Key Innovation vs. Plan 13b-v3

Plan 13b-v3 failed partly because Optuna TPE had to cold-start its exploration of a large search space. For the first ~20-30 trials, TPE is essentially random. On a signal-sparse problem like this one, those random trials poison the posterior and TPE never recovers.

**Plan 13c-v2 uses Plan 13a's Meta-Learner outputs as informed priors.**

### 4.2 Mechanism

```python
def create_warm_start_trials(regime, tier, plan_13a_results):
    """
    Extract the top-3 configs that Plan 13a's TierSelector most frequently
    chose in this (regime, tier) combination during training.
    """
    # From Plan 13a's training logs: which algos+HPs were active
    # when TierSelector gave high β for this tier in this regime?
    tier_winners = plan_13a_results[regime][tier].top_k(k=3)

    seed_trials = []
    for winner in tier_winners:
        seed_trials.append({
            "algo": winner["algo"],
            "lookback": winner["lookback"],
            # ... other HPs from winner
        })
    return seed_trials

# Applied to study:
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    direction="maximize"
)

# Enqueue warm-start trials BEFORE TPE takes over
for seed in create_warm_start_trials(regime, tier, plan_13a_results):
    study.enqueue_trial(seed)

study.optimize(objective, n_trials=TOTAL_TRIALS)
```

### 4.3 Why This Should Help

1. **First 3-5 trials are informed** — not random
2. **TPE builds better posterior** — early good trials anchor the search
3. **Handles small sample regime** — only ~100-200 actual TPE trials, making initial seeds more influential
4. **Follows Prof's explicit instruction** — *"warm start from option 1"*

### 4.4 Ablation (Optional but Valuable)

If time permits, run 13c-v2 both with and without warm start:
- If warm start doesn't help → structural BO problem confirmed (same as 13b-v3)
- If warm start helps significantly → validates Prof's Option 2 methodology

---

## 5. Validation Framework

### 5.1 Chronological 80/20 Val-Split (from Plan 13b-v3)

```
Per fold training block (e.g., 2006-2012):
┌──────────────────────────────────┬──────────────┐
│ Inner Train (80%): 2006-2011     │ Val (20%)    │
│                                  │ 2012         │
└──────────────────────────────────┴──────────────┘

Per regime:
  - Filter days where s_t == regime_k within inner train
  - Filter days where s_t == regime_k within val
  - BO fits algorithm on inner-train-regime days
  - BO evaluates Sharpe on val-regime days
```

### 5.2 Min Regime Size Fallback

If `len(val_regime_days) < 20`:
- Skip BO for this (regime, tier)
- Fallback to EqualWeight for this (regime, tier)
- Log fallback event

This was NOT triggered in 13b-v3 (sufficient regime representation with min=100), but we reduce the threshold here since per-tier optimization is more forgiving.

### 5.3 Hyperparameter Stability Diagnostics

**Critical post-hoc analysis (Lesson from 13b-v3):**

For each (regime, tier), collect selected HPs across folds and compute:

- **Coefficient of Variation (CoV)** for continuous HPs (lookback, alpha, etc.)
- **Algorithm selection frequency** for categorical (which algo chosen how often)

**Interpretation:**
- CoV < 20%: Stable, BO finds consistent optima
- CoV 20-30%: Moderately stable, acceptable
- CoV > 30%: **Red flag** — BO is fitting noise, same failure mode as 13b-v3

Document these in the final report regardless of performance outcome.

---

## 6. Validation Pipeline

### 6.1 12-Fold Walk-Forward (unchanged)

| Fold | Train Period | Test Period |
|:-:|:-:|:-:|
| 1 | 2006-2012 | 2013 |
| 2 | 2006-2013 | 2014 |
| ... | ... | ... |
| 12 | 2006-2023 | 2024 |

All folds restart BO from scratch (except warm-start seeds from Plan 13a, which are fold-agnostic).

### 6.2 Per-Fold Execution (mandatory for memory management)

From Lesson 3 (13b-v2 crashed after 7h due to memory leak):

```powershell
# Fresh Python process per fold
python -u -m experiments.plan13c_v2_real_bo --fold 1 *> "results/plan13c_v2/fold1.log"
python -u -m experiments.plan13c_v2_real_bo --fold 2 *> "results/plan13c_v2/fold2.log"
# ... etc.
```

Each fold script:
- Loads Plan 13a's TierSelector
- Runs 12 BO studies
- Writes `fold_{k}_result.csv`, `fold_{k}_best_configs.json`, `fold_{k}_trial_log.csv` immediately after completion
- Supports `--skip-existing` flag for resume logic
- Calls `del study; gc.collect()` between studies to free memory

---

## 7. Success Criteria & Decision Rules

### 7.1 Phase-Gated Execution

**Phase A: Smoke Test (before any full run)**
- Run Fold 1 with 10 trials per tier instead of 30/50/80
- Verify all infrastructure: val-split, output files, NaN handling, memory cleanup
- Check that `train_sharpe != 0.0` for all trials (NaN-bug detector from Plan 13b-v3)
- Expected runtime: 15-20 minutes
- **Gate:** If any mechanical failure → fix and re-run smoke

**Phase B: Diagnostic Folds (1-3)**
- Run Folds 1, 2, 3 with full trial budget
- After Fold 3: evaluate decision rule below
- Expected runtime: ~15 hours

**Phase C: Full Run (Folds 4-12)**
- Only if Phase B passes decision rule
- Expected runtime: ~45 hours

### 7.2 Decision Rule (After Phase B)

Compute Avg Sharpe across Folds 1-3:

| Avg Sharpe (Folds 1-3) | Decision |
|:-:|:-:|
| < 0.00 | **STOP.** Same pattern as 13b-v3. Pivot to new approach. |
| 0.00 – 0.50 | **CAUTION.** Continue but with skepticism; prepare pivot narrative. |
| 0.50 – 0.88 | **CONTINUE.** Comparable to Plan 13c (old). |
| > 0.88 | **CONTINUE.** Real BO provides value over exhaustive search. |
| > 1.02 (= EW) | **Exceptional.** First method to beat EW. |

This decision rule mirrors Plan 13b-v3's decision rule, which correctly triggered abort and saved ~40 hours of compute.

### 7.3 Quality Gates Beyond Sharpe

Even if Sharpe is acceptable, check HP stability:

- If CoV > 30% for most (regime, tier) combinations across Folds 1-3:
  - BO is fitting noise, not signal
  - Performance will likely degrade on unseen folds
  - **Document and continue carefully** — this would be a valuable negative finding

---

## 8. Expected Outcomes (Honest Assessment)

Based on the methodological heritage:

**Optimistic scenario (~20% probability):**
- Warm start + smaller per-tier search space gives BO enough signal
- Avg Sharpe: +0.90 to +1.05
- Beats Plan 13c (+0.88), potentially beats EW (+1.02)
- HP stability: CoV < 25% for most (regime, tier)

**Realistic scenario (~50% probability):**
- Avg Sharpe: +0.80 to +0.95
- Comparable to Plan 13c (+0.88)
- HP stability: CoV 20-40% (mixed results)
- **Interpretation:** Real BO provides marginal improvement over exhaustive search

**Pessimistic scenario (~30% probability):**
- Avg Sharpe: +0.30 to +0.75 (or worse)
- Warm start insufficient to overcome signal-sparsity
- HP stability: CoV > 40% (same as 13b-v3)
- **Interpretation:** HP tuning has limited value on 5-ETF problem, regardless of search space granularity

All three outcomes produce valuable thesis contributions. The pessimistic case would be the strongest negative finding: *"We tested all three Options systematically (13a, 13b-v3, 13c-v2) and none beat Equal Weight. The bottleneck is search-space signal, not optimization methodology."*

---

## 9. Baselines for Final Comparison

| Method | Avg Sharpe | Role |
|:-:|:-:|:-:|
| Oracle Per-Regime | +1.65 | Upper bound (hindsight) |
| **Equal Weight** | **+1.02** | **Primary baseline** |
| Plan 13c (old, exhaustive) | +0.88 | **Direct ablation comparison for BO** |
| Plan 13b (exhaustive blend) | +0.79 | Option 3 baseline |
| Plan 13b (exhaustive hard) | +0.75 | Option 3 baseline |
| **Plan 13c-v2 (real BO)** | **? ? ?** | **The candidate** |
| Plan 13a (hierarchical NN) | -0.65 | Option 1 baseline |
| Plan 13b-v3 (BO + val-split) | -0.61 | **Negative reference** |

**Key comparison:** Plan 13c-v2 vs. Plan 13c (old) isolates the effect of real BO. Plan 13c-v2 vs. Plan 13b-v3 isolates the effect of smaller search space + warm start.

---

## 10. Lessons Learned Applied

All 7 lessons from Plan 13b-v3 are applied:

- ✅ **Lesson 1** — Val-split is mandatory (chronological 80/20)
- ✅ **Lesson 2** — HP stability measured per fold (CoV diagnostic)
- ✅ **Lesson 3** — BO runtime controlled (per-fold CLI, immediate persistence, memory cleanup, `--skip-existing`)
- ✅ **Lesson 4** — NaN-handling in predict path (`finite_mask`, `return -inf` for failures)
- ✅ **Lesson 5** — Smoke test before full run (Phase A)
- ✅ **Lesson 6** — Fresh Python process per fold (PowerShell invocation)
- ✅ **Lesson 7** — Decision rule after diagnostic phase (Phase B gate)

---

## 11. Deliverables

### 11.1 Code Files

```
experiments/plan13c_v2_real_bo.py        # Main experiment script
experiments/plan13c_v2_aggregate.py       # Aggregate fold results
experiments/plan13c_v2_stability.py       # HP stability diagnostic
```

### 11.2 Output Files (per Fold)

```
results/plan13c_v2/
  ├── fold_01_result.csv           # Per-fold summary metrics
  ├── fold_01_best_configs.json    # 12 BO-optimized configs
  ├── fold_01_trial_log.csv        # All BO trials (for diagnostics)
  ├── fold_01_val_test_correlation.json  # Val-test Pearson r
  └── optuna_studies/
      ├── fold_01_calm_tier1.pkl
      ├── fold_01_calm_tier2.pkl
      └── ...
```

### 11.3 Analysis Artifacts

```
results/plan13c_v2/analysis/
  ├── hp_stability_report.md        # CoV per (regime, tier)
  ├── algorithm_selection_freq.csv  # Which algos chosen how often
  ├── comparison_table.csv          # 13c-v2 vs all baselines
  └── warm_start_ablation.md        # If ablation run
```

---

## 12. Timeline Estimate

| Phase | Task | Duration |
|:-:|:-:|:-:|
| Setup | Refactor algorithm_factory, integrate Plan 13a warm-start loader | 2-3 hours |
| Phase A | Smoke test Fold 1 (10 trials/tier) | 20 min |
| Phase B | Folds 1-3 full run | ~15 hours |
| Decision | Analyze Phase B, apply decision rule | 1 hour |
| Phase C (if green) | Folds 4-12 | ~45 hours |
| Analysis | Aggregate, stability report, comparison | 3-4 hours |
| **Total (optimistic)** | All 12 folds complete | **~65-70 hours** |
| **Total (pessimistic)** | Abort after Phase B | **~17 hours** |

---

## 13. Open Design Decisions (for Record)

These were resolved during design discussion but documented here for traceability:

| Decision | Choice | Rationale |
|:-:|:-:|:-:|
| TierSelector architecture | [64, 32] (from 13a) | Consistency with 13a, enables clean ablation |
| TierSelector training | Frozen (reused from 13a) | Follows Prof's "just do the 1st" instruction |
| BO granularity | 12 studies per fold (regime × tier) | Smallest search space, regime-specific |
| Within-tier selection | BO (hard — 1 best config per tier, regime) | Soft-mix would add second HP layer |
| Val-split | Chronological 80/20 | Matches 13b-v3, Feurer & Hutter (2019) standard |
| Warm start source | Plan 13a top-K configs | Prof's explicit instruction |
| Trial budget | 30/50/80 per tier | Scales with HP dimensionality |

---

## 14. References

- **Bergstra et al. (2011)** — Algorithms for Hyper-Parameter Optimization (TPE)
- **Feurer & Hutter (2019)** — Hyperparameter Optimization (AutoML book chapter)
- **Plan 13a** — Hierarchical Meta-Learner (Option 1 implementation)
- **Plan 13b-v3** — BO with Val-Split (Option 3 implementation, structurally failed)
- **Plan 13c (old)** — Hybrid with exhaustive within-tier search (precursor to this plan)

---

**END OF IMPLEMENTATION PLAN 13c-v2**
