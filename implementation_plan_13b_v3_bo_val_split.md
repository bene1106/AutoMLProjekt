# Implementation Plan 13b-v3: Bayesian Optimization with Inner Validation Split

**Status:** Planned
**Supersedes:** Plan 13b-v2 (retained in repo as historical baseline)
**Expected runtime:** ~5-6 hours for 12 folds
**Author:** Ben | Supervisor: Prof. Daren Ler

---

## 1. Motivation

### 1.1 What Plan 13b-v2 Showed

Plan 13b-v2 implemented true Bayesian Optimization (Optuna TPE) over a continuous
hyperparameter space across 13 algorithm families, replacing the exhaustive search
over 117 fixed configurations from Plan 13b. The partial run (Fold 1 + partial Fold 2)
revealed a severe train-test divergence:

| Regime | Train Sharpe | Best Algorithm          |
|--------|--------------|--------------------------|
| Calm   | +1.36        | RandomForest             |
| Normal | +2.23        | MinVariance              |
| Tense  | +0.96        | MaxDiversification       |
| Crisis | +1.65        | GradientBoosting         |
| **Avg** | **+1.55**   |                          |

**Fold 1 Test result:**
- Hard BO Sharpe: **-0.23**
- Top-3 BO Sharpe: **-0.23**
- Equal Weight: +0.00

Train-Test gap of ~1.78 Sharpe points indicates systematic overfitting of the
TPE sampler to the regime-specific training data.

### 1.2 Root Cause

Plan 13b-v2 evaluated each sampled hyperparameter configuration on the **full
regime-filtered training period**, returning the in-sample Sharpe ratio to
Optuna. With 200 trials per regime over a continuous search space, TPE
aggressively converged on configurations that captured idiosyncratic patterns
in the training data — particularly for regimes with small training samples
(Crisis: 291 days).

Plan 13b (exhaustive search) was less susceptible to this because:
1. Discrete grid of 117 configurations acted as implicit regularization.
2. No adaptive sampling → no "exploitation" of training-set pathologies.

In contrast, true BO without a validation split is structurally prone to
overfit on small samples, which is well-documented in the AutoML literature
(e.g., Feurer & Hutter, 2019; Snoek et al., 2012).

### 1.3 Hypothesis

Introducing an inner chronological train/validation split per regime — where
Optuna optimizes on *validation* Sharpe rather than training Sharpe — will:

- Reduce the train-test gap substantially
- Produce test-set performance comparable to or better than Plan 13b (+0.75 avg)
- Lead TPE to favor more conservative configurations (fewer high-capacity
  Tier 3 models, more Tier 1 heuristics)

---

## 2. Key Changes vs. Plan 13b-v2

### 2.1 Inner Validation Split (Primary Change)

For each regime within each fold's training period:

1. Filter training data to days where `regime == r` (chronologically ordered)
2. Split chronologically:
   - **Inner Train:** first 80% of regime days
   - **Inner Validation:** last 20% of regime days
3. Optuna optimizes on **inner validation Sharpe**:
   - Each trial: sample config → fit on inner train → evaluate on inner val
   - TPE uses val Sharpe as the objective value
4. After BO completes, best config is **refit on the full regime training data**
   (inner train + inner val combined) before deploying on the test set.

**Rationale for refit:** The val split is only used for model selection.
Once the best config is identified, training on all available regime data
maximizes the information used.

### 2.2 Reduced Trial Budget

Trials per regime: **200 → 100**.

Rationale:
- TPE literature (Bergstra & Bengio, 2012; Akiba et al., 2019) reports
  convergence within 50-100 trials on typical ML problems
- Less aggressive sampling reduces overfitting risk on the inner val set itself
- Halves runtime per regime without meaningful loss in search quality

### 2.3 Minimum Regime Size Fallback

If `regime_days_count < 100`:
- Skip BO for that regime
- Use `EqualWeight` as fallback algorithm
- Log the fallback

Rationale: With < 100 days, an 80/20 split leaves < 20 validation days —
not enough for reliable Sharpe estimation. Falling back to EW prevents
forcing BO on underpowered data.

### 2.4 Early Stopping Within Optuna

Add a median-stopping pruner:

```python
pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0)
```

Trials performing worse than the median after 20 warmup trials are pruned
early. Reduces runtime on clearly bad configs without affecting quality.

### 2.5 Per-Fold Resilient Execution

Carries forward the resilience fixes intended for Plan 13b-v2:

- `--fold N` CLI argument for single-fold runs
- Per-fold CSV/JSON persistence *immediately* after each fold completes
- `--skip-existing` flag that bypasses folds with existing result files
- Explicit `gc.collect()` and `del bo_studies` at end of each fold
- Memory logging (`psutil.Process().memory_info().rss`)

### 2.6 Warm Start (Unchanged)

Continue using Plan 13b's best configurations as TPE seed configurations.
Only the *evaluation metric* changes (val instead of train Sharpe) — the
search space and seed configs remain identical.

---

## 3. What Stays the Same

- **Algorithm space:** 13 families across Tiers 1–3 (~81 canonical algos,
  expanded to continuous HP space)
- **Asset universe:** SPY, TLT, GLD, EFA, VNQ (5 ETFs)
- **Regime definition:** Oracle regimes from VIX thresholds [15, 20, 30]
- **Walk-forward scheme:** 12 folds, expanding window, test years 2013–2024
- **Switching cost:** κ = 0.001
- **Test strategies:** Hard BO (best config per regime), Top-3 Blend
  (equal-weight mix of top-3 val-Sharpe configs per regime)
- **Batch precompute:** 21.6× speedup from Plan 13a/13b retained

---

## 4. Methodological Positioning

### 4.1 Why This is the Correct BO Baseline

Plan 13b-v2 represents "BO without regularization" — a design error in retrospect.
Plan 13b-v3 represents "BO with methodologically sound regularization via held-out
validation". This is the standard AutoML practice (Feurer & Hutter, 2019; Auto-sklearn;
Hyperopt-Sklearn).

### 4.2 Comparison Matrix for the Thesis

Plan 13b-v3 enables a clean three-way comparison:

| Plan    | Method                               | Implicit Regularization    | Train-Test Gap |
|---------|--------------------------------------|----------------------------|---------------|
| 13b     | Exhaustive search (117 discrete)     | Coarse grid                | Small         |
| 13b-v2  | BO, no val-split                     | None                       | Large         |
| 13b-v3  | BO + inner val-split                 | Held-out validation        | (hypothesis: small) |

This trio directly tests whether BO *with sound methodology* outperforms
the coarser exhaustive search.

### 4.3 Limitations to Acknowledge in the Thesis

- Single val-split (no inner CV) → val-set Sharpe itself is a noisy estimate
- 80/20 temporal split assumes stationarity *within* a regime — reasonable
  but not guaranteed (e.g., volatility clustering within Crisis periods)
- Fallback threshold of 100 days is a practical choice, not theoretically
  derived

---

## 5. Expected Runtime

| Component                        | Plan 13b-v2 | Plan 13b-v3 |
|----------------------------------|-------------|-------------|
| Trials per regime                | 200         | 100         |
| Pruning                          | None        | MedianPruner |
| Validation overhead              | None        | +~15%       |
| Net runtime per regime (avg)     | ~30 min     | ~15 min     |
| Total runtime (12 folds)         | ~9 h        | **~5-6 h** |

---

## 6. Evaluation & Decision Protocol

### 6.1 Diagnostic Phase (Folds 1–3)

Run Folds 1, 2, 3 in sequential fresh processes. Compare test Sharpes
against Plan 13b and Equal Weight.

**Decision rule:**

| Avg test Sharpe (F1-F3) | Action                                      |
|--------------------------|---------------------------------------------|
| > +0.80                  | Proceed to Folds 4–12 (Plan 13b-v3 wins)    |
| +0.40 to +0.80           | Proceed cautiously, reassess after Fold 6  |
| +0.00 to +0.40           | Investigate — possibly reduce trials further |
| < +0.00                  | BO fundamentally unsuitable → pivot to 13c-v3 |

### 6.2 Final Phase (Folds 4–12)

If diagnostic phase passes: chain Folds 4–12 in a loop of fresh Python processes
(see Phase 3 of Claude Code prompt). Expected overnight run.

### 6.3 Analysis Artifacts

For each fold, record:
- `train_sharpe` and `val_sharpe` for best config per regime
- Train-val gap per regime (diagnostic for residual overfitting)
- Selected algorithm family per regime
- Test-set Sharpe, Sortino, Max Drawdown, Turnover

Aggregate across folds into `plan13b_v3_summary.csv` matching the format
of prior plans.

---

## 7. Success Criteria

**Primary:** Average test Sharpe across 12 folds > Plan 13b's +0.75.

**Secondary:**
- Mean train-val gap < 0.5 Sharpe (indicating BO found generalizable configs)
- At least 4/12 folds beat Equal Weight (Plan 13b: 3/12)
- Algorithm selection is interpretable (not dominated by high-capacity models
  in small-sample regimes)

**Negative result (also valuable):** If 13b-v3 does *not* beat 13b, this
constitutes evidence that continuous-space BO provides no benefit over
exhaustive discrete search on this problem — a publishable finding that
shifts thesis emphasis toward the regime classifier and meta-learner
architecture rather than hyperparameter search methodology.

---

## 8. File Structure

```
experiments/
├── plan13b_v2_true_bo.py          # unchanged, retained as historical
├── plan13b_v3_bo_val_split.py     # NEW: this plan
└── plan13b_v3_aggregate.py        # NEW: aggregation utility

results/
├── plan13b_v2_true_bo/            # unchanged, retained
└── plan13b_v3/                    # NEW
    ├── fold_{N}_result.csv        # per-fold test metrics
    ├── fold_{N}_best_configs.json # per-fold selected configs
    ├── fold_{N}_trial_log.csv     # per-trial val-sharpe log (diagnostic)
    ├── optuna_studies/            # saved .pkl studies
    └── plan13b_v3_summary.csv     # final aggregated summary
```

---

## 9. Open Questions

1. **Should refit-on-full use the *same* HPs or re-tune slightly?**
   Default: same HPs. Re-tuning introduces leakage.

2. **Should we evaluate on val using walk-forward inside the regime?**
   Default: No. Single 80/20 temporal split. Inner-CV is future work.

3. **Should Top-3 Blend use top-3 by val-Sharpe or top-3 by refit-train-Sharpe?**
   Default: top-3 by val-Sharpe. The val-ranking is the generalization signal.
