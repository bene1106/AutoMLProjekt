# Implementation Plan 4 — Tier 2 Algorithms, Per-Regime Rankings & Final Analysis

## Context

Plans 1–3 are complete. The professor's tasks for this week include:
- ✅ Reflex Agent baseline (Plan 1)
- ✅ Regime classifier + confidence analysis (Plan 2)
- ✅ Walk-Forward Validation (Plan 3)
- ❌ **"try with all tier 1 AND tier 2 → which one is the winner"** ← THIS PLAN
- ❌ **Per-regime algorithm ranking** ← THIS PLAN
- ❌ **"Present actual setup, interpret the results, analyse, propose modifications"** ← THIS PLAN

This plan adds Tier 2 algorithms (Ridge, Lasso, Elastic Net), reruns the full pipeline, creates detailed per-regime rankings, and produces a clean summary for the professor.

---

## Step 1: Implement Tier 2 Algorithms (Stage 0 Pre-Training)

### What are Tier 2 Algorithms?

Tier 1 algorithms (Equal Weight, MinVar, Momentum, etc.) are heuristics — they use simple rules and rolling statistics, no training required.

Tier 2 algorithms are **linear ML models** that learn from historical data. They predict next-period asset returns from features, then convert predictions to portfolio weights. They must be **pre-trained and frozen** before the meta-learner/reflex agent sees them (this is "Stage 0" from the problem definition).

### Algorithm Specifications

**File:** `algorithms/tier2_linear.py` (new file)

```
All Tier 2 algorithms follow the same pattern:
1. TRAINING (Stage 0): Fit a regularized linear regression on training data
   - Input: asset features x_{t,i} (lagged returns, volatilities, momentum)
   - Target: next-period return r_{t+1,i}
   - Each algorithm is trained ONCE on the training period, then FROZEN
   
2. INFERENCE (at each time step t):
   - Predict expected returns: mu_hat = model.predict(x_t)
   - Convert predictions to portfolio weights via softmax:
     w_i = exp(mu_hat_i) / sum(exp(mu_hat_j))
   - This ensures w ∈ Delta_N (non-negative, sums to 1)
```

**Family F8: Ridge Portfolio**
```
Class: RidgePortfolio(PortfolioAlgorithm)
- Hyperparams: regularization lambda_ridge, lookback L
- Training: sklearn Ridge(alpha=lambda_ridge)
  - Features: per-asset features from the last L days (flattened)
  - Target: next-day return per asset
- Inference: predict returns → softmax → weights
- Configurations: lambda_ridge ∈ {0.01, 0.1, 1, 10} × L ∈ {60, 120, 252}
- Total: 12 configurations
```

**Family F9: Lasso Portfolio**
```
Class: LassoPortfolio(PortfolioAlgorithm)
- Hyperparams: regularization lambda_lasso, lookback L
- Training: sklearn Lasso(alpha=lambda_lasso)
  - Same feature/target setup as Ridge
- Inference: predict returns → softmax → weights
- Configurations: lambda_lasso ∈ {0.001, 0.01, 0.1} × L ∈ {60, 120, 252}
- Total: 9 configurations
```

**Family F10: Elastic Net Portfolio**
```
Class: ElasticNetPortfolio(PortfolioAlgorithm)
- Hyperparams: regularization lambda, l1_ratio rho, lookback L
- Training: sklearn ElasticNet(alpha=lambda, l1_ratio=rho)
- Inference: predict returns → softmax → weights
- Configurations: lambda ∈ {0.01, 0.1} × rho ∈ {0.25, 0.5, 0.75} × L ∈ {60, 120}
- Total: 12 configurations
```

### Base Class Extension

**File:** `algorithms/base.py` — extend

```
Class: TrainablePortfolioAlgorithm(PortfolioAlgorithm)
- Inherits from PortfolioAlgorithm
- Adds:
  - fit(self, features_train, returns_train) -> self
    Train the model on historical data. Called once during Stage 0.
  - is_fitted: bool property
- After fit() is called, the model is FROZEN — no further updates

The compute_weights() method uses the frozen model to predict returns
and convert to weights.
```

### Stage 0 Integration

**File:** `main.py` or `algorithms/stage0.py` (new file)

```
Function: pretrain_tier2_algorithms(algorithms, features, returns, train_mask) -> list
- For each Tier 2 algorithm:
  1. Extract training features and returns for the training period
  2. Call algorithm.fit(features_train, returns_train)
  3. Verify: algorithm.is_fitted == True
  4. Freeze: algorithm parameters are now fixed
- Return the list of fitted algorithms
- Print: "Stage 0 complete: trained {n} Tier 2 algorithms"

CRITICAL: Stage 0 training must happen INSIDE each walk-forward fold,
using only that fold's training data. Never train on test data.
```

### Algorithm Registry Update

**File:** `algorithms/tier1_heuristics.py` — modify `build_tier1_algorithm_space()`

```
Rename to: build_algorithm_space(tiers=[1]) -> list[PortfolioAlgorithm]

- If tiers=[1]: return 48 Tier 1 algorithms (current behavior)
- If tiers=[1,2]: return 48 + 33 = 81 algorithms (Tier 1 + Tier 2)
- Print: "Built K={len(algorithms)} algorithms: {n1} Tier 1, {n2} Tier 2"
```

### Feature Preparation for Tier 2

```
Tier 2 algorithms need per-asset features as input for prediction.
Use the features already computed in data/features.py:
- Per-asset: ret_1d, ret_5d, ret_20d, ret_60d, vol_20d, vol_60d, 
  mom_20d, mom_60d, mom_120d (9 features per asset)
- For N=5 assets: 45 input features total

For the lookback parameter L:
- The algorithm uses a ROLLING WINDOW of the last L days of features
  to retrain or to define its training set
- Option A (simpler): Train once on all training data, ignore L
- Option B (as specified): Train on the last L days only
  
RECOMMENDATION: Use Option A for simplicity — train once on the full 
training period. The lookback L then controls how much RECENT history 
is used for feature computation, not for model retraining.

CRITICAL: All features must be lagged (use only t-1 information).
```

---

## Step 2: Walk-Forward with Tier 1 + Tier 2

**File:** `evaluation/walk_forward.py` — modify

```
Modify WalkForwardValidator.run_fold() to support Tier 2:

For each fold:
1. Split data into train/test
2. Build algorithm space: build_algorithm_space(tiers=[1,2])
3. NEW — Stage 0: pretrain_tier2_algorithms() on training data
4. Train regime classifier on training data
5. Fit reflex agent on training data (now selecting from 81 algorithms)
6. Backtest on test data
7. Record all metrics

Run walk-forward twice:
- Run A: Tier 1 only (K=48) — this already exists from Plan 3
- Run B: Tier 1 + Tier 2 (K=81) — new

Compare results side by side.
```

---

## Step 3: Per-Regime Algorithm Rankings

This is the detailed analysis the professor wants: for each regime, which algorithms perform best?

**File:** `experiments/regime_algorithm_ranking.py` (new file)

### Analysis 3.1: Per-Regime Ranking (Aggregated Over Walk-Forward)

```
For each of the 12 walk-forward folds:
  For each regime r ∈ {Calm, Normal, Tense, Crisis}:
    For each algorithm A_k (k = 1, ..., K):
      - Collect all test days in this fold where the TRUE regime s*_t == r
      - Compute A_k's average daily return on those days
      - Compute A_k's Sharpe ratio on those days (if enough days)

Then AGGREGATE across folds:
  For each regime r:
    For each algorithm A_k:
      - Average the per-regime Sharpe across all folds
      - Record: mean Sharpe, std Sharpe, number of folds with data

Output: 4 ranking tables (one per regime), showing ALL algorithms ranked

Table: Best Algorithms in Regime 1 (Calm)
| Rank | Algorithm          | Family         | Tier | Avg Sharpe | Std  | Consistent? |
|------|--------------------|----------------|------|------------|------|-------------|
| 1    | ???                | ???            | ???  | ???        | ???  | ???         |
| 2    | ???                | ???            | ???  | ???        | ???  | ???         |
| ...  | ...                | ...            | ...  | ...        | ...  | ...         |
| 81   | ???                | ???            | ???  | ???        | ???  | ???         |

(Same table for Regime 2: Normal, Regime 3: Tense, Regime 4: Crisis)

"Consistent?" = does this algorithm rank in top 10 in >50% of folds for this regime

Save to:
- results/regime_ranking_calm.csv
- results/regime_ranking_normal.csv
- results/regime_ranking_tense.csv
- results/regime_ranking_crisis.csv
```

### Analysis 3.2: Per-Regime Top-5 Summary

```
Compact summary table across all regimes:

| Rank | Calm              | Normal            | Tense             | Crisis            |
|------|-------------------|-------------------|-------------------|-------------------|
| 1    | ??? (Sharpe ???)  | ??? (Sharpe ???)  | ??? (Sharpe ???)  | ??? (Sharpe ???)  |
| 2    | ??? (Sharpe ???)  | ??? (Sharpe ???)  | ??? (Sharpe ???)  | ??? (Sharpe ???)  |
| 3    | ??? (Sharpe ???)  | ??? (Sharpe ???)  | ??? (Sharpe ???)  | ??? (Sharpe ???)  |
| 4    | ??? (Sharpe ???)  | ??? (Sharpe ???)  | ??? (Sharpe ???)  | ??? (Sharpe ???)  |
| 5    | ??? (Sharpe ???)  | ??? (Sharpe ???)  | ??? (Sharpe ???)  | ??? (Sharpe ???)  |

Key questions:
- Do Tier 2 algorithms appear in the top 5 for any regime?
- Are the top algorithms DIFFERENT across regimes? (if yes → algorithm selection matters)
- Are the top algorithms CONSISTENT across folds? (if no → fixed mapping fails)

Save to: results/regime_top5_summary.csv
Plot: results/23_regime_top5_summary.png
```

### Analysis 3.3: Tier 1 vs Tier 2 Per Regime

```
For each regime, compare the average rank of Tier 1 vs Tier 2 algorithms:

| Regime  | Avg Rank Tier 1 | Avg Rank Tier 2 | Tier 2 in Top 10? | Best Tier 2          |
|---------|-----------------|-----------------|--------------------|-----------------------|
| Calm    | ???             | ???             | ???                | ??? (Sharpe ???)     |
| Normal  | ???             | ???             | ???                | ??? (Sharpe ???)     |
| Tense   | ???             | ???             | ???                | ??? (Sharpe ???)     |
| Crisis  | ???             | ???             | ???                | ??? (Sharpe ???)     |

This directly answers the professor's question: does Tier 2 add value?

Save to: results/tier1_vs_tier2_per_regime.csv
Plot: results/24_tier1_vs_tier2_comparison.png
```

---

## Step 4: Reflex Agent Comparison (Tier 1 only vs. Tier 1+2)

**File:** `experiments/tier_comparison.py` (new file)

```
Run the full walk-forward pipeline twice:
- Pipeline A: Reflex Agent with K=48 (Tier 1 only) — reuse Plan 3 results
- Pipeline B: Reflex Agent with K=81 (Tier 1 + Tier 2)

For each fold, record:
- Which algorithms the Reflex Agent selects per regime
- Whether any Tier 2 algorithm gets selected
- Sharpe, cumulative return, turnover

Output: Head-to-Head Comparison

| Fold | Year | EW Sharpe | Reflex T1 | Reflex T1+T2 | T2 Selected? |
|------|------|-----------|-----------|--------------|--------------|
| 1    | 2013 | +0.00     | -0.01     | ???          | ???          |
| 2    | 2014 | +2.05     | +0.89     | ???          | ???          |
| ...  | ...  | ...       | ...       | ???          | ???          |
| 12   | 2024 | +1.05     | +1.48     | ???          | ???          |
| AVG  | —    | +1.02     | +0.87     | ???          | ???          |

Key question: Does adding Tier 2 to the pool improve the Reflex Agent?
- If yes: the pool was the bottleneck, as diagnosed
- If no: the rigid mapping is the bottleneck → meta-learner needed

Output: Reflex Agent Mapping Comparison

| Regime  | Tier 1 Only Pick    | Tier 1+2 Pick       | Changed? |
|---------|---------------------|---------------------|----------|
| Calm    | MinVar_L252         | ???                 | ???      |
| Normal  | MinVar_L120         | ???                 | ???      |
| Tense   | RiskParity_L20      | ???                 | ???      |
| Crisis  | RiskParity_L60      | ???                 | ???      |

Save to: results/tier_comparison.csv
Plot: results/25_tier_comparison.png
```

---

## Step 5: Final Summary for Professor

**File:** `experiments/final_summary.py` (new file)

Generate a comprehensive summary that covers everything the professor asked for.

### 5.1: System Overview

```
Print a clean system description:

"SYSTEM SETUP:
- Asset Universe: 5 ETFs (SPY, TLT, GLD, EFA, VNQ), daily data from 2005
- VIX data from 2000 for regime labels
- Regime Definition: 4 regimes based on VIX thresholds (≤15, 15-20, 20-30, >30)
- Regime Classifier: Logistic Regression on 7 lagged VIX features
- Algorithm Space: K={K} algorithms
  - Tier 1: {n1} classical heuristics (7 families × hyperparameters)
  - Tier 2: {n2} linear ML models (3 families × hyperparameters)
- Selection Agent: Reflex Agent (regime → best algorithm lookup, fitted with net-Sharpe)
- Validation: Walk-Forward, 12 folds (8yr train / 1yr test), 2013–2024
- Switching Cost: kappa = 0.001"
```

### 5.2: Key Results Table

```
| Finding | Result |
|---------|--------|
| Regime Classifier Accuracy | ~84.5% (= naive baseline, no added value) |
| Classifier fails on transition days | 99.4% stable → 69.3% transition |
| # Algorithms beating EW (Tier 1 only) | avg ??? / 48 per fold |
| # Algorithms beating EW (Tier 1+2) | avg ??? / 81 per fold |
| Reflex Agent vs EW (Tier 1) | avg gap: −0.147 Sharpe, wins 2/12 folds |
| Reflex Agent vs EW (Tier 1+2) | avg gap: ??? Sharpe, wins ???/12 folds |
| Oracle Gap | −0.099 (regime prediction not the bottleneck) |
| Algorithm stability per regime | 33–42% (mapping changes across folds) |
| Best overall algorithm | ??? |
| Best per regime (Calm/Normal/Tense/Crisis) | ???/???/???/??? |
```

### 5.3: Interpretation & Analysis

```
Print analysis covering:

1. WHY the regime classifier doesn't help
   - VIX autocorrelation explains 84.5% accuracy
   - No model or feature set improves on "use yesterday's regime"
   - The real difficulty is transition days (30pp accuracy drop)

2. WHY the reflex agent underperforms EW
   - Algorithm pool: few algorithms consistently beat EW
   - Rigid mapping: "best algo per regime" changes across training windows
   - Switching costs: active strategies generate turnover that EW avoids

3. WHETHER Tier 2 helps
   - [Based on results: do Tier 2 algos appear in top rankings per regime?]
   - [Does the Reflex Agent pick Tier 2 algos for any regime?]
   - [Does overall performance improve?]

4. WHAT the per-regime rankings reveal
   - Are different algorithms optimal in different regimes? (→ selection matters)
   - Or does one algorithm dominate all regimes? (→ just use that one)
```

### 5.4: Proposed Modifications for Next Week

```
Print proposals:

"PROPOSED MODIFICATIONS FOR NEXT WEEK:
1. Replace Reflex Agent with a learned Meta-Learner (neural network)
   - Uses asset features + regime probabilities as input
   - Outputs soft mixture weights over all algorithms (not hard selection)
   - Can adapt within a regime based on current market conditions
   
2. Add Tier 3 algorithms (Random Forest, Gradient Boosting, Neural Network)
   - Expands pool to K ≈ 100–150
   - Non-linear models may capture patterns that heuristics and linear models miss

3. Investigate ensemble approaches
   - Instead of selecting ONE algorithm, blend top-N per regime
   - May be more robust than hard selection

4. Refine regime definition
   - Current 4 VIX-based regimes may be too coarse
   - Test: 2 regimes, 6 regimes, data-driven clustering"
```

---

## Execution Instructions

```bash
cd regime_algo_selection

# Run everything in sequence
python -m experiments.plan4_full

# Or step by step:
python -m algorithms.stage0_test          # Verify Tier 2 training works
python -m experiments.tier_comparison      # Walk-forward Tier1 vs Tier1+2
python -m experiments.regime_algorithm_ranking  # Per-regime rankings
python -m experiments.final_summary       # Generate professor summary
```

Create a single entry point `experiments/plan4_full.py` that runs all steps in order.

Expected runtime: ~10-15 minutes (81 algorithms × 12 folds, with Stage 0 training per fold).

---

## Expected Outputs

```
results/
├── (Plans 1–3: plots 01–22, already exist)
├── 23_regime_top5_summary.png
├── 24_tier1_vs_tier2_comparison.png
├── 25_tier_comparison.png
├── regime_ranking_calm.csv
├── regime_ranking_normal.csv
├── regime_ranking_tense.csv
├── regime_ranking_crisis.csv
├── regime_top5_summary.csv
├── tier1_vs_tier2_per_regime.csv
├── tier_comparison.csv
└── final_summary.txt
```

---

## What the Professor Gets

A complete package answering all his questions:

1. **"Regime certainty"** → Classifier analysis: 84.5% accuracy, equals naive baseline, fails on transitions
2. **"Baseline model / reflex agent"** → Reflex Agent: Sharpe +0.87 avg, loses to EW in 10/12 folds
3. **"Try with all tier 1 and tier 2"** → Head-to-head: Tier 1 (K=48) vs Tier 1+2 (K=81)
4. **"Which one is the winner"** → Per-regime rankings showing the best algorithm per market condition
5. **"Unbiased result"** → Walk-forward validation over 12 folds, not a single cherry-picked window
6. **"Present, interpret, analyse"** → Full summary with system description, results, and interpretation
7. **"Propose modifications"** → Meta-learner, Tier 3, ensemble approaches, refined regimes
