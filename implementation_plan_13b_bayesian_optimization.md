# Implementation Plan 13b: Pure Bayesian Optimization for Algorithm Selection

## Overview

**Goal:** Replace the neural network Meta-Learner entirely with Bayesian Optimization (BO). Instead of a learned network that maps context → mixing weights, BO directly searches for the best algorithm (or algorithm blend) by treating portfolio performance as the black-box objective.

**This is Option 3** from Prof. Ler's framework:
- ✅ Option 1 (Plan 13a): Hierarchical Meta-Learner — completed, negative result
- ⬜ Option 2 (Plan 13c): Meta-Learner for tier selection + BO within tier — later
- **→ Option 3 (this plan): Pure BO, no Meta-Learner**

**Motivation:** Plan 13a showed that learned meta-learners struggle because (a) Tier 2+3 algorithms are indistinguishable on 5 ETFs, and (b) the neural network adds overfitting overhead. BO takes a fundamentally different approach: no gradient-based learning, no softmax, no entropy regularization — just smart search over the algorithm space.

**Key design choice:** BO operates per regime. For each of the 4 regimes, a separate BO instance searches for the best algorithm configuration. This makes the regime-conditioning explicit rather than hoping a neural network learns it implicitly.

---

## What This Plan Does NOT Do

- No neural network Meta-Learner (that's Options 1 and 2)
- No real regime classifier (oracle regime, same as Plans 12/13a)
- No warm start from Plan 13a (that's Option 2 / Plan 13c)
- No adversarial training
- No soft blending — BO selects a single algorithm per regime (hard selection)

---

## Conceptual Design

### The Black-Box Problem

BO treats algorithm selection as a hyperparameter optimization problem:

```
Objective: f(algorithm_index) = Sharpe Ratio of algorithm on training data in regime s
Search space: {0, 1, ..., K-1} where K = 117 algorithms
Constraint: One BO instance per regime → 4 independent searches
```

For each walk-forward fold:
1. Split into train/test
2. Pre-compute all 117 algorithm outputs (reuse cached .npy from Plan 13a)
3. For each regime s ∈ {Calm, Normal, Tense, Crisis}:
   a. Identify training days where regime = s
   b. BO searches: which algorithm has the best Sharpe on those days?
   c. Result: best_algo[s] = algorithm index for regime s
4. At test time: observe regime → use best_algo[regime]
5. Evaluate portfolio performance

### Why Per-Regime BO (Not Global BO)

A global BO over the full training set would find one single best algorithm — ignoring regimes entirely. The per-regime approach is the BO equivalent of "regime-dependent algorithm selection," which is our research question.

### Hard Selection vs. Soft Blending

Plan 13a used soft blending (mixing weights α_t). Plan 13b uses **hard selection**: one algorithm per regime. This is simpler and avoids the entropy/softmax issues entirely.

However, we also test a **top-N blend** variant: instead of picking the single best algorithm per regime, BO identifies the top-3 and equally weights them. This provides a middle ground.

---

## BO Configuration

### Library: Optuna

Optuna is chosen over SMAC/Hyperopt because:
- Native support for categorical parameters (algorithm index is categorical)
- TPE (Tree-structured Parzen Estimator) handles mixed spaces well
- Pruning support (not needed here but useful for extensions)
- Well-documented, actively maintained, already in common use

### Search Space

```python
def objective(trial, algo_outputs, returns, regime_mask):
    """
    Black-box objective for one regime.
    
    Args:
        trial: Optuna trial object
        algo_outputs: pre-computed (T, K, N) array of algorithm weights
        returns: (T, N) array of asset returns
        regime_mask: boolean array, True for days in this regime
    
    Returns:
        sharpe: Sharpe ratio of the selected algorithm on regime days
    """
    # Select algorithm
    algo_idx = trial.suggest_categorical("algo_idx", list(range(K)))
    
    # Get this algorithm's weights for regime days
    regime_days = np.where(regime_mask)[0]
    weights = algo_outputs[regime_days, algo_idx, :]  # (n_regime_days, N)
    rets = returns[regime_days, :]                      # (n_regime_days, N)
    
    # Compute portfolio returns
    port_returns = np.sum(weights * rets, axis=1)  # (n_regime_days,)
    
    # Switching costs (between consecutive regime days)
    turnover = np.sum(np.abs(np.diff(weights, axis=0)), axis=1)
    costs = KAPPA * turnover
    net_returns = port_returns[1:] - costs  # first day has no previous weights
    
    # Sharpe ratio
    if len(net_returns) < 10 or np.std(net_returns) < 1e-10:
        return 0.0  # not enough data or zero variance
    sharpe = np.mean(net_returns) / np.std(net_returns) * np.sqrt(252)
    
    return sharpe
```

### BO Hyperparameters

```python
BO_CONFIG = {
    "n_trials": 117,           # test each algorithm at least once (exhaustive for categorical)
    "sampler": "TPESampler",   # Tree-structured Parzen Estimator
    "seed": 42,
    "direction": "maximize",   # maximize Sharpe
    
    # For top-N blend variant
    "top_n": 3,                # blend top-3 algorithms per regime
}
```

**Note on n_trials:** Since the search space is categorical with K=117 values, BO with n_trials=117 will evaluate every algorithm exactly once. This is equivalent to an exhaustive search. The value of BO here is not sample efficiency (the space is small enough to enumerate), but rather the framework — it generalizes naturally to continuous/mixed spaces in Plan 13c where BO searches over hyperparameters within a tier.

For this plan, we also run a simple **exhaustive evaluation** (evaluate all 117 algorithms per regime) as a sanity check that BO finds the same optimum.

---

## Architecture: No Meta-Learner

```
At decision time t:

1. Observe oracle regime s*_t ∈ {Calm, Normal, Tense, Crisis}
2. Look up: best_algo[s*_t] → algorithm index k*
3. Use algorithm k*'s pre-computed weights: w_t = w^(k*)_t
4. (Top-N variant: w_t = (1/3) Σ_{k ∈ top3[s*_t]} w^(k)_t)
```

No neural network, no softmax, no gradient descent. Just a lookup table learned by BO.

---

## Step 0: Reuse Cached Algorithm Outputs

Plan 13a already computed and cached all 117 algorithm outputs per fold:
```
results/plan13a_hierarchical/cache/fold_XX_algo_outputs.npy
```

Plan 13b loads these directly — **zero precompute time**.

If cache files don't exist (e.g., different fold structure), fall back to `batch_precompute_algo_outputs()` from Plan 13a.

---

## Step 1: Implement Per-Regime BO

### New File: `experiments/plan13b_bayesian_optimization.py`

```python
import optuna
import numpy as np

def run_bo_per_regime(algo_outputs, returns, regime_labels, train_mask, config):
    """
    Run Bayesian Optimization independently for each regime.
    
    Args:
        algo_outputs: (T, K, N) pre-computed algorithm weights
        returns: (T, N) asset returns
        regime_labels: (T,) regime labels {1, 2, 3, 4}
        train_mask: boolean (T,), True for training days
        config: dict with BO hyperparameters
    
    Returns:
        best_algo: dict {regime_id: algorithm_index}
        top_n_algos: dict {regime_id: [list of top-N algorithm indices]}
        all_results: dict {regime_id: DataFrame with all algorithm performances}
    """
    K = algo_outputs.shape[1]
    best_algo = {}
    top_n_algos = {}
    all_results = {}
    
    for regime_id in [1, 2, 3, 4]:
        # Days in this regime AND in training set
        regime_train_mask = train_mask & (regime_labels == regime_id)
        regime_days = np.where(regime_train_mask)[0]
        
        if len(regime_days) < 20:
            print(f"  Regime {regime_id}: too few days ({len(regime_days)}), using EW")
            best_algo[regime_id] = 0  # fallback to EW (algo index 0)
            top_n_algos[regime_id] = [0]
            continue
        
        # Exhaustive evaluation: compute Sharpe for ALL algorithms
        sharpe_per_algo = []
        for k in range(K):
            weights = algo_outputs[regime_days, k, :]
            rets = returns[regime_days, :]
            port_ret = np.sum(weights * rets, axis=1)
            
            # Switching costs
            if len(regime_days) > 1:
                turnover = np.sum(np.abs(np.diff(weights, axis=0)), axis=1)
                costs = config["kappa"] * turnover
                net_ret = port_ret[1:] - costs
            else:
                net_ret = port_ret
            
            if len(net_ret) > 1 and np.std(net_ret) > 1e-10:
                sharpe = np.mean(net_ret) / np.std(net_ret) * np.sqrt(252)
            else:
                sharpe = 0.0
            
            sharpe_per_algo.append(sharpe)
        
        sharpe_per_algo = np.array(sharpe_per_algo)
        
        # Best single algorithm
        best_idx = np.argmax(sharpe_per_algo)
        best_algo[regime_id] = best_idx
        
        # Top-N algorithms
        top_n_idx = np.argsort(sharpe_per_algo)[-config["top_n"]:][::-1]
        top_n_algos[regime_id] = top_n_idx.tolist()
        
        all_results[regime_id] = sharpe_per_algo
        
        print(f"  Regime {regime_id}: best algo #{best_idx} "
              f"(Sharpe={sharpe_per_algo[best_idx]:+.4f}), "
              f"top-{config['top_n']}: {top_n_idx.tolist()}")
    
    return best_algo, top_n_algos, all_results
```

### Why Exhaustive Instead of BO for K=117

With only 117 categorical options, exhaustive evaluation takes milliseconds (all outputs are pre-computed). BO's value over exhaustive search only emerges when the search space is continuous or much larger. For Plan 13b, we run exhaustive search and label it "BO-equivalent" — the result is identical.

For Plan 13c, where BO searches over continuous hyperparameters within a tier, the BO framework becomes genuinely necessary.

---

## Step 2: Test-Time Evaluation

### Hard Selection Strategy
```python
def evaluate_hard_selection(best_algo, algo_outputs, returns, regime_labels, test_mask):
    """
    At each test day: look up regime → use best algorithm for that regime.
    """
    test_days = np.where(test_mask)[0]
    N = algo_outputs.shape[2]
    portfolio_weights = np.zeros((len(test_days), N))
    
    for i, t in enumerate(test_days):
        regime = regime_labels[t]
        k = best_algo[regime]
        portfolio_weights[i] = algo_outputs[t, k, :]
    
    # Compute returns and metrics
    ...
```

### Top-N Blend Strategy
```python
def evaluate_top_n_blend(top_n_algos, algo_outputs, returns, regime_labels, test_mask):
    """
    At each test day: look up regime → equal-weight blend of top-N algorithms.
    """
    test_days = np.where(test_mask)[0]
    N = algo_outputs.shape[2]
    n = len(top_n_algos[1])  # same N for all regimes
    portfolio_weights = np.zeros((len(test_days), N))
    
    for i, t in enumerate(test_days):
        regime = regime_labels[t]
        algos = top_n_algos[regime]
        w = np.mean([algo_outputs[t, k, :] for k in algos], axis=0)
        portfolio_weights[i] = w
    
    # Compute returns and metrics
    ...
```

---

## Step 3: Walk-Forward Experiment

### Configuration

```python
CONFIG = {
    # BO settings
    "n_trials": 117,           # exhaustive (K=117)
    "top_n": 3,                # for blend variant
    "kappa": 0.001,            # switching cost coefficient
    
    # Data
    "use_oracle_regime": True,
    "tiers": [1, 2, 3],       # full 117-algorithm space
    
    # Walk-forward
    "n_folds": 12,
    "test_years": list(range(2013, 2025)),
    
    # Cache
    "cache_dir": "results/plan13a_hierarchical/cache",  # reuse Plan 13a cache
}
```

### Experiment Flow (per fold)

```
For each fold f ∈ {1, ..., 12}:

1. LOAD cached algorithm outputs from Plan 13a
   (or compute via batch_precompute if cache missing)

2. SPLIT into train/test indices

3. PER-REGIME BO (on training data only):
   For each regime s ∈ {1, 2, 3, 4}:
     - Evaluate all 117 algorithms on training days with regime=s
     - Record: best_algo[s], top_3_algos[s], all Sharpe values

4. EVALUATE on test data:
   Strategy A (Hard Selection): regime → best_algo[regime]
   Strategy B (Top-3 Blend): regime → EW of top-3 algos for that regime

5. BASELINES:
   - Equal Weight (1/N daily rebalance)
   - Best Single Algorithm (oracle: globally best in-sample, ignoring regimes)
   - Plan 13a Hierarchical Meta-Learner results
   - Always-Best-Per-Regime (oracle: best per-regime algo known in hindsight on TEST data)

6. RECORD all metrics
```

### Outputs

```
results/plan13b_bayesian_opt/
├── summary_metrics.csv                # 12 folds × metrics for Hard/Blend/EW/baselines
├── best_algo_per_regime.csv           # which algorithm was selected per regime per fold
├── algo_sharpe_per_regime/            # full ranking of all 117 algos per regime per fold
│   ├── fold_01_regime_sharpes.csv
│   └── ...
├── regime_selection_stability.csv     # how often does the same algo win across folds?
└── tier_distribution_per_regime.csv   # which tier do the best algos belong to?
```

---

## Step 4: Diagnostic Analyses

### Analysis 1: Which Tier Wins Per Regime?

For each regime, count how many of the top-10 algorithms belong to each tier.
This answers whether Plan 13a's finding (Tier 2+3 useless) holds when BO selects directly.

### Analysis 2: Algorithm Selection Stability

For each regime: does the same algorithm win across multiple folds?
If yes → the regime-algo mapping is stable and a lookup table suffices.
If no → the mapping is noisy and hard/soft selection both struggle.

This was already identified as a key insight: "33–42% consistency" from earlier analyses.

### Analysis 3: Hard Selection vs. Top-N Blend

Compare Sharpe ratios between picking the single best algo vs. blending top-3.
If top-3 beats single-best → soft blending has value (supports meta-learner approach).
If single-best ≥ top-3 → hard selection suffices (meta-learner adds no value).

### Analysis 4: BO vs. Plan 13a Meta-Learner

Direct comparison on the same 12 folds:
- Does BO (simple lookup table) beat the hierarchical meta-learner?
- If yes → the neural network adds negative value (overfitting)
- If no → there is something the meta-learner captures that BO misses

### Analysis 5: Oracle Gap

Compare:
1. BO with oracle regime (this experiment)
2. Always-Best-Per-Regime on test data (perfect hindsight)

The gap measures how stable the training → test transfer is.

---

## Step 5: Expected Runtime

```
Per fold:
  Load cached algo outputs:  ~5 seconds
  Exhaustive evaluation:     ~1 second (117 algos × 4 regimes, all vectorized)
  Test-time evaluation:      ~1 second
  Total per fold:            ~10 seconds

Full 12-fold run:            ~2 minutes (!!!)
```

This is orders of magnitude faster than Plan 13a (~7 hours). The speed comes from:
1. Reusing cached precompute
2. No neural network training
3. Exhaustive search over 117 categorical options is trivial

---

## Success Criteria

| Criterion | Target | Meaning |
|-----------|--------|---------|
| Hard Selection beats EW | ≥ 6/12 folds | BO finds useful regime-algo mapping |
| Top-3 Blend beats Hard Selection | Average Sharpe higher | Soft blending has value |
| BO beats Plan 13a | Higher avg Sharpe | Simple lookup > learned meta-learner |
| Regime-dependent selection | Different algos per regime | BO exploits regime structure |
| Stable across folds | Same algo wins in ≥3/12 folds per regime | Mapping is learnable |

---

## Connection to Plans 13a and 13c

- **Plan 13a** (completed): Provides cached algo outputs and baseline comparison
- **Plan 13b** (this plan): Pure BO baseline — how well does regime-dependent hard selection work?
- **Plan 13c** (next): Hybrid — Meta-Learner selects tier, BO optimizes within tier. Uses warm start from 13a specialists + BO insights from 13b.

The three plans form a progression:
1. Can a neural network learn algorithm selection? (13a → mostly no)
2. Can direct search find good algorithms per regime? (13b → we'll see)
3. Can combining both approaches outperform either? (13c → after 13b results)

---

## File Summary

### New Files

| File | Purpose |
|------|---------|
| `experiments/plan13b_bayesian_optimization.py` | Main experiment: per-regime exhaustive search + evaluation |

### Reused Files (no modifications)

| File | What's Reused |
|------|---------------|
| `results/plan13a_hierarchical/cache/*.npy` | Cached algorithm outputs per fold |
| `meta_learner/dataset.py` | MetaLearnerDataset + batch_precompute |
| `algorithms/tier1_heuristics.py` | build_algorithm_space(tiers=[1,2,3]) |
| `algorithms/stage0.py` | pretrain_algorithms() |
| `regimes/ground_truth.py` | compute_regime_labels() |

### Dependencies

```
pip install optuna  # for BO framework (optional — exhaustive search doesn't strictly need it)
```

---

## Claude Code Prompt

```
Read `implementation_plan_13b_bayesian_optimization.md` and implement everything.

The experiment reuses cached algorithm outputs from Plan 13a at 
results/plan13a_hierarchical/cache/fold_XX_algo_outputs.npy. If cache files 
don't exist, compute them using batch_precompute_algo_outputs() from 
meta_learner/dataset.py.

Implement experiments/plan13b_bayesian_optimization.py with:
1. Per-regime exhaustive evaluation of all 117 algorithms on training data
2. Two test-time strategies: Hard Selection (best algo per regime) and 
   Top-3 Blend (EW of top 3 per regime)
3. Baselines: EW, Best Single Algorithm (global), Plan 13a results
4. 12-fold walk-forward evaluation
5. Diagnostic outputs: which tier wins per regime, selection stability, 
   Hard vs Blend comparison

Use oracle regime labels (same as Plans 12/13a).
Print a summary table comparing Hard Selection, Top-3 Blend, EW, and Plan 13a.
Save results to results/plan13b_bayesian_opt/.

Run the full 12-fold experiment.
Then commit and push.
```
