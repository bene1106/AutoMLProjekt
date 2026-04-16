# Implementation Plan 13b-v2: True Bayesian Optimization with Continuous Hyperparameter Search

## Overview

**Goal:** Replace the exhaustive search over 117 fixed algorithm configurations (Plan 13b) with **true Bayesian Optimization** over the continuous hyperparameter space of each algorithm family. Instead of choosing from a pre-defined grid, BO discovers the optimal hyperparameter configuration per regime directly.

**Motivation:** Plan 13b used a discrete grid of 117 pre-defined configurations. But these were arbitrary samples from a much larger continuous space — Momentum with Lookback=45 might outperform both Lookback=20 and Lookback=60, but we never tested it. True BO explores this continuous space efficiently using Optuna's TPE (Tree-structured Parzen Estimator).

**This replaces Plan 13b** (Option 3 from Prof. Ler) with what the Prof actually meant: *"substitute the algo-selection with Bayesian Optimization"* over the full hyperparameter space, not just a discrete lookup.

---

## What Changes from Plan 13b

| Aspect | Plan 13b (old) | Plan 13b-v2 (this) |
|--------|---------------|-------------------|
| Search space | 117 fixed configs (categorical) | Continuous hyperparameters per family |
| Method | Exhaustive (`for k in range(117)`) | Optuna TPE (~200 trials per regime) |
| Algo creation | Pre-built, frozen | Built on-the-fly per trial with sampled HPs |
| Speed | 6 seconds | ~30-60 min (algorithms must be fitted + evaluated per trial) |
| Can find | Only the best of 117 pre-defined | Any point in the continuous HP space |

---

## The Continuous Search Space

### Tier 1: Heuristics

```python
def sample_tier1(trial):
    family = trial.suggest_categorical("family", [
        "EqualWeight", "MinVariance", "RiskParity", "MaxDiversification",
        "MeanVariance", "Momentum", "TrendFollowing"
    ])
    
    if family == "EqualWeight":
        return {"family": "EqualWeight"}
        # No hyperparameters
    
    elif family == "MinVariance":
        lookback = trial.suggest_int("minvar_lookback", 10, 252)
        return {"family": "MinVariance", "lookback": lookback}
    
    elif family == "RiskParity":
        lookback = trial.suggest_int("riskparity_lookback", 10, 252)
        return {"family": "RiskParity", "lookback": lookback}
    
    elif family == "MaxDiversification":
        lookback = trial.suggest_int("maxdiv_lookback", 10, 252)
        return {"family": "MaxDiversification", "lookback": lookback}
    
    elif family == "MeanVariance":
        lookback = trial.suggest_int("mv_lookback", 10, 252)
        gamma = trial.suggest_float("mv_gamma", 0.1, 10.0, log=True)
        return {"family": "MeanVariance", "lookback": lookback, "gamma": gamma}
    
    elif family == "Momentum":
        lookback = trial.suggest_int("mom_lookback", 5, 252)
        mom_type = trial.suggest_categorical("mom_type", ["linear", "exp"])
        return {"family": "Momentum", "lookback": lookback, "type": mom_type}
    
    elif family == "TrendFollowing":
        lookback = trial.suggest_int("trend_lookback", 5, 252)
        trend_type = trial.suggest_categorical("trend_type", ["linear", "exp"])
        return {"family": "TrendFollowing", "lookback": lookback, "type": trend_type}
```

### Tier 2: Linear ML

```python
def sample_tier2(trial):
    family = trial.suggest_categorical("family", ["Ridge", "Lasso", "ElasticNet"])
    lookback = trial.suggest_int("lookback", 20, 252)
    
    if family == "Ridge":
        alpha = trial.suggest_float("ridge_alpha", 1e-4, 100.0, log=True)
        return {"family": "Ridge", "alpha": alpha, "lookback": lookback}
    
    elif family == "Lasso":
        alpha = trial.suggest_float("lasso_alpha", 1e-4, 10.0, log=True)
        return {"family": "Lasso", "alpha": alpha, "lookback": lookback}
    
    elif family == "ElasticNet":
        alpha = trial.suggest_float("enet_alpha", 1e-4, 10.0, log=True)
        l1_ratio = trial.suggest_float("enet_l1_ratio", 0.1, 0.9)
        return {"family": "ElasticNet", "alpha": alpha, 
                "l1_ratio": l1_ratio, "lookback": lookback}
```

### Tier 3: Non-Linear ML

```python
def sample_tier3(trial):
    family = trial.suggest_categorical("family", [
        "RandomForest", "GradientBoosting", "MLP"
    ])
    lookback = trial.suggest_int("lookback", 20, 252)
    
    if family == "RandomForest":
        n_estimators = trial.suggest_int("rf_n_estimators", 50, 500)
        max_depth = trial.suggest_int("rf_max_depth", 3, 30)
        min_samples_leaf = trial.suggest_int("rf_min_leaf", 1, 20)
        return {"family": "RandomForest", "n_estimators": n_estimators,
                "max_depth": max_depth, "min_samples_leaf": min_samples_leaf,
                "lookback": lookback}
    
    elif family == "GradientBoosting":
        n_estimators = trial.suggest_int("gbm_n_estimators", 50, 500)
        max_depth = trial.suggest_int("gbm_max_depth", 2, 15)
        learning_rate = trial.suggest_float("gbm_lr", 0.001, 0.5, log=True)
        subsample = trial.suggest_float("gbm_subsample", 0.5, 1.0)
        return {"family": "GradientBoosting", "n_estimators": n_estimators,
                "max_depth": max_depth, "learning_rate": learning_rate,
                "subsample": subsample, "lookback": lookback}
    
    elif family == "MLP":
        hidden1 = trial.suggest_int("mlp_hidden1", 16, 256)
        hidden2 = trial.suggest_int("mlp_hidden2", 0, 128)
        alpha = trial.suggest_float("mlp_alpha", 1e-5, 0.1, log=True)
        hidden = (hidden1,) if hidden2 == 0 else (hidden1, hidden2)
        return {"family": "MLP", "hidden_layer_sizes": hidden,
                "alpha": alpha, "lookback": lookback}
```

### Combined Search Space (All Tiers)

```python
def sample_algorithm(trial):
    """
    Single unified search space across all tiers and families.
    Optuna's TPE handles the conditional (tree-structured) space natively.
    """
    tier = trial.suggest_categorical("tier", [1, 2, 3])
    
    if tier == 1:
        return sample_tier1(trial)
    elif tier == 2:
        return sample_tier2(trial)
    elif tier == 3:
        return sample_tier3(trial)
```

**Total search space dimensionality:** ~5-8 continuous/integer parameters (depending on family) + 2 categorical (tier, family). This is exactly the kind of mixed, conditional space that TPE excels at.

---

## The Objective Function

```python
def objective(trial, prices_train, returns_train, regime_days, kappa):
    """
    Black-box objective for one regime.
    
    1. Sample hyperparameters from the continuous space
    2. Create and fit the algorithm with those hyperparameters
    3. Compute portfolio weights for all regime days
    4. Compute Sharpe ratio (net of switching costs)
    
    This is the "expensive" evaluation that BO is designed for —
    each trial requires fitting a model and computing weights.
    """
    # 1. Sample algorithm configuration
    config = sample_algorithm(trial)
    
    # 2. Create algorithm instance with sampled hyperparameters
    algo = create_algorithm_from_config(config)
    
    # 3. Fit if trainable (Tier 2+3)
    if hasattr(algo, 'fit'):
        algo.fit(prices_train)
    
    # 4. Compute weights for regime days
    weights = []
    for t in regime_days:
        w = algo.compute_weights(prices_train.iloc[:t+1])
        weights.append(w)
    weights = np.array(weights)
    
    # 5. Compute portfolio returns
    rets = returns_train[regime_days]
    port_returns = np.sum(weights * rets, axis=1)
    
    # 6. Switching costs
    if len(weights) > 1:
        turnover = np.sum(np.abs(np.diff(weights, axis=0)), axis=1)
        costs = kappa * turnover
        net_returns = port_returns[1:] - costs
    else:
        net_returns = port_returns
    
    # 7. Sharpe ratio
    if len(net_returns) < 10 or np.std(net_returns) < 1e-10:
        return 0.0
    sharpe = np.mean(net_returns) / np.std(net_returns) * np.sqrt(252)
    
    return sharpe
```

---

## Algorithm Factory

```python
def create_algorithm_from_config(config):
    """
    Create a portfolio algorithm instance from sampled hyperparameters.
    
    This is the bridge between Optuna's search space and the existing
    algorithm classes from tier1_heuristics.py, tier2_linear.py, tier3_nonlinear.py.
    """
    family = config["family"]
    
    # Tier 1
    if family == "EqualWeight":
        return EqualWeightAlgorithm()
    elif family == "MinVariance":
        return MinVarianceAlgorithm(lookback=config["lookback"])
    elif family == "Momentum":
        return MomentumAlgorithm(lookback=config["lookback"], 
                                  mom_type=config["type"])
    # ... etc for all families
    
    # Tier 2
    elif family == "Ridge":
        return Tier2Algorithm(
            model_class=Ridge, 
            model_params={"alpha": config["alpha"]},
            lookback=config["lookback"]
        )
    # ... etc
    
    # Tier 3
    elif family == "RandomForest":
        return Tier3Algorithm(
            family="RandomForest",
            model_class=RandomForestRegressor,
            model_params={
                "n_estimators": config["n_estimators"],
                "max_depth": config["max_depth"],
                "min_samples_leaf": config["min_samples_leaf"],
                "random_state": 42, "n_jobs": -1
            },
            lookback=config["lookback"]
        )
    # ... etc
```

---

## Experiment Flow (per fold)

```
For each fold f ∈ {1, ..., 12}:

1. SPLIT into train/test

2. FOR EACH REGIME s ∈ {Calm, Normal, Tense, Crisis}:
   a. Identify training days where regime = s
   b. Create Optuna study (direction="maximize", sampler=TPESampler)
   c. Run N_TRIALS = 200 trials
      - Each trial: sample HP config → create algo → fit → evaluate → Sharpe
   d. Record: best_config[s] = study.best_trial.params
   e. Record: best_sharpe[s] = study.best_value

3. TEST-TIME EVALUATION:
   Strategy A (Hard BO): For each test day, use best_config[regime] algo
   Strategy B (Top-3 BO): Blend top-3 configs per regime
   
4. BASELINES:
   - Equal Weight
   - Plan 13b (exhaustive over 117 fixed configs)
   - Plan 13a (Hierarchical ML)
   - Oracle Per-Regime

5. RECORD all metrics
```

---

## Configuration

```python
CONFIG = {
    # BO settings
    "n_trials": 200,                # trials per regime (4 regimes × 200 = 800 per fold)
    "sampler": "TPESampler",
    "seed": 42,
    "direction": "maximize",
    
    # Pruning (optional — stop bad trials early)
    "use_pruning": False,           # start without, add if too slow
    
    # Costs
    "kappa": 0.001,
    
    # Data
    "use_oracle_regime": True,
    "tiers": [1, 2, 3],
    
    # Walk-forward
    "n_folds": 12,
    "test_years": list(range(2013, 2025)),
    
    # Warm start
    "warm_start_from_13b": True,    # seed first trials with Plan 13b's best configs
}
```

---

## Warm Start from Plan 13b

Prof. Ler said: *"use the Bayesian optimizer, but not start from scratch — with warm start from Option 1."*

We implement this by seeding the Optuna study with the best configurations found in Plan 13b:

```python
def add_warm_start_trials(study, best_configs_from_13b, regime_id):
    """
    Enqueue the best configs from Plan 13b as the first trials.
    Optuna evaluates these first, then explores from there.
    """
    for config in best_configs_from_13b[regime_id]:
        study.enqueue_trial(config)
```

This means BO starts from a known-good point and explores outward — exactly what the Prof suggested.

---

## Speedup Strategies

Each trial requires fitting an algorithm and computing weights — this is slower than Plan 13b's cached lookup. Estimated cost per trial:

| Tier | Fit Time | Eval Time (per day) | Total (200 regime days) |
|------|----------|-------------------|------------------------|
| Tier 1 Heuristic | 0s | ~1ms | ~0.2s |
| Tier 1 MinVar/MaxDiv | 0s | ~50ms (scipy) | ~10s |
| Tier 2 Linear ML | ~0.5s | ~0.1ms | ~0.5s |
| Tier 3 RF/GBM | ~2-5s | ~0.1ms | ~3-5s |
| Tier 3 MLP | ~3-10s | ~0.1ms | ~4-10s |

Average per trial: ~3-5 seconds. With 200 trials × 4 regimes × 12 folds = 9,600 trials total → **~8-13 hours**.

### Optimization: Vectorized Evaluation

For Tier 2+3, reuse the batch prediction pattern from Plan 13a:
1. Fit the model once on training data
2. Predict all regime days at once (`model.predict(X_all)`)
3. Compute Sharpe vectorized

This reduces Tier 2+3 trial time from ~5s to ~1s. Estimated total: **~3-5 hours**.

### Optimization: Reduce Trials for Later Folds

Folds 1-3 use full 200 trials. Folds 4-12 can warm-start from previous folds' best configs and use only 100 trials (the landscape is similar with expanding windows). Estimated total: **~2-3 hours**.

---

## Outputs

```
results/plan13b_v2_true_bo/
├── summary_metrics.csv                  # 12 folds × metrics
├── best_config_per_regime.json          # optimal HP config per regime per fold
├── optuna_studies/                      # saved Optuna study objects (for analysis)
│   ├── fold_01_regime_1.pkl
│   ├── fold_01_regime_2.pkl
│   └── ...
├── search_analysis/
│   ├── hp_importance.csv                # which HPs matter most (Optuna built-in)
│   ├── best_family_per_regime.csv       # which family wins per regime
│   └── continuous_vs_grid_comparison.csv # do BO configs beat grid configs?
└── comparison_all.csv                   # 13b-v2 vs 13b vs 13a vs 13c vs EW
```

---

## Key Diagnostic Analyses

### Analysis 1: Does BO Find Better Configs Than the Grid?

Compare Plan 13b's grid-best vs Plan 13b-v2's BO-best per regime:
- If BO Sharpe > Grid Sharpe → the discrete grid was missing good regions
- If BO Sharpe ≈ Grid Sharpe → the grid was already sufficient

### Analysis 2: Hyperparameter Importance

Optuna provides built-in HP importance analysis (fANOVA). Which parameters matter most?
- If lookback dominates → the timing of the signal matters more than the model
- If family dominates → the algorithm choice is more important than tuning

### Analysis 3: Optimal Lookback Distribution

For Momentum/TrendFollowing: what lookback does BO converge to per regime?
- Calm → longer lookback? (stable trends)
- Crisis → shorter lookback? (fast reactions needed)

### Analysis 4: Does True BO Close the Gap to EW?

```
Plan 13b (grid):    +0.75
Plan 13b-v2 (BO):   ???
EW:                 +1.02
Oracle:             +1.65
```

If 13b-v2 > +1.02 → BO finds configurations the grid missed, and regime-selection works
If 13b-v2 ≈ +0.75 → the grid was already good enough; the problem is not the search

---

## Expected Runtime

```
Per fold (200 trials × 4 regimes):
  Tier 1 trials (~60%): ~1-2s each
  Tier 2 trials (~15%): ~1s each  
  Tier 3 trials (~25%): ~2-5s each
  Average: ~2s per trial
  Total: 800 trials × 2s = ~27 min per fold

Full 12-fold run: ~5-6 hours
With warm start + reduced trials: ~3-4 hours
```

---

## Success Criteria

| Criterion | Target | Meaning |
|-----------|--------|---------|
| BO Sharpe > Grid Sharpe (13b) | +0.75 → higher | BO finds better configs than the fixed grid |
| BO beats EW | ≥ 4/12 folds | True BO overcomes the instability problem |
| Optimal lookback varies by regime | Different per regime | Regime-conditional HP optimization has value |
| HP importance shows family < lookback | fANOVA result | Model choice matters less than timing |

---

## Dependencies

```
pip install optuna   # required — core BO framework
```

---

## Claude Code Prompt

```
Read `implementation_plan_13b_v2_true_bo.md` and implement everything in 
experiments/plan13b_v2_true_bo.py.

Key implementation details:
1. Use Optuna TPESampler for Bayesian Optimization
2. Define the continuous search space for all 13 algorithm families 
   (7 Tier 1 + 3 Tier 2 + 3 Tier 3) with continuous/integer hyperparameters
3. Create an algorithm factory function that builds algorithm instances 
   from sampled hyperparameters, reusing existing classes from 
   algorithms/tier1_heuristics.py, tier2_linear.py, tier3_nonlinear.py
4. Run 200 trials per regime (4 regimes) per fold (12 folds)
5. For Tier 2+3: fit the model once, then batch-predict all regime days
   (use the vectorized pattern from batch_precompute_algo_outputs)
6. Warm start: seed each study with Plan 13b's best config per regime 
   (load from results/plan13b_bayesian_opt/best_algo_per_regime.csv)
7. Two test strategies: Hard BO (best config per regime) and Top-3 BO 
   (blend top 3 configs per regime)
8. Print comparison table: 13b-v2 vs 13b vs 13a vs 13c vs EW
9. Save Optuna studies for HP importance analysis

Use oracle regime labels. Save results to results/plan13b_v2_true_bo/.
Run the full 12-fold experiment, then commit and push.
```
