# Implementation Plan 13c: Hybrid — Meta-Learner for Tier Selection + BO Within Tier

## Overview

**Goal:** Combine the Meta-Learner (Option 1) with Bayesian Optimization (Option 3) into a hybrid system. The Meta-Learner selects which tier to trust, and BO selects the best algorithm within the chosen tier. This is the standard AutoML architecture (cf. SMAC, Auto-sklearn) adapted to the sequential portfolio setting.

**This is Option 2** from Prof. Ler's framework:
- ✅ Option 1 (Plan 13a): Hierarchical Meta-Learner — Avg Sharpe -0.65, 1/12 > EW
- ✅ Option 3 (Plan 13b): Pure BO — Avg Sharpe +0.75, 3/12 > EW
- **→ Option 2 (this plan): Hybrid ML + BO**

**Motivation:** Prof. Ler's original suggestion: "Meta-Learner only on model family, then substitute the algo-selection with Bayesian Optimization." Plan 13a showed the ML tier selector works (H(β) in target range, regime sensitivity detected) but the within-tier specialists fail. Plan 13b showed that per-regime exhaustive search finds reasonable algorithms but can't beat EW. The hybrid combines the working part of 13a (tier selection) with the working part of 13b (within-tier search).

**Warm start:** Prof. Ler specifically asked: "Use the Bayesian optimizer, but not start from scratch — with warm start from Option 1." The BO within each tier is warm-started using Plan 13b's per-regime algorithm rankings.

---

## What This Plan Does NOT Do

- No neural network for within-tier selection (that's Plan 13a's failed specialists)
- No real regime classifier (oracle regime, same as Plans 12/13a/13b)
- No adversarial training
- No end-to-end joint training of ML + BO

---

## Architecture

```
Level 1: Tier Selection — Meta-Learner (from Plan 13a)
   Input: X_t (29 dims: 25 asset features + 4 regime one-hot)
   Output: β_t ∈ Δ_3 (mixing weights over 3 tiers)
   Architecture: Feedforward NN [64, 32] → Softmax(3)
   Training: Reward maximization with entropy regularization

Level 2: Within-Tier Algorithm Selection — BO (from Plan 13b)
   Per regime, per tier: BO finds the best algorithm within that tier
   Result: best_algo[regime][tier] — a lookup table
   Warm start: initialized from Plan 13b's per-regime rankings

Combination at test time:
   For each tier f: w^(tier_f)_t = w^(best_algo[regime][f])_t
   Final: w_t = Σ_f β_{t,f} · w^(tier_f)_t
```

### Key Difference from Plans 13a and 13b

| Component | Plan 13a | Plan 13b | Plan 13c (this) |
|-----------|----------|----------|-----------------|
| Tier selection | ML (TierSelector NN) | Implicit (best overall) | **ML (TierSelector NN)** |
| Within-tier selection | ML (Specialist NNs) ❌ | BO (exhaustive) | **BO (exhaustive)** |
| Test-time decision | Soft blend all 117 | Hard select 1 algo | **Soft blend 3 tier-portfolios, each is BO-selected best** |

The hybrid replaces the failed Specialist NNs with BO's proven per-regime search, while keeping the TierSelector NN that showed regime sensitivity in Plan 13a.

---

## Step 0: Reuse Cached Data

From Plan 13a:
- `results/plan13a_hierarchical/cache/fold_XX_algo_outputs.npy` — pre-computed algorithm outputs
- `tier_algorithm_indices` — which algorithms belong to which tier

From Plan 13b:
- Per-regime algorithm rankings — warm start for BO

**Zero additional precompute needed.**

---

## Step 1: Per-Regime, Per-Tier BO (Within-Tier Selection)

### Approach

For each walk-forward fold, for each regime, for each tier:
evaluate all algorithms in that tier and select the best one.

```python
def bo_within_tier(algo_outputs, returns, regime_labels, train_mask, 
                   tier_indices, kappa):
    """
    Per-regime, per-tier exhaustive search.
    
    Returns:
        best_algo_per_regime_tier: dict {regime_id: {tier_id: algo_global_index}}
        tier_sharpe_per_regime: dict {regime_id: {tier_id: best_sharpe}}
    """
    best = {}  # {regime: {tier: global_algo_index}}
    tier_sharpe = {}  # {regime: {tier: sharpe}}
    
    for regime_id in [1, 2, 3, 4]:
        regime_train = train_mask & (regime_labels == regime_id)
        regime_days = np.where(regime_train)[0]
        
        if len(regime_days) < 20:
            # Fallback: EW for all tiers
            best[regime_id] = {f: tier_indices[f][0] for f in range(3)}
            tier_sharpe[regime_id] = {f: 0.0 for f in range(3)}
            continue
        
        best[regime_id] = {}
        tier_sharpe[regime_id] = {}
        
        for tier_id in range(3):
            t_indices = tier_indices[tier_id]
            
            # Evaluate all algorithms in this tier on regime days
            best_sharpe = -np.inf
            best_idx = t_indices[0]
            
            for k in t_indices:
                weights = algo_outputs[regime_days, k, :]
                rets = returns[regime_days, :]
                port_ret = np.sum(weights * rets, axis=1)
                
                if len(regime_days) > 1:
                    turnover = np.sum(np.abs(np.diff(weights, axis=0)), axis=1)
                    costs = kappa * turnover
                    net_ret = port_ret[1:] - costs
                else:
                    net_ret = port_ret
                
                if len(net_ret) > 1 and np.std(net_ret) > 1e-10:
                    sharpe = np.mean(net_ret) / np.std(net_ret) * np.sqrt(252)
                else:
                    sharpe = 0.0
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_idx = k
            
            best[regime_id][tier_id] = best_idx
            tier_sharpe[regime_id][tier_id] = best_sharpe
    
    return best, tier_sharpe
```

This gives us, per regime, the single best algorithm from each tier. At test time, the TierSelector decides how to blend these three tier-level portfolios.

---

## Step 2: Train Tier Selector (ML Component)

### Approach

The TierSelector from Plan 13a is retrained, but with a critical difference: instead of blending 48/33/36 specialist outputs, it blends **3 single-algorithm portfolios** (the BO-selected best per tier per regime).

```python
def train_tier_selector(tier_selector, best_algo_per_regime_tier,
                        algo_outputs, returns, regime_labels,
                        train_indices, config):
    """
    Train TierSelector NN to blend 3 tier-level portfolios.
    
    Each tier-level portfolio is the single BO-selected best algorithm
    for the current regime.
    
    This is much cleaner than Plan 13a because:
    - No Specialist NNs (replaced by BO lookup)
    - Each tier contributes its single best algo, not a soft blend of many
    - TierSelector only needs to learn: given regime, how much to trust 
      each tier's best algo?
    """
    optimizer = torch.optim.Adam(
        tier_selector.parameters(), 
        lr=config["selector_lr"],
        weight_decay=config["weight_decay"]
    )
    
    tier_selector.train()
    
    for epoch in range(config["selector_epochs"]):
        w_prev = torch.ones(5) / 5.0
        epoch_reward = 0.0
        
        for t in train_indices:
            X_t = torch.tensor(dataset.get_input(t), dtype=torch.float32)
            regime = dataset.get_regime(t)
            
            # Get each tier's best algo's portfolio for this regime
            tier_portfolios = []
            for tier_id in range(3):
                k = best_algo_per_regime_tier[regime][tier_id]
                w_tier = torch.tensor(
                    algo_outputs[t, k, :], dtype=torch.float32
                )
                tier_portfolios.append(w_tier)
            
            tier_portfolios = torch.stack(tier_portfolios)  # (3, N)
            
            # TierSelector produces β_t
            beta_t = tier_selector(X_t)  # (3,)
            
            # Blended portfolio
            w_t = torch.matmul(beta_t, tier_portfolios)  # (N,)
            
            # Reward
            r_next = torch.tensor(dataset.get_returns(t), dtype=torch.float32)
            portfolio_ret = torch.dot(w_t, r_next)
            port_cost = kappa * smooth_l1(w_t - w_prev)
            reward = portfolio_ret - port_cost
            
            # Entropy regularization on β
            entropy = -torch.sum(beta_t * torch.log(beta_t + 1e-10))
            loss = -(reward * 252.0 + config["lambda_tier"] * entropy)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tier_selector.parameters(), 1.0)
            optimizer.step()
            
            w_prev = w_t.detach()
            epoch_reward += reward.item()
    
    tier_selector.eval()
    return tier_selector
```

### Key Simplification vs. Plan 13a

In Plan 13a, the selector blended 3 **specialist-produced** portfolios (each a soft mix of many algos). Here, the selector blends 3 **single-algorithm** portfolios (each the BO-best for the current regime and tier). This means:

- **No Specialist NNs needed** (Phase A from Plan 13a is entirely replaced by BO)
- **Much cleaner gradient signal** — each tier contributes one concrete portfolio, not a blurry average
- **Only Phase B** (Tier Selector training) is needed

---

## Step 3: Test-Time Evaluation

### Strategy A: Hybrid ML+BO (Primary)
```python
def evaluate_hybrid(tier_selector, best_algo_per_regime_tier,
                    algo_outputs, returns, regime_labels, test_indices):
    """
    At each test day:
    1. Look up regime → get best algo per tier (from BO)
    2. Run TierSelector NN → get β_t (tier weights)
    3. Blend: w_t = Σ_f β_{t,f} · w^(best[regime][f])_t
    """
    ...
```

### Strategy B: BO-Only Per-Tier (Ablation)
```python
def evaluate_bo_per_tier(best_algo_per_regime_tier, tier_sharpe_per_regime,
                         algo_outputs, returns, regime_labels, test_indices):
    """
    Ablation: Instead of ML selector, use training Sharpe to weight tiers.
    β_f ∝ max(0, sharpe_f) — tier weight proportional to its in-sample Sharpe.
    No neural network needed.
    """
    ...
```

### Strategy C: Best-Tier-Only (Ablation)
```python
def evaluate_best_tier_only(best_algo_per_regime_tier, tier_sharpe_per_regime,
                            algo_outputs, returns, regime_labels, test_indices):
    """
    Ablation: Always pick the tier with the highest in-sample Sharpe per regime.
    Hard tier selection + hard within-tier selection = single algorithm per regime.
    This is equivalent to Plan 13b's Hard Selection.
    """
    ...
```

---

## Step 4: Walk-Forward Experiment

### Configuration

```python
CONFIG = {
    # Tier Selector (ML)
    "input_dim": 29,
    "selector_hidden": [64, 32],
    "dropout": 0.1,
    "selector_lr": 0.005,
    "selector_epochs": 50,
    "lambda_tier": 0.05,
    "weight_decay": 1e-4,
    
    # BO (within-tier)
    "kappa": 0.001,
    "top_n_per_tier": 1,       # single best per tier (BO result)
    
    # Data
    "use_oracle_regime": True,
    "tiers": [1, 2, 3],
    
    # Walk-forward
    "n_folds": 12,
    "test_years": list(range(2013, 2025)),
    
    # Cache
    "cache_dir": "results/plan13a_hierarchical/cache",
}
```

### Experiment Flow (per fold)

```
For each fold f ∈ {1, ..., 12}:

1. LOAD cached algorithm outputs (from Plan 13a)

2. SPLIT into train/test indices

3. BO WITHIN-TIER (on training data):
   For each regime × tier combination:
     Evaluate all algos in that tier on regime's training days
     Record: best_algo[regime][tier] and sharpe[regime][tier]

4. TRAIN TIER SELECTOR (on training data):
   TierSelector NN blends 3 BO-selected tier portfolios
   Uses the same architecture and λ_tier as Plan 13a Phase B
   
5. EVALUATE on test data:
   Strategy A: Hybrid (ML selector + BO within-tier)
   Strategy B: BO-only per-tier (Sharpe-weighted, no NN)
   Strategy C: Best-tier-only (hard selection, = Plan 13b equivalent)
   
6. BASELINES:
   - Equal Weight
   - Plan 13a (Hierarchical ML)
   - Plan 13b (BO Hard Selection)
   - Plan 13b (BO Top-3 Blend)
   - Oracle Per-Regime

7. RECORD all metrics
```

### Outputs

```
results/plan13c_hybrid/
├── summary_metrics.csv                # 12 folds × all strategies
├── tier_selection_by_regime.csv       # β_t per regime (same format as 13a)
├── bo_selected_algos.csv              # best algo per regime per tier per fold
├── tier_sharpe_by_regime.csv          # in-sample Sharpe per tier per regime
└── comparison_all_options.csv         # head-to-head: 13a vs 13b vs 13c vs EW
```

---

## Step 5: Final Comparison (All Three Options)

The central output of Plan 13c is the **head-to-head comparison** of all approaches:

```
======================================================================
FINAL COMPARISON: All Three Options (12-fold Walk-Forward, 2013-2024)
======================================================================

Strategy                          Avg Sharpe   Beats EW   Folds Won
----------------------------------------------------------------------
Oracle Per-Regime                  +1.65        —          (upper bound)
Equal Weight (EW)                  +1.02        —          baseline
Best Single Algorithm              +1.01        —          
Plan 13b: BO Hard Selection        +0.75        3/12       
Plan 13b: BO Top-3 Blend           +0.79        4/12       
Plan 13c: Hybrid ML+BO            ???          ???        
Plan 13c: BO-Only Per-Tier         ???          ???        (ablation)
Plan 13a: Hierarchical ML          -0.65        1/12       
----------------------------------------------------------------------
```

This table is the key deliverable for the thesis — it answers Prof. Ler's question:
"Try all 3 options and compare."

---

## Expected Runtime

```
Per fold:
  Load cached algo outputs:    ~5 seconds
  BO within-tier (exhaustive): ~1 second
  Train Tier Selector:         ~2 minutes (50 epochs, 3 tier portfolios)
  Test evaluation:             ~1 second
  Total per fold:              ~2.5 minutes

Full 12-fold run:              ~30 minutes
```

Much faster than Plan 13a (~7h), slightly slower than Plan 13b (~6s), because the Tier Selector NN still needs training.

---

## Success Criteria

| Criterion | Target | Meaning |
|-----------|--------|---------|
| Hybrid beats Plan 13a | Higher avg Sharpe | BO within-tier > ML within-tier |
| Hybrid ≈ Plan 13b | Similar avg Sharpe | ML tier selector adds no value over BO alone |
| Hybrid beats EW | ≥ 4/12 folds | Combined approach captures regime structure |
| H(β) in [0.1, 1.0] | ≥ 8/12 folds | Tier selector doesn't collapse |
| Regime-conditional β differs | Across ≥ 2 regimes | ML captures regime signal |

### Expected Outcome

Based on Plans 13a and 13b results, the most likely outcome is:

**Hybrid ≈ Plan 13b > Plan 13a, but still < EW.**

Reasoning: The ML tier selector showed slight regime sensitivity in Plan 13a (Tense → more Tier 1), but the effect was too weak to overcome the noise. Adding it to BO's already-decent per-regime selection is unlikely to improve significantly, because:
1. BO already selects per-regime (the regime conditioning is explicit)
2. The tier selector's regime sensitivity was only on the tier level, not within-tier
3. With BO handling within-tier selection perfectly, the only role of ML is "which tier overall" — and Plan 13b showed that Tier 1 dominates anyway

This makes Plan 13c primarily a **completeness experiment** — Prof. Ler asked for all three, and the "all three fail to beat EW" conclusion is cleaner with all three tested.

---

## File Summary

### New Files

| File | Purpose |
|------|---------|
| `experiments/plan13c_hybrid.py` | Main experiment: BO within-tier + ML tier selector |
| `implementation_plan_13c_hybrid.md` | This implementation plan |

### Reused Files (no modifications)

| File | What's Reused |
|------|---------------|
| `results/plan13a_hierarchical/cache/*.npy` | Cached algorithm outputs |
| `meta_learner/hierarchical_network.py` | TierSelector class |
| `meta_learner/dataset.py` | MetaLearnerDataset + batch_precompute |
| `algorithms/*` | All 117 algorithms |
| `regimes/ground_truth.py` | Oracle regime labels |

---

## Claude Code Prompt

```
Read `implementation_plan_13c_hybrid.md` and implement everything in a single 
file: experiments/plan13c_hybrid.py.

The experiment combines:
- BO within-tier: For each fold, for each regime, for each tier, evaluate 
  all algorithms in that tier on training data and pick the best one.
  This gives best_algo[regime][tier] — a lookup table.
  
- ML tier selector: Reuse TierSelector from meta_learner/hierarchical_network.py.
  Train it to blend 3 BO-selected tier portfolios (not specialist outputs).
  Same config as Plan 13a Phase B: lr=0.005, epochs=50, lambda_tier=0.05.

Reuse cached algo outputs from results/plan13a_hierarchical/cache/fold_XX_algo_outputs.npy.
Reuse MetaLearnerDataset from meta_learner/dataset.py.

Test three strategies:
A) Hybrid: ML tier selector + BO within-tier
B) BO-only per-tier: weight tiers by in-sample Sharpe (no NN)
C) Best-tier-only: always pick the tier with best in-sample Sharpe per regime

Baselines: EW, Plan 13b Hard Selection, Plan 13b Top-3 Blend.

Print a final comparison table of ALL approaches across all plans (13a, 13b, 13c).
Save results to results/plan13c_hybrid/.

Run the full 12-fold experiment, then commit and push.
```
