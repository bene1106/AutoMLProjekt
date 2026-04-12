# Implementation Plan 13a: Hierarchical Meta-Learner with Tier 3

## Overview

**Goal:** Implement a hierarchical meta-learner that decomposes algorithm selection into two levels: (1) a Tier Selector choosing among 3 tiers, and (2) specialized within-tier models selecting among algorithms within each tier. Additionally, implement Tier 3 non-linear ML algorithms to complete the algorithm space.

**Motivation:** Plan 12 demonstrated that a flat softmax over K=81 algorithms is fundamentally unstable — entropy either collapses to 0 (one algorithm gets 100%) or stays at log(81)≈4.4 (uniform/no learning), with a sharp phase transition between the two. Prof. Ler confirmed: "81 is too much for less data" and directed hierarchical decomposition with specialized models.

**Training approach:** Sequential (3-Stage) — Stage 0 (frozen algos, now including Tier 3) → Stage 1 (oracle regime, no classifier) → Stage 2 (hierarchical meta-learner). Within Stage 2, training is two-phase: Phase A trains each specialist independently, Phase B trains the tier selector with frozen specialists.

**Key design choices:**
- Oracle regime (same as Plan 12) to isolate meta-learner quality
- Independent specialists (not one joint network) — per Prof. Ler: "have specialized models, not a general one"
- Entropy regularization on both levels (learned from Plan 12's experiments)

**This is Option 1** of Prof. Ler's three-option framework:
- **Option 1 (this plan):** Hierarchical Meta-Learner, no Bayesian Optimization
- **Option 2 (Plan 13c, later):** Meta-Learner for tier selection + BO for within-tier configuration
- **Option 3 (Plan 13b, later):** Pure BO, no Meta-Learner

---

## What This Plan Does NOT Do

- No Bayesian Optimization (that's Plans 13b/13c)
- No real regime classifier (oracle only — same as Plan 12)
- No end-to-end or alternating training (sequential only)
- No algorithm outputs in the meta-learner input
- No adversarial training (Prof. Ler mentioned this for later)
- No pairwise decomposition

---

## Why Plan 12's Full-Run Is Skipped

Plan 12's last recommended step was a full 12-fold run with λ=0.08. We skip this because:

1. **Prof. Ler rejected the flat architecture.** Running a full experiment on a discarded architecture wastes time.
2. **The diagnostic data is already sufficient.** 3 Runs + 3 λ-Sweeps (12 data points) clearly show the phase transition problem. Only 1/12 configurations reached the target entropy range.
3. **The negative result stands as-is.** In the thesis, Plan 12's results motivate the hierarchical decomposition. No full-run needed for that.

---

## Prerequisites

Everything from Plans 1–4 and Plan 12 must be available in `regime_algo_selection/`:

- `data/loader.py` — `load_data()` returns `{"prices": DataFrame, "vix": Series}` (2006–2024)
- `data/features.py` — `compute_asset_features(prices)` returns MultiIndex DataFrame with 9 features per asset. Also `compute_returns(prices)` for forward returns.
- `algorithms/tier1_heuristics.py` — 48 Tier 1 algorithms (EW, MinVar, RiskParity, MaxDiv, Momentum, TrendFollowing families)
- `algorithms/tier2_linear.py` — 33 Tier 2 algorithms (Ridge, Lasso, ElasticNet with various hyperparameters)
- `algorithms/base.py` — `PortfolioAlgorithm` base class with `compute_weights(prices_history)` and `TrainablePortfolioAlgorithm` with `fit()`, `is_fitted`, `_compute_feature_row()`, `_softmax()`
- `algorithms/stage0.py` — `pretrain_tier2_algorithms(algorithms, asset_features, returns, train_start, train_end)`
- `regimes/ground_truth.py` — `compute_regime_labels(vix)` returns Series {1,2,3,4}
- `config.py` — `ASSETS=["SPY","TLT","GLD","EFA","VNQ"]`, `KAPPA=0.001`, `REGIME_THRESHOLDS=[15,20,30]`
- `meta_learner/dataset.py` — `MetaLearnerDataset` from Plan 12 (reused for feature assembly)

**IMPORTANT:** `build_algorithm_space()` currently supports `tiers=[1,2]` returning 81 algorithms. This plan extends it to `tiers=[1,2,3]`.

---

## File Structure (New and Modified Files)

```
regime_algo_selection/
├── algorithms/
│   ├── tier3_nonlinear.py          # NEW: Tier 3 non-linear ML algorithms
│   ├── __init__.py                 # MODIFIED: add Tier 3 to build_algorithm_space()
│   └── stage0.py                   # MODIFIED: extend pre-training to Tier 3
├── meta_learner/
│   ├── hierarchical_network.py     # NEW: TierSelector + TierSpecialist + HierarchicalMetaLearner
│   ├── hierarchical_trainer.py     # NEW: Two-phase training (specialists → tier selector)
│   ├── dataset.py                  # UNCHANGED (reused from Plan 12)
│   ├── network.py                  # UNCHANGED (kept for Plan 12 comparison)
│   ├── trainer.py                  # UNCHANGED (kept for Plan 12 comparison)
│   └── inference.py                # UNCHANGED
├── experiments/
│   └── plan13a_hierarchical.py     # NEW: Main experiment script
└── results/
    └── plan13a_hierarchical/       # NEW: Output directory
```

---

## Step 0: Implement Tier 3 Non-Linear ML Algorithms

**File:** `algorithms/tier3_nonlinear.py`

### Design

Tier 3 algorithms follow the same pattern as Tier 2: predict next-period asset returns from lagged features, then convert predictions to portfolio weights via softmax allocation. The difference is the model family — non-linear instead of linear.

### Model Families and Hyperparameter Grid

```
Family F11: Random Forest
  n_estimators ∈ {100, 300}
  max_depth ∈ {5, 10, None}
  lookback L ∈ {60, 120}
  → 2 × 3 × 2 = 12 configurations

Family F12: Gradient Boosting
  n_estimators ∈ {100, 300}
  max_depth ∈ {3, 5}
  learning_rate ∈ {0.05, 0.1}
  lookback L ∈ {60, 120}
  → 2 × 2 × 2 × 2 = 16 configurations

Family F13: MLP (Neural Network)
  hidden_layer_sizes ∈ {(64,), (64,32)}
  alpha ∈ {0.0001, 0.001}
  lookback L ∈ {60, 120}
  → 2 × 2 × 2 = 8 configurations

Total Tier 3: 12 + 16 + 8 = 36 algorithm configurations
```

### Implementation

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

class Tier3Algorithm(TrainablePortfolioAlgorithm):
    """
    Non-linear ML algorithm for return prediction → portfolio weights.
    Same interface as Tier 2: fit() on training data, compute_weights() at decision time.
    
    Pipeline:
    1. fit(): Train sklearn model on (features, next_period_returns) pairs
    2. compute_weights(): Predict returns for next period → softmax → portfolio weights
    """
    
    def __init__(self, family: str, model_class, model_params: dict, 
                 lookback: int, name: str):
        """
        Args:
            family: "RandomForest", "GradientBoosting", or "MLP"
            model_class: sklearn class
            model_params: dict of hyperparameters
            lookback: L, number of days of history used as features
            name: human-readable name, e.g. "RF_n100_d5_L60"
        """
        super().__init__(name=name, tier=3)
        self.family = family
        self.model_class = model_class
        self.model_params = model_params
        self.lookback = lookback
        self.models = {}  # asset_name → fitted model
    
    def fit(self, prices_history, features=None):
        """
        Train one model per asset on (lagged_features, next_return) pairs.
        Uses _compute_feature_row() from TrainablePortfolioAlgorithm (same as Tier 2).
        """
        pass
    
    def compute_weights(self, prices_history):
        """
        Predict next-period returns → softmax → portfolio weights ∈ Δ_N.
        Uses _softmax() from TrainablePortfolioAlgorithm (same as Tier 2).
        """
        pass


def build_tier3_algorithms():
    """Returns list of all 36 Tier 3 algorithm instances."""
    algorithms = []
    
    # F11: Random Forest
    for n_est in [100, 300]:
        for max_d in [5, 10, None]:
            for L in [60, 120]:
                depth_str = str(max_d) if max_d else "None"
                name = f"RF_n{n_est}_d{depth_str}_L{L}"
                algorithms.append(Tier3Algorithm(
                    family="RandomForest",
                    model_class=RandomForestRegressor,
                    model_params={"n_estimators": n_est, "max_depth": max_d, 
                                  "random_state": 42, "n_jobs": -1},
                    lookback=L, name=name
                ))
    
    # F12: Gradient Boosting
    for n_est in [100, 300]:
        for max_d in [3, 5]:
            for lr in [0.05, 0.1]:
                for L in [60, 120]:
                    name = f"GBM_n{n_est}_d{max_d}_lr{lr}_L{L}"
                    algorithms.append(Tier3Algorithm(
                        family="GradientBoosting",
                        model_class=GradientBoostingRegressor,
                        model_params={"n_estimators": n_est, "max_depth": max_d,
                                      "learning_rate": lr, "random_state": 42},
                        lookback=L, name=name
                    ))
    
    # F13: MLP
    for hidden in [(64,), (64, 32)]:
        for alpha in [0.0001, 0.001]:
            for L in [60, 120]:
                hidden_str = "x".join(str(h) for h in hidden)
                name = f"MLP_h{hidden_str}_a{alpha}_L{L}"
                algorithms.append(Tier3Algorithm(
                    family="MLP",
                    model_class=MLPRegressor,
                    model_params={"hidden_layer_sizes": hidden, "alpha": alpha,
                                  "max_iter": 500, "random_state": 42,
                                  "early_stopping": True, "validation_fraction": 0.1},
                    lookback=L, name=name
                ))
    
    return algorithms  # 36 algorithms
```

### Modifications to Existing Files

**`algorithms/__init__.py`** — Update `build_algorithm_space()`:
```python
def build_algorithm_space(tiers=[1, 2]):
    algorithms = []
    if 1 in tiers:
        algorithms.extend(build_tier1_algorithms())    # 48
    if 2 in tiers:
        algorithms.extend(build_tier2_algorithms())    # 33
    if 3 in tiers:
        algorithms.extend(build_tier3_algorithms())    # 36
    return algorithms
```

**`algorithms/stage0.py`** — Extend pre-training to include Tier 3:
```python
def pretrain_algorithms(algorithms, prices, train_start, train_end):
    """Pre-train all trainable algorithms (Tier 2 + Tier 3)."""
    prices_train = prices.loc[train_start:train_end]
    for algo in algorithms:
        if hasattr(algo, 'fit') and algo.tier in [2, 3]:
            algo.fit(prices_train)
```

### Verification

After implementation, verify with a single fold:
- All 36 Tier 3 algorithms produce valid weights (≥0, sum to 1, no NaN)
- `build_algorithm_space(tiers=[1,2,3])` returns exactly 117 algorithms (48+33+36)
- Stage 0 pre-training completes without errors

---

## Step 1: Hierarchical Meta-Learner Network

**File:** `meta_learner/hierarchical_network.py`

### Architecture Overview

```
Level 1: TierSelector
   Input: X_t (29 dims: 25 asset features + 4 regime one-hot)
   Hidden: [64, 32] → ReLU → Dropout(0.1)
   Output: β_t ∈ Δ_3 (softmax over 3 tiers)

Level 2: TierSpecialist (one per tier, INDEPENDENT networks)
   Tier 1 Specialist: Input (29) → [64, 32] → Softmax(48)   → γ^(1)_t
   Tier 2 Specialist: Input (29) → [64, 32] → Softmax(33)   → γ^(2)_t
   Tier 3 Specialist: Input (29) → [64, 32] → Softmax(36)   → γ^(3)_t

Combined weight:
   α_{t,k} = β_{t,f} · γ^(f)_{t,j}   for algorithm k in tier f, position j

Final portfolio:
   w_t = Σ_k α_{t,k} · w^(k)_t
```

### Implementation

```python
import torch
import torch.nn as nn


class TierSelector(nn.Module):
    """
    Level 1: Selects mixing weights over 3 tiers.
    Small network — only 3 output dims, very stable softmax.
    H_max = log(3) ≈ 1.10, no phase transition issues expected.
    """
    
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 3)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        h = self.feature_extractor(x)
        logits = self.output_layer(h)
        return torch.softmax(logits, dim=-1)


class TierSpecialist(nn.Module):
    """
    Level 2: Selects mixing weights within one tier.
    Independent network — does not share parameters with selector or other specialists.
    Largest softmax is K1≈48 (vs. K=81 in Plan 12). H_max ≈ 3.87 vs. 4.39.
    """
    
    def __init__(self, input_dim, n_algorithms, hidden_dims=[64, 32], dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, n_algorithms)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        h = self.feature_extractor(x)
        logits = self.output_layer(h)
        return torch.softmax(logits, dim=-1)


class HierarchicalMetaLearner(nn.Module):
    """
    Combines TierSelector + 3 TierSpecialists.
    
    Combined weight: α_{t,k} = β_{t,f} · γ^(f)_{t,j}
    
    Guarantees α_t ∈ Δ_K because:
        Σ_k α_{t,k} = Σ_f β_{t,f} · Σ_j γ^(f)_{t,j} = Σ_f β_{t,f} · 1 = 1
    """
    
    def __init__(self, input_dim, tier_sizes, 
                 selector_hidden=[64, 32], specialist_hidden=[64, 32], dropout=0.1):
        super().__init__()
        self.tier_sizes = tier_sizes
        self.n_tiers = len(tier_sizes)
        self.total_algorithms = sum(tier_sizes)
        
        self.tier_selector = TierSelector(input_dim, selector_hidden, dropout)
        self.specialists = nn.ModuleList([
            TierSpecialist(input_dim, k_f, specialist_hidden, dropout)
            for k_f in tier_sizes
        ])
    
    def forward(self, x):
        beta_t = self.tier_selector(x)
        gammas = [spec(x) for spec in self.specialists]
        
        alpha_parts = []
        for f in range(self.n_tiers):
            alpha_parts.append(beta_t[..., f:f+1] * gammas[f])
        alpha_t = torch.cat(alpha_parts, dim=-1)
        
        return alpha_t, beta_t, gammas
```

### Input Composition

Same as Plan 12:
- Asset features: 25 dims (5 features × 5 ETFs)
- Regime signal: 4-dim one-hot of oracle regime s*_t
- Total input: 29 dims
- No algorithm outputs in input

### Parameter Count Comparison

```
Plan 12 (Flat): Input(29) → [128, 64] → Softmax(81) = 17,169 params (1 network)

Plan 13a (Hierarchical):
  TierSelector:    29→[64,32]→3   = ~4,067 params
  Specialist T1:   29→[64,32]→48  = ~5,504 params
  Specialist T2:   29→[64,32]→33  = ~5,023 params
  Specialist T3:   29→[64,32]→36  = ~5,120 params
  Total: ~19,714 params (4 independent networks)
```

More total parameters, but each network is smaller and more focused. Largest softmax: 48 (vs. 81).

---

## Step 2: Two-Phase Hierarchical Trainer

**File:** `meta_learner/hierarchical_trainer.py`

### Training Strategy

**Phase A — Train each Specialist independently:**

For each tier f ∈ {1, 2, 3}:
1. Use ONLY the algorithms belonging to tier f
2. Specialist computes γ^(f)_t → composite portfolio using only tier f's algorithms
3. Each specialist optimizes as if it controls the entire portfolio (β_f = 1)
4. Loss: -(reward × 252 + λ_spec × H(γ^(f)_t))
5. Separate optimizer per specialist

**Phase B — Train Tier Selector (freeze specialists):**

1. Freeze all 3 specialist networks
2. Each specialist produces its tier-level composite portfolio (frozen)
3. Tier selector learns β_t to blend the 3 tier-level portfolios
4. Loss: -(reward × 252 + λ_tier × H(β_t))

### Entropy Regularization

```
Phase A loss (per specialist): loss = -(reward × 252.0 + λ_spec × H(γ^(f)_t))
Phase B loss (tier selector):  loss = -(reward × 252.0 + λ_tier × H(β_t))
```

Why entropy is easier now:
```
Plan 12: Softmax(81) → H_max = 4.39 → sharp phase transition
Plan 13a:
  TierSelector Softmax(3)  → H_max = 1.10 → very smooth
  Specialist 1 Softmax(48) → H_max = 3.87 → better than 4.39
  Specialist 2 Softmax(33) → H_max = 3.50
  Specialist 3 Softmax(36) → H_max = 3.58
```

### Implementation

```python
class HierarchicalTrainer:
    """
    Two-phase trainer for the hierarchical meta-learner.
    
    Phase A: Train each specialist independently on its own tier's algorithms.
    Phase B: Freeze specialists, train tier selector on blended portfolio.
    
    Both phases use sequential (non-shuffled) processing through time
    because switching costs create temporal dependencies (same as Plan 12).
    """
    
    def __init__(self, model, tier_algorithm_indices, kappa=0.001, kappa_a=0.0,
                 specialist_lr=0.005, selector_lr=0.005, 
                 specialist_epochs=80, selector_epochs=50,
                 lambda_spec=0.05, lambda_tier=0.05,
                 weight_decay=1e-4, grad_clip=1.0):
        """
        Args:
            model: HierarchicalMetaLearner instance
            tier_algorithm_indices: list of 3 lists, each containing global indices
                                   of algorithms belonging to that tier.
                                   E.g. [[0..47], [48..80], [81..116]]
            kappa: portfolio switching cost coefficient
            kappa_a: algorithm switching cost coefficient (0 for first run)
            specialist_lr: learning rate for Phase A
            selector_lr: learning rate for Phase B
            specialist_epochs: number of epochs per specialist (Phase A)
            selector_epochs: number of epochs for tier selector (Phase B)
            lambda_spec: entropy reg weight for specialists
            lambda_tier: entropy reg weight for tier selector
        """
        self.model = model
        self.tier_indices = tier_algorithm_indices
        self.kappa = kappa
        self.kappa_a = kappa_a
        self.lambda_spec = lambda_spec
        self.lambda_tier = lambda_tier
        self.specialist_epochs = specialist_epochs
        self.selector_epochs = selector_epochs
        self.grad_clip = grad_clip
        
        self.specialist_optimizers = [
            torch.optim.Adam(spec.parameters(), lr=specialist_lr, weight_decay=weight_decay)
            for spec in model.specialists
        ]
        self.selector_optimizer = torch.optim.Adam(
            model.tier_selector.parameters(), lr=selector_lr, weight_decay=weight_decay
        )
    
    def smooth_l1(self, x, eps=1e-8):
        return torch.sum(torch.sqrt(x**2 + eps))
    
    def train_phase_a(self, dataset, train_indices):
        """Phase A: Train each specialist independently."""
        # See detailed pseudocode in architecture section
        pass
    
    def train_phase_b(self, dataset, train_indices):
        """Phase B: Freeze specialists, train tier selector."""
        # See detailed pseudocode in architecture section
        pass
    
    def train_fold(self, dataset, train_indices):
        """Full two-phase training for one walk-forward fold."""
        print("--- Phase A: Training Specialists ---")
        specialist_histories = self.train_phase_a(dataset, train_indices)
        print("--- Phase B: Training Tier Selector ---")
        selector_history = self.train_phase_b(dataset, train_indices)
        return {
            'specialist_histories': specialist_histories,
            'selector_history': selector_history
        }
```

### Important Implementation Notes (same as Plan 12)

1. **Sequential processing, not batched:** Switching cost depends on w_{t-1}, process days chronologically.
2. **Detach previous states:** w_prev and alpha_prev must be detached after each step.
3. **Algorithm outputs are constants:** No gradients through w_t^(k).
4. **Warm-up period:** First ~60 days skipped (insufficient lookback for momentum_60d).
5. **Gradient clipping:** Prevents explosion in early training.

---

## Step 3: Experiment Script

**File:** `experiments/plan13a_hierarchical.py`

### Configuration

```python
CONFIG = {
    # Network architecture
    "input_dim": 29,
    "selector_hidden": [64, 32],
    "specialist_hidden": [64, 32],
    "dropout": 0.1,
    
    # Training Phase A (Specialists)
    "specialist_lr": 0.005,
    "specialist_epochs": 80,
    "lambda_spec": 0.05,
    
    # Training Phase B (Tier Selector)
    "selector_lr": 0.005,
    "selector_epochs": 50,
    "lambda_tier": 0.05,
    
    # Costs
    "kappa": 0.001,
    "kappa_a": 0.0,
    
    # Data
    "asset_features": ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "mom_60d"],
    "regime_encoding": "onehot",
    "use_oracle_regime": True,
    
    # Walk-forward
    "train_years": 8,
    "n_folds": 12,
    "test_years": list(range(2013, 2025)),
    
    # Algorithm space
    "tiers": [1, 2, 3],
}
```

### Experiment Flow (per fold)

```
For each fold f ∈ {1, ..., 12}:

1. SPLIT data into train/test (expanding window)

2. STAGE 0 — Pre-compute ALL algorithm outputs
   - build_algorithm_space(tiers=[1,2,3]) → K≈117 algorithms
   - Pre-train Tier 2 + Tier 3 on training block
   - Compute w_t^(k) for all K algorithms, all time steps
   - Record tier_indices: [[0..47], [48..80], [81..116]]

3. PREPARE dataset
   - Compute asset features (25 dims)
   - Get oracle regime labels (4-dim one-hot)
   - Fit StandardScaler on training features only
   - Create MetaLearnerDataset (reuse from Plan 12)

4. INITIALIZE hierarchical meta-learner
   - HierarchicalMetaLearner(input_dim=29, tier_sizes=[K1, K2, K3])

5. TRAIN (Phase A + Phase B)
   - HierarchicalTrainer.train_fold(dataset, train_indices)

6. EVALUATE on test year
   - Run full hierarchical model on test data
   - Compute metrics: Sharpe, Sortino, MaxDD, Turnover
   - Compute entropies: H(α_t), H(β_t), H(γ^(f)_t) per specialist

7. BASELINES
   - Equal Weight (1/N daily rebalance)
   - Best Individual Algorithm (oracle: best in-sample Sharpe)
   - Plan 12 Flat Meta-Learner results (if available)
   - Per-Tier Equal Weight (EW within each tier, then EW across tiers)
```

### Outputs

```
results/plan13a_hierarchical/
├── summary_metrics.csv              # 12 folds × all metrics
├── tier_selection_by_regime.csv     # mean β_t per regime (4×3) per fold
├── specialist_entropy.csv           # H(γ^(f)_t) per specialist per fold
├── training_history.json            # loss curves for all components
├── per_fold/
│   ├── fold_01_2013.json
│   ├── ...
│   └── fold_12_2024.json
└── comparison_vs_baselines.csv
```

### Summary Table (printed to console)

```
Fold | Year | EW_Sharpe | BestAlgo | Plan13a_Hier | H(β) | H(γ1) | H(γ2) | H(γ3) | H(α)
-----|------|-----------|----------|--------------|------|--------|--------|--------|------
  1  | 2013 |   ...     |   ...    |     ...      | ...  |  ...   |  ...   |  ...   | ...
 ...
 12  | 2024 |   ...     |   ...    |     ...      | ...  |  ...   |  ...   |  ...   | ...
 AVG |      |   ...     |   ...    |     ...      | ...  |  ...   |  ...   |  ...   | ...
```

---

## Step 4: Sanity Check (Run FIRST)

### Quick Verification

Run only **Fold 1 (test 2013) and Fold 6 (test 2018)** with reduced epochs:

```python
SANITY_CONFIG = {
    **CONFIG,
    "specialist_epochs": 20,
    "selector_epochs": 15,
}
```

### Checklist

```
[ ] All 117 algorithms produce valid weights (no NaN, ≥0, sum to 1)
[ ] Tier 3 Stage 0 pre-training completes without errors
[ ] Each specialist entropy H(γ^(f)) in [0.3, H_max - 0.3]:
      Tier 1: [0.3, 3.57]   (H_max = 3.87)
      Tier 2: [0.3, 3.20]   (H_max = 3.50)
      Tier 3: [0.3, 3.28]   (H_max = 3.58)
[ ] Tier selector entropy H(β) ∈ [0.1, 1.0]  (H_max = 1.10)
[ ] Composite portfolio weights valid (≥0, sum to 1)
[ ] Sharpe ratio computable (no NaN)
[ ] Training loss decreases over epochs
```

### If Entropy Issues Occur

Quick λ-sweep on Fold 1 and Fold 6:
```python
LAMBDA_SPEC_VALUES = [0.01, 0.03, 0.05, 0.1, 0.2]
LAMBDA_TIER_VALUES = [0.01, 0.03, 0.05, 0.1]
```

---

## Step 5: Analysis

### Central Question

> Does the Tier Selector learn to assign different tiers to different regimes?

### Key Diagnostics

**Regime-conditional tier selection (4×3 matrix):**
```
For each regime s ∈ {Calm, Normal, Tense, Crisis}:
    mean_beta[s] = average β_t over all t where s*_t = s
```

**Specialist standalone performance:**
```
For each tier f:
    standalone_sharpe[f] = Sharpe of specialist f alone (β_f = 1)
```

**Oracle Gap (hierarchical version):**
```
Gap 1: Hierarchical ML vs. Perfect Tier Selection → Tier Selector quality
Gap 2: Perfect Tier vs. Perfect Algorithm → cost of tier grouping
```

---

## Expected Runtime

```
Per fold:
  Stage 0 (pre-train Tier 2+3):    ~3-5 min
  Pre-compute algorithm outputs:     ~2-3 min
  Phase A (3 specialists × 80 ep):  ~5-8 min
  Phase B (selector × 50 ep):       ~2-3 min
  Evaluation:                        ~1 min
  Total per fold:                    ~13-20 min

Full 12-fold run:          ~2.5-4 hours
Sanity check (2 folds):    ~10-15 min
```

---

## Success Criteria

| Criterion | Target | If Failed |
|-----------|--------|-----------|
| Specialist entropy H(γ^(f)) | In [0.3, H_max-0.3] for ≥8/12 folds | λ_spec sweep |
| Tier selector entropy H(β) | In [0.1, 1.0] for ≥8/12 folds | λ_tier sweep |
| Regime sensitivity | β_t differs across ≥2 regimes | Document as negative result |
| Sharpe ratio | ≥ EW in ≥4/12 folds | Adjust hyperparameters |
| Stability | No NaN/crashes in 12 folds | Debug individual failures |

---

## Connection to Plans 13b and 13c

This plan produces:
1. **Trained specialists** → warm start for BO in Plan 13c
2. **Baseline performance** → comparison target for Plan 13b (pure BO)
3. **Tier selection patterns** → evidence whether regime-dependent tier selection exists
4. **Tier 3 implementation** → reused by all subsequent plans
