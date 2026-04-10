# Implementation Plan 12: Meta-Learner with Oracle Regime

## Overview

**Goal:** Implement the first real Meta-Learner (Stage 2) that learns mixing weights α_t ∈ Δ_K over all K≈81 base algorithms, using the **oracle regime** (true s*_t ∈ {1,2,3,4}) as input.

**Training approach:** Sequential (3-Stage) — Stage 0 (frozen algos from Plan 4) → Stage 1 (oracle regime, no classifier needed) → Stage 2 (Meta-Learner training).

**Input composition:** Asset features + oracle regime. **No** algorithm outputs w_t^(k) in the input.

**Selection mechanism:** Flat softmax over K algorithms.

**Key design choice:** Using the oracle regime isolates the Meta-Learner's learning ability from classifier errors. This establishes the **upper bound** for the sequential training approach. Later, we swap in the real classifier and measure the Oracle Gap.

---

## What This Plan Does NOT Do

- No hierarchical softmax (flat only)
- No algorithm outputs in the meta-learner input
- No real regime classifier (oracle only)
- No end-to-end or alternating training
- No Tier 3 algorithms
- No pairwise decomposition

---

## Prerequisites

Everything from Plans 1–4 must be available in `regime_algo_selection/`:
- `data/loader.py` — `load_data()` returns `{"prices": DataFrame, "vix": Series}` (2006–2024, cached in `data/cache/`)
- `data/features.py` — `compute_asset_features(prices)` returns MultiIndex DataFrame (asset, feature) with 9 features per asset (ret_1d, ret_5d, ret_20d, ret_60d, vol_20d, vol_60d, mom_20d, mom_60d, mom_120d). Also `compute_returns(prices)` for forward returns.
- `algorithms/tier1_heuristics.py` — `build_algorithm_space(tiers=[1,2])` returns all 81 algorithms (48 Tier 1 + 33 Tier 2)
- `algorithms/tier2_linear.py` — Ridge/Lasso/ElasticNet with `fit()` and `compute_weights(prices_history)`
- `algorithms/base.py` — `PortfolioAlgorithm` base class with `compute_weights(prices_history)` signature; `TrainablePortfolioAlgorithm` extends with `fit()`, `is_fitted`, `_compute_feature_row()`, `_softmax()`
- `algorithms/stage0.py` — `pretrain_tier2_algorithms(algorithms, asset_features, returns, train_start, train_end)`
- `regimes/ground_truth.py` — `compute_regime_labels(vix)` returns Series {1,2,3,4}
- `evaluation/walk_forward.py` — `WalkForwardValidator` with `generate_folds()` and `run_fold()`. Uses **expanding windows** from 2005, 8yr train + 1yr test, test years 2013–2024.
- `evaluation/backtest.py` — `Backtester` class and `BacktestResult` dataclass
- `evaluation/metrics.py` — `compute_all_metrics(result)` returns dict with sharpe_ratio, sortino, max_drawdown, etc.
- `config.py` — `ASSETS=["SPY","TLT","GLD","EFA","VNQ"]`, `KAPPA=0.001`, `REGIME_THRESHOLDS=[15,20,30]`

**IMPORTANT:** The existing `WalkForwardValidator.run_fold()` is built around the ReflexAgent pattern (classifier + agent.select(regime)). Plan 12 does NOT use `run_fold()` — it implements its own walk-forward loop because the meta-learner has a fundamentally different interface (X_t → α_t → w_t instead of regime → algorithm name).

**IMPORTANT:** The existing `compute_asset_features()` already computes 9 features per asset (45 total). Plan 12 uses a SUBSET of 5 features per asset (25 total) for the initial meta-learner input. The full 9-feature set can be tested later as an ablation.

---

## File Structure (New Files Only)

```
regime_algo_selection/
├── meta_learner/
│   ├── __init__.py
│   ├── network.py              # PyTorch neural network (flat softmax)
│   ├── dataset.py              # Dataset class: assembles X_t from features + regime
│   ├── trainer.py              # Training loop: reward maximization
│   └── inference.py            # At decision time: X_t → α_t → w_t
├── experiments/
│   └── plan5_meta_learner.py   # Main experiment script
└── results/
    └── (outputs from this plan)
```

---

## Step 1: Asset Feature Engineering for the Meta-Learner

**File:** `meta_learner/dataset.py`

The meta-learner receives asset features x_{t,i} for each of the N=5 ETFs. These are the **candidate features** from the Architecture Document (Table 2). For the first implementation, use a **minimal but informative** feature set:

### Initial Asset Feature Set (per asset, per day)

The existing `data/features.py:compute_asset_features()` computes 9 features per asset. For the initial meta-learner, use a **subset of 5** (drop ret_60d, vol_60d, mom_20d, mom_60d which overlap heavily with other features):

```
Feature                  | Existing column name        | Lookback
-------------------------|-----------------------------|--------
return_1d                | (asset, "ret_1d")           | 1d
return_5d                | (asset, "ret_5d")           | 5d
return_20d               | (asset, "ret_20d")          | 20d
rolling_vol_20d          | (asset, "vol_20d")          | 20d
momentum_60d             | (asset, "mom_60d")          | 60d
```

This gives d=5 features per asset, total asset feature block: N×d = 5×5 = 25 dimensions.

**How to extract:** The existing `compute_asset_features(prices)` returns a DataFrame with MultiIndex columns `(asset, feature)`. Select the 5 features:
```python
SELECTED_FEATURES = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "mom_60d"]
asset_feats = compute_asset_features(prices)
# Select only the 5 features we want
selected = asset_feats.loc[:, asset_feats.columns.get_level_values("feature").isin(SELECTED_FEATURES)]
# Flatten to (T, 25) array, sorted by (asset, feature)
X_assets = selected.values  # shape (T, N_assets * 5) = (T, 25)
```

### Regime Feature

The oracle regime s*_t ∈ {1,2,3,4} is encoded as a **one-hot vector** of dimension 4:
- Calm:   [1, 0, 0, 0]
- Normal: [0, 1, 0, 0]
- Tense:  [0, 0, 1, 0]
- Crisis: [0, 0, 0, 1]

One-hot is preferred over scalar encoding because the regimes are categorical, not ordinal in terms of what the meta-learner should do (Calm doesn't mean "less" than Crisis in terms of algorithm selection).

### Full Input Vector

```
X_t = [asset_features (25), regime_onehot (4)] → total input dim = 29
```

### Implementation Details

```python
class MetaLearnerDataset:
    """
    Assembles the input X_t and computes all quantities needed for training.
    
    For each time step t in the dataset:
    - X_t: input vector (asset features + regime one-hot) — shape (29,)
    - w_t^(k): outputs of all K algorithms at time t — shape (K, N)
    - r_{t→t+1}: next-period return vector — shape (N,)
    - s*_t: true regime label (for oracle)
    
    All features use LAGGED data only (t-1 and earlier).
    """
    
    def __init__(self, prices, vix, algorithms, regime_labels):
        # prices: DataFrame (dates × N assets)
        # vix: Series (dates)
        # algorithms: list of K PortfolioAlgorithm objects
        # regime_labels: Series (dates → {1,2,3,4})
        pass
    
    def compute_asset_features(self, t):
        """
        Returns feature vector for all assets at time t.
        Shape: (N*d,) = (25,)
        
        CRITICAL: All lookback windows must have enough history.
        The longest lookback is 60d (momentum_60d), so the first 
        valid time step is day 60 of the dataset.
        
        StandardScaler: fit on training data only, transform both 
        train and test. This is done per walk-forward fold.
        """
        pass
    
    def compute_regime_onehot(self, t):
        """
        Returns one-hot encoding of s*_t.
        Shape: (4,)
        """
        pass
    
    def compute_algorithm_outputs(self, t):
        """
        Runs all K algorithms and collects their weight vectors.
        Shape: (K, N)
        
        NOTE: These are NOT part of the meta-learner input in this plan,
        but they ARE needed for computing the composite portfolio w_t
        during training and inference.
        """
        pass
    
    def get_returns(self, t):
        """
        Returns the realized return vector r_{t→t+1}.
        Shape: (N,)
        """
        pass
```

### Data Leakage Prevention Checklist

- [ ] All features use data up to t-1 (never t or later)
- [ ] StandardScaler fitted on training block only
- [ ] Regime labels are ground truth from VIX (no look-ahead — VIX_t is known at close of day t)
- [ ] Algorithm outputs w_t^(k) use only lagged data (this is guaranteed by the algorithms themselves, already verified in Plans 1-4)
- [ ] No feature uses future returns

---

## Step 2: Meta-Learner Neural Network

**File:** `meta_learner/network.py`

### Architecture

```python
import torch
import torch.nn as nn

class MetaLearnerNetwork(nn.Module):
    """
    Feedforward neural network with softmax output.
    Maps X_t → α_t ∈ Δ_K.
    
    Architecture:
        Input (29) → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear → Softmax (K)
    """
    
    def __init__(self, input_dim, n_algorithms, hidden_dims=[128, 64], dropout=0.2):
        """
        Args:
            input_dim: dimension of X_t (= 29 for initial setup)
            n_algorithms: K (≈81)
            hidden_dims: list of hidden layer sizes
            dropout: dropout probability
        """
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
        # NOTE: No softmax here — we apply it in forward() for numerical stability
        # When computing loss, use the logits directly with appropriate loss function
    
    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, input_dim) or (input_dim,)
        
        Returns:
            alpha: mixing weights, shape (batch_size, K) or (K,), values in Δ_K
        """
        h = self.feature_extractor(x)
        logits = self.output_layer(h)
        alpha = torch.softmax(logits, dim=-1)
        return alpha
    
    def get_logits(self, x):
        """Returns raw logits before softmax (useful for some loss formulations)."""
        h = self.feature_extractor(x)
        return self.output_layer(h)
```

### Hyperparameter Candidates (for later tuning)

```
hidden_dims:  [128, 64] (default), [64, 32], [256, 128, 64]
dropout:      0.1, 0.2, 0.3
learning_rate: 1e-4, 5e-4, 1e-3
weight_decay:  1e-5, 1e-4, 1e-3
```

For the **first run**, use: hidden_dims=[128, 64], dropout=0.2, lr=1e-3, weight_decay=1e-4.

---

## Step 3: Training Loop (Reward Maximization)

**File:** `meta_learner/trainer.py`

### Loss Function Design

The meta-learner maximizes cumulative reward. Since we use gradient-based optimization (Adam), we minimize the **negative reward**:

```
Loss_t = -reward_{t+1} = -(w_t^T r_{t→t+1} - κ ||w_t - w_{t-1}||_1 - κ_a ||α_t - α_{t-1}||_1)
```

where:
```
w_t = Σ_k α_{t,k} · w_t^(k)     (composite portfolio from mixing weights)
```

**Why this is differentiable:** 
- α_t = softmax(logits) — differentiable
- w_t = Σ_k α_{t,k} · w_t^(k) — linear in α_t, differentiable
- w_t^T r_{t+1} — linear in w_t, differentiable
- ||w_t - w_{t-1}||_1 — NOT differentiable at zero, but we can use a smooth approximation:
  ||x||_1 ≈ Σ_i sqrt(x_i^2 + ε) with ε=1e-8 (Huber-like smoothing)

### Training Procedure

```python
class MetaLearnerTrainer:
    """
    Trains the meta-learner on a single walk-forward fold.
    
    Training is SEQUENTIAL through time (not shuffled batches), because:
    1. The switching cost depends on w_{t-1} and α_{t-1}
    2. The temporal ordering matters for the cost structure
    
    We use MINI-EPOCHS: sweep through the training data multiple times,
    each time processing days sequentially.
    """
    
    def __init__(self, network, algorithms, kappa=0.001, kappa_a=0.0, 
                 lr=1e-3, weight_decay=1e-4, n_epochs=50):
        """
        Args:
            network: MetaLearnerNetwork instance
            algorithms: list of K frozen PortfolioAlgorithm instances
            kappa: portfolio switching cost coefficient
            kappa_a: algorithm switching cost coefficient (0 for first experiment)
            lr: learning rate for Adam
            weight_decay: L2 regularization
            n_epochs: number of passes through the training data
        """
        self.network = network
        self.algorithms = algorithms
        self.kappa = kappa
        self.kappa_a = kappa_a
        self.optimizer = torch.optim.Adam(
            network.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.n_epochs = n_epochs
    
    def smooth_l1(self, x, eps=1e-8):
        """Smooth approximation to L1 norm for differentiability."""
        return torch.sum(torch.sqrt(x**2 + eps))
    
    def train_fold(self, dataset, train_indices):
        """
        Train the meta-learner on one walk-forward fold.
        
        IMPORTANT: Process days SEQUENTIALLY within each epoch.
        The switching cost creates temporal dependencies.
        
        Args:
            dataset: MetaLearnerDataset instance
            train_indices: list of time indices for training
        
        Returns:
            training_history: dict with losses per epoch
        """
        self.network.train()
        history = {'epoch_loss': [], 'epoch_reward': []}
        
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            epoch_reward = 0.0
            
            # Initialize previous weights and alphas
            w_prev = torch.ones(len(self.algorithms[0].compute_weights(...))) / N  # equal weight
            alpha_prev = torch.ones(len(self.algorithms)) / len(self.algorithms)  # uniform
            
            for t in train_indices:
                # 1. Get inputs
                X_t = dataset.get_input(t)  # shape (29,)
                X_t = torch.tensor(X_t, dtype=torch.float32)
                
                # 2. Forward pass: X_t → α_t
                alpha_t = self.network(X_t)  # shape (K,)
                
                # 3. Get algorithm outputs (frozen, no gradient)
                W_algos = dataset.get_algorithm_outputs(t)  # shape (K, N), numpy
                W_algos = torch.tensor(W_algos, dtype=torch.float32)
                
                # 4. Composite portfolio: w_t = Σ_k α_{t,k} · w_t^(k)
                w_t = torch.matmul(alpha_t, W_algos)  # shape (N,)
                
                # 5. Get realized returns
                r_next = dataset.get_returns(t)  # shape (N,), numpy
                r_next = torch.tensor(r_next, dtype=torch.float32)
                
                # 6. Compute reward components
                portfolio_return = torch.dot(w_t, r_next)
                portfolio_cost = self.kappa * self.smooth_l1(w_t - w_prev)
                algo_cost = self.kappa_a * self.smooth_l1(alpha_t - alpha_prev)
                
                reward = portfolio_return - portfolio_cost - algo_cost
                loss = -reward  # minimize negative reward
                
                # 7. Backward pass and update
                self.optimizer.zero_grad()
                loss.backward()
                
                # Optional: gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # 8. Update previous state (detach from graph!)
                w_prev = w_t.detach()
                alpha_prev = alpha_t.detach()
                
                epoch_loss += loss.item()
                epoch_reward += reward.item()
            
            history['epoch_loss'].append(epoch_loss / len(train_indices))
            history['epoch_reward'].append(epoch_reward / len(train_indices))
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.n_epochs}: "
                      f"avg_reward={epoch_reward/len(train_indices):.6f}")
        
        return history
```

### Important Implementation Notes

**1. Sequential processing, not batched:** Because the switching cost depends on w_{t-1} and α_{t-1}, we MUST process days in chronological order. Each day's gradient depends on the previous day's output. Standard shuffled mini-batching would break the temporal dependency.

**2. Detaching previous states:** After each step, w_prev and alpha_prev must be detached from the computation graph. Otherwise the gradient would flow back through all previous time steps (like BPTT), which is not what we want — we want to optimize each day's decision given the previous day's outcome.

**3. Algorithm outputs are constants:** The w_t^(k) values come from frozen algorithms and should NOT have gradients. Wrap them in `torch.tensor(..., requires_grad=False)` or `.detach()`.

**4. Warm-up period:** The first ~60 days of each fold cannot be used (insufficient lookback for momentum_60d). Skip these in train_indices.

**5. Gradient clipping:** With K≈81 algorithms, the softmax output is high-dimensional. Gradient clipping prevents explosion in early training.

---

## Step 4: Inference (Decision Time)

**File:** `meta_learner/inference.py`

```python
class MetaLearnerAgent:
    """
    At decision time, the meta-learner is the only active component.
    Everything else (algorithms, regime) is given.
    
    This class wraps the trained network for use in the backtesting engine.
    """
    
    def __init__(self, network, algorithms, scaler=None):
        """
        Args:
            network: trained MetaLearnerNetwork (in eval mode)
            algorithms: list of K frozen PortfolioAlgorithm instances
            scaler: fitted StandardScaler for input normalization
        """
        self.network = network
        self.network.eval()
        self.algorithms = algorithms
        self.scaler = scaler
    
    def select(self, X_t, algorithm_outputs):
        """
        Given the current context and algorithm outputs, compute the 
        composite portfolio.
        
        Args:
            X_t: input vector (asset features + regime one-hot), shape (29,)
            algorithm_outputs: matrix of algo weights, shape (K, N)
        
        Returns:
            w_t: composite portfolio weights, shape (N,)
            alpha_t: mixing weights, shape (K,)
        """
        with torch.no_grad():
            if self.scaler is not None:
                X_t = self.scaler.transform(X_t.reshape(1, -1)).flatten()
            
            X_t = torch.tensor(X_t, dtype=torch.float32)
            alpha_t = self.network(X_t)  # shape (K,)
            
            W = torch.tensor(algorithm_outputs, dtype=torch.float32)
            w_t = torch.matmul(alpha_t, W)  # shape (N,)
            
            return w_t.numpy(), alpha_t.numpy()
```

---

## Step 5: Walk-Forward Experiment

**File:** `experiments/plan5_meta_learner.py`

### Experiment Structure

For each of the 12 walk-forward folds:

```
1. SPLIT data into train and test (expanding window, test = 1 year)

2. STAGE 0 — Pre-compute algorithm outputs
   - For ALL time steps (train + test), compute w_t^(k) for all K algorithms
   - Tier 1: compute from rolling statistics (no training needed)
   - Tier 2: re-train on training block, then compute outputs for all time steps
   - Store as matrix: algo_outputs[t, k, :] = w_t^(k) ∈ R^N

3. PREPARE dataset
   - Compute asset features for all time steps
   - Get oracle regime labels s*_t for all time steps
   - Fit StandardScaler on training features only, transform both train and test
   - Create MetaLearnerDataset

4. TRAIN meta-learner (Stage 2)
   - Initialize MetaLearnerNetwork(input_dim=29, n_algorithms=K)
   - Train via MetaLearnerTrainer.train_fold(dataset, train_indices)
   - Record training history (loss curve)

5. EVALUATE on test year
   - Run MetaLearnerAgent on test data
   - Record: w_t, α_t, portfolio returns, switching costs
   - Compute all metrics (Sharpe, Sortino, MaxDD, Turnover, etc.)
   - Compute Algorithm Entropy H(α_t) per day

6. BASELINES (for comparison within this fold)
   - Equal Weight (Buy & Rebalance)
   - Reflex Agent (from Plan 4, using oracle regime)
   - Best individual algorithm (oracle: best in-sample Sharpe)
```

### Configuration for First Run

```python
CONFIG_PLAN5 = {
    # Meta-learner architecture
    'input_dim': 29,           # 25 asset features + 4 regime one-hot
    'hidden_dims': [128, 64],
    'dropout': 0.2,
    'activation': 'relu',
    
    # Training
    'n_epochs': 50,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'gradient_clip': 1.0,
    
    # Costs
    'kappa': 0.001,            # Portfolio switching cost (same as Plan 4)
    'kappa_a': 0.0,            # Algorithm switching cost (OFF for first run)
    
    # Data
    'asset_features': ['return_1d', 'return_5d', 'return_20d', 
                       'rolling_vol_20d', 'momentum_60d'],
    'regime_encoding': 'onehot',  # 'onehot' or 'scalar'
    'use_oracle_regime': True,
    
    # Walk-forward
    'n_folds': 12,
    'test_years': list(range(2013, 2025)),  # 2013–2024
}
```

### Outputs

For each fold, save:
- `alpha_t` time series (K-dimensional, test period)
- `w_t` time series (N-dimensional, test period)
- Portfolio returns and net returns
- All metrics

Across folds, compute:
- Average Sharpe, Sortino, MaxDD, etc.
- Average Algorithm Entropy
- Oracle Gap (oracle regime meta-learner vs. what we'd get with real classifier — this is for LATER)

---

## Step 6: Analysis and Visualization

### Plot 1: Training Convergence
- For each fold: average reward per epoch
- Shows whether 50 epochs is enough, or if the model needs more/less

### Plot 2: Meta-Learner vs. Baselines (Walk-Forward)
- Bar chart: Sharpe Ratio per fold for:
  - Equal Weight
  - Reflex Agent (oracle)
  - Meta-Learner (oracle)
  - Best Individual Algorithm (oracle)
- This is THE key result: does the Meta-Learner beat the Reflex Agent?

### Plot 3: Strategy Mixing Weights Over Time (α_t)
- Stacked area chart for 1-2 selected test years
- Shows which algorithms the meta-learner trusts over time
- Overlay: VIX level and regime labels
- This is the qualitative evidence for RQ1: "Does the meta-learner select different algorithms under different regimes?"

### Plot 4: Algorithm Entropy Over Time
- H(α_t) per day, overlaid with VIX and regime boundaries
- Low entropy = confident selection, high entropy = uncertain
- Expected: entropy decreases during stable regimes, spikes during transitions

### Plot 5: Regime-Conditional Selection Pattern
- Heatmap: for each regime, show the average α_t distribution across algorithms
- Group by algorithm family (F1–F10)
- This answers: "Under each regime, which algorithm families does the meta-learner prefer?"

### Plot 6: Summary Table
- CSV with all metrics across all 12 folds
- Columns: Fold, EW_Sharpe, Reflex_Sharpe, ML_Sharpe, BestAlgo_Sharpe, ML_Entropy, ML_Turnover

---

## Step 7: Sanity Checks

Before trusting the results, verify:

### Check 1: Does the meta-learner learn anything?
- Compare to a RANDOM meta-learner (α_t ~ Uniform) across folds
- If the trained meta-learner's Sharpe ≈ random, something is wrong

### Check 2: Does training converge?
- Check training loss curves — if flat from epoch 1, the learning rate is too low or the signal is too weak
- If diverging, reduce learning rate

### Check 3: Are the weights reasonable?
- Check w_t values: are they all ≈ 0.2 (= Equal Weight)? That means the meta-learner learned nothing
- Check α_t values: are they all ≈ 1/K (uniform)? Same problem
- Check if a few algorithms dominate: that's expected and good

### Check 4: Is the composite portfolio valid?
- Verify: sum(w_t) ≈ 1.0 and all w_t,i ≥ 0 for all t
- This should be guaranteed by the convex combination, but verify numerically

### Check 5: Switching cost impact
- Run once with kappa=0 and once with kappa=0.001
- The kappa=0 version should have higher gross returns but much higher turnover
- If turnover is the same with and without costs, the switching cost term has no effect

---

## Expected Outcomes

### Optimistic scenario
The meta-learner beats the Reflex Agent by learning soft mixing instead of hard selection. It shows different mixing patterns across regimes and achieves a higher Sharpe Ratio, especially in transition periods.

### Realistic scenario
The meta-learner performs comparably to the Reflex Agent in most folds but shows smoother transitions and lower turnover. The mixing weights reveal that a few algorithm families dominate, with the distribution shifting across regimes.

### Pessimistic scenario
The meta-learner converges to near-Equal-Weight (α_t ≈ 1/K for all t) because the signal-to-noise ratio is too low for the network to learn meaningful regime-dependent selection with only ~1750–4500 training days. If this happens:
- Try reducing K (e.g., only Tier 1, K=48, or only the top-10 algorithms per regime from Plan 4)
- Try simpler architecture (single hidden layer)
- Try longer training (more epochs)
- Try higher learning rate

---

## Execution Instructions

```bash
cd regime_algo_selection

# Run the full Plan 5 experiment
python -m experiments.plan5_meta_learner
```

The script should:
1. Load all data (cached from previous plans)
2. Initialize all K≈81 algorithms
3. Run 12-fold walk-forward with meta-learner training + evaluation
4. Generate all plots and save to results/
5. Print summary table to console

Expected runtime: ~30-60 minutes (12 folds × 50 epochs × ~4000 training days per fold).

---

## Connection to Research Questions

| RQ | What This Plan Tests |
|----|---------------------|
| RQ1: Algorithm Selection under Regimes | Plot 3 + 5: Does α_t change across regimes? |
| RQ2: Effect of Switching Costs | Check 5: κ=0 vs κ>0 comparison |
| RQ3: Algorithm Space Size | Not directly — but meta-learner over K=81 vs. Reflex Agent over K=81 is the first comparison |
| RQ4: Regime Estimation Quality | Not yet — oracle only. Real classifier comes in Plan 6 |
| RQ5: Meta-Learner vs. Direct Learning | Not yet — direct policy learning comparison is a later plan |

---

## What Comes Next (After Plan 5)

1. **Plan 6:** Swap oracle regime for real classifier → measure Oracle Gap
2. **Hyperparameter tuning:** hidden dims, dropout, lr, n_epochs
3. **Input composition ablation:** add algorithm outputs w_t^(k) to input
4. **Flat vs. hierarchical softmax** comparison
5. **Algorithm switching cost** (κ_a > 0)
6. **Pairwise decomposition** (Prof. Ler's suggestion)
