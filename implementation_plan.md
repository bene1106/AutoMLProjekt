# Implementation Plan: Regime-Aware Reflex Agent Baseline

## Context & Goal

This project implements a **Regime-Aware Cost-Constrained Algorithm Selection** system for multi-asset allocation. The formal problem definition is in `ProblemDef4_5_hybrid.pdf`.

**This implementation plan covers the FIRST MILESTONE**: a Reflex Agent baseline using only Tier 1 heuristic algorithms + a regime classifier. The professor wants this for next week's meeting.

The deliverable: a complete, runnable system that:
1. Classifies market regimes from lagged VIX features
2. Uses a Reflex Agent (lookup table) to map regimes → best algorithm
3. Evaluates performance with proper metrics
4. Presents results clearly (tables, plots)

---

## Project Structure

```
regime_algo_selection/
├── config.py                  # All hyperparameters, constants, paths
├── data/
│   ├── loader.py              # Download & cache price data + VIX
│   └── features.py            # Feature engineering (asset features + VIX features)
├── regimes/
│   ├── ground_truth.py        # VIX → regime label mapping (ground truth)
│   └── classifier.py          # Regime classifier (Logistic Regression, etc.)
├── algorithms/
│   ├── base.py                # Abstract base class for all algorithms
│   └── tier1_heuristics.py    # All Tier 1 algorithms (EW, MinVar, Momentum, etc.)
├── agents/
│   ├── reflex_agent.py        # Reflex agent: regime → fixed algorithm mapping
│   └── oracle_agent.py        # Oracle agent: uses true regime (upper bound)
├── evaluation/
│   ├── metrics.py             # All evaluation metrics
│   ├── backtest.py            # Backtesting engine
│   └── visualization.py       # Plots and tables
├── main.py                    # Main entry point: runs everything
└── results/                   # Output directory for plots and tables
```

---

## Step 1: Configuration (`config.py`)

Define all constants in one place:

```python
# === Asset Universe ===
ASSETS = ["SPY", "TLT", "GLD", "EFA", "VNQ"]  # 5 ETFs covering equity, bonds, gold, intl, real estate
N_ASSETS = len(ASSETS)

# === Time ===
START_DATE = "2006-01-01"   # VIX + ETF data available from here
END_DATE = "2024-12-31"
TRAIN_END = "2018-12-31"    # Training: 2006-2018
VAL_END = "2020-12-31"      # Validation: 2019-2020
# Test: 2021-2024

# === Regime Thresholds (VIX-based) ===
REGIME_THRESHOLDS = [15, 20, 30]  # → 4 regimes: <=15, 15-20, 20-30, >30
REGIME_NAMES = {1: "Calm", 2: "Normal", 3: "Tense", 4: "Crisis"}
N_REGIMES = 4

# === Switching Cost ===
KAPPA = 0.001  # Portfolio-level switching cost coefficient (start conservative)

# === Tier 1 Algorithm Hyperparameter Grids ===
LOOKBACKS_COV = [20, 40, 60, 120, 252]       # For covariance-based methods
LOOKBACKS_MOM = [5, 10, 20, 60, 120]          # For momentum
LOOKBACKS_TREND = [5, 10, 20, 60]             # For trend-following
TREND_BETAS = [1, 2, 3]                        # Concentration parameter
RISK_AVERSIONS = [0.5, 1, 2, 5]               # For mean-variance
```

---

## Step 2: Data Pipeline (`data/loader.py` and `data/features.py`)

### `data/loader.py`

```
Function: load_data() -> dict
- Download daily adjusted close prices for all assets in ASSETS using yfinance
- Download VIX (ticker: "^VIX") daily close
- Cache data locally as CSV to avoid re-downloading
- Return dict: {"prices": pd.DataFrame, "vix": pd.Series}
  - prices: index=date, columns=asset tickers, values=adjusted close
  - vix: index=date, values=VIX close
- Handle missing data: forward-fill, then drop any remaining NaNs
- Align all series to common dates
```

### `data/features.py`

```
Function: compute_asset_features(prices: pd.DataFrame) -> pd.DataFrame
- For each asset, compute:
  - Lagged returns: ret_1d, ret_5d, ret_20d, ret_60d
  - Rolling volatility: vol_20d, vol_60d (annualized std of daily returns)
  - Momentum score: cumulative return over [20d, 60d, 120d]
- Return: MultiIndex DataFrame (date × asset × feature)

Function: compute_vix_features(vix: pd.Series) -> pd.DataFrame
- These are the lagged features z_t used for regime estimation
- Compute:
  - vix_prev: v_{t-1} (yesterday's VIX close)
  - vix_change_1d: v_{t-1} - v_{t-2}
  - vix_change_5d: v_{t-1} - v_{t-6}
  - vix_ma5: 5-day moving average of VIX (lagged by 1)
  - vix_ma20: 20-day moving average of VIX (lagged by 1)
  - vix_std20: 20-day rolling std of VIX (lagged by 1)
  - vix_relative: v_{t-1} / vix_ma20 (VIX relative to its recent average)
- CRITICAL: All features must be LAGGED — only use information available at decision time t
- Return: DataFrame with index=date, columns=feature names

Function: compute_returns(prices: pd.DataFrame) -> pd.DataFrame
- Simple daily returns: r_{t+1,i} = (P_{t+1,i} - P_{t,i}) / P_{t,i}
- Return: DataFrame, same shape as prices but shifted (return at date t is the return FROM t TO t+1)
```

---

## Step 3: Regime Ground Truth (`regimes/ground_truth.py`)

```
Function: compute_regime_labels(vix: pd.Series) -> pd.Series
- Apply the deterministic mapping g(v_t):
  - v_t <= 15       → regime 1 (Calm)
  - 15 < v_t <= 20  → regime 2 (Normal)
  - 20 < v_t <= 30  → regime 3 (Tense)
  - v_t > 30        → regime 4 (Crisis)
- Return: pd.Series with index=date, values=regime label (1-4)
- This is s*_t — the TRUE regime, known only after market close on day t

Function: compute_lagged_regime(regime_labels: pd.Series) -> pd.Series
- Return s*_{t-1}: yesterday's true regime
- This is the "naive baseline" for regime estimation
```

---

## Step 4: Regime Classifier (`regimes/classifier.py`)

```
Class: RegimeClassifier
- __init__(self, model_type="logistic_regression")
  - model_type options: "logistic_regression", "random_forest", "xgboost"
  - Start with logistic_regression (simplest)

- fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> self
  - X_train: VIX features z_t (from compute_vix_features)
  - y_train: true regime labels s*_t (from compute_regime_labels)
  - IMPORTANT: y_train uses the regime label for the SAME day t
    (because at training time, we know the true regime)
  - Train a standard classifier (sklearn)
  - Store the trained model

- predict_proba(self, X: pd.DataFrame) -> pd.DataFrame
  - Return p_hat_t: probability distribution over 4 regimes
  - Columns: ["prob_regime_1", "prob_regime_2", "prob_regime_3", "prob_regime_4"]
  
- predict(self, X: pd.DataFrame) -> pd.Series
  - Return s_hat_t = argmax of predict_proba
  
- evaluate(self, X_test, y_test) -> dict
  - Return: accuracy, classification_report, confusion_matrix
```

---

## Step 5: Tier 1 Algorithms (`algorithms/base.py` and `algorithms/tier1_heuristics.py`)

### `algorithms/base.py`

```python
from abc import ABC, abstractmethod

class PortfolioAlgorithm(ABC):
    """Base class for all portfolio construction algorithms."""
    
    def __init__(self, name: str, family: str, hyperparams: dict):
        self.name = name          # e.g., "MinVar_L60"
        self.family = family      # e.g., "MinimumVariance"
        self.hyperparams = hyperparams
    
    @abstractmethod
    def compute_weights(self, prices_history: pd.DataFrame, features: dict) -> np.ndarray:
        """
        Given historical prices up to time t and features, 
        return portfolio weights w_t in Delta_N (non-negative, sum to 1).
        
        Args:
            prices_history: price DataFrame up to and including day t-1
            features: dict with any additional info needed
        
        Returns:
            np.ndarray of shape (N,) — portfolio weights
        """
        pass
```

### `algorithms/tier1_heuristics.py`

Implement each strategy family. Each concrete instance is one algorithm A_k.

**Family F1: Equal Weight**
```
Class: EqualWeight(PortfolioAlgorithm)
- No hyperparameters
- compute_weights: return np.ones(N) / N
- Only 1 configuration
```

**Family F2: Minimum Variance**
```
Class: MinimumVariance(PortfolioAlgorithm)
- Hyperparams: lookback L_sigma
- compute_weights:
  1. Compute rolling covariance matrix from last L_sigma daily returns
  2. Solve: min w^T Σ w  subject to  w >= 0, sum(w) = 1
  3. Use scipy.optimize.minimize with constraints
  4. If optimization fails, fall back to equal weight
- Configurations: L_sigma in [20, 40, 60, 120, 252] → 5 algorithms
```

**Family F3: Risk Parity**
```
Class: RiskParity(PortfolioAlgorithm)
- Hyperparams: lookback L_sigma
- compute_weights:
  1. Compute rolling volatility for each asset over L_sigma days
  2. w_i = (1/sigma_i) / sum(1/sigma_j)  (inverse volatility weighting)
  3. Normalize to sum to 1
- Configurations: L_sigma in [20, 60, 120, 252] → 4 algorithms
```

**Family F4: Maximum Diversification**
```
Class: MaxDiversification(PortfolioAlgorithm)
- Hyperparams: lookback L_sigma
- compute_weights:
  1. Compute correlation matrix and volatilities over L_sigma days
  2. Maximize diversification ratio: (w^T σ) / sqrt(w^T Σ w)
  3. Use scipy.optimize.minimize (negate for minimization)
- Configurations: L_sigma in [20, 60, 120, 252] → 4 algorithms
```

**Family F5: Momentum**
```
Class: Momentum(PortfolioAlgorithm)
- Hyperparams: lookback L_m, weighting_scheme ("linear" or "exp")
- compute_weights:
  1. Compute momentum score for each asset: cumulative return over last L_m days
  2. If linear: score_i = cum_return_i (raw momentum)
  3. If exp: score_i = ewm return over L_m
  4. Convert scores to weights via softmax: w_i = exp(score_i) / sum(exp(score_j))
  5. Ensure non-negative (softmax guarantees this)
- Configurations: L_m in [5, 10, 20, 60, 120] × weighting in ["linear", "exp"] → 10 algorithms
```

**Family F6: Trend-Following**
```
Class: TrendFollowing(PortfolioAlgorithm)
- Hyperparams: lookback L_m, concentration beta
- compute_weights:
  1. For each asset, compute trend signal: sign of L_m-day return (or SMA crossover)
     - trend_i = 1 if price > SMA(L_m), else 0  (binary trend signal)
  2. Score = trend_i (binary: in trend or not)
  3. Apply concentration: raw_w_i = score_i^beta
  4. Normalize: w_i = raw_w_i / sum(raw_w_j)
  5. If all signals are 0 (no asset trending), fall back to equal weight
- Configurations: L_m in [5, 10, 20, 60] × beta in [1, 2, 3] → 12 algorithms
```

**Family F7: Mean-Variance**
```
Class: MeanVariance(PortfolioAlgorithm)
- Hyperparams: lookback L_sigma, risk_aversion gamma
- compute_weights:
  1. Estimate expected returns: mu = rolling mean of daily returns over L_sigma
  2. Estimate covariance: Sigma = rolling cov over L_sigma
  3. Solve: max w^T mu - (gamma/2) w^T Sigma w  s.t. w >= 0, sum(w) = 1
  4. Use scipy.optimize.minimize
- Configurations: L_sigma in [20, 60, 120] × gamma in [0.5, 1, 2, 5] → 12 algorithms
```

### Algorithm Registry

```
Function: build_tier1_algorithm_space() -> list[PortfolioAlgorithm]
- Instantiate ALL Tier 1 algorithm configurations
- Return a list of PortfolioAlgorithm objects
- Print summary: "Built K={len(algorithms)} Tier 1 algorithms across 7 families"
- Expected: ~48 algorithms total (1 + 5 + 4 + 4 + 10 + 12 + 12)
```

---

## Step 6: Backtesting Engine (`evaluation/backtest.py`)

This is the core simulation loop.

```
Class: Backtester
- __init__(self, algorithms, regime_classifier, agent, returns, vix_features, 
           regime_labels, kappa=0.001)

- run(self, start_date, end_date) -> BacktestResult
  Core loop — for each trading day t in [start_date, end_date]:
  
  1. Get VIX features z_t (all lagged, available at decision time)
  2. Regime classifier predicts: p_hat_t = classifier.predict_proba(z_t)
  3. Point estimate: s_hat_t = argmax(p_hat_t)
  4. Agent selects action based on s_hat_t:
     - Reflex agent: look up pre-assigned algorithm for this regime
     - Oracle agent: use true regime s*_t to look up algorithm
  5. Selected algorithm(s) compute weights: w_t = A_k.compute_weights(...)
  6. Compute portfolio return: R_t = w_t^T @ r_{t→t+1}
  7. Compute switching cost: C_t = kappa * ||w_t - w_{t-1}||_1
  8. Compute net return: R_net_t = R_t - C_t
  9. Store: w_t, alpha_t (which algorithm), R_t, C_t, s_hat_t, s*_t
  
  Return a BacktestResult object containing all stored time series

Class: BacktestResult
- portfolio_returns: pd.Series (gross returns)
- net_returns: pd.Series (after switching costs)  
- weights_history: pd.DataFrame (N assets × T dates)
- algorithm_selections: pd.Series (which algorithm was selected each day)
- regime_predictions: pd.Series
- regime_true: pd.Series
- switching_costs: pd.Series
```

---

## Step 7: Reflex Agent (`agents/reflex_agent.py`)

```
Class: ReflexAgent
- The reflex agent is a simple lookup table: regime → algorithm

- fit(self, algorithms, returns, regime_labels, prices_history)
  For each regime r in {1, 2, 3, 4}:
    For each algorithm A_k:
      Backtest A_k on TRAINING DATA restricted to days where s*_t == r
      Compute average daily return (or Sharpe ratio) of A_k in regime r
    Assign: best_algo[r] = argmax over A_k of performance in regime r
  
  Store the mapping: {1: algo_X, 2: algo_Y, 3: algo_Z, 4: algo_W}
  Print the mapping for interpretability

- select(self, regime_estimate: int) -> PortfolioAlgorithm
  Return best_algo[regime_estimate]
```

```
Class: OracleAgent(ReflexAgent)
- Same as ReflexAgent but:
  - At decision time, uses the TRUE regime s*_t instead of predicted s_hat_t
  - This is the upper bound — what you'd get with perfect regime knowledge
```

---

## Step 8: Evaluation Metrics (`evaluation/metrics.py`)

```
Function: compute_all_metrics(result: BacktestResult) -> dict
- cumulative_return: total wealth growth (product of 1 + r_t)
- annualized_return: geometric mean annualized
- annualized_volatility: std of daily returns * sqrt(252)
- sharpe_ratio: annualized_return / annualized_volatility (assume rf=0 for simplicity)
- sortino_ratio: annualized_return / downside_deviation
- max_drawdown: worst peak-to-trough decline
- total_turnover: sum of ||w_t - w_{t-1}||_1 over all t
- avg_daily_turnover: total_turnover / T
- total_switching_cost: sum of all C_t
- regime_accuracy: accuracy of s_hat_t vs s*_t
- oracle_gap: sharpe(oracle_agent) - sharpe(reflex_agent)
  → "how much performance is lost due to imperfect regime prediction"

All metrics should be returned as a clean dict that can be displayed as a table.
```

---

## Step 9: Visualization (`evaluation/visualization.py`)

Generate the following plots:

```
1. Cumulative Wealth Curves
   - Plot cumulative wealth for: Equal Weight, Best Single Algorithm, Reflex Agent, Oracle Agent
   - X-axis: date, Y-axis: cumulative wealth (starting at $1)
   - Include vertical shading by regime (color-coded)

2. Regime Classification Over Time
   - Top panel: VIX level over time with threshold lines at 15, 20, 30
   - Bottom panel: True regime vs predicted regime (colored bars)

3. Algorithm Selection Over Time
   - Stacked area plot showing which algorithm the reflex agent selects
   - Color-coded by algorithm family

4. Per-Regime Performance Table
   - Table: rows = regimes, columns = [# days, avg return, volatility, best algo, sharpe]
   - Shows which algorithm dominates in each regime

5. Regime Classifier Confusion Matrix
   - Standard confusion matrix heatmap

6. Metrics Summary Table
   - Compare all strategies side by side: EW, Buy&Hold, Best Single Algo, Reflex Agent, Oracle Agent
```

---

## Step 10: Main Script (`main.py`)

```python
"""
Main entry point: Regime-Aware Reflex Agent Baseline
"""

def main():
    # 1. Load data
    data = load_data()
    
    # 2. Compute features
    returns = compute_returns(data["prices"])
    asset_features = compute_asset_features(data["prices"])
    vix_features = compute_vix_features(data["vix"])
    regime_labels = compute_regime_labels(data["vix"])
    
    # 3. Train/test split (temporal)
    train_mask = returns.index <= TRAIN_END
    val_mask = (returns.index > TRAIN_END) & (returns.index <= VAL_END)
    test_mask = returns.index > VAL_END
    
    # 4. Build Tier 1 algorithm space
    algorithms = build_tier1_algorithm_space()
    print(f"Algorithm space: K = {len(algorithms)}")
    
    # 5. Train regime classifier
    classifier = RegimeClassifier("logistic_regression")
    classifier.fit(vix_features[train_mask], regime_labels[train_mask])
    classifier_eval = classifier.evaluate(vix_features[test_mask], regime_labels[test_mask])
    print(f"Regime classification accuracy (test): {classifier_eval['accuracy']:.3f}")
    
    # 6. Fit reflex agent (find best algo per regime on training data)
    reflex = ReflexAgent()
    reflex.fit(algorithms, returns[train_mask], regime_labels[train_mask], data["prices"])
    print(f"Reflex agent mapping: {reflex.mapping}")
    
    # 7. Backtest on TEST set
    backtester = Backtester(algorithms, classifier, reflex, returns, 
                            vix_features, regime_labels, kappa=KAPPA)
    result_reflex = backtester.run(test_start, test_end)
    
    # Also run oracle agent and equal-weight baseline
    oracle = OracleAgent()
    oracle.fit(algorithms, returns[train_mask], regime_labels[train_mask], data["prices"])
    result_oracle = backtester.run(test_start, test_end, agent=oracle)
    result_ew = backtester.run(test_start, test_end, agent=EqualWeightAgent())
    
    # 8. Compute metrics
    metrics = {
        "Equal Weight": compute_all_metrics(result_ew),
        "Reflex Agent": compute_all_metrics(result_reflex),
        "Oracle Agent": compute_all_metrics(result_oracle),
    }
    
    # 9. Generate visualizations
    plot_cumulative_wealth([result_ew, result_reflex, result_oracle])
    plot_regime_classification(vix_features, regime_labels, classifier)
    plot_algorithm_selection(result_reflex)
    print_metrics_table(metrics)
    
    # 10. Print analysis summary
    print("\n=== ANALYSIS ===")
    print(f"Oracle Gap (Sharpe): {metrics['Oracle Agent']['sharpe'] - metrics['Reflex Agent']['sharpe']:.4f}")
    print(f"Reflex vs EW (Sharpe): {metrics['Reflex Agent']['sharpe'] - metrics['Equal Weight']['sharpe']:.4f}")

if __name__ == "__main__":
    main()
```

---

## Key Implementation Notes for Claude Code

### Data Leakage Prevention
- **ALL features must be lagged by at least 1 day**. At decision time t, we only know information up to t-1.
- The regime label s*_t is defined by VIX close on day t — NOT available at decision time.
- Train/test split is TEMPORAL, never random.

### Numerical Stability
- Covariance matrices can be singular for short lookbacks — add a small ridge (1e-6 * I) before inverting.
- Softmax can overflow — use the log-sum-exp trick.
- Some algorithms may produce NaN weights (e.g., MinVar with singular covariance) — always validate and fall back to equal weight.

### Dependencies
```
pip install yfinance pandas numpy scipy scikit-learn matplotlib seaborn
```

### Output
- Save all plots to `results/` directory as PNG
- Print metrics tables to console AND save as CSV
- The professor wants: "Present actual setup, interpret the results, analyse, propose modifications"

---

## What Comes AFTER This Baseline

Once the Reflex Agent baseline works:
1. **Replace Reflex Agent with a learned Meta-Learner** (neural net with softmax over K algorithms)
2. **Add Tier 2 algorithms** (Ridge, Lasso, Elastic Net) → requires Stage 0 pre-training
3. **Add Tier 3 algorithms** (RF, GBM, NN)
4. **Compare**: Reflex Agent vs Meta-Learner vs Direct Policy Learning
5. **Run full experiment suite** from ProblemDef4.5 Section 16
