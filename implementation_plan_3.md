# Implementation Plan 3 — Extended Data & Walk-Forward Validation

## Context

Plans 1 and 2 are complete. Key problem identified: all results are based on a single test window (2021–2024), which was a low-volatility bull market. We don't know if the findings are systematic or specific to this period. Additionally, the data starts in 2006, missing the Dot-Com crash, 9/11, and the 2003–2005 recovery.

This plan fixes both issues before we move to Tier 2/3 and the Meta-Learner.

---

## Step 1: Extend Data History to 2000

### Problem

The current asset universe (SPY, TLT, GLD, EFA, VNQ) starts in 2006 because:
- GLD launched November 2004
- VNQ launched September 2004
- TLT launched July 2002

This excludes: the Dot-Com crash (2000–2002), the 9/11 shock (2001), and the recovery (2003–2005) — all important for learning robust regime-algorithm mappings.

### Solution: Staggered Asset Universe

Instead of dropping assets or finding proxies, use a staggered approach where the number of assets grows over time as ETFs become available.

**File:** `config.py` (modify) and `data/loader.py` (modify)

```
New configuration:

ASSET_PHASES = [
    {
        "start": "2000-01-01",
        "assets": ["SPY"],           # SPY available since 1993
        "n_assets": 1
    },
    {
        "start": "2002-08-01",
        "assets": ["SPY", "TLT"],    # TLT launched Jul 2002
        "n_assets": 2
    },
    {
        "start": "2004-12-01",
        "assets": ["SPY", "TLT", "GLD", "EFA", "VNQ"],  # All 5 available
        "n_assets": 5
    }
]

# VIX is available from 1990, so regime labels work for the entire period
START_DATE = "2000-01-01"
END_DATE = "2024-12-31"
```

**IMPORTANT DESIGN DECISION:** A staggered universe adds significant complexity to the algorithms (covariance matrices change size, weight vectors change dimension). A simpler alternative:

```
ALTERNATIVE (recommended): Start with 2 assets from 2000

ASSETS_EXTENDED = ["SPY", "TLT"]           # Available from 2002-08
ASSETS_FULL = ["SPY", "TLT", "GLD", "EFA", "VNQ"]  # Available from 2004-12

Option A — "2-Asset Long History":
  - Use only SPY + TLT from 2002-08 to 2024-12
  - Pro: Simple, no dimension changes, covers Dot-Com tail + Iraq War
  - Con: Only 2 assets, less diversification to test

Option B — "5-Asset from 2005":
  - Use all 5 ETFs from 2005-01 to 2024-12 (one year earlier than current)
  - Pro: Full asset universe, gains one extra year
  - Con: Still misses Dot-Com crash

Option C — "Staggered" (most complete):
  - 2 assets 2002–2004, then 5 assets 2005–2024
  - Pro: Maximum data coverage
  - Con: Algorithms must handle changing N, more complex implementation
  
RECOMMENDATION: Implement Option C but with a clean separation:
  - Phase 1 (2002-08 to 2004-11): SPY + TLT only, N=2
  - Phase 2 (2004-12 to 2024-12): All 5 ETFs, N=5
  - Walk-forward windows that span the boundary use Phase 2 data only
  - Phase 1 is used ONLY for additional training data (regime classifier benefits 
    from seeing the 2002-2003 crisis, even if only 2 assets were tradeable)
```

### Implementation Details

**File:** `data/loader.py` — modify `load_data()`

```
Function: load_data_extended() -> dict
- Download SPY from 2000-01-01
- Download TLT from 2002-08-01
- Download GLD, EFA, VNQ from 2004-12-01
- Download VIX from 2000-01-01
- Cache all to CSV
- Return dict with:
  - "prices_phase1": DataFrame (SPY, TLT) from 2002-08 to 2004-11
  - "prices_phase2": DataFrame (all 5 ETFs) from 2004-12 to 2024-12
  - "prices_full": DataFrame with NaN for missing assets before their launch
  - "vix": Series from 2000-01-01 to 2024-12-31
- Validate: check that all ETFs have data from their expected start dates
```

**File:** `data/features.py` — modify feature functions

```
- compute_vix_features(): Extend to work from 2000-01-01
  (VIX features don't depend on assets, so no changes needed)
  
- compute_returns(): Handle NaN columns for assets not yet available
  
- compute_asset_features(): Handle variable number of assets
  Return NaN for assets not yet launched

- compute_cross_asset_features(): Only compute for periods where 
  all referenced assets exist (e.g., SPY-TLT correlation needs both)
```

**File:** `regimes/ground_truth.py` — no changes needed
(VIX-based regime labels work for the full period)

---

## Step 2: Rebuild Tier-1 Algorithms for Variable N

**File:** `algorithms/tier1_heuristics.py` — modify

The algorithms must handle N=2 (Phase 1) and N=5 (Phase 2) cleanly.

```
Changes needed:
- All algorithms already take prices_history as input and infer N from it
- Verify: covariance estimation, optimization constraints work for N=2
- MinVar, MaxDiv, MeanVar: scipy.optimize constraints must adapt to N
- Risk Parity, Momentum, Trend-Following: should work automatically
- Equal Weight: trivially works for any N

Test: Run each algorithm on a 2-asset subset (SPY, TLT) and verify 
outputs are valid weight vectors in Delta_2.
```

---

## Step 3: Walk-Forward Validation Framework

This is the core of Plan 3.

**File:** `evaluation/walk_forward.py` (new file)

### Design

```
Walk-Forward Parameters:
  - TRAIN_WINDOW = 8 years (training period length)
  - TEST_WINDOW = 1 year (test period length)
  - STEP = 1 year (how much to advance each fold)
  - MIN_START = "2005-01-01" (earliest test start — ensures 5 assets available)
  
This generates approximately 12 folds:
  
  Fold  | Training Period  | Test Period  | Market Character
  ------|------------------|--------------|------------------
  1     | 2005–2012        | 2013         | Post-crisis recovery
  2     | 2006–2013        | 2014         | Calm bull market
  3     | 2007–2014        | 2015         | China crash, VIX spike
  4     | 2008–2015        | 2016         | Low vol, post-oil-crash
  5     | 2009–2016        | 2017         | Extremely calm, low VIX
  6     | 2010–2017        | 2018         | Vol spike (Feb), trade war
  7     | 2011–2018        | 2019         | Strong equity rally
  8     | 2012–2019        | 2020         | COVID crash + recovery
  9     | 2013–2020        | 2021         | Post-COVID bull
  10    | 2014–2021        | 2022         | Rate hikes, bear market
  11    | 2015–2022        | 2023         | AI rally, recovery
  12    | 2016–2023        | 2024         | Continued bull
  
Note: Regime classifier can use VIX data from 2000 onward for all folds
(additional training data from Phase 1), but algorithm backtests use only 
Phase 2 data (5 assets from 2005+).
```

### Implementation

```
Class: WalkForwardValidator

- __init__(self, train_years=8, test_years=1, step_years=1, min_test_start="2005-01-01")

- generate_folds(self, data_start, data_end) -> list[dict]
  Returns a list of fold specifications:
  [
    {"fold": 1, "train_start": "2005-01-01", "train_end": "2012-12-31", 
     "test_start": "2013-01-01", "test_end": "2013-12-31"},
    {"fold": 2, ...},
    ...
  ]

- run_fold(self, fold_spec, prices, vix, vix_features, regime_labels, 
           algorithms, kappa) -> FoldResult
  For ONE fold:
  1. Split data according to fold_spec
  2. Train regime classifier on training period
  3. Fit reflex agent (net-Sharpe) on training period
  4. Backtest on test period
  5. Also run: Oracle Agent, Equal Weight, all individual algorithms
  6. Compute all metrics
  7. Return FoldResult with everything

- run_all(self, ...) -> WalkForwardResult
  Run all folds sequentially
  Print progress: "Fold 1/12: Training 2005-2012, Testing 2013..."
  Return aggregated results

Class: FoldResult
- fold_spec: dict (which dates)
- metrics_reflex: dict (Sharpe, return, drawdown, turnover, ...)
- metrics_oracle: dict
- metrics_ew: dict
- algorithm_ranking: DataFrame (all 48 algos ranked for this fold)
- regime_accuracy: float
- reflex_mapping: dict (which algo per regime)
- dominant_regime: str (most common regime in test period)

Class: WalkForwardResult
- folds: list[FoldResult]
- summary_table: DataFrame (one row per fold, all metrics)
- aggregate_metrics: dict (mean, std, median across folds)
```

---

## Step 4: Walk-Forward Analysis & Visualization

**File:** `experiments/walk_forward_analysis.py` (new file)

### Analysis 1: Performance Across All Folds

```
Output Table: Per-Fold Performance Comparison

| Fold | Test Year | Dominant Regime | EW Sharpe | Reflex Sharpe | Oracle Sharpe | Best Algo | Best Sharpe |
|------|-----------|-----------------|-----------|---------------|---------------|-----------|-------------|
| 1    | 2013      | ???             | ???       | ???           | ???           | ???       | ???         |
| 2    | 2014      | ???             | ???       | ???           | ???           | ???       | ???         |
| ...  | ...       | ...             | ...       | ...           | ...           | ...       | ...         |
| 12   | 2024      | ???             | ???       | ???           | ???           | ???       | ???         |
|------|-----------|-----------------|-----------|---------------|---------------|-----------|-------------|
| AVG  | —         | —               | ???       | ???           | ???           | —         | ???         |
| STD  | —         | —               | ???       | ???           | ???           | —         | ???         |

Key questions:
- Does the Reflex Agent beat EW on AVERAGE across all folds?
- In which market conditions (regimes) does it win/lose?
- How large is the Oracle Gap on average?

Save to: results/walk_forward_performance.csv
Plot: results/17_walk_forward_sharpe_comparison.png
  (grouped bar chart: EW vs Reflex vs Oracle per fold)
```

### Analysis 2: Regime-Conditional Performance

```
Group folds by their dominant test-period regime:

| Dominant Regime | # Folds | Avg EW Sharpe | Avg Reflex Sharpe | Reflex Wins? |
|-----------------|---------|---------------|-------------------|--------------|
| Calm            | ???     | ???           | ???               | ???          |
| Normal          | ???     | ???           | ???               | ???          |
| Tense           | ???     | ???           | ???               | ???          |
| Crisis          | ???     | ???           | ???               | ???          |

Key question: Does the Reflex Agent outperform EW specifically during 
volatile/crisis periods (where algorithm selection should matter most)?

Plot: results/18_regime_conditional_performance.png
```

### Analysis 3: Algorithm Stability Across Folds

```
For each fold, record which algorithm the Reflex Agent selects per regime.

Output: Stability Matrix

| Regime  | Fold 1      | Fold 2      | ... | Fold 12     | Most Common     | Stability |
|---------|-------------|-------------|-----|-------------|-----------------|-----------|
| Calm    | ???         | ???         | ... | ???         | ???             | ???%      |
| Normal  | ???         | ???         | ... | ???         | ???             | ???%      |
| Tense   | ???         | ???         | ... | ???         | ???             | ???%      |
| Crisis  | ???         | ???         | ... | ???         | ???             | ???%      |

"Stability" = percentage of folds where the same algorithm is selected.
Low stability means the optimal algorithm per regime changes over time
→ strong argument for a learned meta-learner that can adapt.

Plot: results/19_algorithm_stability.png
  (heatmap: folds × regimes, color = which algorithm family)
```

### Analysis 4: Algorithm Ranking Stability

```
For each algorithm, compute its average rank across all 12 folds.

| Algorithm        | Avg Rank | Std Rank | # Times in Top 5 | # Times Beats EW |
|------------------|----------|----------|-------------------|-------------------|
| RiskParity_L60   | ???      | ???      | ???               | ???               |
| MinVar_L120      | ???      | ???      | ???               | ???               |
| EqualWeight      | ???      | ???      | ???               | ???               |
| ...              | ...      | ...      | ...               | ...               |

Key question: Is there a single algorithm that CONSISTENTLY performs well,
or does the best algorithm change across periods?
If it changes → strong motivation for algorithm selection.
If RiskParity_L60 always wins → maybe just use that, no meta-learner needed.

Plot: results/20_algorithm_rank_stability.png
  (box plot of ranks across folds, per algorithm)
```

### Analysis 5: Oracle Gap Over Time

```
For each fold, compute:
- Oracle Gap = Oracle Sharpe - Reflex Sharpe

Plot over time: Is the Oracle Gap larger during volatile periods?
If yes → better regime prediction matters more when markets are turbulent.
If no → regime prediction quality is uniformly (ir)relevant.

Plot: results/21_oracle_gap_over_time.png
```

---

## Step 5: Updated Baseline Comparison

**File:** `experiments/walk_forward_analysis.py`

After Walk-Forward, recompute the key metrics from Plan 2 but now aggregated over all folds:

```
Aggregated Metrics Table:

| Metric                          | Plan 2 (single window) | Plan 3 (walk-forward avg) |
|---------------------------------|------------------------|---------------------------|
| Regime Classifier Accuracy      | 84.5%                  | ???                       |
| Naive Baseline Accuracy         | 84.7%                  | ???                       |
| # Algos Beating EW (avg/fold)   | 2/48                   | ???/48                    |
| EW Sharpe                       | +0.37                  | ???                       |
| Reflex Agent Sharpe (net fit)   | +0.16                  | ???                       |
| Oracle Agent Sharpe             | -0.30                  | ???                       |
| Oracle Gap                      | -0.009                 | ???                       |
| Reflex vs EW Gap                | -0.21                  | ???                       |

This directly shows whether Plan 2's findings hold up or were artifacts 
of the specific 2021–2024 test window.

Save to: results/plan2_vs_plan3_comparison.csv
Plot: results/22_plan2_vs_plan3.png
```

---

## Execution Instructions

```bash
cd regime_algo_selection

# Step 1: Extend data (downloads new data, caches to CSV)
python -m data.loader_extended

# Steps 2-5: Run walk-forward validation and all analyses
python -m experiments.walk_forward_analysis
```

The walk_forward_analysis.py script should:
1. Load extended data
2. Build algorithm space (verify N=2 and N=5 both work)
3. Generate all folds
4. Run each fold (with progress output)
5. Generate all analysis tables and plots
6. Print the Plan 2 vs Plan 3 comparison table
7. Save everything to results/

Expected runtime: ~5-10 minutes (48 algorithms × 12 folds × backtest loop).

---

## Expected Outputs

```
results/
├── (Plans 1+2: plots 01–16, already exist)
├── 17_walk_forward_sharpe_comparison.png
├── 18_regime_conditional_performance.png
├── 19_algorithm_stability.png
├── 20_algorithm_rank_stability.png
├── 21_oracle_gap_over_time.png
├── 22_plan2_vs_plan3.png
├── walk_forward_performance.csv
├── algorithm_stability.csv
├── algorithm_rank_stability.csv
└── plan2_vs_plan3_comparison.csv
```

---

## What to Present to the Professor

With Walk-Forward results, the story becomes much stronger:

1. **"Across 12 test windows spanning 2013–2024, the Reflex Agent beats/loses to EW in X/12 folds."**
   → Systematic finding, not a single data point.

2. **"The Reflex Agent outperforms EW during [crisis/volatile] periods but underperforms during [calm/bull] periods."**
   → Or the opposite — either way, it's a robust, nuanced finding.

3. **"The optimal algorithm per regime changes across folds — stability is only X%."**
   → If low: strong argument that a static lookup table can't work, need a learned meta-learner.
   → If high: the regime-algorithm mapping is robust, problem is elsewhere.

4. **"The Oracle Gap is larger during [specific conditions], suggesting regime prediction matters most when [X]."**
   → Connects classifier quality to portfolio impact in a differentiated way.

5. **"These findings motivate Plan 4: expanding the algorithm pool (Tier 2/3) and replacing the rigid reflex agent with a learned meta-learner."**
   → Evidence-based roadmap.
