# config.py — All hyperparameters, constants, and paths for the Regime-Aware Reflex Agent

import os

# === Paths ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cache")
RESULTS_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Asset Universe ===
ASSETS = ["SPY", "TLT", "GLD", "EFA", "VNQ"]  # equity, bonds, gold, intl, real estate
N_ASSETS = len(ASSETS)

# === Time Range ===
START_DATE = "2006-01-01"   # VIX + ETF data available from here
END_DATE   = "2024-12-31"
TRAIN_END  = "2018-12-31"   # Training: 2006-2018
VAL_END    = "2020-12-31"   # Validation: 2019-2020
# Test: 2021-2024

# === Regime Thresholds (VIX-based) ===
# g(v_t): v<=15 → 1, 15<v<=20 → 2, 20<v<=30 → 3, v>30 → 4
REGIME_THRESHOLDS = [15, 20, 30]
REGIME_NAMES = {1: "Calm", 2: "Normal", 3: "Tense", 4: "Crisis"}
N_REGIMES = 4

# === Switching Cost ===
KAPPA = 0.001  # Portfolio-level switching cost coefficient (L1 turnover penalty)

# === Tier 1 Algorithm Hyperparameter Grids ===
LOOKBACKS_COV   = [20, 40, 60, 120, 252]   # Covariance-based methods
LOOKBACKS_MOM   = [5, 10, 20, 60, 120]     # Momentum
LOOKBACKS_TREND = [5, 10, 20, 60]          # Trend-following
TREND_BETAS     = [1, 2, 3]                # Concentration parameter for trend
RISK_AVERSIONS  = [0.5, 1, 2, 5]          # For mean-variance

# === Random Seed ===
RANDOM_SEED = 42
