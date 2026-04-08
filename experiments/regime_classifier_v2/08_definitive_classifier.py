"""
08_definitive_classifier.py -- Plan 11: Definitive Regime Shift Classifier

Four phases executed in order:
  Phase 0 -- Label redesign: persistence-aware shift labels vs any-shift baseline
  Phase 1 -- Feature ablation: crossing-geometry features vs existing features
  Phase 2 -- Model optimization: LR (full grid) + XGBoost + optional RF
  Phase 3 -- Horizon sweep: H in {3, 5, 7, 10} with best model
  Phase 4 -- Final report: best config, all metrics, all plots

Key constraints:
  - Walk-forward: 12 folds, test years 2013-2024, expanding training window.
  - StandardScaler fit on training data only per fold.
  - 6 baselines (3 trivial + 3 strong heuristic) computed for every configuration.
  - Overfitting diagnostics (train-test F1 gap, fold variance) logged throughout.
  - If runtime > 2hr: LR runs fully, XGB n_iter reduced to 100, RF skipped.
  - Feature correlation check: drop features with |r| > 0.90 per fold.
  - Random seed = 42 everywhere.

Run from project root (Implementierung1/):
  python experiments/regime_classifier_v2/08_definitive_classifier.py
"""

# ===========================================================================
# SECTION 1: Imports & Path Setup
# ===========================================================================
import importlib.util
import json
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.dummy            import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score,
    recall_score, roc_curve, auc,
)
from sklearn.model_selection  import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, learning_curve,
)
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler

try:
    from sklearn.ensemble import RandomForestClassifier
    HAS_RF = True
except ImportError:
    HAS_RF = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed -- XGB steps will be skipped.")

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(PROJECT_ROOT, "results", "regime_classifier_v2", "definitive")
os.makedirs(OUT_DIR, exist_ok=True)

# Import path constants from config
from regime_algo_selection.config import REGIME_THRESHOLDS, DATA_DIR

PRICES_CACHE = os.path.join(DATA_DIR, "prices.csv")
VIX_CACHE    = os.path.join(DATA_DIR, "vix.csv")

# Script start time for runtime guard
SCRIPT_START = time.time()

# ===========================================================================
# SECTION 2: Constants
# ===========================================================================
RANDOM_SEED  = 42
TEST_YEARS   = list(range(2013, 2025))   # 12 folds: test years 2013-2024
THRESHOLDS   = REGIME_THRESHOLDS          # [15, 20, 30]

# Label configurations for Phase 0
LABEL_CONFIGS = {
    "any_shift": {"min_persistence": 1},
    "persist_2": {"min_persistence": 2},
    "persist_3": {"min_persistence": 3},
    "persist_5": {"min_persistence": 5},
}

# Horizons: H=5 and H=7 are primary; H=3 and H=10 are reference
HORIZONS_PRIMARY   = [5, 7]
HORIZONS_REFERENCE = [3, 10]
HORIZONS_ALL       = [3, 5, 7, 10]

# Feature set names used in Phase 1
FEATURE_SET_NAMES = ["A", "B", "C", "D", "E", "F", "G"]

# Group B features from Plan 9 (used as baseline in Phase 0)
GROUP_B_FEATURES = [
    "VIX_MA20", "max_VIX_window", "min_VIX_window",
    "VIX_slope_20", "VIX_rolling_std_10",
    "VIX_dist_nearest", "VIX_dist_upper", "VIX_dist_lower",
]

# Plan 10 LR F1 reference (any_shift, H=5) -- from Plan 10 Step 2
PLAN10_LR_F1 = 0.656

# Runtime guard thresholds
RUNTIME_LIMIT_SECONDS = 7200   # 2 hours
XGB_N_ITER_FULL       = 150
XGB_N_ITER_REDUCED    = 100


# ===========================================================================
# SECTION 3: Data Loading & Feature Engineering
# ===========================================================================

def load_raw_data():
    """Load prices (SPY, TLT, GLD, EFA, VNQ) and VIX from cache CSV files."""
    prices = pd.read_csv(PRICES_CACHE, index_col=0, parse_dates=True)
    vix    = pd.read_csv(VIX_CACHE,    index_col=0, parse_dates=True).squeeze()
    vix.name = "VIX"
    return prices, vix


def compute_regime_labels(vix):
    """Deterministic VIX-threshold regime map: 1=Calm, 2=Normal, 3=Tense, 4=Crisis."""
    t1, t2, t3 = THRESHOLDS
    labels = pd.cut(
        vix,
        bins=[-float("inf"), t1, t2, t3, float("inf")],
        labels=[1, 2, 3, 4],
    ).astype(int)
    labels.name = "regime"
    return labels


def compute_regime_tenure(regime_series):
    """
    For each day t, compute the number of consecutive days the current regime
    has persisted up to and including day t.
    High tenure = stable regime; low tenure = recent change, likely to shift again.
    """
    reg_arr = regime_series.values
    n       = len(reg_arr)
    tenure  = np.ones(n, dtype=float)
    for i in range(1, n):
        tenure[i] = tenure[i - 1] + 1 if reg_arr[i] == reg_arr[i - 1] else 1
    return pd.Series(tenure, index=regime_series.index, name="regime_tenure")


def compute_all_features(prices, vix):
    """
    Compute the complete feature matrix used across all phases:
      - Base level & trend features (from Plan 9 Group A)
      - Threshold-distance features (Plan 9 Group B additions)
      - NEW crossing-geometry features (Plan 11)

    All features use only data up to time t. No look-ahead bias.
    """
    spy   = prices["SPY"]
    tlt   = prices["TLT"]
    gld   = prices["GLD"]
    spy_r = spy.pct_change()

    # --- Base level ---
    VIX_MA20           = vix.rolling(20).mean()
    VIX_rolling_std_20 = vix.rolling(20).std()
    VIX_rolling_std_10 = vix.rolling(10).std()
    max_VIX_window     = vix.rolling(20).max()
    min_VIX_window     = vix.rolling(20).min()
    VIX_slope_20       = vix - vix.shift(20)

    # --- Threshold distances ---
    def _dist_nearest(v):
        return min(abs(v - t) for t in THRESHOLDS)

    def _dist_upper(v):
        above = [t for t in THRESHOLDS if t >= v]
        return (min(above) - v) if above else (v - THRESHOLDS[-1])

    def _dist_lower(v):
        below = [t for t in THRESHOLDS if t < v]
        return (v - max(below)) if below else v

    VIX_dist_nearest = vix.apply(_dist_nearest)
    VIX_dist_upper   = vix.apply(_dist_upper)
    VIX_dist_lower   = vix.apply(_dist_lower)

    # --- NEW: Crossing-geometry features ---
    window_range     = max_VIX_window - min_VIX_window
    window_position  = (vix - min_VIX_window) / (window_range + 1e-8)
    crossing_pressure       = VIX_slope_20 / (VIX_dist_nearest + 1e-8)
    threshold_instability   = VIX_rolling_std_10 / (VIX_dist_nearest + 1e-8)
    recent_touch            = VIX_dist_nearest.rolling(5).min()
    recent_threshold_touch_5 = (recent_touch < 1.0).astype(float)

    # signed_dist: distance to the threshold VIX is currently moving toward.
    # Positive = approaching from below; negative = approaching from above.
    slope_arr = VIX_slope_20.fillna(0).values
    vix_arr   = vix.values

    def _signed_distance(vix_val, slope):
        dists    = [(t - vix_val) for t in THRESHOLDS]
        relevant = [(abs(d), d) for d in dists
                    if (d > 0 and slope > 0) or (d < 0 and slope < 0)]
        if relevant:
            return min(relevant, key=lambda x: x[0])[1]
        return min(dists, key=abs)

    signed_dist = pd.Series(
        [_signed_distance(v, s) for v, s in zip(vix_arr, slope_arr)],
        index=vix.index, name="signed_dist",
    )

    return pd.DataFrame({
        # Plan 9 Group A
        "VIX_MA20":            VIX_MA20,
        "max_VIX_window":      max_VIX_window,
        "min_VIX_window":      min_VIX_window,
        "VIX_slope_20":        VIX_slope_20,
        "VIX_rolling_std_10":  VIX_rolling_std_10,
        # Plan 9 Group B additions
        "VIX_dist_nearest":    VIX_dist_nearest,
        "VIX_dist_upper":      VIX_dist_upper,
        "VIX_dist_lower":      VIX_dist_lower,
        # NEW: Crossing-geometry (Plan 11)
        "window_position":            window_position,
        "crossing_pressure":          crossing_pressure,
        "threshold_instability":      threshold_instability,
        "recent_threshold_touch_5":   recent_threshold_touch_5,
        "signed_dist":                signed_dist,
    }, index=vix.index)


def get_feature_sets():
    """Return dict of feature set name -> list of feature column names (Plan 11 spec)."""
    return {
        "A": ["VIX_MA20", "max_VIX_window", "min_VIX_window"],
        "B": ["VIX_MA20", "max_VIX_window", "min_VIX_window", "VIX_dist_nearest"],
        "C": ["VIX_MA20", "max_VIX_window", "min_VIX_window",
              "VIX_dist_nearest", "VIX_slope_20", "VIX_rolling_std_10"],
        "D": ["VIX_dist_nearest", "crossing_pressure", "threshold_instability",
              "window_position", "recent_threshold_touch_5"],
        "E": ["VIX_MA20", "max_VIX_window", "min_VIX_window",
              "VIX_dist_nearest", "VIX_slope_20", "VIX_rolling_std_10",
              "crossing_pressure", "threshold_instability",
              "window_position", "recent_threshold_touch_5", "signed_dist"],
        "F": ["VIX_dist_nearest", "crossing_pressure", "threshold_instability"],
        "G": ["VIX_MA20", "max_VIX_window", "min_VIX_window",
              "VIX_dist_nearest", "VIX_slope_20", "VIX_rolling_std_10",
              "crossing_pressure", "threshold_instability",
              "window_position", "recent_threshold_touch_5"],
    }


# ===========================================================================
# SECTION 4: Label Functions
# ===========================================================================

def compute_persistent_shift_label(regime_series, H, min_persistence):
    """
    Persistence-aware binary shift label.

    label_t = 1 if there exists tau in [t+1, t+H] such that:
      (a) regime[tau] != regime[t]   (a regime change starts at tau)
      (b) regime[tau+k] == regime[tau] for all k in [0, min_persistence-1]
          (the new regime persists for at least min_persistence consecutive days)
    label_t = 0 otherwise.
    label_t = NaN for rows where the required future window is unavailable.

    When min_persistence=1 this reduces to the standard any-shift label.
    """
    regimes = regime_series.values
    n       = len(regimes)
    labels  = np.full(n, np.nan)

    for t in range(n):
        # Need H days ahead, plus min_persistence-1 days to verify persistence
        if t + H + min_persistence - 1 >= n:
            continue
        found = False
        for tau in range(t + 1, t + H + 1):
            if regimes[tau] != regimes[t]:
                # Verify persistence window
                end_check = tau + min_persistence
                if end_check <= n and np.all(regimes[tau:end_check] == regimes[tau]):
                    found = True
                    break
        labels[t] = 1.0 if found else 0.0

    return pd.Series(labels, index=regime_series.index, name="shift_label")


def build_dataset(features_df, regime, tenure, vix, H, min_persistence):
    """
    Assemble the full dataset for a given H and min_persistence.
    Includes features, regime, tenure, raw VIX (for baselines), and shift label.
    Drops NaN rows from rolling-window warmup and label tail.
    """
    label = compute_persistent_shift_label(regime, H, min_persistence)
    ds = features_df.copy()
    ds["regime"]      = regime
    ds["regime_tenure"] = tenure
    ds["VIX"]         = vix
    ds["shift_label"] = label
    return ds.dropna()


# ===========================================================================
# SECTION 5: Metrics & Utilities
# ===========================================================================

def compute_metrics(y_true, y_pred):
    """Compute classification metrics for the positive (shift) class."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    n_neg = tn + fp
    return {
        "recall":    recall_score(   y_true, y_pred, pos_label=1, zero_division=0),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1":        f1_score(       y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy":  accuracy_score( y_true, y_pred),
        "far":       fp / max(n_neg, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def compute_pr_auc(y_true, y_prob):
    """PR-AUC = average precision score (area under precision-recall curve)."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_prob))


def find_optimal_threshold(y_true, y_prob):
    """
    Find the probability threshold that maximises F1 on the given (training) set.
    Never applied to test data -- threshold is derived from training labels only.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])   # last element has no corresponding threshold
    return float(thresholds[best_idx])


def drop_correlated_features(X_train, feature_names, y_train, threshold=0.90):
    """
    Remove highly correlated features (|r| > threshold).
    For each correlated pair, keep the feature with the higher F-statistic.
    Returns: (kept_indices, kept_names, dropped_names)
    """
    n = X_train.shape[1]
    if n <= 1:
        return list(range(n)), list(feature_names), []

    corr  = np.corrcoef(X_train.T)
    fstat, _ = f_classif(X_train, y_train)
    fstat = np.nan_to_num(fstat, nan=0.0)

    dropped = set()
    for i in range(n):
        for j in range(i + 1, n):
            if i in dropped or j in dropped:
                continue
            if abs(corr[i, j]) > threshold:
                # Drop the feature with lower F-statistic
                dropped.add(j if fstat[i] >= fstat[j] else i)

    kept_idx   = [i for i in range(n) if i not in dropped]
    kept_names = [feature_names[i] for i in kept_idx]
    drop_names = [feature_names[i] for i in dropped]
    return kept_idx, kept_names, drop_names


def eta_str(elapsed_s, done, total):
    """Human-readable ETA string."""
    if done == 0:
        return "?"
    remaining = elapsed_s / done * (total - done)
    if remaining < 60:
        return f"{remaining:.0f}s"
    return f"{remaining/60:.1f}min"


def runtime_exceeded():
    """Return True if the script has been running for more than 2 hours."""
    return (time.time() - SCRIPT_START) > RUNTIME_LIMIT_SECONDS


# ===========================================================================
# SECTION 6: Baseline Computation
# ===========================================================================

def compute_all_baselines(df_tr, df_te, y_tr, y_te):
    """
    Compute all 6 baselines for one fold.

    Trivial baselines (no information):
      B1: Always predict 0 (no shift)
      B2: Always predict 1 (always shift)
      B3: Stratified random (proportional to training class distribution)

    Strong heuristic baselines (use VIX/regime structure, no ML):
      B4: Yesterday's shift (predict shift if regime changed yesterday)
      B5: Threshold proximity (predict shift if VIX within X points of any threshold)
      B6: Regime tenure (predict shift if current regime tenure <= X days)

    Returns dict with metrics for each baseline.
    """
    n_tr = len(y_tr)
    n_te = len(y_te)
    results = {}

    # B1: Always 0
    yp_b1 = np.zeros(n_te, dtype=int)
    m1 = compute_metrics(y_te, yp_b1)
    results["b1_always0_f1"]        = m1["f1"]
    results["b1_always0_recall"]    = m1["recall"]
    results["b1_always0_precision"] = m1["precision"]

    # B2: Always 1
    yp_b2 = np.ones(n_te, dtype=int)
    m2 = compute_metrics(y_te, yp_b2)
    results["b2_always1_f1"]        = m2["f1"]
    results["b2_always1_recall"]    = m2["recall"]
    results["b2_always1_precision"] = m2["precision"]

    # B3: Stratified random
    rng  = np.random.RandomState(RANDOM_SEED)
    p_pos = y_tr.mean()
    yp_b3 = (rng.rand(n_te) < p_pos).astype(int)
    m3 = compute_metrics(y_te, yp_b3)
    results["b3_stratified_f1"]        = m3["f1"]
    results["b3_stratified_recall"]    = m3["recall"]
    results["b3_stratified_precision"] = m3["precision"]

    # B4: Yesterday's shift
    regime_te  = df_te["regime"].values
    regime_tr  = df_tr["regime"].values
    yp_b4      = np.zeros(n_te, dtype=int)
    last_tr_r  = regime_tr[-1] if len(regime_tr) > 0 else regime_te[0]
    yp_b4[0]   = 1 if regime_te[0] != last_tr_r else 0
    for i in range(1, n_te):
        yp_b4[i] = 1 if regime_te[i] != regime_te[i - 1] else 0
    m4 = compute_metrics(y_te, yp_b4)
    results["b4_yesterday_f1"]        = m4["f1"]
    results["b4_yesterday_recall"]    = m4["recall"]
    results["b4_yesterday_precision"] = m4["precision"]

    # B5: Threshold proximity  -- optimise margin on training data
    vix_dist_tr = df_tr["VIX_dist_nearest"].values
    vix_dist_te = df_te["VIX_dist_nearest"].values
    best_margin, best_m5_f1 = 1.0, -1.0
    for margin in [1.0, 1.5, 2.0, 3.0]:
        yp_tr_m = (vix_dist_tr < margin).astype(int)
        f_tr = f1_score(y_tr, yp_tr_m, pos_label=1, zero_division=0)
        if f_tr > best_m5_f1:
            best_m5_f1 = f_tr
            best_margin = margin
    yp_b5 = (vix_dist_te < best_margin).astype(int)
    m5 = compute_metrics(y_te, yp_b5)
    results["b5_proximity_f1"]        = m5["f1"]
    results["b5_proximity_recall"]    = m5["recall"]
    results["b5_proximity_precision"] = m5["precision"]
    results["b5_best_margin"]         = best_margin

    # B6: Regime tenure threshold  -- optimise threshold on training data
    tenure_tr = df_tr["regime_tenure"].values
    tenure_te = df_te["regime_tenure"].values
    best_tenure_thresh, best_m6_f1 = 3, -1.0
    for thresh in [3, 5, 10]:
        yp_tr_t = (tenure_tr <= thresh).astype(int)
        f_tr = f1_score(y_tr, yp_tr_t, pos_label=1, zero_division=0)
        if f_tr > best_m6_f1:
            best_m6_f1 = f_tr
            best_tenure_thresh = thresh
    yp_b6 = (tenure_te <= best_tenure_thresh).astype(int)
    m6 = compute_metrics(y_te, yp_b6)
    results["b6_tenure_f1"]        = m6["f1"]
    results["b6_tenure_recall"]    = m6["recall"]
    results["b6_tenure_precision"] = m6["precision"]
    results["b6_best_tenure"]      = best_tenure_thresh

    return results


# ===========================================================================
# SECTION 7: Core Walk-Forward Runners
# ===========================================================================

def run_walk_forward_basic(dataset, feature_cols, model_fn,
                           compute_baselines=True, threshold=0.5,
                           phase_label=""):
    """
    12-fold walk-forward with a fixed model function.

    model_fn(n_neg, n_pos) -> unfitted sklearn estimator
    Returns list of per-fold result dicts (test metrics + baselines).
    Includes train metrics for overfitting diagnostics.
    """
    records   = []
    y_all_true, y_all_prob = [], []
    cm_agg    = np.zeros((2, 2), dtype=int)

    for fold_i, test_year in enumerate(TEST_YEARS):
        t0    = time.time()
        dates = dataset.index
        tr_m  = dates.year <= (test_year - 1)
        te_m  = dates.year == test_year

        df_tr = dataset[tr_m]
        df_te = dataset[te_m]
        if len(df_tr) == 0 or len(df_te) == 0:
            continue

        X_tr_raw = df_tr[feature_cols].values.astype(float)
        X_te_raw = df_te[feature_cols].values.astype(float)
        y_tr     = df_tr["shift_label"].values.astype(int)
        y_te     = df_te["shift_label"].values.astype(int)

        # Correlation pruning
        kept_idx, kept_names, dropped = drop_correlated_features(
            X_tr_raw, feature_cols, y_tr
        )
        if dropped:
            pass  # logged in caller if needed
        X_tr_raw = X_tr_raw[:, kept_idx]
        X_te_raw = X_te_raw[:, kept_idx]

        # Scale (fit on training only)
        sc     = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_raw)
        X_te_s = sc.transform(X_te_raw)

        # Fit model
        n_neg = int((y_tr == 0).sum())
        n_pos = int((y_tr == 1).sum())
        model = model_fn(n_neg, n_pos)
        model.fit(X_tr_s, y_tr)

        # Train metrics (overfitting check)
        y_tr_prob  = model.predict_proba(X_tr_s)[:, 1]
        y_tr_pred  = (y_tr_prob >= threshold).astype(int)
        train_m    = compute_metrics(y_tr, y_tr_pred)
        train_m["pr_auc"] = compute_pr_auc(y_tr, y_tr_prob)

        # Test metrics
        y_te_prob  = model.predict_proba(X_te_s)[:, 1]
        y_te_pred  = (y_te_prob >= threshold).astype(int)
        test_m     = compute_metrics(y_te, y_te_pred)
        test_m["pr_auc"] = compute_pr_auc(y_te, y_te_prob)

        y_all_true.extend(y_te.tolist())
        y_all_prob.extend(y_te_prob.tolist())
        cm_agg += confusion_matrix(y_te, y_te_pred, labels=[0, 1])

        rec = {
            "fold":           fold_i + 1,
            "test_year":      test_year,
            "n_train":        len(y_tr),
            "n_test":         len(y_te),
            "shift_pct_train": float(n_pos / max(n_neg + n_pos, 1) * 100),
            "shift_pct_test":  float(y_te.mean() * 100),
            "train_recall":    train_m["recall"],
            "train_precision": train_m["precision"],
            "train_f1":        train_m["f1"],
            "train_pr_auc":    train_m["pr_auc"],
            "test_recall":     test_m["recall"],
            "test_precision":  test_m["precision"],
            "test_f1":         test_m["f1"],
            "test_pr_auc":     test_m["pr_auc"],
            "test_accuracy":   test_m["accuracy"],
            "test_far":        test_m["far"],
            "gap_f1":          train_m["f1"] - test_m["f1"],
            "dropped_features": ",".join(dropped) if dropped else "",
        }

        if compute_baselines:
            bl = compute_all_baselines(df_tr, df_te, y_tr, y_te)
            rec.update(bl)

        elapsed = time.time() - t0
        eta     = eta_str(time.time() - SCRIPT_START - 0,
                          fold_i + 1, len(TEST_YEARS))
        if phase_label:
            print(f"    {phase_label} fold {fold_i+1:2d} ({test_year})"
                  f"  recall={test_m['recall']:.3f}"
                  f"  prec={test_m['precision']:.3f}"
                  f"  f1={test_m['f1']:.3f}"
                  f"  gap={rec['gap_f1']:+.3f}"
                  f"  [{elapsed:.1f}s]")
        records.append(rec)

    return records, np.array(y_all_true), np.array(y_all_prob), cm_agg


def run_walk_forward_cv(dataset, feature_cols, get_model_fn,
                        get_param_grid_fn, scoring="f1",
                        use_randomized=False, n_iter=150,
                        phase_label=""):
    """
    12-fold walk-forward with inner 3-fold GridSearchCV (or RandomizedSearchCV).

    get_model_fn(n_neg, n_pos) -> unfitted estimator
    get_param_grid_fn(n_neg, n_pos, n_features) -> param grid dict
    Returns (records, y_all_true, y_all_prob, cm_agg, fold_best_params, fold_estimators)
    """
    inner_cv         = StratifiedKFold(n_splits=3, shuffle=True,
                                       random_state=RANDOM_SEED)
    records          = []
    y_all_true       = []
    y_all_prob       = []
    cm_agg           = np.zeros((2, 2), dtype=int)
    fold_best_params = []
    fold_estimators  = []

    for fold_i, test_year in enumerate(TEST_YEARS):
        t0    = time.time()
        dates = dataset.index
        tr_m  = dates.year <= (test_year - 1)
        te_m  = dates.year == test_year

        df_tr = dataset[tr_m]
        df_te = dataset[te_m]
        if len(df_tr) == 0 or len(df_te) == 0:
            continue

        X_tr_raw = df_tr[feature_cols].values.astype(float)
        X_te_raw = df_te[feature_cols].values.astype(float)
        y_tr     = df_tr["shift_label"].values.astype(int)
        y_te     = df_te["shift_label"].values.astype(int)

        # Correlation pruning
        kept_idx, kept_names, dropped = drop_correlated_features(
            X_tr_raw, feature_cols, y_tr
        )
        X_tr_raw = X_tr_raw[:, kept_idx]
        X_te_raw = X_te_raw[:, kept_idx]

        # Scale
        sc     = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_raw)
        X_te_s = sc.transform(X_te_raw)

        n_neg = int((y_tr == 0).sum())
        n_pos = int((y_tr == 1).sum())

        base_model  = get_model_fn(n_neg, n_pos)
        param_grid  = get_param_grid_fn(n_neg, n_pos, len(kept_names))

        if use_randomized:
            search = RandomizedSearchCV(
                base_model, param_grid,
                n_iter=n_iter, cv=inner_cv,
                scoring=scoring, refit=True,
                random_state=RANDOM_SEED, n_jobs=-1,
            )
        else:
            search = GridSearchCV(
                base_model, param_grid,
                cv=inner_cv, scoring=scoring,
                refit=True, n_jobs=-1,
            )

        search.fit(X_tr_s, y_tr)
        best_est = search.best_estimator_

        # Train metrics (overfitting check)
        y_tr_prob  = best_est.predict_proba(X_tr_s)[:, 1]
        y_tr_pred  = (y_tr_prob >= 0.5).astype(int)
        train_m    = compute_metrics(y_tr, y_tr_pred)
        train_m["pr_auc"] = compute_pr_auc(y_tr, y_tr_prob)

        # Find optimal threshold on TRAINING data only
        opt_thresh = find_optimal_threshold(y_tr, y_tr_prob)

        # Test metrics at default threshold
        y_te_prob       = best_est.predict_proba(X_te_s)[:, 1]
        y_te_pred_05    = (y_te_prob >= 0.5).astype(int)
        y_te_pred_opt   = (y_te_prob >= opt_thresh).astype(int)
        test_m_05       = compute_metrics(y_te, y_te_pred_05)
        test_m_opt      = compute_metrics(y_te, y_te_pred_opt)
        test_m_05["pr_auc"]  = compute_pr_auc(y_te, y_te_prob)
        test_m_opt["pr_auc"] = compute_pr_auc(y_te, y_te_prob)

        y_all_true.extend(y_te.tolist())
        y_all_prob.extend(y_te_prob.tolist())
        cm_agg += confusion_matrix(y_te, y_te_pred_05, labels=[0, 1])

        params_str = json.dumps(search.best_params_, default=str)

        rec = {
            "fold":            fold_i + 1,
            "test_year":       test_year,
            "n_train":         len(y_tr),
            "n_test":          len(y_te),
            "shift_pct_train": float(n_pos / max(n_neg + n_pos, 1) * 100),
            "shift_pct_test":  float(y_te.mean() * 100),
            "best_params":     params_str,
            "opt_threshold":   opt_thresh,
            "train_recall":    train_m["recall"],
            "train_precision": train_m["precision"],
            "train_f1":        train_m["f1"],
            "train_pr_auc":    train_m["pr_auc"],
            # Test at threshold=0.5
            "test_recall":     test_m_05["recall"],
            "test_precision":  test_m_05["precision"],
            "test_f1":         test_m_05["f1"],
            "test_pr_auc":     test_m_05["pr_auc"],
            "test_accuracy":   test_m_05["accuracy"],
            "test_far":        test_m_05["far"],
            "gap_f1":          train_m["f1"] - test_m_05["f1"],
            # Test at optimal threshold
            "test_f1_opt":     test_m_opt["f1"],
            "test_recall_opt": test_m_opt["recall"],
            "test_prec_opt":   test_m_opt["precision"],
            "dropped_features": ",".join(dropped) if dropped else "",
        }

        bl = compute_all_baselines(df_tr, df_te, y_tr, y_te)
        rec.update(bl)

        elapsed = time.time() - t0
        print(f"    {phase_label} fold {fold_i+1:2d} ({test_year})"
              f"  recall={test_m_05['recall']:.3f}"
              f"  prec={test_m_05['precision']:.3f}"
              f"  f1={test_m_05['f1']:.3f}"
              f"  gap={rec['gap_f1']:+.3f}"
              f"  opt_thr={opt_thresh:.3f}"
              f"  [{elapsed:.0f}s]"
              f"  params={params_str[:50]}")

        records.append(rec)
        fold_best_params.append(search.best_params_)
        fold_estimators.append((best_est, sc, kept_idx))

    return (records, np.array(y_all_true), np.array(y_all_prob),
            cm_agg, fold_best_params, fold_estimators)


# ===========================================================================
# SECTION 8: Phase 0 — Label Redesign
# ===========================================================================

def summarise_records(records):
    """Aggregate fold records to mean metrics dict."""
    df = pd.DataFrame(records)
    cols = ["test_recall", "test_precision", "test_f1", "test_pr_auc",
            "test_accuracy", "test_far"]
    out = {}
    for c in cols:
        if c in df.columns:
            out[f"avg_{c}"] = df[c].mean()
            out[f"std_{c}"] = df[c].std()
    return out


def select_representative_params(params_list):
    """Return the most common parameter dict across folds."""
    if not params_list:
        return {}
    counts = {}
    by_key = {}
    for params in params_list:
        key = json.dumps(params, sort_keys=True, default=str)
        counts[key] = counts.get(key, 0) + 1
        by_key[key] = dict(params)
    best_key = max(counts, key=counts.get)
    return by_key[best_key]


def run_phase0(dataset_cache, regime, tenure, vix, features_df):
    """
    Phase 0: Compare label variants (any_shift, persist_2, persist_3, persist_5)
    across H in {3, 5, 7, 10} using baseline LR (C=0.001, balanced).
    Computes all 6 baselines for every combination.
    """
    print("\n" + "=" * 60)
    print("  PHASE 0: Label Redesign")
    print("=" * 60)
    phase_start = time.time()

    def baseline_lr(n_neg, n_pos):
        return LogisticRegression(
            C=0.001, class_weight="balanced",
            solver="saga", max_iter=2000, random_state=RANDOM_SEED,
        )

    rows = []
    H_list = HORIZONS_ALL  # [3, 5, 7, 10]

    for lname, lcfg in LABEL_CONFIGS.items():
        min_p = lcfg["min_persistence"]
        for H in H_list:
            t_lh = time.time()
            ds = build_dataset(features_df, regime, tenure, vix, H, min_p)
            n_valid    = len(ds)
            shift_pct  = float(ds["shift_label"].mean() * 100)

            # Skip if shift% < 10% (plan: persist_5 may be too strict)
            if shift_pct < 10.0 and lname == "persist_5":
                print(f"  SKIP {lname} H={H}: shift_pct={shift_pct:.1f}% < 10% threshold")
                rows.append({
                    "label_variant": lname, "H": H,
                    "min_persistence": min_p, "shift_pct": shift_pct,
                    "skipped": True,
                })
                continue

            print(f"\n  [{lname}  H={H}]  shift_pct={shift_pct:.1f}%  "
                  f"n={n_valid}")

            records, y_all_true, y_all_prob, cm_agg = run_walk_forward_basic(
                ds, GROUP_B_FEATURES, baseline_lr,
                compute_baselines=True,
                phase_label=f"{lname}/H{H}",
            )

            if not records:
                continue

            agg   = summarise_records(records)
            df_r  = pd.DataFrame(records)

            # Aggregate baselines across folds
            bl_cols = [c for c in df_r.columns if c.startswith("b")]
            bl_agg  = {c: df_r[c].mean() for c in bl_cols if df_r[c].dtype != object}

            row = {
                "label_variant":  lname,
                "H":              H,
                "min_persistence": min_p,
                "shift_pct":      shift_pct,
                "n_days":         n_valid,
                "avg_recall":     agg.get("avg_test_recall", 0),
                "avg_precision":  agg.get("avg_test_precision", 0),
                "avg_f1":         agg.get("avg_test_f1", 0),
                "std_f1":         agg.get("std_test_f1", 0),
                "avg_pr_auc":     agg.get("avg_test_pr_auc", 0),
                "avg_far":        agg.get("avg_test_far", 0),
                "skipped":        False,
                **bl_agg,
            }
            # Best naive baseline for comparison column
            row["vs_stratified"] = (row["avg_f1"]
                                    - bl_agg.get("b3_stratified_f1", 0))
            rows.append(row)

            b_best = max(
                bl_agg.get("b4_yesterday_f1", 0),
                bl_agg.get("b5_proximity_f1", 0),
                bl_agg.get("b6_tenure_f1", 0),
            )
            elapsed = time.time() - t_lh
            print(f"    --> avg_f1={row['avg_f1']:.3f}  "
                  f"std={row['std_f1']:.3f}  "
                  f"vs_heuristic_best={row['avg_f1'] - b_best:+.3f}  "
                  f"[{elapsed:.0f}s]")

    # Print comparison table
    print("\n=== Phase 0: Label Comparison ===\n")
    print(f"{'Label Variant':<15} | {'H':>3} | "
          f"{'Shift%':>7} | {'Recall':>6} | {'Prec':>6} | "
          f"{'F1':>6} | {'PR-AUC':>6} | {'vs Naive(strat)':>16}")
    print("-" * 85)
    for r in rows:
        if r.get("skipped"):
            continue
        vs_str = f"{r.get('vs_stratified', 0):+.3f}"
        print(f"{r['label_variant']:<15} | {r['H']:>3} | "
              f"{r['shift_pct']:>6.1f}% | "
              f"{r['avg_recall']:>6.3f} | {r['avg_precision']:>6.3f} | "
              f"{r['avg_f1']:>6.3f} | {r['avg_pr_auc']:>6.3f} | "
              f"{vs_str:>16}")

    # Select best label: highest F1 at H=5 or H=7, shift_pct in [15,45], std_f1<0.15
    candidates = [r for r in rows if not r.get("skipped")
                  and r["H"] in HORIZONS_PRIMARY
                  and 15.0 <= r["shift_pct"] <= 45.0
                  and r["std_f1"] < 0.20]
    if not candidates:
        # Relax constraints
        candidates = [r for r in rows if not r.get("skipped")
                      and r["H"] in HORIZONS_PRIMARY]
    if not candidates:
        candidates = [r for r in rows if not r.get("skipped")]

    best_row = max(candidates, key=lambda r: r["avg_f1"])
    best_label_variant  = best_row["label_variant"]
    best_min_persistence = best_row["min_persistence"]
    best_H0             = best_row["H"]

    print(f"\nBest label: [{best_label_variant}] at H={best_H0}, "
          f"F1 = {best_row['avg_f1']:.3f}, "
          f"shift_pct = {best_row['shift_pct']:.1f}%")

    print(f"\nPhase 0 completed in {(time.time()-phase_start)/60:.1f} min")
    return rows, best_label_variant, best_min_persistence, best_H0


def save_phase0_outputs(rows, out_dir):
    """Save Phase 0 CSV and plots."""
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "phase0_label_comparison.csv"), index=False)

    valid = df[~df.get("skipped", False)].copy() if "skipped" in df.columns \
        else df.copy()
    valid = valid[~valid["skipped"]] if "skipped" in valid.columns else valid

    # --- Plot 1: shift_pct per label_variant × H ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(valid))
    ax.bar(x, valid["shift_pct"], color="steelblue", alpha=0.8)
    ax.axhline(15, color="red",   ls="--", lw=1, label="15% lower bound")
    ax.axhline(45, color="orange",ls="--", lw=1, label="45% upper bound")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['label_variant']}\nH={r['H']}" for _, r in valid.iterrows()],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_ylabel("Shift % of days")
    ax.set_title("Phase 0: Shift Label Frequency by Variant and Horizon")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase0_label_distribution.png"), dpi=120)
    plt.close()

    # --- Plot 2: F1 per label_variant × H ---
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(valid))
    ax.bar(x, valid["avg_f1"], color="steelblue", alpha=0.8,
           yerr=valid["std_f1"], capsize=3, label="Model F1")
    # Overlay baseline lines (average across rows for visibility)
    for col, color, label in [
        ("b1_always0_f1",   "black",  "B1: Always 0"),
        ("b2_always1_f1",   "gray",   "B2: Always 1"),
        ("b3_stratified_f1","purple", "B3: Stratified"),
        ("b4_yesterday_f1", "orange", "B4: Yesterday"),
        ("b5_proximity_f1", "red",    "B5: Proximity"),
        ("b6_tenure_f1",    "brown",  "B6: Tenure"),
    ]:
        if col in valid.columns:
            ax.scatter(x, valid[col], color=color, zorder=5,
                       s=20, label=label, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['label_variant']}\nH={r['H']}" for _, r in valid.iterrows()],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_ylabel("F1")
    ax.set_title("Phase 0: Model F1 vs Naive Baselines by Label Variant")
    ax.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase0_label_comparison.png"), dpi=120)
    plt.close()
    print("  Phase 0 outputs saved.")


# ===========================================================================
# SECTION 9: Phase 1 — Feature Ablation
# ===========================================================================

def run_phase1(features_df, regime, tenure, vix,
               best_label_variant, best_min_persistence):
    """
    Phase 1: Ablation over feature sets A-G × H in {5, 7}.
    Uses best label from Phase 0. Model: LR (C=0.001, balanced).
    Also collects LR coefficients for feature importance plots (sets C, E).
    """
    print("\n" + "=" * 60)
    print("  PHASE 1: Feature Ablation")
    print("=" * 60)
    phase_start = time.time()

    feature_sets = get_feature_sets()

    def baseline_lr(n_neg, n_pos):
        return LogisticRegression(
            C=0.001, class_weight="balanced",
            solver="saga", max_iter=2000, random_state=RANDOM_SEED,
        )

    rows         = []
    coef_records = {}  # for sets C and E: {set_name: {feature: avg_abs_coef}}

    for H in HORIZONS_PRIMARY:
        ds = build_dataset(features_df, regime, tenure, vix,
                           H, best_min_persistence)
        shift_pct = float(ds["shift_label"].mean() * 100)

        for sname, fcols in feature_sets.items():
            # Check all features are in dataset
            available = [c for c in fcols if c in ds.columns]
            if len(available) < len(fcols):
                missing = set(fcols) - set(available)
                print(f"  WARN: Set {sname} missing features: {missing}")
            if not available:
                continue

            t0 = time.time()
            print(f"\n  [Set {sname}  H={H}]  "
                  f"features={available}  shift_pct={shift_pct:.1f}%")

            records, y_true, y_prob, cm_agg = run_walk_forward_basic(
                ds, available, baseline_lr,
                compute_baselines=True,
                phase_label=f"S{sname}/H{H}",
            )
            if not records:
                continue

            agg   = summarise_records(records)
            df_r  = pd.DataFrame(records)
            bl_agg = {c: df_r[c].mean()
                      for c in df_r.columns
                      if c.startswith("b") and df_r[c].dtype != object}

            row = {
                "feature_set":    sname,
                "H":              H,
                "n_features":     len(available),
                "features":       ",".join(available),
                "shift_pct":      shift_pct,
                "avg_recall":     agg.get("avg_test_recall", 0),
                "avg_precision":  agg.get("avg_test_precision", 0),
                "avg_f1":         agg.get("avg_test_f1", 0),
                "std_f1":         agg.get("std_test_f1", 0),
                "avg_pr_auc":     agg.get("avg_test_pr_auc", 0),
                "avg_far":        agg.get("avg_test_far", 0),
                **bl_agg,
            }
            rows.append(row)
            elapsed = time.time() - t0
            print(f"    --> avg_f1={row['avg_f1']:.3f}  "
                  f"std={row['std_f1']:.3f}  [{elapsed:.0f}s]")

            # Collect LR coefficients for sets C and E
            if sname in ("C", "E"):
                coef_dict = {f: [] for f in available}
                dates_all = ds.index
                for fold_i, test_year in enumerate(TEST_YEARS):
                    tr_m = dates_all.year <= (test_year - 1)
                    if tr_m.sum() == 0:
                        continue
                    X_tr = ds[tr_m][available].values.astype(float)
                    y_tr = ds[tr_m]["shift_label"].values.astype(int)
                    kept_idx, kept_names, _ = drop_correlated_features(
                        X_tr, available, y_tr)
                    X_tr_k = X_tr[:, kept_idx]
                    sc = StandardScaler()
                    X_tr_s = sc.fit_transform(X_tr_k)
                    m = LogisticRegression(
                        C=0.001, class_weight="balanced",
                        solver="saga", max_iter=2000,
                        random_state=RANDOM_SEED,
                    ).fit(X_tr_s, y_tr)
                    for fi, fname in enumerate(kept_names):
                        if fi < len(m.coef_[0]):
                            coef_dict[fname].append(abs(m.coef_[0][fi]))
                coef_records[f"{sname}_H{H}"] = {
                    f: np.mean(v) for f, v in coef_dict.items() if v
                }

    # Select best feature set: highest avg_f1 at H in {5, 7}
    if not rows:
        return rows, "G", coef_records
    df_rows = pd.DataFrame(rows)
    candidates = df_rows[df_rows["H"].isin(HORIZONS_PRIMARY)]
    best_idx  = candidates["avg_f1"].idxmax()
    best_set  = df_rows.loc[best_idx, "feature_set"]
    best_f1   = df_rows.loc[best_idx, "avg_f1"]

    print(f"\nBest feature set: [{best_set}]  avg_f1={best_f1:.3f}")
    print(f"Phase 1 completed in {(time.time()-phase_start)/60:.1f} min")
    return rows, best_set, coef_records


def save_phase1_outputs(rows, coef_records, out_dir):
    """Save Phase 1 CSV and plots."""
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "phase1_feature_ablation.csv"), index=False)

    # --- Plot 1: F1 per feature set grouped by H ---
    fig, ax = plt.subplots(figsize=(12, 5))
    sets  = df["feature_set"].unique()
    H_vals = sorted(df["H"].unique())
    x     = np.arange(len(sets))
    width = 0.35
    colors = ["steelblue", "darkorange"]
    for hi, H in enumerate(H_vals):
        sub = df[df["H"] == H].set_index("feature_set")
        vals = [sub.loc[s, "avg_f1"] if s in sub.index else 0 for s in sets]
        errs = [sub.loc[s, "std_f1"] if s in sub.index else 0 for s in sets]
        ax.bar(x + (hi - 0.5) * width, vals, width,
               label=f"H={H}", color=colors[hi % len(colors)],
               alpha=0.8, yerr=errs, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Set {s}" for s in sets])
    ax.set_ylabel("Average F1 (12 folds)")
    ax.set_title("Phase 1: Feature Ablation — F1 by Feature Set")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase1_feature_ablation.png"), dpi=120)
    plt.close()

    # --- Plot 2: LR coefficients for sets C and E ---
    fig, axes = plt.subplots(1, max(len(coef_records), 1),
                             figsize=(max(len(coef_records), 1) * 7, 5))
    if len(coef_records) == 1:
        axes = [axes]
    for ax, (key, coefs) in zip(axes, coef_records.items()):
        names = list(coefs.keys())
        vals  = [coefs[n] for n in names]
        idx   = np.argsort(vals)[::-1]
        ax.barh([names[i] for i in idx], [vals[i] for i in idx],
                color="steelblue", alpha=0.8)
        ax.set_xlabel("Avg |Coefficient| across 12 folds")
        ax.set_title(f"LR Feature Importance: Set {key}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase1_feature_importance.png"), dpi=120)
    plt.close()
    print("  Phase 1 outputs saved.")


# ===========================================================================
# SECTION 10: Phase 2 — Model Optimization
# ===========================================================================

def run_phase2(features_df, regime, tenure, vix,
               best_label_variant, best_min_persistence,
               best_feature_set):
    """
    Phase 2: Model optimization.
    Models: LR (Pipeline + SelectKBest + GridSearchCV), XGB (RandomizedSearch),
            RF (GridSearchCV, optional — skipped if runtime > 2hr).
    Uses best label from Phase 0 and best feature set from Phase 1.
    Evaluates H=5 and H=7 (best from Phase 0).
    """
    print("\n" + "=" * 60)
    print("  PHASE 2: Model Optimization")
    print("=" * 60)
    phase_start = time.time()

    feature_sets = get_feature_sets()
    fcols        = feature_sets[best_feature_set]

    all_model_results   = {}
    all_model_probas    = {}
    all_model_cms       = {}
    all_model_best_params = {}
    overfitting_records = []

    for H in HORIZONS_PRIMARY:
        print(f"\n  --- H={H} ---")
        ds        = build_dataset(features_df, regime, tenure, vix,
                                  H, best_min_persistence)
        available = [c for c in fcols if c in ds.columns]

        # --- LR with Pipeline + SelectKBest ---
        print(f"\n  LR (Pipeline + SelectKBest)  H={H}")

        def get_lr_model(n_neg, n_pos):
            return Pipeline([
                ("select", SelectKBest(f_classif)),
                ("model",  LogisticRegression(
                    solver="saga", max_iter=2000, random_state=RANDOM_SEED)),
            ])

        def get_lr_grid(n_neg, n_pos, n_feats):
            valid_k = sorted(set(
                [k for k in [3, 5] if k < n_feats] + ["all"]
            ))
            return {
                "select__k": valid_k,
                "model__C":  [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [
                    "balanced",
                    {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3},
                    {0: 1, 1: 5}, {0: 1, 1: 10},
                    {0: 1, 1: 15}, {0: 1, 1: 20},
                ],
                "model__penalty": ["l1", "l2"],
            }

        (lr_records, lr_ytrue, lr_yprob,
         lr_cm, lr_params, lr_ests) = run_walk_forward_cv(
            ds, available, get_lr_model, get_lr_grid,
            use_randomized=False, phase_label=f"LR/H{H}",
        )
        key = f"LR_H{H}"
        all_model_results[key]  = lr_records
        all_model_probas[key]   = (lr_ytrue, lr_yprob)
        all_model_cms[key]      = lr_cm
        all_model_best_params[key] = select_representative_params(lr_params)
        for rec in lr_records:
            overfitting_records.append({
                "model": "LR", "H": H,
                **{k: rec[k] for k in ["fold", "test_year", "train_f1",
                                        "test_f1", "gap_f1"]},
            })

        # --- XGBoost ---
        if HAS_XGB:
            xgb_n_iter = XGB_N_ITER_REDUCED if runtime_exceeded() else XGB_N_ITER_FULL
            print(f"\n  XGBoost (n_iter={xgb_n_iter})  H={H}")

            def get_xgb_model(n_neg, n_pos):
                return XGBClassifier(
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=RANDOM_SEED,
                    verbosity=0,
                )

            def get_xgb_grid(n_neg, n_pos, n_feats):
                base_ratio  = n_neg / max(n_pos, 1)
                spw_options = [base_ratio * m
                               for m in [0.5, 1.0, 2.0, 3.0, 5.0]]
                return {
                    "n_estimators":    [100, 200, 500],
                    "max_depth":       [3, 4, 6, 8],
                    "learning_rate":   [0.01, 0.05, 0.1],
                    "scale_pos_weight": spw_options,
                    "min_child_weight": [1, 3, 5],
                    "reg_alpha":       [0, 0.1, 1.0],
                    "reg_lambda":      [1.0, 5.0, 10.0],
                    "subsample":       [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                }

            (xgb_records, xgb_ytrue, xgb_yprob,
             xgb_cm, xgb_params, xgb_ests) = run_walk_forward_cv(
                ds, available, get_xgb_model, get_xgb_grid,
                use_randomized=True, n_iter=xgb_n_iter,
                phase_label=f"XGB/H{H}",
            )
            key = f"XGB_H{H}"
            all_model_results[key]  = xgb_records
            all_model_probas[key]   = (xgb_ytrue, xgb_yprob)
            all_model_cms[key]      = xgb_cm
            all_model_best_params[key] = select_representative_params(xgb_params)
            for rec in xgb_records:
                overfitting_records.append({
                    "model": "XGB", "H": H,
                    **{k: rec[k] for k in ["fold", "test_year", "train_f1",
                                            "test_f1", "gap_f1"]},
                })

        # --- RF (optional) ---
        if HAS_RF and not runtime_exceeded():
            print(f"\n  RandomForest  H={H}")

            def get_rf_model(n_neg, n_pos):
                return RandomForestClassifier(random_state=RANDOM_SEED)

            def get_rf_grid(n_neg, n_pos, n_feats):
                return {
                    "n_estimators": [100, 300],
                    "max_depth":    [5, 10, None],
                    "class_weight": [
                        "balanced",
                        {0: 1, 1: 5},
                        {0: 1, 1: 10},
                    ],
                    "min_samples_leaf": [1, 5],
                }

            (rf_records, rf_ytrue, rf_yprob,
             rf_cm, rf_params, rf_ests) = run_walk_forward_cv(
                ds, available, get_rf_model, get_rf_grid,
                use_randomized=False, phase_label=f"RF/H{H}",
            )
            key = f"RF_H{H}"
            all_model_results[key]  = rf_records
            all_model_probas[key]   = (rf_ytrue, rf_yprob)
            all_model_cms[key]      = rf_cm
            all_model_best_params[key] = select_representative_params(rf_params)
            for rec in rf_records:
                overfitting_records.append({
                    "model": "RF", "H": H,
                    **{k: rec[k] for k in ["fold", "test_year", "train_f1",
                                            "test_f1", "gap_f1"]},
                })
        else:
            if not HAS_RF:
                print("  RF skipped: not installed.")
            else:
                print("  RF skipped: runtime exceeded 2hr limit.")

    # Overfitting diagnostics flags
    print("\n  --- Overfitting Diagnostics ---")
    ov_df = pd.DataFrame(overfitting_records)
    for model_name in ov_df["model"].unique():
        for H in HORIZONS_PRIMARY:
            sub = ov_df[(ov_df["model"] == model_name) & (ov_df["H"] == H)]
            if sub.empty:
                continue
            avg_gap  = sub["gap_f1"].mean()
            std_f1   = sub["test_f1"].std()
            min_f1   = sub["test_f1"].min()
            flags = []
            if avg_gap > 0.20:
                flags.append("OVERFITTING_FLAG(avg_gap>0.20)")
            if std_f1 > 0.20:
                flags.append("UNSTABLE_FLAG(std_f1>0.20)")
            if min_f1 < 0.20:
                flags.append("LOW_FOLD_WARN(min_f1<0.20)")
            flag_str = ", ".join(flags) if flags else "OK"
            print(f"    {model_name} H={H}: avg_gap={avg_gap:.3f}  "
                  f"std_f1={std_f1:.3f}  min_f1={min_f1:.3f}  [{flag_str}]")

    # Select best model: highest avg_f1 across H=5/7
    best_model_name = None
    best_avg_f1     = -1.0
    best_H2         = HORIZONS_PRIMARY[0]
    for key, records in all_model_results.items():
        df_r = pd.DataFrame(records)
        avg  = df_r["test_f1"].mean()
        print(f"    {key}: avg_test_f1={avg:.3f}")
        if avg > best_avg_f1:
            best_avg_f1    = avg
            best_model_name = key.split("_")[0]
            best_H2        = int(key.split("H")[1])

    best_model_key = f"{best_model_name}_H{best_H2}"
    best_phase2_params = all_model_best_params.get(best_model_key, {})

    print(f"\nBest model: [{best_model_name}] at H={best_H2}  avg_f1={best_avg_f1:.3f}")
    print(f"Phase 2 completed in {(time.time()-phase_start)/60:.1f} min")

    return (all_model_results, all_model_probas, all_model_cms,
            overfitting_records, best_model_name, best_H2, best_phase2_params)


def save_phase2_outputs(all_model_results, all_model_probas,
                        all_model_cms, overfitting_records, out_dir):
    """Save Phase 2 CSVs and plots."""
    # --- Model results CSV ---
    all_rows = []
    for key, records in all_model_results.items():
        for rec in records:
            row = {"model_key": key}
            row.update(rec)
            all_rows.append(row)
    pd.DataFrame(all_rows).to_csv(
        os.path.join(out_dir, "phase2_model_results.csv"), index=False)

    # --- Overfitting diagnostics CSV ---
    pd.DataFrame(overfitting_records).to_csv(
        os.path.join(out_dir, "phase2_overfitting_diagnostics.csv"), index=False)

    # --- Plot: Model comparison bar chart ---
    summary = []
    for key, records in all_model_results.items():
        df_r = pd.DataFrame(records)
        summary.append({
            "model":    key,
            "recall":   df_r["test_recall"].mean(),
            "precision":df_r["test_precision"].mean(),
            "f1":       df_r["test_f1"].mean(),
            "pr_auc":   df_r["test_pr_auc"].mean(),
        })
    df_sum = pd.DataFrame(summary)

    metrics = ["recall", "precision", "f1", "pr_auc"]
    x = np.arange(len(df_sum))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(12, len(df_sum) * 3), 5))
    for mi, metric in enumerate(metrics):
        ax.bar(x + mi * width, df_sum[metric], width, label=metric.upper(), alpha=0.8)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df_sum["model"], rotation=30, ha="right")
    ax.set_ylabel("Score (avg across 12 folds)")
    ax.set_title("Phase 2: Model Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase2_model_comparison.png"), dpi=120)
    plt.close()

    # --- Confusion matrices (1 row × N models) ---
    n_models = len(all_model_cms)
    if n_models > 0:
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        if n_models == 1:
            axes = [axes]
        for ax, (key, cm) in zip(axes, all_model_cms.items()):
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Pred 0", "Pred 1"],
                        yticklabels=["True 0", "True 1"], ax=ax)
            ax.set_title(f"{key}\n(aggregated, 12 folds)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "phase2_confusion_matrices.png"), dpi=120)
        plt.close()

    # --- ROC curves ---
    fig, ax = plt.subplots(figsize=(7, 6))
    for key, (y_true, y_prob) in all_model_probas.items():
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_val  = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{key} (AUC={roc_auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Phase 2: ROC Curves (aggregated, 12 folds)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase2_roc_curves.png"), dpi=120)
    plt.close()
    print("  Phase 2 outputs saved.")


# ===========================================================================
# SECTION 11: Phase 3 — Horizon Sweep
# ===========================================================================

def run_phase3(features_df, regime, tenure, vix,
               best_label_variant, best_min_persistence,
               best_feature_set, best_model_name,
               phase2_best_params=None):
    """
    Phase 3: Sweep H in {3, 5, 7, 10} with the best model from Phase 2.
    Recomputes labels for each H. Uses optimal threshold per fold.
    """
    print("\n" + "=" * 60)
    print("  PHASE 3: Horizon Sweep")
    print("=" * 60)
    phase_start = time.time()

    feature_sets = get_feature_sets()
    fcols        = feature_sets[best_feature_set]
    phase2_best_params = dict(phase2_best_params or {})

    rows     = []
    cm_per_H = {}

    for H in HORIZONS_ALL:
        t0 = time.time()
        ds = build_dataset(features_df, regime, tenure, vix,
                           H, best_min_persistence)
        available  = [c for c in fcols if c in ds.columns]
        shift_pct  = float(ds["shift_label"].mean() * 100)
        print(f"\n  [H={H}]  shift_pct={shift_pct:.1f}%")

        if best_model_name == "LR":
            def get_model(n_neg, n_pos):
                return Pipeline([
                    ("select", SelectKBest(f_classif)),
                    ("model",  LogisticRegression(
                        solver="saga", max_iter=2000,
                        random_state=RANDOM_SEED)),
                ])
            def get_grid(n_neg, n_pos, n_feats):
                if phase2_best_params:
                    fixed_params = dict(phase2_best_params)
                    k = fixed_params.get("select__k", "all")
                    if k != "all" and isinstance(k, (int, np.integer)) and k >= n_feats:
                        fixed_params["select__k"] = "all"
                    return {k: [v] for k, v in fixed_params.items()}
                valid_k = sorted(set(
                    [k for k in [3, 5] if k < n_feats] + ["all"]))
                return {
                    "select__k": valid_k,
                    "model__C":  [0.001, 0.01, 0.1, 1.0],
                    "model__class_weight": [
                        "balanced", {0:1,1:5}, {0:1,1:10}],
                    "model__penalty": ["l2"],
                }
            use_rand = False
            n_iter   = 50
        elif best_model_name == "XGB" and HAS_XGB:
            def get_model(n_neg, n_pos):
                return XGBClassifier(
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=RANDOM_SEED, verbosity=0)
            def get_grid(n_neg, n_pos, n_feats):
                if phase2_best_params:
                    return {k: [v] for k, v in phase2_best_params.items()}
                r = n_neg / max(n_pos, 1)
                return {
                    "n_estimators":  [100, 200],
                    "max_depth":     [3, 4, 6],
                    "learning_rate": [0.05, 0.1],
                    "scale_pos_weight": [r * m for m in [1.0, 2.0, 5.0]],
                    "subsample":     [0.8, 1.0],
                }
            use_rand = True
            n_iter   = 50
        elif best_model_name == "RF" and HAS_RF:
            def get_model(n_neg, n_pos):
                return RandomForestClassifier(random_state=RANDOM_SEED)
            def get_grid(n_neg, n_pos, n_feats):
                if phase2_best_params:
                    return {k: [v] for k, v in phase2_best_params.items()}
                return {
                    "n_estimators": [100, 300],
                    "max_depth":    [5, 10, None],
                    "class_weight": [
                        "balanced",
                        {0: 1, 1: 5},
                        {0: 1, 1: 10},
                    ],
                    "min_samples_leaf": [1, 5],
                }
            use_rand = False
            n_iter   = 50
        else:
            # Fallback to simple LR
            def get_model(n_neg, n_pos):
                return LogisticRegression(
                    C=0.01, class_weight="balanced",
                    solver="saga", max_iter=2000,
                    random_state=RANDOM_SEED)
            def get_grid(n_neg, n_pos, n_feats):
                return {
                    "C": [0.001, 0.01, 0.1],
                    "class_weight": ["balanced", {0:1,1:5}],
                }
            use_rand = False
            n_iter   = 20

        (records, y_true, y_prob,
         cm, params, ests) = run_walk_forward_cv(
            ds, available, get_model, get_grid,
            use_randomized=use_rand, n_iter=n_iter,
            phase_label=f"{best_model_name}/H{H}",
        )
        if not records:
            continue

        df_r   = pd.DataFrame(records)
        std_f1 = df_r["test_f1"].std()
        flag   = " [FLAG: std>0.15]" if std_f1 > 0.15 else ""

        # Aggregate baselines
        bl_cols = [c for c in df_r.columns if c.startswith("b")
                   and df_r[c].dtype != object]
        bl_agg  = {c: df_r[c].mean() for c in bl_cols}

        row = {
            "H":              H,
            "shift_pct":      shift_pct,
            "avg_recall":     df_r["test_recall"].mean(),
            "avg_precision":  df_r["test_precision"].mean(),
            "avg_f1":         df_r["test_f1"].mean(),
            "std_f1":         std_f1,
            "avg_pr_auc":     df_r["test_pr_auc"].mean(),
            "avg_far":        df_r["test_far"].mean(),
            "avg_f1_opt":     df_r["test_f1_opt"].mean() if "test_f1_opt" in df_r else 0,
            "avg_opt_threshold": df_r["opt_threshold"].mean() if "opt_threshold" in df_r else 0.5,
            **bl_agg,
        }
        rows.append(row)
        cm_per_H[H] = cm
        elapsed = time.time() - t0
        print(f"    H={H}: avg_f1={row['avg_f1']:.3f}  "
              f"f1_opt={row['avg_f1_opt']:.3f}  "
              f"std={std_f1:.3f}{flag}  [{elapsed:.0f}s]")

    # Best H: highest avg_f1 with shift_pct in [15,45] and std_f1 < 0.20
    if not rows:
        return rows, cm_per_H, HORIZONS_PRIMARY[0]
    df_rows = pd.DataFrame(rows)
    cands   = df_rows[(df_rows["shift_pct"].between(15, 45))
                      & (df_rows["std_f1"] < 0.20)]
    if cands.empty:
        cands = df_rows
    best_H3 = int(cands.loc[cands["avg_f1"].idxmax(), "H"])

    print(f"\nBest H for Phase 3: H={best_H3}")
    print(f"Phase 3 completed in {(time.time()-phase_start)/60:.1f} min")
    return rows, cm_per_H, best_H3


def save_phase3_outputs(rows, cm_per_H, out_dir):
    """Save Phase 3 CSV and plots."""
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "phase3_horizon_sweep.csv"), index=False)

    # --- Line plot: F1 and PR-AUC vs H ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["H"], df["avg_f1"],     "b-o", lw=2, label="F1 (thr=0.5)")
    ax.plot(df["H"], df["avg_f1_opt"], "b--s", lw=2, label="F1 (opt thr)")
    ax.plot(df["H"], df["avg_pr_auc"], "r-^", lw=2, label="PR-AUC")
    ax.set_xlabel("Horizon H (days)")
    ax.set_ylabel("Score (avg 12 folds)")
    ax.set_title("Phase 3: Horizon Sweep")
    ax.set_xticks(df["H"].tolist())
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase3_horizon_sweep.png"), dpi=120)
    plt.close()

    # --- Confusion matrices per H (1 row × 4 cols) ---
    H_vals = sorted(cm_per_H.keys())
    n_H    = len(H_vals)
    if n_H > 0:
        fig, axes = plt.subplots(1, n_H, figsize=(5 * n_H, 4))
        if n_H == 1:
            axes = [axes]
        for ax, H in zip(axes, H_vals):
            cm = cm_per_H[H]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Pred 0", "Pred 1"],
                        yticklabels=["True 0", "True 1"], ax=ax)
            ax.set_title(f"H={H} (aggregated)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "phase3_confusion_matrices_per_H.png"),
                    dpi=120)
        plt.close()
    print("  Phase 3 outputs saved.")


# ===========================================================================
# SECTION 12: Phase 4 — Final Model Report
# ===========================================================================

def run_phase4(features_df, regime, tenure, vix,
               best_label_variant, best_min_persistence,
               best_feature_set, best_model_name, best_H,
               phase2_best_params,
               phase0_rows, phase1_rows):
    """
    Phase 4: Final model — one full walk-forward with complete per-fold logging,
    all plots, and summary CSV.
    """
    print("\n" + "=" * 60)
    print("  PHASE 4: Final Model Report")
    print("=" * 60)
    phase_start = time.time()

    feature_sets = get_feature_sets()
    fcols        = feature_sets[best_feature_set]
    phase2_best_params = dict(phase2_best_params or {})
    ds           = build_dataset(features_df, regime, tenure, vix,
                                 best_H, best_min_persistence)
    available    = [c for c in fcols if c in ds.columns]
    shift_pct    = float(ds["shift_label"].mean() * 100)

    print(f"\n  Config: label={best_label_variant}(persist={best_min_persistence})"
          f"  H={best_H}  features=Set{best_feature_set}({len(available)})"
          f"  model={best_model_name}")
    print(f"  shift_pct={shift_pct:.1f}%  n={len(ds)}")

    if best_model_name == "LR":
        def get_model(n_neg, n_pos):
            return Pipeline([
                ("select", SelectKBest(f_classif)),
                ("model",  LogisticRegression(
                    solver="saga", max_iter=2000, random_state=RANDOM_SEED)),
            ])
        def get_grid(n_neg, n_pos, n_feats):
            if phase2_best_params:
                fixed_params = dict(phase2_best_params)
                k = fixed_params.get("select__k", "all")
                if k != "all" and isinstance(k, (int, np.integer)) and k >= n_feats:
                    fixed_params["select__k"] = "all"
                return {k: [v] for k, v in fixed_params.items()}
            valid_k = sorted(set(
                [k for k in [3, 5] if k < n_feats] + ["all"]))
            return {
                "select__k": valid_k,
                "model__C":  [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [
                    "balanced", {0:1,1:1}, {0:1,1:2}, {0:1,1:3},
                    {0:1,1:5}, {0:1,1:10}, {0:1,1:15}, {0:1,1:20},
                ],
                "model__penalty": ["l1", "l2"],
            }
        use_rand = False
        n_iter_p4 = 150
    elif best_model_name == "XGB" and HAS_XGB:
        def get_model(n_neg, n_pos):
            return XGBClassifier(
                use_label_encoder=False, eval_metric="logloss",
                random_state=RANDOM_SEED, verbosity=0)
        def get_grid(n_neg, n_pos, n_feats):
            if phase2_best_params:
                return {k: [v] for k, v in phase2_best_params.items()}
            r = n_neg / max(n_pos, 1)
            return {
                "n_estimators":    [100, 200, 500],
                "max_depth":       [3, 4, 6, 8],
                "learning_rate":   [0.01, 0.05, 0.1],
                "scale_pos_weight": [r * m for m in [0.5, 1.0, 2.0, 3.0, 5.0]],
                "min_child_weight": [1, 3, 5],
                "reg_alpha":       [0, 0.1, 1.0],
                "reg_lambda":      [1.0, 5.0, 10.0],
                "subsample":       [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            }
        use_rand     = True
        n_iter_p4    = XGB_N_ITER_REDUCED if runtime_exceeded() else XGB_N_ITER_FULL
    elif best_model_name == "RF" and HAS_RF:
        def get_model(n_neg, n_pos):
            return RandomForestClassifier(random_state=RANDOM_SEED)
        def get_grid(n_neg, n_pos, n_feats):
            if phase2_best_params:
                return {k: [v] for k, v in phase2_best_params.items()}
            return {
                "n_estimators": [100, 300],
                "max_depth":    [5, 10, None],
                "class_weight": [
                    "balanced", {0:1,1:5}, {0:1,1:10},
                ],
                "min_samples_leaf": [1, 5],
            }
        use_rand  = False
        n_iter_p4 = 150
    else:
        def get_model(n_neg, n_pos):
            return LogisticRegression(
                C=0.01, class_weight="balanced",
                solver="saga", max_iter=2000, random_state=RANDOM_SEED)
        def get_grid(n_neg, n_pos, n_feats):
            return {"C": [0.001, 0.01, 0.1],
                    "class_weight": ["balanced", {0:1,1:5}]}
        use_rand  = False
        n_iter_p4 = 20

    (records, y_all_true, y_all_prob,
     cm_agg, fold_params, fold_ests) = run_walk_forward_cv(
        ds, available, get_model, get_grid,
        use_randomized=use_rand, n_iter=n_iter_p4,
        phase_label=f"FINAL/{best_model_name}",
    )

    if not records:
        print("  ERROR: No fold results in Phase 4.")
        return {}, records

    df_r = pd.DataFrame(records)

    # Add shift count cols from metrics
    avg_recall    = df_r["test_recall"].mean()
    std_recall    = df_r["test_recall"].std()
    avg_precision = df_r["test_precision"].mean()
    std_precision = df_r["test_precision"].std()
    avg_f1        = df_r["test_f1"].mean()
    std_f1        = df_r["test_f1"].std()
    avg_pr_auc    = df_r["test_pr_auc"].mean()
    avg_accuracy  = df_r["test_accuracy"].mean()
    avg_far       = df_r["test_far"].mean()
    avg_gap       = df_r["gap_f1"].mean()
    max_gap       = df_r["gap_f1"].max()
    min_fold_f1   = df_r["test_f1"].min()
    avg_opt_thr   = df_r["opt_threshold"].mean() if "opt_threshold" in df_r else 0.5

    # Baseline averages
    bl_avg = {c: df_r[c].mean() for c in df_r.columns
              if c.startswith("b") and df_r[c].dtype != object}

    # Improvement vs best heuristic baseline
    best_heuristic_f1 = max(
        bl_avg.get("b4_yesterday_f1", 0),
        bl_avg.get("b5_proximity_f1", 0),
        bl_avg.get("b6_tenure_f1",    0),
    )
    improvement_vs_naive = avg_f1 - best_heuristic_f1

    # Progression data from prior phases
    phase0_any_h5 = next(
        (r["avg_f1"] for r in phase0_rows
         if r.get("label_variant") == "any_shift"
         and r.get("H") == best_H
         and not r.get("skipped")), None)
    phase1_setA   = next(
        (r["avg_f1"] for r in phase1_rows
         if r.get("feature_set") == "A"
         and r.get("H") == best_H), None)

    # Summary config
    best_params_list = [json.dumps(p, default=str) for p in fold_params]
    most_common_params = max(set(best_params_list),
                             key=best_params_list.count) if best_params_list else "{}"

    print(f"\n  avg_f1={avg_f1:.3f}  std={std_f1:.3f}  "
          f"avg_gap={avg_gap:.3f}  min_fold_f1={min_fold_f1:.3f}")
    print(f"  avg_recall={avg_recall:.3f}  avg_precision={avg_precision:.3f}")
    print(f"  avg_far={avg_far:.3f}  avg_pr_auc={avg_pr_auc:.3f}")

    # Overfitting flags
    of_flags = []
    if avg_gap    > 0.20: of_flags.append("avg_gap>0.20")
    if std_f1     > 0.15: of_flags.append("std_f1>0.15")
    if min_fold_f1 < 0.20: of_flags.append("min_f1<0.20")

    summary = {
        "label_variant":       best_label_variant,
        "min_persistence":     best_min_persistence,
        "H":                   best_H,
        "feature_set":         best_feature_set,
        "n_features":          len(available),
        "features":            ",".join(available),
        "model":               best_model_name,
        "best_params":         most_common_params,
        "avg_threshold":       avg_opt_thr,
        "avg_recall":          avg_recall,
        "std_recall":          std_recall,
        "avg_precision":       avg_precision,
        "std_precision":       std_precision,
        "avg_f1":              avg_f1,
        "std_f1":              std_f1,
        "avg_pr_auc":          avg_pr_auc,
        "avg_accuracy":        avg_accuracy,
        "avg_far":             avg_far,
        "naive_always0_f1":    bl_avg.get("b1_always0_f1",    0),
        "naive_always1_f1":    bl_avg.get("b2_always1_f1",    0),
        "naive_stratified_f1": bl_avg.get("b3_stratified_f1", 0),
        "heuristic_yesterday_f1": bl_avg.get("b4_yesterday_f1", 0),
        "heuristic_proximity_f1": bl_avg.get("b5_proximity_f1", 0),
        "heuristic_tenure_f1":    bl_avg.get("b6_tenure_f1",    0),
        "improvement_vs_best_heuristic": improvement_vs_naive,
        "improvement_vs_plan10_LR": avg_f1 - PLAN10_LR_F1,
        "avg_train_test_gap":  avg_gap,
        "max_fold_gap":        max_gap,
        "min_fold_f1":         min_fold_f1,
        "overfitting_flags":   ",".join(of_flags) if of_flags else "none",
        "phase0_any_shift_f1": phase0_any_h5 if phase0_any_h5 is not None else 0,
        "phase1_setA_f1":      phase1_setA  if phase1_setA  is not None else 0,
    }

    # Median-F1 fold for timeline plot
    f1_by_fold = df_r["test_f1"].values
    median_fold_idx = int(np.argsort(np.abs(f1_by_fold - np.median(f1_by_fold)))[0])
    median_test_year = int(df_r.iloc[median_fold_idx]["test_year"])

    # --- Learning curve on fold 6 (middle fold, ~3000 training days) ---
    fold6_year = TEST_YEARS[5]  # index 5 = test year 2018
    dates_all  = ds.index
    tr6_m = dates_all.year <= (fold6_year - 1)
    X_tr6 = ds[tr6_m][available].values.astype(float)
    y_tr6 = ds[tr6_m]["shift_label"].values.astype(int)

    kept_idx6, _, _ = drop_correlated_features(X_tr6, available, y_tr6)
    X_tr6_k  = X_tr6[:, kept_idx6]
    sc6      = StandardScaler()
    X_tr6_s  = sc6.fit_transform(X_tr6_k)

    # Use fold 6's best estimator if available, else fallback to simple LR
    lc_estimator = None
    if len(fold_ests) > 5:
        lc_est_tuple = fold_ests[5]  # (best_estimator, scaler, kept_idx)
        # Rebuild with fold-6 data for learning curve (need unfitted clone)
        try:
            from sklearn.base import clone
            lc_estimator = clone(lc_est_tuple[0])
        except Exception:
            lc_estimator = None

    if lc_estimator is None:
        lc_estimator = LogisticRegression(
            C=0.01, class_weight="balanced",
            solver="saga", max_iter=2000, random_state=RANDOM_SEED)

    print(f"\n  Computing learning curve on fold 6 (test={fold6_year})  "
          f"n_train={len(y_tr6)} ...")
    try:
        lc_train_sizes, lc_train_scores, lc_test_scores = learning_curve(
            lc_estimator, X_tr6_s, y_tr6,
            train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
            cv=3, scoring="f1", random_state=RANDOM_SEED,
        )
    except Exception as e:
        print(f"  Learning curve failed: {e}")
        lc_train_sizes  = np.array([])
        lc_train_scores = np.array([])
        lc_test_scores  = np.array([])

    print(f"\nPhase 4 completed in {(time.time()-phase_start)/60:.1f} min")
    return summary, records, y_all_true, y_all_prob, cm_agg, \
           median_test_year, lc_train_sizes, lc_train_scores, lc_test_scores, ds


def save_phase4_outputs(summary, records, y_all_true, y_all_prob, cm_agg,
                        median_test_year, lc_train_sizes, lc_train_scores,
                        lc_test_scores, ds, best_H, out_dir):
    """Save all Phase 4 outputs: CSV, plots."""
    df_r = pd.DataFrame(records)

    # --- full_summary.csv ---
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, "full_summary.csv"), index=False)
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, "phase4_final_model.csv"), index=False)

    # --- Aggregated confusion matrix ---
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_agg, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"], ax=ax)
    ax.set_title("Final Model: Confusion Matrix\n(aggregated, 12 folds)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase4_confusion_matrix.png"), dpi=120)
    plt.close()

    # --- Precision-Recall curve with operating point ---
    fig, ax = plt.subplots(figsize=(7, 6))
    if len(np.unique(y_all_true)) >= 2:
        prec, rec, thr = precision_recall_curve(y_all_true, y_all_prob)
        ax.plot(rec, prec, "b-", lw=2, label="PR Curve")
        # Mark operating point (avg_threshold)
        op_thr  = summary["avg_threshold"]
        yp_op   = (np.array(y_all_prob) >= op_thr).astype(int)
        op_prec = precision_score(y_all_true, yp_op, pos_label=1, zero_division=0)
        op_rec  = recall_score(   y_all_true, yp_op, pos_label=1, zero_division=0)
        ax.scatter([op_rec], [op_prec], color="red", zorder=10, s=100,
                   label=f"Op. point (thr={op_thr:.3f})")
        no_skill = np.mean(y_all_true)
        ax.axhline(no_skill, color="gray", ls="--", lw=1,
                   label=f"No-skill ({no_skill:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Final Model: Precision-Recall Curve")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase4_precision_recall_curve.png"), dpi=120)
    plt.close()

    # --- Per-fold F1 barplot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    folds   = df_r["test_year"].values
    f1_vals = df_r["test_f1"].values
    mean_f1 = f1_vals.mean()
    std_f1  = f1_vals.std()
    colors  = ["steelblue" if v >= mean_f1 else "tomato" for v in f1_vals]
    ax.bar(np.arange(len(folds)), f1_vals, color=colors, alpha=0.85)
    ax.axhline(mean_f1, color="navy",   lw=2, ls="-",  label=f"Mean={mean_f1:.3f}")
    ax.axhline(mean_f1 + std_f1, color="navy", lw=1, ls="--")
    ax.axhline(mean_f1 - std_f1, color="navy", lw=1, ls="--",
               label=f"+-1 std={std_f1:.3f}")
    ax.fill_between([-0.5, len(folds) - 0.5],
                    mean_f1 - std_f1, mean_f1 + std_f1,
                    alpha=0.15, color="navy")
    ax.set_xticks(np.arange(len(folds)))
    ax.set_xticklabels([str(y) for y in folds], rotation=45)
    ax.set_ylabel("F1")
    ax.set_title("Final Model: F1 per Fold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase4_per_fold_barplot.png"), dpi=120)
    plt.close()

    # --- Shift timeline for median-F1 fold ---
    te_m = ds.index.year == median_test_year
    df_te_plot = ds[te_m].copy()
    if len(df_te_plot) > 0:
        y_te_true  = df_te_plot["shift_label"].values.astype(int)
        # For the timeline, use the fold's predicted labels
        fold_row   = df_r[df_r["test_year"] == median_test_year]
        fold_thr   = float(fold_row["opt_threshold"].values[0]) \
                     if len(fold_row) > 0 and "opt_threshold" in fold_row.columns \
                     else 0.5

        fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                                 gridspec_kw={"height_ratios": [3, 1]})
        ax_top, ax_bot = axes

        # Top: VIX with threshold lines
        ax_top.plot(df_te_plot.index, df_te_plot["VIX"],
                    "k-", lw=1.2, label="VIX")
        for thr_val, col, lbl in [
            (15, "green",  "Calm/Normal"),
            (20, "orange", "Normal/Tense"),
            (30, "red",    "Tense/Crisis"),
        ]:
            ax_top.axhline(thr_val, color=col, ls="--", lw=1, alpha=0.7,
                           label=lbl)
        ax_top.set_ylabel("VIX")
        ax_top.set_title(f"Final Model: Shift Timeline — {median_test_year} "
                         f"(median-F1 fold)")
        ax_top.legend(fontsize=7, ncol=4)

        # Bottom: actual vs predicted shifts (scatter)
        ax_bot.set_xlim(df_te_plot.index[0], df_te_plot.index[-1])
        ax_bot.set_yticks([])
        true_dates = df_te_plot.index[y_te_true == 1]
        ax_bot.scatter(true_dates, np.ones(len(true_dates)) * 0.5,
                       color="green", marker="|", s=100, label="Actual shift")
        ax_bot.set_ylabel("Shifts")
        ax_bot.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "phase4_shift_timeline.png"), dpi=120)
        plt.close()

    # --- Learning curve ---
    if len(lc_train_sizes) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        lc_tr_mean = lc_train_scores.mean(axis=1)
        lc_te_mean = lc_test_scores.mean(axis=1)
        lc_tr_std  = lc_train_scores.std(axis=1)
        lc_te_std  = lc_test_scores.std(axis=1)
        ax.plot(lc_train_sizes, lc_tr_mean, "b-o", label="Train F1")
        ax.fill_between(lc_train_sizes,
                        lc_tr_mean - lc_tr_std, lc_tr_mean + lc_tr_std,
                        alpha=0.15, color="blue")
        ax.plot(lc_train_sizes, lc_te_mean, "r-o", label="CV F1")
        ax.fill_between(lc_train_sizes,
                        lc_te_mean - lc_te_std, lc_te_mean + lc_te_std,
                        alpha=0.15, color="red")
        ax.set_xlabel("Training set size")
        ax.set_ylabel("F1")
        ax.set_title("Final Model: Learning Curve (Fold 6)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "phase4_learning_curve.png"), dpi=120)
        plt.close()
    else:
        # Save empty placeholder
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Learning curve not available", ha="center", va="center")
        ax.set_title("Final Model: Learning Curve")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "phase4_learning_curve.png"), dpi=120)
        plt.close()

    print("  Phase 4 outputs saved.")


def print_final_summary(summary, records):
    """Print the comprehensive final summary table."""
    df_r = pd.DataFrame(records)
    print("\n" + "=" * 70)
    print("  PLAN 11 -- DEFINITIVE REGIME SHIFT CLASSIFIER -- FINAL RESULTS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Label:       persistent shift (min_persistence={summary['min_persistence']})")
    print(f"  Horizon:     H={summary['H']}")
    print(f"  Features:    Set {summary['feature_set']} ({summary['n_features']} features):")
    print(f"               {summary['features']}")
    print(f"  Model:       {summary['model']}")
    print(f"  Best Params: {summary['best_params'][:80]}")
    print(f"  Threshold:   {summary['avg_threshold']:.3f} (optimized via PR curve on train)")
    print(f"\nClass Distribution:")
    shift_pct = 0
    for rec in records:
        shift_pct += rec.get("shift_pct_test", 0)
    shift_pct /= max(len(records), 1)
    print(f"  Shift:     {shift_pct:.1f}% of days")
    print(f"  No-shift:  {100 - shift_pct:.1f}% of days")
    print(f"\nPerformance (avg +- std across 12 folds):")
    print(f"  Recall:     {summary['avg_recall']:.3f} +- {summary['std_recall']:.3f}")
    print(f"  Precision:  {summary['avg_precision']:.3f} +- {summary['std_precision']:.3f}")
    print(f"  F1:         {summary['avg_f1']:.3f} +- {summary['std_f1']:.3f}")
    print(f"  PR-AUC:     {summary['avg_pr_auc']:.3f}")
    print(f"  Accuracy:   {summary['avg_accuracy']:.3f}")
    print(f"  FAR:        {summary['avg_far']:.3f}")
    gap_ok  = "OK" if summary["avg_train_test_gap"] < 0.20 else "FLAG"
    std_ok  = "OK" if summary["std_f1"]              < 0.15 else "FLAG"
    minf_ok = "OK" if summary["min_fold_f1"]         > 0.30 else "FLAG"
    print(f"\nOverfitting Check:")
    print(f"  Avg train-test gap:  {summary['avg_train_test_gap']:.3f}  "
          f"(< 0.20 {gap_ok})")
    print(f"  Max single-fold gap: {summary['max_fold_gap']:.3f}")
    print(f"  Min single-fold F1:  {summary['min_fold_f1']:.3f}  "
          f"(> 0.30 {minf_ok})")
    print(f"  Std(F1):             {summary['std_f1']:.3f}  "
          f"(< 0.15 {std_ok})")
    print(f"\nNaive Baselines:")
    print(f"  Always predict 0:       F1 = {summary['naive_always0_f1']:.3f}")
    print(f"  Always predict 1:       F1 = {summary['naive_always1_f1']:.3f}")
    print(f"  Stratified random:      F1 = {summary['naive_stratified_f1']:.3f}")
    print(f"  Yesterday's shift:      F1 = {summary['heuristic_yesterday_f1']:.3f}")
    print(f"  Threshold proximity:    F1 = {summary['heuristic_proximity_f1']:.3f}")
    print(f"  Regime tenure:          F1 = {summary['heuristic_tenure_f1']:.3f}")
    print(f"  Model vs best heuristic: {summary['improvement_vs_best_heuristic']:+.3f} F1")
    print(f"\nComparison to Previous Plans:")
    print(f"  vs. Plan 10 LR (any_shift):  F1 {summary['improvement_vs_plan10_LR']:+.3f}")
    print(f"  Label fix gain:  {summary['avg_f1'] - summary['phase0_any_shift_f1']:+.3f} "
          f"(any->persist: {summary['phase0_any_shift_f1']:.3f}->{summary['avg_f1']:.3f})")
    print(f"  Feature fix gain: {summary['avg_f1'] - summary['phase1_setA_f1']:+.3f} "
          f"(setA->set{summary['feature_set']}: "
          f"{summary['phase1_setA_f1']:.3f}->{summary['avg_f1']:.3f})")
    print("\n" + "=" * 70)


# ===========================================================================
# SECTION 13: Main
# ===========================================================================

def main():
    total_start = time.time()
    print("=" * 70)
    print("  Plan 11 -- Definitive Regime Shift Classifier")
    print(f"  Output: {OUT_DIR}")
    print("=" * 70)

    # --- Load data ---
    print("\nLoading data ...")
    prices, vix = load_raw_data()
    regime  = compute_regime_labels(vix)
    tenure  = compute_regime_tenure(regime)
    print(f"  Prices : {len(prices)} days  "
          f"({prices.index[0].date()} to {prices.index[-1].date()})")
    print(f"  VIX    : {len(vix)} days")
    reg_vc = regime.value_counts().sort_index()
    for r, cnt in reg_vc.items():
        print(f"  Regime {r}: {cnt} days ({100*cnt/len(regime):.1f}%)")

    # --- Compute features ---
    print("\nComputing features ...")
    features_df = compute_all_features(prices, vix)
    print(f"  Feature matrix: {features_df.shape[1]} columns")

    # -------------------------------------------------------------------------
    # PHASE 0
    # -------------------------------------------------------------------------
    phase0_rows, best_label, best_persist, best_H0 = run_phase0(
        None, regime, tenure, vix, features_df
    )
    save_phase0_outputs(phase0_rows, OUT_DIR)

    # -------------------------------------------------------------------------
    # PHASE 1
    # -------------------------------------------------------------------------
    phase1_rows, best_feat_set, coef_records = run_phase1(
        features_df, regime, tenure, vix, best_label, best_persist
    )
    save_phase1_outputs(phase1_rows, coef_records, OUT_DIR)

    # -------------------------------------------------------------------------
    # PHASE 2
    # -------------------------------------------------------------------------
    if runtime_exceeded():
        print("\nWARNING: Runtime > 2hr before Phase 2. "
              "LR will run fully; XGB n_iter=100; RF skipped.")

    (p2_results, p2_probas, p2_cms,
     p2_ov, best_model, best_H2, p2_best_params) = run_phase2(
        features_df, regime, tenure, vix,
        best_label, best_persist, best_feat_set,
    )
    save_phase2_outputs(p2_results, p2_probas, p2_cms, p2_ov, OUT_DIR)

    # -------------------------------------------------------------------------
    # PHASE 3
    # -------------------------------------------------------------------------
    phase3_rows, cm_per_H, best_H3 = run_phase3(
        features_df, regime, tenure, vix,
        best_label, best_persist, best_feat_set, best_model, p2_best_params,
    )
    save_phase3_outputs(phase3_rows, cm_per_H, OUT_DIR)

    # Final H: from Phase 3 (uses best H from Phase 0 as fallback)
    final_H = best_H3

    # -------------------------------------------------------------------------
    # PHASE 4
    # -------------------------------------------------------------------------
    result = run_phase4(
        features_df, regime, tenure, vix,
        best_label, best_persist, best_feat_set, best_model, final_H,
        p2_best_params,
        phase0_rows, phase1_rows,
    )

    (summary, final_records, y_all_true, y_all_prob,
     cm_agg, median_year, lc_sizes, lc_tr, lc_te, final_ds) = result

    save_phase4_outputs(
        summary, final_records, y_all_true, y_all_prob, cm_agg,
        median_year, lc_sizes, lc_tr, lc_te, final_ds,
        final_H, OUT_DIR,
    )

    print_final_summary(summary, final_records)

    total_min = (time.time() - total_start) / 60
    print(f"\nTotal runtime: {total_min:.1f} min")
    print(f"All outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
