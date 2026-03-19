"""
07_regime_simplification.py -- Plan 10: Regime Definition Simplification + Full Weight Optimization

Steps:
  1. Compare 5 regime definitions (A-E) using LR baseline, H in {3,5,7,10}
  2. Full weight optimization on best regime definition (all models, full grids)
  3. Horizon sweep with best model + best regime definition
  4. Final model report

Walk-forward: 12 folds, test years 2013-2024 (identical to 01-06).
All overfitting prevention measures A-G are applied.

Run from project root (Implementierung1/):
  python experiments/regime_classifier_v2/07_regime_simplification.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import collections
import importlib.util
import json
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import mode as scipy_mode

from sklearn.base            import BaseEstimator, ClassifierMixin
from sklearn.ensemble        import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (accuracy_score, auc, confusion_matrix,
                                     f1_score, precision_recall_curve,
                                     precision_score, recall_score, roc_auc_score,
                                     roc_curve)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, learning_curve
from sklearn.neural_network  import MLPClassifier
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.svm             import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed -- XGBoost skipped.")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    torch.manual_seed(42)
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: torch not installed -- LSTM/GRU skipped.")

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    print("WARNING: hmmlearn not installed -- HMM skipped.")

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.path.join(PROJECT_ROOT, "results", "regime_classifier_v2", "regime_simplification")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Import helpers from 01_build_dataset.py via importlib
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "build_dataset", os.path.join(SCRIPT_DIR, "01_build_dataset.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_raw_data              = _mod.load_raw_data
compute_binary_shift_label = _mod.compute_binary_shift_label
compute_features           = _mod.compute_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_YEARS  = list(range(2013, 2025))   # 12 folds
TARGET_COL  = "shift_label"
SEQ_LEN     = 20
SEQ_FEAT    = ["VIX_raw", "delta_VIX", "SPY_return_1", "VIX_rolling_std_10"]

# Group B features (Plan 9 winner) -- base
GROUP_B_BASE = ["VIX_MA20", "max_VIX_window", "min_VIX_window",
                "VIX_slope_20", "VIX_rolling_std_10"]
# Distance features vary per regime definition -- built dynamically

# Tabular features for LSTM/GRU (non-distance part)
TAB_FEATURES_BASE = ["VIX_MA20", "max_VIX_window", "min_VIX_window",
                     "VIX_slope_20", "VIX_rolling_std_10"]

SCRIPT_START_TIME = time.time()


# ===========================================================================
# Regime Definition Functions
# ===========================================================================

def regime_A(vix: pd.Series) -> pd.Series:
    """Original 4 regimes: thresholds at 15, 20, 30."""
    labels = np.ones(len(vix), dtype=int)
    v = vix.values
    labels[v > 15]  = 2
    labels[v > 20]  = 3
    labels[v > 30]  = 4
    return pd.Series(labels, index=vix.index, name="regime")


def regime_B(vix: pd.Series) -> pd.Series:
    """2 regimes: threshold at 20. 0=Calm, 1=Volatile."""
    return pd.Series((vix.values > 20).astype(int), index=vix.index, name="regime")


def regime_C(vix: pd.Series) -> pd.Series:
    """2 regimes: threshold at 25. 0=Calm, 1=Volatile."""
    return pd.Series((vix.values > 25).astype(int), index=vix.index, name="regime")


def regime_D(vix: pd.Series) -> pd.Series:
    """2 regimes with hysteresis: threshold 20, buffer 2.
    Calm -> Volatile only when VIX > 22.
    Volatile -> Calm only when VIX < 18.
    """
    v = vix.values
    regime = np.zeros(len(v), dtype=int)
    regime[0] = int(v[0] > 20)
    for t in range(1, len(v)):
        if regime[t - 1] == 0:   # currently Calm
            regime[t] = 1 if v[t] > 22 else 0
        else:                     # currently Volatile
            regime[t] = 0 if v[t] < 18 else 1
    return pd.Series(regime, index=vix.index, name="regime")


def regime_E(vix: pd.Series) -> pd.Series:
    """Smoothed 4 regimes: original 4-class with 5-day backward-looking majority vote."""
    raw = regime_A(vix).values.astype(float)
    smoothed = pd.Series(raw, index=vix.index).rolling(5, min_periods=1).apply(
        lambda x: float(scipy_mode(x, keepdims=False)[0])
    ).astype(int)
    smoothed.name = "regime"
    return smoothed


REGIME_DEFS = {
    "A_4reg":      (regime_A, [15, 20, 30],   "4 regimes: 15/20/30"),
    "B_2r_thr20":  (regime_B, [20],            "2 regimes: thr=20"),
    "C_2r_thr25":  (regime_C, [25],            "2 regimes: thr=25"),
    "D_hysteresis":(regime_D, [18, 22],        "hysteresis: 18/22"),
    "E_smooth4":   (regime_E, [15, 20, 30],    "smoothed 4r: 15/20/30"),
}


# ===========================================================================
# Distance features (threshold-specific)
# ===========================================================================

def compute_dist_features(vix: pd.Series, thresholds: list) -> pd.DataFrame:
    """Compute VIX_dist_nearest, VIX_dist_upper, VIX_dist_lower for given thresholds."""
    v = vix.values

    def _dist_nearest(val):
        return min(abs(val - t) for t in thresholds)

    def _dist_upper(val):
        above = [t for t in thresholds if t >= val]
        return (min(above) - val) if above else (val - thresholds[-1])

    def _dist_lower(val):
        below = [t for t in thresholds if t < val]
        return (val - max(below)) if below else val

    nearest = np.array([_dist_nearest(x) for x in v])
    upper   = np.array([_dist_upper(x)   for x in v])
    lower   = np.array([_dist_lower(x)   for x in v])

    return pd.DataFrame({
        "VIX_dist_nearest": nearest,
        "VIX_dist_upper":   upper,
        "VIX_dist_lower":   lower,
    }, index=vix.index)


def get_group_b_features(vix: pd.Series, base_feats: pd.DataFrame, thresholds: list) -> pd.DataFrame:
    """Return Group B feature set: base5 + dist3, with dist recomputed for given thresholds."""
    dist = compute_dist_features(vix, thresholds)
    return pd.concat([base_feats[GROUP_B_BASE], dist], axis=1)


# ===========================================================================
# Overfitting Prevention -- Feature Correlation Handling (A)
# ===========================================================================

def drop_correlated(X_train: np.ndarray, y_train: np.ndarray,
                    feature_names: list, threshold: float = 0.90) -> tuple:
    """
    Drop one of any pair with |r| > threshold.
    Keep the one with higher univariate F-statistic. Returns (kept_names, dropped_set).
    """
    if X_train.shape[1] <= 1:
        return feature_names, set()

    corr = np.abs(np.corrcoef(X_train.T))
    to_drop = set()
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if corr[i, j] > threshold and feature_names[j] not in to_drop:
                if feature_names[i] in to_drop:
                    continue
                # Keep the feature with higher F-statistic
                try:
                    f_i = f_classif(X_train[:, [i]], y_train)[0][0]
                    f_j = f_classif(X_train[:, [j]], y_train)[0][0]
                except Exception:
                    f_i, f_j = 0.0, 0.0
                if np.isnan(f_i):
                    f_i = 0.0
                if np.isnan(f_j):
                    f_j = 0.0
                drop = feature_names[j] if f_i >= f_j else feature_names[i]
                to_drop.add(drop)
    kept = [f for f in feature_names if f not in to_drop]
    return kept, to_drop


# ===========================================================================
# Metrics
# ===========================================================================

def wf_metrics(y_true, y_pred):
    return {
        "recall":    recall_score(   y_true, y_pred, pos_label=1, zero_division=0),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1":        f1_score(       y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy":  accuracy_score( y_true, y_pred),
        "far":       (int(((y_pred == 1) & (y_true == 0)).sum()) /
                      max(int((y_true == 0).sum()), 1)),
    }


def agg(records):
    df = pd.DataFrame(records)
    return {k: (df[k].mean(), df[k].std())
            for k in ["recall", "precision", "f1", "accuracy", "far"]}


def eta_str(elapsed_s, done, total):
    if done == 0:
        return "ETA: --"
    rate = elapsed_s / done
    remaining = rate * (total - done)
    if remaining > 3600:
        return f"ETA: ~{remaining/3600:.1f}h"
    elif remaining > 60:
        return f"ETA: ~{remaining/60:.0f}m"
    else:
        return f"ETA: ~{remaining:.0f}s"


# ===========================================================================
# Custom Wrappers (WeightedGB, WeightedMLP)
# ===========================================================================

class WeightedGB(BaseEstimator, ClassifierMixin):
    """GradientBoostingClassifier with shift sample weighting."""
    def __init__(self, shift_weight=1.0, n_estimators=100, max_depth=3,
                 learning_rate=0.1, min_samples_leaf=5):
        self.shift_weight     = shift_weight
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        sw = np.where(y == 1, self.shift_weight, 1.0)
        self._gb = GradientBoostingClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, min_samples_leaf=self.min_samples_leaf,
            random_state=RANDOM_SEED,
        ).fit(X, y, sample_weight=sw)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):       return self._gb.predict(X)
    def predict_proba(self, X): return self._gb.predict_proba(X)


class WeightedMLP(BaseEstimator, ClassifierMixin):
    """MLPClassifier with shift oversampling to emulate class weighting."""
    def __init__(self, shift_weight=1.0, hidden_layer_sizes=(64, 32),
                 alpha=0.001, learning_rate_init=0.001, max_iter=500):
        self.shift_weight       = shift_weight
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha              = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter           = max_iter

    def fit(self, X, y):
        reps    = max(1, int(round(self.shift_weight)))
        pos_idx = np.where(y == 1)[0]
        if reps > 1 and len(pos_idx) > 0:
            extra_X = np.repeat(X[pos_idx], reps - 1, axis=0)
            extra_y = np.repeat(y[pos_idx], reps - 1)
            X_aug = np.vstack([X, extra_X])
            y_aug = np.concatenate([y, extra_y])
        else:
            X_aug, y_aug = X, y
        self._mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
            learning_rate_init=self.learning_rate_init, max_iter=self.max_iter,
            random_state=RANDOM_SEED,
        ).fit(X_aug, y_aug)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):       return self._mlp.predict(X)
    def predict_proba(self, X): return self._mlp.predict_proba(X)


# ===========================================================================
# LSTM / GRU models (PyTorch)
# ===========================================================================

if TORCH_AVAILABLE:
    class ShiftLSTM(nn.Module):
        def __init__(self, seq_dim=4, tab_dim=5, hidden_dim=32,
                     num_layers=1, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(seq_dim, hidden_dim, num_layers,
                                batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim + tab_dim, 32), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(32, 1),
            )

        def forward(self, seq, tab):
            _, (h, _) = self.lstm(seq)
            return self.head(torch.cat([h[-1], tab], dim=1)).squeeze(1)

    class ShiftGRU(nn.Module):
        def __init__(self, seq_dim=4, tab_dim=5, hidden_dim=32,
                     num_layers=1, dropout=0.3):
            super().__init__()
            self.gru = nn.GRU(seq_dim, hidden_dim, num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim + tab_dim, 32), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(32, 1),
            )

        def forward(self, seq, tab):
            _, h = self.gru(seq)
            return self.head(torch.cat([h[-1], tab], dim=1)).squeeze(1)


def build_sequences(arr, seq_len):
    """arr: (n, d) -> (n - seq_len + 1, seq_len, d)"""
    n, d = arr.shape
    return np.stack([arr[i - seq_len: i] for i in range(seq_len, n + 1)])


def run_rnn_fold(seq_tr, tab_tr, y_tr, seq_te, tab_te, y_te,
                 hidden_dim, num_layers, dropout, lr, pos_weight,
                 model_class, max_epochs=30, patience=4, batch_size=64):
    """Train and evaluate one LSTM or GRU config on one fold. Returns y_prob (test)."""
    if not TORCH_AVAILABLE:
        return np.zeros(len(y_te))
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tab_dim = tab_tr.shape[1]
    seq_dim = seq_tr.shape[2]
    n_val   = max(int(len(y_tr) * 0.2), 1)
    n_tr2   = len(y_tr) - n_val

    def _tens(s, t, y):
        return (torch.tensor(s, dtype=torch.float32),
                torch.tensor(t, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))

    tr_loader  = DataLoader(TensorDataset(*_tens(seq_tr[:n_tr2], tab_tr[:n_tr2], y_tr[:n_tr2])),
                            batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(*_tens(seq_tr[n_tr2:], tab_tr[n_tr2:], y_tr[n_tr2:])),
                            batch_size=batch_size)

    model = model_class(seq_dim=seq_dim, tab_dim=tab_dim, hidden_dim=hidden_dim,
                        num_layers=num_layers, dropout=dropout).to(device)
    crit  = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    best_val, p_cnt, best_state = float("inf"), 0, None
    for _ in range(max_epochs):
        model.train()
        for sb, tb, yb in tr_loader:
            sb, tb, yb = sb.to(device), tb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(sb, tb), yb).backward(); opt.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sb, tb, yb in val_loader:
                val_loss += crit(model(sb.to(device), tb.to(device)), yb.to(device)).item()
        val_loss /= max(len(val_loader), 1)
        if val_loss < best_val:
            best_val, p_cnt = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            p_cnt += 1
            if p_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        s = torch.tensor(seq_te, dtype=torch.float32).to(device)
        t = torch.tensor(tab_te, dtype=torch.float32).to(device)
        probs = torch.sigmoid(model(s, t)).cpu().numpy()
    return probs


# ===========================================================================
# HMM wrapper (unsupervised shift detection)
# ===========================================================================

class HMMShiftDetector:
    """
    Wraps GaussianHMM for binary shift prediction.
    Fits on VIX sequence, decodes test states, predicts shift when state changes.
    """
    def __init__(self, n_components=2, covariance_type="diag"):
        self.n_components     = n_components
        self.covariance_type  = covariance_type
        self._model           = None
        self._train_state_seq = None
        self.classes_         = np.array([0, 1])

    def fit(self, X, y=None):
        """X is scaled feature matrix. Uses only first column (VIX-like) for HMM."""
        obs = X[:, :1]   # single univariate observation
        m = GaussianHMM(n_components=self.n_components,
                        covariance_type=self.covariance_type,
                        n_iter=200, random_state=RANDOM_SEED)
        m.fit(obs)
        self._model = m
        states = m.predict(obs)
        self._last_train_state = states[-1]
        return self

    def predict_proba(self, X):
        obs    = X[:, :1]
        states = self._model.predict(obs)
        # Shift = state changes relative to previous state in sequence
        probs  = np.zeros((len(states), 2))
        probs[0, int(states[0] != self._last_train_state)] = 1.0
        for i in range(1, len(states)):
            shifted = int(states[i] != states[i - 1])
            probs[i, shifted] = 1.0
        return probs

    def predict(self, X):
        return self.predict_proba(X)[:, 1].astype(int)


# ===========================================================================
# Optimal threshold via PR curve on training data
# ===========================================================================

def optimal_threshold_pr(y_train, y_prob_train):
    """Return threshold maximizing F1 on training PR curve."""
    prec, rec, thrs = precision_recall_curve(y_train, y_prob_train)
    f1s = np.where((prec + rec) > 0,
                   2 * prec * rec / (prec + rec + 1e-9), 0.0)
    idx = np.argmax(f1s)
    thr = thrs[idx] if idx < len(thrs) else 0.5
    return float(np.clip(thr, 0.01, 0.99))


# ===========================================================================
# Build dataset for a given regime definition + H
# ===========================================================================

def build_dataset(prices, vix, base_feats, reg_fn, thresholds, h):
    """Build complete feature+label dataset for one (regime_def, H) combo."""
    regime  = reg_fn(vix)
    sl      = compute_binary_shift_label(regime, h)
    dist    = compute_dist_features(vix, thresholds)

    # Full feature set: Group B = base5 + dist3
    feat_df = pd.concat([base_feats[GROUP_B_BASE], dist], axis=1)
    feat_df["VIX_raw"]      = vix
    feat_df["SPY_return_1"] = prices["SPY"].pct_change()

    ds              = feat_df.copy()
    ds["regime"]    = regime
    ds[TARGET_COL]  = sl
    return ds.dropna()


# ===========================================================================
# STEP 1 -- Regime Definition Comparison
# ===========================================================================

def step1(prices, vix, base_feats):
    print("\n" + "=" * 70)
    print("STEP 1 -- Regime Definition Comparison")
    print("=" * 70)
    t0_step = time.time()

    H_VALUES = [3, 5, 7, 10]
    LR_MODEL = lambda: LogisticRegression(C=0.001, class_weight="balanced",
                                          penalty="l2", solver="saga",
                                          max_iter=2000, random_state=RANDOM_SEED)

    rows          = []
    cm_data       = {}   # regime_def -> aggregated cm at best H
    best_h_per_def = {}

    total_runs = len(REGIME_DEFS) * len(H_VALUES)
    run_count  = 0

    for def_name, (reg_fn, thresholds, desc) in REGIME_DEFS.items():
        print(f"\n  [{def_name}]  {desc}")
        best_h_f1  = -1.0
        best_h_val = H_VALUES[1]

        for h in H_VALUES:
            run_count += 1
            elapsed = time.time() - t0_step
            print(f"    H={h:2d}  {eta_str(elapsed, run_count - 1, total_runs)} ...", end=" ")

            ds = build_dataset(prices, vix, base_feats, reg_fn, thresholds, h)
            n_shift   = int((ds[TARGET_COL] == 1).sum())
            n_noshift = int((ds[TARGET_COL] == 0).sum())
            n_total   = n_shift + n_noshift
            shift_pct = 100.0 * n_shift / max(n_total, 1)

            FEAT_COLS = [c for c in GROUP_B_BASE + ["VIX_dist_nearest",
                                                      "VIX_dist_upper",
                                                      "VIX_dist_lower"] if c in ds.columns]

            X_all = ds[FEAT_COLS].values
            y_all = ds[TARGET_COL].values.astype(int)
            dates = ds.index

            fold_recs = []
            fold_cms  = []
            for fold_idx, test_year in enumerate(TEST_YEARS):
                train_mask = dates.year <= (test_year - 1)
                test_mask  = dates.year == test_year
                X_tr, y_tr = X_all[train_mask], y_all[train_mask]
                X_te, y_te = X_all[test_mask],  y_all[test_mask]
                if len(X_tr) == 0 or len(X_te) == 0 or len(np.unique(y_tr)) < 2:
                    continue
                sc = StandardScaler()
                X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)

                # Feature correlation drop (Measure A)
                kept, dropped = drop_correlated(X_tr_s, y_tr, FEAT_COLS)
                if dropped:
                    kidx = [FEAT_COLS.index(f) for f in kept]
                    X_tr_s = X_tr_s[:, kidx]
                    X_te_s = X_te_s[:, kidx]

                m = LR_MODEL()
                m.fit(X_tr_s, y_tr)
                y_pred = m.predict(X_te_s)
                fold_recs.append(wf_metrics(y_te, y_pred))
                fold_cms.append(confusion_matrix(y_te, y_pred, labels=[0, 1]))

            if not fold_recs:
                continue

            a = agg(fold_recs)
            row = {
                "regime_def": def_name,
                "description": desc,
                "H": h,
                "n_days": n_total,
                "n_shift": n_shift,
                "n_no_shift": n_noshift,
                "shift_pct": round(shift_pct, 2),
                "avg_recall":    round(a["recall"][0],    4),
                "avg_precision": round(a["precision"][0], 4),
                "avg_f1":        round(a["f1"][0],        4),
                "avg_accuracy":  round(a["accuracy"][0],  4),
            }
            rows.append(row)

            if a["f1"][0] > best_h_f1:
                best_h_f1  = a["f1"][0]
                best_h_val = h
                # Aggregate CMs
                if fold_cms:
                    cm_data[def_name] = sum(fold_cms)

            print(f"shift={shift_pct:.0f}%  recall={a['recall'][0]:.3f}  "
                  f"f1={a['f1'][0]:.3f}")

        best_h_per_def[def_name] = best_h_val

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "step1_regime_comparison.csv"), index=False)
    print(f"\n  Saved: step1_regime_comparison.csv")

    # ---- Overfitting prevention D: Robustness check ----
    print("\n  Robustness check (Measure D):")
    for def_name in REGIME_DEFS:
        sub = df[df["regime_def"] == def_name]
        if sub.empty:
            continue
        # Best for each H?
        for h in H_VALUES:
            h_sub = df[df["H"] == h]
            if h_sub.empty:
                continue
            best_at_h = h_sub.loc[h_sub["avg_f1"].idxmax(), "regime_def"]
            if best_at_h == def_name:
                print(f"    {def_name} is best at H={h}")

    # ---- Determine best regime definition at H=5 ----
    df5 = df[df["H"] == 5]
    if df5.empty:
        df5 = df[df["H"] == 3]
    best_idx     = df5["avg_f1"].idxmax()
    best_def     = df5.loc[best_idx, "regime_def"]
    best_def_f1  = df5.loc[best_idx, "avg_f1"]

    print(f"\n  Best for H=5: {best_def}  F1={best_def_f1:.4f}")

    # ---- Plots ----
    _plot_step1_f1_vs_h(df, H_VALUES)
    _plot_step1_shift_distribution(df, H_VALUES)
    _plot_step1_confusion_matrices(cm_data)

    # ---- Print summary table ----
    print("\n=== Step 1: Regime Definition Comparison ===\n")
    header = f"{'Regime Def':16s} | {'Thresholds':12s} | {'H=3':16s} | {'H=5':16s} | {'H=10':16s}"
    print(header)
    print(f"{'':16s} | {'':12s} | {'shft%':6s} {'F1':6s} | {'shft%':6s} {'F1':6s} | {'shft%':6s} {'F1':6s}")
    print("-" * len(header))
    for def_name, (_, thresholds, _) in REGIME_DEFS.items():
        thr_str = "/".join(str(t) for t in thresholds)
        vals = {}
        for h in [3, 5, 10]:
            sub = df[(df["regime_def"] == def_name) & (df["H"] == h)]
            if not sub.empty:
                vals[h] = (sub.iloc[0]["shift_pct"], sub.iloc[0]["avg_f1"])
            else:
                vals[h] = (0.0, 0.0)
        print(f"{def_name:16s} | {thr_str:12s} | "
              f"{vals[3][0]:5.0f}% {vals[3][1]:.3f} | "
              f"{vals[5][0]:5.0f}% {vals[5][1]:.3f} | "
              f"{vals[10][0]:5.0f}% {vals[10][1]:.3f}")
    print(f"\nBest for H=5: {best_def}  F1={best_def_f1:.4f}")

    elapsed = time.time() - t0_step
    print(f"\n  Step 1 done ({elapsed:.0f}s)")
    return best_def, df


def _plot_step1_f1_vs_h(df, H_values):
    fig, ax = plt.subplots(figsize=(9, 5))
    for def_name in df["regime_def"].unique():
        sub = df[df["regime_def"] == def_name].sort_values("H")
        ax.plot(sub["H"], sub["avg_f1"], marker="o", label=def_name)
    ax.set_xlabel("Horizon H"); ax.set_ylabel("Avg F1 (12 folds)")
    ax.set_title("Step 1: F1 vs H per Regime Definition")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step1_regime_comparison.png"), dpi=120)
    plt.close(fig)


def _plot_step1_shift_distribution(df, H_values):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(H_values))
    width = 0.15
    defs  = list(df["regime_def"].unique())
    for i, def_name in enumerate(defs):
        sub = df[df["regime_def"] == def_name].sort_values("H")
        vals = [sub[sub["H"] == h]["shift_pct"].values[0]
                if len(sub[sub["H"] == h]) > 0 else 0
                for h in H_values]
        ax.bar(x + i * width, vals, width, label=def_name)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f"H={h}" for h in H_values])
    ax.set_ylabel("Shift %"); ax.set_title("Step 1: Shift% per Regime Definition and H")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step1_shift_distributions.png"), dpi=120)
    plt.close(fig)


def _plot_step1_confusion_matrices(cm_data):
    defs  = list(REGIME_DEFS.keys())
    n     = len(defs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i, def_name in enumerate(defs):
        ax = axes[i]
        if def_name in cm_data:
            cm = cm_data[def_name]
            sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                        xticklabels=["No Shift", "Shift"],
                        yticklabels=["No Shift", "Shift"])
            ax.set_title(f"{def_name}\n(agg. 12 folds)", fontsize=8)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(def_name, fontsize=8)
    fig.suptitle("Step 1: Confusion Matrices at Best H per Regime Definition")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step1_confusion_matrices.png"), dpi=120)
    plt.close(fig)


# ===========================================================================
# STEP 2 -- Full Weight Optimization
# ===========================================================================

def step2(prices, vix, base_feats, best_def_name, best_h=5):
    print("\n" + "=" * 70)
    print(f"STEP 2 -- Full Weight Optimization  [def={best_def_name}, H={best_h}]")
    print("=" * 70)
    t0_step = time.time()

    reg_fn, thresholds, desc = REGIME_DEFS[best_def_name]
    ds = build_dataset(prices, vix, base_feats, reg_fn, thresholds, best_h)

    FEAT_COLS = [c for c in GROUP_B_BASE + ["VIX_dist_nearest",
                                              "VIX_dist_upper",
                                              "VIX_dist_lower"] if c in ds.columns]
    print(f"  Features ({len(FEAT_COLS)}): {FEAT_COLS}")

    X_all  = ds[FEAT_COLS].values
    y_all  = ds[TARGET_COL].values.astype(int)
    dates  = ds.index

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    all_fold_rows   = []
    weight_rows     = []
    diag_rows       = []    # overfitting diagnostics
    model_fold_probs= {}    # model -> list of (y_te, y_prob)
    rejected_models = set()
    flagged_models  = set()

    # ---- Full grids ----
    # LR pipeline with SelectKBest (Measure C)
    lr_pipe_grid = {
        "select__k":             [3, 5, "all"],
        "model__C":              [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "model__class_weight":   ["balanced",
                                  {0:1,1:1}, {0:1,1:2}, {0:1,1:3}, {0:1,1:5},
                                  {0:1,1:10}, {0:1,1:15}, {0:1,1:20}],
        "model__penalty":        ["l1", "l2"],
        "model__solver":         ["saga"],
        "model__max_iter":       [2000],
    }

    rf_grid = {
        "n_estimators":     [100, 300, 500],
        "max_depth":        [3, 5, 10, None],
        "class_weight":     ["balanced", "balanced_subsample",
                             {0:1,1:2}, {0:1,1:5}, {0:1,1:10}],
        "min_samples_leaf": [1, 5, 10],
    }

    svm_grid = {
        "C":            [0.01, 0.1, 1.0, 10.0, 100.0],
        "gamma":        ["scale", "auto", 0.01, 0.1],
        "class_weight": ["balanced", {0:1,1:2}, {0:1,1:5}, {0:1,1:10}],
    }

    gb_grid = {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "shift_weight":  [1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0],
        "min_samples_leaf": [5, 10],
    }

    mlp_grid = {
        "hidden_layer_sizes": [(32,), (64, 32), (128, 64, 32), (64, 32, 16)],
        "alpha":              [0.0001, 0.001, 0.01, 0.1],
        "shift_weight":       [1, 2, 5, 10, 15, 20],
        "learning_rate_init": [0.001, 0.01],
        "max_iter":           [500],
    }

    # Compute n_combos
    def _n_combos(grid):
        n = 1
        for v in grid.values():
            n *= len(v)
        return n

    n_lr  = _n_combos(lr_pipe_grid)
    n_rf  = _n_combos(rf_grid)
    n_svm = _n_combos(svm_grid)
    # GB and MLP use RandomizedSearchCV
    n_gb_iter  = 100
    n_mlp_iter = 80
    n_xgb_iter = 200

    print(f"\n  LR  pipeline: {n_lr} combos (GridSearch)")
    print(f"  RF : {n_rf} combos (GridSearch)")
    print(f"  SVM: {n_svm} combos (GridSearch)")
    print(f"  XGB: RandomizedSearch n_iter={n_xgb_iter}")
    print(f"  GB : RandomizedSearch n_iter={n_gb_iter}")
    print(f"  MLP: RandomizedSearch n_iter={n_mlp_iter}")

    # ---- Model spec list: (name, base_fn, grid, search_type) ----
    model_specs = [
        ("LR",  None,                                     lr_pipe_grid, "grid_pipe"),
        ("RF",  lambda nn, np_: RandomForestClassifier(random_state=RANDOM_SEED),
         rf_grid, "grid"),
        ("SVM", lambda nn, np_: SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
         svm_grid, "grid"),
        ("GB",  lambda nn, np_: WeightedGB(),
         gb_grid, "random"),
        ("MLP", lambda nn, np_: WeightedMLP(),
         mlp_grid, "random"),
    ]

    total_model_folds = len(model_specs) * len(TEST_YEARS)
    done_mf = 0

    for mname, base_fn, grid, search_type in model_specs:
        print(f"\n  --- [{mname}] ---")
        t0_m = time.time()
        fold_recs = []
        fold_probs_list = []
        fold_diag = []

        for fold_idx, test_year in enumerate(TEST_YEARS):
            train_mask = dates.year <= (test_year - 1)
            test_mask  = dates.year == test_year
            X_tr, y_tr = X_all[train_mask], y_all[train_mask]
            X_te, y_te = X_all[test_mask],  y_all[test_mask]
            if len(X_tr) == 0 or len(X_te) == 0 or len(np.unique(y_tr)) < 2:
                done_mf += 1
                continue

            sc = StandardScaler()
            X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)

            # Measure A: drop correlated features
            kept, dropped = drop_correlated(X_tr_s, y_tr, FEAT_COLS)
            if dropped:
                kidx    = [FEAT_COLS.index(f) for f in kept]
                X_tr_use = X_tr_s[:, kidx]
                X_te_use = X_te_s[:, kidx]
            else:
                X_tr_use, X_te_use = X_tr_s, X_te_s
                kept = FEAT_COLS

            n_neg = int((y_tr == 0).sum())
            n_pos = int((y_tr == 1).sum())

            try:
                if search_type == "grid_pipe":
                    pipe = Pipeline([
                        ("select", SelectKBest(f_classif)),
                        ("model",  LogisticRegression(random_state=RANDOM_SEED)),
                    ])
                    search = GridSearchCV(pipe, grid, cv=inner_cv,
                                         scoring="f1", refit=True, n_jobs=1)
                    search.fit(X_tr_use, y_tr)

                elif search_type == "grid":
                    base = base_fn(n_neg, n_pos)
                    search = GridSearchCV(base, grid, cv=inner_cv,
                                         scoring="f1", refit=True, n_jobs=1)
                    search.fit(X_tr_use, y_tr)

                else:  # random
                    base = base_fn(n_neg, n_pos)
                    n_iter = n_gb_iter if mname == "GB" else n_mlp_iter
                    search = RandomizedSearchCV(base, grid, n_iter=n_iter,
                                               cv=inner_cv, scoring="f1",
                                               refit=True, n_jobs=1,
                                               random_state=RANDOM_SEED)
                    search.fit(X_tr_use, y_tr)

                y_prob_te = search.best_estimator_.predict_proba(X_te_use)[:, 1]
                y_pred_te = (y_prob_te >= 0.5).astype(int)

                # Measure B: train performance gap
                y_prob_tr   = search.best_estimator_.predict_proba(X_tr_use)[:, 1]
                y_pred_tr   = (y_prob_tr >= 0.5).astype(int)
                train_met   = wf_metrics(y_tr, y_pred_tr)
                test_met    = wf_metrics(y_te, y_pred_te)
                gap_f1      = train_met["f1"] - test_met["f1"]

                diag_rows.append({
                    "step": 2, "fold": fold_idx + 1, "test_year": test_year,
                    "model": mname,
                    "train_recall": round(train_met["recall"], 4),
                    "train_f1":     round(train_met["f1"],     4),
                    "test_recall":  round(test_met["recall"],  4),
                    "test_f1":      round(test_met["f1"],      4),
                    "gap_f1":       round(gap_f1,              4),
                })
                fold_diag.append({"gap_f1": gap_f1})

                rec = dict(test_met)
                rec.update({"fold": fold_idx + 1, "test_year": test_year,
                            "model": mname,
                            "best_params": json.dumps(search.best_params_, default=str)})
                fold_recs.append(rec)
                fold_probs_list.append((y_te, y_prob_te))

                cw_key = str(search.best_params_.get("model__class_weight",
                             search.best_params_.get("class_weight",
                             search.best_params_.get("shift_weight", "?"))))
                weight_rows.append({"fold": fold_idx + 1, "model": mname,
                                    "best_weight": cw_key,
                                    "shift_ratio": round(n_neg / max(n_pos, 1), 3)})

            except Exception as e:
                print(f"      fold {fold_idx+1} ERROR: {e}")
                fold_recs.append({"fold": fold_idx+1, "test_year": test_year,
                                  "model": mname, "recall":0,"precision":0,
                                  "f1":0,"accuracy":0,"far":0})

            done_mf += 1
            elapsed = time.time() - t0_step
            print(f"    fold {fold_idx+1:2d} ({test_year})  "
                  f"f1={fold_recs[-1].get('f1', 0):.3f}  "
                  f"gap={fold_diag[-1]['gap_f1']:.3f}  "
                  f"{eta_str(elapsed, done_mf, total_model_folds)}",
                  flush=True)

        if fold_recs:
            a = agg(fold_recs)
            all_fold_rows.extend(fold_recs)
            model_fold_probs[mname] = fold_probs_list

            # Measure E: rejection
            avg_gap = np.mean([d["gap_f1"] for d in fold_diag]) if fold_diag else 0.0
            if avg_gap > 0.20:
                print(f"  WARNING: {mname} REJECTED -- avg train-test gap = {avg_gap:.3f} > 0.20")
                rejected_models.add(mname)

            # Measure F: stability
            f1s = [r.get("f1", 0) for r in fold_recs]
            std_f1 = float(np.std(f1s))
            if std_f1 > 0.20:
                print(f"  FLAG: {mname} unstable -- std(F1)={std_f1:.3f} > 0.20")
                flagged_models.add(mname)

            elapsed_m = time.time() - t0_m
            print(f"  [{mname}] done ({elapsed_m:.0f}s)  "
                  f"avg recall={a['recall'][0]:.3f}  avg f1={a['f1'][0]:.3f}  "
                  f"avg gap={avg_gap:.3f}  "
                  f"{'REJECTED' if mname in rejected_models else ''}")

    # ---- XGBoost (per-fold scale_pos_weight) ----
    if HAS_XGB:
        mname = "XGB"
        print(f"\n  --- [{mname}] RandomizedSearch n_iter={n_xgb_iter} ---")
        t0_m      = time.time()
        fold_recs = []
        fold_probs_list = []
        fold_diag = []

        for fold_idx, test_year in enumerate(TEST_YEARS):
            train_mask = dates.year <= (test_year - 1)
            test_mask  = dates.year == test_year
            X_tr, y_tr = X_all[train_mask], y_all[train_mask]
            X_te, y_te = X_all[test_mask],  y_all[test_mask]
            if len(X_tr) == 0 or len(X_te) == 0 or len(np.unique(y_tr)) < 2:
                continue

            sc = StandardScaler()
            X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)
            kept, dropped = drop_correlated(X_tr_s, y_tr, FEAT_COLS)
            if dropped:
                kidx    = [FEAT_COLS.index(f) for f in kept]
                X_tr_use = X_tr_s[:, kidx]
                X_te_use = X_te_s[:, kidx]
            else:
                X_tr_use, X_te_use = X_tr_s, X_te_s

            n_neg = int((y_tr == 0).sum())
            n_pos = int((y_tr == 1).sum())
            base_ratio = n_neg / max(n_pos, 1)

            xgb_grid = {
                "n_estimators":     [100, 200, 500],
                "max_depth":        [3, 4, 6, 8],
                "learning_rate":    [0.01, 0.05, 0.1],
                "scale_pos_weight": [base_ratio * f for f in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]],
                "min_child_weight": [1, 3, 5],
                "reg_alpha":        [0, 0.1, 1.0],
                "reg_lambda":       [1.0, 5.0, 10.0],
            }

            try:
                base = XGBClassifier(eval_metric="logloss", verbosity=0,
                                     random_state=RANDOM_SEED)
                search = RandomizedSearchCV(base, xgb_grid, n_iter=n_xgb_iter,
                                           cv=inner_cv, scoring="f1",
                                           refit=True, n_jobs=1,
                                           random_state=RANDOM_SEED)
                search.fit(X_tr_use, y_tr)

                y_prob_te = search.best_estimator_.predict_proba(X_te_use)[:, 1]
                y_pred_te = (y_prob_te >= 0.5).astype(int)
                y_prob_tr = search.best_estimator_.predict_proba(X_tr_use)[:, 1]
                y_pred_tr = (y_prob_tr >= 0.5).astype(int)
                train_met = wf_metrics(y_tr, y_pred_tr)
                test_met  = wf_metrics(y_te, y_pred_te)
                gap_f1    = train_met["f1"] - test_met["f1"]

                diag_rows.append({
                    "step": 2, "fold": fold_idx + 1, "test_year": test_year,
                    "model": mname,
                    "train_recall": round(train_met["recall"], 4),
                    "train_f1":     round(train_met["f1"],     4),
                    "test_recall":  round(test_met["recall"],  4),
                    "test_f1":      round(test_met["f1"],      4),
                    "gap_f1":       round(gap_f1,              4),
                })
                fold_diag.append({"gap_f1": gap_f1})

                rec = dict(test_met)
                rec.update({"fold": fold_idx + 1, "test_year": test_year, "model": mname,
                            "best_params": json.dumps(search.best_params_, default=str)})
                fold_recs.append(rec)
                fold_probs_list.append((y_te, y_prob_te))
                weight_rows.append({"fold": fold_idx+1, "model": mname,
                                    "best_weight": str(search.best_params_.get("scale_pos_weight")),
                                    "shift_ratio": round(base_ratio, 3)})

            except Exception as e:
                print(f"      fold {fold_idx+1} ERROR: {e}")

            elapsed = time.time() - t0_step
            if fold_recs:
                print(f"    fold {fold_idx+1:2d} ({test_year})  "
                      f"f1={fold_recs[-1].get('f1', 0):.3f}  "
                      f"gap={fold_diag[-1]['gap_f1']:.3f}  "
                      f"{eta_str(elapsed, fold_idx+1, len(TEST_YEARS))}", flush=True)

        if fold_recs:
            a = agg(fold_recs)
            all_fold_rows.extend(fold_recs)
            model_fold_probs[mname] = fold_probs_list

            avg_gap = np.mean([d["gap_f1"] for d in fold_diag]) if fold_diag else 0.0
            if avg_gap > 0.20:
                print(f"  WARNING: XGB REJECTED -- avg gap = {avg_gap:.3f} > 0.20")
                rejected_models.add("XGB")
            f1s = [r.get("f1", 0) for r in fold_recs]
            if float(np.std(f1s)) > 0.20:
                flagged_models.add("XGB")

            elapsed_m = time.time() - t0_m
            print(f"  [XGB] done ({elapsed_m:.0f}s)  "
                  f"avg recall={a['recall'][0]:.3f}  avg f1={a['f1'][0]:.3f}  "
                  f"avg gap={avg_gap:.3f}")

    # ---- LSTM ----
    lstm_probs_out = []
    best_lstm_config = _run_lstm_search(ds, FEAT_COLS, prices, vix, ShiftLSTM, "LSTM",
                                        all_fold_rows, lstm_probs_out,
                                        diag_rows, weight_rows,
                                        rejected_models, flagged_models,
                                        t0_step)
    if best_lstm_config:
        model_fold_probs["LSTM"] = lstm_probs_out

    # ---- GRU ----
    gru_probs_out = []
    best_gru_config = _run_lstm_search(ds, FEAT_COLS, prices, vix, ShiftGRU, "GRU",
                                       all_fold_rows, gru_probs_out,
                                       diag_rows, weight_rows,
                                       rejected_models, flagged_models,
                                       t0_step)
    if best_gru_config:
        model_fold_probs["GRU"] = gru_probs_out

    # ---- HMM ----
    if HAS_HMM:
        _run_hmm(ds, FEAT_COLS, all_fold_rows, model_fold_probs,
                 diag_rows, weight_rows, rejected_models, flagged_models, t0_step)

    # ---- Save CSVs ----
    pd.DataFrame(all_fold_rows).to_csv(
        os.path.join(OUT_DIR, "step2_full_weight_optimization.csv"), index=False)
    pd.DataFrame(weight_rows).to_csv(
        os.path.join(OUT_DIR, "step2_best_weights_per_fold.csv"), index=False)
    pd.DataFrame(diag_rows).to_csv(
        os.path.join(OUT_DIR, "overfitting_diagnostics.csv"), index=False)

    # ---- Plots ----
    _plot_step2_model_comparison(all_fold_rows, rejected_models)
    _plot_step2_confusion_matrices(all_fold_rows, rejected_models)
    _plot_step2_roc_curves(model_fold_probs, rejected_models)

    # ---- Determine best non-rejected model ----
    df_s2 = pd.DataFrame(all_fold_rows)
    best_model = _pick_best_model(df_s2, rejected_models)

    elapsed = time.time() - t0_step
    print(f"\n  Step 2 done ({elapsed:.0f}s)")
    print(f"  Rejected models: {rejected_models}")
    print(f"  Flagged (unstable): {flagged_models}")
    print(f"  Best model: {best_model}")

    return best_model, all_fold_rows, rejected_models, df_s2


def _run_lstm_search(ds, FEAT_COLS, prices, vix, model_class, model_name,
                     all_fold_rows, fold_probs_out, diag_rows, weight_rows,
                     rejected_models, flagged_models, t0_step):
    """Architecture search on folds 1, 6, 12; then full 12-fold evaluation."""
    if not TORCH_AVAILABLE:
        return None

    print(f"\n  --- [{model_name}] architecture search ---")

    lstm_configs = [
        {"hidden_dim": 16, "num_layers": 1, "dropout": 0.2},
        {"hidden_dim": 32, "num_layers": 1, "dropout": 0.3},
        {"hidden_dim": 64, "num_layers": 1, "dropout": 0.3},
        {"hidden_dim": 32, "num_layers": 2, "dropout": 0.3},
        {"hidden_dim": 64, "num_layers": 2, "dropout": 0.4},
    ]
    pos_weight_values = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0]
    lr_values         = [0.0005, 0.001, 0.005]
    search_fold_indices = [0, 5, 11]  # folds 1, 6, 12

    X_all = ds[FEAT_COLS].values
    y_all = ds[TARGET_COL].values.astype(int)
    dates = ds.index

    # Need VIX raw and SPY_return_1 for sequences
    vix_arr = vix.reindex(ds.index).values
    spy_r   = prices["SPY"].pct_change().reindex(ds.index).values

    # Reconstruct sequence features
    delta_vix = ds["VIX_dist_nearest"].values  # placeholder; use rolling std if available
    vix_raw   = ds["VIX_raw"].values if "VIX_raw" in ds.columns else vix_arr
    spy_ret1  = ds["SPY_return_1"].values if "SPY_return_1" in ds.columns else spy_r
    vix_std10 = ds["VIX_rolling_std_10"].values
    delta_v   = np.gradient(vix_raw)

    seq_arr = np.column_stack([vix_raw, delta_v, spy_ret1, vix_std10])  # (n, 4)

    tab_cols = [c for c in GROUP_B_BASE if c in FEAT_COLS]
    tab_arr  = ds[tab_cols].values  # (n, tab_dim)

    best_config = None
    best_avg_f1 = -1.0
    total_search = len(lstm_configs) * len(pos_weight_values) * len(lr_values)
    done_cfg = 0

    for cfg in lstm_configs:
        for pw in pos_weight_values:
            for lr_ in lr_values:
                fold_f1s = []
                for si in search_fold_indices:
                    if si >= len(TEST_YEARS):
                        continue
                    test_year = TEST_YEARS[si]
                    train_mask = dates.year <= (test_year - 1)
                    test_mask  = dates.year == test_year
                    if not train_mask.any() or not test_mask.any():
                        continue

                    # Build sequences
                    sc_seq = StandardScaler()
                    sc_tab = StandardScaler()
                    seq_tr_raw = seq_arr[train_mask]
                    seq_te_raw = seq_arr[test_mask]
                    tab_tr_raw = tab_arr[train_mask]
                    tab_te_raw = tab_arr[test_mask]
                    y_tr = y_all[train_mask]
                    y_te = y_all[test_mask]

                    seq_tr_s = sc_seq.fit_transform(seq_tr_raw.reshape(-1, seq_tr_raw.shape[-1])).reshape(seq_tr_raw.shape)
                    seq_te_s = sc_seq.transform(seq_te_raw.reshape(-1, seq_te_raw.shape[-1])).reshape(seq_te_raw.shape)
                    tab_tr_s = sc_tab.fit_transform(tab_tr_raw)
                    tab_te_s = sc_tab.transform(tab_te_raw)

                    # Build rolling sequences
                    seq_tr_3d = build_sequences(seq_tr_s, SEQ_LEN)
                    seq_te_3d = build_sequences(seq_te_s, SEQ_LEN)
                    y_tr_seq  = y_tr[SEQ_LEN - 1:]
                    tab_tr_seq = tab_tr_s[SEQ_LEN - 1:]
                    y_te_seq  = y_te[SEQ_LEN - 1:]
                    tab_te_seq = tab_te_s[SEQ_LEN - 1:]

                    if len(y_tr_seq) < 20 or len(np.unique(y_tr_seq)) < 2:
                        continue

                    try:
                        probs = run_rnn_fold(
                            seq_tr_3d, tab_tr_seq, y_tr_seq.astype(float),
                            seq_te_3d, tab_te_seq, y_te_seq.astype(float),
                            cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"],
                            lr_, pw, model_class
                        )
                        y_pred = (probs >= 0.5).astype(int)
                        fold_f1s.append(f1_score(y_te_seq, y_pred, pos_label=1, zero_division=0))
                    except Exception:
                        pass

                avg_f1_cfg = float(np.mean(fold_f1s)) if fold_f1s else 0.0
                done_cfg += 1

                if avg_f1_cfg > best_avg_f1:
                    best_avg_f1 = avg_f1_cfg
                    best_config = {"cfg": cfg, "pw": pw, "lr": lr_}

    if best_config is None:
        print(f"  [{model_name}] search failed -- no valid configs")
        return None

    print(f"  [{model_name}] best config: {best_config}  avg_F1={best_avg_f1:.3f}")

    # Full 12-fold evaluation with best config
    print(f"  [{model_name}] running 12 folds ...")
    fold_recs = []
    fold_diag = []

    for fold_idx, test_year in enumerate(TEST_YEARS):
        train_mask = dates.year <= (test_year - 1)
        test_mask  = dates.year == test_year
        if not train_mask.any() or not test_mask.any():
            continue
        y_tr = y_all[train_mask]
        y_te = y_all[test_mask]
        if len(np.unique(y_tr)) < 2:
            continue

        sc_seq = StandardScaler()
        sc_tab = StandardScaler()
        seq_tr_s = sc_seq.fit_transform(seq_arr[train_mask])
        seq_te_s = sc_seq.transform(seq_arr[test_mask])
        tab_tr_s = sc_tab.fit_transform(tab_arr[train_mask])
        tab_te_s = sc_tab.transform(tab_arr[test_mask])

        seq_tr_3d  = build_sequences(seq_tr_s, SEQ_LEN)
        seq_te_3d  = build_sequences(seq_te_s, SEQ_LEN)
        y_tr_seq   = y_tr[SEQ_LEN - 1:]
        tab_tr_seq = tab_tr_s[SEQ_LEN - 1:]
        y_te_seq   = y_te[SEQ_LEN - 1:]
        tab_te_seq = tab_te_s[SEQ_LEN - 1:]

        if len(y_tr_seq) < 20 or len(np.unique(y_tr_seq)) < 2:
            continue

        try:
            cfg    = best_config["cfg"]
            probs  = run_rnn_fold(
                seq_tr_3d, tab_tr_seq, y_tr_seq.astype(float),
                seq_te_3d, tab_te_seq, y_te_seq.astype(float),
                cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"],
                best_config["lr"], best_config["pw"], model_class
            )
            y_pred = (probs >= 0.5).astype(int)

            # Train metrics (refit on train)
            probs_tr = run_rnn_fold(
                seq_tr_3d, tab_tr_seq, y_tr_seq.astype(float),
                seq_tr_3d, tab_tr_seq, y_tr_seq.astype(float),
                cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"],
                best_config["lr"], best_config["pw"], model_class
            )
            y_pred_tr = (probs_tr >= 0.5).astype(int)
            train_met = wf_metrics(y_tr_seq, y_pred_tr)
            test_met  = wf_metrics(y_te_seq, y_pred)
            gap_f1    = train_met["f1"] - test_met["f1"]

            diag_rows.append({
                "step": 2, "fold": fold_idx + 1, "test_year": test_year,
                "model": model_name,
                "train_recall": round(train_met["recall"], 4),
                "train_f1":     round(train_met["f1"],     4),
                "test_recall":  round(test_met["recall"],  4),
                "test_f1":      round(test_met["f1"],      4),
                "gap_f1":       round(gap_f1,              4),
            })
            fold_diag.append({"gap_f1": gap_f1})
            fold_probs_out.append((y_te_seq, probs))

            rec = dict(test_met)
            rec.update({"fold": fold_idx + 1, "test_year": test_year,
                        "model": model_name,
                        "best_params": json.dumps(best_config, default=str)})
            fold_recs.append(rec)
            weight_rows.append({"fold": fold_idx+1, "model": model_name,
                                 "best_weight": str(best_config["pw"]),
                                 "shift_ratio": "n/a"})

            elapsed = time.time() - t0_step
            print(f"    fold {fold_idx+1:2d} ({test_year})  "
                  f"f1={test_met['f1']:.3f}  gap={gap_f1:.3f}  "
                  f"{eta_str(elapsed, fold_idx+1, len(TEST_YEARS))}", flush=True)

        except Exception as e:
            print(f"    fold {fold_idx+1} ERROR: {e}")

    if fold_recs:
        a = agg(fold_recs)
        all_fold_rows.extend(fold_recs)
        avg_gap = np.mean([d["gap_f1"] for d in fold_diag]) if fold_diag else 0.0
        if avg_gap > 0.20:
            print(f"  WARNING: {model_name} REJECTED -- avg gap={avg_gap:.3f}")
            rejected_models.add(model_name)
        f1s = [r.get("f1", 0) for r in fold_recs]
        if float(np.std(f1s)) > 0.20:
            flagged_models.add(model_name)
        print(f"  [{model_name}] avg f1={a['f1'][0]:.3f}  avg gap={avg_gap:.3f}")

    return best_config


def _run_hmm(ds, FEAT_COLS, all_fold_rows, model_fold_probs,
             diag_rows, weight_rows, rejected_models, flagged_models, t0_step):
    """HMM: unsupervised shift detection via state transition."""
    print(f"\n  --- [HMM] grid: n_components in {{2,3,4}} x covariance in {{diag,full}} ---")

    X_all = ds[FEAT_COLS].values
    y_all = ds[TARGET_COL].values.astype(int)
    dates = ds.index

    hmm_configs = [{"n_components": n, "covariance_type": ct}
                   for n in [2, 3, 4] for ct in ["diag", "full"]]

    fold_recs  = []
    fold_probs = []
    fold_diag  = []

    for fold_idx, test_year in enumerate(TEST_YEARS):
        train_mask = dates.year <= (test_year - 1)
        test_mask  = dates.year == test_year
        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_te, y_te = X_all[test_mask],  y_all[test_mask]
        if len(X_tr) == 0 or len(X_te) == 0:
            continue

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        # Try all configs, pick best on training recall
        best_cfg_f1 = -1.0
        best_hmm    = None
        for cfg in hmm_configs:
            try:
                det = HMMShiftDetector(**cfg)
                det.fit(X_tr_s, y_tr)
                y_pred_tr = det.predict(X_tr_s)
                f1_tr = f1_score(y_tr, y_pred_tr, pos_label=1, zero_division=0)
                if f1_tr > best_cfg_f1:
                    best_cfg_f1 = f1_tr
                    best_hmm    = det
            except Exception:
                pass

        if best_hmm is None:
            continue

        y_prob_te = best_hmm.predict_proba(X_te_s)[:, 1]
        y_pred_te = (y_prob_te >= 0.5).astype(int)
        y_prob_tr = best_hmm.predict_proba(X_tr_s)[:, 1]
        y_pred_tr = (y_prob_tr >= 0.5).astype(int)
        train_met = wf_metrics(y_tr, y_pred_tr)
        test_met  = wf_metrics(y_te, y_pred_te)
        gap_f1    = train_met["f1"] - test_met["f1"]

        diag_rows.append({
            "step": 2, "fold": fold_idx+1, "test_year": test_year,
            "model": "HMM",
            "train_recall": round(train_met["recall"], 4),
            "train_f1":     round(train_met["f1"],     4),
            "test_recall":  round(test_met["recall"],  4),
            "test_f1":      round(test_met["f1"],      4),
            "gap_f1":       round(gap_f1,              4),
        })
        fold_diag.append({"gap_f1": gap_f1})
        fold_probs.append((y_te, y_prob_te))

        rec = dict(test_met)
        rec.update({"fold": fold_idx+1, "test_year": test_year,
                    "model": "HMM",
                    "best_params": json.dumps({"n_components": best_hmm.n_components,
                                               "cov": best_hmm.covariance_type})})
        fold_recs.append(rec)
        weight_rows.append({"fold": fold_idx+1, "model": "HMM",
                             "best_weight": "n/a", "shift_ratio": "n/a"})

        elapsed = time.time() - t0_step
        print(f"    fold {fold_idx+1:2d} ({test_year})  "
              f"f1={test_met['f1']:.3f}  gap={gap_f1:.3f}  "
              f"{eta_str(elapsed, fold_idx+1, len(TEST_YEARS))}", flush=True)

    if fold_recs:
        a = agg(fold_recs)
        all_fold_rows.extend(fold_recs)
        model_fold_probs["HMM"] = fold_probs
        avg_gap = np.mean([d["gap_f1"] for d in fold_diag]) if fold_diag else 0.0
        if avg_gap > 0.20:
            print(f"  WARNING: HMM REJECTED -- avg gap={avg_gap:.3f}")
            rejected_models.add("HMM")
        f1s = [r.get("f1", 0) for r in fold_recs]
        if float(np.std(f1s)) > 0.20:
            flagged_models.add("HMM")
        print(f"  [HMM] avg f1={a['f1'][0]:.3f}  avg gap={avg_gap:.3f}")


def _pick_best_model(df_s2, rejected_models):
    """Return model name with highest avg F1 among non-rejected models."""
    valid = df_s2[~df_s2["model"].isin(rejected_models)]
    if valid.empty:
        valid = df_s2
    agg_m = valid.groupby("model")["f1"].mean()
    return agg_m.idxmax() if not agg_m.empty else "LR"


def _plot_step2_model_comparison(fold_rows, rejected_models):
    df = pd.DataFrame(fold_rows)
    if df.empty:
        return
    agg_df = df.groupby("model")[["recall", "precision", "f1"]].mean().reset_index()
    n = len(agg_df)
    x = np.arange(n)
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), 5))
    ax.bar(x - w, agg_df["recall"],    w, label="Recall")
    ax.bar(x,     agg_df["precision"], w, label="Precision")
    ax.bar(x + w, agg_df["f1"],        w, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(agg_df["model"], rotation=30)
    ax.set_ylabel("Score"); ax.set_title("Step 2: Model Comparison (avg over 12 folds)")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    # Mark rejected
    for i, row in agg_df.iterrows():
        if row["model"] in rejected_models:
            ax.get_xticklabels()[i].set_color("red")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step2_model_comparison_barplot.png"), dpi=120)
    plt.close(fig)


def _plot_step2_confusion_matrices(fold_rows, rejected_models):
    df = pd.DataFrame(fold_rows)
    if df.empty:
        return
    models = df["model"].unique()
    n_models = len(models)
    ncols = min(5, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, mname in enumerate(models):
        ax = axes[i]
        mdf = df[df["model"] == mname]
        # We don't have individual predictions stored here; show avg metrics as text
        avg_f1  = mdf["f1"].mean()
        avg_rec = mdf["recall"].mean()
        avg_pre = mdf["precision"].mean()
        color   = "red" if mname in rejected_models else "black"
        title   = f"{mname}\nF1={avg_f1:.3f} R={avg_rec:.3f} P={avg_pre:.3f}"
        if mname in rejected_models:
            title += "\nREJECTED (overfit)"
        ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=9, color=color,
                transform=ax.transAxes)
        for sp in ax.spines.values():
            sp.set_color("red" if mname in rejected_models else "grey")
            sp.set_linewidth(2 if mname in rejected_models else 1)
        ax.set_xticks([]); ax.set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Step 2: Model Summary (avg across 12 folds)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step2_confusion_matrices.png"), dpi=120)
    plt.close(fig)


def _plot_step2_roc_curves(model_fold_probs, rejected_models):
    fig, ax = plt.subplots(figsize=(7, 6))
    for mname, probs_list in model_fold_probs.items():
        if not probs_list:
            continue
        y_true_all = np.concatenate([p[0] for p in probs_list])
        y_prob_all = np.concatenate([p[1] for p in probs_list])
        if len(np.unique(y_true_all)) < 2:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
            roc_auc     = auc(fpr, tpr)
            ls = "--" if mname in rejected_models else "-"
            ax.plot(fpr, tpr, ls, label=f"{mname} AUC={roc_auc:.3f}")
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Step 2: ROC Curves (aggregated across folds)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step2_roc_curves.png"), dpi=120)
    plt.close(fig)


# ===========================================================================
# STEP 3 -- Horizon Sweep
# ===========================================================================

def step3(prices, vix, base_feats, best_def_name, best_model_name, df_s2):
    print("\n" + "=" * 70)
    print(f"STEP 3 -- Horizon Sweep  [model={best_model_name}, def={best_def_name}]")
    print("=" * 70)
    t0_step = time.time()

    H_VALUES = [3, 5, 7, 10, 15, 20]
    reg_fn, thresholds, desc = REGIME_DEFS[best_def_name]

    # Get best params from Step 2 for this model (most frequent)
    s2_model = df_s2[df_s2["model"] == best_model_name]
    best_params_str = (s2_model["best_params"].mode().iloc[0]
                       if not s2_model.empty and "best_params" in s2_model.columns
                       else "{}")
    try:
        best_params = json.loads(best_params_str)
    except Exception:
        best_params = {}

    rows         = []
    cm_per_h     = {}

    for h_idx, h in enumerate(H_VALUES):
        elapsed = time.time() - t0_step
        print(f"\n  H={h}  {eta_str(elapsed, h_idx, len(H_VALUES))} ...", flush=True)

        ds = build_dataset(prices, vix, base_feats, reg_fn, thresholds, h)
        FEAT_COLS = [c for c in GROUP_B_BASE + ["VIX_dist_nearest",
                                                  "VIX_dist_upper",
                                                  "VIX_dist_lower"] if c in ds.columns]
        X_all = ds[FEAT_COLS].values
        y_all = ds[TARGET_COL].values.astype(int)
        dates = ds.index

        fold_recs_05  = []
        fold_recs_opt = []
        fold_cms      = []

        for fold_idx, test_year in enumerate(TEST_YEARS):
            train_mask = dates.year <= (test_year - 1)
            test_mask  = dates.year == test_year
            X_tr, y_tr = X_all[train_mask], y_all[train_mask]
            X_te, y_te = X_all[test_mask],  y_all[test_mask]
            if len(X_tr) == 0 or len(X_te) == 0 or len(np.unique(y_tr)) < 2:
                continue

            sc = StandardScaler()
            X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)
            kept, dropped = drop_correlated(X_tr_s, y_tr, FEAT_COLS)
            if dropped:
                kidx = [FEAT_COLS.index(f) for f in kept]
                X_tr_s, X_te_s = X_tr_s[:, kidx], X_te_s[:, kidx]

            try:
                m = _make_model_from_name(best_model_name, best_params, y_tr)
                m.fit(X_tr_s, y_tr)
                y_prob_te = m.predict_proba(X_te_s)[:, 1]
                y_prob_tr = m.predict_proba(X_tr_s)[:, 1]

                # threshold = 0.5
                y_pred_05 = (y_prob_te >= 0.5).astype(int)
                fold_recs_05.append(wf_metrics(y_te, y_pred_05))

                # optimal threshold from training PR curve
                thr_opt   = optimal_threshold_pr(y_tr, y_prob_tr)
                y_pred_opt = (y_prob_te >= thr_opt).astype(int)
                fold_recs_opt.append(wf_metrics(y_te, y_pred_opt))
                fold_cms.append(confusion_matrix(y_te, y_pred_opt, labels=[0, 1]))

                print(f"    fold {fold_idx+1:2d} ({test_year})  "
                      f"f1@0.5={fold_recs_05[-1]['f1']:.3f}  "
                      f"f1@opt={fold_recs_opt[-1]['f1']:.3f}  "
                      f"thr_opt={thr_opt:.3f}", flush=True)

            except Exception as e:
                print(f"    fold {fold_idx+1} ERROR: {e}")

        if fold_cms:
            cm_per_h[h] = sum(fold_cms)

        def _row(fold_recs, thr_label, h):
            if not fold_recs:
                return {}
            a = agg(fold_recs)
            return {
                "H": h, "threshold": thr_label,
                "avg_recall":    round(a["recall"][0],    4),
                "std_recall":    round(a["recall"][1],    4),
                "avg_precision": round(a["precision"][0], 4),
                "avg_f1":        round(a["f1"][0],        4),
                "std_f1":        round(a["f1"][1],        4),
                "avg_accuracy":  round(a["accuracy"][0],  4),
                "avg_far":       round(a["far"][0],       4),
            }
        rows.append(_row(fold_recs_05,  "0.5",     h))
        rows.append(_row(fold_recs_opt, "optimal", h))

    df = pd.DataFrame([r for r in rows if r])
    df.to_csv(os.path.join(OUT_DIR, "step3_horizon_sweep.csv"), index=False)

    _plot_step3_horizon(df, H_VALUES)
    _plot_step3_cm_per_h(cm_per_h, H_VALUES)

    # Best H for optimal threshold
    opt_df  = df[df["threshold"] == "optimal"]
    if not opt_df.empty:
        best_h_row = opt_df.loc[opt_df["avg_f1"].idxmax()]
        best_h     = int(best_h_row["H"])
        print(f"\n  Best H (optimal threshold): H={best_h}  F1={best_h_row['avg_f1']:.4f}")
    else:
        best_h = 5

    elapsed = time.time() - t0_step
    print(f"\n  Step 3 done ({elapsed:.0f}s)")
    return best_h, df


def _make_model_from_name(model_name, params, y_tr):
    """Reconstruct best model from name + params dict."""
    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())

    if model_name == "LR":
        c  = params.get("model__C", params.get("C", 0.001))
        cw = params.get("model__class_weight", params.get("class_weight", "balanced"))
        pen = params.get("model__penalty", params.get("penalty", "l2"))
        return LogisticRegression(C=c, class_weight=cw, penalty=pen,
                                  solver="saga", max_iter=2000,
                                  random_state=RANDOM_SEED)
    elif model_name == "RF":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 5),
            class_weight=params.get("class_weight", "balanced"),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            random_state=RANDOM_SEED)
    elif model_name == "SVM":
        return SVC(kernel="rbf", probability=True,
                   C=params.get("C", 1.0),
                   gamma=params.get("gamma", "scale"),
                   class_weight=params.get("class_weight", "balanced"),
                   random_state=RANDOM_SEED)
    elif model_name == "GB":
        return WeightedGB(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.05),
            shift_weight=params.get("shift_weight", 5.0))
    elif model_name == "MLP":
        return WeightedMLP(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (64, 32)),
            alpha=params.get("alpha", 0.001),
            shift_weight=params.get("shift_weight", 5.0))
    elif model_name == "XGB" and HAS_XGB:
        base_ratio = n_neg / max(n_pos, 1)
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.05),
            scale_pos_weight=params.get("scale_pos_weight", base_ratio),
            eval_metric="logloss", verbosity=0,
            random_state=RANDOM_SEED)
    else:
        return LogisticRegression(C=0.001, class_weight="balanced",
                                  solver="saga", max_iter=2000,
                                  random_state=RANDOM_SEED)


def _plot_step3_horizon(df, H_values):
    fig, ax = plt.subplots(figsize=(9, 5))
    for thr in ["0.5", "optimal"]:
        sub = df[df["threshold"] == thr].sort_values("H")
        if not sub.empty:
            ax.plot(sub["H"], sub["avg_f1"], marker="o", label=f"thr={thr}")
            ax.fill_between(sub["H"],
                            sub["avg_f1"] - sub.get("std_f1", 0),
                            sub["avg_f1"] + sub.get("std_f1", 0),
                            alpha=0.2)
    ax.set_xlabel("Horizon H"); ax.set_ylabel("Avg F1")
    ax.set_title(f"Step 3: Horizon Sweep")
    ax.legend(); ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step3_horizon_sweep.png"), dpi=120)
    plt.close(fig)


def _plot_step3_cm_per_h(cm_per_h, H_values):
    valid_H = [h for h in H_values if h in cm_per_h]
    if not valid_H:
        return
    n = len(valid_H)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i, h in enumerate(valid_H):
        cm = cm_per_h[h]
        sns.heatmap(cm, annot=True, fmt="d", ax=axes[i], cmap="Blues",
                    xticklabels=["No Shift", "Shift"],
                    yticklabels=["No Shift", "Shift"])
        axes[i].set_title(f"H={h}"); axes[i].set_xlabel("Predicted"); axes[i].set_ylabel("True")
    fig.suptitle("Step 3: Confusion Matrices per H (optimal threshold)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step3_confusion_matrices_per_H.png"), dpi=120)
    plt.close(fig)


# ===========================================================================
# STEP 4 -- Final Model Report
# ===========================================================================

def step4(prices, vix, base_feats, best_def_name, best_model_name, best_h, df_s2):
    print("\n" + "=" * 70)
    print(f"STEP 4 -- Final Model Report  "
          f"[def={best_def_name}, model={best_model_name}, H={best_h}]")
    print("=" * 70)
    t0_step = time.time()

    reg_fn, thresholds, desc = REGIME_DEFS[best_def_name]
    ds = build_dataset(prices, vix, base_feats, reg_fn, thresholds, best_h)

    FEAT_COLS = [c for c in GROUP_B_BASE + ["VIX_dist_nearest",
                                              "VIX_dist_upper",
                                              "VIX_dist_lower"] if c in ds.columns]
    X_all = ds[FEAT_COLS].values
    y_all = ds[TARGET_COL].values.astype(int)
    dates = ds.index

    s2_model = df_s2[df_s2["model"] == best_model_name]
    best_params_str = (s2_model["best_params"].mode().iloc[0]
                       if not s2_model.empty and "best_params" in s2_model.columns
                       else "{}")
    try:
        best_params = json.loads(best_params_str)
    except Exception:
        best_params = {}

    fold_rows = []
    all_y_true, all_y_pred, all_y_prob = [], [], []
    all_y_true_tr = []
    all_y_pred_tr = []
    fold_cms = []
    fold_f1s = []

    opt_thresholds = []

    for fold_idx, test_year in enumerate(TEST_YEARS):
        train_mask = dates.year <= (test_year - 1)
        test_mask  = dates.year == test_year
        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_te, y_te = X_all[test_mask],  y_all[test_mask]
        if len(X_tr) == 0 or len(X_te) == 0 or len(np.unique(y_tr)) < 2:
            continue

        sc = StandardScaler()
        X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)
        kept, dropped = drop_correlated(X_tr_s, y_tr, FEAT_COLS)
        if dropped:
            kidx = [FEAT_COLS.index(f) for f in kept]
            X_tr_s, X_te_s = X_tr_s[:, kidx], X_te_s[:, kidx]

        m = _make_model_from_name(best_model_name, best_params, y_tr)
        m.fit(X_tr_s, y_tr)

        y_prob_te = m.predict_proba(X_te_s)[:, 1]
        y_prob_tr = m.predict_proba(X_tr_s)[:, 1]
        thr_opt   = optimal_threshold_pr(y_tr, y_prob_tr)
        opt_thresholds.append(thr_opt)

        y_pred_te = (y_prob_te >= thr_opt).astype(int)
        y_pred_tr = (y_prob_tr >= thr_opt).astype(int)

        train_met = wf_metrics(y_tr, y_pred_tr)
        test_met  = wf_metrics(y_te, y_pred_te)
        gap_f1    = train_met["f1"] - test_met["f1"]

        fold_rows.append({
            "fold": fold_idx + 1, "test_year": test_year,
            **{f"test_{k}": round(v, 4) for k, v in test_met.items()},
            **{f"train_{k}": round(v, 4) for k, v in train_met.items()},
            "gap_f1": round(gap_f1, 4),
            "opt_threshold": round(thr_opt, 4),
        })
        fold_cms.append(confusion_matrix(y_te, y_pred_te, labels=[0, 1]))
        fold_f1s.append(test_met["f1"])
        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred_te.tolist())
        all_y_prob.extend(y_prob_te.tolist())
        all_y_true_tr.extend(y_tr.tolist())
        all_y_pred_tr.extend(y_pred_tr.tolist())

        print(f"    fold {fold_idx+1:2d} ({test_year})  "
              f"f1={test_met['f1']:.3f}  gap={gap_f1:.3f}  thr={thr_opt:.3f}", flush=True)

    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(os.path.join(OUT_DIR, "step4_final_model.csv"), index=False)

    if not df_folds.empty:
        metrics_cols = ["test_recall", "test_precision", "test_f1", "test_accuracy", "test_far"]
        means = df_folds[metrics_cols].mean()
        stds  = df_folds[metrics_cols].std()
    else:
        means = pd.Series({c: 0 for c in ["test_recall","test_precision","test_f1","test_accuracy","test_far"]})
        stds  = means.copy()

    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)
    y_prob_arr = np.array(all_y_prob)
    avg_thr    = float(np.mean(opt_thresholds)) if opt_thresholds else 0.5

    # ---- Plots ----
    # Confusion matrix
    if fold_cms:
        cm_agg = sum(fold_cms)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_agg, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Shift", "Shift"],
                    yticklabels=["No Shift", "Shift"])
        ax.set_title(f"Step 4 Final: Confusion Matrix (agg 12 folds)\n"
                     f"{best_def_name} | {best_model_name} | H={best_h}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "step4_confusion_matrix.png"), dpi=120)
        plt.close(fig)

    # PR curve
    if len(np.unique(y_true_arr)) > 1:
        prec, rec, thrs = precision_recall_curve(y_true_arr, y_prob_arr)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rec, prec, label=f"{best_model_name}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Step 4: Precision-Recall Curve (aggregated)")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "step4_precision_recall_curve.png"), dpi=120)
        plt.close(fig)

    # Per-fold barplot
    if fold_f1s:
        fig, ax = plt.subplots(figsize=(10, 4))
        fold_nums = [r["fold"] for r in fold_rows]
        ax.bar(fold_nums, fold_f1s, color="steelblue")
        ax.axhline(float(np.mean(fold_f1s)), color="red", ls="--",
                   label=f"Mean F1={np.mean(fold_f1s):.3f}")
        ax.set_xlabel("Fold (test year)"); ax.set_ylabel("F1")
        ax.set_xticks(fold_nums)
        ax.set_xticklabels([str(r["test_year"]) for r in fold_rows], rotation=45)
        ax.set_title("Step 4: F1 per Fold")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "step4_per_fold_barplot.png"), dpi=120)
        plt.close(fig)

    # Learning curve (Fold 6, Measure G)
    _plot_learning_curve(prices, vix, base_feats, reg_fn, thresholds,
                         best_h, best_model_name, best_params, FEAT_COLS,
                         fold_idx_to_use=5)

    # Shift timeline (representative test year = median F1 fold)
    if fold_rows:
        median_f1   = float(np.median(fold_f1s))
        best_fold_r = min(fold_rows,
                          key=lambda r: abs(r["test_f1"] - median_f1))
        rep_year    = best_fold_r["test_year"]
        _plot_shift_timeline(ds, X_all, y_all, dates, FEAT_COLS, best_params,
                             best_model_name, best_h, rep_year, avg_thr)

    # ---- Print comprehensive summary ----
    elapsed = time.time() - t0_step
    print(f"\n  Step 4 done ({elapsed:.0f}s)")

    print("\n" + "=" * 70)
    print("=== Plan 10 Final Results ===\n")
    print(f"Regime Definition: {best_def_name} ({desc})")
    print(f"Model:             {best_model_name}")
    print(f"Best Params:       {best_params_str[:120]}")
    print(f"Features:          {FEAT_COLS}  ({len(FEAT_COLS)} features)")
    print(f"Horizon:           H={best_h}")
    print(f"Threshold:         {avg_thr:.3f} (avg optimal across folds)")
    print()
    print("Performance (avg +/- std, 12 folds):")
    print(f"  Accuracy:  {means['test_accuracy']:.3f} +/- {stds['test_accuracy']:.3f}")
    print(f"  Recall:    {means['test_recall']:.3f} +/- {stds['test_recall']:.3f}")
    print(f"  Precision: {means['test_precision']:.3f} +/- {stds['test_precision']:.3f}")
    print(f"  F1:        {means['test_f1']:.3f} +/- {stds['test_f1']:.3f}")
    print(f"  FAR:       {means['test_far']:.3f} +/- {stds['test_far']:.3f}")

    plan9_f1  = 0.720   # Plan 9 best LR H=20
    plan7_f1  = 0.680   # Plan 7 RF H=10
    print(f"\nvs. Plan 9 best (4 regimes, LR, H=20): F1 change: {means['test_f1']-plan9_f1:+.3f}")
    print(f"vs. Plan 7 RF (4 regimes, H=10):       F1 change: {means['test_f1']-plan7_f1:+.3f}")

    if means["test_accuracy"] >= 0.95:
        finding = "Yes: >95% accuracy achieved with simplified regime definition."
    else:
        finding = f"No: best accuracy={means['test_accuracy']:.3f} < 95%."
    print(f"\nKey finding: {finding}")
    print("=" * 70)

    return df_folds


def _plot_learning_curve(prices, vix, base_feats, reg_fn, thresholds, h,
                         model_name, params, feat_cols, fold_idx_to_use=5):
    """Measure G: learning curve on Fold 6."""
    from sklearn.model_selection import learning_curve as sklearn_lc

    ds = build_dataset(prices, vix, base_feats, reg_fn, thresholds, h)
    FEAT_COLS_LC = [c for c in feat_cols if c in ds.columns]
    X_all = ds[FEAT_COLS_LC].values
    y_all = ds[TARGET_COL].values.astype(int)
    dates = ds.index

    if fold_idx_to_use >= len(TEST_YEARS):
        return
    test_year  = TEST_YEARS[fold_idx_to_use]
    train_mask = dates.year <= (test_year - 1)
    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    if len(X_tr) < 50 or len(np.unique(y_tr)) < 2:
        return

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)

    try:
        m = _make_model_from_name(model_name, params, y_tr)
        train_sizes, tr_scores, te_scores = sklearn_lc(
            m, X_tr_s, y_tr,
            train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
            cv=3, scoring="f1", random_state=RANDOM_SEED,
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(train_sizes, tr_scores.mean(axis=1), "o-", label="Train F1")
        ax.plot(train_sizes, te_scores.mean(axis=1), "s-", label="CV F1")
        ax.fill_between(train_sizes,
                        tr_scores.mean(axis=1) - tr_scores.std(axis=1),
                        tr_scores.mean(axis=1) + tr_scores.std(axis=1), alpha=0.2)
        ax.fill_between(train_sizes,
                        te_scores.mean(axis=1) - te_scores.std(axis=1),
                        te_scores.mean(axis=1) + te_scores.std(axis=1), alpha=0.2)
        ax.set_xlabel("Training size"); ax.set_ylabel("F1")
        ax.set_title(f"Step 4: Learning Curve (Fold {fold_idx_to_use+1}, {test_year})")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "step4_learning_curve.png"), dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  Learning curve error: {e}")


def _plot_shift_timeline(ds, X_all, y_all, dates, FEAT_COLS, best_params,
                         model_name, h, rep_year, avg_thr):
    """Timeline of actual vs predicted shifts for representative year."""
    train_mask = dates.year <= (rep_year - 1)
    test_mask  = dates.year == rep_year
    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    X_te, y_te = X_all[test_mask],  y_all[test_mask]
    if len(X_tr) == 0 or len(X_te) == 0 or len(np.unique(y_tr)) < 2:
        return

    sc = StandardScaler()
    X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)
    kept, dropped = drop_correlated(X_tr_s, y_tr, FEAT_COLS)
    if dropped:
        kidx = [FEAT_COLS.index(f) for f in kept]
        X_tr_s, X_te_s = X_tr_s[:, kidx], X_te_s[:, kidx]

    try:
        m = _make_model_from_name(model_name, best_params, y_tr)
        m.fit(X_tr_s, y_tr)
        y_prob = m.predict_proba(X_te_s)[:, 1]
        y_pred = (y_prob >= avg_thr).astype(int)

        test_dates = ds.index[test_mask]
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(test_dates, y_te, "k", linewidth=1.5, label="Actual Shift")
        axes[0].set_ylabel("Actual"); axes[0].legend(loc="upper right")
        axes[0].set_title(f"Step 4: Shift Timeline  [{rep_year}]  "
                          f"model={model_name}  H={h}")

        axes[1].plot(test_dates, y_pred, "steelblue", linewidth=1.5, label="Predicted Shift")
        axes[1].set_ylabel("Predicted"); axes[1].legend(loc="upper right")

        axes[2].plot(test_dates, y_prob, "orange", linewidth=1.0, label="Probability")
        axes[2].axhline(avg_thr, color="red", ls="--", label=f"Threshold={avg_thr:.3f}")
        axes[2].set_ylabel("Prob"); axes[2].legend(loc="upper right")
        axes[2].set_xlabel("Date")

        # Mark hits and misses
        hit_mask  = (y_pred == 1) & (y_te == 1)
        miss_mask = (y_pred == 0) & (y_te == 1)
        fa_mask   = (y_pred == 1) & (y_te == 0)
        for mask, color, label in [(hit_mask, "green", "Hit"),
                                   (miss_mask, "red", "Miss"),
                                   (fa_mask, "orange", "FA")]:
            idxs = np.where(mask)[0]
            if len(idxs) > 0:
                axes[0].scatter(test_dates[idxs], y_te[idxs], color=color,
                                s=30, zorder=5, label=label)
        axes[0].legend(loc="upper right", fontsize=7)

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "step4_shift_timeline.png"), dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  Shift timeline error: {e}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 70)
    print("Plan 10: Regime Definition Simplification + Full Weight Optimization")
    print(f"Output directory: {OUT_DIR}")
    print(f"Models: LR, RF, SVM, GB, MLP, XGB, LSTM, GRU, HMM")
    print("=" * 70)

    WALL_T0 = time.time()

    # ---- Load data ----
    prices, vix = load_raw_data()
    base_feats  = compute_features(prices, vix)
    print(f"\nData loaded: {len(vix)} VIX days, {len(prices)} price days")

    # ---- Step 1 ----
    best_def_name, df_s1 = step1(prices, vix, base_feats)

    wall_elapsed = time.time() - WALL_T0
    print(f"\nWall time after Step 1: {wall_elapsed/60:.1f} min")

    # ---- Step 2 ----
    best_model_name, all_fold_rows, rejected_models, df_s2 = step2(
        prices, vix, base_feats, best_def_name, best_h=5
    )

    wall_elapsed = time.time() - WALL_T0
    print(f"\nWall time after Step 2: {wall_elapsed/60:.1f} min")

    if wall_elapsed > 3 * 3600:
        print("\nWARNING: 3h wall time exceeded after Step 2. "
              "Skipping Steps 3 and 4 for runtime safety.")
        return

    # ---- Step 3 ----
    best_h, df_s3 = step3(prices, vix, base_feats, best_def_name, best_model_name, df_s2)

    wall_elapsed = time.time() - WALL_T0
    print(f"\nWall time after Step 3: {wall_elapsed/60:.1f} min")

    # ---- Step 4 ----
    df_s4 = step4(prices, vix, base_feats, best_def_name, best_model_name, best_h, df_s2)

    wall_elapsed = time.time() - WALL_T0
    print(f"\nTotal wall time: {wall_elapsed/60:.1f} min  ({wall_elapsed/3600:.2f} h)")
    print(f"\nAll outputs saved to: {OUT_DIR}")

    # List generated files
    files = [f for f in os.listdir(OUT_DIR) if os.path.isfile(os.path.join(OUT_DIR, f))]
    print(f"\nGenerated {len(files)} files:")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f:55s}  {size//1024:5d} KB")


if __name__ == "__main__":
    main()
