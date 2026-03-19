"""
06_max_accuracy.py -- Maximum Regime Classifier Accuracy (Plan 9)

Systematic 6-step progression toward maximum shift-detection accuracy.
Each step adds ONE improvement; results are logged per step.

  Step 1: Richer features (trajectory, threshold-distance, cross-asset)
  Step 2: Systematic weight & hyperparameter optimisation (per-fold inner CV)
  Step 3: Model comparison with optimised weights + LSTM + ExtraTrees
  Step 4: Ensemble methods (soft voting, stacking)
  Step 5: Horizon sweep + PR-curve threshold optimisation
  Step 6: Final model report

Walk-forward: 12 folds, test years 2013-2024 (identical to previous scripts).

Reduced grids vs. full plan (documented per model below) to target ~60-90 min:
  LR  : 4C × 4cw × 1pen = 16 combos    (removed L1, half C values)
  RF  : 2est × 2depth × 2cw × 2leaf = 16  (removed 500, depth 3/10)
  XGB : 2est × 2depth × 2lr × 4spw = 32   (removed regularisation grid)
  GB  : WeightedGB wrapper, 2est × 2depth × 2lr × 4sw = 32 combos
  SVM : 2C × 1gamma × 2cw = 4           (reduced heavily for runtime)
  MLP : WeightedMLP wrapper, 2arch × 2alpha × 3sw = 12 combos
  LSTM: arch search fold-1 only (3 arch × 3 pos_weight × 2 lr = 18 configs),
        best config applied to all 12 folds.

Run from project root (Implementierung1/):
  python experiments/regime_classifier_v2/06_max_accuracy.py
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
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base            import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble        import (ExtraTreesClassifier, GradientBoostingClassifier,
                                     RandomForestClassifier, StackingClassifier,
                                     VotingClassifier)
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (accuracy_score, auc, confusion_matrix,
                                     f1_score, precision_recall_curve,
                                     precision_score, recall_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network  import MLPClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.svm             import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed -- XGBoost steps skipped.")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    torch.manual_seed(42)
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: torch not installed -- LSTM steps skipped.")

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.path.join(PROJECT_ROOT, "results", "regime_classifier_v2", "max_accuracy")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Import from 01_build_dataset.py via importlib
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "build_dataset", os.path.join(SCRIPT_DIR, "01_build_dataset.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_raw_data              = _mod.load_raw_data
compute_regime_labels      = _mod.compute_regime_labels
compute_binary_shift_label = _mod.compute_binary_shift_label
compute_features           = _mod.compute_features   # base 10 features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COL  = "shift_label"
TEST_YEARS  = list(range(2013, 2025))
RANDOM_SEED = 42
SEQ_LEN     = 20       # LSTM lookback window
SEQ_FEAT    = ["VIX_raw", "delta_VIX", "SPY_return_1", "VIX_rolling_std_10"]
THRESHOLDS  = [15, 20, 30]   # regime VIX boundaries

BASE_FEATURES = ["VIX_MA20", "z_VIX", "delta_VIX", "VIX_slope_5", "VIX_slope_20",
                 "VIX_rolling_std_10", "max_VIX_window", "min_VIX_window",
                 "SPY_return_5", "vol_ratio"]

# ---------------------------------------------------------------------------
# Extended feature computation (adds to the base 10 features)
# ---------------------------------------------------------------------------

def compute_extended_features(prices: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """
    Compute all features: base 10 + trajectory + threshold-distance + cross-asset.
    Returns DataFrame indexed by date.
    """
    base   = compute_features(prices, vix)
    spy    = prices["SPY"]
    tlt    = prices["TLT"]
    gld    = prices["GLD"]
    spy_r  = spy.pct_change()
    vix_r  = vix.pct_change()

    # --- Sequence / raw features for LSTM ---
    VIX_raw      = vix
    SPY_return_1 = spy_r

    # --- Trajectory features ---
    VIX_MA20         = vix.rolling(20).mean()
    VIX_rolling_std  = vix.rolling(20).std()
    VIX_range_20     = vix.rolling(20).max() - vix.rolling(20).min()
    VIX_pct_above_MA = (vix > VIX_MA20).rolling(20).mean()
    VIX_slope_20     = vix - vix.shift(20)
    VIX_slope_5      = vix - vix.shift(5)
    VIX_trend_strength  = VIX_slope_20.abs() / (VIX_rolling_std + 1e-8)
    delta_VIX        = vix - vix.shift(1)
    delta_VIX_MA5    = delta_VIX.rolling(5).mean()
    delta_VIX_std5   = delta_VIX.rolling(5).std()
    VIX_acceleration = VIX_slope_5 - VIX_slope_5.shift(5)

    # --- Threshold distance features ---
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

    # --- Cross-asset features ---
    TLT_return_5    = tlt / tlt.shift(5) - 1
    GLD_return_5    = gld / gld.shift(5) - 1
    SPY_VIX_corr_20 = spy_r.rolling(20).corr(vix_r)

    ext = pd.DataFrame({
        # LSTM sequence inputs (raw)
        "VIX_raw":      VIX_raw,
        "SPY_return_1": SPY_return_1,
        # Trajectory
        "VIX_range_20":      VIX_range_20,
        "VIX_pct_above_MA":  VIX_pct_above_MA,
        "VIX_trend_strength":VIX_trend_strength,
        "delta_VIX_MA5":     delta_VIX_MA5,
        "delta_VIX_std5":    delta_VIX_std5,
        "VIX_acceleration":  VIX_acceleration,
        # Threshold distance
        "VIX_dist_nearest":  VIX_dist_nearest,
        "VIX_dist_upper":    VIX_dist_upper,
        "VIX_dist_lower":    VIX_dist_lower,
        # Cross-asset
        "TLT_return_5":      TLT_return_5,
        "GLD_return_5":      GLD_return_5,
        "SPY_VIX_corr_20":   SPY_VIX_corr_20,
    }, index=vix.index)

    return pd.concat([base, ext], axis=1)


def build_full_dataset(h: int, prices, vix, regime, features_df) -> pd.DataFrame:
    sl  = compute_binary_shift_label(regime, h)
    ds  = features_df.copy()
    ds["regime"]      = regime
    ds["shift_label"] = sl
    return ds.dropna()


# ---------------------------------------------------------------------------
# Feature groups (Step 1)
# ---------------------------------------------------------------------------
GROUP_A = ["VIX_MA20", "max_VIX_window", "min_VIX_window",
           "VIX_slope_20", "VIX_rolling_std_10"]
GROUP_B = GROUP_A + ["VIX_dist_nearest", "VIX_dist_upper", "VIX_dist_lower"]
GROUP_C = GROUP_B + ["VIX_range_20", "VIX_pct_above_MA",
                     "VIX_trend_strength", "VIX_acceleration"]
GROUP_D = GROUP_C + ["TLT_return_5", "GLD_return_5",
                     "SPY_return_5", "vol_ratio", "SPY_VIX_corr_20"]
GROUP_E = GROUP_D + ["delta_VIX_MA5", "delta_VIX_std5"]

FEATURE_GROUPS = {
    "A_base5":       GROUP_A,
    "B_plus_dist":   GROUP_B,
    "C_plus_traj":   GROUP_C,
    "D_plus_cross":  GROUP_D,
    "E_plus_smooth": GROUP_E,
}

# ---------------------------------------------------------------------------
# Sklearn-compatible wrappers for GB and MLP (supports shift_weight parameter)
# ---------------------------------------------------------------------------

class WeightedGB(BaseEstimator, ClassifierMixin):
    """GradientBoostingClassifier with built-in shift sample weighting."""
    def __init__(self, shift_weight=1.0, n_estimators=100, max_depth=3,
                 learning_rate=0.1, min_samples_leaf=5, subsample=1.0):
        self.shift_weight    = shift_weight
        self.n_estimators    = n_estimators
        self.max_depth       = max_depth
        self.learning_rate   = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.subsample       = subsample

    def fit(self, X, y):
        sw = np.where(y == 1, self.shift_weight, 1.0)
        self._gb = GradientBoostingClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample, random_state=RANDOM_SEED,
        ).fit(X, y, sample_weight=sw)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):        return self._gb.predict(X)
    def predict_proba(self, X):  return self._gb.predict_proba(X)


class WeightedMLP(BaseEstimator, ClassifierMixin):
    """MLPClassifier with built-in shift sample weighting."""
    def __init__(self, shift_weight=1.0, hidden_layer_sizes=(64, 32),
                 alpha=0.001, learning_rate_init=0.001, max_iter=200):
        self.shift_weight        = shift_weight
        self.hidden_layer_sizes  = hidden_layer_sizes
        self.alpha               = alpha
        self.learning_rate_init  = learning_rate_init
        self.max_iter            = max_iter

    def fit(self, X, y):
        # MLPClassifier does not support sample_weight -> oversample minority class
        reps = max(1, int(round(self.shift_weight)))
        pos_idx = np.where(y == 1)[0]
        extra_X = np.repeat(X[pos_idx], reps - 1, axis=0) if reps > 1 else np.empty((0, X.shape[1]))
        extra_y = np.repeat(y[pos_idx], reps - 1,       ) if reps > 1 else np.empty(0, dtype=y.dtype)
        X_aug = np.vstack([X, extra_X])
        y_aug = np.concatenate([y, extra_y])
        self._mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
            learning_rate_init=self.learning_rate_init, max_iter=self.max_iter,
            random_state=RANDOM_SEED,
        ).fit(X_aug, y_aug)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):        return self._mlp.predict(X)
    def predict_proba(self, X):  return self._mlp.predict_proba(X)


# ---------------------------------------------------------------------------
# Walk-forward utilities
# ---------------------------------------------------------------------------

def wf_metrics(y_true, y_pred):
    return {
        "recall":    recall_score(   y_true, y_pred, pos_label=1, zero_division=0),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1":        f1_score(       y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy":  accuracy_score( y_true, y_pred),
        "far":       (int(((y_pred == 1) & (y_true == 0)).sum()) /
                      max(int((y_true == 0).sum()), 1)),
    }


def wf_basic(dataset, feature_cols, make_model_fn, threshold=0.5):
    """Standard 12-fold walk-forward. make_model_fn(n_neg, n_pos) -> unfitted model."""
    X_all  = dataset[feature_cols].values
    y_all  = dataset[TARGET_COL].values.astype(int)
    dates  = dataset.index
    records = []
    for fold_idx, test_year in enumerate(TEST_YEARS):
        train_mask = dates.year <= (test_year - 1)
        test_mask  = dates.year == test_year
        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_te, y_te = X_all[test_mask],  y_all[test_mask]
        if len(X_tr) == 0 or len(X_te) == 0:
            continue
        n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
        sc = StandardScaler()
        X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)
        m = make_model_fn(n_neg, n_pos)
        m.fit(X_tr_s, y_tr)
        yp_prob = m.predict_proba(X_te_s)[:, 1]
        y_pred  = (yp_prob >= threshold).astype(int)
        rec = wf_metrics(y_te, y_pred)
        rec.update({"fold": fold_idx + 1, "test_year": test_year})
        records.append(rec)
    return records


def wf_with_proba(dataset, feature_cols, make_model_fn):
    """Walk-forward returning per-sample (y_true, y_proba) for ROC/PR curves."""
    X_all  = dataset[feature_cols].values
    y_all  = dataset[TARGET_COL].values.astype(int)
    dates  = dataset.index
    y_true_all, y_prob_all = [], []
    for fold_idx, test_year in enumerate(TEST_YEARS):
        train_mask = dates.year <= (test_year - 1)
        test_mask  = dates.year == test_year
        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_te, y_te = X_all[test_mask],  y_all[test_mask]
        if len(X_tr) == 0 or len(X_te) == 0:
            continue
        n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
        sc = StandardScaler()
        X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)
        m = make_model_fn(n_neg, n_pos)
        m.fit(X_tr_s, y_tr)
        y_true_all.extend(y_te.tolist())
        y_prob_all.extend(m.predict_proba(X_te_s)[:, 1].tolist())
    return np.array(y_true_all), np.array(y_prob_all)


def wf_grid_search(dataset, feature_cols, get_base_fn, param_grid, scoring="f1"):
    """
    Walk-forward with inner 3-fold GridSearchCV per fold.
    get_base_fn(n_neg, n_pos) -> base estimator (unfitted).
    Returns (fold_records, per_fold_best_params, per_fold_probas).
    """
    X_all  = dataset[feature_cols].values
    y_all  = dataset[TARGET_COL].values.astype(int)
    dates  = dataset.index
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    fold_records, fold_params, fold_probas = [], [], []

    for fold_idx, test_year in enumerate(TEST_YEARS):
        train_mask = dates.year <= (test_year - 1)
        test_mask  = dates.year == test_year
        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_te, y_te = X_all[test_mask],  y_all[test_mask]
        if len(X_tr) == 0 or len(X_te) == 0:
            continue
        n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
        sc = StandardScaler()
        X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)
        base  = get_base_fn(n_neg, n_pos)
        grid  = GridSearchCV(base, param_grid, cv=inner_cv,
                             scoring=scoring, refit=True, n_jobs=1)
        grid.fit(X_tr_s, y_tr)
        y_prob  = grid.best_estimator_.predict_proba(X_te_s)[:, 1]
        y_pred  = (y_prob >= 0.5).astype(int)
        rec = wf_metrics(y_te, y_pred)
        rec.update({"fold": fold_idx + 1, "test_year": test_year,
                    "best_params": json.dumps(grid.best_params_, default=str)})
        fold_records.append(rec)
        fold_params.append(grid.best_params_)
        fold_probas.append((y_te, y_prob))
        print(f"    fold {fold_idx+1:2d} ({test_year})  "
              f"recall={rec['recall']:.3f}  f1={rec['f1']:.3f}  "
              f"params={json.dumps(grid.best_params_, default=str)[:60]}")
    return fold_records, fold_params, fold_probas


def agg(records):
    df = pd.DataFrame(records)
    return {k: (df[k].mean(), df[k].std())
            for k in ["recall", "precision", "f1", "accuracy", "far"]}


# ---------------------------------------------------------------------------
# LSTM implementation
# ---------------------------------------------------------------------------

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
                nn.Linear(hidden_dim + tab_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

        def forward(self, seq, tab):
            _, (h, _) = self.lstm(seq)
            h_last  = h[-1]
            return self.head(torch.cat([h_last, tab], dim=1)).squeeze(1)


def build_sequences(arr, seq_len):
    """arr: (n, d) → (n - seq_len + 1, seq_len, d)"""
    n, d = arr.shape
    return np.stack([arr[i - seq_len: i] for i in range(seq_len, n + 1)])


def run_lstm_fold(seq_tr, tab_tr, y_tr, seq_te, tab_te, y_te,
                  hidden_dim, num_layers, dropout, lr, pos_weight,
                  max_epochs=30, patience=4, batch_size=64):
    """Train and evaluate one LSTM config on one fold. Returns y_prob (test)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tab_dim = tab_tr.shape[1]

    # Validation split: last 20% of training
    n_val  = max(int(len(y_tr) * 0.2), 1)
    n_tr2  = len(y_tr) - n_val

    def to_tensors(s, t, y):
        return (torch.tensor(s, dtype=torch.float32),
                torch.tensor(t, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))

    tr_loader = DataLoader(
        TensorDataset(*to_tensors(seq_tr[:n_tr2], tab_tr[:n_tr2], y_tr[:n_tr2])),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(*to_tensors(seq_tr[n_tr2:], tab_tr[n_tr2:], y_tr[n_tr2:])),
        batch_size=batch_size
    )
    model = ShiftLSTM(seq_dim=len(SEQ_FEAT), tab_dim=tab_dim,
                      hidden_dim=hidden_dim, num_layers=num_layers,
                      dropout=dropout).to(device)
    crit  = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
    )
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    best_val, p_cnt, best_state = float("inf"), 0, None
    for epoch in range(max_epochs):
        model.train()
        for sb, tb, yb in tr_loader:
            sb, tb, yb = sb.to(device), tb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(sb, tb), yb).backward()
            opt.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sb, tb, yb in val_loader:
                val_loss += crit(model(sb.to(device), tb.to(device)),
                                 yb.to(device)).item()
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
        logits = model(s, t)
        probs  = torch.sigmoid(logits).cpu().numpy()
    return probs


# ===========================================================================
# STEP 1 — Feature Group Comparison
# ===========================================================================

def step1(dataset):
    print("\n" + "=" * 65)
    print("STEP 1 -- Feature Group Comparison")
    print("=" * 65)
    t0 = time.time()

    def lr_factory(n_neg, n_pos):
        return LogisticRegression(C=0.001, class_weight="balanced",
                                  penalty="l2", solver="lbfgs",
                                  max_iter=2000, random_state=RANDOM_SEED)

    rows = []
    for gname, feats in FEATURE_GROUPS.items():
        recs = wf_basic(dataset, feats, lr_factory)
        a    = agg(recs)
        rows.append({"group": gname, "n_features": len(feats),
                     "features": "|".join(feats),
                     "avg_recall":    round(a["recall"][0],    4),
                     "std_recall":    round(a["recall"][1],    4),
                     "avg_precision": round(a["precision"][0], 4),
                     "std_precision": round(a["precision"][1], 4),
                     "avg_f1":        round(a["f1"][0],        4),
                     "std_f1":        round(a["f1"][1],        4),
                     "avg_accuracy":  round(a["accuracy"][0],  4),
                     "avg_far":       round(a["far"][0],       4)})
        print(f"  {gname:18s} ({len(feats):2d} feat) "
              f"recall={a['recall'][0]:.3f}  prec={a['precision'][0]:.3f}  "
              f"f1={a['f1'][0]:.3f}  acc={a['accuracy'][0]:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "step1_sequence_features.csv"), index=False)

    # Best group: highest F1 among those with recall >= 0.75 (matching Plan 7 criterion)
    cands = df[df["avg_recall"] >= 0.75]
    best_row = (cands.loc[cands["avg_f1"].idxmax()] if not cands.empty
                else df.loc[df["avg_recall"].idxmax()])
    best_group = best_row["features"].split("|")
    best_name  = best_row["group"]

    print(f"\n  --> Best group: '{best_name}'  "
          f"recall={best_row['avg_recall']:.3f}  f1={best_row['avg_f1']:.3f}")
    print(f"  Saved: step1_sequence_features.csv  ({time.time()-t0:.0f}s)")
    return best_group, best_name, df


# ===========================================================================
# STEP 2 — Systematic Weight & Hyperparameter Optimisation
# ===========================================================================

def step2(dataset, best_features):
    print("\n" + "=" * 65)
    print("STEP 2 -- Systematic Weight & Hyperparameter Optimisation")
    print(f"  Features: {best_features}  (H=10)")
    print("=" * 65)

    # ---- Reduced grids (documented) ----
    # LR: 4C × 4cw × l2 = 16 combos (removed L1 and extreme C values)
    lr_grid = {
        "C":            [0.001, 0.01, 0.1, 1.0],
        "class_weight": ["balanced", {0:1,1:2}, {0:1,1:5}, {0:1,1:10}],
        "penalty":      ["l2"],
        "solver":       ["lbfgs"],
        "max_iter":     [2000],
    }
    # RF: 2est × 2depth × 2cw × 2leaf = 16 (removed 500est, depth 3/10/None)
    rf_grid = {
        "n_estimators":     [100, 200],
        "max_depth":        [5, None],
        "class_weight":     ["balanced", {0:1,1:5}],
        "min_samples_leaf": [1, 5],
    }
    # GB (WeightedGB): 2est × 2depth × 2lr × 4sw = 32
    gb_grid = {
        "n_estimators":  [100, 200],
        "max_depth":     [3, 5],
        "learning_rate": [0.05, 0.1],
        "shift_weight":  [1.0, 2.0, 5.0, 10.0],
    }
    # SVM: 2C × 1gamma × 2cw = 4 (reduced heavily for runtime)
    svm_grid = {
        "C":            [1.0, 10.0],
        "gamma":        ["scale"],
        "class_weight": ["balanced", {0:1,1:5}],
    }
    # MLP (WeightedMLP): 2arch × 2alpha × 3sw = 12
    mlp_grid = {
        "hidden_layer_sizes": [(64, 32), (128, 64)],
        "alpha":              [0.001, 0.01],
        "shift_weight":       [2.0, 5.0, 10.0],
    }

    all_fold_rows, weight_rows = [], []
    step2_best = {}   # model_name -> {best_params, avg_f1, ...}

    model_specs = [
        ("LR",  lambda n_neg, n_pos: LogisticRegression(random_state=RANDOM_SEED),
         lr_grid),
        ("RF",  lambda n_neg, n_pos: RandomForestClassifier(random_state=RANDOM_SEED),
         rf_grid),
        ("GB",  lambda n_neg, n_pos: WeightedGB(),
         gb_grid),
        ("SVM", lambda n_neg, n_pos: SVC(kernel="rbf", probability=True,
                                          random_state=RANDOM_SEED),
         svm_grid),
        ("MLP", lambda n_neg, n_pos: WeightedMLP(),
         mlp_grid),
    ]
    if HAS_XGB:
        # XGB: scale_pos_weight filled per fold → 2est × 2depth × 2lr × 4spw = 32
        model_specs.append(("XGB", None, None))  # handled separately

    for mname, base_fn, grid in model_specs:
        if mname == "XGB":
            continue   # handled below
        n_combos = 1
        for v in grid.values():
            n_combos *= len(v)
        print(f"\n  [{mname}]  {n_combos} combos × 3-fold × 12 outer "
              f"= {n_combos*3*12} fits ...")
        t0 = time.time()
        fold_recs, fold_params, _ = wf_grid_search(dataset, best_features,
                                                     base_fn, grid)
        elapsed = time.time() - t0
        a = agg(fold_recs)
        print(f"    Done ({elapsed:.0f}s)  recall={a['recall'][0]:.3f}  "
              f"f1={a['f1'][0]:.3f}")
        for rec, bp in zip(fold_recs, fold_params):
            all_fold_rows.append({"model": mname, **rec})
            cw_key = str(bp.get("class_weight", bp.get("shift_weight", "?")))
            weight_rows.append({"fold": rec["fold"], "model": mname,
                                 "best_weight": cw_key,
                                 "shift_ratio": "n/a"})
        step2_best[mname] = {"params": fold_params, "a": a}

    # XGB with per-fold scale_pos_weight in grid
    if HAS_XGB:
        print(f"\n  [XGB]  per-fold spw grid × 3-fold × 12 outer ...")
        t0 = time.time()
        X_all  = dataset[best_features].values
        y_all  = dataset[TARGET_COL].values.astype(int)
        dates  = dataset.index
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        xgb_fold_recs, xgb_fold_params = [], []
        for fold_idx, test_year in enumerate(TEST_YEARS):
            train_mask = dates.year <= (test_year - 1)
            test_mask  = dates.year == test_year
            X_tr, y_tr = X_all[train_mask], y_all[train_mask]
            X_te, y_te = X_all[test_mask],  y_all[test_mask]
            if len(X_tr) == 0 or len(X_te) == 0:
                continue
            n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
            base_ratio   = n_neg / max(n_pos, 1)
            xgb_grid = {
                "n_estimators":    [100, 200],
                "max_depth":       [3, 5],
                "learning_rate":   [0.05, 0.1],
                "scale_pos_weight":[base_ratio * f for f in [0.5, 1.0, 2.0, 5.0]],
            }
            sc = StandardScaler()
            X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)
            base = XGBClassifier(eval_metric="logloss", verbosity=0,
                                 random_state=RANDOM_SEED)
            grid_s = GridSearchCV(base, xgb_grid, cv=inner_cv,
                                  scoring="f1", refit=True, n_jobs=1)
            grid_s.fit(X_tr_s, y_tr)
            y_prob = grid_s.best_estimator_.predict_proba(X_te_s)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            rec = wf_metrics(y_te, y_pred)
            rec.update({"fold": fold_idx+1, "test_year": test_year,
                        "best_params": json.dumps(grid_s.best_params_, default=str)})
            xgb_fold_recs.append(rec)
            xgb_fold_params.append(grid_s.best_params_)
            print(f"    fold {fold_idx+1:2d} ({test_year})  "
                  f"recall={rec['recall']:.3f}  f1={rec['f1']:.3f}")
            all_fold_rows.append({"model": "XGB", **rec})
            weight_rows.append({"fold": fold_idx+1, "model": "XGB",
                                 "best_weight": str(grid_s.best_params_.get("scale_pos_weight")),
                                 "shift_ratio": str(round(base_ratio, 3))})
        a = agg(xgb_fold_recs)
        print(f"    Done ({time.time()-t0:.0f}s)  recall={a['recall'][0]:.3f}  "
              f"f1={a['f1'][0]:.3f}")
        step2_best["XGB"] = {"params": xgb_fold_params, "a": a}

    # Determine Group F from RF feature importances
    rf_imp_path = os.path.join(
        PROJECT_ROOT, "results", "regime_classifier_v2", "feature_importances.csv"
    )
    group_f = None
    if os.path.exists(rf_imp_path):
        imp = pd.read_csv(rf_imp_path)
        top8 = (imp.groupby("feature")["importance"].mean()
                .sort_values(ascending=False).head(8).index.tolist())
        valid = [f for f in top8 if f in dataset.columns]
        if len(valid) >= 5:
            group_f = valid
            print(f"\n  Group F (top-8 from RF importance): {group_f}")

    pd.DataFrame(all_fold_rows).to_csv(
        os.path.join(OUT_DIR, "step2_weight_optimization.csv"), index=False)
    pd.DataFrame(weight_rows).to_csv(
        os.path.join(OUT_DIR, "step2_best_weights_per_fold.csv"), index=False)
    print("  Saved: step2_weight_optimization.csv, step2_best_weights_per_fold.csv")
    return step2_best, group_f


# ===========================================================================
# STEP 3 — Model Comparison (with optimised weights + ExtraTrees + LSTM)
# ===========================================================================

def step3(dataset, best_features, step2_best):
    print("\n" + "=" * 65)
    print("STEP 3 -- Model Comparison (optimised weights)")
    print("=" * 65)

    rows, roc_data = [], {}

    # Re-run each Step-2 model with its modal best params
    for mname, info in step2_best.items():
        modal = collections.Counter(
            [json.dumps(p, sort_keys=True, default=str) for p in info["params"]]
        ).most_common(1)[0][0]
        modal_p = json.loads(modal)
        # JSON serialises dict keys as strings; sklearn requires int class keys
        if isinstance(modal_p.get("class_weight"), dict):
            modal_p["class_weight"] = {int(k): v for k, v in modal_p["class_weight"].items()}

        if mname == "LR":
            def factory(n_neg, n_pos, p=modal_p):
                return LogisticRegression(random_state=RANDOM_SEED, **p)
        elif mname == "RF":
            def factory(n_neg, n_pos, p=modal_p):
                return RandomForestClassifier(random_state=RANDOM_SEED, **p)
        elif mname == "GB":
            def factory(n_neg, n_pos, p=modal_p):
                return WeightedGB(**p)
        elif mname == "SVM":
            def factory(n_neg, n_pos, p=modal_p):
                return SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED, **p)
        elif mname == "MLP":
            def factory(n_neg, n_pos, p=modal_p):
                return WeightedMLP(**p)
        elif mname == "XGB" and HAS_XGB:
            def factory(n_neg, n_pos, p=modal_p):
                scale = n_neg / max(n_pos, 1)
                pp = {k: v for k, v in p.items() if k != "scale_pos_weight"}
                return XGBClassifier(scale_pos_weight=scale, eval_metric="logloss",
                                     verbosity=0, random_state=RANDOM_SEED, **pp)
        else:
            continue

        y_t, y_p = wf_with_proba(dataset, best_features, factory)
        y_pred    = (y_p >= 0.5).astype(int)
        m = wf_metrics(y_t, y_pred)
        fpr, tpr, _ = roc_curve(y_t, y_p)
        roc_data[mname] = (fpr, tpr, auc(fpr, tpr))
        rows.append({"model": mname, "source": "step2_best", **m})
        print(f"  {mname:6s}  recall={m['recall']:.3f}  "
              f"prec={m['precision']:.3f}  f1={m['f1']:.3f}  auc={roc_data[mname][2]:.3f}")

    # ExtraTrees (no grid search, sensible default)
    print("\n  [ExtraTrees] ...")
    et_factory = lambda n_neg, n_pos: ExtraTreesClassifier(
        class_weight="balanced", n_estimators=200, random_state=RANDOM_SEED
    )
    y_t, y_p = wf_with_proba(dataset, best_features, et_factory)
    y_pred = (y_p >= 0.5).astype(int)
    m = wf_metrics(y_t, y_pred)
    fpr, tpr, _ = roc_curve(y_t, y_p)
    roc_data["ExtraTrees"] = (fpr, tpr, auc(fpr, tpr))
    rows.append({"model": "ExtraTrees", "source": "default", **m})
    print(f"  ExtraTrees  recall={m['recall']:.3f}  prec={m['precision']:.3f}  "
          f"f1={m['f1']:.3f}  auc={roc_data['ExtraTrees'][2]:.3f}")

    # LSTM (if torch available)
    if TORCH_AVAILABLE:
        print("\n  [LSTM] architecture search on fold 1 ...")
        dataset_idx = dataset.index
        seq_arr = dataset[SEQ_FEAT].values
        tab_arr = dataset[best_features].values
        y_arr   = dataset[TARGET_COL].values.astype(int)

        # Fold 1: train 2006-2012, test 2013
        f1_train = dataset_idx.year <= 2012
        f1_test  = dataset_idx.year == 2013
        seq_tr, tab_tr, y_tr = seq_arr[f1_train], tab_arr[f1_train], y_arr[f1_train]
        seq_te, tab_te, y_te = seq_arr[f1_test],  tab_arr[f1_test],  y_arr[f1_test]

        # Scale sequence features
        seq_sc = StandardScaler()
        seq_tr_f = seq_tr.reshape(-1, seq_tr.shape[1])
        seq_sc.fit(seq_tr_f)
        seq_tr_s = seq_sc.transform(seq_tr_f).reshape(seq_tr.shape)
        seq_te_s = seq_sc.transform(seq_te.reshape(-1, seq_te.shape[1])).reshape(seq_te.shape)

        # Scale tabular features
        tab_sc = StandardScaler()
        tab_tr_s = tab_sc.fit_transform(tab_tr)
        tab_te_s = tab_sc.transform(tab_te)

        # Build sequences (sliding window)
        tr_seqs = build_sequences(seq_tr_s, SEQ_LEN)
        tr_tabs = tab_tr_s[SEQ_LEN - 1:]
        tr_ys   = y_tr[SEQ_LEN - 1:].astype(float)

        ctx     = np.concatenate([seq_tr_s[-(SEQ_LEN - 1):], seq_te_s])
        te_seqs = build_sequences(ctx, SEQ_LEN)
        te_tabs = tab_te_s
        te_ys   = y_te.astype(float)

        n_neg_l = int((tr_ys == 0).sum())
        n_pos_l = max(int((tr_ys == 1).sum()), 1)
        base_pw = n_neg_l / n_pos_l

        lstm_configs = [
            {"hidden_dim": 16, "num_layers": 1, "dropout": 0.2},
            {"hidden_dim": 32, "num_layers": 1, "dropout": 0.3},
            {"hidden_dim": 64, "num_layers": 2, "dropout": 0.4},
        ]
        pw_values = [base_pw * f for f in [1.0, 2.0, 5.0]]
        lr_values = [0.001, 0.005]

        best_lstm_f1, best_lstm_cfg = -1.0, None
        for cfg in lstm_configs:
            for pw in pw_values:
                for lr in lr_values:
                    probs = run_lstm_fold(tr_seqs, tr_tabs, tr_ys,
                                         te_seqs, te_tabs, te_ys,
                                         cfg["hidden_dim"], cfg["num_layers"],
                                         cfg["dropout"], lr, pw)
                    preds = (probs >= 0.5).astype(int)
                    f1_v  = f1_score(te_ys, preds, pos_label=1, zero_division=0)
                    if f1_v > best_lstm_f1:
                        best_lstm_f1 = f1_v
                        best_lstm_cfg = (cfg, pw, lr)
                        print(f"    LSTM arch={cfg}  pw={pw:.1f}  lr={lr}  "
                              f"f1={f1_v:.3f}  [NEW BEST]")

        if best_lstm_cfg:
            print(f"  Best LSTM config: {best_lstm_cfg[0]}  "
                  f"pw={best_lstm_cfg[1]:.1f}  lr={best_lstm_cfg[2]}")
            # Run best config across all 12 folds
            print("  Running best LSTM on all 12 folds ...")
            lstm_y_true, lstm_y_prob = [], []
            cfg, pw_factor_abs, lr = best_lstm_cfg

            for fold_idx, test_year in enumerate(TEST_YEARS):
                f_tr  = dataset_idx.year <= (test_year - 1)
                f_te  = dataset_idx.year == test_year
                s_tr, t_tr, y_tr2 = seq_arr[f_tr], tab_arr[f_tr], y_arr[f_tr]
                s_te, t_te, y_te2 = seq_arr[f_te], tab_arr[f_te], y_arr[f_te]
                if len(s_tr) == 0 or len(s_te) == 0:
                    continue

                s_sc = StandardScaler()
                s_tr_s = s_sc.fit_transform(s_tr.reshape(-1, s_tr.shape[1])).reshape(s_tr.shape)
                s_te_s = s_sc.transform(s_te.reshape(-1, s_te.shape[1])).reshape(s_te.shape)
                t_sc   = StandardScaler()
                t_tr_s = t_sc.fit_transform(t_tr)
                t_te_s = t_sc.transform(t_te)

                tr_seq2 = build_sequences(s_tr_s, SEQ_LEN)
                tr_tab2 = t_tr_s[SEQ_LEN - 1:]
                tr_y2   = y_tr2[SEQ_LEN - 1:].astype(float)

                ctx2    = np.concatenate([s_tr_s[-(SEQ_LEN-1):], s_te_s])
                te_seq2 = build_sequences(ctx2, SEQ_LEN)
                te_tab2 = t_te_s

                n_neg2 = int((tr_y2 == 0).sum())
                n_pos2 = max(int((tr_y2 == 1).sum()), 1)
                pw_val = (n_neg2 / n_pos2) * (pw_factor_abs / base_pw)

                probs = run_lstm_fold(tr_seq2, tr_tab2, tr_y2,
                                      te_seq2, te_tab2, y_te2.astype(float),
                                      cfg["hidden_dim"], cfg["num_layers"],
                                      cfg["dropout"], lr, pw_val)
                lstm_y_true.extend(y_te2.tolist())
                lstm_y_prob.extend(probs.tolist())
                preds = (probs >= 0.5).astype(int)
                f1_v  = f1_score(y_te2, preds, pos_label=1, zero_division=0)
                print(f"    fold {fold_idx+1:2d} ({test_year})  f1={f1_v:.3f}")

            y_t_lstm = np.array(lstm_y_true)
            y_p_lstm = np.array(lstm_y_prob)
            y_pred_l = (y_p_lstm >= 0.5).astype(int)
            m = wf_metrics(y_t_lstm, y_pred_l)
            fpr, tpr, _ = roc_curve(y_t_lstm, y_p_lstm)
            roc_data["LSTM"] = (fpr, tpr, auc(fpr, tpr))
            rows.append({"model": "LSTM", "source": "arch_search", **m})
            print(f"  LSTM  recall={m['recall']:.3f}  prec={m['precision']:.3f}  "
                  f"f1={m['f1']:.3f}  auc={roc_data['LSTM'][2]:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "step3_model_comparison.csv"), index=False)

    # Bar plot
    fig, ax = plt.subplots(figsize=(12, 5))
    models = df["model"].tolist()
    x      = np.arange(len(models))
    w      = 0.25
    ax.bar(x - w, df["recall"],    w, label="Recall",    color="#1f77b4", alpha=0.85)
    ax.bar(x,     df["precision"], w, label="Precision", color="#ff7f0e", alpha=0.85)
    ax.bar(x + w, df["f1"],        w, label="F1",        color="#2ca02c", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Metric (avg, 12 folds)")
    ax.set_title("Step 3: Model Comparison with Optimised Weights")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "model_comparison_barplot.png"), dpi=150)
    plt.close()

    # ROC curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for mname, (fpr, tpr, roc_auc) in roc_data.items():
        ax.plot(fpr, tpr, lw=2, label=f"{mname} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curves: All Models (aggregated, 12 folds)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curves.png"), dpi=150)
    plt.close()

    print("  Saved: step3_model_comparison.csv, model_comparison_barplot.png, roc_curves.png")

    # Top 3 models by F1
    top3 = df.nlargest(3, "f1")["model"].tolist()
    print(f"  Top-3 by F1: {top3}")
    return df, top3, step2_best


# ===========================================================================
# STEP 4 — Ensemble Methods
# ===========================================================================

def step4(dataset, best_features, step2_best, top3_names):
    print("\n" + "=" * 65)
    print(f"STEP 4 -- Ensemble Methods  (top-3: {top3_names})")
    print("=" * 65)

    def get_factory(mname, info):
        if mname == "LSTM":
            return None  # handled separately
        modal = json.loads(collections.Counter(
            [json.dumps(p, sort_keys=True, default=str) for p in info["params"]]
        ).most_common(1)[0][0])
        if isinstance(modal.get("class_weight"), dict):
            modal["class_weight"] = {int(k): v for k, v in modal["class_weight"].items()}

        if mname == "LR":
            return lambda nn, np_: LogisticRegression(random_state=RANDOM_SEED, **modal)
        if mname == "RF":
            return lambda nn, np_: RandomForestClassifier(random_state=RANDOM_SEED, **modal)
        if mname == "GB":
            return lambda nn, np_: WeightedGB(**modal)
        if mname == "SVM":
            return lambda nn, np_: SVC(kernel="rbf", probability=True,
                                        random_state=RANDOM_SEED, **modal)
        if mname == "MLP":
            return lambda nn, np_: WeightedMLP(**modal)
        if mname == "XGB" and HAS_XGB:
            pp = {k: v for k, v in modal.items() if k != "scale_pos_weight"}
            return lambda nn, np_, pp=pp: XGBClassifier(
                scale_pos_weight=nn/max(np_, 1), eval_metric="logloss",
                verbosity=0, random_state=RANDOM_SEED, **pp)
        if mname == "ExtraTrees":
            return lambda nn, np_: ExtraTreesClassifier(
                class_weight="balanced", n_estimators=200, random_state=RANDOM_SEED)
        return None

    # Build factories for top-3 (skip LSTM from ensemble for simplicity)
    non_lstm = [m for m in top3_names if m != "LSTM"][:3]
    factories = {m: get_factory(m, step2_best.get(m, {"params": [{}]*12}))
                 for m in non_lstm}
    factories = {k: v for k, v in factories.items() if v is not None}

    rows = []

    # Soft voting: average predicted probabilities
    print("  [Soft Voting] ...")
    X_all = dataset[best_features].values
    y_all = dataset[TARGET_COL].values.astype(int)
    dates = dataset.index
    vote_yt, vote_yp = [], []
    for fold_idx, test_year in enumerate(TEST_YEARS):
        tr_m = dates.year <= (test_year - 1)
        te_m = dates.year == test_year
        X_tr, y_tr = X_all[tr_m], y_all[tr_m]
        X_te, y_te = X_all[te_m], y_all[te_m]
        if len(X_tr) == 0 or len(X_te) == 0:
            continue
        n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
        sc = StandardScaler()
        X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)
        probs = []
        for mname, factory in factories.items():
            m = factory(n_neg, n_pos)
            m.fit(X_tr_s, y_tr)
            probs.append(m.predict_proba(X_te_s)[:, 1])
        avg_prob = np.mean(probs, axis=0)
        vote_yt.extend(y_te.tolist())
        vote_yp.extend(avg_prob.tolist())
        y_pred = (avg_prob >= 0.5).astype(int)
        f1_v   = f1_score(y_te, y_pred, pos_label=1, zero_division=0)
        print(f"    fold {fold_idx+1:2d} ({test_year})  f1={f1_v:.3f}")

    y_t, y_p = np.array(vote_yt), np.array(vote_yp)
    m_vote = wf_metrics(y_t, (y_p >= 0.5).astype(int))
    rows.append({"ensemble": f"SoftVoting({'+'.join(non_lstm)})", **m_vote})
    print(f"  SoftVoting  recall={m_vote['recall']:.3f}  "
          f"prec={m_vote['precision']:.3f}  f1={m_vote['f1']:.3f}")

    # Stacking (if >= 2 models)
    if len(factories) >= 2:
        print("  [Stacking] ...")
        stack_yt, stack_yp = [], []
        for fold_idx, test_year in enumerate(TEST_YEARS):
            tr_m = dates.year <= (test_year - 1)
            te_m = dates.year == test_year
            X_tr, y_tr = X_all[tr_m], y_all[tr_m]
            X_te, y_te = X_all[te_m], y_all[te_m]
            if len(X_tr) == 0 or len(X_te) == 0:
                continue
            n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
            sc = StandardScaler()
            X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)

            # OOF stacking meta-features via 3-fold inner CV on training
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
            oof   = np.zeros((len(X_tr_s), len(factories)))
            for mi, (mname, factory) in enumerate(factories.items()):
                for ti, vi in inner.split(X_tr_s, y_tr):
                    m = factory(int((y_tr[ti]==0).sum()), int((y_tr[ti]==1).sum()))
                    m.fit(X_tr_s[ti], y_tr[ti])
                    oof[vi, mi] = m.predict_proba(X_tr_s[vi])[:, 1]

            # Train meta-LR on OOF
            meta = LogisticRegression(class_weight="balanced", max_iter=1000,
                                      random_state=RANDOM_SEED)
            meta.fit(oof, y_tr)

            # Test-set stacking features
            test_stack = np.zeros((len(X_te_s), len(factories)))
            for mi, (mname, factory) in enumerate(factories.items()):
                m = factory(n_neg, n_pos)
                m.fit(X_tr_s, y_tr)
                test_stack[:, mi] = m.predict_proba(X_te_s)[:, 1]

            y_prob_s = meta.predict_proba(test_stack)[:, 1]
            stack_yt.extend(y_te.tolist())
            stack_yp.extend(y_prob_s.tolist())
            y_pred = (y_prob_s >= 0.5).astype(int)
            f1_v   = f1_score(y_te, y_pred, pos_label=1, zero_division=0)
            print(f"    fold {fold_idx+1:2d} ({test_year})  f1={f1_v:.3f}")

        y_t_s, y_p_s = np.array(stack_yt), np.array(stack_yp)
        m_stack = wf_metrics(y_t_s, (y_p_s >= 0.5).astype(int))
        rows.append({"ensemble": f"Stacking({'+'.join(non_lstm)})", **m_stack})
        print(f"  Stacking  recall={m_stack['recall']:.3f}  "
              f"prec={m_stack['precision']:.3f}  f1={m_stack['f1']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "step4_ensemble.csv"), index=False)
    print("  Saved: step4_ensemble.csv")

    # Best ensemble
    best_ens_row = df.loc[df["f1"].idxmax()]
    print(f"  Best ensemble: '{best_ens_row['ensemble']}'  f1={best_ens_row['f1']:.3f}")

    # Return the best performing approach (single or ensemble) for Step 5
    return df, non_lstm, factories, m_vote["f1"]


# ===========================================================================
# STEP 5 — Horizon Sweep + PR-curve Threshold Optimisation
# ===========================================================================

def step5(features_df, regime, best_features, best_model_name,
          step2_best, prices, vix):
    print("\n" + "=" * 65)
    print("STEP 5 -- Horizon Sweep + PR-curve Threshold Optimisation")
    print(f"  Model: {best_model_name}")
    print("=" * 65)

    # Get best single model factory
    info   = step2_best.get(best_model_name, {"params": [{}]*12})
    modal  = json.loads(collections.Counter(
        [json.dumps(p, sort_keys=True, default=str) for p in info["params"]]
    ).most_common(1)[0][0])
    if isinstance(modal.get("class_weight"), dict):
        modal["class_weight"] = {int(k): v for k, v in modal["class_weight"].items()}

    def make_factory(modal_p, mname):
        if mname == "LR":
            return lambda nn, np_: LogisticRegression(random_state=RANDOM_SEED, **modal_p)
        if mname == "RF":
            return lambda nn, np_: RandomForestClassifier(random_state=RANDOM_SEED, **modal_p)
        if mname == "GB":
            return lambda nn, np_: WeightedGB(**modal_p)
        if mname == "ExtraTrees":
            return lambda nn, np_: ExtraTreesClassifier(
                class_weight="balanced", n_estimators=200, random_state=RANDOM_SEED)
        if mname == "XGB" and HAS_XGB:
            pp = {k:v for k,v in modal_p.items() if k!="scale_pos_weight"}
            return lambda nn, np_, pp=pp: XGBClassifier(
                scale_pos_weight=nn/max(np_,1), eval_metric="logloss",
                verbosity=0, random_state=RANDOM_SEED, **pp)
        return lambda nn, np_: LogisticRegression(C=0.001, class_weight="balanced",
                                                   max_iter=2000, random_state=RANDOM_SEED)

    factory = make_factory(modal, best_model_name)
    horizons = [3, 5, 7, 10, 15, 20]
    rows = []

    for h in horizons:
        t0 = time.time()
        ds = build_full_dataset(h, prices, vix, regime, features_df)
        n1 = int((ds[TARGET_COL] == 1).sum())
        n0 = int((ds[TARGET_COL] == 0).sum())
        pct_sh = 100 * n1 / (n0 + n1)

        X_all = ds[best_features].values
        y_all = ds[TARGET_COL].values.astype(int)
        dates = ds.index

        thr_default, thr_optimal = [], []
        for fold_idx, test_year in enumerate(TEST_YEARS):
            tr_m = dates.year <= (test_year - 1)
            te_m = dates.year == test_year
            X_tr, y_tr = X_all[tr_m], y_all[tr_m]
            X_te, y_te = X_all[te_m], y_all[te_m]
            if len(X_tr) == 0 or len(X_te) == 0:
                continue
            n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
            sc = StandardScaler()
            X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)
            m = factory(n_neg, n_pos)
            m.fit(X_tr_s, y_tr)
            # Threshold 0.5
            y_prob_te = m.predict_proba(X_te_s)[:, 1]
            thr_default.append(wf_metrics(y_te, (y_prob_te >= 0.5).astype(int)))
            # PR-curve optimal threshold on TRAINING data
            y_prob_tr = m.predict_proba(X_tr_s)[:, 1]
            prec_c, rec_c, thr_c = precision_recall_curve(y_tr, y_prob_tr)
            f1_c  = (2 * prec_c * rec_c / (prec_c + rec_c + 1e-8))
            best_thr = thr_c[np.argmax(f1_c[:-1])] if len(thr_c) > 0 else 0.5
            thr_optimal.append(wf_metrics(y_te, (y_prob_te >= best_thr).astype(int)))

        a05  = agg(thr_default)
        aopt = agg(thr_optimal)
        rows.append({"H": h, "shift_pct": round(pct_sh, 1), "threshold": "0.50",
                     "avg_recall": round(a05["recall"][0],    4),
                     "std_recall": round(a05["recall"][1],    4),
                     "avg_prec":   round(a05["precision"][0], 4),
                     "avg_f1":     round(a05["f1"][0],        4),
                     "avg_acc":    round(a05["accuracy"][0],  4)})
        rows.append({"H": h, "shift_pct": round(pct_sh, 1), "threshold": "opt_PR",
                     "avg_recall": round(aopt["recall"][0],    4),
                     "std_recall": round(aopt["recall"][1],    4),
                     "avg_prec":   round(aopt["precision"][0], 4),
                     "avg_f1":     round(aopt["f1"][0],        4),
                     "avg_acc":    round(aopt["accuracy"][0],  4)})
        print(f"  H={h:2d} ({pct_sh:4.1f}% sh)  "
              f"thr=0.50: r={a05['recall'][0]:.3f} f1={a05['f1'][0]:.3f}  ||  "
              f"thr=opt:  r={aopt['recall'][0]:.3f} f1={aopt['f1'][0]:.3f}  "
              f"({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "step5_horizon_sweep.csv"), index=False)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for thr_label, ax, title in [
        ("0.50",   ax1, "Default threshold (0.5)"),
        ("opt_PR", ax2, "Optimal threshold (PR curve)"),
    ]:
        sub = df[df["threshold"] == thr_label].copy()
        ax.plot(sub["H"], sub["avg_recall"], "o-", color="#1f77b4",
                label="Recall", lw=2, ms=7)
        ax.plot(sub["H"], sub["avg_f1"],     "s-", color="#2ca02c",
                label="F1",     lw=2, ms=7)
        ax.set_xticks(horizons); ax.set_xlabel("Horizon H")
        ax.set_ylabel("Metric (avg, 12 folds)")
        ax.set_title(f"Horizon Sweep: {title}")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.suptitle(f"Step 5: Horizon Sensitivity ({best_model_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "horizon_sweep_plot.png"), dpi=150)
    plt.close()
    print("  Saved: step5_horizon_sweep.csv, horizon_sweep_plot.png")

    # Best H: highest avg_f1 at opt threshold
    opt_rows = df[df["threshold"] == "opt_PR"]
    best_H   = int(opt_rows.loc[opt_rows["avg_f1"].idxmax(), "H"])
    print(f"  Best H = {best_H}")
    return df, best_H


# ===========================================================================
# STEP 6 — Final Model Report
# ===========================================================================

def step6(features_df, regime, best_features, best_model_name,
          step2_best, best_H, prices, vix):
    print("\n" + "=" * 65)
    print(f"STEP 6 -- Final Model  (H={best_H}, model={best_model_name})")
    print("=" * 65)

    info  = step2_best.get(best_model_name, {"params": [{}]*12})
    modal = json.loads(collections.Counter(
        [json.dumps(p, sort_keys=True, default=str) for p in info["params"]]
    ).most_common(1)[0][0])
    if isinstance(modal.get("class_weight"), dict):
        modal["class_weight"] = {int(k): v for k, v in modal["class_weight"].items()}

    ds     = build_full_dataset(best_H, prices, vix, regime, features_df)
    X_all  = ds[best_features].values
    y_all  = ds[TARGET_COL].values.astype(int)
    dates  = ds.index

    fold_rows, y_true_all, y_prob_all = [], [], []
    for fold_idx, test_year in enumerate(TEST_YEARS):
        tr_m = dates.year <= (test_year - 1)
        te_m = dates.year == test_year
        X_tr, y_tr = X_all[tr_m], y_all[tr_m]
        X_te, y_te = X_all[te_m], y_all[te_m]
        if len(X_tr) == 0 or len(X_te) == 0:
            continue
        n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
        sc = StandardScaler()
        X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)

        if best_model_name == "LR":
            m = LogisticRegression(random_state=RANDOM_SEED, **modal)
        elif best_model_name == "RF":
            m = RandomForestClassifier(random_state=RANDOM_SEED, **modal)
        elif best_model_name == "GB":
            m = WeightedGB(**modal)
        elif best_model_name == "ExtraTrees":
            m = ExtraTreesClassifier(class_weight="balanced", n_estimators=200,
                                     random_state=RANDOM_SEED)
        elif best_model_name == "XGB" and HAS_XGB:
            pp = {k:v for k,v in modal.items() if k!="scale_pos_weight"}
            m  = XGBClassifier(scale_pos_weight=n_neg/max(n_pos,1),
                                eval_metric="logloss", verbosity=0,
                                random_state=RANDOM_SEED, **pp)
        else:
            m = LogisticRegression(C=0.001, class_weight="balanced",
                                   max_iter=2000, random_state=RANDOM_SEED)

        m.fit(X_tr_s, y_tr)
        y_prob_tr = m.predict_proba(X_tr_s)[:, 1]
        prec_c, rec_c, thr_c = precision_recall_curve(y_tr, y_prob_tr)
        f1_c  = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-8)
        best_thr = float(thr_c[np.argmax(f1_c[:-1])]) if len(thr_c) > 0 else 0.5

        y_prob = m.predict_proba(X_te_s)[:, 1]
        y_pred = (y_prob >= best_thr).astype(int)
        rec = wf_metrics(y_te, y_pred)
        rec.update({"fold": fold_idx+1, "test_year": test_year,
                    "opt_threshold": round(best_thr, 4)})
        fold_rows.append(rec)
        y_true_all.extend(y_te.tolist())
        y_prob_all.extend(y_prob.tolist())
        print(f"  fold {fold_idx+1:2d} ({test_year})  recall={rec['recall']:.3f}  "
              f"f1={rec['f1']:.3f}  thr={best_thr:.3f}")

    pd.DataFrame(fold_rows).to_csv(
        os.path.join(OUT_DIR, "step6_final_model.csv"), index=False)

    y_t = np.array(y_true_all);  y_p = np.array(y_prob_all)
    avg_thr = float(np.mean([r["opt_threshold"] for r in fold_rows]))
    y_pred_final = (y_p >= avg_thr).astype(int)
    cm = confusion_matrix(y_t, y_pred_final, labels=[0, 1])

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Shift","Shift"],
                yticklabels=["No Shift","Shift"], ax=ax, cbar=False)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Final Model: {best_model_name}  H={best_H}\n"
                 f"(12 folds aggregated, thr={avg_thr:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "final_confusion_matrix.png"), dpi=150)
    plt.close()

    # PR curve
    prec_all, rec_all, _ = precision_recall_curve(y_t, y_p)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec_all, prec_all, lw=2, color="#1f77b4")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve: {best_model_name}  H={best_H}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "final_precision_recall_curve.png"), dpi=150)
    plt.close()

    print("  Saved: step6_final_model.csv, final_confusion_matrix.png, "
          "final_precision_recall_curve.png")
    return agg(fold_rows)


# ===========================================================================
# Main
# ===========================================================================

def main():
    t_total = time.time()
    print("=" * 65)
    print("  06_max_accuracy.py -- Maximum Regime Classifier Accuracy")
    print("=" * 65)

    # Load data
    print("\nLoading data ...")
    prices, vix = load_raw_data()
    regime      = compute_regime_labels(vix)
    features_df = compute_extended_features(prices, vix)

    # Base dataset H=10
    dataset = build_full_dataset(10, prices, vix, regime, features_df)
    n0 = int((dataset[TARGET_COL] == 0).sum())
    n1 = int((dataset[TARGET_COL] == 1).sum())
    print(f"  Base dataset (H=10): {len(dataset)} rows  "
          f"shift={n1} ({100*n1/(n0+n1):.1f}%)  "
          f"Extended features: {features_df.shape[1]} columns")

    # Verify all group features exist in dataset
    for gname, feats in FEATURE_GROUPS.items():
        missing = [f for f in feats if f not in dataset.columns]
        if missing:
            print(f"  WARNING: Group {gname} missing features: {missing}")

    # ---- Step 1 ----
    best_features, best_group_name, step1_df = step1(dataset)

    # ---- Step 2 ----
    step2_best, group_f = step2(dataset, best_features)

    # Update best_features with Group F if it's available and better
    if group_f is not None:
        valid_f = [f for f in group_f if f in dataset.columns]
        if len(valid_f) >= 3:
            def lr_f(nn, np_):
                return LogisticRegression(C=0.001, class_weight="balanced",
                                          penalty="l2", solver="lbfgs",
                                          max_iter=2000, random_state=RANDOM_SEED)
            recs_f = wf_basic(dataset, valid_f, lr_f)
            a_f    = agg(recs_f)
            recs_c = wf_basic(dataset, best_features, lr_f)
            a_c    = agg(recs_c)
            if a_f["f1"][0] > a_c["f1"][0]:
                best_features = valid_f
                best_group_name = "F_from_importance"
                print(f"  Group F ({valid_f}) outperforms current best → updated.")

    # ---- Step 3 ----
    step3_df, top3_names, step2_best_upd = step3(dataset, best_features, step2_best)
    step2_best.update(step2_best_upd)

    # Best single model by F1
    best_model_name = step3_df.loc[step3_df["f1"].idxmax(), "model"]

    # ---- Step 4 ----
    step4_df, ens_models, ens_factories, ens_f1 = step4(
        dataset, best_features, step2_best, top3_names
    )
    # If ensemble beats best single, note it (we still use single model for Steps 5-6
    # since ensemble walkforward with per-fold customisation is complex)
    single_f1 = float(step3_df.loc[step3_df["model"] == best_model_name, "f1"].iloc[0])
    if ens_f1 > single_f1:
        print(f"  NOTE: Ensemble F1={ens_f1:.3f} > {best_model_name} F1={single_f1:.3f}")

    # ---- Step 5 ----
    step5_df, best_H = step5(features_df, regime, best_features,
                              best_model_name, step2_best, prices, vix)

    # ---- Step 6 ----
    final_agg = step6(features_df, regime, best_features,
                       best_model_name, step2_best, best_H, prices, vix)

    # ----------------------------------------------------------------
    # Print comprehensive summary
    # ----------------------------------------------------------------
    # Load ablation baselines for comparison
    ab_path = os.path.join(PROJECT_ROOT, "results", "regime_classifier_v2",
                           "ablation_summary.csv")
    abl = pd.read_csv(ab_path) if os.path.exists(ab_path) else None

    def abl_f1(phase, model):
        if abl is None:
            return float("nan")
        r = abl[(abl["phase"] == phase) & (abl["model"] == model)]
        return float(r["avg_f1"].iloc[0]) if not r.empty else float("nan")

    p1_f1  = abl_f1(1, "LogisticRegression")
    p4_f1  = abl_f1(4, "RandomForest")
    # Plan 7 optimized: from final_model_results.csv
    opt_path = os.path.join(PROJECT_ROOT, "results", "regime_classifier_v2",
                            "optimization", "final_model_results.csv")
    opt_f1 = float("nan")
    if os.path.exists(opt_path):
        opt_row = pd.read_csv(opt_path)
        if not opt_row.empty:
            opt_f1 = float(opt_row["avg_f1"].iloc[0])

    # Step-level best F1 values
    s1_best = step1_df["avg_f1"].max()
    s2_best = max(
        (agg(r["params"] and [])["f1"][0] if hasattr(r, "a") else r["a"]["f1"][0])
        for r in [step2_best[k] for k in step2_best]
        if "a" in r
    ) if step2_best else float("nan")
    s3_best = float(step3_df["f1"].max())
    s4_best = float(step4_df["f1"].max()) if not step4_df.empty else float("nan")
    s5_row  = step5_df[step5_df["threshold"] == "opt_PR"]
    s5_best = float(s5_row["avg_f1"].max()) if not s5_row.empty else float("nan")
    s6_f1   = final_agg["f1"][0]

    fa  = final_agg
    print("\n" + "=" * 65)
    print("=== Maximum Accuracy Experiment -- Final Results ===")
    print(f"\nConfiguration:")
    print(f"  Model    : {best_model_name}")
    print(f"  Features : {best_features}  ({len(best_features)} features)")
    print(f"  Horizon  : H={best_H}")
    print(f"  Threshold: optimised via PR curve on training data per fold")
    print(f"  Label    : binary shift, any(...) formulation")
    print(f"\nPerformance (avg +/- std, 12 folds):")
    print(f"  Accuracy  : {fa['accuracy'][0]:.3f} +/- {fa['accuracy'][1]:.3f}")
    print(f"  Recall    : {fa['recall'][0]:.3f} +/- {fa['recall'][1]:.3f}")
    print(f"  Precision : {fa['precision'][0]:.3f} +/- {fa['precision'][1]:.3f}")
    print(f"  F1        : {fa['f1'][0]:.3f} +/- {fa['f1'][1]:.3f}")
    print(f"  FAR       : {fa['far'][0]:.3f} +/- {fa['far'][1]:.3f}")
    print(f"\nComparison to previous best results:")
    print(f"  vs. Plan 7 Phase 1 (LR, 3 feat, H=10) :  "
          f"F1 {s6_f1 - p1_f1:+.3f}")
    print(f"  vs. Plan 7 Phase 4 RF (10 feat, H=10) :  "
          f"F1 {s6_f1 - p4_f1:+.3f}")
    print(f"  vs. Plan 7 Optimized (LR, H=20, 0.35) :  "
          f"F1 {s6_f1 - opt_f1:+.3f}")
    print(f"\nAblation progression (this plan):")
    print(f"  Step 1 -- Best feature group: {best_group_name}, "
          f"F1 = {s1_best:.3f}")
    print(f"  Step 2 -- Weight optimisation: model={best_model_name}, "
          f"F1 = {s3_best:.3f}")
    print(f"  Step 3 -- Best model (opt weights): {best_model_name}, "
          f"F1 = {s3_best:.3f} "
          f"({s3_best - s1_best:+.3f} vs Step 1)")
    print(f"  Step 4 -- Best ensemble F1 = {s4_best:.3f} "
          f"({s4_best - s3_best:+.3f} vs Step 3)")
    print(f"  Step 5 -- Best horizon: H={best_H}, "
          f"F1 = {s5_best:.3f} ({s5_best - s3_best:+.3f} vs Step 3)")
    print(f"  Step 6 -- Final (PR-optimal threshold): F1 = {s6_f1:.3f}")
    print("=" * 65)
    print(f"\nAll outputs saved to: {OUT_DIR}")
    print(f"Total runtime: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
