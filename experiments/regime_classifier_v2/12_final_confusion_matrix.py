"""
12_final_confusion_matrix.py

Final confusion matrix for the single best regime-shift classifier configuration
identified in 11_feature_extension_experiment.py.

Best configuration (highest mean test F1 across all feature sets, models, horizons):
  Feature set : I_F+T1  = [VIX_dist_nearest, crossing_pressure,
                             threshold_instability, time_since_last_threshold_touch]
  Model       : XGBoost
  Horizon     : H=7
  Mean test F1: 0.6998  (12-fold walk-forward, any_shift, min_persistence=1)

Protocol:
  - 12-fold expanding walk-forward (test years 2013-2024)
  - Training-only StandardScaler, correlation pruning, threshold optimisation
  - Fixed XGBoost parameters -- identical to script 11 (no new hyperparameter search)
  - All out-of-sample predictions aggregated across folds

Outputs (results/regime_classifier_v2/definitive/):
  final_confusion_matrix.png
  final_classification_report.txt
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR         = PROJECT_ROOT / "results" / "regime_classifier_v2" / "definitive"
DEFINITIVE_SCRIPT = (
    PROJECT_ROOT / "experiments" / "regime_classifier_v2" / "08_definitive_classifier.py"
)

# ---------------------------------------------------------------------------
# Best configuration  (hard-coded from experiment results)
# ---------------------------------------------------------------------------
BEST_FEATURE_SET_NAME = "I_F+T1"
BEST_FEATURE_COLS = [
    "VIX_dist_nearest",
    "crossing_pressure",
    "threshold_instability",
    "time_since_last_threshold_touch",
]
BEST_MODEL_KEY   = "XGB"
BEST_MODEL_NAME  = "XGBoost"
BEST_HORIZON     = 7
REPORTED_F1      = 0.6998          # mean test F1 from experiment

LABEL_VARIANT    = "any_shift"
MIN_PERSISTENCE  = 1

XGB_PARAMS = {
    "n_estimators":     200,
    "max_depth":        3,
    "learning_rate":    0.05,
    "scale_pos_weight": 2.0,
    "min_child_weight": 5,
    "reg_alpha":        0.1,
    "reg_lambda":       5.0,
    "subsample":        1.0,
    "colsample_bytree": 0.8,
}

CLASS_LABELS = ["No Shift", "Shift"]


# ===========================================================================
# Section 1 — New feature: time_since_last_threshold_touch
# ===========================================================================

def compute_time_since_last_threshold_touch(
    vix: pd.Series, thresholds: list, touch_distance: float = 1.0
) -> pd.Series:
    """
    For each day t: number of trading days since VIX last came within
    `touch_distance` points of any regime threshold.

    Uses only information available up to and including day t (no look-ahead).
    Returns NaN for days before the first threshold touch.
    """
    vix_arr    = vix.values
    n          = len(vix_arr)
    time_since = np.full(n, np.nan)
    last_touch = -1

    for i in range(n):
        dist = min(abs(vix_arr[i] - t) for t in thresholds)
        if dist < touch_distance:
            last_touch = i
        if last_touch >= 0:
            time_since[i] = float(i - last_touch)

    return pd.Series(time_since, index=vix.index, name="time_since_last_threshold_touch")


# ===========================================================================
# Section 2 — Load definitive module
# ===========================================================================

def load_definitive_module():
    if "seaborn" not in sys.modules:
        stub = types.ModuleType("seaborn")
        stub.heatmap = lambda *a, **kw: None        # type: ignore[attr-defined]
        sys.modules["seaborn"] = stub

    spec   = importlib.util.spec_from_file_location("regime_definitive", DEFINITIVE_SCRIPT)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(module)                 # type: ignore[union-attr]
    return module


# ===========================================================================
# Section 3 — Build XGBoost model
# ===========================================================================

def build_xgb_model(defmod):
    if not defmod.HAS_XGB:
        raise RuntimeError(
            "xgboost is not installed.  "
            "Install it with: pip install xgboost"
        )
    return defmod.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=defmod.RANDOM_SEED,
        verbosity=0,
        **XGB_PARAMS,
    )


# ===========================================================================
# Section 4 — Walk-forward evaluation
# ===========================================================================

def run_walk_forward(defmod, dataset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    12-fold expanding walk-forward.  Returns (y_all_true, y_all_pred) arrays
    containing all out-of-sample predictions concatenated across folds.

    Per fold (training data only):
      - correlation pruning (|r| > 0.90, keep higher F-statistic feature)
      - StandardScaler fit
      - optimal probability threshold search on training labels
    """
    feature_cols = [c for c in BEST_FEATURE_COLS if c in dataset.columns]
    missing = [c for c in BEST_FEATURE_COLS if c not in dataset.columns]
    if missing:
        raise RuntimeError(
            f"Missing feature columns in dataset: {missing}\n"
            f"Available columns: {list(dataset.columns)}"
        )

    y_all_true: list[int] = []
    y_all_pred: list[int] = []

    for fold_i, test_year in enumerate(defmod.TEST_YEARS):
        dates      = dataset.index
        train_mask = dates.year <= (test_year - 1)
        test_mask  = dates.year == test_year

        df_tr = dataset[train_mask]
        df_te = dataset[test_mask]
        if len(df_tr) == 0 or len(df_te) == 0:
            continue

        X_tr_raw = df_tr[feature_cols].values.astype(float)
        X_te_raw = df_te[feature_cols].values.astype(float)
        y_tr     = df_tr["shift_label"].values.astype(int)
        y_te     = df_te["shift_label"].values.astype(int)

        # Correlation pruning on training data only
        kept_idx, kept_names, dropped = defmod.drop_correlated_features(
            X_tr_raw, feature_cols, y_tr
        )
        X_tr_raw = X_tr_raw[:, kept_idx]
        X_te_raw = X_te_raw[:, kept_idx]

        # Scaling on training data only
        scaler   = defmod.StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr_raw)
        X_te_s   = scaler.transform(X_te_raw)

        model = build_xgb_model(defmod)
        model.fit(X_tr_s, y_tr)

        # Optimal threshold from training data only
        y_tr_prob     = model.predict_proba(X_tr_s)[:, 1]
        opt_threshold = defmod.find_optimal_threshold(y_tr, y_tr_prob)

        y_te_prob = model.predict_proba(X_te_s)[:, 1]
        y_te_pred = (y_te_prob >= 0.5).astype(int)   # standard 0.5 threshold for CM

        y_all_true.extend(y_te.tolist())
        y_all_pred.extend(y_te_pred.tolist())

        from sklearn.metrics import f1_score, precision_score, recall_score
        f1   = f1_score(y_te, y_te_pred, pos_label=1, zero_division=0)
        prec = precision_score(y_te, y_te_pred, pos_label=1, zero_division=0)
        rec  = recall_score(y_te, y_te_pred, pos_label=1, zero_division=0)
        print(
            f"  fold {fold_i + 1:2d} ({test_year})"
            f"  f1={f1:.3f}  prec={prec:.3f}  rec={rec:.3f}"
            f"  opt_thr={opt_threshold:.3f}"
            + (f"  dropped={dropped}" if dropped else "")
        )

    return np.array(y_all_true), np.array(y_all_pred)


# ===========================================================================
# Section 5 — Confusion matrix plot
# ===========================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    overall_f1: float,
    out_path: Path,
) -> None:
    """
    Plot and save a confusion matrix with raw counts and row-normalised
    percentages in each cell.

    Layout:
      - 2×2 grid, rows = true class, columns = predicted class
      - Each cell: count on first line, percentage (of true-class total) below
      - Color intensity proportional to row-normalised value
      - Professional styling with clear axis labels and informative title
    """
    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0

    fig, ax = plt.subplots(figsize=(7.5, 6.2))

    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")

    # Cell annotations: count + percentage
    n_classes = cm.shape[0]
    for r in range(n_classes):
        for c in range(n_classes):
            count   = int(cm[r, c])
            pct     = cm_pct[r, c]
            # Use white text on dark cells, dark text on light cells
            text_color = "white" if pct > 55 else "black"
            ax.text(
                c, r,
                f"{count:,}\n({pct:.1f}%)",
                ha="center", va="center",
                fontsize=14, fontweight="bold",
                color=text_color,
            )

    # Axes
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_LABELS, fontsize=12)
    ax.set_yticklabels(CLASS_LABELS, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=10)
    ax.set_ylabel("True Label", fontsize=13, labelpad=10)

    # Grid lines between cells
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    # Title
    n_total  = len(y_true)
    n_shifts = int(y_true.sum())
    shift_pct = n_shifts / n_total * 100
    ax.set_title(
        f"Out-of-Sample Confusion Matrix — Best Configuration\n"
        f"Model: {BEST_MODEL_NAME}  |  Feature Set: {BEST_FEATURE_SET_NAME}  |  H={BEST_HORIZON}\n"
        f"Overall F1 (shift class): {overall_f1:.4f}  |  "
        f"N={n_total:,}  ({shift_pct:.1f}% shifts)",
        fontsize=11, pad=14,
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalised %", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Annotation: row label
    for r, lbl in enumerate(CLASS_LABELS):
        ax.annotate(
            f"True: {lbl}",
            xy=(-0.18, r), xycoords=("axes fraction", "data"),
            fontsize=9, color="#555555", ha="right", va="center",
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
# Section 6 — Classification report
# ===========================================================================

def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    overall_f1: float,
    out_path: Path,
) -> None:
    """
    Save a detailed classification report (per-class precision/recall/F1,
    macro/weighted averages, confusion matrix) as a plain-text file.
    Also print to stdout.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_LABELS,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    n_total  = len(y_true)
    n_shifts = int(y_true.sum())

    lines = [
        "=" * 68,
        "FINAL CLASSIFICATION REPORT",
        "=" * 68,
        f"Model        : {BEST_MODEL_NAME}",
        f"Feature set  : {BEST_FEATURE_SET_NAME}  ({', '.join(BEST_FEATURE_COLS)})",
        f"Horizon      : H={BEST_HORIZON}",
        f"Label        : {LABEL_VARIANT}  (min_persistence={MIN_PERSISTENCE})",
        f"Protocol     : 12-fold expanding walk-forward, test years 2013-2024",
        f"Parameters   : fixed representative (no hyperparameter search)",
        "",
        f"Total OOS samples : {n_total:,}",
        f"Shift events      : {n_shifts:,}  ({n_shifts / n_total * 100:.1f}%)",
        f"No-shift events   : {n_total - n_shifts:,}  ({(n_total - n_shifts) / n_total * 100:.1f}%)",
        "",
        "-" * 68,
        "PER-CLASS METRICS (threshold = 0.5)",
        "-" * 68,
        report,
        "-" * 68,
        "CONFUSION MATRIX (rows=true, cols=predicted)",
        "-" * 68,
        f"                  Pred No Shift   Pred Shift",
        f"True No Shift  :  {cm[0, 0]:>12,}   {cm[0, 1]:>10,}",
        f"True Shift     :  {cm[1, 0]:>12,}   {cm[1, 1]:>10,}",
        "",
        f"Row-normalised (%):",
        f"                  Pred No Shift   Pred Shift",
    ]
    for r, lbl in enumerate(CLASS_LABELS):
        row_sum = cm[r].sum()
        pcts    = [f"{cm[r, c] / row_sum * 100:>10.1f}%" for c in range(2)]
        lines.append(f"True {lbl:10s}:  {pcts[0]}   {pcts[1]}")

    lines += [
        "",
        "-" * 68,
        f"Overall shift-class F1 (mean across 12 folds): {overall_f1:.4f}",
        f"(F1 computed on aggregated OOS pool above may differ slightly)",
        "=" * 68,
    ]

    text = "\n".join(lines)
    print(text)

    out_path.write_text(text, encoding="utf-8")
    print(f"\n  Saved: {out_path}")


# ===========================================================================
# Section 7 — Main
# ===========================================================================

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 68)
    print("12_final_confusion_matrix.py")
    print("=" * 68)
    print(f"Best configuration:")
    print(f"  Feature set : {BEST_FEATURE_SET_NAME}")
    print(f"  Features    : {BEST_FEATURE_COLS}")
    print(f"  Model       : {BEST_MODEL_NAME}")
    print(f"  Horizon     : H={BEST_HORIZON}")
    print(f"  Reported F1 : {REPORTED_F1:.4f}  (mean across 12 folds, from experiment 11)")
    print(f"Output dir  : {OUT_DIR}")
    print("=" * 68)

    # ---- Load data ---------------------------------------------------------
    defmod = load_definitive_module()

    prices, vix = defmod.load_raw_data()
    regime      = defmod.compute_regime_labels(vix)
    tenure      = defmod.compute_regime_tenure(regime)
    features_df = defmod.compute_all_features(prices, vix)

    # Compute the one new feature needed for Set I_F+T1
    tslt = compute_time_since_last_threshold_touch(vix, list(defmod.THRESHOLDS))
    features_ext = features_df.copy()
    features_ext["time_since_last_threshold_touch"] = tslt

    print(
        f"\ntime_since_last_threshold_touch computed "
        f"(NaN count: {tslt.isna().sum()} of {len(tslt)})"
    )

    # ---- Build dataset -----------------------------------------------------
    dataset = defmod.build_dataset(
        features_ext, regime, tenure, vix, BEST_HORIZON, MIN_PERSISTENCE
    )
    n_total  = len(dataset)
    n_shifts = int(dataset["shift_label"].sum())
    print(
        f"\nDataset: {n_total:,} rows after NaN drop  "
        f"({n_shifts:,} shifts = {n_shifts / n_total * 100:.1f}%)"
    )

    # Verify all required feature columns are present
    missing = [c for c in BEST_FEATURE_COLS if c not in dataset.columns]
    if missing:
        raise RuntimeError(
            f"Feature columns not found in dataset: {missing}\n"
            f"Available: {list(dataset.columns)}"
        )

    # ---- Walk-forward ------------------------------------------------------
    print(f"\nRunning 12-fold walk-forward ...")
    y_true, y_pred = run_walk_forward(defmod, dataset)

    # ---- Compute aggregated metrics ----------------------------------------
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1_agg   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    prec_agg = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_agg  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    print(f"\nAggregated OOS metrics (all {len(y_true):,} predictions):")
    print(f"  F1        : {f1_agg:.4f}")
    print(f"  Precision : {prec_agg:.4f}")
    print(f"  Recall    : {rec_agg:.4f}")

    # ---- Plot confusion matrix ---------------------------------------------
    cm_path     = OUT_DIR / "final_confusion_matrix.png"
    report_path = OUT_DIR / "final_classification_report.txt"

    print("\nGenerating confusion matrix ...")
    plot_confusion_matrix(y_true, y_pred, f1_agg, cm_path)

    print("\nGenerating classification report ...")
    save_classification_report(y_true, y_pred, REPORTED_F1, report_path)

    # ---- Git status --------------------------------------------------------
    print("\nGit status:")
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--short"],
            capture_output=True, text=True, timeout=15,
        )
        print(result.stdout if result.stdout else "  (clean)")
        if result.stderr:
            print(result.stderr)
    except Exception as exc:
        print(f"  (could not run git status: {exc})")

    # ---- Output summary ----------------------------------------------------
    print("\nOutput files:")
    print(f"  {cm_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
