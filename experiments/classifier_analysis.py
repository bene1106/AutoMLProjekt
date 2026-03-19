"""
experiments/classifier_analysis.py
===================================
Part 1: Regime Classifier Deep-Dive

Task 1.1 -- Naive Baseline Measurement
Task 1.2 -- Feature Ablation Study (Feature Sets A-G)
Task 1.3 -- Model Comparison (LR, RF, XGBoost, GradientBoosting)
Task 1.4 -- Confidence Analysis

Run from Implementierung1/ as:
    python -m experiments.classifier_analysis
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
import xgboost as xgb

from regime_algo_selection.data.loader   import load_data
from regime_algo_selection.data.features import (
    compute_asset_features, compute_vix_features, compute_cross_asset_features,
)
from regime_algo_selection.regimes.ground_truth import compute_regime_labels
from regime_algo_selection.config import TRAIN_END, VAL_END, REGIME_NAMES, RESULTS_DIR, RANDOM_SEED

# ── Helpers ────────────────────────────────────────────────────────────────────

def _sharpe_series(arr):
    if arr.std() < 1e-12:
        return 0.0
    return float((arr.mean() / arr.std()) * np.sqrt(252))

def _recall_per_regime(y_true, y_pred):
    recalls = {}
    for r in [1, 2, 3, 4]:
        mask = y_true == r
        if mask.sum() == 0:
            recalls[r] = float("nan")
        else:
            recalls[r] = float((y_pred[mask] == r).mean())
    return recalls

def _f1_per_regime(y_true, y_pred):
    f1s = {}
    for r in [1, 2, 3, 4]:
        f1s[r] = float(f1_score(y_true, y_pred, labels=[r], average="macro",
                                zero_division=0))
    return f1s

def _eval_model(model, scaler, X_test_raw, y_test, use_scale=True):
    """Evaluate a fitted model. Returns dict of metrics."""
    valid = X_test_raw.notna().all(axis=1)
    X_v   = X_test_raw.loc[valid]
    y_v   = y_test.loc[valid]
    if use_scale:
        X_arr = scaler.transform(X_v)
    else:
        X_arr = X_v.values
    y_pred = model.predict(X_arr)
    acc    = accuracy_score(y_v, y_pred)
    f1s    = _f1_per_regime(y_v.values, y_pred)
    recs   = _recall_per_regime(y_v.values, y_pred)
    cm     = confusion_matrix(y_v, y_pred, labels=[1, 2, 3, 4])
    return {"accuracy": acc, "f1": f1s, "recall": recs, "confusion": cm,
            "y_pred": y_pred, "y_true": y_v.values}


def _fit_logreg(X_train, y_train):
    scaler = StandardScaler()
    X_arr  = scaler.fit_transform(X_train.dropna())
    y_arr  = y_train.loc[X_train.dropna().index].values
    model  = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial",
                                solver="lbfgs", random_state=RANDOM_SEED)
    model.fit(X_arr, y_arr)
    return model, scaler


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all():
    data   = load_data()
    prices = data["prices"]
    vix    = data["vix"]

    # All feature sets
    vix_feats    = compute_vix_features(vix)
    asset_feats  = compute_asset_features(prices)
    cross_feats  = compute_cross_asset_features(prices)
    regime_labels= compute_regime_labels(vix)

    # Flatten asset features to wide format
    af = asset_feats.copy()
    af.columns = [f"{a}_{f}" for a, f in af.columns]

    # Sub-groups of asset features
    ret_cols  = [c for c in af.columns if any(c.endswith(s) for s in ["_ret_1d","_ret_5d","_ret_20d"])]
    vol_cols  = [c for c in af.columns if any(c.endswith(s) for s in ["_vol_20d","_vol_60d"])]
    mom_cols  = [c for c in af.columns if any(c.endswith(s) for s in ["_mom_20d","_mom_60d","_mom_120d"])]

    # Align all to common index
    idx    = vix_feats.index
    vix_df = vix_feats.loc[idx]
    af_df  = af.reindex(idx)
    cr_df  = cross_feats.reindex(idx)

    # Feature sets
    set_A = vix_df
    set_B = pd.concat([vix_df, af_df[ret_cols]], axis=1)
    set_C = pd.concat([vix_df, af_df[ret_cols], af_df[vol_cols]], axis=1)
    set_D = pd.concat([vix_df, af_df[ret_cols], af_df[vol_cols], af_df[mom_cols]], axis=1)
    set_E = af_df[ret_cols + vol_cols + mom_cols]
    set_F = pd.concat([vix_df, cr_df], axis=1)
    set_G = pd.concat([vix_df, af_df[ret_cols], af_df[vol_cols], af_df[mom_cols], cr_df], axis=1)

    feature_sets = {
        "A: VIX Only"          : set_A,
        "B: +Asset Returns"    : set_B,
        "C: +Volatilities"     : set_C,
        "D: +Momentum"         : set_D,
        "E: Assets Only"       : set_E,
        "F: VIX+Stress"        : set_F,
        "G: Kitchen Sink"      : set_G,
    }

    # Temporal split (on common dates with returns)
    common = regime_labels.index
    train_mask = common <= TRAIN_END
    test_mask  = common > VAL_END

    return {
        "regime_labels": regime_labels,
        "feature_sets" : feature_sets,
        "train_mask"   : train_mask,
        "test_mask"    : test_mask,
        "prices"       : prices,
        "vix"          : vix,
    }


# ── Task 1.1: Naive Baseline ───────────────────────────────────────────────────

def task_1_1_naive_baseline(regime_labels, test_mask):
    print("\n" + "="*60)
    print("TASK 1.1: Naive Baseline (yesterday's regime)")
    print("="*60)

    y_pred = regime_labels.shift(1)
    y_test  = regime_labels.loc[test_mask]
    y_pred_test = y_pred.loc[test_mask]

    valid = y_pred_test.notna()
    y_t   = y_test.loc[valid].astype(int).values
    y_p   = y_pred_test.loc[valid].astype(int).values

    acc  = accuracy_score(y_t, y_p)
    recs = _recall_per_regime(y_t, y_p)
    cm   = confusion_matrix(y_t, y_p, labels=[1, 2, 3, 4])

    print(f"  Overall accuracy: {acc:.4f}")
    for r in [1, 2, 3, 4]:
        print(f"  Recall Regime {r} ({REGIME_NAMES[r]}): {recs[r]:.4f}")
    print(f"  Confusion matrix:\n{cm}")

    return {
        "accuracy": acc,
        "recall"  : recs,
        "confusion": cm,
        "y_pred"  : y_p,
        "y_true"  : y_t,
    }


# ── Task 1.2: Feature Ablation ─────────────────────────────────────────────────

def task_1_2_feature_ablation(feature_sets, regime_labels, train_mask, test_mask,
                               naive_result):
    print("\n" + "="*60)
    print("TASK 1.2: Feature Ablation Study")
    print("="*60)

    rows = []

    # Naive row
    nr = naive_result
    rows.append({
        "Feature Set"  : "Naive Baseline",
        "# Features"   : 0,
        "Overall Acc"  : f"{nr['accuracy']:.4f}",
        "Calm F1"      : "—",
        "Normal F1"    : "—",
        "Tense F1"     : "—",
        "Crisis F1"    : "—",
    })

    best_acc    = 0.0
    best_set_name = "A: VIX Only"
    models = {}

    for name, feat_df in feature_sets.items():
        X_train = feat_df.loc[train_mask].dropna()
        y_train = regime_labels.loc[X_train.index]
        X_test  = feat_df.loc[test_mask]
        y_test  = regime_labels.loc[test_mask]

        model, scaler = _fit_logreg(X_train, y_train)
        res = _eval_model(model, scaler, X_test, y_test)
        models[name] = (model, scaler, feat_df, res)

        n_features = X_train.shape[1]
        f1s = res["f1"]
        acc = res["accuracy"]

        rows.append({
            "Feature Set"  : name,
            "# Features"   : n_features,
            "Overall Acc"  : f"{acc:.4f}",
            "Calm F1"      : f"{f1s[1]:.4f}",
            "Normal F1"    : f"{f1s[2]:.4f}",
            "Tense F1"     : f"{f1s[3]:.4f}",
            "Crisis F1"    : f"{f1s[4]:.4f}",
        })

        print(f"  {name:<25} n={n_features:<4} acc={acc:.4f}  "
              f"F1: C={f1s[1]:.3f} N={f1s[2]:.3f} T={f1s[3]:.3f} Cr={f1s[4]:.3f}")

        if acc > best_acc:
            best_acc = acc
            best_set_name = name

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "feature_ablation.csv"), index=False)
    print(f"\n  Best feature set: '{best_set_name}' (acc={best_acc:.4f})")

    # Plot
    _plot_feature_ablation(df)

    return models, best_set_name


def _plot_feature_ablation(df):
    fig, ax = plt.subplots(figsize=(11, 5))
    sets = df["Feature Set"].tolist()
    accs = []
    for v in df["Overall Acc"]:
        try:
            accs.append(float(v))
        except ValueError:
            accs.append(0.0)

    colors = ["#CFD8DC" if s == "Naive Baseline" else "#1565C0" for s in sets]
    bars = ax.barh(sets, accs, color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, accs):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Test Accuracy (2021-2024)")
    ax.set_title("Feature Ablation: Regime Classifier Accuracy by Feature Set",
                 fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.axvline(accs[0], color="red", linewidth=1.2, linestyle="--",
               label=f"Naive={accs[0]:.4f}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "07_feature_ablation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Task 1.3: Model Comparison ─────────────────────────────────────────────────

def task_1_3_model_comparison(feature_sets, regime_labels, train_mask, test_mask,
                               best_set_name, naive_result):
    print("\n" + "="*60)
    print("TASK 1.3: Model Comparison")
    print(f"  Using best feature set: {best_set_name}")
    print("="*60)

    feat_df = feature_sets[best_set_name]
    X_train = feat_df.loc[train_mask].dropna()
    y_train = regime_labels.loc[X_train.index]
    X_test  = feat_df.loc[test_mask]
    y_test  = regime_labels.loc[test_mask]

    scaler_main = StandardScaler()
    X_train_sc  = scaler_main.fit_transform(X_train)
    X_test_valid= X_test.dropna()
    X_test_sc   = scaler_main.transform(X_test_valid)
    y_test_valid= y_test.loc[X_test_valid.index]

    models_cfg = [
        ("Logistic Regression",
         LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial",
                            solver="lbfgs", random_state=RANDOM_SEED),
         True),
        ("Random Forest",
         RandomForestClassifier(n_estimators=200, max_depth=10,
                                random_state=RANDOM_SEED, n_jobs=-1),
         False),
        ("XGBoost",
         xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                           random_state=RANDOM_SEED, eval_metric="mlogloss",
                           verbosity=0),
         False),
        ("Gradient Boosting",
         GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                    random_state=RANDOM_SEED),
         False),
    ]

    rows = []
    best_model_name = None
    best_model_acc  = 0.0
    best_proba_df   = None

    # Naive baseline row
    nr = naive_result
    rows.append({
        "Model"      : "Naive Baseline",
        "Accuracy"   : f"{nr['accuracy']:.4f}",
        "Calm F1"    : "—",
        "Normal F1"  : "—",
        "Tense F1"   : "—",
        "Crisis F1"  : "—",
        "Train Time" : "0.0s",
    })

    fitted = {}
    for mname, model, use_scale in models_cfg:
        t0 = time.time()
        X_fit = X_train_sc if use_scale else X_train.values
        # XGBoost: labels must be 0-indexed
        if "XGBoost" in mname:
            model.fit(X_fit, y_train.values - 1)
        else:
            model.fit(X_fit, y_train.values)
        elapsed = time.time() - t0

        # Predict
        X_ev = X_test_sc if use_scale else X_test_valid.values
        if "XGBoost" in mname:
            y_pred = model.predict(X_ev) + 1   # back to 1-indexed
            proba  = model.predict_proba(X_ev) # shape (N, 4)
        else:
            y_pred = model.predict(X_ev)
            proba  = model.predict_proba(X_ev) if hasattr(model, "predict_proba") else None

        acc  = accuracy_score(y_test_valid, y_pred)
        f1s  = _f1_per_regime(y_test_valid.values, y_pred)

        rows.append({
            "Model"      : mname,
            "Accuracy"   : f"{acc:.4f}",
            "Calm F1"    : f"{f1s[1]:.4f}",
            "Normal F1"  : f"{f1s[2]:.4f}",
            "Tense F1"   : f"{f1s[3]:.4f}",
            "Crisis F1"  : f"{f1s[4]:.4f}",
            "Train Time" : f"{elapsed:.1f}s",
        })
        fitted[mname] = (model, y_pred, proba, y_test_valid, use_scale)

        print(f"  {mname:<22} acc={acc:.4f}  "
              f"F1: C={f1s[1]:.3f} N={f1s[2]:.3f} T={f1s[3]:.3f} Cr={f1s[4]:.3f}  "
              f"time={elapsed:.1f}s")

        if acc > best_model_acc:
            best_model_acc  = acc
            best_model_name = mname
            if proba is not None:
                best_proba_df = pd.DataFrame(
                    proba,
                    index=X_test_valid.index,
                    columns=[f"prob_{r}" for r in range(proba.shape[1])],
                )

    df_cmp = pd.DataFrame(rows)
    df_cmp.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    print(f"\n  Best model: {best_model_name} (acc={best_model_acc:.4f})")

    _plot_model_comparison(df_cmp)

    return fitted, best_model_name, best_proba_df, y_test_valid, scaler_main


def _plot_model_comparison(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    models = df["Model"].tolist()
    accs   = [float(v) if v != "—" else 0.0 for v in df["Accuracy"]]
    colors = ["#CFD8DC"] + ["#1565C0", "#6A1B9A", "#2E7D32", "#E65100"][:len(models)-1]

    bars = ax.barh(models, accs, color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, accs):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Test Accuracy (2021-2024)")
    ax.set_title("Model Comparison: Regime Classifier", fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.axvline(accs[0], color="red", linewidth=1.2, linestyle="--",
               label=f"Naive={accs[0]:.4f}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "08_model_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Task 1.4: Confidence Analysis ─────────────────────────────────────────────

def task_1_4_confidence_analysis(fitted, best_model_name, regime_labels, test_mask):
    print("\n" + "="*60)
    print(f"TASK 1.4: Confidence Analysis (model: {best_model_name})")
    print("="*60)

    model, y_pred, proba, y_true_series, use_scale = fitted[best_model_name]

    if proba is None:
        print("  No probability output available for this model -- skipping")
        return

    y_true = y_true_series.values
    conf   = proba.max(axis=1)        # max predicted probability = confidence
    correct= (y_pred == y_true)

    # 1. Confidence Distribution
    _plot_confidence_histogram(conf, correct)

    # 2. Reliability Diagram
    _plot_reliability_diagram(conf, correct)

    # 3. Regime Transition Analysis
    regime_test = regime_labels.loc[test_mask]
    _plot_transition_analysis(y_pred, y_true, y_true_series.index, regime_test)

    print(f"  Mean confidence (correct  predictions): {conf[correct].mean():.4f}")
    print(f"  Mean confidence (incorrect predictions): {conf[~correct].mean():.4f}")


def _plot_confidence_histogram(conf, correct):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(conf[correct],  bins=30, alpha=0.7, color="#1565C0", label="Correct")
    ax.hist(conf[~correct], bins=30, alpha=0.7, color="#C62828", label="Incorrect")
    ax.set_xlabel("Max Predicted Probability (Confidence)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution: Correct vs Incorrect Predictions",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "09_confidence_histogram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_reliability_diagram(conf, correct):
    bins   = np.linspace(0.25, 1.0, 16)
    bin_acc   = []
    bin_conf  = []
    bin_count = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() > 0:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(conf[mask].mean())
            bin_count.append(mask.sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    scatter = ax.scatter(bin_conf, bin_acc, c=bin_count, cmap="Blues",
                         s=80, zorder=5, edgecolors="navy")
    ax.plot(bin_conf, bin_acc, color="#1565C0", linewidth=2)
    plt.colorbar(scatter, ax=ax, label="# samples in bin")
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Actual Accuracy")
    ax.set_title("Reliability Diagram (Calibration Curve)", fontweight="bold")
    ax.set_xlim(0.25, 1.0)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(range(len(bin_count)), bin_count, color="#1565C0", alpha=0.7)
    ax2.set_xticks(range(len(bin_conf)))
    ax2.set_xticklabels([f"{v:.2f}" for v in bin_conf], rotation=45, fontsize=7)
    ax2.set_xlabel("Confidence Bin")
    ax2.set_ylabel("# Predictions")
    ax2.set_title("Confidence Histogram by Bin", fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "10_reliability_diagram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_transition_analysis(y_pred, y_true, pred_index, regime_test):
    """
    Compare accuracy on stable days vs transition days.
    Stable: regime unchanged for +-3 days.
    Transition: within 3 days of a regime change.
    """
    reg = regime_test.reindex(pred_index)
    shifted_fwd  = reg.shift(-1)
    shifted_back = reg.shift(1)

    # Transition day: regime changes within window of +/-3
    window = 3
    rolling_max = reg.rolling(window=2*window+1, center=True, min_periods=1).max()
    rolling_min = reg.rolling(window=2*window+1, center=True, min_periods=1).min()
    is_transition = (rolling_max != rolling_min)

    # Align
    common = pred_index.intersection(is_transition.index)
    trans  = is_transition.loc[common].values
    correct= (y_pred == y_true)  # already aligned to pred_index

    if len(correct) != len(trans):
        # safety: truncate
        n = min(len(correct), len(trans))
        correct = correct[:n]
        trans   = trans[:n]

    stable_acc = correct[~trans].mean() if (~trans).sum() > 0 else float("nan")
    trans_acc  = correct[trans].mean()  if  trans.sum()  > 0 else float("nan")

    n_stable = (~trans).sum()
    n_trans  = trans.sum()

    print(f"\n  Transition Analysis:")
    print(f"    Stable days     ({n_stable:4d}): accuracy = {stable_acc:.4f}")
    print(f"    Transition days ({n_trans:4d}):  accuracy = {trans_acc:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    cats = ["Stable Days", "Transition Days"]
    accs = [stable_acc, trans_acc]
    ns   = [n_stable, n_trans]
    bars = ax.bar(cats, accs, color=["#1565C0", "#E65100"], width=0.5, edgecolor="white")
    for bar, val, n in zip(bars, accs, ns):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.4f}\n(n={n})", ha="center", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Classifier Accuracy:\nStable vs Transition Days", fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Also show rolling accuracy over time
    ax2 = axes[1]
    correct_series = pd.Series(
        (y_pred == y_true).astype(float), index=common[:len(y_pred)]
    )
    rolling_acc = correct_series.rolling(30, min_periods=5).mean()
    ax2.plot(rolling_acc, color="#1565C0", linewidth=1.5, label="30-day rolling accuracy")
    ax2.axhline(stable_acc, color="#2E7D32", linestyle="--", linewidth=1,
                label=f"Stable avg={stable_acc:.3f}")
    ax2.axhline(trans_acc,  color="#E65100", linestyle="--", linewidth=1,
                label=f"Trans avg={trans_acc:.3f}")
    ax2.set_ylabel("Accuracy (30d rolling)")
    ax2.set_title("Classifier Accuracy Over Time", fontweight="bold")
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "11_transition_analysis.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Part 1: Regime Classifier Deep-Dive")
    print("="*60)

    print("\nLoading data...")
    bundle = load_all()
    regime_labels = bundle["regime_labels"]
    feature_sets  = bundle["feature_sets"]
    train_mask    = bundle["train_mask"]
    test_mask     = bundle["test_mask"]

    # Task 1.1
    naive_result = task_1_1_naive_baseline(regime_labels, test_mask)

    # Task 1.2
    models, best_set_name = task_1_2_feature_ablation(
        feature_sets, regime_labels, train_mask, test_mask, naive_result
    )

    # Task 1.3
    fitted, best_model_name, best_proba, y_test_valid, scaler = \
        task_1_3_model_comparison(
            feature_sets, regime_labels, train_mask, test_mask,
            best_set_name, naive_result,
        )

    # Task 1.4
    task_1_4_confidence_analysis(fitted, best_model_name, regime_labels, test_mask)

    print("\n" + "="*60)
    print("  Part 1 complete. All plots saved to results/")
    print("="*60)


if __name__ == "__main__":
    main()
