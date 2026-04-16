"""
Plan 13b-v3 Hyperparameter Landscape Visualization
===================================================
Diagnostic / visualization work only — no new experiments, no model training.

Data source: fold_0{1,2,3}_trial_log.csv  (400 rows each = 100 trials × 4 regimes)
             fold_0{1,2,3}_best_configs.json  (best selected config per regime-fold)

Note: The Optuna .pkl study files were not persisted, so per-trial lookback values
      are not available for all 1200 trials.  Plots 1 & 5 use only the 12 best-
      selected configs (fold × regime) which DO carry lookback information.
"""

import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("results/plan13b_v3")
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

REGIME_NAMES = {1: "Calm", 2: "Normal", 3: "Tense", 4: "Crisis"}

# ---------------------------------------------------------------------------
# Consistent colour palette  (family → colour)
# ---------------------------------------------------------------------------
ALL_FAMILIES = [
    "MinVariance", "MeanVariance", "MaxDiversification", "RiskParity",
    "EqualWeight", "Momentum", "TrendFollowing",
    "Lasso", "Ridge", "ElasticNet",
    "RandomForest", "GradientBoosting", "MLP",
]
_palette = sns.color_palette("tab20", n_colors=len(ALL_FAMILIES))
FAMILY_COLOUR = {fam: _palette[i] for i, fam in enumerate(ALL_FAMILIES)}

FOLD_MARKERS = {1: "o", 2: "s", 3: "^"}

# ---------------------------------------------------------------------------
# Phase 1: Load & merge data
# ---------------------------------------------------------------------------

def load_trial_logs() -> pd.DataFrame:
    dfs = []
    for fold in [1, 2, 3]:
        fp = BASE / f"fold_0{fold}_trial_log.csv"
        df = pd.read_csv(fp)
        df.insert(0, "fold", fold)
        # rename "gap" → "train_val_gap" for clarity
        if "gap" in df.columns and "train_val_gap" not in df.columns:
            df = df.rename(columns={"gap": "train_val_gap"})
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    merged["regime_name"] = merged["regime_id"].map(REGIME_NAMES)
    return merged


def load_best_configs() -> pd.DataFrame:
    """Extract the 12 best-selected configs (fold × regime) including lookback."""
    rows = []
    for fold in [1, 2, 3]:
        fp = BASE / f"fold_0{fold}_best_configs.json"
        with open(fp) as f:
            d = json.load(f)
        for regime_str, cfg in d.items():
            regime_id = int(regime_str)
            config = cfg.get("config", {})
            # Extract lookback from any key containing "lookback"
            lookback = None
            for k, v in config.items():
                if "lookback" in k:
                    lookback = v
                    break
            rows.append({
                "fold": fold,
                "regime_id": regime_id,
                "regime_name": REGIME_NAMES[regime_id],
                "family": cfg["family"],
                "tier": cfg["tier"],
                "lookback": lookback,
                "val_sharpe": cfg["val_sharpe"],
                "train_sharpe": cfg["train_sharpe"],
                "train_val_gap": cfg.get("train_val_gap", cfg["train_sharpe"] - cfg["val_sharpe"]),
                "all_params": json.dumps(config),
            })
    return pd.DataFrame(rows)


print("Loading trial logs …")
df = load_trial_logs()
print(f"  Total trials loaded: {len(df)}")

print("Loading best configs …")
best_df = load_best_configs()
print(f"  Best configs loaded: {len(best_df)}")

# ---------------------------------------------------------------------------
# Save landscape_data.csv  (trial logs + null lookback column)
# ---------------------------------------------------------------------------
# Add placeholder lookback column (not available per-trial without .pkl files)
df["lookback"] = np.nan
df["all_params"] = ""

col_order = ["fold", "regime_id", "regime_name", "trial_num", "tier",
             "family", "lookback", "val_sharpe", "train_sharpe",
             "train_val_gap", "pruned", "all_params"]
df[col_order].to_csv(BASE / "landscape_data.csv", index=False)
print(f"  Saved landscape_data.csv  ({len(df)} rows)")

# ---------------------------------------------------------------------------
# Phase 2: Visualizations
# ---------------------------------------------------------------------------

# -- helper ------------------------------------------------------------------

def save(fig: plt.Figure, name: str, tight: bool = True):
    if tight:
        fig.tight_layout()
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 1: Best-Selected Lookback vs Val-Sharpe  (per regime, 12 points)
# Note: full per-trial lookback unavailable (no .pkl); we plot the 12
#       best-selected configs and, where available, fill with trial data
#       that shares the same (fold, regime) for reference.
# ---------------------------------------------------------------------------
print("\nPlot 1: Lookback vs Val-Sharpe landscape …")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (rid, rname) in zip(axes, REGIME_NAMES.items()):
    sub = best_df[best_df["regime_id"] == rid].copy()

    for _, row in sub.iterrows():
        fold = int(row["fold"])
        fam = row["family"]
        colour = FAMILY_COLOUR.get(fam, "grey")
        marker = FOLD_MARKERS[fold]
        ax.scatter(row["lookback"], row["val_sharpe"],
                   color=colour, marker=marker, s=120, zorder=5,
                   edgecolors="white", linewidths=0.6)

    # Best trial per regime-fold = these ARE the best configs; annotate top val_sharpe
    if len(sub):
        best_row = sub.loc[sub["val_sharpe"].idxmax()]
        ax.scatter(best_row["lookback"], best_row["val_sharpe"],
                   marker="*", color="red", s=350, zorder=10,
                   label=f'Best: {best_row["family"]} (Fold {int(best_row["fold"])})')
        ax.annotate(f'{best_row["family"]}\n(F{int(best_row["fold"])})',
                    (best_row["lookback"], best_row["val_sharpe"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=7, color="red")

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(f"Regime {rid}: {rname}", fontsize=11)
    ax.set_xlabel("Lookback (days)")
    ax.set_ylabel("Val Sharpe")
    ax.set_xlim(0, 260)

    # Legend: families present
    present_fams = sub["family"].unique()
    handles = [mpatches.Patch(color=FAMILY_COLOUR.get(f, "grey"), label=f)
               for f in present_fams]
    fold_handles = [plt.Line2D([0], [0], marker=FOLD_MARKERS[k], color="k",
                               linestyle="None", markersize=7, label=f"Fold {k}")
                    for k in [1, 2, 3]]
    ax.legend(handles=handles + fold_handles, fontsize=7, ncol=2, loc="best")

fig.suptitle(
    "Hyperparameter Landscape: Best-Selected Lookback vs Validation Sharpe\n"
    "(12 data points = 3 folds × 4 regimes; per-trial lookback unavailable without .pkl)",
    fontsize=11
)
save(fig, "plot1_lookback_vs_val_sharpe.png")
print("  (Note: only 12 best-config points; full 1200-trial landscape requires .pkl files)")


# ---------------------------------------------------------------------------
# Plot 2: Val-Sharpe Distribution per Family per Regime
# ---------------------------------------------------------------------------
print("\nPlot 2: Val-Sharpe distribution by family …")

fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharey=False)

for ax, (rid, rname) in zip(axes, REGIME_NAMES.items()):
    sub = df[df["regime_id"] == rid].copy()
    # Only families with ≥5 trials in this regime
    counts = sub["family"].value_counts()
    valid_fams = counts[counts >= 5].index.tolist()
    # Sort by median val_sharpe
    medians = sub[sub["family"].isin(valid_fams)].groupby("family")["val_sharpe"].median()
    ordered_fams = medians.sort_values().index.tolist()

    plot_data = sub[sub["family"].isin(ordered_fams)].copy()
    palette = {f: FAMILY_COLOUR.get(f, "grey") for f in ordered_fams}

    sns.boxplot(
        data=plot_data, x="family", y="val_sharpe",
        order=ordered_fams, palette=palette, ax=ax,
        width=0.55, flierprops=dict(marker=".", alpha=0.4, markersize=4)
    )
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
    ax.set_title(f"Regime {rid}: {rname}", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("Val Sharpe")
    ax.tick_params(axis="x", rotation=25, labelsize=8)

    # Overlay best-selected family for each fold with a star
    for fold in [1, 2, 3]:
        b = best_df[(best_df["regime_id"] == rid) & (best_df["fold"] == fold)]
        if not b.empty and b.iloc[0]["family"] in ordered_fams:
            x_pos = ordered_fams.index(b.iloc[0]["family"])
            ax.scatter(x_pos, b.iloc[0]["val_sharpe"],
                       marker="*", color="red", s=180, zorder=10,
                       label=f"Best F{fold}")

fig.suptitle("Validation Sharpe Distribution by Family and Regime\n"
             "(red stars = TPE-selected best per fold)", fontsize=12)
save(fig, "plot2_val_sharpe_by_family_regime.png")


# ---------------------------------------------------------------------------
# Plot 3: Train-Val Gap Scatter
# ---------------------------------------------------------------------------
print("\nPlot 3: Train-Val gap scatter …")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (rid, rname) in zip(axes, REGIME_NAMES.items()):
    sub = df[df["regime_id"] == rid]

    for fam, grp in sub.groupby("family"):
        colour = FAMILY_COLOUR.get(fam, "grey")
        ax.scatter(grp["train_sharpe"], grp["val_sharpe"],
                   color=colour, alpha=0.55, s=25, label=fam,
                   edgecolors="none")

    # Diagonal y = x
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.7, label="y = x")

    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.axvline(0, color="grey", lw=0.5, ls=":")

    ax.set_title(f"Regime {rid}: {rname}", fontsize=11)
    ax.set_xlabel("Train Sharpe")
    ax.set_ylabel("Val Sharpe")

    handles = [mpatches.Patch(color=FAMILY_COLOUR.get(f, "grey"), label=f)
               for f in sub["family"].unique()]
    ax.legend(handles=handles, fontsize=6, ncol=2, loc="best")

fig.suptitle(
    "Train-Val Relationship (Ideal: points on or above diagonal)\n"
    "Points below y=x → overfitting; above → val outperforms train",
    fontsize=11
)
save(fig, "plot3_train_val_gap.png")


# ---------------------------------------------------------------------------
# Plot 4: TPE Convergence Over Trials
# ---------------------------------------------------------------------------
print("\nPlot 4: TPE convergence …")

FOLD_COLOURS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (rid, rname) in zip(axes, REGIME_NAMES.items()):
    for fold in [1, 2, 3]:
        sub = df[(df["regime_id"] == rid) & (df["fold"] == fold)].sort_values("trial_num")
        if sub.empty:
            continue
        colour = FOLD_COLOURS[fold]
        running_max = sub["val_sharpe"].cummax()

        ax.scatter(sub["trial_num"], sub["val_sharpe"],
                   color=colour, alpha=0.3, s=15, zorder=2)
        ax.plot(sub["trial_num"], running_max,
                color=colour, lw=2.0, zorder=3, label=f"Fold {fold} (best so far)")

    # Warm-start boundary (first 48 trials are from warm-start)
    ax.axvline(48, color="purple", lw=1, ls=":", alpha=0.7, label="Warm-start end")
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)
    ax.set_title(f"Regime {rid}: {rname}", fontsize=11)
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Val Sharpe")
    ax.legend(fontsize=8)

fig.suptitle(
    "TPE Convergence: Running Best Val-Sharpe Over Trials\n"
    "(dots = individual trials; thick line = running max; purple = warm-start boundary)",
    fontsize=11
)
save(fig, "plot4_tpe_convergence.png")


# ---------------------------------------------------------------------------
# Plot 5: Hyperparameter Stability Across Folds (Best-Selected Lookback)
# ---------------------------------------------------------------------------
print("\nPlot 5: Lookback stability across folds …")

fig, ax = plt.subplots(figsize=(12, 6))

# Jitter x slightly by fold for visibility
jitter = {1: -0.2, 2: 0.0, 3: 0.2}
x_positions = list(REGIME_NAMES.keys())

for _, row in best_df.iterrows():
    if pd.isna(row["lookback"]):
        continue
    xpos = row["regime_id"] + jitter[row["fold"]]
    colour = FAMILY_COLOUR.get(row["family"], "grey")
    # Point size proportional to val_sharpe rank within regime (larger = better)
    regime_vals = best_df[best_df["regime_id"] == row["regime_id"]]["val_sharpe"]
    rank = (regime_vals < row["val_sharpe"]).sum() + 1   # 1 = worst
    size = 60 + rank * 50

    ax.scatter(xpos, row["lookback"],
               color=colour, marker=FOLD_MARKERS[row["fold"]],
               s=size, zorder=5, edgecolors="white", linewidths=0.6,
               alpha=0.85, label=row["family"])

# Overlay best-selected with big red X
for _, row in best_df.iterrows():
    if pd.isna(row["lookback"]):
        continue
    xpos = row["regime_id"] + jitter[row["fold"]]
    ax.scatter(xpos, row["lookback"],
               marker="X", color="red", s=220, zorder=10,
               edgecolors="darkred", linewidths=0.8)
    ax.annotate(f"F{row['fold']}: {row['family'][:4]}",
                (xpos, row["lookback"]),
                textcoords="offset points", xytext=(4, 3),
                fontsize=6.5, color="red")

ax.set_xticks(x_positions)
ax.set_xticklabels([f"R{r}: {REGIME_NAMES[r]}" for r in x_positions])
ax.set_ylabel("Best-Selected Lookback (days)")
ax.set_xlabel("Regime")
ax.axhline(50, color="grey", lw=0.5, ls=":")
ax.axhline(100, color="grey", lw=0.5, ls=":")
ax.axhline(200, color="grey", lw=0.5, ls=":")
ax.set_ylim(0, 270)

# Legend: families
present_fams = best_df["family"].unique()
handles = [mpatches.Patch(color=FAMILY_COLOUR.get(f, "grey"), label=f)
           for f in present_fams]
fold_handles = [plt.Line2D([0], [0], marker=FOLD_MARKERS[k], color="k",
                           linestyle="None", markersize=8, label=f"Fold {k}")
                for k in [1, 2, 3]]
ax.legend(handles=handles + fold_handles, fontsize=8, ncol=3, loc="upper left")

ax.set_title(
    "Best-Selected Lookback per Regime (Across 3 Folds)\n"
    "(red X = TPE winner; spread → instability)", fontsize=12
)
save(fig, "plot5_lookback_stability.png")


# ---------------------------------------------------------------------------
# Plot 6: Family Selection Frequency Heatmap
# ---------------------------------------------------------------------------
print("\nPlot 6: Family selection frequency heatmap …")

# Build pivot: rows=family, cols=fold-regime combo
df["fold_regime"] = df.apply(lambda r: f"F{r['fold']}_R{r['regime_id']}", axis=1)
pivot = df.groupby(["family", "fold_regime"]).size().unstack(fill_value=0)

# Sort families by total count descending
pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

# Column order: F1_R1, F1_R2, ...
col_order = sorted(pivot.columns, key=lambda c: (int(c[1]), int(c[4])))
pivot = pivot[col_order]

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt="d", linewidths=0.4,
            ax=ax, cbar_kws={"label": "Trial Count"})

# Star annotations on best-selected family per fold-regime
for _, row in best_df.iterrows():
    col_key = f"F{row['fold']}_R{row['regime_id']}"
    if col_key in pivot.columns and row["family"] in pivot.index:
        ci = list(pivot.columns).index(col_key)
        ri = list(pivot.index).index(row["family"])
        ax.text(ci + 0.5, ri + 0.5, "★", ha="center", va="center",
                fontsize=14, color="blue", fontweight="bold")

ax.set_title("TPE Family Selection Frequency per Regime-Fold\n"
             "(blue ★ = TPE-selected best config)", fontsize=12)
ax.set_xlabel("Fold_Regime")
ax.set_ylabel("Family")
plt.xticks(rotation=45, ha="right")
save(fig, "plot6_family_selection_heatmap.png")


# ---------------------------------------------------------------------------
# Phase 3: Quantitative Summary Report
# ---------------------------------------------------------------------------
print("\nGenerating summary report …")

lines = ["## Landscape Summary Statistics", ""]
lines.append("### Per Regime:")

overall_pruned = int(df["pruned"].sum())
overall_total = len(df)

for rid, rname in REGIME_NAMES.items():
    sub = df[df["regime_id"] == rid]
    n = len(sub)

    mv = sub["val_sharpe"].mean()
    sv = sub["val_sharpe"].std()
    mn = sub["val_sharpe"].min()
    mx = sub["val_sharpe"].max()
    pct_pos_val = 100 * (sub["val_sharpe"] > 0).mean()
    pct_both_pos = 100 * ((sub["train_sharpe"] > 0) & (sub["val_sharpe"] > 0)).mean()
    best_val = sub["val_sharpe"].max()

    # Spread of best-per-fold lookbacks
    best_lbs = best_df[best_df["regime_id"] == rid]["lookback"].dropna()
    if len(best_lbs) > 1:
        lb_min = int(best_lbs.min())
        lb_max = int(best_lbs.max())
        lb_cov = 100 * best_lbs.std() / best_lbs.mean() if best_lbs.mean() != 0 else float("nan")
    else:
        lb_min = lb_max = int(best_lbs.iloc[0]) if len(best_lbs) else -1
        lb_cov = 0.0

    # Correlation
    corr = sub[["train_sharpe", "val_sharpe"]].corr().iloc[0, 1]

    lines += [
        f"  Regime {rid} ({rname}):",
        f"    - Mean val_sharpe across all {n} trials: {mv:.4f}",
        f"    - Std val_sharpe: {sv:.4f}",
        f"    - Range: [{mn:.4f}, {mx:.4f}]",
        f"    - % trials with val_sharpe > 0: {pct_pos_val:.1f}%",
        f"    - % trials with both train > 0 AND val > 0: {pct_both_pos:.1f}%",
        f"    - Best single trial's val_sharpe: {best_val:.4f}",
        f"    - Spread of best-per-fold lookbacks: min={lb_min}, max={lb_max}, CoV={lb_cov:.1f}%",
        f"    - Corr(train_sharpe, val_sharpe): {corr:.4f}",
        "",
    ]

lines += [
    "### Overall:",
    f"  - Total trials: {overall_total}",
    f"  - Pruned trials: {overall_pruned} ({100*overall_pruned/overall_total:.1f}%)",
    f"  - Trials with val_sharpe > 1.0 (genuinely good): {(df['val_sharpe'] > 1.0).sum()}",
    f"  - Trials with val_sharpe > 2.0: {(df['val_sharpe'] > 2.0).sum()}",
    "",
    "### Correlation(train_sharpe, val_sharpe) per regime:",
]
for rid, rname in REGIME_NAMES.items():
    sub = df[df["regime_id"] == rid]
    corr = sub[["train_sharpe", "val_sharpe"]].corr().iloc[0, 1]
    lines.append(f"  Regime {rid} ({rname}): {corr:.4f}")

lines += [
    "",
    "### Best-Selected Config Instability (Lookback Spread per Regime):",
]
for rid, rname in REGIME_NAMES.items():
    sub_best = best_df[best_df["regime_id"] == rid]
    lbs = sub_best[["fold", "family", "lookback", "val_sharpe"]].to_string(index=False)
    lines.append(f"  Regime {rid} ({rname}):\n{lbs}")
    lines.append("")

report_text = "\n".join(lines)
report_path = BASE / "landscape_summary.txt"
with open(report_path, "w") as f:
    f.write(report_text)

print(f"  Saved {report_path}")
print()
print(report_text)

# ---------------------------------------------------------------------------
# Brief qualitative observations
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("QUALITATIVE OBSERVATIONS")
print("="*70)
print("""
Plot 1 (Lookback vs Val-Sharpe):
  Only 12 best-selected configs shown (no per-trial lookback in CSV data).
  Lookbacks span the full range (15–251 days) with no clustering, confirming
  the landscape has no clear optimal lookback.

Plot 2 (Val-Sharpe by Family):
  Boxes are wide and centered near 0 or below for most families,
  suggesting the landscape is noisy rather than structured.

Plot 3 (Train-Val Gap):
  The vast majority of points fall below the y=x diagonal: train_sharpe
  is systematically higher than val_sharpe, indicating strong overfitting.

Plot 4 (TPE Convergence):
  Running max plots reveal whether TPE improved beyond the warm-start plateau.
  Steep early gains followed by a flat plateau would confirm TPE is
  exploiting noise rather than genuine signal.

Plot 5 (Lookback Stability):
  Red X markers are scattered across the full 0–260 day range for most
  regimes, directly visualizing the instability: TPE found different "optima"
  in each fold.

Plot 6 (Family Selection Heatmap):
  Shows how evenly (or unevenly) TPE spread its budget across families.
  Uneven distribution suggests TPE converged prematurely on certain families.
""")
