"""
Module 4: Granger Causality
============================
Tests whether each variable Granger-causes SP500 returns,
and also runs a full pairwise Granger causality matrix.

Steps:
  1. Select optimal lag order via AIC (VAR model)
  2. Pairwise Granger causality: X → SP500_ret for each X
  3. Full pairwise matrix across all variables
  4. Visualize results as heatmaps

Input:  data/processed_data.csv   ← uses non-normalized for interpretability
Output: figures/granger_sp500.png
        figures/granger_matrix.png
        results/granger_results.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
import os
import warnings
warnings.filterwarnings("ignore")

INPUT_FILE  = "data/processed_data.csv"
FIGURES_DIR = "figures"
RESULTS_DIR = "results"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_LAG = 6   # test up to 6 months lag

LABELS = {
    "SP500_ret"  : "S&P 500 Return",
    "VIX"        : "VIX (log)",
    "FEDFUNDS_d" : "Fed Funds Δ",
    "CPI_yoy"    : "CPI YoY",
    "UNRATE_d"   : "Unemployment Δ",
    "M2_yoy"     : "M2 Growth",
    "SENTIMENT_d": "Consumer Sentiment Δ",
    "OIL_ret"    : "Oil Return",
    "GS10_d"     : "10Y Treasury Δ",
}

df = pd.read_csv(INPUT_FILE, index_col="Date", parse_dates=True).dropna()


# ── 1. Optimal lag via VAR AIC ────────────────────────────────────────────────
print("Selecting optimal lag order via VAR AIC...")
model    = VAR(df)
lag_order = model.select_order(maxlags=MAX_LAG)
optimal_lag = lag_order.aic
print(f"  Optimal lag (AIC): {optimal_lag}")
optimal_lag = max(1, optimal_lag)   # ensure at least 1


# ── 2. Granger causality: each variable → SP500_ret ──────────────────────────
print(f"\nGranger causality tests (lag={optimal_lag}): X → SP500_ret")
print(f"{'Variable':<25} {'F-stat':>10} {'p-value':>10} {'Significant?':>14}")
print("-" * 62)

granger_sp500 = {}
macro_vars = [c for c in df.columns if c != "SP500_ret"]

for col in macro_vars:
    data_pair = df[["SP500_ret", col]].dropna()
    result    = grangercausalitytests(data_pair, maxlag=optimal_lag, verbose=False)
    # Use F-test p-value at optimal lag
    p_val  = result[optimal_lag][0]["ssr_ftest"][1]
    f_stat = result[optimal_lag][0]["ssr_ftest"][0]
    sig    = "✅ Yes" if p_val < 0.05 else "  No"
    granger_sp500[col] = {"f_stat": f_stat, "p_value": p_val}
    print(f"{LABELS[col]:<25} {f_stat:>10.3f} {p_val:>10.4f} {sig:>14}")


# ── 3. Full pairwise Granger matrix ──────────────────────────────────────────
print(f"\nBuilding full pairwise Granger matrix (lag={optimal_lag})...")
all_vars = df.columns.tolist()
n        = len(all_vars)
p_matrix = pd.DataFrame(np.nan, index=all_vars, columns=all_vars)

for cause in all_vars:
    for effect in all_vars:
        if cause == effect:
            p_matrix.loc[cause, effect] = np.nan
            continue
        data_pair = df[[effect, cause]].dropna()
        try:
            result = grangercausalitytests(data_pair, maxlag=optimal_lag, verbose=False)
            p_matrix.loc[cause, effect] = result[optimal_lag][0]["ssr_ftest"][1]
        except Exception:
            p_matrix.loc[cause, effect] = np.nan

# Save results
p_matrix.to_csv(f"{RESULTS_DIR}/granger_results.csv")
print(f"  Saved → {RESULTS_DIR}/granger_results.csv")


# ── 4. Visualizations ─────────────────────────────────────────────────────────

# Plot A: Bar chart — X → SP500
print("\nPlotting Granger → SP500 bar chart...")
results_df = pd.DataFrame(granger_sp500).T
results_df = results_df.sort_values("p_value")
colors     = ["#16a34a" if p < 0.05 else "#94a3b8" for p in results_df["p_value"]]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh([LABELS[c] for c in results_df.index], results_df["p_value"], color=colors)
ax.axvline(0.05, color="red", linewidth=1.5, linestyle="--", label="p = 0.05")
ax.set_xlabel("p-value (Granger F-test)")
ax.set_title(f"Granger Causality: X → S&P 500 Return (lag = {optimal_lag})",
             fontsize=12, fontweight="bold")
ax.legend()
ax.set_xlim(0, max(results_df["p_value"].max() * 1.1, 0.2))
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/granger_sp500.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {FIGURES_DIR}/granger_sp500.png")

# Plot B: Full pairwise heatmap
print("Plotting full pairwise Granger heatmap...")
pretty_labels = [LABELS[v] for v in all_vars]

fig, ax = plt.subplots(figsize=(11, 9))
sig_matrix = (p_matrix < 0.05).astype(float)   # 1 = significant, 0 = not
sns.heatmap(p_matrix.astype(float),
            annot=True, fmt=".2f", cmap="RdYlGn_r",
            center=0.05, vmin=0, vmax=0.3,
            xticklabels=pretty_labels,
            yticklabels=pretty_labels,
            ax=ax, linewidths=0.5)
ax.set_title(f"Pairwise Granger Causality p-values (row → col, lag={optimal_lag})\nGreen = significant (p<0.05)",
             fontsize=11, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
ax.set_xlabel("Effect variable")
ax.set_ylabel("Cause variable")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/granger_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {FIGURES_DIR}/granger_matrix.png")

print("\nGranger causality analysis complete.")
