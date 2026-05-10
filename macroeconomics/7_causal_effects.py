"""
Module 7: Causal Effect Estimation
====================================
Based on the 7 edges found by PC algorithm, estimates the causal effect
of each cause on its effect using the backdoor criterion.

PC Algorithm discovered edges:
  1. VIX          → SP500 Return
  2. CPI YoY      → SP500 Return
  3. Oil Return   → SP500 Return
  4. VIX          → FedFunds Δ
  5. CPI YoY      → FedFunds Δ
  6. Unemploy Δ   → FedFunds Δ
  7. Oil Return   → FedFunds Δ

For each edge X → Y, we identify the adjustment set (backdoor criterion):
all variables that are parents of X in the DAG, to block backdoor paths.

Input:  data/processed_data.csv   ← non-normalized for interpretable coefficients
Output: figures/causal_effects_sp500.png
        figures/causal_effects_fedfunds.png
        results/causal_effects.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")

INPUT_FILE  = "data/processed_data.csv"
FIGURES_DIR = "figures"
RESULTS_DIR = "results"

df = pd.read_csv(INPUT_FILE, index_col="Date", parse_dates=True).dropna()

# ── Variable name mapping ─────────────────────────────────────────────────────
COLS = {
    "SP500_ret"  : "S&P 500 Return",
    "VIX"        : "VIX (log)",
    "FEDFUNDS_d" : "Fed Funds Δ",
    "CPI_yoy"    : "CPI YoY",
    "UNRATE_d"   : "Unemployment Δ",
    "M2_yoy"     : "M2 Growth",
    "SENTIMENT_d": "Sentiment Δ",
    "OIL_ret"    : "Oil Return",
    "GS10_d"     : "10Y Treasury Δ",
}

# ── Causal graph structure from PC algorithm ──────────────────────────────────
# Parents of each node (used to construct adjustment sets)
# Based on PC graph: VIX→SP500, CPI→SP500, Oil→SP500,
#                    VIX→FedFunds, CPI→FedFunds, Unemploy→FedFunds, Oil→FedFunds
# No variable has identified parents in the PC graph (all are roots),
# so backdoor adjustment set = all other variables that could confound.
# We use the direct parents of the EFFECT variable as controls (standard adjustment).

EDGES = [
    # (cause, effect, adjustment_set)
    # For X → SP500: control for other direct causes of SP500
    ("VIX",        "SP500_ret",  ["CPI_yoy", "OIL_ret"]),
    ("CPI_yoy",    "SP500_ret",  ["VIX",     "OIL_ret"]),
    ("OIL_ret",    "SP500_ret",  ["VIX",     "CPI_yoy"]),
    # For X → FedFunds: control for other direct causes of FedFunds
    ("VIX",        "FEDFUNDS_d", ["CPI_yoy", "UNRATE_d", "OIL_ret"]),
    ("CPI_yoy",    "FEDFUNDS_d", ["VIX",     "UNRATE_d", "OIL_ret"]),
    ("UNRATE_d",   "FEDFUNDS_d", ["VIX",     "CPI_yoy",  "OIL_ret"]),
    ("OIL_ret",    "FEDFUNDS_d", ["VIX",     "CPI_yoy",  "UNRATE_d"]),
]

N_BOOTSTRAP = 1000
results = []

print(f"Estimating causal effects with backdoor adjustment (bootstrap n={N_BOOTSTRAP})...\n")
print(f"{'Cause':<22} {'Effect':<18} {'Coef':>8} {'95% CI':>20} {'p≈0?':>8}")
print("-" * 80)

for cause, effect, adjustment in EDGES:
    X_cols = [cause] + adjustment
    X = df[X_cols].values
    y = df[effect].values

    # OLS estimate
    reg = LinearRegression().fit(X, y)
    coef = reg.coef_[0]   # coefficient for cause (first column)

    # Bootstrap 95% CI
    boot_coefs = []
    for _ in range(N_BOOTSTRAP):
        X_b, y_b = resample(X, y, random_state=None)
        reg_b = LinearRegression().fit(X_b, y_b)
        boot_coefs.append(reg_b.coef_[0])

    ci_lo = np.percentile(boot_coefs, 2.5)
    ci_hi = np.percentile(boot_coefs, 97.5)
    significant = "✅" if (ci_lo > 0 or ci_hi < 0) else "  "

    results.append({
        "cause"   : COLS[cause],
        "effect"  : COLS[effect],
        "coef"    : coef,
        "ci_lo"   : ci_lo,
        "ci_hi"   : ci_hi,
        "sig"     : significant.strip() == "✅",
    })
    print(f"{COLS[cause]:<22} {COLS[effect]:<18} {coef:>8.4f} "
          f"[{ci_lo:>7.4f}, {ci_hi:>7.4f}] {significant:>8}")

results_df = pd.DataFrame(results)
results_df.to_csv(f"{RESULTS_DIR}/causal_effects.csv", index=False)
print(f"\nSaved → {RESULTS_DIR}/causal_effects.csv")


# ── Plot: SP500 effects ───────────────────────────────────────────────────────
def plot_effects(df_sub, title, filename):
    fig, ax = plt.subplots(figsize=(9, 4))
    y_pos = range(len(df_sub))

    colors = ["#16a34a" if s else "#94a3b8" for s in df_sub["sig"]]

    ax.barh(list(y_pos), df_sub["coef"], color=colors, alpha=0.85, height=0.5)
    ax.errorbar(
        df_sub["coef"], list(y_pos),
        xerr=[df_sub["coef"] - df_sub["ci_lo"],
              df_sub["ci_hi"] - df_sub["coef"]],
        fmt="none", color="black", capsize=5, linewidth=1.5
    )
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df_sub["cause"], fontsize=10)
    ax.set_xlabel("Regression coefficient (with backdoor adjustment)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")

    sig_patch   = mpatches.Patch(color="#16a34a", alpha=0.85, label="Significant (95% CI excludes 0)")
    insig_patch = mpatches.Patch(color="#94a3b8", alpha=0.85, label="Not significant")
    ax.legend(handles=[sig_patch, insig_patch], fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {FIGURES_DIR}/{filename}")


sp500_df   = results_df[results_df["effect"] == COLS["SP500_ret"]].reset_index(drop=True)
fedfunds_df = results_df[results_df["effect"] == COLS["FEDFUNDS_d"]].reset_index(drop=True)

plot_effects(sp500_df,
             "Causal Effects on S&P 500 Return\n(OLS with backdoor adjustment, 95% bootstrap CI)",
             "causal_effects_sp500.png")

plot_effects(fedfunds_df,
             "Causal Effects on Fed Funds Rate Δ\n(OLS with backdoor adjustment, 95% bootstrap CI)",
             "causal_effects_fedfunds.png")

print("\nCausal effect estimation complete.")
