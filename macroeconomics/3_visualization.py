"""
Module 3: Visualization
=======================
Exploratory visualizations before running causal models.

Plots:
  1. Time series of all processed variables (with crisis annotations)
  2. Correlation heatmap
  3. Rolling correlation: each macro variable vs SP500_ret

Input:  data/processed_data.csv
Output: figures/timeseries.png
        figures/correlation_heatmap.png
        figures/rolling_corr.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

INPUT_FILE  = "data/processed_data.csv"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE, index_col="Date", parse_dates=True)

# Crisis / regime periods for annotation
REGIMES = [
    ("2000-03", "2002-10", "Dot-com bust",   "#ffd6d6"),
    ("2007-12", "2009-06", "GFC",             "#ffd6d6"),
    ("2020-02", "2020-04", "COVID crash",     "#ffd6d6"),
    ("2022-03", "2023-07", "Rate hike cycle", "#fff3cd"),
]

LABELS = {
    "SP500_ret"  : "S&P 500 Log Return",
    "VIX"        : "VIX (log)",
    "FEDFUNDS_d" : "Fed Funds Δ (pp)",
    "CPI_yoy"    : "CPI YoY (%)",
    "UNRATE_d"   : "Unemployment Δ (pp)",
    "M2_yoy"     : "M2 YoY Growth (%)",
    "SENTIMENT_d": "Consumer Sentiment Δ",
    "OIL_ret"    : "Oil Log Return",
    "GS10_d"     : "10Y Treasury Δ (pp)",
}


def add_regimes(ax):
    for start, end, label, color in REGIMES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.3, color=color, label=label)


# ── Plot 1: Time series grid ──────────────────────────────────────────────────
print("Plotting time series...")
fig, axes = plt.subplots(9, 1, figsize=(14, 28), sharex=True)
fig.suptitle("Macroeconomic Indicators & Market Variables (2000–2024)",
             fontsize=14, fontweight="bold", y=1.001)

for ax, col in zip(axes, df.columns):
    ax.plot(df.index, df[col], color="#2563eb", linewidth=1.2)
    add_regimes(ax)
    ax.set_ylabel(LABELS[col], fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.grid(axis="y", alpha=0.3)

# Legend for regimes (only on last ax)
handles = [mpatches.Patch(color="#ffd6d6", alpha=0.5, label="Recession/Crisis"),
           mpatches.Patch(color="#fff3cd", alpha=0.5, label="Rate Hike Cycle")]
axes[-1].legend(handles=handles, loc="lower left", fontsize=8)
axes[-1].set_xlabel("Date")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {FIGURES_DIR}/timeseries.png")


# ── Plot 2: Correlation heatmap ───────────────────────────────────────────────
print("Plotting correlation heatmap...")
corr = df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # show lower triangle only
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1,
            xticklabels=[LABELS[c] for c in corr.columns],
            yticklabels=[LABELS[c] for c in corr.index],
            ax=ax, linewidths=0.5)
ax.set_title("Correlation Matrix of Processed Variables", fontsize=13, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {FIGURES_DIR}/correlation_heatmap.png")


# ── Plot 3: Rolling 24-month correlation with SP500_ret ──────────────────────
print("Plotting rolling correlations...")
macro_vars = [c for c in df.columns if c != "SP500_ret"]
colors     = plt.cm.tab10.colors

fig, ax = plt.subplots(figsize=(14, 5))
for i, col in enumerate(macro_vars):
    roll_corr = df["SP500_ret"].rolling(24).corr(df[col])
    ax.plot(df.index, roll_corr, label=LABELS[col], color=colors[i], linewidth=1.2)

add_regimes(ax)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Rolling 24-Month Correlation with S&P 500 Returns", fontsize=13, fontweight="bold")
ax.set_ylabel("Pearson Correlation")
ax.set_xlabel("Date")
ax.legend(loc="lower left", fontsize=8, ncol=2)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/rolling_corr.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {FIGURES_DIR}/rolling_corr.png")

print("\nAll figures saved.")
