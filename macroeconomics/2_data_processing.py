"""
Module 2: Data Processing
=========================
Transforms raw data into stationary, model-ready variables.

Steps:
  1. Load raw_data.csv
  2. Transform variables (log returns, YoY%, first differences)
  3. ADF stationarity test on all variables
  4. Forward-fill remaining NaNs, drop incomplete rows
  5. Normalize (zero mean, unit variance) → model_data.csv
  6. Save both processed (human-readable) and normalized (model-ready) versions

Input:  data/raw_data.csv
Output: data/processed_data.csv   ← interpretable, for visualization
        data/model_data.csv       ← normalized, for PC / Granger / CD-NOD
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

INPUT_FILE     = "data/raw_data.csv"
PROCESSED_FILE = "data/processed_data.csv"
MODEL_FILE     = "data/model_data.csv"


# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE, index_col="Date", parse_dates=True)
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")


# ── 2. Transform variables ────────────────────────────────────────────────────
processed = pd.DataFrame(index=df.index)

# S&P 500 → monthly log return
processed["SP500_ret"] = np.log(df["SP500"] / df["SP500"].shift(1))

# VIX → log level (already stationary in levels, but log compresses extremes)
processed["VIX"] = np.log(df["VIX"])

# Federal Funds Rate → first difference (MoM change in rate)
processed["FEDFUNDS_d"] = df["FEDFUNDS"].diff()

# CPI → year-over-year % change (annualized inflation rate)
processed["CPI_yoy"] = df["CPI"].pct_change(12) * 100

# Unemployment → first difference
processed["UNRATE_d"] = df["UNRATE"].diff()

# M2 → year-over-year % change (growth rate)
processed["M2_yoy"] = df["M2"].pct_change(12) * 100

# Consumer Sentiment → first difference
processed["SENTIMENT_d"] = df["SENTIMENT"].diff()

# Oil → monthly log return
processed["OIL_ret"] = np.log(df["OIL"] / df["OIL"].shift(1))

# 10Y Treasury → first difference
processed["GS10_d"] = df["GS10"].diff()

# Forward-fill any remaining NaNs (at most a few from data gaps), then drop
processed = processed.ffill().dropna()

print(f"\nProcessed data shape: {processed.shape}")
print(f"Date range: {processed.index.min()} → {processed.index.max()}")


# ── 3. ADF Stationarity Tests ─────────────────────────────────────────────────
print("\n── ADF Stationarity Test Results ──────────────────────────────")
print(f"{'Variable':<20} {'ADF Stat':>10} {'p-value':>10} {'Stationary?':>12}")
print("-" * 55)

for col in processed.columns:
    result = adfuller(processed[col].dropna(), autolag="AIC")
    adf_stat = result[0]
    p_val    = result[1]
    stationary = "✅ Yes" if p_val < 0.05 else "❌ No"
    print(f"{col:<20} {adf_stat:>10.3f} {p_val:>10.4f} {stationary:>12}")

print()


# ── 4. Save processed (human-readable) ───────────────────────────────────────
processed.to_csv(PROCESSED_FILE)
print(f"Saved processed data → {PROCESSED_FILE}")


# ── 5. Normalize → model_data ────────────────────────────────────────────────
scaler     = StandardScaler()
normalized = pd.DataFrame(
    scaler.fit_transform(processed),
    index   = processed.index,
    columns = processed.columns
)

normalized.to_csv(MODEL_FILE)
print(f"Saved normalized model data → {MODEL_FILE}")

print("\nVariable descriptions:")
descriptions = {
    "SP500_ret"  : "S&P 500 monthly log return",
    "VIX"        : "VIX log level",
    "FEDFUNDS_d" : "Fed Funds Rate MoM change (pp)",
    "CPI_yoy"    : "CPI year-over-year inflation (%)",
    "UNRATE_d"   : "Unemployment Rate MoM change (pp)",
    "M2_yoy"     : "M2 Money Supply YoY growth (%)",
    "SENTIMENT_d": "Consumer Sentiment MoM change",
    "OIL_ret"    : "WTI Oil monthly log return",
    "GS10_d"     : "10Y Treasury Yield MoM change (pp)",
}
for k, v in descriptions.items():
    print(f"  {k:<20} {v}")
