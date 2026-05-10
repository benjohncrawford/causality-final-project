"""
Module 1: Data Pulling
======================
Pulls all required data from Yahoo Finance and FRED.
Time range: 2000-01-01 to 2024-12-31, monthly frequency.

Variables:
  Yahoo Finance:
    - S&P 500       (^GSPC)
    - VIX           (^VIX)

  FRED:
    - Federal Funds Rate    (FEDFUNDS)
    - CPI                   (CPIAUCSL)
    - Unemployment Rate     (UNRATE)
    - M2 Money Supply       (M2SL)
    - Consumer Sentiment    (UMCSENT)
    - WTI Oil Price         (DCOILWTICO)
    - 10Y Treasury Yield    (GS10)

Output:
    data/raw_data.csv
"""

import pandas as pd
import yfinance as yf
from fredapi import Fred
import os

# ── Config ──────────────────────────────────────────────────────────────────
FRED_API_KEY = "APIKEY"   # <-- replace with your key
START        = "2000-01-01"
END          = "2024-12-31"
OUTPUT_DIR   = "data"
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "raw_data.csv")
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Yahoo Finance ─────────────────────────────────────────────────────────
def pull_yahoo(ticker: str, col_name: str) -> pd.Series:
    """Download monthly close price from Yahoo Finance."""
    df = yf.download(ticker, start=START, end=END, interval="1mo", auto_adjust=True, progress=False)
    series = df["Close"].squeeze()
    series.index = series.index.to_period("M").to_timestamp("M")  # month-end
    series.name = col_name
    return series


print("Pulling Yahoo Finance data...")
sp500 = pull_yahoo("^GSPC", "SP500")
vix   = pull_yahoo("^VIX",  "VIX")


# ── 2. FRED ──────────────────────────────────────────────────────────────────
def pull_fred(series_id: str, col_name: str, fred: Fred) -> pd.Series:
    """Download a FRED series and resample to month-end."""
    s = fred.get_series(series_id, observation_start=START, observation_end=END)
    s = s.resample("ME").last()          # month-end; for monthly series this is a no-op
    s.index = s.index.to_period("M").to_timestamp("M")
    s.name = col_name
    return s


print("Pulling FRED data...")
fred = Fred(api_key=FRED_API_KEY)

fedfunds  = pull_fred("FEDFUNDS",   "FEDFUNDS",  fred)
cpi       = pull_fred("CPIAUCSL",   "CPI",       fred)
unrate    = pull_fred("UNRATE",     "UNRATE",    fred)
m2        = pull_fred("M2SL",       "M2",        fred)
sentiment = pull_fred("UMCSENT",    "SENTIMENT", fred)
oil       = pull_fred("DCOILWTICO", "OIL",       fred)
gs10      = pull_fred("GS10",       "GS10",      fred)


# ── 3. Merge ─────────────────────────────────────────────────────────────────
print("Merging all series...")
df = pd.concat([sp500, vix, fedfunds, cpi, unrate, m2, sentiment, oil, gs10], axis=1)
df.index.name = "Date"

# Keep only rows where ALL series have data (inner join on time)
df = df.dropna(how="all")

print(f"\nRaw data shape: {df.shape}")
print(f"Date range:     {df.index.min()} → {df.index.max()}")
print(f"Missing values per column:\n{df.isnull().sum()}")

df.to_csv(OUTPUT_FILE)
print(f"\nSaved to {OUTPUT_FILE}")
