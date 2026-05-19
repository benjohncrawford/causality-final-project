"""Phase 3: Conditional VAR Granger causality for sector ETF returns."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR

warnings.filterwarnings("ignore")

INPUT_CSV = "sector_data_all.csv"
OUTPUT_DIR = "phase3_outputs"
TICKERS = ["XLF", "XLB", "XLE", "XLI", "XLK", "XLV", "XLY", "XLP", "XLU", "^GSPC"]
ALPHA = 0.05
USE_LAG_CRITERION = "bic"
FREQUENCIES = ["daily", "weekly", "monthly"]


def setup_output():
    base = Path(OUTPUT_DIR)
    (base / "tables").mkdir(parents=True, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    out = {}
    for freq in FREQUENCIES:
        sub = df.loc[df["type"] == freq, TICKERS + ["Date"]].set_index("Date")
        sub = sub.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        out[freq] = sub
        print(f"{freq}: {sub.shape[0]} obs, {sub.index.min().date()} – {sub.index.max().date()}")
    return out


def run_adf(data_dict):
    rows = []
    for resolution, df in data_dict.items():
        for ticker in TICKERS:
            stat, pval, used_lag, _, _, _ = adfuller(df[ticker], autolag="AIC")
            rows.append({
                "resolution": resolution,
                "ticker": ticker,
                "adf_stat": stat,
                "p_value": pval,
                "used_lag": used_lag,
                "n_obs": len(df),
                "stationary_5pct": pval < ALPHA,
            })
    adf_df = pd.DataFrame(rows)
    adf_df.to_csv(Path(OUTPUT_DIR) / "tables" / "adf_stationarity_results.csv", index=False)
    return adf_df


def select_var_lags(data_dict):
    selected, rows = {}, []
    caps = {"daily": 10, "weekly": 8, "monthly": 4}
    for resolution, df in data_dict.items():
        n_obs = len(df)
        maxlags = min(caps[resolution], max(1, n_obs // 20))
        order = VAR(df).select_order(maxlags=maxlags)
        lag = max(1, getattr(order, USE_LAG_CRITERION))
        selected[resolution] = lag
        rows.append({
            "resolution": resolution,
            "n_obs": n_obs,
            "maxlags_used": maxlags,
            "aic_lag": order.aic,
            "bic_lag": order.bic,
            "hqic_lag": order.hqic,
            f"selected_lag_{USE_LAG_CRITERION}": lag,
        })
        print(f"{resolution}: lag={lag} (BIC={order.bic})")
    lag_df = pd.DataFrame(rows)
    lag_df.to_csv(Path(OUTPUT_DIR) / "tables" / "var_lag_selection_results.csv", index=False)
    return lag_df, selected


def fit_var_models(data_dict, selected_lags):
    models = {}
    for resolution, df in data_dict.items():
        lag = selected_lags[resolution]
        models[resolution] = VAR(df).fit(lag)
        print(f"{resolution}: VAR({lag}) fitted")
    return models


def run_conditional_granger(fitted_models):
    rows = []
    for resolution, model in fitted_models.items():
        lag = model.k_ar
        for cause in TICKERS:
            for effect in TICKERS:
                if cause == effect:
                    continue
                try:
                    res = model.test_causality(caused=effect, causing=cause, kind="f", signif=ALPHA)
                    rows.append({
                        "resolution": resolution,
                        "cause": cause,
                        "effect": effect,
                        "lag": lag,
                        "test_statistic": res.test_statistic,
                        "p_value": res.pvalue,
                        "df": res.df,
                        "error_msg": None,
                    })
                except Exception as exc:
                    rows.append({
                        "resolution": resolution,
                        "cause": cause,
                        "effect": effect,
                        "lag": lag,
                        "test_statistic": np.nan,
                        "p_value": np.nan,
                        "df": np.nan,
                        "error_msg": str(exc),
                    })
    return pd.DataFrame(rows)


def apply_fdr(raw_df):
    parts = []
    for resolution in raw_df["resolution"].unique():
        part = raw_df[raw_df["resolution"] == resolution].copy()
        valid = part["p_value"].notna()
        if valid.any():
            reject, p_adj, _, _ = multipletests(part.loc[valid, "p_value"], alpha=ALPHA, method="fdr_bh")
            part["p_adj"] = np.nan
            part["significant"] = False
            part.loc[valid, "p_adj"] = p_adj
            part.loc[valid, "significant"] = reject
        else:
            part["p_adj"] = np.nan
            part["significant"] = False
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def main():
    print("Phase 3: Conditional VAR Granger")
    setup_output()
    data = load_data(INPUT_CSV)
    run_adf(data)
    _, selected_lags = select_var_lags(data)
    models = fit_var_models(data, selected_lags)
    raw = run_conditional_granger(models)
    results = apply_fdr(raw)
    out = Path(OUTPUT_DIR) / "tables" / "conditional_var_granger_fdr_adjusted_results.csv"
    results.to_csv(out, index=False)
    sig = results["significant"].sum()
    print(f"Saved {out} ({sig} significant edges)")


if __name__ == "__main__":
    main()
