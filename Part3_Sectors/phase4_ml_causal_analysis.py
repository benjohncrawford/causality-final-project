"""Phase 4: CVAR + PCMCI + XGBoost consensus for predictive lead-lag edges."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

XGBOOST_AVAILABLE = False
XGBRegressor = None
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_IMPORT_ERROR = e

TIGRAMITE_AVAILABLE = False
try:
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI
    TIGRAMITE_AVAILABLE = True
except ImportError:
    pass

INPUT_CSV = "sector_data_all.csv"
PHASE3_DIR = "phase3_outputs/tables"
OUTPUT_DIR = "phase4_outputs"
TICKERS = ["XLF", "XLB", "XLE", "XLI", "XLK", "XLV", "XLY", "XLP", "XLU", "^GSPC"]
FREQUENCIES = ["daily", "weekly", "monthly"]
ALPHA = 0.05
XGB_TOP_N = 3
DEFAULT_LAGS = {"daily": 1, "weekly": 1, "monthly": 1}
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
    "verbosity": 0,
}

RANK_COLS = [
    "total_conditional_score",
    "core_causal_score",
    "conditional_var_p_adj",
    "pcmci_min_p_value",
]
RANK_ASC = [False, False, True, True]


def setup_output():
    base = Path(OUTPUT_DIR)
    (base / "tables").mkdir(parents=True, exist_ok=True)


def rank_edges(df, extra_sort=None):
    cols = list(RANK_COLS)
    asc = list(RANK_ASC)
    if extra_sort:
        cols = cols + [extra_sort]
        asc = asc + [True]
    return df.sort_values(cols, ascending=asc, na_position="last")


def load_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    out = {}
    for freq in FREQUENCIES:
        sub = df.loc[df["type"] == freq, TICKERS + ["Date"]].set_index("Date")
        sub = sub.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        out[freq] = sub
    return out


def load_selected_lags(phase3_dir):
    lag_file = Path(phase3_dir) / "var_lag_selection_results.csv"
    if not lag_file.exists():
        return DEFAULT_LAGS.copy()
    lag_df = pd.read_csv(lag_file)
    lag_col = next((c for c in lag_df.columns if c.startswith("selected_lag")), "bic_lag")
    return {row["resolution"]: max(1, int(row[lag_col])) for _, row in lag_df.iterrows()}


def load_conditional_var(phase3_dir):
    path = Path(phase3_dir) / "conditional_var_granger_fdr_adjusted_results.csv"
    if not path.exists():
        print("Warning: no Phase 3 CVAR results found.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} CVAR tests ({df['significant'].sum()} significant)")
    return df


def create_lagged_dataset(data, target_col, max_lag):
    feats = {
        f"{t}_lag{lag}": data[t].shift(lag)
        for t in TICKERS
        for lag in range(1, max_lag + 1)
    }
    X = pd.DataFrame(feats)
    y = data[target_col]
    mask = ~(X.isna().any(axis=1) | y.isna())
    return X[mask], y[mask]


def run_xgboost(data_dict, selected_lags):
    print("XGBoost feature importance")
    perf_cached = Path(OUTPUT_DIR) / "tables" / "xgboost_model_performance.csv"
    if not XGBOOST_AVAILABLE:
        raise RuntimeError(f"XGBoost unavailable: {XGBOOST_IMPORT_ERROR}")

    results, perf = [], []
    for resolution, data in data_dict.items():
        max_lag = selected_lags[resolution]
        for target in TICKERS:
            X, y = create_lagged_dataset(data, target, max_lag)
            if len(X) < 10:
                continue
            split = int(0.8 * len(X))
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(X.iloc[:split], y.iloc[:split])
            pred = model.predict(X.iloc[split:])
            y_test = y.iloc[split:]
            metrics = {
                "resolution": resolution,
                "target": target,
                "test_rmse": np.sqrt(mean_squared_error(y_test, pred)),
                "test_mae": mean_absolute_error(y_test, pred),
                "test_r2": r2_score(y_test, pred),
            }
            perf.append(metrics)

            cause_imp = {c: 0.0 for c in TICKERS}
            for name, imp in zip(X.columns, model.feature_importances_):
                cause = name.split("_lag")[0]
                if cause in cause_imp:
                    cause_imp[cause] += imp
            total = sum(cause_imp.values()) or 1.0
            for cause, imp in cause_imp.items():
                results.append({
                    "resolution": resolution,
                    "cause": cause,
                    "effect": target,
                    "xgb_importance": imp / total,
                    **metrics,
                })

    res_df = pd.DataFrame(results)
    perf_df = pd.DataFrame(perf)
    perf_df.to_csv(perf_cached, index=False)
    return res_df, perf_df


def top_xgb_edges(xgb_df, top_n=XGB_TOP_N):
    parts = []
    for resolution in xgb_df["resolution"].unique():
        for effect in TICKERS:
            sub = xgb_df[(xgb_df["resolution"] == resolution) & (xgb_df["effect"] == effect)]
            sub = sub[sub["cause"] != effect].sort_values("xgb_importance", ascending=False).head(top_n)
            sub = sub.copy()
            sub["xgb_top_edge"] = True
            sub["xgb_rank_for_effect"] = range(1, len(sub) + 1)
            parts.append(sub)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.to_csv(Path(OUTPUT_DIR) / "tables" / "xgboost_top_edges.csv", index=False)
    return out


def run_pcmci(data_dict, selected_lags):
    if not TIGRAMITE_AVAILABLE:
        print("PCMCI skipped (tigramite not installed)")
        return pd.DataFrame()

    rows = []
    for resolution, data in data_dict.items():
        tau_max = selected_lags[resolution]
        try:
            tdata = pp.DataFrame(data.values, datatime=np.arange(len(data)), var_names=list(data.columns))
            pcmci = PCMCI(tdata, cond_ind_test=ParCorr(significance="analytic"))
            out = pcmci.run_pcmci(tau_min=1, tau_max=tau_max, pc_alpha=ALPHA, alpha_level=ALPHA)
            p_matrix, val_matrix = out["p_matrix"], out.get("val_matrix")
            names = list(data.columns)
            for i, cause in enumerate(names):
                for j, effect in enumerate(names):
                    if cause == effect:
                        continue
                    for lag in range(1, p_matrix.shape[2]):
                        p = p_matrix[i, j, lag]
                        if np.isnan(p):
                            continue
                        rows.append({
                            "resolution": resolution,
                            "cause": cause,
                            "effect": effect,
                            "lag": lag,
                            "p_value": p,
                            "strength": val_matrix[i, j, lag] if val_matrix is not None else np.nan,
                            "significant": p < ALPHA,
                        })
            print(f"{resolution}: PCMCI done")
        except Exception as exc:
            print(f"{resolution}: PCMCI failed ({exc})")

    df = pd.DataFrame(rows)
    if len(df):
        sig = df[df["significant"] & (df["cause"] != df["effect"])]
        sig.to_csv(Path(OUTPUT_DIR) / "tables" / "pcmci_significant_edges.csv", index=False)
    return df


def build_consensus(cvar_df, xgb_top, pcmci_df):
    rows = []
    for resolution in FREQUENCIES:
        for cause in TICKERS:
            for effect in TICKERS:
                if cause == effect:
                    continue
                row = {
                    "resolution": resolution,
                    "cause": cause,
                    "effect": effect,
                    "conditional_var_edge": False,
                    "conditional_var_p_adj": np.nan,
                    "xgb_top_edge": False,
                    "xgb_importance": np.nan,
                    "pcmci_edge": False,
                    "pcmci_min_p_value": np.nan,
                }

                if len(cvar_df):
                    hit = cvar_df[
                        (cvar_df["resolution"] == resolution)
                        & (cvar_df["cause"] == cause)
                        & (cvar_df["effect"] == effect)
                    ]
                    if len(hit):
                        row["conditional_var_p_adj"] = hit.iloc[0]["p_adj"]
                        row["conditional_var_edge"] = bool(hit.iloc[0]["significant"])

                if len(xgb_top):
                    hit = xgb_top[
                        (xgb_top["resolution"] == resolution)
                        & (xgb_top["cause"] == cause)
                        & (xgb_top["effect"] == effect)
                    ]
                    if len(hit):
                        row["xgb_top_edge"] = True
                        row["xgb_importance"] = hit.iloc[0]["xgb_importance"]

                if len(pcmci_df):
                    hit = pcmci_df[
                        (pcmci_df["resolution"] == resolution)
                        & (pcmci_df["cause"] == cause)
                        & (pcmci_df["effect"] == effect)
                        & (pcmci_df["significant"])
                    ]
                    if len(hit):
                        row["pcmci_edge"] = True
                        row["pcmci_min_p_value"] = hit["p_value"].min()

                rows.append(row)

    df = pd.DataFrame(rows)
    df["cvar"] = df["conditional_var_edge"].astype(int)
    df["pcmci"] = df["pcmci_edge"].astype(int)
    df["xgb"] = df["xgb_top_edge"].astype(int)
    df["total_conditional_score"] = df["cvar"] + df["pcmci"] + df["xgb"]
    df["core_causal_score"] = df["cvar"] + df["pcmci"]
    return rank_edges(df, extra_sort="resolution")


def sp500_tables(consensus):
    leaders = consensus[(consensus["effect"] == "^GSPC") & (consensus["cause"] != "^GSPC")].copy()
    leaders = rank_edges(leaders, extra_sort="resolution")

    overall = leaders.groupby("cause", as_index=False).agg(
        total_conditional_score=("total_conditional_score", "sum"),
        core_causal_score=("core_causal_score", "sum"),
        cvar_support=("cvar", "sum"),
        pcmci_support=("pcmci", "sum"),
        xgb_support=("xgb", "sum"),
        conditional_var_p_adj=("conditional_var_p_adj", "min"),
        pcmci_min_p_value=("pcmci_min_p_value", "min"),
        num_resolutions=("resolution", "nunique"),
    )
    overall = rank_edges(overall)
    overall.to_csv(Path(OUTPUT_DIR) / "tables" / "sp500_overall_sector_leader_ranking.csv", index=False)
    return leaders, overall


def save_derived_tables(consensus):
    consensus.to_csv(Path(OUTPUT_DIR) / "tables" / "consensus_edges.csv", index=False)
    strong = consensus[consensus["total_conditional_score"] >= 2].copy()
    strong = rank_edges(strong, extra_sort="resolution")
    strong.to_csv(Path(OUTPUT_DIR) / "tables" / "strongest_consensus_edges.csv", index=False)
    print(f"consensus: {len(consensus)} edges, score>=2: {len(strong)}, core=2: {(consensus['core_causal_score']==2).sum()}")


def main():
    print("Phase 4: CVAR + PCMCI + XGBoost consensus")
    setup_output()
    data = load_data(INPUT_CSV)
    lags = load_selected_lags(PHASE3_DIR)
    cvar = load_conditional_var(PHASE3_DIR)
    xgb_all, _ = run_xgboost(data, lags)
    xgb_top = top_xgb_edges(xgb_all)
    pcmci = run_pcmci(data, lags)
    consensus = build_consensus(cvar, xgb_top, pcmci)
    save_derived_tables(consensus)
    sp500_detail, sp500_rank = sp500_tables(consensus)
    print("\nTop sectors predicting ^GSPC:")
    for i, row in sp500_rank.head(5).iterrows():
        print(f"  {row['cause']}: total={int(row['total_conditional_score'])}, core={int(row['core_causal_score'])}")


if __name__ == "__main__":
    main()
