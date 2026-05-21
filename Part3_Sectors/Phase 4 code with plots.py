#!/usr/bin/env python
# coding: utf-8

# In[14]:


"""
Phase 4: CVAR + PCMCI + XGBoost consensus for predictive lead-lag edges.

"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

PHASE3_DIR = "."

OUTPUT_DIR = "phase4_outputs"

TICKERS = ["XLF", "XLB", "XLE", "XLI", "XLK", "XLV", "XLY", "XLP", "XLU", "^GSPC"]

FREQUENCIES = ["daily", "weekly", "monthly"]

ALPHA = 0.05
XGB_TOP_N = 3

DEFAULT_LAGS = {
    "daily": 1,
    "weekly": 1,
    "monthly": 1,
}

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

SECTOR_NAMES = {
    "XLF": "Financials",
    "XLB": "Materials",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "^GSPC": "S&P 500",
}

def setup_output():
    base = Path(OUTPUT_DIR)
    (base / "tables").mkdir(parents=True, exist_ok=True)
    (base / "figures").mkdir(parents=True, exist_ok=True)


def setup_figures():
    fig_dir = Path(OUTPUT_DIR) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def label_sector(x):
    return SECTOR_NAMES.get(x, x)


def rank_edges(df, extra_sort=None):
    cols = list(RANK_COLS)
    asc = list(RANK_ASC)

    if extra_sort:
        cols = cols + [extra_sort]
        asc = asc + [True]

    existing_cols = [c for c in cols if c in df.columns]
    existing_asc = [asc[i] for i, c in enumerate(cols) if c in df.columns]

    return df.sort_values(existing_cols, ascending=existing_asc, na_position="last")

def load_data(path):

    df = pd.read_csv(path)

    print("Columns in input file:")
    print(df.columns.tolist())

    if "Date" not in df.columns:
        raise ValueError("Input CSV must contain a Date column.")

    missing_tickers = [t for t in TICKERS if t not in df.columns]
    if missing_tickers:
        raise ValueError(f"Missing ticker columns in {path}: {missing_tickers}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    possible_freq_cols = ["type", "resolution", "frequency", "freq"]
    freq_col = None

    for col in possible_freq_cols:
        if col in df.columns:
            freq_col = col
            break

    out = {}

    # Case 1: already has daily / weekly / monthly stacked rows
    if freq_col is not None:
        print(f"Using existing frequency column: {freq_col}")

        for freq in FREQUENCIES:
            sub = df.loc[df[freq_col] == freq, TICKERS + ["Date"]].set_index("Date")
            sub = sub.apply(pd.to_numeric, errors="coerce")
            sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
            out[freq] = sub

            print(f"{freq}: {sub.shape[0]} obs, {sub.index.min().date()} – {sub.index.max().date()}")

        return out

    # Case 2: wide raw price file, no type column
    print("No type/resolution/frequency column found.")
    print("Treating input as wide daily price data and converting to returns.")

    df = df.set_index("Date")
    prices = df[TICKERS].apply(pd.to_numeric, errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna()

    # Daily returns
    daily_returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    # Weekly returns: resample prices first, then compute returns
    weekly_prices = prices.resample("W-FRI").last().dropna()
    weekly_returns = weekly_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    # Monthly returns: use ME if available; if not, fallback to M
    try:
        monthly_prices = prices.resample("ME").last().dropna()
    except ValueError:
        monthly_prices = prices.resample("M").last().dropna()

    monthly_returns = monthly_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    out["daily"] = daily_returns
    out["weekly"] = weekly_returns
    out["monthly"] = monthly_returns

    for freq, sub in out.items():
        print(f"{freq}: {sub.shape[0]} obs, {sub.index.min().date()} – {sub.index.max().date()}")

    return out


def load_selected_lags(phase3_dir):

    lag_file = Path(phase3_dir) / "var_lag_selection_results.csv"


    if not lag_file.exists():
        print("Warning: no Phase 3 lag selection file found. Using default lag = 1.")
        return DEFAULT_LAGS.copy()

    lag_df = pd.read_csv(lag_file)

    print("Loaded Phase 3 lag selection results:")
    print(lag_df.head())

    lag_col = next((c for c in lag_df.columns if c.startswith("selected_lag")), None)

    if lag_col is None:
        if "bic_lag" in lag_df.columns:
            lag_col = "bic_lag"
        else:
            print("Warning: no selected_lag or bic_lag column found. Using default lag = 1.")
            return DEFAULT_LAGS.copy()

    selected = {}

    for _, row in lag_df.iterrows():
        resolution = row["resolution"]
        selected[resolution] = max(1, int(row[lag_col]))

    for freq in FREQUENCIES:
        if freq not in selected:
            selected[freq] = DEFAULT_LAGS[freq]

    print("Selected lags used in Phase 4:")
    print(selected)

    return selected


def load_conditional_var(phase3_dir):

    path = Path(phase3_dir) / "conditional_var_granger_fdr_adjusted_results.csv"

    if not path.exists():
        print("Warning: no Phase 3 CVAR results found.")
        return pd.DataFrame()

    df = pd.read_csv(path)

    print(f"Loaded {len(df)} CVAR tests ({df['significant'].sum()} significant)")
    print(df.head())

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

    if not XGBOOST_AVAILABLE:
        raise RuntimeError(f"XGBoost unavailable: {XGBOOST_IMPORT_ERROR}")

    results = []
    perf = []

    for resolution, data in data_dict.items():
        max_lag = selected_lags.get(resolution, 1)

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

    perf_df.to_csv(Path(OUTPUT_DIR) / "tables" / "xgboost_model_performance.csv", index=False)
    res_df.to_csv(Path(OUTPUT_DIR) / "tables" / "xgboost_all_importance.csv", index=False)

    return res_df, perf_df


def top_xgb_edges(xgb_df, top_n=XGB_TOP_N):
    parts = []

    for resolution in xgb_df["resolution"].unique():
        for effect in TICKERS:
            sub = xgb_df[
                (xgb_df["resolution"] == resolution)
                & (xgb_df["effect"] == effect)
            ]

            sub = (
                sub[sub["cause"] != effect]
                .sort_values("xgb_importance", ascending=False)
                .head(top_n)
            )

            sub = sub.copy()
            sub["xgb_top_edge"] = True
            sub["xgb_rank_for_effect"] = range(1, len(sub) + 1)

            parts.append(sub)

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    out.to_csv(Path(OUTPUT_DIR) / "tables" / "xgboost_top_edges.csv", index=False)

    return out

def run_pcmci(data_dict, selected_lags):
    if not TIGRAMITE_AVAILABLE:
        print("PCMCI skipped because tigramite is not installed.")
        return pd.DataFrame()

    rows = []

    for resolution, data in data_dict.items():
        tau_max = selected_lags.get(resolution, 1)

        try:
            tdata = pp.DataFrame(
                data.values,
                datatime=np.arange(len(data)),
                var_names=list(data.columns),
            )

            pcmci = PCMCI(
                tdata,
                cond_ind_test=ParCorr(significance="analytic"),
            )

            out = pcmci.run_pcmci(
                tau_min=1,
                tau_max=tau_max,
                pc_alpha=ALPHA,
                alpha_level=ALPHA,
            )

            p_matrix = out["p_matrix"]
            val_matrix = out.get("val_matrix")

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
        df.to_csv(Path(OUTPUT_DIR) / "tables" / "pcmci_all_edges.csv", index=False)

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

                # Conditional VAR / Granger
                if len(cvar_df):
                    hit = cvar_df[
                        (cvar_df["resolution"] == resolution)
                        & (cvar_df["cause"] == cause)
                        & (cvar_df["effect"] == effect)
                    ]

                    if len(hit):
                        row["conditional_var_p_adj"] = hit.iloc[0]["p_adj"]
                        row["conditional_var_edge"] = bool(hit.iloc[0]["significant"])

                # XGBoost top edge
                if len(xgb_top):
                    hit = xgb_top[
                        (xgb_top["resolution"] == resolution)
                        & (xgb_top["cause"] == cause)
                        & (xgb_top["effect"] == effect)
                    ]

                    if len(hit):
                        row["xgb_top_edge"] = True
                        row["xgb_importance"] = hit.iloc[0]["xgb_importance"]

                # PCMCI
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
    leaders = consensus[
        (consensus["effect"] == "^GSPC")
        & (consensus["cause"] != "^GSPC")
    ].copy()

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

    leaders.to_csv(Path(OUTPUT_DIR) / "tables" / "sp500_sector_leader_detail.csv", index=False)
    overall.to_csv(Path(OUTPUT_DIR) / "tables" / "sp500_overall_sector_leader_ranking.csv", index=False)

    return leaders, overall


def save_derived_tables(consensus):
    consensus.to_csv(Path(OUTPUT_DIR) / "tables" / "consensus_edges.csv", index=False)

    strong = consensus[consensus["total_conditional_score"] >= 2].copy()
    strong = rank_edges(strong, extra_sort="resolution")

    strong.to_csv(Path(OUTPUT_DIR) / "tables" / "strongest_consensus_edges.csv", index=False)

    print(
        f"consensus: {len(consensus)} edges, "
        f"score>=2: {len(strong)}, "
        f"core=2: {(consensus['core_causal_score'] == 2).sum()}"
    )


### Plots

def plot_sp500_sector_ranking(sp500_rank, fig_dir):
    df = sp500_rank.copy()
    df["sector_label"] = df["cause"].map(label_sector)
    df = df.sort_values("total_conditional_score", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(df["sector_label"], df["total_conditional_score"])

    plt.xlabel("Total Consensus Score")
    plt.ylabel("Sector")
    plt.title("Sectors Leading the S&P 500: Script Consensus Ranking")

    plt.tight_layout()
    plt.savefig(fig_dir / "sp500_sector_leader_ranking.png", dpi=300)
    plt.show()
    plt.close()


def plot_sp500_method_support(sp500_rank, fig_dir):
    df = sp500_rank.copy()
    df["sector_label"] = df["cause"].map(label_sector)
    df = df.sort_values("total_conditional_score", ascending=False)

    x = np.arange(len(df))

    plt.figure(figsize=(11, 6))

    plt.bar(x, df["cvar_support"], label="Conditional VAR")
    plt.bar(
        x,
        df["pcmci_support"],
        bottom=df["cvar_support"],
        label="PCMCI",
    )
    plt.bar(
        x,
        df["xgb_support"],
        bottom=df["cvar_support"] + df["pcmci_support"],
        label="XGBoost",
    )

    plt.xticks(x, df["sector_label"], rotation=45, ha="right")

    plt.ylabel("Support Count Across Resolutions")
    plt.title("Method Support for Sectors Leading the S&P 500")
    plt.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "sp500_method_support_stacked.png", dpi=300)
    plt.show()
    plt.close()


def plot_consensus_heatmaps(consensus, fig_dir):
    """
    Blue-green heatmaps.

    Score meaning:
        0 = no method support
        1 = one method supports edge
        2 = two methods support edge
        3 = all three methods support edge
    """
    labels = [label_sector(t) for t in TICKERS]

    for resolution in FREQUENCIES:
        sub = consensus[consensus["resolution"] == resolution].copy()

        matrix = sub.pivot(
            index="cause",
            columns="effect",
            values="total_conditional_score",
        )

        matrix = matrix.reindex(index=TICKERS, columns=TICKERS)

        plt.figure(figsize=(10, 8))

        # Blue-green color palette
        plt.imshow(
            matrix,
            aspect="auto",
            cmap="YlGnBu",
            vmin=0,
            vmax=3,
        )

        cbar = plt.colorbar(label="Total Consensus Score")
        cbar.set_ticks([0, 1, 2, 3])

        plt.xticks(np.arange(len(TICKERS)), labels, rotation=45, ha="right")
        plt.yticks(np.arange(len(TICKERS)), labels)

        plt.xlabel("Effect")
        plt.ylabel("Cause")
        plt.title(f"{resolution.capitalize()} Consensus Edge Heatmap")

        for i in range(len(TICKERS)):
            for j in range(len(TICKERS)):
                val = matrix.iloc[i, j]

                if pd.notna(val):
                    text_color = "white" if val >= 2 else "black"
                    plt.text(
                        j,
                        i,
                        int(val),
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=9,
                    )

        plt.tight_layout()
        plt.savefig(fig_dir / f"consensus_heatmap_{resolution}.png", dpi=300)
        plt.show()
        plt.close()


def plot_strongest_edges(consensus, fig_dir, top_n=15):
    strong = consensus[consensus["total_conditional_score"] > 0].copy()

    strong["edge_label"] = (
        strong["cause"].map(label_sector)
        + " → "
        + strong["effect"].map(label_sector)
        + " ("
        + strong["resolution"]
        + ")"
    )

    strong = strong.sort_values(
        [
            "total_conditional_score",
            "core_causal_score",
            "conditional_var_p_adj",
            "pcmci_min_p_value",
        ],
        ascending=[False, False, True, True],
        na_position="last",
    ).head(top_n)

    strong = strong.sort_values("total_conditional_score", ascending=True)

    plt.figure(figsize=(12, 7))
    plt.barh(strong["edge_label"], strong["total_conditional_score"])

    plt.xlabel("Total Consensus Score")
    plt.ylabel("Directed Edge")
    plt.title(f"Top {top_n} Strongest Consensus Edges")

    plt.tight_layout()
    plt.savefig(fig_dir / "top_strongest_consensus_edges.png", dpi=300)
    plt.show()
    plt.close()


def plot_first_mover_sector_scores(consensus, fig_dir):
    df = consensus.copy()

    leader_scores = df.groupby("cause", as_index=False).agg(
        total_outgoing_score=("total_conditional_score", "sum"),
        core_outgoing_score=("core_causal_score", "sum"),
        cvar_outgoing=("cvar", "sum"),
        pcmci_outgoing=("pcmci", "sum"),
        xgb_outgoing=("xgb", "sum"),
    )

    leader_scores = leader_scores[leader_scores["cause"] != "^GSPC"]
    leader_scores["sector_label"] = leader_scores["cause"].map(label_sector)

    leader_scores = leader_scores.sort_values(
        [
            "total_outgoing_score",
            "core_outgoing_score",
            "cvar_outgoing",
            "pcmci_outgoing",
            "xgb_outgoing",
        ],
        ascending=[True, True, True, True, True],
    )

    leader_scores.to_csv(
        Path(OUTPUT_DIR) / "tables" / "first_mover_sector_ranking.csv",
        index=False,
    )

    plt.figure(figsize=(10, 6))
    plt.barh(leader_scores["sector_label"], leader_scores["total_outgoing_score"])

    plt.xlabel("Total Outgoing Consensus Score")
    plt.ylabel("Sector")
    plt.title("First-Mover Sector Ranking Across All Directed Edges")

    plt.tight_layout()
    plt.savefig(fig_dir / "first_mover_sector_ranking.png", dpi=300)
    plt.show()
    plt.close()


def make_all_plots(consensus, sp500_rank):
    fig_dir = setup_figures()

    plot_sp500_sector_ranking(sp500_rank, fig_dir)
    plot_sp500_method_support(sp500_rank, fig_dir)
    plot_consensus_heatmaps(consensus, fig_dir)
    plot_strongest_edges(consensus, fig_dir, top_n=15)
    plot_first_mover_sector_scores(consensus, fig_dir)

    print(f"Saved figures to: {fig_dir.resolve()}")


def main():
    print("Phase 4: CVAR + PCMCI + XGBoost consensus")

    setup_output()

    data = load_data(INPUT_CSV)

    lags = load_selected_lags(PHASE3_DIR)

    cvar = load_conditional_var(PHASE3_DIR)

    xgb_all, xgb_perf = run_xgboost(data, lags)

    xgb_top = top_xgb_edges(xgb_all)

    pcmci = run_pcmci(data, lags)

    consensus = build_consensus(cvar, xgb_top, pcmci)

    save_derived_tables(consensus)

    sp500_detail, sp500_rank = sp500_tables(consensus)

    make_all_plots(consensus, sp500_rank)

    print("\nTop sectors predicting / leading ^GSPC:")
    for _, row in sp500_rank.head(10).iterrows():
        print(
            f"{row['cause']}: "
            f"total={int(row['total_conditional_score'])}, "
            f"core={int(row['core_causal_score'])}, "
            f"CVAR={int(row['cvar_support'])}, "
            f"PCMCI={int(row['pcmci_support'])}, "
            f"XGB={int(row['xgb_support'])}"
        )

main()


# In[ ]:




