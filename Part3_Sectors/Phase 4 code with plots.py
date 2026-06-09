# Plots

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

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

# Color Palet
C_NAVY      = "#0D2D6B" 
C_ROYAL     = "#1A56B0"
C_PERIWINK  = "#5B8FE8"
C_STEEL     = "#B8CBF0" 
C_STEM      = "#111111" 

C_CVAR      = C_NAVY
C_PCMCI     = C_PERIWINK
C_XGB       = C_STEEL

BLUE_CMAP = LinearSegmentedColormap.from_list(
    "blue_ramp",
    ["#FFFFFF", C_STEEL, C_PERIWINK, C_ROYAL, C_NAVY],
)

def _style_ax(ax):
    """Apply shared axis styling to all plots."""
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", length=0, pad=6)
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")


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

    if freq_col is not None:

        for freq in FREQUENCIES:
            sub = df.loc[df[freq_col] == freq, TICKERS + ["Date"]].set_index("Date")
            sub = sub.apply(pd.to_numeric, errors="coerce")
            sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
            out[freq] = sub

        return out

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


    return selected


def load_conditional_var(phase3_dir):
    
    path = Path(phase3_dir) / "conditional_var_granger_fdr_adjusted_results.csv"

    if not path.exists():
        print("Warning: no Phase 3 CVAR results found.")
        return pd.DataFrame()

    df = pd.read_csv(path)

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

                # XGBoost
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


### Plots

def plot_sp500_sector_ranking(sp500_rank, fig_dir):

    df = sp500_rank.copy()
    df["sector_label"] = df["cause"].map(label_sector)
    df = df.sort_values(
        ["core_causal_score", "total_conditional_score"],
        ascending=[True, True]
    )

    max_score = df["core_causal_score"].max() or 1
    bar_colors = [
        BLUE_CMAP(v / max_score) for v in df["core_causal_score"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    ax.barh(df["sector_label"], df["core_causal_score"],
            color=bar_colors, linewidth=0)

    ax.set_xlabel("Core Causal Score (Conditional VAR + PCMCI)", fontsize=11, color="#555", labelpad=8)
    ax.set_title(
        "Sectors Leading the S&P 500: Consensus Ranking",
        fontsize=14, fontweight="bold", color="#111", pad=14, loc="left"
    )
    ax.text(
        0, 1.01,
        "Sorted by core causal score (CVAR + PCMCI)  ·  XGBoost excluded from ranking",
        transform=ax.transAxes, fontsize=9, color="#888", va="bottom"
    )
    ax.xaxis.grid(True, color="#E5E5E5", linewidth=0.8, zorder=0)
    ax.yaxis.grid(False)
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["left"].set_linewidth(0.8)
    ax.tick_params(axis="both", length=0, pad=6)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    ax.tick_params(axis="y", labelsize=10, labelcolor="#222")
    ax.tick_params(axis="x", labelsize=10, labelcolor="#555")

    plt.tight_layout()
    plt.savefig(fig_dir / "sp500_sector_leader_ranking.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close()


def plot_sp500_method_support(sp500_rank, fig_dir):

    df = sp500_rank.copy()
    df["sector_label"] = df["cause"].map(label_sector)
    df = df.sort_values("total_conditional_score", ascending=False)

    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")

    ax.bar(x, df["cvar_support"],  color=C_CVAR,   linewidth=0, label="Conditional VAR", zorder=3)
    ax.bar(x, df["pcmci_support"], color=C_PCMCI,  linewidth=0, label="PCMCI",
           bottom=df["cvar_support"], zorder=3)
    ax.bar(x, df["xgb_support"],   color=C_XGB,    linewidth=0, label="XGBoost",
           bottom=df["cvar_support"] + df["pcmci_support"], zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(df["sector_label"], rotation=45, ha="right",
                       fontsize=10, color="#222")
    ax.set_ylabel("Support Count Across Resolutions", fontsize=11,
                  color="#555", labelpad=8)
    ax.set_title(
        "Method Support for Sectors Leading the S&P 500",
        fontsize=14, fontweight="bold", color="#111", pad=14, loc="left"
    )

    legend_handles = [
        mpatches.Patch(facecolor=C_CVAR,  label="Conditional VAR (Granger)"),
        mpatches.Patch(facecolor=C_PCMCI, label="PCMCI"),
        mpatches.Patch(facecolor=C_XGB,   label="XGBoost"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=10,
              labelcolor="#333", handlelength=1.2)

    _style_ax(ax)

    plt.tight_layout()
    plt.savefig(fig_dir / "sp500_method_support_stacked.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close()


def plot_consensus_heatmaps(consensus, fig_dir):

    labels = [label_sector(t) for t in TICKERS]

    for resolution in FREQUENCIES:
        sub = consensus[consensus["resolution"] == resolution].copy()

        matrix = sub.pivot(
            index="cause",
            columns="effect",
            values="total_conditional_score",
        )

        matrix = matrix.reindex(index=TICKERS, columns=TICKERS)

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap=BLUE_CMAP,
            vmin=0,
            vmax=3,
        )

        cbar = plt.colorbar(im, ax=ax, label="Total Consensus Score")
        cbar.set_ticks([0, 1, 2, 3])
        cbar.ax.tick_params(labelsize=10, labelcolor="#555")
        cbar.ax.yaxis.label.set_color("#555")
        cbar.ax.yaxis.label.set_fontsize(11)
        cbar.outline.set_visible(False)

        ax.set_xticks(np.arange(len(TICKERS)))
        ax.set_yticks(np.arange(len(TICKERS)))
        ax.set_xticklabels(labels, rotation=45, ha="right",
                           fontsize=10, color="#222")
        ax.set_yticklabels(labels, fontsize=10, color="#222")

        ax.set_xlabel("Effect", fontsize=11, color="#555", labelpad=8)
        ax.set_ylabel("Cause",  fontsize=11, color="#555", labelpad=8)
        ax.set_title(
            f"{resolution.capitalize()} Consensus Edge Heatmap",
            fontsize=14, fontweight="bold", color="#111", pad=14, loc="left"
        )

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0, pad=6)

        for i in range(len(TICKERS)):
            for j in range(len(TICKERS)):
                val = matrix.iloc[i, j]
                if pd.notna(val):
                    text_color = "white" if val >= 2 else C_NAVY
                    ax.text(j, i, int(val), ha="center", va="center",
                            color=text_color, fontsize=9, fontweight="500")

        plt.tight_layout()
        plt.savefig(fig_dir / f"consensus_heatmap_{resolution}.png", dpi=300,
                    bbox_inches="tight", facecolor="white")
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

    max_score = strong["total_conditional_score"].max() or 1
    bar_colors = [
        BLUE_CMAP(v / max_score) for v in strong["total_conditional_score"]
    ]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")

    ax.barh(strong["edge_label"], strong["total_conditional_score"],
            color=bar_colors, linewidth=0)

    ax.set_xlabel("Total Consensus Score", fontsize=11, color="#555", labelpad=8)
    ax.set_title(
        f"Top {top_n} Strongest Consensus Edges",
        fontsize=14, fontweight="bold", color="#111", pad=14, loc="left"
    )
    ax.xaxis.grid(True, color="#E5E5E5", linewidth=0.8, zorder=0)
    ax.yaxis.grid(False)
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["left"].set_linewidth(0.8)
    ax.tick_params(axis="both", length=0, pad=6)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    ax.tick_params(axis="y", labelsize=10, labelcolor="#222")
    ax.tick_params(axis="x", labelsize=10, labelcolor="#555")

    plt.tight_layout()
    plt.savefig(fig_dir / "top_strongest_consensus_edges.png", dpi=300,
                bbox_inches="tight", facecolor="white")
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

    max_score = leader_scores["total_outgoing_score"].max() or 1
    bar_colors = [
        BLUE_CMAP(v / max_score) for v in leader_scores["total_outgoing_score"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    ax.barh(leader_scores["sector_label"], leader_scores["total_outgoing_score"],
            color=bar_colors, linewidth=0)

    ax.set_xlabel("Total Outgoing Consensus Score", fontsize=11, color="#555", labelpad=8)
    ax.set_title(
        "First-Mover Sector Ranking Across All Directed Edges",
        fontsize=14, fontweight="bold", color="#111", pad=14, loc="left"
    )
    ax.xaxis.grid(True, color="#E5E5E5", linewidth=0.8, zorder=0)
    ax.yaxis.grid(False)
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["left"].set_linewidth(0.8)
    ax.tick_params(axis="both", length=0, pad=6)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    ax.tick_params(axis="y", labelsize=10, labelcolor="#222")
    ax.tick_params(axis="x", labelsize=10, labelcolor="#555")

    plt.tight_layout()
    plt.savefig(fig_dir / "first_mover_sector_ranking.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close()


def plot_core_causal_score(sp500_rank, fig_dir):

    df = sp500_rank.copy()
    df["sector_label"] = df["cause"].map(label_sector)
    df["cvar_bar"]  = df["cvar_support"].clip(0, 1)
    df["pcmci_bar"] = df["pcmci_support"].clip(0, 1)
    df["core_causal_score"] = df["cvar_bar"] + df["pcmci_bar"]
    df = df.sort_values("core_causal_score", ascending=False).reset_index(drop=True)

    n = len(df)
    x = np.arange(n)
    stem_top = 2.35

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, (_, row) in enumerate(df.iterrows()):
        if row["core_causal_score"] == 0:
            ax.bar(i, 0.08, color=C_STEEL, width=0.55, linewidth=0, zorder=3)
        else:
            ax.bar(i, row["cvar_bar"], color=C_CVAR,
                   width=0.55, linewidth=0, zorder=3)
            ax.bar(i, row["pcmci_bar"], bottom=row["cvar_bar"],
                   color=C_PCMCI, width=0.55, linewidth=0, zorder=3)

    for i, (_, row) in enumerate(df.iterrows()):
        bar_top = max(row["core_causal_score"], 0.08)
        ax.plot([i, i], [bar_top, stem_top],
                color=C_STEM, lw=1.4, zorder=2, solid_capstyle="butt")
        ax.scatter(i, stem_top, color=C_STEM, s=38, zorder=4, clip_on=False)

    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 2.6)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["0", "1", "2"], fontsize=11, color="#555")
    ax.set_ylabel("Core causal score  (max 2)", fontsize=11,
                  color="#555", labelpad=10)

    ax.set_xticks(x)
    tick_labels = [
        f"{row['cause']}\n({row['sector_label']})"
        for _, row in df.iterrows()
    ]
    ax.set_xticklabels(tick_labels, fontsize=10, color="#222", ha="center")

    _style_ax(ax)

    ax.set_title(
        "Which sectors causally lead the S&P 500?",
        fontsize=14, fontweight="bold", color="#111", pad=18, loc="left"
    )
    ax.text(
        0, 1.01,
        "Core causal evidence only: Conditional VAR + PCMCI  ·  Daily lag-1  ·  XGBoost excluded",
        transform=ax.transAxes,
        fontsize=9, color="#888", va="bottom"
    )

    legend_handles = [
        mpatches.Patch(facecolor=C_CVAR,  label="Conditional VAR (Granger)"),
        mpatches.Patch(facecolor=C_PCMCI, label="PCMCI"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        fontsize=10,
        labelcolor="#333",
        handlelength=1.2,
        handleheight=1.0,
    )

    plt.tight_layout()
    plt.savefig(fig_dir / "core_causal_score.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.show()
    plt.close()


def make_all_plots(consensus, sp500_rank):
    fig_dir = setup_figures()

    plot_sp500_sector_ranking(sp500_rank, fig_dir)
    plot_sp500_method_support(sp500_rank, fig_dir)
    plot_consensus_heatmaps(consensus, fig_dir)
    plot_strongest_edges(consensus, fig_dir, top_n=15)
    plot_first_mover_sector_scores(consensus, fig_dir)
    plot_core_causal_score(sp500_rank, fig_dir)

    print(f"Saved figures to: {fig_dir.resolve()}")


def main():

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

main()
