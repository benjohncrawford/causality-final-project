"""
Module 6: CD-NOD (Nonstationary Causal Discovery)
===================================================
Uses CD-NOD to detect whether causal structures change across economic regimes.

Regime definitions (c_indx):
  0 → 2000–2007  Pre-GFC
  1 → 2008–2019  Post-GFC / ZIRP era
  2 → 2020–2024  Post-COVID / rate hike cycle

Steps:
  1. Load model data, attach regime labels
  2. Run CD-NOD
  3. Compare resulting graph vs PC algorithm graph
  4. Visualize

Input:  data/model_data.csv
Output: figures/cdnod_graph.png
        figures/cdnod_vs_pc.png
        results/cdnod_adjacency.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.GraphUtils import GraphUtils

INPUT_FILE  = "data/model_data.csv"
FIGURES_DIR = "figures"
RESULTS_DIR = "results"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

LABELS = {
    "SP500_ret"  : "SP500\nReturn",
    "VIX"        : "VIX",
    "FEDFUNDS_d" : "FedFunds\nΔ",
    "CPI_yoy"    : "CPI\nYoY",
    "UNRATE_d"   : "Unemploy\nΔ",
    "M2_yoy"     : "M2\nGrowth",
    "SENTIMENT_d": "Sentiment\nΔ",
    "OIL_ret"    : "Oil\nReturn",
    "GS10_d"     : "10Y Treas\nΔ",
}

REGIME_NAMES = {0: "Pre-GFC (2000–07)", 1: "ZIRP Era (2008–19)", 2: "Post-COVID (2020–24)"}


# ── 1. Load data & assign regimes ─────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE, index_col="Date", parse_dates=True).dropna()

def assign_regime(date):
    if date < pd.Timestamp("2008-01-01"):
        return 0
    elif date < pd.Timestamp("2020-01-01"):
        return 1
    else:
        return 2

regime_series = df.index.map(assign_regime)

# Use continuous time index so CD-NOD can detect finer-grained structural
# changes rather than only comparing three discrete blocks.
c_indx = np.arange(len(df)).reshape(-1, 1).astype(float)

node_names   = df.columns.tolist()
pretty_names = [LABELS[n] for n in node_names]
data         = df.to_numpy()

print(f"Data shape: {data.shape}")
print("Regime reference counts (for interpretation):")
for r, name in REGIME_NAMES.items():
    print(f"  {name}: {(regime_series == r).sum()} months")
print("c_indx: continuous time index (0 to 287)")


# ── 2. Run CD-NOD ─────────────────────────────────────────────────────────────
print("\nRunning CD-NOD (alpha=0.05)...")
cg_cdnod = cdnod(
    data,
    c_indx,
    alpha         = 0.05,
    indep_test    = "fisherz",
    stable        = True,
    uc_rule       = 0,
    uc_priority   = 2,
    node_names    = pretty_names,
    show_progress = True,
)
print("CD-NOD complete.\n")


# ── 3. Save adjacency matrix ──────────────────────────────────────────────────
G_cdnod = cg_cdnod.G.graph
pyd = GraphUtils.to_pydot(cg_cdnod.G)
pyd.write_png(f'figures/cdnod_graph.png')
print(G_cdnod)
# CD-NOD appends a "c_indx" node — exclude it from the adjacency output
n_vars   = len(pretty_names)
G_vars   = G_cdnod[:n_vars, :n_vars]

adj_df = pd.DataFrame(G_vars, index=pretty_names, columns=pretty_names)
adj_df.to_csv(f"{RESULTS_DIR}/cdnod_adjacency.csv")
print("CD-NOD adjacency matrix saved.")


# ── 4. Identify regime-sensitive variables (needed before visualization) ──────
regime_sensitive = []
if G_cdnod.shape[0] > n_vars:
    c_row = G_cdnod[n_vars, :n_vars]
    c_col = G_cdnod[:n_vars, n_vars]
    for j in range(n_vars):
        if c_col[j] == 1 and c_row[j] == -1:
            regime_sensitive.append(pretty_names[j])


# ── 5. Visualize CD-NOD graph ─────────────────────────────────────────────────
print("Generating CD-NOD graph...")
import networkx as nx
from matplotlib.lines import Line2D

G_nx = nx.DiGraph()
G_nx.add_nodes_from(pretty_names)

for i, src in enumerate(pretty_names):
    for j, tgt in enumerate(pretty_names):
        if i >= j:
            continue
        if G_cdnod[j, i] == 1 and G_cdnod[i, j] == -1:      # i → j
            G_nx.add_edge(src, tgt, edge_type="directed")
        elif G_cdnod[i, j] == 1 and G_cdnod[j, i] == -1:    # j → i
            G_nx.add_edge(tgt, src, edge_type="directed")
        elif G_cdnod[i, j] == -1 and G_cdnod[j, i] == -1:     # undirected
            G_nx.add_edge(src, tgt, edge_type="undirected")
            G_nx.add_edge(tgt, src, edge_type="undirected")

# Color regime-sensitive nodes differently
node_colors = []
for n in pretty_names:
    if n in regime_sensitive:
        node_colors.append("#fde68a")   # yellow = regime-sensitive
    elif G_nx.degree(n) == 0:
        node_colors.append("#f1f5f9")   # grey = isolated
    else:
        node_colors.append("#fef9c3")   # cream = normal

edge_colors = []
for u, v in G_nx.edges():
    if u in regime_sensitive or v in regime_sensitive:
        edge_colors.append("#b45309")   # darker for regime-sensitive edges
    else:
        edge_colors.append("#92400e")

try:
    pos = nx.nx_agraph.graphviz_layout(G_nx, prog="dot")
    print("  Using graphviz 'dot' layout")
except Exception:
    pos = nx.spring_layout(G_nx, seed=42, k=2.5)
    print("  Using spring layout")

# Directed edges
directed_edges = [
    (u, v) for u, v, d in G_nx.edges(data=True) if d.get("edge_type") == "directed"
]
undirected_edges = [
    (u, v) for u, v, d in G_nx.edges(data=True) if d.get("edge_type") == "undirected"
]

fig, ax = plt.subplots(figsize=(13, 10))
nx.draw_networkx_nodes(G_nx, pos, ax=ax,
                       node_size=2500, node_color=node_colors,
                       edgecolors="#92400e", linewidths=1.5)
nx.draw_networkx_edges(G_nx, pos, ax=ax, edgelist=undirected_edges,
                       edge_color=edge_colors, width=2.0, arrows=False,
                       connectionstyle="arc3,rad=0.05",
                       min_source_margin=30, min_target_margin=30)
nx.draw_networkx_edges(G_nx, pos, ax=ax, edgelist=directed_edges,
                       edge_color=edge_colors, width=2.0,
                       arrowsize=25, arrowstyle="-|>",
                       connectionstyle="arc3,rad=0.05",
                       min_source_margin=30, min_target_margin=30)
nx.draw_networkx_labels(G_nx, pos, ax=ax, font_size=9, font_weight="bold")

legend_elements = [
    Line2D([0], [0], color="#92400e", linewidth=2, label="Causal edge"),
    mpatches.Patch(facecolor="#fde68a", edgecolor="#92400e", label="Regime-sensitive variable"),
    mpatches.Patch(facecolor="#fef9c3", edgecolor="#92400e", label="Stable variable"),
    mpatches.Patch(facecolor="#f1f5f9", edgecolor="#94a3b8", label="Isolated variable"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=9)
ax.set_title("CD-NOD — Nonstationary Causal Graph (α = 0.05)\n"
             f"Regimes: {', '.join(REGIME_NAMES.values())}", fontsize=13, fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/cdnod_graph_pretty.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {FIGURES_DIR}/cdnod_graph_pretty.png")


# ── 6. Print edges & highlight regime-sensitive ones ─────────────────────────
print("\n── CD-NOD discovered edges ───────────────────────────────")

edge_count = 0
for i in range(n_vars):
    for j in range(n_vars):
        if i >= j:
            continue
        src = pretty_names[i].replace("\n", " ")
        tgt = pretty_names[j].replace("\n", " ")
        if G_vars[j, i] == 1 and G_vars[i, j] == -1:
            print(f"  {src} → {tgt}")
            edge_count += 1
        elif G_vars[i, j] == 1 and G_vars[j, i] == -1:
            print(f"  {tgt} → {src}")
            edge_count += 1
        elif G_vars[i, j] == 1 and G_vars[j, i] == 1:
            print(f"  {src} — {tgt}  (undirected)")
            edge_count += 1

print(f"\nTotal edges: {edge_count}")

if regime_sensitive:
    print(f"\n⚠️  Regime-sensitive variables (causal mechanism changed across periods):")
    for v in regime_sensitive:
        print(f"   {v.replace(chr(10), ' ')}")
else:
    print("\nNo regime-sensitive variables detected at α=0.05.")

print("\nCD-NOD analysis complete.")
