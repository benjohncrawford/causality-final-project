"""
Module 5: PC Algorithm
=======================
Runs the PC algorithm on all variables to discover the causal DAG structure.

Steps:
  1. Load normalized model data
  2. Run PC algorithm (causal-learn)
  3. Visualize the causal graph
  4. Print adjacency matrix

Input:  data/model_data.csv
Output: figures/pc_graph.png
        results/pc_adjacency.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import fisherz

INPUT_FILE  = "data/model_data.csv"
FIGURES_DIR = "figures"
RESULTS_DIR = "results"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

LABELS = {
    "SP500_ret"  : "SP500\nReturn",
    "VIX"        : "VIX",
    "FEDFUNDS_d" : "FedFunds\n",
    "CPI_yoy"    : "CPI\nYoY",
    "UNRATE_d"   : "Unemploy\n",
    "M2_yoy"     : "M2\nGrowth",
    "SENTIMENT_d": "Sentiment\n",
    "OIL_ret"    : "Oil\nReturn",
    "GS10_d"     : "10Y Treas\n",
}
def save_graph(pretty_names, G, filename):
    print("\nGenerating graph visualization...")

    import networkx as nx
    print(G)
    # Build directed graph from adjacency matrix
    G_nx = nx.DiGraph()
    G_nx.add_nodes_from(pretty_names)

    for i, src in enumerate(pretty_names):
        for j, tgt in enumerate(pretty_names):
            if i >= j:
                continue
            if G[j, i] == 1 and G[i, j] == -1:      # i → j
                G_nx.add_edge(src, tgt, edge_type="directed")
            elif G[i, j] == 1 and G[j, i] == -1:    # j → i
                G_nx.add_edge(tgt, src, edge_type="directed")
            elif G[i, j] == -1 and G[j, i] == -1:     # undirected
                G_nx.add_edge(src, tgt, edge_type="undirected")
                G_nx.add_edge(tgt, src, edge_type="undirected")
    # Use graphviz hierarchical layout if available, else spring
    try:
        pos = nx.nx_agraph.graphviz_layout(G_nx, prog="dot")
        print("  Using graphviz 'dot' layout")
    except Exception:
        pos = nx.spring_layout(G_nx, seed=42, k=2.5)
        print("  Using spring layout (graphviz not available)")

    # Separate isolated nodes
    isolated = [n for n in pretty_names if G_nx.degree(n) == 0]
    connected = [n for n in pretty_names if G_nx.degree(n) > 0]

    fig, ax = plt.subplots(figsize=(13, 10))

    # Draw connected nodes
    nx.draw_networkx_nodes(G_nx, pos, nodelist=connected, ax=ax,
                        node_size=2500, node_color="#dbeafe", edgecolors="#1e40af", linewidths=1.5)
    # Draw isolated nodes
    if isolated:
        nx.draw_networkx_nodes(G_nx, pos, nodelist=isolated, ax=ax,
                            node_size=2500, node_color="#f1f5f9", edgecolors="#94a3b8",
                            linewidths=1.5)

    # Directed edges
    directed_edges = [
        (u, v) for u, v, d in G_nx.edges(data=True) if d.get("edge_type") == "directed"
    ]
    undirected_edges = [
        (u, v) for u, v, d in G_nx.edges(data=True) if d.get("edge_type") == "undirected"
    ]
    nx.draw_networkx_edges(G_nx, pos, ax=ax, edgelist=directed_edges,
                        edge_color="#1e40af", width=2.0,
                        arrowsize=25, arrowstyle="-|>",
                        connectionstyle="arc3,rad=0.05",
                        min_source_margin=30, min_target_margin=30)
    nx.draw_networkx_edges(G_nx, pos, ax=ax, edgelist=undirected_edges,
                        edge_color="#1e40af", width=2.0, arrows=False,
                        connectionstyle="arc3,rad=0.05",
                        min_source_margin=30, min_target_margin=30)
    nx.draw_networkx_labels(G_nx, pos, ax=ax, font_size=9, font_weight="bold")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#1e40af", linewidth=2, label="Directed causal edge"),
        plt.scatter([], [], s=100, facecolors="#dbeafe", edgecolors="#1e40af", linewidths=1.5, label="Connected variable"),
        plt.scatter([], [], s=100, facecolors="#f1f5f9", edgecolors="#94a3b8", linewidths=1.5, label="Isolated variable"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9)
    ax.set_title("PC Algorithm — Causal Graph (α = 0.05)", fontsize=14, fontweight="bold", pad=20)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {FIGURES_DIR}/{filename}")


# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE, index_col="Date", parse_dates=True).dropna()
node_names   = df.columns.tolist()
pretty_names = [LABELS[n] for n in node_names]
data         = df.to_numpy()

print(f"Data shape: {data.shape}  ({data.shape[0]} months × {data.shape[1]} variables)")
print(f"Variables: {node_names}\n")


# ── 2. Run PC algorithm ───────────────────────────────────────────────────────
print("Running PC algorithm (alpha=0.05)...")
cg = pc(
    data,
    alpha      = 0.05,       # significance level for conditional independence tests
    indep_test = fisherz,    # Fisher Z test (appropriate for continuous data)
    node_names = pretty_names,
    show_progress = True,
)
print("PC algorithm complete.\n")
pyd = GraphUtils.to_pydot(cg.G)
pyd.write_png(f'figures/pc_graph.png')

regimes = [("2000-01-01", "2007-01-01"), ("2008-01-01", "2019-01-01"), ("2020-01-01", "2024-01-01"),]
for regime in regimes:
    df_regime = df.loc[regime[0]:regime[1]]
    data_regime         = df_regime.to_numpy()
    print("Running PC algorithm (alpha=0.05)...")
    cg = pc(
        data_regime,
        alpha      = 0.05,       # significance level for conditional independence tests
        indep_test = fisherz,    # Fisher Z test (appropriate for continuous data)
        node_names = pretty_names,
        show_progress = True,
    )
    print("PC algorithm complete.\n")
    pyd = GraphUtils.to_pydot(cg.G)
    pyd.write_png(f'figures/pc_graph_{regime[0]}_{regime[1]}.png')
    save_graph(pretty_names, cg.G.graph, f"pc_graph_{regime[0]}_{regime[1]}_pretty.png")

# ── 2. Run FCI algorithm ───────────────────────────────────────────────────────
print("Running FCI algorithm (alpha=0.05)...")
cg_fci, edges = fci(
    data,
    alpha      = 0.05,       # significance level for conditional independence tests
    indep_test = fisherz,    # Fisher Z test (appropriate for continuous data)
    node_names = pretty_names,
    show_progress = True,
)
print("FCI algorithm complete.\n")
pyd = GraphUtils.to_pydot(cg_fci)
pyd.write_png('figures/fci_graph.png')

# # ── 2. Run GES algorithm ───────────────────────────────────────────────────────
# print("Running GES algorithm...")
# parameters = {}
# parameters["kfold"] = 2
# parameters["lambda"] = 0.2
# record = ges(data, parameters = parameters, maxP = 4, score_func="local_score_CV_general", node_names = pretty_names)
# print(record)
# print("GES algorithm complete.\n")

# pyd = GraphUtils.to_pydot(record['G'])
# pyd.write_png('figures/ges_graph.png')
# ── 3. Adjacency matrix ───────────────────────────────────────────────────────
# causal-learn stores edges in cg.G.graph
# graph[i,j]=1 & graph[j,i]=-1 means i→j
# graph[i,j]=1 & graph[j,i]=1  means i—j (undirected)
G = cg.G.graph
adj_df = pd.DataFrame(G, index=pretty_names, columns=pretty_names)
adj_df.to_csv(f"{RESULTS_DIR}/pc_adjacency.csv")
print("Adjacency matrix (1=arrowhead, -1=tail, 0=no edge):")
print(adj_df)
print(f"\nSaved → {RESULTS_DIR}/pc_adjacency.csv")


# ── 4. Visualize ──────────────────────────────────────────────────────────────
save_graph(pretty_names, G, "pc_graph_pretty.png")


# ── 5. Print discovered edges in plain English ────────────────────────────────
print("\n── Discovered edges ──────────────────────────────────────")
edge_count = 0
for i, src in enumerate(pretty_names):
    for j, tgt in enumerate(pretty_names):
        if i >= j:
            continue
        if G[j, i] == 1 and G[i, j] == -1:
            print(f"  {src.replace(chr(10),' ')} → {tgt.replace(chr(10),' ')}")
            edge_count += 1
        elif G[i, j] == 1 and G[j, i] == -1:
            print(f"  {tgt.replace(chr(10),' ')} → {src.replace(chr(10),' ')}")
            edge_count += 1
        elif G[i, j] == 1 and G[j, i] == 1:
            print(f"  {src.replace(chr(10),' ')} — {tgt.replace(chr(10),' ')}  (undirected)")
            edge_count += 1

if edge_count == 0:
    print("  No significant edges found. Try increasing alpha (e.g. 0.10).")

print(f"\nTotal edges: {edge_count}")
print("PC algorithm analysis complete.")
