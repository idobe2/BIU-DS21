import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import pandas as pd

# ---------- Step 1: Creating a network graph ----------

# Creating a directed graph
G = nx.DiGraph()

# Adding edges
edges = [
    # "Normal" part of the network
    ("A", "B"),
    ("A", "C"),
    ("B", "C"),
    ("B", "D"),
    ("C", "A"),
    ("C", "D"),
    # Link into the trap
    ("C", "E"),
    ("D", "C"),
    ("D", "H"),
    # Link into the trap + dead-end
    ("D", "E"),
    # Spider trap: E,F,G connected only to each other (no outward edges)
    ("E", "F"),
    ("E", "G"),
    ("F", "E"),
    ("F", "G"),
    ("G", "E"),
    ("G", "F"),
]
G.add_edges_from(edges)

# Short tests to make sure we met the requirements
dead_ends = [n for n in G.nodes() if G.out_degree(n) == 0]
print("Dead-ends (out_degree==0):", dead_ends)

trap_set = {"E", "F", "G"}
is_trap = all(succ in trap_set for n in trap_set for succ in G.successors(n))
print("Is {E,F,G} a spider trap?", is_trap)

# Drawing the graph
plt.figure(figsize=(9, 6))
pos = nx.kamada_kawai_layout(G)
nx.draw_networkx(
    G, pos=pos, with_labels=True, arrows=True, node_size=1200, arrowsize=15
)
plt.axis("off")
plt.show()

# ---------- Step 2: Manual implementation of PageRank ----------


def build_transition_matrix(G, nodes=None):
    if nodes is None:
        nodes = list(G.nodes())
    n = len(nodes)
    idx = {node: k for k, node in enumerate(nodes)}

    M = np.zeros((n, n), dtype=float)

    for j, src in enumerate(nodes):
        out_neighbors = list(G.successors(src))
        if len(out_neighbors) == 0:
            # Dangling node: Distribute evenly to all nodes
            M[:, j] = 1.0 / n
        else:
            p = 1.0 / len(out_neighbors)
            for dst in out_neighbors:
                i = idx[dst]
                M[i, j] += p

    return M, nodes


M, nodes = build_transition_matrix(G)
print("Column sums:", M.sum(axis=0))


def pagerank_power_iteration(M, d=0.85, tol=1e-12, max_iter=10_000):
    n = M.shape[0]
    r = np.ones(n) / n
    teleport = np.ones(n) / n

    for it in range(max_iter):
        r_next = d * (M @ r) + (1 - d) * teleport
        # Convergence index
        if np.linalg.norm(r_next - r, ord=1) < tol:
            return r_next
        r = r_next

    raise RuntimeError("PageRank did not converge within max_iter")


rank = pagerank_power_iteration(M, d=0.85)
print("sum(rank) =", rank.sum())
print("min(rank), max(rank) =", rank.min(), rank.max())

# ---------- Step 3: Comparing results for different values ​​of the attenuation factor ----------

ds = [0.95, 0.85, 0.50]

ranks = {}
for d in ds:
    ranks[d] = pagerank_power_iteration(M, d=d)

for d in ds:
    print(f"\n=== d={d} ===")
    for node, val in sorted(zip(nodes, ranks[d]), key=lambda x: x[1], reverse=True):
        print(f"{node}: {val:.6f}")

for d in ds:
    print(f"sum(rank) for d={d} -> {ranks[d].sum()}")


def pagerank_residual(M, r, d):
    n = M.shape[0]
    teleport = np.ones(n) / n
    r_check = d * (M @ r) + (1 - d) * teleport
    return np.linalg.norm(r - r_check, ord=1), np.linalg.norm(r - r_check, ord=2)


for d in ds:
    res1, res2 = pagerank_residual(M, ranks[d], d)
    print(f"d={d}: residual L1={res1:.3e}, residual L2={res2:.3e}")

# ---------- Step 4: Calculating distances between results with L1 and L2 norms ----------

pairs = [(0.95, 0.85), (0.85, 0.50)]  # Two sample pairs

for d1, d2 in pairs:
    diff = ranks[d1] - ranks[d2]
    l1 = norm(diff, ord=1)
    l2 = norm(diff, ord=2)
    print(f"Distance between d={d1} and d={d2}:  L1={l1:.6f},  L2={l2:.6f}")

# ---------- Step 5: Table + Graph generation ----------

df = pd.DataFrame(
    {
        "node": nodes,
        "rank_d0.95": ranks[0.95],
        "rank_d0.85": ranks[0.85],
        "rank_d0.50": ranks[0.50],
    }
).set_index("node")

print(df.sort_values("rank_d0.85", ascending=False))

# Distance table
pairs = [(0.95, 0.85), (0.85, 0.50)]
dist_rows = []
for d1, d2 in pairs:
    diff = ranks[d1] - ranks[d2]
    dist_rows.append(
        {
            "pair": f"{d1} vs {d2}",
            "L1": norm(diff, 1),
            "L2": norm(diff, 2),
        }
    )
dist_df = pd.DataFrame(dist_rows)
print(dist_df)

# Chart graph for comparing rankings between d values ​​(for each node)
ax = df[["rank_d0.95", "rank_d0.85", "rank_d0.50"]].plot(kind="bar", figsize=(10, 5))
ax.set_ylabel("PageRank")
ax.set_title("PageRank comparison for different damping factors")
plt.tight_layout()
plt.show()
