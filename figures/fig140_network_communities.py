"""Fig. 140 - Network with detected communities and hubs (1.5 column, 120 mm).

A node-link diagram in which nothing is placed by hand: the graph is a
degree-corrected three-block stochastic block model, the layout is the
Fruchterman-Reingold spring embedder coded in numpy (seeded start, fixed
iterations, linear cooling, then rotated onto its principal axes), and
the colours are communities found by spectral clustering on the two
leading eigenvectors of the modularity matrix with a deterministic
k-means. Node area encodes degree, within-community edges are darker
than the bridges between communities, and the three highest-degree hubs
are labelled outside the hull of the drawing, where no edge can run
under the type. The self-check is that the adjacency matrix is symmetric
with an empty diagonal, that modularity Q computed from its definition
exceeds 0.3, and that the detected communities reproduce the planted
blocks (agreement >= 0.95 after label matching).

Data: one undirected graph, n = 60 nodes in three planted blocks of 20;
no hypothesis test. All data are simulated.
"""

from itertools import permutations

import numpy as np
from matplotlib.collections import LineCollection

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(140)

# ------------------------------------------------------------- DATA ----
# Degree-corrected stochastic block model: P(edge ij) = w_i w_j P[b_i, b_j]
BLOCK_SIZE, K = 20, 3
P_IN, P_OUT = 0.30, 0.02
block = np.repeat(np.arange(K), BLOCK_SIZE)
n = block.size
weight = rng.lognormal(0.0, 0.35, n)             # a few sociable nodes
prob = np.where(block[:, None] == block[None, :], P_IN, P_OUT)
prob = np.clip(prob * np.outer(weight, weight), 0, 1)
upper = np.triu(rng.random((n, n)) < prob, k=1)
A = (upper | upper.T).astype(float)              # adjacency matrix
degree = A.sum(axis=1)
m = A.sum() / 2                                  # number of edges


# ---------------------------------------------------------- METHODS ----
def fruchterman_reingold(adjacency, iterations=300):
    """Spring layout: repulsion k^2/d between all pairs, attraction d^2/k
    along edges, steps capped by a temperature that cools linearly."""
    count = adjacency.shape[0]
    pos = rng.uniform(-1, 1, (count, 2))
    k = 2.0 / np.sqrt(count)                     # ideal edge length
    for temperature in np.linspace(0.2, 0.0, iterations, endpoint=False):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.maximum(np.hypot(delta[..., 0], delta[..., 1]), 1e-3)
        force = k ** 2 / dist ** 2 - adjacency * dist / k   # per unit delta
        move = (delta * force[..., None]).sum(axis=1)
        length = np.maximum(np.hypot(move[:, 0], move[:, 1]), 1e-9)
        step = np.minimum(length, temperature) / length
        pos += move * step[:, None]
    pos -= pos.mean(axis=0)
    _u, _s, axes = np.linalg.svd(pos, full_matrices=False)
    return pos @ axes.T                          # long axis horizontal


def kmeans(points, k, iterations=50):
    """Lloyd's algorithm from a farthest-first start (no random draws)."""
    centres = [points[np.argmax((points ** 2).sum(axis=1))]]
    for _ in range(k - 1):
        gap = np.min([((points - c) ** 2).sum(axis=1) for c in centres],
                     axis=0)
        centres.append(points[np.argmax(gap)])
    centres = np.array(centres)
    for _ in range(iterations):
        label = np.argmin(((points[:, None, :] - centres) ** 2).sum(axis=2),
                          axis=1)
        centres = np.array([points[label == c].mean(axis=0) for c in range(k)])
    return label


# Modularity matrix B = A - k k^T / 2m; its K - 1 leading eigenvectors
# separate K communities (Newman's spectral method).
B = A - np.outer(degree, degree) / (2 * m)
eigenvalue, eigenvector = np.linalg.eigh(B)
lead = slice(-(K - 1), None)
detected = kmeans(eigenvector[:, lead] * np.sqrt(eigenvalue[lead]), K)

# Cluster numbers are arbitrary: rename them to agree best with the blocks
best = max(permutations(range(K)),
           key=lambda perm: np.mean(np.array(perm)[detected] == block))
community = np.array(best)[detected]
agreement = float(np.mean(community == block))
same = community[:, None] == community[None, :]
Q = float((B * same).sum() / (2 * m))            # modularity, by definition

pos = fruchterman_reingold(A)
pos -= (pos.max(axis=0) + pos.min(axis=0)) / 2   # centre the bounding box
pos *= np.min(np.array([60.0, 56.0]) / np.ptp(pos, axis=0))   # fit, in mm
hubs = np.argsort(-degree, kind="stable")[:3]

# ------------------------------------------------------- SELF-CHECK ---
assert np.array_equal(A, A.T) and np.all(np.diag(A) == 0)
assert set(np.unique(A)) <= {0.0, 1.0} and degree.min() >= 1
assert Q > 0.3, Q
assert agreement >= 0.95, agreement
print(f"fig140: self-check passed (n = {n}, m = {m:.0f} edges; adjacency "
      f"symmetric, zero diagonal; Q = {Q:.3f} > 0.3; agreement with "
      f"planted blocks = {agreement:.3f})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 82)
ax = ms.axes(fig, 1, 2, 88, 78)          # network; data units are mm
ax.set_xlim(-44, 44)
ax.set_ylim(-39, 39)
ax.set_aspect("equal")
ax.axis("off")
key = ms.axes(fig, 91, 2, 28, 78)        # key column, also in mm
key.set_xlim(0, 28)
key.set_ylim(0, 78)
key.axis("off")

COLOURS = np.array([ms.BLUE, ms.ORANGE, ms.GREEN])
EDGE = {True: ("#8C8C8C", 0.45), False: (ms.GREY_LIGHT, 0.45)}   # within?


def node_area(k):
    """Marker area in pt^2: proportional to degree."""
    return 3.2 * np.asarray(k, float)


rows, cols = np.nonzero(np.triu(A))
within = community[rows] == community[cols]
for inside in (False, True):             # bridges underneath
    colour, width = EDGE[inside]
    pick = within == inside
    ax.add_collection(LineCollection(
        np.stack([pos[rows[pick]], pos[cols[pick]]], axis=1), colors=colour,
        linewidths=width, capstyle="round", zorder=1 + inside))
ax.scatter(pos[:, 0], pos[:, 1], s=node_area(degree), c=COLOURS[community],
           edgecolors="white", linewidths=0.5, zorder=4)

# Hub labels sit outside the convex hull of the drawing, where no edge can
# pass under them. Of 24 directions each hub takes the shortest way out
# (distance to the supporting line of all nodes) whose leader does not
# run through another node; the label box is then pushed along the ray
# until it clears that line (support distance of a w x h box).
angle = np.deg2rad(np.arange(0, 360, 15))
rays = np.column_stack([np.cos(angle), np.sin(angle)])
renderer = fig.canvas.get_renderer()
for hub in hubs:
    way_out = (pos @ rays.T).max(axis=0) - pos[hub] @ rays.T        # mm
    rel = np.delete(pos, hub, axis=0) - pos[hub]
    along = rel @ rays.T                                 # (node, ray)
    across = np.abs(rel @ (rays[:, ::-1] * [1, -1]).T)
    blocked = ((along > 0) & (across < 0.9)).any(axis=0)
    r = np.argmin(np.where(blocked, np.inf, way_out))
    ray, exit_point = rays[r], pos[hub] + way_out[r] * rays[r]
    ax.plot(*np.column_stack([pos[hub], exit_point + 1.2 * ray]),
            color=ms.INK, lw=0.5, zorder=3)
    ax.scatter(*pos[hub], s=node_area(degree[hub]), facecolors="none",
               edgecolors=ms.INK, linewidths=0.8, zorder=5)
    label = ax.text(0, 0, f"Node {hub + 1}\ndegree {degree[hub]:.0f}",
                    ha="center", va="center", fontsize=ms.FS_TICK,
                    linespacing=1.15)
    box = label.get_window_extent(renderer)
    size = np.array([box.width, box.height]) * 25.4 / fig.dpi        # mm
    label.set_position(exit_point
                       + ray * (2.2 + (size * np.abs(ray)).sum() / 2))

# key: communities, node size, edge classes, summary numbers
key.text(0, 74, "Detected community", fontsize=ms.FS_TICK, fontweight="bold")
for c in range(K):
    y = 69.5 - 4.2 * c
    key.scatter(2, y, s=node_area(8), color=COLOURS[c], linewidths=0)
    key.text(5, y, f"Community {c + 1}, n = {np.sum(community == c)}",
             fontsize=ms.FS_TICK, va="center")

key.text(0, 52, "Node area, degree", fontsize=ms.FS_TICK, fontweight="bold")
for col, k in enumerate((3, 8, 16)):
    key.scatter(3 + 9 * col, 46.5, s=node_area(k), color=ms.GREY,
                linewidths=0)
    key.text(3 + 9 * col, 42.6, str(k), fontsize=ms.FS_TICK, ha="center",
             va="center")

key.text(0, 35, "Edge", fontsize=ms.FS_TICK, fontweight="bold")
for row, inside in enumerate((True, False)):
    y = 30.5 - 4.2 * row
    colour, width = EDGE[inside]
    key.plot([0.5, 5.5], [y, y], color=colour, lw=width * 2)
    count = np.sum(within == inside)
    key.text(7.5, y, f"{'Within' if inside else 'Between'}  ({count})",
             fontsize=ms.FS_TICK, va="center")

key.text(0, 17.5, f"Modularity Q = {Q:.2f}\n"
         f"Planted blocks recovered:\n{np.sum(community == block)} of {n} "
         "nodes", fontsize=ms.FS_TICK, color=ms.GREY_DARK, va="top",
         linespacing=1.3)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig140_network_communities.{ext}")
print("fig140_network_communities: saved png + pdf")
