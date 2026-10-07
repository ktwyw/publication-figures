"""Fig. 132 - UPGMA tree aligned with a trait heatmap (1.5 column, 120 mm).

A tree beside a heatmap is only an argument if row k of the matrix is
tip k of the tree. Traits evolve by Brownian motion along a random
coalescent tree; the plotted tree is rebuilt from the standardised
traits by UPGMA (average linkage on Euclidean distance). scipy supplies
the leaf order, but the branches are drawn here from the linkage matrix,
so each one can take the colour of the clade it belongs to (four clades
from cutting the tree) and the tips can be labelled in a straight
column. The self-check: the tip labels read back from the figure equal
the dendrogram leaf order and each tip sits at the centre of its heatmap
row, the tree is ultrametric (every root-to-tip sum of the drawn branch
lengths is equal within 1e-9), and the cophenetic correlation exceeds
0.8.

Statistics: n = 24 taxa x 8 traits; colour, trait z-score across taxa
(sample s.d.); branch length, average-linkage Euclidean distance in
z-score units; clades, four-cluster cut. All data are simulated.
"""

import numpy as np
from matplotlib.lines import Line2D
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(1322)
N_TAXA, N_TRAITS, N_CLADES, Z_LIMIT = 24, 8, 4, 2.5
CLADE_COLOURS = [ms.BLUE, ms.ORANGE, ms.GREEN, ms.PINK]
taxa = [f"Taxon {k}" for k in range(1, N_TAXA + 1)]

# ------------------------------------------------------------- DATA ----
# True tree: a random coalescent (nodes 0-23 are tips, the last node is
# the root). Traits: Brownian motion from the root, variance = branch length.
N_NODES = 2 * N_TAXA - 1
age, parent = np.zeros(N_NODES), np.full(N_NODES, -1)
active, now = list(range(N_TAXA)), 0.0
for node in range(N_TAXA, N_NODES):
    k = len(active)
    now += rng.exponential(2.0 / (k * (k - 1)))
    pair = rng.choice(k, 2, replace=False)
    parent[[active[q] for q in pair]] = node
    age[node] = now
    active = [n for q, n in enumerate(active) if q not in pair] + [node]
value = np.zeros((N_NODES, N_TRAITS))
for node in range(N_NODES - 2, -1, -1):              # parents come first
    branch = age[parent[node]] - age[node]
    value[node] = value[parent[node]] + rng.normal(0, np.sqrt(branch),
                                                   N_TRAITS)
traits = value[:N_TAXA]

# ------------------------------------------------------------- TREE ----
z = (traits - traits.mean(axis=0)) / traits.std(axis=0, ddof=1)
distance = pdist(z)
link = hierarchy.linkage(distance, method="average")            # UPGMA
leaves = hierarchy.dendrogram(link, no_plot=True)["leaves"]
cophenetic = hierarchy.cophenet(link, distance)[0]
cut = hierarchy.fcluster(link, N_CLADES, criterion="maxclust")
first_seen = list(dict.fromkeys(cut[leaves]))        # numbered top-down
clade = np.array([first_seen.index(c) for c in cut])

# node coordinates: tips on their rows, a parent midway between its children
row, height, members = np.zeros(N_NODES), np.zeros(N_NODES), {}
row[leaves] = np.arange(N_TAXA)
for tip in range(N_TAXA):
    members[tip] = [tip]
for k, (a, b, h, _size) in enumerate(link):
    a, b, node = int(a), int(b), N_TAXA + k
    row[node], height[node] = (row[a] + row[b]) / 2, h
    members[node] = members[a] + members[b]


def colour_of(node):
    """Clade colour if the whole subtree is one clade, else neutral."""
    inside = set(clade[members[node]])
    return CLADE_COLOURS[inside.pop()] if len(inside) == 1 else ms.GREY_DARK


# ------------------------------------------------------------ FIGURE --
ROW_MM, COL_MM = 3.0, 4.4
LEFT, BOTTOM, TREE_W, NAME_W = 5.0, 9.0, 40.0, 14.5
HEAT_X = LEFT + TREE_W + NAME_W
HEAT_W, HEAT_H = N_TRAITS * COL_MM, N_TAXA * ROW_MM
SIDE = HEAT_X + HEAT_W + 5.0                         # colour bar, clade key
fig = ms.figure(120, BOTTOM + HEAT_H + 9.0)
ax_tree = ms.axes(fig, LEFT, BOTTOM, TREE_W, HEAT_H)
ax_heat = ms.axes(fig, HEAT_X, BOTTOM, HEAT_W, HEAT_H)
ax_cbar = ms.axes(fig, SIDE, BOTTOM + HEAT_H - 24.0, 2.2, 22.0)

# a, the tree: x is minus the node height, so all tips end at x = 0
root_to_tip = np.zeros(N_TAXA)       # drawn branch lengths, summed per tip
for k, (a, b, h, _size) in enumerate(link):
    a, b, node = int(a), int(b), N_TAXA + k
    ax_tree.plot([-h, -h], [row[a], row[b]], color=colour_of(node), lw=0.9,
                 solid_capstyle="projecting")
    for child in (a, b):
        ax_tree.plot([-h, -height[child]], [row[child]] * 2,
                     color=colour_of(child), lw=0.9, solid_capstyle="butt")
        root_to_tip[members[child]] += h - height[child]
root = height.max()
tips = [ax_tree.text(0.03 * root, row[tip], taxa[tip], va="center",
                     fontsize=ms.FS_SMALL) for tip in leaves]
ax_tree.set_xlim(-1.03 * root, 0)
ax_tree.set_ylim(N_TAXA - 0.5, -0.5)
ax_tree.set_axis_off()

# branch-length scale bar: the largest of 1, 2, 5 x 10^k within a third
# of the tree depth, drawn in the tree's own x units
steps = np.array([1, 2, 5]) * 10.0 ** np.floor(np.log10(root / 3))
scale = steps[steps <= root / 3].max()
y_bar = N_TAXA + 0.4
ax_tree.plot([-root, -root + scale], [y_bar, y_bar], color=ms.INK, lw=1.0,
             solid_capstyle="butt", clip_on=False)
ax_tree.text(-root, y_bar + 0.45, f"Branch length {scale:g}", va="top",
             fontsize=ms.FS_TICK)

# b, the heatmap, rows in the order of the tips
image = ax_heat.imshow(z[leaves], cmap="RdBu_r", vmin=-Z_LIMIT, vmax=Z_LIMIT,
                       aspect="auto")
ax_heat.set_xticks(range(N_TRAITS), range(1, N_TRAITS + 1))
ax_heat.set_yticks([])
ax_heat.xaxis.tick_top()
ax_heat.xaxis.set_label_position("top")
ax_heat.set_xlabel("Trait")
ax_heat.tick_params(length=0, pad=2)
for spine in ax_heat.spines.values():
    spine.set_visible(False)
cbar = fig.colorbar(image, cax=ax_cbar, ticks=[-2, -1, 0, 1, 2],
                    extend="both")
cbar.ax.yaxis.set_major_formatter(
    lambda v, _pos: f"{v:g}".replace("-", "−"))
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(length=2, width=0.5)
cbar.ax.set_title("z-score", fontsize=ms.FS_TICK, pad=5, loc="left")
fig.legend(handles=[Line2D([], [], color=colour, lw=1.4,
                           label=f"Clade {k + 1}")
                    for k, colour in enumerate(CLADE_COLOURS)],
           loc="upper left", handlelength=1.1,
           bbox_to_anchor=(SIDE / 120, (BOTTOM + HEAT_H - 34.0)
                           / (BOTTOM + HEAT_H + 9.0)), borderaxespad=0)

# ------------------------------------------------------- SELF-CHECK ---
# (after drawing: labels and positions are read back from the figure)
assert [t.get_text() for t in tips] == [taxa[k] for k in leaves]
assert leaves == list(hierarchy.leaves_list(link))
fig.canvas.draw()
for r, tip in enumerate(tips):
    y_tip = ax_tree.transData.transform((0, tip.get_position()[1]))[1]
    y_row = ax_heat.transData.transform((0, r))[1]
    assert abs(y_tip - y_row) < 0.05, (r, y_tip, y_row)
assert np.ptp(root_to_tip) < 1e-9 and np.isclose(root_to_tip[0], root)
assert np.all(height[N_TAXA:] >= height[link[:, :2].astype(int)].max(axis=1))
assert cophenetic > 0.8, cophenetic
print(f"fig132: self-check passed (heatmap rows = {N_TAXA} tips in "
      f"dendrogram leaf order; root-to-tip {root:.3f} for every tip, "
      f"spread {np.ptp(root_to_tip):.1e}; cophenetic r = {cophenetic:.3f}; "
      f"clade sizes {np.bincount(clade).tolist()})")

ms.panel_label(ax_tree, "a", dx_pt=-8, dy_pt=12)
ms.panel_label(ax_heat, "b", dx_pt=-10, dy_pt=12)
ms.assert_aligned([ax_tree, ax_heat])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig132_tree_trait_heatmap.{ext}")
print("fig132_tree_trait_heatmap: saved png + pdf")
