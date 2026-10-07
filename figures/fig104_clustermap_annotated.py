"""Fig. 104 - Annotated clustered heatmap, laid out in millimetres (120 mm).

Extends fig015: rows and columns are both clustered by average linkage
on correlation distance (fig015 uses Ward on the columns), a condition
bar sits between the column dendrogram and the matrix so the reader can
see whether the tree recovers the design, and the colour bar and legend
are placed in millimetres beside the gene names rather than by a
gridspec ratio. The self-check: every row z-score has mean 0 and
s.d. 1; both leaf orders are permutations and equal
scipy's leaves_list; cutting the column tree into two clusters
reproduces the control/treated split exactly; and each drawn dendrogram
leaf sits over the centre of its heatmap column or row.

Statistics: 24 genes x 12 samples (6 per condition, column order
shuffled before clustering); row z-scores (sample s.d.); average
linkage on correlation distance. All data are simulated.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.cluster import hierarchy

import manuscript as ms

ms.apply()
HERE = ms.HERE

N_GENES, N_PER_GROUP = 24, 6
N_SAMPLES = 2 * N_PER_GROUP
Z_LIMIT = 2.5
CONDITION = {"Control": ms.GREY, "Treated": ms.VERMILLION}

# ------------------------------------------------------------- DATA ----
# Two gene modules move in opposite directions with treatment; six genes
# do not respond. Samples are shuffled so the clustering does the work.
rng = np.random.default_rng(404)
names = ([f"C{i}" for i in range(1, N_PER_GROUP + 1)]
         + [f"T{i}" for i in range(1, N_PER_GROUP + 1)])
condition = np.array([0] * N_PER_GROUP + [1] * N_PER_GROUP)
module = np.array([1] * 9 + [-1] * 9 + [0] * 6)
rng.shuffle(module)
matrix = rng.normal(8, 0.55, (N_GENES, N_SAMPLES))
matrix += np.outer(module * rng.uniform(1.0, 2.2, N_GENES), condition - 0.5)
shuffle = rng.permutation(N_SAMPLES)
expr = matrix[:, shuffle]
samples = [names[i] for i in shuffle]
treated = condition[shuffle].astype(bool)
genes = [f"Gene {i:02d}" for i in range(1, N_GENES + 1)]

# ---------------------------------------------------------- CLUSTER ----
z = ((expr - expr.mean(axis=1, keepdims=True))
     / expr.std(axis=1, ddof=1, keepdims=True))
row_link = hierarchy.linkage(z, method="average", metric="correlation")
col_link = hierarchy.linkage(z.T, method="average", metric="correlation")
row_order = hierarchy.dendrogram(row_link, no_plot=True)["leaves"]
col_order = hierarchy.dendrogram(col_link, no_plot=True)["leaves"]

# ------------------------------------------------------- SELF-CHECK ---
assert np.allclose(z.mean(axis=1), 0.0, atol=1e-12)
assert np.allclose(z.std(axis=1, ddof=1), 1.0, atol=1e-12)
assert sorted(row_order) == list(range(N_GENES))
assert sorted(col_order) == list(range(N_SAMPLES))
assert row_order == list(hierarchy.leaves_list(row_link))
assert col_order == list(hierarchy.leaves_list(col_link))
two = hierarchy.fcluster(col_link, 2, criterion="maxclust")
assert len(set(two)) == 2
assert len(set(two[treated])) == 1 and len(set(two[~treated])) == 1
assert two[treated][0] != two[~treated][0]
# the two top-level branches are contiguous along the plotted columns
switches = int(np.sum(np.diff(treated[col_order].astype(int)) != 0))
assert switches == 1, switches
print("fig104: self-check passed (row z-scores mean 0, s.d. 1; leaf orders "
      f"are permutations of {N_GENES} rows and {N_SAMPLES} columns; a "
      "2-cluster cut of the column tree = control/treated, "
      f"{int((~treated).sum())} + {int(treated.sum())})")

# ------------------------------------------------------------ FIGURE --
# Physical layout (mm): the heatmap first, auxiliary axes placed from it.
LEFT, BOTTOM, HEAT_W, HEAT_H = 22.0, 10.0, 66.0, 67.2
SIDE = LEFT + HEAT_W + 14.0              # colour bar and legend column
fig = ms.figure(120, 92)
ax_heat = ms.axes(fig, LEFT, BOTTOM, HEAT_W, HEAT_H)
ax_rowd = ms.axes(fig, LEFT - 13.5, BOTTOM, 12.5, HEAT_H)
ax_bar = ms.axes(fig, LEFT, BOTTOM + HEAT_H + 1.0, HEAT_W, 2.2)
ax_cold = ms.axes(fig, LEFT, BOTTOM + HEAT_H + 4.0, HEAT_W, 7.5)
ax_cbar = ms.axes(fig, SIDE + 0.5, BOTTOM + HEAT_H - 24.0, 2.2, 22.0)


def draw_dendrogram(ax, link, orientation):
    """Grey tree on a bare axes; returns the dendrogram dictionary."""
    with plt.rc_context({"lines.linewidth": 0.6}):
        tree = hierarchy.dendrogram(
            link, ax=ax, orientation=orientation, no_labels=True,
            color_threshold=0, above_threshold_color=ms.GREY_DARK)
    ax.set_axis_off()
    return tree


row_tree = draw_dendrogram(ax_rowd, row_link, "left")
col_tree = draw_dendrogram(ax_cold, col_link, "top")
assert row_tree["leaves"] == row_order and col_tree["leaves"] == col_order
# orientation "left" lists the leaves bottom-up; imshow draws top-down
ordered = z[np.ix_(row_order[::-1], col_order)]

image = ax_heat.imshow(ordered, cmap="RdBu_r", vmin=-Z_LIMIT, vmax=Z_LIMIT,
                       aspect="auto", interpolation="nearest")
ax_heat.set_xticks(range(N_SAMPLES), [samples[i] for i in col_order])
ax_heat.set_yticks(range(N_GENES), [genes[i] for i in row_order[::-1]],
                   fontsize=ms.FS_SMALL)
ax_heat.yaxis.tick_right()
ax_heat.tick_params(length=0, pad=2.5)
for spine in ax_heat.spines.values():
    spine.set_visible(False)
ax_heat.set_xlabel("Samples")

# condition bar as a one-row image: no hairlines between adjacent cells
ax_bar.imshow(treated[col_order][None, :].astype(int), aspect="auto",
              cmap=ListedColormap(list(CONDITION.values())), vmin=0, vmax=1,
              interpolation="nearest")
ax_bar.set_axis_off()

cbar = fig.colorbar(image, cax=ax_cbar, ticks=[-2, -1, 0, 1, 2],
                    extend="both")
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(length=2, width=0.5)
cbar.ax.set_title("Row z-score", fontsize=ms.FS_TICK, pad=5, loc="left")

fig.legend(handles=[Patch(color=colour, label=name)
                    for name, colour in CONDITION.items()],
           title="Condition", alignment="left", loc="upper left",
           bbox_to_anchor=(SIDE / 120, (BOTTOM + 30.0) / 92))

# drawn leaves (at 5, 15, 25, ... in tree units) sit on the cell centres
fig.canvas.draw()
to_fig = fig.transFigure.inverted()
for k in range(N_SAMPLES):
    x_leaf = to_fig.transform(ax_cold.transData.transform((10 * k + 5, 0)))[0]
    x_cell = to_fig.transform(ax_heat.transData.transform((k, 0)))[0]
    assert abs(x_leaf - x_cell) * 120 < 0.05, (k, x_leaf, x_cell)
for k in range(N_GENES):
    y_leaf = to_fig.transform(ax_rowd.transData.transform((0, 10 * k + 5)))[1]
    y_cell = to_fig.transform(
        ax_heat.transData.transform((0, N_GENES - 1 - k)))[1]
    assert abs(y_leaf - y_cell) * 92 < 0.05, (k, y_leaf, y_cell)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig104_clustermap_annotated.{ext}")
print("fig104_clustermap_annotated: saved png + pdf")
