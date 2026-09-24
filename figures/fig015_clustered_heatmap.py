"""Fig. 15 - Clustered heatmap with dendrograms (a "clustermap" from scratch).

Techniques shown: scipy hierarchical clustering, drawing dendrograms on
side axes aligned with the reordered matrix, a dedicated colorbar axis,
and group-coloured tick labels. Alignment trick: dendrogram leaves sit
at 5, 15, 25, ... so setting the side-axis limits to (0, 10*n) matches
imshow's default extent exactly.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(16)

# ------------------------------------- SIMULATED EXPRESSION MATRIX ----
n_genes, n_samples = 24, 12
X = rng.normal(0, 1, (n_genes, n_samples))
X[:10, :6] += 2.2          # module 1: high in group A
X[10:20, 6:] += 2.2        # module 2: high in group B
Xz = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

genes = [f"Gene {i + 1:02d}" for i in range(n_genes)]
samples = [f"A{i + 1}" for i in range(6)] + [f"B{i + 1}" for i in range(6)]

# --------------------------------------------------------- CLUSTER ----
link_rows = hierarchy.linkage(Xz, method="average", metric="correlation")
link_cols = hierarchy.linkage(Xz.T, method="ward", metric="euclidean")

# ------------------------------------------------------------ LAYOUT ----
fig = plt.figure(figsize=(4.0, 3.6))
gs = fig.add_gridspec(2, 3, width_ratios=(0.16, 1, 0.045),
                      height_ratios=(0.16, 1))
ax_col = fig.add_subplot(gs[0, 1])
ax_row = fig.add_subplot(gs[1, 0])
ax_hm = fig.add_subplot(gs[1, 1])
cax = fig.add_subplot(gs[1, 2])

grey = dict(link_color_func=lambda _: "0.4", no_labels=True)
dc = hierarchy.dendrogram(link_cols, ax=ax_col, **grey)
dr = hierarchy.dendrogram(link_rows, ax=ax_row, orientation="left", **grey)
ax_col.axis("off")
ax_col.set_xlim(0, 10 * n_samples)
ax_row.axis("off")
ax_row.set_ylim(10 * n_genes, 0)      # inverted to match imshow origin

# ----------------------------------------------------------- HEATMAP ----
order_r, order_c = dr["leaves"], dc["leaves"]
im = ax_hm.imshow(Xz[np.ix_(order_r, order_c)], cmap="RdBu_r",
                  vmin=-2.5, vmax=2.5, aspect="auto",
                  interpolation="nearest")
ax_hm.set_xticks(range(n_samples), [samples[i] for i in order_c],
                 fontsize=6)
ax_hm.yaxis.tick_right()
ax_hm.set_yticks(range(n_genes), [genes[i] for i in order_r], fontsize=5)
for lbl in ax_hm.get_xticklabels():
    lbl.set_color("C0" if lbl.get_text().startswith("A") else "C1")
for spine in ax_hm.spines.values():
    spine.set_visible(False)
ax_hm.tick_params(length=0)

cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Row $z$-score")
cbar.outline.set_linewidth(0.8)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig015_clustered_heatmap.{ext}")
print("saved fig015_clustered_heatmap.png / .pdf")
