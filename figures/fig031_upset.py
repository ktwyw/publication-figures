"""Fig. 31 - UpSet plot: set intersections beyond what Venn diagrams can show.

Exclusive intersection sizes as bars, membership as a dot matrix with
connecting lines, and set sizes at the left - assembled from GridSpec
with shared axes so everything stays aligned. No external package.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(31)

# ------------------------------------------------------------- DATA ----
sets = ["Liver", "Kidney", "Heart", "Brain"]
M = rng.uniform(size=(1500, 4)) < np.array([0.30, 0.26, 0.22, 0.18])
M[:120] = True                    # core genes shared by all tissues
M[120:260, :2] = True             # liver-kidney block
M[260:340, 2:] = True             # heart-brain block
M = M[M.any(axis=1)]

patterns, counts = np.unique(M, axis=0, return_counts=True)
order = np.argsort(-counts)[:12]  # twelve largest exclusive intersections
patterns, counts = patterns[order], counts[order]
set_sizes = M.sum(axis=0)
n_sets, n_int = len(sets), len(counts)

# ------------------------------------------------------------ LAYOUT ----
fig = plt.figure(figsize=(4.6, 3.2))
gs = fig.add_gridspec(2, 2, width_ratios=(1, 3.6), height_ratios=(2.4, 1.0))
ax_bar = fig.add_subplot(gs[0, 1])
ax_mat = fig.add_subplot(gs[1, 1], sharex=ax_bar)
ax_set = fig.add_subplot(gs[1, 0], sharey=ax_mat)

# intersection sizes
x = np.arange(n_int)
ax_bar.bar(x, counts, width=0.6, color="0.25")
for xi, c in zip(x, counts):
    ax_bar.text(xi, c + counts.max() * 0.02, str(c), ha="center",
                va="bottom", fontsize=6)
ax_bar.set_ylabel("Intersection size")
ax_bar.set_ylim(0, counts.max() * 1.15)
ax_bar.tick_params(labelbottom=False, bottom=False)
ax_bar.spines["bottom"].set_visible(False)

# membership matrix
y = np.arange(n_sets)
for yi in y[::2]:
    ax_mat.axhspan(yi - 0.5, yi + 0.5, color="0.94", lw=0, zorder=0)
for xi, pat in zip(x, patterns):
    ax_mat.scatter(np.full(n_sets, xi), y, s=26, lw=0, zorder=2,
                   color=np.where(pat, "0.15", "0.82"))
    members = np.flatnonzero(pat)
    if members.size > 1:
        ax_mat.plot([xi, xi], [members.min(), members.max()],
                    color="0.15", lw=1.4, zorder=1)
ax_mat.set_yticks(y, sets)
ax_mat.set_ylim(-0.5, n_sets - 0.5)
ax_mat.invert_yaxis()
ax_mat.tick_params(left=False, bottom=False, labelbottom=False)
for spine in ax_mat.spines.values():
    spine.set_visible(False)

# set sizes
ax_set.barh(y, set_sizes, height=0.6, color="0.45")
ax_set.invert_xaxis()
ax_set.set_xlabel("Set size")
ax_set.tick_params(left=False, labelleft=False)
ax_set.spines["left"].set_visible(False)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig031_upset.{ext}")
print("saved fig031_upset.png / .pdf")
