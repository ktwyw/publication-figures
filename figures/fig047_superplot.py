"""Fig. 47 - SuperPlot: cell-level swarm + replicate-level statistics.

Following Lord et al. (2020, J. Cell Biol.): every cell is shown as a
swarm point coloured by biological replicate, replicate means are
overlaid as large markers and joined across conditions, and the
statistical test is run on the replicate means (n = 3), not on cells.
The beeswarm layout is a compact from-scratch implementation.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(47)


def swarm_x(values, center, dx=0.028, width=0.32, nbins=45):
    """Beeswarm x-positions: alternate points outward within thin y-strips."""
    x = np.full(values.size, float(center))
    edges = np.linspace(values.min(), values.max(), nbins + 1)
    strip = np.clip(np.digitize(values, edges[1:-1]), 0, nbins - 1)
    for b in np.unique(strip):
        idx = np.flatnonzero(strip == b)
        k = np.arange(idx.size)
        offsets = ((k + 1) // 2) * np.where(k % 2 == 0, 1, -1) * dx
        x[idx] = center + np.clip(offsets, -width, width)
    return x


# ------------------------------------------------------------- DATA ----
conditions = ["Control", "Treated"]
N_REP, N_CELLS = 3, 55
rep_offset = rng.normal(0, 9, N_REP)                      # replicate-level variation
cells = {c: [rng.normal(120 + 32 * i + rep_offset[r], 20, N_CELLS)
             for r in range(N_REP)] for i, c in enumerate(conditions)}
rep_means = np.array([[v.mean() for v in cells[c]] for c in conditions])  # (2, 3)
t_res = stats.ttest_rel(rep_means[1], rep_means[0])

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(2.9, 3.0))
for xi, c in enumerate(conditions):
    all_vals = np.concatenate(cells[c])
    rep_id = np.repeat(np.arange(N_REP), N_CELLS)
    xs = swarm_x(all_vals, xi)
    ax.scatter(xs, all_vals, s=7, c=[f"C{r}" for r in rep_id], alpha=0.55,
               lw=0, zorder=2)
for r in range(N_REP):
    ax.plot([0, 1], rep_means[:, r], color=f"C{r}", lw=0.8, alpha=0.7, zorder=3)
    ax.scatter([0, 1], rep_means[:, r], s=55, color=f"C{r}", edgecolor="black",
               lw=0.8, zorder=4)

top = max(np.concatenate(v).max() for v in cells.values())
ax.plot([0, 0, 1, 1], [top + 8, top + 14, top + 14, top + 8], lw=0.8,
        color="black", clip_on=False)
p_txt = "$P$ < 0.001" if t_res.pvalue < 1e-3 else f"$P$ = {t_res.pvalue:.3f}"
ax.text(0.5, top + 16, f"{p_txt} (paired $t$, $n$ = {N_REP})", ha="center",
        va="bottom", fontsize=6.5)

ax.set_xticks([0, 1], conditions)
ax.set_xlim(-0.6, 1.6)
bottom = min(np.concatenate(v).min() for v in cells.values())
ax.set_ylim(bottom - 12, top + 40)
ax.set_ylabel("Cell area (\u00b5m$^2$)")
ax.legend(handles=[Line2D([], [], ls="", marker="o", color=f"C{r}",
                          mec="black", ms=5, label=f"Replicate {r + 1}")
                   for r in range(N_REP)],
          loc="lower right", fontsize=6)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig047_superplot.{ext}")
print("saved fig047_superplot.png / .pdf")
