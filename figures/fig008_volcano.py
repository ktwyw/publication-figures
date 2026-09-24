"""Fig. 8 - Volcano plot for differential expression (single column).

Notes for real use:
- scatter(..., rasterized=True) keeps the PDF small and fast to render
  even with 10^4-10^6 points, while axes/text stay vector.
- In practice use FDR-adjusted P values (e.g. Benjamini-Hochberg) for
  the significance threshold, and report thresholds in the caption.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(6)

# ------------------------------------------------- SIMULATED RESULTS ----
n_genes = 2400
fc = rng.normal(0, 0.55, n_genes)                      # log2 fold change
hits = rng.choice(n_genes, 90, replace=False)          # true positives
fc[hits] += rng.choice([-1, 1], hits.size) * rng.uniform(1.0, 2.0, hits.size)
se = rng.uniform(0.28, 0.55, n_genes)
p = 2 * stats.norm.sf(np.abs(fc / se))
p = np.clip(p, 1e-14, 1.0)
neglog = -np.log10(p)

FC_THR, P_THR = 1.0, 1e-2
sig = (np.abs(fc) > FC_THR) & (p < P_THR)
up, down = sig & (fc > 0), sig & (fc < 0)
ns = ~sig

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.5, 3.0))
ax.scatter(fc[ns], neglog[ns], s=6, color="0.75", lw=0, rasterized=True)
ax.scatter(fc[down], neglog[down], s=8, color="C0", lw=0, rasterized=True)
ax.scatter(fc[up], neglog[up], s=8, color="C1", lw=0, rasterized=True)

ax.axhline(-np.log10(P_THR), ls="--", lw=0.7, color="0.5")
for x0 in (-FC_THR, FC_THR):
    ax.axvline(x0, ls="--", lw=0.7, color="0.5")

# label the strongest hits on each side (gene symbols in italics)
def top_of(mask, k=3):
    idx = np.where(mask)[0]
    return idx[np.argsort(neglog[idx])[::-1][:k]]


# offsets staggered by rank to avoid collisions; for dozens of labels,
# the adjustText package automates this
for idxs, gene_names, right, dys in [
        (top_of(up), ["MMP9", "SERPINE1", "EGR1"], True, (9, -9, 2)),
        (top_of(down), ["IL6", "CXCL8", "COL1A1"], False, (-10, -9, 3))]:
    for i, name, dy in zip(idxs, gene_names, dys):
        ax.annotate(name, (fc[i], neglog[i]),
                    xytext=(7 if right else -7, dy), textcoords="offset points",
                    fontsize=6, style="italic",
                    ha="left" if right else "right",
                    va="bottom" if dy >= 0 else "top",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="0.35",
                                    shrinkA=0, shrinkB=2))

lim = np.ceil(np.abs(fc).max()) + 0.3
ax.set_xlim(-lim, lim)
ax.set_ylim(0, neglog.max() * 1.1)
ax.set_xlabel(r"$\log_2$ fold change")
ax.set_ylabel(r"$-\log_{10}(P)$")
ax.text(0.98, 0.97, f"Up: {up.sum()}", transform=ax.transAxes,
        ha="right", va="top", color="C1")
ax.text(0.02, 0.97, f"Down: {down.sum()}", transform=ax.transAxes,
        ha="left", va="top", color="C0")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig008_volcano.{ext}")
print("saved fig008_volcano.png / .pdf")
