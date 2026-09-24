"""Fig. 14 - Scatter with marginal distributions (a "joint plot" from scratch).

Techniques shown: GridSpec with width/height ratios, shared axes,
2-D KDE contours over the scatter, and 1-D KDE curves on the margins.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, pearsonr

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(14)

# ------------------------------------------------------------- DATA ----
n = 500
x = rng.normal(1.6, 0.55, n)                      # log10 mRNA abundance
y = 0.75 * x + rng.normal(0.9, 0.42, n)           # log10 protein abundance
r, _ = pearsonr(x, y)

# ------------------------------------------------------------ LAYOUT ----
fig = plt.figure(figsize=(3.6, 3.6))
gs = fig.add_gridspec(2, 2, width_ratios=(4, 1.1), height_ratios=(1.1, 4))
ax = fig.add_subplot(gs[1, 0])
ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
ax_corner = fig.add_subplot(gs[0, 1])
ax_corner.axis("off")

# ------------------------------------------------------- MAIN PANEL ----
ax.scatter(x, y, s=8, color="C0", alpha=0.45, lw=0, rasterized=True)
kde2 = gaussian_kde(np.vstack([x, y]))
gx, gy = np.mgrid[x.min():x.max():120j, y.min():y.max():120j]
dens = kde2(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
ax.contour(gx, gy, dens, levels=4, colors="0.25", linewidths=0.6)
ax.set_xlabel(r"mRNA abundance (log$_{10}$ TPM)")
ax.set_ylabel(r"Protein abundance (log$_{10}$ iBAQ)")

# --------------------------------------------------------- MARGINALS ----
bins_x = np.linspace(x.min(), x.max(), 28)
bins_y = np.linspace(y.min(), y.max(), 28)
ax_top.hist(x, bins_x, density=True, color="C0", alpha=0.4, lw=0)
ax_right.hist(y, bins_y, density=True, color="C0", alpha=0.4, lw=0,
              orientation="horizontal")
xx = np.linspace(x.min(), x.max(), 200)
yy = np.linspace(y.min(), y.max(), 200)
ax_top.plot(xx, gaussian_kde(x)(xx), color="C0", lw=1.1)
ax_right.plot(gaussian_kde(y)(yy), yy, color="C0", lw=1.1)

ax_top.tick_params(labelbottom=False)
ax_top.set_yticks([])
ax_top.spines["left"].set_visible(False)
ax_right.tick_params(labelleft=False)
ax_right.set_xticks([])
ax_right.spines["bottom"].set_visible(False)

ax_corner.text(0.05, 0.35, rf"$r$ = {r:.2f}" + "\n" + rf"$n$ = {n}",
               fontsize=7, va="center")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig014_joint_marginals.{ext}")
print("saved fig014_joint_marginals.png / .pdf")
