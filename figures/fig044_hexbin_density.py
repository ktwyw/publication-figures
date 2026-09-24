"""Fig. 44 - Hexbin density for very large datasets (single column).

Scatter plots saturate past ~10^4 points; hexagonal binning with a log
colour scale shows structure at every density. A polygonal "gate" with
the enclosed fraction demonstrates Path.contains_points - the flow
cytometry idiom, equally useful for any 2-D population data.
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(44)

# ------------------------------------------------------------- DATA ----
pops = [((300, 180), (60, 40), 120_000),      # lymphocyte-like
        ((520, 420), (90, 110), 90_000),      # monocyte-like
        ((650, 750), (110, 130), 60_000),     # granulocyte-like
        ((150, 80), (50, 30), 30_000)]        # debris
fsc = np.concatenate([rng.normal(m[0], s[0], n) for m, s, n in pops])
ssc = np.concatenate([rng.normal(m[1], s[1], n) for m, s, n in pops])
keep = (fsc > 0) & (ssc > 0) & (fsc < 1000) & (ssc < 1000)
fsc, ssc = fsc[keep], ssc[keep]

gate = np.array([[190, 90], [420, 90], [460, 250], [330, 320], [180, 260]])
inside = MplPath(gate).contains_points(np.column_stack([fsc, ssc]))
pct = 100 * inside.mean()

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.4, 3.0))
hb = ax.hexbin(fsc, ssc, gridsize=90, cmap="viridis", mincnt=1,
               norm=mpl.colors.LogNorm(), linewidths=0.1, rasterized=True)
ax.add_patch(PathPatch(MplPath(np.vstack([gate, gate[:1]]), closed=True),
                       facecolor="none", edgecolor="white", lw=1.0))
ax.text(gate[:, 0].mean(), gate[:, 1].max() + 25,
        f"Lymphocytes\n{pct:.1f}%", ha="center", va="bottom", fontsize=6.5,
        color="white", path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])

ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
ax.set_xlabel("Forward scatter (a.u.)")
ax.set_ylabel("Side scatter (a.u.)")
ax.text(0.97, 0.03, f"$n$ = {fsc.size:,} cells", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=6.5)

cbar = fig.colorbar(hb, ax=ax, pad=0.02)
cbar.set_label("Cells per bin")
cbar.outline.set_linewidth(0.8)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig044_hexbin_density.{ext}")
print("saved fig044_hexbin_density.png / .pdf")
