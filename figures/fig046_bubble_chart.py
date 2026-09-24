"""Fig. 46 - Bubble chart with a size legend (single column).

Three variables per point: x, y, and marker area for a third (population).
Shows a log axis with plain-number ticks, colour by group, and the piece
people usually get wrong - a size legend built from proxy markers whose
diameters are sqrt(area), so it truly matches the plotted bubbles.
Simulated data.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(46)

# ------------------------------------------------------------- DATA ----
regions = ["Africa", "Asia", "Europe", "Americas"]
n_per = [12, 12, 10, 10]
gdp_mu = [3.4, 3.9, 4.5, 4.1]                         # log10 GDP per capita
gdp, life, pop, region = [], [], [], []
for r, n, mu in zip(regions, n_per, gdp_mu):
    g = 10 ** rng.normal(mu, 0.35, n)
    gdp.append(g)
    life.append(48 + 8.5 * np.log10(g / 1000) + rng.normal(0, 2.2, n))
    pop.append(10 ** rng.uniform(6.3, 8.9, n))
    region += [r] * n
gdp, life, pop = map(np.concatenate, (gdp, life, pop))
region = np.array(region)

SIZE_MIN, SIZE_SCALE = 8, 520
sizes = SIZE_MIN + SIZE_SCALE * pop / pop.max()

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.6, 3.0))
for i, r in enumerate(regions):
    m = region == r
    ax.scatter(gdp[m], life[m], s=sizes[m], color=f"C{i}", alpha=0.7,
               edgecolor="white", lw=0.5, label=r)

ax.set_xscale("log")
ax.set_xticks([1_000, 3_000, 10_000, 30_000, 100_000],
              ["1k", "3k", "10k", "30k", "100k"])
ax.minorticks_off()
ax.set_xlabel("GDP per capita (US$)")
ax.set_ylabel("Life expectancy (years)")

region_handles = [Line2D([], [], ls="", marker="o", color=f"C{i}", alpha=0.7,
                         ms=6, label=r) for i, r in enumerate(regions)]
region_legend = ax.legend(handles=region_handles, loc="lower right", fontsize=6,
                          title="Region", title_fontsize=6.5)
ax.add_artist(region_legend)

pop_levels = [10e6, 100e6, 500e6]
handles = [Line2D([], [], ls="", marker="o", color="0.6", alpha=0.7,
                  ms=np.sqrt(SIZE_MIN + SIZE_SCALE * p / pop.max()),
                  label=f"{p / 1e6:.0f} M")
           for p in pop_levels]
ax.legend(handles=handles, loc="upper left", fontsize=6, title="Population",
          title_fontsize=6.5, labelspacing=1.4, borderpad=1.0)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig046_bubble_chart.{ext}")
print("saved fig046_bubble_chart.png / .pdf")
