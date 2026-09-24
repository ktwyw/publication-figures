"""Fig. 21 - Stacked composition over time with direct labels (single column).

Direct labels at the right edge beat a legend: the reader never has to
map colours back and forth. Semantic colours (coal grey, solar yellow)
also carry meaning. Bands are normalised to 100%.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))


def logistic(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


# ------------------------------------------------------------- DATA ----
year = np.arange(2000, 2026)
raw = {
    "Coal":    40 * (1 - logistic(year, 2014, 0.35)) + 6,
    "Gas":     22 + 10 * logistic(year, 2010, 0.3),
    "Nuclear": 18 - 3 * logistic(year, 2012, 0.3),
    "Hydro":   14 + 0 * year,
    "Wind":    1 + 20 * logistic(year, 2016, 0.35),
    "Solar":   0.3 + 13 * logistic(year, 2019, 0.45),
}
total = sum(raw.values())
shares = {k: 100 * v / total for k, v in raw.items()}

colors = {"Coal": "#4d4d4d", "Gas": "#E69F00", "Nuclear": "#CC79A7",
          "Hydro": "#56B4E9", "Wind": "#009E73", "Solar": "#F0E442"}
label_colors = dict(colors, Solar="#8a7d00")   # darker text for yellow

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.6, 2.7))
ax.stackplot(year, shares.values(), colors=colors.values(),
             edgecolor="white", linewidth=0.4)

# direct labels centred on each band at the final year
cum = 0.0
for name, s in shares.items():
    center = cum + s[-1] / 2
    ax.text(year[-1] + 0.6, center, name, color=label_colors[name],
            fontsize=6.5, va="center")
    cum += s[-1]

ax.set_xlim(year[0], year[-1])
ax.set_ylim(0, 100)
ax.set_xlabel("Year")
ax.set_ylabel("Share of generation (%)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig021_stacked_composition.{ext}")
print("saved fig021_stacked_composition.png / .pdf")
