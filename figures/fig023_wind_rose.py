"""Fig. 23 - Wind rose: directional data on a polar axis (single column).

Stacked polar bars by speed class, compass orientation (N up, clockwise),
radial ticks in percent, and an outside legend. The same pattern serves
any orientation data: animal headings, fracture strikes, phase angles.
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(23)

# ------------------------------------------------------------- DATA ----
n = 2000
theta = rng.vonmises(np.deg2rad(225), 2.0, n) % (2 * np.pi)  # prevailing SW
speed = rng.weibull(2.0, n) * 4.5

n_sect = 16
edges = np.linspace(0, 2 * np.pi, n_sect + 1)
centers = edges[:-1] + np.pi / n_sect
speed_bins = [(0, 3), (3, 6), (6, 9), (9, np.inf)]
speed_labels = ["0\u20133", "3\u20136", "6\u20139", "> 9"]
shades = [mpl.colormaps["YlGnBu"](v) for v in np.linspace(0.30, 0.95, 4)]

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.3, 3.1),
                       subplot_kw=dict(projection="polar"))
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

bottom = np.zeros(n_sect)
for (lo, hi), lab, col in zip(speed_bins, speed_labels, shades):
    mask = (speed >= lo) & (speed < hi)
    pct = 100 * np.histogram(theta[mask], bins=edges)[0] / n
    ax.bar(centers, pct, width=0.9 * 2 * np.pi / n_sect, bottom=bottom,
           color=col, edgecolor="white", linewidth=0.3, label=lab)
    bottom += pct

ax.set_thetagrids(np.arange(0, 360, 45),
                  ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
ax.set_rlabel_position(112)
ax.set_yticks([4, 8, 12])
ax.set_yticklabels(["4%", "8%", "12%"], fontsize=6)
ax.grid(True, lw=0.4, color="0.85")
ax.spines["polar"].set_linewidth(0.8)

ax.legend(title="Wind speed (m s$^{-1}$)", loc="upper left",
          bbox_to_anchor=(1.08, 1.05), fontsize=6, title_fontsize=6.5)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig023_wind_rose.{ext}")
print("saved fig023_wind_rose.png / .pdf")
