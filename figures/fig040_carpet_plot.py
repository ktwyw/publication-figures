"""Fig. 40 - Carpet plot: hour of day x day of year (double column).

A year of hourly data as one image: rows are hours, columns are days.
Diurnal and seasonal structure (and the weekly rhythm) become visible at
once. Uses pcolormesh with explicit cell edges and month-centred ticks.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(40)

# ------------------------------------------------------------- DATA ----
days, hours = np.arange(365), np.arange(24)
D, H = np.meshgrid(days, hours)                            # (24, 365)
season = 1 + 0.25 * np.cos(2 * np.pi * (D - 15) / 365)     # winter peak
diurnal = (0.55 + 0.5 * np.exp(-((H - 8) / 2.0) ** 2)
           + 0.8 * np.exp(-((H - 18.5) / 2.5) ** 2) + 0.15 * (H > 7) * (H < 22))
weekend = np.where(D % 7 >= 5, 0.85, 1.0)
demand = 2000 * season * diurnal * weekend * (1 + rng.normal(0, 0.05, D.shape))

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(7.1, 2.3))
pc = ax.pcolormesh(np.arange(366), np.arange(25), demand, cmap="viridis",
                   shading="flat", rasterized=True)

month_start = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])
ax.set_xticks((month_start[:-1] + month_start[1:]) / 2,
              ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.tick_params(axis="x", length=0)
ax.set_yticks([0, 6, 12, 18, 24])
ax.set_ylim(0, 24)
ax.set_ylabel("Hour of day")

cbar = fig.colorbar(pc, ax=ax, pad=0.015)
cbar.set_label("Electricity demand (MW)")
cbar.outline.set_linewidth(0.8)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig040_carpet_plot.{ext}")
print("saved fig040_carpet_plot.png / .pdf")
