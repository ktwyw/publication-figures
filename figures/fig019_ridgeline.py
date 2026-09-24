"""Fig. 19 - Ridgeline plot: distributions across an ordered variable.

Overlapping KDE ridges with occlusion (later rows drawn in front),
white halo edges for separation, and fill colours mapped to each
row's mean so the colour itself carries information.
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(19)

# ------------------------------------------------------------- DATA ----
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
mu = 10 - 12 * np.cos(2 * np.pi * (np.arange(12) + 0.5) / 12)
samples = [rng.normal(m, 3.0, 300) for m in mu]

grid = np.linspace(-14, 34, 400)
kdes = [gaussian_kde(s)(grid) for s in samples]
scale = 1.55 / max(k.max() for k in kdes)      # ridge height vs spacing

cmap = mpl.colormaps["coolwarm"]
norm = mpl.colors.Normalize(mu.min(), mu.max())

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.5, 3.6))

offsets = []
for i, (k, m) in enumerate(zip(kdes, mu)):
    y0 = 11 - i                                # Jan on top
    offsets.append(y0)
    ax.fill_between(grid, y0, y0 + k * scale, facecolor=cmap(norm(m)),
                    lw=0, zorder=i + 1)
    ax.plot(grid, y0 + k * scale, color="white", lw=1.4, zorder=i + 1)
    ax.plot(grid, y0 + k * scale, color="0.25", lw=0.6, zorder=i + 1)

ax.set_yticks(offsets, months)
ax.tick_params(left=False)
ax.spines["left"].set_visible(False)
ax.set_xlim(grid[0], grid[-1])
ax.set_ylim(-0.3, 13.4)
ax.set_xlabel("Daily mean temperature (\u00b0C)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig019_ridgeline.{ext}")
print("saved fig019_ridgeline.png / .pdf")
