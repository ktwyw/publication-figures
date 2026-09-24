"""Fig. 38 - Small multiples: one panel per site, shared axes (double column).

Tufte's small multiples: twelve sites on identical axes so the eye can
compare directly, panels sorted by trend, a per-panel fitted trend line
coloured by sign, and shared axis titles via fig.supxlabel/supylabel.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(38)

# ------------------------------------------------------------- DATA ----
years = np.arange(2000, 2021)
sites = [f"Site {chr(65 + i)}" for i in range(12)]
slopes = rng.normal(0, 1.2, 12)
series = (50 + rng.normal(0, 8, (12, 1)) + slopes[:, None] * (years - 2010)
          + rng.normal(0, 3.5, (12, years.size)))
fits = [stats.linregress(years, s) for s in series]
order = np.argsort([-f.slope for f in fits])       # steepest increase first

# ------------------------------------------------------------- PLOT ----
fig, axs = plt.subplots(3, 4, figsize=(7.1, 4.4), sharex=True, sharey=True)

for ax, idx in zip(axs.flat, order):
    res = fits[idx]
    color = "C1" if res.slope > 0 else "C0"
    ax.plot(years, series[idx], color="0.55", lw=0.8, marker="o", ms=1.8)
    ax.plot(years, res.intercept + res.slope * years, color=color, lw=1.2, ls="--")
    ax.set_title(sites[idx], fontsize=7, loc="left", pad=2)
    sign = "+" if res.slope >= 0 else "\u2212"
    label = f"{sign}{abs(res.slope):.2f} yr$^{{-1}}$"
    if res.pvalue < 0.05:
        label += "*"
    ax.text(0.96, 0.06, label, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6, color=color)

axs[0, 0].set_xticks([2000, 2010, 2020])
fig.supxlabel("Year", fontsize=8)
fig.supylabel("Abundance index", fontsize=8)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig038_small_multiples.{ext}")
print("saved fig038_small_multiples.png / .pdf")
