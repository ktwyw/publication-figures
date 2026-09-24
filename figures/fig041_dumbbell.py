"""Fig. 41 - Dumbbell chart: change between two time points (single column).

Two values per category joined by a bar, sorted by the later value,
with the change labelled at the right edge. Far clearer than grouped
bars for "then vs now" comparisons. Illustrative values, not statistics.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

# ------------------------------------------------------------- DATA ----
countries = ["Nigeria", "India", "Indonesia", "Egypt", "Brazil", "Mexico",
             "China", "T\u00fcrkiye", "Russia", "United States", "Germany",
             "Italy", "Spain", "Japan"]
y2000 = np.array([46.5, 62.7, 66.3, 68.6, 70.1, 74.8, 71.4, 70.0, 65.3,
                  76.7, 78.0, 79.5, 79.0, 81.1])
y2023 = np.array([54.5, 72.0, 71.9, 71.8, 75.8, 75.1, 78.6, 77.3, 73.2,
                  79.3, 81.4, 83.7, 83.8, 84.7])

order = np.argsort(y2023)
countries = [countries[i] for i in order]
y2000, y2023 = y2000[order], y2023[order]

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.6, 3.4))
y = np.arange(len(countries))

ax.hlines(y, y2000, y2023, color="0.78", lw=2.2, zorder=1)
ax.scatter(y2000, y, s=22, color="0.45", zorder=3, label="2000")
ax.scatter(y2023, y, s=22, color="C1", zorder=3, label="2023")

tform = ax.get_yaxis_transform()
for yi, d in zip(y, y2023 - y2000):
    ax.text(1.03, yi, f"+{d:.1f}", transform=tform, va="center",
            fontsize=6, color="0.3")
ax.text(1.03, len(y) - 0.4, "\u0394", transform=tform, fontsize=6.5,
        color="0.3", va="center")

ax.set_yticks(y, countries)
ax.set_ylim(-0.7, len(y) - 0.3)
ax.tick_params(left=False)
ax.spines["left"].set_visible(False)
ax.set_xlabel("Life expectancy at birth (years)")
ax.set_xlim(42, 88)
ax.legend(loc="lower right", fontsize=6.5, markerscale=0.9)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig041_dumbbell.{ext}")
print("saved fig041_dumbbell.png / .pdf")
