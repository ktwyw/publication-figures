"""Fig. 6 - Raincloud plots: half-violin + box + raw data (single column).

Shows the full distribution, the summary statistics, and every raw data
point at once - increasingly expected by reviewers instead of bare bars.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(9)
OKABE = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# ------------------------------------------------------------- DATA ----
conditions = ["Congruent", "Neutral", "Incongruent"]
data = [420 + rng.gamma(6, 18, 80),
        450 + rng.gamma(6, 20, 80),
        520 + rng.gamma(7, 24, 80)]

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.5, 2.8))
pos = np.arange(1, len(data) + 1)

# half-violins (the "cloud"): clip each violin body to its left half
parts = ax.violinplot(data, positions=pos, widths=0.9, showextrema=False)
for body, color, p in zip(parts["bodies"], OKABE, pos):
    verts = body.get_paths()[0].vertices
    verts[:, 0] = np.clip(verts[:, 0], -np.inf, p)
    body.set_facecolor(color)
    body.set_alpha(0.55)
    body.set_edgecolor("none")

# narrow box plots (the summary)
ax.boxplot(data, positions=pos, widths=0.10, showfliers=False, zorder=3,
           patch_artist=True,
           boxprops=dict(facecolor="white", lw=0.8),
           medianprops=dict(color="black", lw=1.1),
           whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))

# jittered raw data (the "rain")
for p, d, color in zip(pos, data, OKABE):
    ax.scatter(p + rng.uniform(0.07, 0.21, d.size), d, s=7,
               color=color, alpha=0.6, lw=0, zorder=2)

ax.set_xticks(pos, conditions)
ax.set_xlim(0.35, len(data) + 0.75)
ax.set_ylabel("Reaction time (ms)")
ax.set_xlabel("Condition")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig006_raincloud.{ext}")
print("saved fig006_raincloud.png / .pdf")
