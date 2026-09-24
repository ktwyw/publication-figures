"""Fig. 26 - Sankey / alluvial flow diagram from scratch.

matplotlib's built-in Sankey class produces dated-looking diagrams, so
this draws its own: nodes as rectangles, flows as smoothstep-interpolated
ribbons via fill_between, ribbon widths proportional to flow, coloured by
source. Works for any two-stage flow matrix (sources x targets).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

# ------------------------------------------------------------- DATA ----
sources = ["Coal", "Gas", "Nuclear", "Renewables"]
targets = ["Industry", "Buildings", "Transport", "Losses"]
flows = np.array([[18, 6, 0, 8],       # TWh, rows = sources, cols = targets
                  [10, 14, 3, 5],
                  [6, 5, 0, 4],
                  [7, 9, 6, 3]], float)
unit = "TWh"
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# ------------------------------------------------------------ LAYOUT ----
X_L, X_R, W = 0.0, 1.0, 0.035         # node x positions and width
tot_l, tot_r = flows.sum(1), flows.sum(0)
gap = 0.06 * flows.sum()


def node_tops(totals, height_max):
    """Top y of each node, stacked from the top with gaps, centred."""
    h = totals.sum() + gap * (len(totals) - 1)
    y = height_max - (height_max - h) / 2
    tops = []
    for t in totals:
        tops.append(y)
        y -= t + gap
    return np.array(tops)


H = max(tot_l.sum() + gap * (len(tot_l) - 1), tot_r.sum() + gap * (len(tot_r) - 1))
top_l, top_r = node_tops(tot_l, H), node_tops(tot_r, H)

fig, ax = plt.subplots(figsize=(4.2, 3.0))
xs = np.linspace(X_L + W, X_R, 80)
t = (xs - xs[0]) / (xs[-1] - xs[0])
s = t * t * (3 - 2 * t)                # smoothstep: flat at both ends

off_l, off_r = np.zeros(len(sources)), np.zeros(len(targets))
for i in range(len(sources)):
    for j in range(len(targets)):
        f = flows[i, j]
        if f == 0:
            continue
        yl_top, yr_top = top_l[i] - off_l[i], top_r[j] - off_r[j]
        y_top = yl_top + (yr_top - yl_top) * s
        y_bot = y_top - f
        ax.fill_between(xs, y_bot, y_top, color=colors[i], alpha=0.45, lw=0)
        off_l[i] += f
        off_r[j] += f

for name, tot, top, col in zip(sources, tot_l, top_l, colors):
    ax.add_patch(Rectangle((X_L, top - tot), W, tot, color=col, lw=0))
    ax.text(X_L - 0.03, top - tot / 2, f"{name}\n{tot:.0f} {unit}",
            ha="right", va="center", fontsize=6.5)
for name, tot, top in zip(targets, tot_r, top_r):
    ax.add_patch(Rectangle((X_R, top - tot), W, tot, color="0.35", lw=0))
    ax.text(X_R + W + 0.03, top - tot / 2, f"{name}\n{tot:.0f} {unit}",
            ha="left", va="center", fontsize=6.5)

ax.set_xlim(-0.32, 1.37)
ax.set_ylim(-gap / 2, H + gap / 2)
ax.axis("off")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig026_sankey.{ext}")
print("saved fig026_sankey.png / .pdf")
