"""Fig. 43 - Broken y-axis for one dominant value (single column).

When one category dwarfs the rest, a broken axis keeps the small bars
readable without a log scale. Two stacked axes show the same data with
different y-limits; diagonal break marks signal the discontinuity.
Use sparingly and always mark the break clearly.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

# ------------------------------------------------------------- DATA ----
labels = ["Wild type", "\u0394abcA", "\u0394abcB", "\u0394abcC", "\u0394abcD", "Double"]
values = np.array([485, 41, 27, 33, 12, 8])
colors = ["0.35"] + ["C0"] * 5

# ------------------------------------------------------------- PLOT ----
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(3.3, 2.8), sharex=True,
                                     height_ratios=[1, 2.4])
x = np.arange(len(labels))
for ax, (lo, hi) in ((ax_top, (440, 520)), (ax_bot, (0, 60))):
    ax.bar(x, values, width=0.65, color=colors)
    ax.set_ylim(lo, hi)
    for xi, v in zip(x, values):
        if lo <= v <= hi:                      # label only where visible
            ax.text(xi, v + (hi - lo) * 0.02, str(v), ha="center",
                    va="bottom", fontsize=6)

ax_top.set_yticks([460, 500])

# hide the joint between the two axes
ax_top.spines["bottom"].set_visible(False)
ax_top.tick_params(bottom=False)

# diagonal break marks on the left spine
d = 0.012
kw = dict(color="black", clip_on=False, lw=0.8)
ax_top.plot((-d, d), (-d, d), transform=ax_top.transAxes, **kw)
ax_bot.plot((-d, d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kw)

ax_bot.set_xticks(x, labels, fontsize=6.5)
fig.supylabel("Colony-forming units (\u00d710$^3$)", fontsize=8)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig043_broken_axis.{ext}")
print("saved fig043_broken_axis.png / .pdf")
