"""Fig. 28 - Waterfall chart: decomposition of a change (single column).

Used for emissions drivers, budget bridges, energy balances, and any
"start -> contributions -> end" story. Shown: floating bars positioned
by cumulative sum, colour by sign, connector lines, signed value labels.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

# ------------------------------------------------------------- DATA ----
start_label, end_label = "2015", "2025"
start = 100.0
steps = [("Activity\ngrowth", 18), ("Energy\nintensity", -12),
         ("Fuel\nswitching", -9), ("Renewables", -14), ("CCS", -3)]
end = start + sum(d for _, d in steps)

COL_UP, COL_DOWN, COL_TOTAL = "C1", "C2", "0.35"

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.9, 2.7))
x = np.arange(len(steps) + 2)

ax.bar(x[0], start, color=COL_TOTAL, width=0.65)
ax.text(x[0], start + 1.5, f"{start:.0f}", ha="center", va="bottom", fontsize=6.5)

level = start
for xi, (label, d) in zip(x[1:-1], steps):
    bottom = level if d >= 0 else level + d
    ax.bar(xi, abs(d), bottom=bottom, width=0.65,
           color=COL_UP if d >= 0 else COL_DOWN)
    ax.plot([xi - 1 + 0.325, xi - 0.325], [level, level],
            color="0.5", lw=0.6, ls="--")
    sign = "+" if d >= 0 else "\u2212"
    ax.text(xi, bottom + abs(d) + 1.5, f"{sign}{abs(d):.0f}",
            ha="center", va="bottom", fontsize=6.5)
    level += d

ax.plot([x[-2] + 0.325, x[-1] - 0.325], [level, level],
        color="0.5", lw=0.6, ls="--")
ax.bar(x[-1], end, color=COL_TOTAL, width=0.65)
ax.text(x[-1], end + 1.5, f"{end:.0f}", ha="center", va="bottom", fontsize=6.5)

ax.set_xticks(x, [start_label] + [s for s, _ in steps] + [end_label],
              fontsize=6.5)
ax.set_ylabel("CO$_2$ emissions (Mt yr$^{-1}$)")
ax.set_ylim(0, max(start, start + steps[0][1]) * 1.12)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig028_waterfall.{ext}")
print("saved fig028_waterfall.png / .pdf")
