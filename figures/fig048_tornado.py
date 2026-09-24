"""Fig. 48 - Tornado diagram: one-way sensitivity analysis (single column).

Each parameter is varied to its low and high bound while the others stay
at base case; bars show the resulting swing in the outcome, sorted so
the most influential parameter is on top. Standard in health economics,
engineering design and any model-based paper.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

# ------------------------------------------------------------- DATA ----
BASE = 25_000                                    # base-case ICER, $/QALY
# parameter: (outcome at low input, outcome at high input)
params = {"Drug price": (18_500, 33_200),
          "Utility gain": (34_800, 19_600),
          "Time horizon": (30_500, 21_900),
          "Discount rate": (23_100, 27_400),
          "Hospitalisation cost": (26_900, 22_800),
          "Adherence": (29_500, 22_400),
          "Mortality reduction": (31_000, 21_700),
          "Monitoring cost": (24_100, 26_200)}

names = np.array(list(params))
low = np.array([v[0] for v in params.values()], float)
high = np.array([v[1] for v in params.values()], float)
order = np.argsort(np.abs(high - low))            # smallest swing at bottom
names, low, high = names[order], low[order], high[order]

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.8, 2.8))
y = np.arange(len(names))
ax.barh(y, low - BASE, left=BASE, height=0.6, color="C0")
ax.barh(y, high - BASE, left=BASE, height=0.6, color="C1")
ax.axvline(BASE, color="black", lw=0.8)

for yi, lo, hi in zip(y, low, high):
    for v in (lo, hi):
        ax.text(v + np.sign(v - BASE) * 300, yi, f"{v / 1000:.1f}k",
                va="center", ha="left" if v > BASE else "right", fontsize=5.5,
                color="0.3")

ax.set_yticks(y, names)
ax.tick_params(left=False)
ax.spines["left"].set_visible(False)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v / 1000:.0f}k"))
ax.set_xlim(low.min() - 4_500, high.max() + 4_500)
ax.set_xlabel("ICER ($ per QALY gained)")
ax.text(BASE - 250, len(y) - 0.3, f"Base case ${BASE / 1000:.0f}k", ha="right",
        va="bottom", fontsize=6.5)
ax.legend(handles=[Patch(color="C0", label="Lower bound of input"),
                   Patch(color="C1", label="Upper bound of input")],
          loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=6,
          borderaxespad=0)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig048_tornado.{ext}")
print("saved fig048_tornado.png / .pdf")
