"""Fig. 49 - Energy-level (Jablonski) diagram drawn programmatically.

Schematic figures are usually drawn by hand; doing them in code makes
them reproducible and restyle-able. Shown: electronic states with
vibrational sublevels, radiative transitions as solid arrows and
non-radiative ones as dashed arrows, all placed in energy units.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

# ------------------------------------------------------------- DATA ----
states = {"S$_0$": (0.0, 0.0, 1.0), "S$_1$": (2.9, 0.0, 1.0),
          "S$_2$": (3.8, 0.0, 1.0), "T$_1$": (2.3, 1.5, 2.5)}   # E (eV), x0, x1
VIB = [0.12, 0.24, 0.36]

COL_ABS, COL_FL, COL_PH, COL_NR = "#5b3fbf", "#009E73", "#D55E00", "0.45"


def arrow(ax, x, e0, e1, color, ls="-", label=None, dx=0.0, offset=(6, 0)):
    ax.annotate("", xy=(x + dx, e1), xytext=(x, e0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, ls=ls,
                                mutation_scale=9, shrinkA=0, shrinkB=0))
    if label:
        ax.annotate(label, ((2 * x + dx) / 2, (e0 + e1) / 2), xytext=offset,
                    textcoords="offset points", fontsize=6, color=color,
                    va="center")


# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.4, 3.0))

for name, (e, x0, x1) in states.items():
    ax.hlines(e, x0, x1, color="black", lw=1.6)
    for v in VIB:
        ax.hlines(e + v, x0, x1, color="0.6", lw=0.5)
    ax.text(x0 - 0.06, e, name, ha="right", va="center", fontsize=7)

arrow(ax, 0.15, 0.0, 2.9 + 0.24, COL_ABS, label="Absorption", offset=(-6, 0))
ax.texts[-1].set_ha("right")
arrow(ax, 0.30, 0.0, 3.8 + 0.12, COL_ABS)
arrow(ax, 0.62, 3.8 + 0.12, 2.9 + 0.36, COL_NR, ls="--", label="IC", offset=(5, 0))
arrow(ax, 0.62, 2.9 + 0.36, 2.9, COL_NR, ls="--")
arrow(ax, 0.80, 2.9, 0.0 + 0.12, COL_FL)
ax.text(0.86, 2.5, "Fluorescence", color=COL_FL, fontsize=6, va="center")
arrow(ax, 1.0, 2.9, 2.3 + 0.36, COL_NR, ls="--", label="ISC", dx=0.5,
      offset=(0, 7))
arrow(ax, 1.75, 2.3, 0.0 + 0.24, COL_PH, label="Phosphorescence", dx=-0.9,
      offset=(8, 0))

ax.set_xlim(-0.35, 2.7)
ax.set_ylim(-0.2, 4.4)
ax.set_ylabel("Energy (eV)")
ax.set_yticks([0, 1, 2, 3, 4])
ax.spines["bottom"].set_visible(False)
ax.tick_params(bottom=False, labelbottom=False)
ax.text(0.5, 4.25, "Singlet manifold", ha="center", fontsize=6.5, color="0.35")
ax.text(2.0, 4.25, "Triplet manifold", ha="center", fontsize=6.5, color="0.35")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig049_energy_levels.{ext}")
print("saved fig049_energy_levels.png / .pdf")
