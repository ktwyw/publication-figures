"""Fig. 52 - Thermal analysis: TGA + DSC on a twin axis (single column).

Dual y-axes are risky; the readable version colour-codes everything:
each spine, tick set, and axis label matches its curve, so no legend is
needed. Mass-loss steps are annotated with double-headed arrows and the
DSC axis carries the mandatory "exo up" marker.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(52)


def sigmoid(x, x0, w):
    return 1.0 / (1.0 + np.exp(-(x - x0) / w))


# ------------------------------------------------------------- DATA ----
T = np.linspace(30, 800, 900)
step1, step2 = 5.1, 39.8                              # % mass losses
mass = (100 - step1 * sigmoid(T, 105, 12) - step2 * sigmoid(T, 355, 18)
        - 0.002 * T + rng.normal(0, 0.06, T.size))
heat = (-0.9 * np.exp(-((T - 108) / 22) ** 2)          # endothermic dehydration
        + 2.6 * np.exp(-((T - 362) / 20) ** 2)         # exothermic decomposition
        + 0.0006 * T + rng.normal(0, 0.03, T.size))

C_TGA, C_DSC = "C0", "C1"

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax2 = ax.twinx()

ax.plot(T, mass, color=C_TGA, lw=1.2)
ax2.plot(T, heat, color=C_DSC, lw=1.0)

# colour-code the two y-axes
ax.set_ylabel("Mass (%)", color=C_TGA)
ax.tick_params(axis="y", colors=C_TGA)
ax.spines["left"].set_color(C_TGA)
ax2.set_ylabel("Heat flow (W g$^{-1}$)", color=C_DSC)
ax2.tick_params(axis="y", colors=C_DSC)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color(C_DSC)
ax2.spines["left"].set_visible(False)

# annotate the two mass-loss steps
for T0, before, loss in [(160, 100 - 0.3, step1), (450, 94.4, step2)]:
    ax.annotate("", xy=(T0, before - loss), xytext=(T0, before),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=C_TGA))
    ax.text(T0 + 12, before - loss / 2, f"\u2212{loss:.1f}%", va="center",
            fontsize=6.5, color=C_TGA)

ax2.text(0.985, 0.97, "exo $\\uparrow$", transform=ax2.transAxes,
         ha="right", va="top", fontsize=6.5, color=C_DSC)
ax.set_xlabel("Temperature (\u00b0C)")
ax.set_xlim(30, 800)
ax.set_ylim(48, 103)
ax2.set_ylim(-1.6, 3.4)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig052_tga_dsc.{ext}")
print("saved fig052_tga_dsc.png / .pdf")
