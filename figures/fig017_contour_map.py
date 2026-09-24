"""Fig. 17 - 2-D parameter map: filled + labelled contours (single column).

The pattern for response surfaces, phase diagrams, and optimisation
landscapes: contourf for the field, a few labelled contour lines for
quantitative reading, the sampled design points, and the optimum marked.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

# ---------------------------------------------------- MODEL SURFACE ----
T = np.linspace(40, 90, 220)
pH = np.linspace(5.0, 9.5, 220)
TT, PP = np.meshgrid(T, pH)
T_opt, pH_opt = 63.0, 7.4
yield_ = 4 + 90 * np.exp(-((TT - T_opt) / 14) ** 2
                         - ((PP - pH_opt) / 0.9) ** 2)

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.4, 2.7))
cf = ax.contourf(TT, PP, yield_, levels=np.linspace(0, 96, 13),
                 cmap="viridis")
cs = ax.contour(TT, PP, yield_, levels=[20, 40, 60, 80],
                colors="white", linewidths=0.8, alpha=0.9)
ax.clabel(cs, fmt="%d", fontsize=6, inline=True)

# sampled design points
Tg, Pg = np.meshgrid(np.linspace(45, 85, 6), np.linspace(5.5, 9.0, 5))
ax.scatter(Tg, Pg, marker="+", s=16, color="white", lw=0.8, alpha=0.7)

# optimum
ax.plot(T_opt, pH_opt, marker="*", ms=9, color="white",
        markeredgecolor="black", markeredgewidth=0.5)
ax.annotate("optimum", (T_opt, pH_opt), xytext=(8, 6),
            textcoords="offset points", fontsize=6, color="0.1")

ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("pH")
cbar = fig.colorbar(cf, ax=ax, pad=0.02)
cbar.set_label("Yield (%)")
cbar.outline.set_linewidth(0.8)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig017_contour_map.{ext}")
print("saved fig017_contour_map.png / .pdf")
