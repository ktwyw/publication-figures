"""Fig. 36 - Ternary diagram from scratch (single column).

Three-component compositions (soil texture, alloys, mixtures) mapped to
an equilateral triangle: grid lines, edge tick labels, rotated axis
titles, and samples coloured by a fourth variable with an inset colorbar.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(36)
SQ3 = np.sqrt(3)


def tern(a, b, c):
    """Barycentric (a, b, c) -> Cartesian; a at bottom-left, b bottom-right, c top."""
    a, b, c = map(np.asarray, (a, b, c))
    s = a + b + c
    return 0.5 * (2 * b + c) / s, SQ3 / 2 * c / s


# ------------------------------------------------------------- DATA ----
comp = rng.dirichlet([2.5, 3.0, 2.0], 70)            # silt, sand, clay
silt, sand, clay = comp.T
organic = 0.8 + 6 * clay + rng.normal(0, 0.4, 70)    # organic carbon, %

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.6, 3.3))
ax.set_aspect("equal")
ax.axis("off")

tri = np.array([tern(1, 0, 0), tern(0, 1, 0), tern(0, 0, 1), tern(1, 0, 0)])
ax.plot(tri[:, 0], tri[:, 1], color="black", lw=0.8, zorder=2)

for t in np.arange(0.2, 0.99, 0.2):
    for p, q in [((t, 1 - t, 0), (t, 0, 1 - t)),        # constant silt
                 ((1 - t, t, 0), (0, t, 1 - t)),        # constant sand
                 ((1 - t, 0, t), (0, 1 - t, t))]:       # constant clay
        (x0, y0), (x1, y1) = tern(*p), tern(*q)
        ax.plot([x0, x1], [y0, y1], color="0.82", lw=0.5, zorder=0)
    lab = f"{100 * t:.0f}"
    xb, yb = tern(1 - t, t, 0)          # sand ticks along the bottom edge
    ax.text(xb, yb - 0.025, lab, ha="center", va="top", fontsize=6, color="0.35")
    xr, yr = tern(0, 1 - t, t)          # clay ticks along the right edge
    ax.text(xr + 0.02, yr, lab, ha="left", va="center", fontsize=6, color="0.35")
    xl, yl = tern(t, 0, 1 - t)          # silt ticks along the left edge
    ax.text(xl - 0.02, yl, lab, ha="right", va="center", fontsize=6, color="0.35")

ax.text(0.5, -0.11, "Sand (%)", ha="center", va="top", fontsize=7)
ax.text(0.88, 0.50, "Clay (%)", rotation=-60, ha="center", va="center", fontsize=7)
ax.text(0.12, 0.50, "Silt (%)", rotation=60, ha="center", va="center", fontsize=7)

x, y = tern(silt, sand, clay)
sc = ax.scatter(x, y, c=organic, s=16, cmap="viridis", edgecolor="white",
                lw=0.3, zorder=3)
cax = ax.inset_axes([0.96, 0.25, 0.035, 0.5])
cbar = fig.colorbar(sc, cax=cax)
cbar.set_label("Organic C (%)")
cbar.outline.set_linewidth(0.8)

ax.set_xlim(-0.15, 1.15)
ax.set_ylim(-0.17, 0.95)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig036_ternary.{ext}")
print("saved fig036_ternary.png / .pdf")
