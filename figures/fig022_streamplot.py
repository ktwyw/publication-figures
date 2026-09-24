"""Fig. 22 - Vector field as streamlines (single column).

Potential flow past a cylinder: streamlines coloured by local speed,
with the solid body masked out via NaNs (streamplot skips them) and a
patch drawn on top. The same pattern works for any 2-D vector field.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

# ------------------------------------------------------- VECTOR FIELD ----
R, U = 1.0, 1.0
x = np.linspace(-3, 3, 240)
y = np.linspace(-2.2, 2.2, 180)
X, Y = np.meshgrid(x, y)
r2 = X**2 + Y**2

u = U * (1 - R**2 * (X**2 - Y**2) / r2**2)
v = -U * 2 * R**2 * X * Y / r2**2
inside = r2 < R**2
u[inside] = v[inside] = np.nan
speed = np.hypot(u, v)

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.6, 2.9))
strm = ax.streamplot(x, y, u, v, color=speed, cmap="viridis",
                     density=1.15, linewidth=0.9, arrowsize=0.7)
ax.add_patch(Circle((0, 0), R, facecolor="0.88", edgecolor="black",
                    lw=0.8, zorder=3))

ax.set_aspect("equal")
ax.set_xlim(x[0], x[-1])
ax.set_ylim(y[0], y[-1])
ax.set_xlabel("$x / R$")
ax.set_ylabel("$y / R$")
# inset cax in axes-fraction coords tracks the aspect-locked axes exactly
cax = ax.inset_axes([1.03, 0.0, 0.04, 1.0])
cbar = fig.colorbar(strm.lines, cax=cax)
cbar.set_label(r"Speed / $U_\infty$")
cbar.outline.set_linewidth(0.8)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig022_streamplot.{ext}")
print("saved fig022_streamplot.png / .pdf")
