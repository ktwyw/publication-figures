"""Fig. 42 - Split violins: two conditions per category (single column).

Each violin is cut down the middle - control on the left half, treated
on the right - so the two distributions share a baseline for direct
comparison. Medians are drawn as short white bars inside each half.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(42)

# ------------------------------------------------------------- DATA ----
cell_types = ["Naive", "Memory", "Effector", "Treg", "NK"]
shift = [0.05, 0.35, 0.8, -0.1, 0.5]                  # treatment effect
control = [rng.lognormal(2.0, 0.35, 70) for _ in cell_types]
treated = [rng.lognormal(2.0 + s, 0.35, 70) for s in shift]


def half_violin(ax, data, pos, side, color):
    parts = ax.violinplot([data], positions=[pos], widths=0.85,
                          showextrema=False)
    body = parts["bodies"][0]
    verts = body.get_paths()[0].vertices
    if side == "left":
        verts[:, 0] = np.clip(verts[:, 0], -np.inf, pos)
    else:
        verts[:, 0] = np.clip(verts[:, 0], pos, np.inf)
    body.set_facecolor(color)
    body.set_edgecolor("none")
    body.set_alpha(0.8)
    med = np.median(data)
    x0, x1 = (pos - 0.30, pos - 0.03) if side == "left" else (pos + 0.03, pos + 0.30)
    ax.hlines(med, x0, x1, color="white", lw=1.4, zorder=3)


# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.6, 2.7))
pos = np.arange(len(cell_types))
for p, c, t in zip(pos, control, treated):
    half_violin(ax, c, p, "left", "C0")
    half_violin(ax, t, p, "right", "C1")

ax.set_xticks(pos, cell_types)
ax.set_xlim(-0.6, len(cell_types) - 0.4)
ax.set_ylabel("Marker expression (MFI, a.u.)")
ax.legend(handles=[Patch(color="C0", label="Control"),
                   Patch(color="C1", label="Treated")],
          loc="upper left", fontsize=6.5)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig042_split_violin.{ext}")
print("saved fig042_split_violin.png / .pdf")
