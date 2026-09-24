"""Fig. 11 - Microscopy-style image with scale bar (single column).

Publication conventions demonstrated here:
- no axis ticks; a scale bar instead (state its length in the caption too)
- interpolation="nearest" so pixels are shown as recorded
- contrast set by percentile stretch, with the colorbar reporting the
  true recorded intensities (never silently clip without saying so)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(10)

# ------------------------------------------ SYNTHETIC "MICROGRAPH" ----
SIZE = 512
PX_UM = 0.2                                   # pixel size: 0.2 um/px
yy, xx = np.mgrid[0:SIZE, 0:SIZE]
lam = 30 + 0.015 * xx                          # gentle illumination gradient
for _ in range(28):                            # fluorescent "cells"
    x0, y0 = rng.uniform(20, SIZE - 20, 2)
    amp, sig = rng.uniform(80, 220), rng.uniform(5, 12)
    lam = lam + amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sig**2))
img = rng.poisson(lam).astype(float)           # shot noise

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.3, 2.9))
vmax = np.percentile(img, 99.5)                # percentile contrast stretch
im = ax.imshow(img, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
ax.set_axis_off()

# scale bar: 20 um = 100 px at 0.2 um/px
bar_px = 20 / PX_UM
x0, y0 = 20, SIZE - 22
ax.plot([x0, x0 + bar_px], [y0, y0], color="white", lw=2.5,
        solid_capstyle="butt")
ax.text(x0 + bar_px / 2, y0 - 10, "20 µm", color="white",
        ha="center", va="bottom", fontsize=7)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label("Photon counts")
cbar.outline.set_linewidth(0.8)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig011_image_scalebar.{ext}")
print("saved fig011_image_scalebar.png / .pdf")
