"""Fig. 9 - Spectrum with annotated peaks and a zoom inset (single column).

The inset uses ax.inset_axes + ax.indicate_inset_zoom, which draws the
zoom rectangle and connector lines for you - useful for resolving
doublets, shoulders, or any crowded region.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(12)

# --------------------------------------------------- SYNTHETIC DATA ----
x = np.linspace(400, 1800, 2800)
peaks = [(621, 5, 0.18), (795, 7, 0.06), (1002, 4, 1.00), (1032, 5, 0.28),
         (1155, 6, 0.10), (1450, 8, 0.22), (1583, 6, 0.16), (1602, 5, 0.30)]
y_clean = 0.03 + 2e-5 * (x - 400)                     # sloping baseline
for c, w, a in peaks:
    y_clean += a * w**2 / ((x - c) ** 2 + w**2)       # Lorentzians
y = y_clean + rng.normal(0, 0.006, x.size)

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.6, 2.7))
ax.plot(x, y, color="C0", lw=1.0)
ax.set_xlim(400, 1800)
ax.set_ylim(-0.02, 1.32)
ax.set_xlabel("Raman shift (cm$^{-1}$)")
ax.set_ylabel("Intensity (a.u.)")

# annotate selected peaks with rotated wavenumber labels
for c in (621, 1002, 1032, 1450):
    ax.text(c, np.interp(c, x, y_clean) + 0.04, str(c), rotation=90,
            ha="center", va="bottom", fontsize=6, color="0.3")

# zoom inset resolving the 1583/1602 doublet
axins = ax.inset_axes([0.06, 0.52, 0.30, 0.44])
axins.plot(x, y, color="C0", lw=1.0)
axins.set_xlim(1550, 1640)
axins.set_ylim(0.0, 0.42)
axins.set_xticks([1560, 1600, 1640])
axins.tick_params(labelsize=6)
for c in (1583, 1602):
    axins.text(c, np.interp(c, x, y_clean) + 0.02, str(c), rotation=90,
               ha="center", va="bottom", fontsize=5.5, color="0.3")
ax.indicate_inset_zoom(axins, edgecolor="0.35", lw=0.7)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig009_spectrum_inset.{ext}")
print("saved fig009_spectrum_inset.png / .pdf")
