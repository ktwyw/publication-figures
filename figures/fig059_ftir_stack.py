"""Fig. 59 - FTIR spectra, stacked with band assignments (single column).

Infrared conventions handled properly: wavenumber axis reversed
(4000 -> 400), transmittance stacked with offsets for comparison,
dotted guide lines at diagnostic bands, and rotated assignment labels
above the axes.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(59)

wn = np.linspace(4000, 400, 3200)


def dip(center, width, depth):
    return depth * np.exp(-((wn - center) / width) ** 2)


# --------------------------------------------------- three samples ----
common = [(2922, 30, 14), (2853, 24, 9), (1455, 22, 10), (1160, 40, 8),
          (720, 15, 5)]
samples = [
    ("Neat polymer", common + [(1715, 20, 6)]),
    ("+ 5% filler", common + [(1715, 20, 8), (1060, 45, 22), (470, 25, 12)]),
    ("Crosslinked", common + [(3350, 160, 16), (1715, 20, 24), (1600, 22, 9),
                              (1240, 30, 14)]),
]

bands = [(3350, r"$\nu$(O$-$H)"), (2920, r"$\nu$(C$-$H)"),
         (1715, r"$\nu$(C$=$O)"), (1600, r"$\nu$(C$=$C)"),
         (1240, r"$\nu$(C$-$O)"), (1060, r"$\nu$(Si$-$O)")]

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.6, 3.0))

for center, _ in bands:
    ax.axvline(center, color="0.88", ls=":", lw=0.7, zorder=0)

for i, (label, peaks) in enumerate(samples):
    y = 97 - sum(dip(*p) for p in peaks) + rng.normal(0, 0.35, wn.size)
    offset = 30 * i
    ax.plot(wn, y + offset, lw=0.7, color=f"C{i}")
    ax.text(430, 99 + offset, label, ha="right", va="bottom", fontsize=6.5,
            color=f"C{i}")

for center, name in bands:
    ax.text(center, 1.01, name, transform=ax.get_xaxis_transform(),
            rotation=60, ha="left", va="bottom", fontsize=5.5, color="0.3")

ax.set_xlim(4000, 400)                       # reversed, IR convention
ax.set_ylim(35, 172)
ax.set_yticks([])
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Transmittance (%, offset)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig059_ftir_stack.{ext}")
print("saved fig059_ftir_stack.png / .pdf")
