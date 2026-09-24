"""Fig. 51 - Powder XRD patterns, stacked with reference sticks (single column).

The materials-paper staple: offset-stacked diffractograms for sample
comparison, a stick pattern for the reference phase along the baseline,
and (hkl) indices over the peaks. Intensities in arbitrary units, so the
y-axis keeps its label but drops its ticks, as is conventional.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(51)

# ---------------------------------------------- anatase-like reference ----
two_theta = np.linspace(10, 80, 3500)
peaks = [(25.3, 100, "101"), (37.8, 22, "004"), (48.0, 30, "200"),
         (53.9, 19, "105"), (55.1, 18, "211"), (62.7, 14, "204"),
         (68.8, 6, "116"), (70.3, 6, "220"), (75.0, 10, "215")]


def pattern(fwhm, scale, noise):
    y = 3 + 18 * np.exp(-two_theta / 25)                 # amorphous background
    for pos, inten, _ in peaks:
        w = fwhm / 2
        y += scale * inten * w**2 / ((two_theta - pos) ** 2 + w**2)
    return y + rng.normal(0, noise, two_theta.size)


samples = [("As-synthesized", pattern(0.55, 0.55, 1.2), 0),
           ("Calcined 450 \u00b0C", pattern(0.28, 1.00, 1.2), 90)]

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.5, 3.0))

for pos, inten, _ in peaks:                              # reference sticks
    ax.vlines(pos, -34, -34 + 0.28 * inten, color="0.35", lw=1.0)
ax.text(79, -30, "Reference\n(anatase)", ha="right", va="bottom",
        fontsize=6, color="0.35")

for label, y, off in samples:
    ax.plot(two_theta, y + off, lw=0.7, color="C0" if off == 0 else "C1")
    ax.text(11, off + 32, label, ha="left", fontsize=6.5)

top = samples[-1][1] + samples[-1][2]
stagger = {"105": 10, "211": 26, "220": 14}                          # avoid label clash
for pos, inten, hkl in peaks:
    ax.text(pos, np.max(top[np.abs(two_theta - pos) < 0.5]) + 6
            + stagger.get(hkl, 0),
            hkl, rotation=90, ha="center", va="bottom", fontsize=5.5,
            color="0.25")

ax.set_xlim(10, 80)
ax.set_ylim(-38, 305)
ax.set_yticks([])
ax.set_xlabel(r"2$\theta$ (degree, Cu K$\alpha$)")
ax.set_ylabel("Intensity (a.u.)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig051_xrd.{ext}")
print("saved fig051_xrd.png / .pdf")
