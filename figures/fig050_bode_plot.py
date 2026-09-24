"""Fig. 50 - Bode plot: magnitude and phase vs frequency (single column).

The control-engineering standard: two stacked semilog panels sharing the
frequency axis, log-spaced grid, -3 dB bandwidth marked, and two damping
ratios compared. Frequency response from scipy.signal.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

# ------------------------------------------------------------- DATA ----
WN = 100.0                                          # natural frequency, rad/s
w = np.logspace(0, 4, 600)
systems = {r"$\zeta$ = 0.2": 0.2, r"$\zeta$ = 0.7": 0.7}

# ------------------------------------------------------------- PLOT ----
fig, (ax_m, ax_p) = plt.subplots(2, 1, figsize=(3.5, 3.4), sharex=True,
                                 height_ratios=[1.4, 1])

for i, (label, zeta) in enumerate(systems.items()):
    sys = signal.TransferFunction([WN**2], [1, 2 * zeta * WN, WN**2])
    _, mag, phase = signal.bode(sys, w)
    ax_m.semilogx(w, mag, color=f"C{i}", label=label)
    ax_p.semilogx(w, phase, color=f"C{i}")
    if i == 0:                                       # mark -3 dB bandwidth
        w3 = w[np.argmax(mag < -3)]
        ax_m.plot(w3, -3, "o", ms=4, mfc="white", mec=f"C{i}", mew=1.1, zorder=4)
        ax_m.annotate(f"\u22123 dB at {w3:.0f} rad s$^{{-1}}$", (w3, -3),
                      xytext=(8, 8), textcoords="offset points", fontsize=6)

# -40 dB/decade asymptote guide
w_asym = np.array([300, 8000])
ax_m.plot(w_asym, -40 * np.log10(w_asym / WN), ls="--", lw=0.7, color="0.5")
ax_m.text(2500, -40 * np.log10(2500 / WN) + 6, "\u221240 dB/decade",
          fontsize=6, color="0.4", rotation=-38, ha="center")

for ax in (ax_m, ax_p):
    ax.grid(True, which="both", lw=0.4, color="0.9")
    ax.set_axisbelow(True)
ax_m.set_ylabel("Magnitude (dB)")
ax_m.set_ylim(-80, 15)
ax_m.legend(loc="lower left", fontsize=6.5)
ax_p.set_ylabel("Phase (\u00b0)")
ax_p.set_yticks([0, -45, -90, -135, -180])
ax_p.set_xlabel("Frequency (rad s$^{-1}$)")
ax_p.set_xlim(w[0], w[-1])

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig050_bode_plot.{ext}")
print("saved fig050_bode_plot.png / .pdf")
