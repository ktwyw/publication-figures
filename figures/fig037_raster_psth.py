"""Fig. 37 - Spike raster + peri-stimulus time histogram (single column).

The canonical single-neuron figure: one row of ticks per trial via
ax.eventplot, a PSTH in Hz below it, a Gaussian-smoothed rate estimate,
and the stimulus window shaded across both panels.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(37)

# ------------------------------------------------------------- DATA ----
N_TRIALS, DT = 40, 0.001
t = np.arange(-0.5, 1.5, DT)
rate = 5 + (t >= 0) * (30 * np.exp(-t / 0.3) + 8 * (t < 0.5))   # Hz
spikes = [t[rng.uniform(size=t.size) < rate * DT] for _ in range(N_TRIALS)]

BIN = 0.02
edges = np.arange(-0.5, 1.5 + 1e-9, BIN)
counts, _ = np.histogram(np.concatenate(spikes), bins=edges)
psth = counts / (N_TRIALS * BIN)
smooth = gaussian_filter1d(psth, sigma=1.5)                     # 30 ms kernel

# ------------------------------------------------------------- PLOT ----
fig, (ax_r, ax_p) = plt.subplots(2, 1, figsize=(3.5, 3.0), sharex=True,
                                 height_ratios=[2, 1])
for ax in (ax_r, ax_p):
    ax.axvspan(0, 0.5, color="C1", alpha=0.12, lw=0, zorder=0)

ax_r.eventplot(spikes, colors="black", linelengths=0.8, linewidths=0.5)
ax_r.set_ylim(-0.5, N_TRIALS - 0.5)
ax_r.set_yticks([0, 19, 39], ["1", "20", "40"])
ax_r.set_ylabel("Trial")
ax_r.text(0.25, 1.02, "Stimulus", transform=ax_r.get_xaxis_transform(),
          ha="center", va="bottom", fontsize=6.5, color="C1")

ax_p.bar(edges[:-1], psth, width=BIN, align="edge", color="0.65", lw=0)
ax_p.plot(edges[:-1] + BIN / 2, smooth, color="C0", lw=1.2)
ax_p.set_xlim(t[0], t[-1])
ax_p.set_ylabel("Rate (Hz)")
ax_p.set_xlabel("Time from stimulus onset (s)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig037_raster_psth.{ext}")
print("saved fig037_raster_psth.png / .pdf")
