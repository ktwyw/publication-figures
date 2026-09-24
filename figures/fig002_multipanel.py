"""Fig. 2 - Multi-panel composite (double column, 183 mm) with a-d labels.

(a) time courses with s.e.m. bands   (b) scatter encoded by a 3rd variable
(c) overlaid distributions           (d) box plots with raw-data overlay
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(3)
OKABE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def panel_label(ax, letter):
    """Bold lowercase panel letter, Nature style."""
    ax.text(-0.16, 1.02, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="right")


fig, axs = plt.subplots(2, 2, figsize=(7.1, 5.2))
ax_a, ax_b, ax_c, ax_d = axs.flat

# ---- (a) time courses: mean line + s.e.m. band -------------------------
t = np.linspace(0, 48, 25)
for label, tau in [("Control", 30), ("Treatment A", 18), ("Treatment B", 11)]:
    mean = 100 * np.exp(-t / tau) * (1 + rng.normal(0, 0.015, t.size))
    sem = 1.2 + 0.03 * mean
    (line,) = ax_a.plot(t, mean, label=label)
    ax_a.fill_between(t, mean - sem, mean + sem,
                      color=line.get_color(), alpha=0.25, lw=0)
ax_a.set_xlabel("Time (h)")
ax_a.set_ylabel("Fluorescence (% of initial)")
ax_a.set_xlim(0, 48)
ax_a.set_ylim(0, 108)
ax_a.legend()

# ---- (b) scatter coloured by a third variable --------------------------
conc = rng.uniform(0, 10, 90)
temp = rng.uniform(20, 60, 90)
rate = 0.85 * conc * (1 + 0.012 * (temp - 20)) + rng.normal(0, 0.7, 90)
sc = ax_b.scatter(conc, rate, c=temp, s=14, cmap="viridis", lw=0)
cbar = fig.colorbar(sc, ax=ax_b, pad=0.02)
cbar.set_label("Temperature (\u00b0C)")
ax_b.set_xlabel("Substrate (\u00b5M)")
ax_b.set_ylabel("Rate (s$^{-1}$)")

# ---- (c) overlaid distributions ----------------------------------------
wt = rng.normal(0.0, 1.0, 400)
mut = rng.normal(1.3, 1.25, 400)
bins = np.linspace(-4, 6, 34)
for data, label, color in [(wt, "Wild type", OKABE[0]),
                           (mut, "Mutant", OKABE[1])]:
    ax_c.hist(data, bins, density=True, histtype="stepfilled",
              alpha=0.35, color=color, lw=0)
    ax_c.hist(data, bins, density=True, histtype="step",
              color=color, lw=1.1, label=label)
ax_c.set_xlabel("Expression (log$_2$ fold change)")
ax_c.set_ylabel("Probability density")
ax_c.legend()

# ---- (d) box plots with raw data overlay -------------------------------
groups = [rng.normal(m, s, 28) for m, s in [(2.1, 0.45), (3.0, 0.55), (2.55, 0.5)]]
ax_d.boxplot(groups, widths=0.55, showfliers=False, patch_artist=True,
             boxprops=dict(facecolor="0.92", lw=0.8),
             medianprops=dict(color="black", lw=1.1),
             whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
for i, g in enumerate(groups, start=1):
    ax_d.scatter(i + rng.uniform(-0.14, 0.14, g.size), g, s=9,
                 color=OKABE[i - 1], alpha=0.75, lw=0, zorder=3)
ax_d.set_xticks([1, 2, 3], ["Control", "Treatment A", "Treatment B"])
ax_d.set_ylabel("Vesicle diameter (\u00b5m)")

for ax, letter in zip(axs.flat, "abcd"):
    panel_label(ax, letter)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig002_multipanel.{ext}")
print("saved fig002_multipanel.png / .pdf")
