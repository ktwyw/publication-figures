"""Fig. 56 - N2 physisorption: isotherm + pore-size distribution.

(a) Type-IV isotherm with H1 hysteresis: closed symbols = adsorption,
open symbols = desorption (the IUPAC convention), branch arrows.
(b) BJH pore-size distribution on a log-diameter axis.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(56)


def sigmoid(x, x0, w):
    return 1.0 / (1.0 + np.exp(-(x - x0) / w))


# ------------------------------------------------------------- DATA ----
p = np.concatenate([np.linspace(0.005, 0.4, 18), np.linspace(0.43, 0.995, 22)])
QM, C = 145.0, 80.0                                     # monolayer capacity
mono = QM * C * p / (1 + (C - 1) * p)                   # Langmuir-type knee
multi = 30.0 * p**2 / (1 - 0.75 * p)                    # bounded multilayer
cond = 150.0                                            # capillary step size
base = mono + multi
q_ads = base + cond * sigmoid(p, 0.72, 0.035) + rng.normal(0, 2.0, p.size)
q_des = base + cond * sigmoid(p, 0.62, 0.030) + rng.normal(0, 2.0, p.size)
q_des = np.where(p < 0.45, q_ads, q_des)                # loop closes at ~0.45

d = np.logspace(np.log10(2), np.log10(60), 120)         # pore diameter, nm
psd = 1.9 * np.exp(-(np.log(d / 8.2) / 0.28) ** 2) \
    + 0.12 * np.exp(-(np.log(d / 30) / 0.5) ** 2) \
    + np.abs(rng.normal(0, 0.02, d.size))

# ------------------------------------------------------------- PLOT ----
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.1, 2.7),
                                 width_ratios=[1.15, 1])

ax_a.plot(p, q_ads, "o-", ms=3.2, lw=0.8, color="C0",
          label="Adsorption")
des = p >= 0.45                                        # loop region only
ax_a.plot(p[des], q_des[des], "o-", ms=3.2, lw=0.8, color="C0", mfc="white",
          label="Desorption")
ax_a.annotate("", xy=(0.88, 398), xytext=(0.79, 362),
              arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.4"))
ax_a.annotate("", xy=(0.565, 195), xytext=(0.655, 285),
              arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.4"))
ax_a.set_xlim(0, 1.0)
ax_a.set_ylim(0, 440)
ax_a.set_xlabel(r"Relative pressure, $P/P_0$")
ax_a.set_ylabel(r"Quantity adsorbed (cm$^3$ g$^{-1}$ STP)")
ax_a.legend(loc="upper left", fontsize=6)
ax_a.text(0.97, 0.05, "77 K, N$_2$", transform=ax_a.transAxes,
          ha="right", fontsize=6.5)

ax_b.plot(d, psd, color="C1", lw=1.2)
ax_b.fill_between(d, 0, psd, color="C1", alpha=0.2, lw=0)
ax_b.set_xscale("log")
ax_b.set_xlim(2, 60)
ax_b.set_ylim(0, 2.2)
ax_b.set_xticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
ax_b.set_xlabel("Pore diameter (nm)")
ax_b.set_ylabel(r"d$V$/dlog$D$ (cm$^3$ g$^{-1}$)")
peak_d = d[np.argmax(psd)]
ax_b.annotate(f"{peak_d:.1f} nm", (peak_d, psd.max()), xytext=(8, -2),
              textcoords="offset points", fontsize=6.5)

for ax, letter in zip((ax_a, ax_b), "ab"):
    ax.text(-0.16, 1.02, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="right")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig056_isotherm_psd.{ext}")
print("saved fig056_isotherm_psd.png / .pdf")
