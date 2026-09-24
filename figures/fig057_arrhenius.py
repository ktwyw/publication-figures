"""Fig. 57 - Arrhenius plot with a secondary temperature axis (single column).

ln k vs 1000/T with a weighted linear fit giving the activation energy,
and - the part people rarely get right - a secondary x-axis on top
showing temperature in degrees Celsius via matplotlib's
secondary_xaxis with an exact forward/inverse transform pair.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(57)
R_GAS = 8.314                                   # J mol^-1 K^-1

# ------------------------------------------------------------- DATA ----
T = np.array([310, 325, 340, 360, 380, 400, 425, 450], float)     # K
EA_TRUE, LNA_TRUE = 78.0e3, 21.5
lnk = LNA_TRUE - EA_TRUE / (R_GAS * T) + rng.normal(0, 0.10, T.size)
lnk_err = np.full(T.size, 0.12)

x = 1000.0 / T
res = stats.linregress(x, lnk)
ea = -res.slope * R_GAS * 1000.0                 # slope is per (1000/T)
ea_err = res.stderr * R_GAS * 1000.0

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.4, 2.8))
xf = np.linspace(x.min() - 0.08, x.max() + 0.08, 100)
ax.plot(xf, res.intercept + res.slope * xf, color="C1", lw=1.1,
        label="Linear fit")
ax.errorbar(x, lnk, yerr=lnk_err, fmt="o", ms=4, mfc="white", color="C0",
            capsize=2, label="Measured")

ax.set_xlabel(r"$1000/T$ (K$^{-1}$)")
ax.set_ylabel(r"$\ln k$  ($k$ in s$^{-1}$)")
ax.text(0.04, 0.06,
        rf"$E_a$ = {ea/1e3:.1f} $\pm$ {ea_err/1e3:.1f} kJ mol$^{{-1}}$"
        "\n"
        rf"$\ln A$ = {res.intercept:.1f}, $R^2$ = {res.rvalue**2:.3f}",
        transform=ax.transAxes, va="bottom", fontsize=6.5)
ax.legend(loc="upper right", fontsize=6)

# secondary axis: temperature in Celsius (exact transform pair)
secax = ax.secondary_xaxis(
    "top", functions=(lambda invT: 1000.0 / invT - 273.15,
                      lambda tc: 1000.0 / (tc + 273.15)))
secax.set_xticks([40, 60, 80, 100, 130, 160])
secax.set_xlabel("Temperature (\u00b0C)")
secax.tick_params(labelsize=7)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig057_arrhenius.{ext}")
print("saved fig057_arrhenius.png / .pdf")
