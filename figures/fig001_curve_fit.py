"""Fig. 1 - Data + model fit with error bars (single column, 89 mm).

Michaelis-Menten kinetics as a stand-in for any "measurements + fitted
model" figure. Replace the DATA block with your own values.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))


def model(s, vmax, km):
    """Michaelis-Menten rate law."""
    return vmax * s / (km + s)


# ------------------------------------------------------------- DATA ----
rng = np.random.default_rng(7)
S = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])  # substrate (mM)
replicates = model(S, 10.2, 3.1) + rng.normal(0.0, 0.35, (3, S.size))
v_mean = replicates.mean(axis=0)
v_sd = replicates.std(axis=0, ddof=1)

# -------------------------------------------------------------- FIT ----
popt, pcov = curve_fit(model, S, v_mean, sigma=v_sd,
                       absolute_sigma=True, p0=(8.0, 2.0))
perr = np.sqrt(np.diag(pcov))
s_fine = np.linspace(0.0, 33.0, 400)

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.5, 2.6))

ax.plot(s_fine, model(s_fine, *popt), color="C1", zorder=2,
        label="Michaelis\u2013Menten fit")
ax.errorbar(S, v_mean, yerr=v_sd, fmt="o", color="C0", mfc="white",
            zorder=3, label="Measured (mean \u00b1 s.d., $n$ = 3)")

ax.set_xlabel("Substrate concentration (mM)")
ax.set_ylabel("Initial rate $v_0$ (\u00b5mol min$^{-1}$)")
ax.set_xlim(0, 33)
ax.set_ylim(0, 11.5)

fit_text = (rf"$V_{{\max}}$ = {popt[0]:.1f} ± {perr[0]:.1f} µmol min$^{{-1}}$"
            "\n"
            rf"$K_\mathrm{{M}}$ = {popt[1]:.1f} ± {perr[1]:.1f} mM")
ax.text(0.97, 0.05, fit_text, transform=ax.transAxes, ha="right", va="bottom")
ax.legend(loc="upper left")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig001_curve_fit.{ext}")
print("saved fig001_curve_fit.png / .pdf")
