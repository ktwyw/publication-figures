"""Fig. 32 - Dose-response curves with 4-parameter logistic fits (single column).

The pharmacology staple: log-spaced concentrations, replicate error bars,
4PL fits done in log10 space for numerical stability, and EC50 values
read off the fit and dropped to the axis.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(32)


def four_pl(logx, bottom, top, log_ec50, hill):
    return bottom + (top - bottom) / (1 + 10 ** ((log_ec50 - logx) * hill))


# ------------------------------------------------------------- DATA ----
conc = np.logspace(-2, 2, 9)                       # µM
compounds = {"Compound A": (0.3, 1.1, 98),          # EC50, Hill slope, top
             "Compound B": (2.0, 0.9, 92),
             "Compound C": (15.0, 1.3, 84)}

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.5, 2.7))
x_fit = np.logspace(-2.4, 2.4, 300)

for i, (name, (ec50, hill, top)) in enumerate(compounds.items()):
    truth = four_pl(np.log10(conc), 2, top, np.log10(ec50), hill)
    reps = truth + rng.normal(0, 4, (3, conc.size))
    mean, sd = reps.mean(0), reps.std(0, ddof=1)

    popt, _ = curve_fit(four_pl, np.log10(conc), mean, sigma=sd,
                        p0=(0, 100, 0, 1), maxfev=5000)
    ec50_fit = 10 ** popt[2]

    ax.errorbar(conc, mean, yerr=sd, fmt="o", ms=3.5, color=f"C{i}",
                capsize=2, lw=0.9)
    ax.plot(x_fit, four_pl(np.log10(x_fit), *popt), color=f"C{i}",
            label=f"{name} (EC$_{{50}}$ = {ec50_fit:.2g} µM)")
    ax.plot([ec50_fit, ec50_fit], [-5, four_pl(popt[2], *popt)],
            ls=":", lw=0.7, color=f"C{i}")

ax.set_xscale("log")
ax.set_xlim(x_fit[0], x_fit[-1])
ax.set_ylim(-5, 110)
ax.set_xlabel("Concentration (µM)")
ax.set_ylabel("Response (% of control)")
ax.legend(loc="upper left", fontsize=6)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig032_dose_response.{ext}")
print("saved fig032_dose_response.png / .pdf")
