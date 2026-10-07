"""Fig. 110 - Dose-response fits with IC50 intervals (single column, 89 mm).

Extends fig032, which fits the means and quotes a point EC50: here the
four-parameter logistic is fitted to all replicate values with log IC50
as the fitted parameter, so each IC50 is reported with a 95% interval
from the covariance matrix. Symbols are mean ± s.d. with a distinct
marker per compound, and the concentration axis carries plain decimal
ticks. The self-check is that each fitted curve evaluated at its own
IC50 equals the midpoint (bottom + top)/2, that the IC50 used to
simulate each compound lies inside its fitted interval, and that the
fitted potencies keep the simulated order.

Statistics: n = 3 independent experiments per concentration; symbols,
mean; error bars, s.d.; curves, least-squares four-parameter logistic
fits to all 27 values per compound; interval, log IC50 ± 1.96 s.e.
All data are simulated.
"""

import numpy as np
from scipy.optimize import curve_fit

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(1010)
CONC_UM = 10.0 ** np.arange(-3.0, 1.01, 0.5)    # 1 nM to 10 µM, half-log
N_EXPERIMENTS = 3
# name: (true IC50 in µM, Hill slope, colour, marker)
COMPOUNDS = {"Compound 1": (0.62, 1.1, ms.GREY, "s"),
             "Compound 2": (0.21, 1.3, ms.GREEN, "^"),
             "Compound 3": (0.031, 1.2, ms.VERMILLION, "o")}
HERO = "Compound 3"


def four_pl(conc, bottom, top, log_ic50, hill):
    """Four-parameter logistic, decreasing in concentration."""
    return bottom + (top - bottom) / (1.0 + (conc / 10.0 ** log_ic50) ** hill)


# ------------------------------------------------------------- DATA ----
# viability (% of control): one row per experiment, one column per dose
viability = {}
for name, (ic50, hill, _colour, _marker) in COMPOUNDS.items():
    viability[name] = np.array([
        four_pl(CONC_UM, 6.0, 100.0, np.log10(ic50), hill)
        + rng.normal(0, 4.5, CONC_UM.size) for _ in range(N_EXPERIMENTS)])
assert np.all(CONC_UM > 0), "a log axis needs positive concentrations"

# -------------------------------------------------------------- FITS ---
fits = {}
for name, response in viability.items():
    params, covariance = curve_fit(
        four_pl, np.tile(CONC_UM, N_EXPERIMENTS), response.ravel(),
        p0=[5.0, 100.0, -1.0, 1.0], maxfev=10000)
    se_log = np.sqrt(covariance[2, 2])
    fits[name] = dict(
        params=params, ic50_nm=10.0 ** params[2] * 1000,
        ci_nm=10.0 ** (params[2] + np.array([-1.96, 1.96]) * se_log) * 1000)

# ------------------------------------------------------- SELF-CHECK ---
for name, f in fits.items():
    bottom, top, log_ic50, _hill = f["params"]
    midpoint = four_pl(10.0 ** log_ic50, *f["params"])
    assert abs(midpoint - (bottom + top) / 2) < 1e-9, (name, midpoint)
    true_nm = COMPOUNDS[name][0] * 1000
    assert f["ci_nm"][0] < true_nm < f["ci_nm"][1], (name, f["ci_nm"])
fitted_order = sorted(fits, key=lambda name: fits[name]["ic50_nm"])
assert fitted_order == sorted(COMPOUNDS, key=lambda name: COMPOUNDS[name][0])
print("fig110: self-check passed (curve at IC50 = midpoint; true IC50 "
      "inside 95% CI: " + ", ".join(
          f"{COMPOUNDS[n][0] * 1000:.0f} in {f['ci_nm'][0]:.0f}–"
          f"{f['ci_nm'][1]:.0f} nM" for n, f in fits.items()) + ")")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 62)
ax = fig.add_subplot(ms.grid(fig, 1, 1, left=13, right=4, top=5,
                             bottom=11)[0])
dense = 10.0 ** np.linspace(-3.2, 1.2, 300)
for name, (_ic50, _hill, colour, marker) in COMPOUNDS.items():
    f = fits[name]
    ax.plot(dense, four_pl(dense, *f["params"]), color=colour,
            lw=1.3 if name == HERO else 1.0)
    ax.errorbar(CONC_UM, viability[name].mean(axis=0),
                yerr=viability[name].std(axis=0, ddof=1), fmt=marker,
                color=colour, markersize=3.2, markeredgewidth=0,
                elinewidth=0.6, capsize=1.5, capthick=0.6,
                label=f"{name}: {f['ic50_nm']:.0f} nM "
                f"({f['ci_nm'][0]:.0f}–{f['ci_nm'][1]:.0f})")

ax.set_xscale("log")
ax.set_xlim(10 ** -3.25, 10 ** 1.25)
ax.set_xticks([0.001, 0.01, 0.1, 1, 10])
ms.plain_log_ticks(ax.xaxis)
ax.set_ylim(-5, 146)
ax.set_yticks(np.arange(0, 101, 25))
ax.spines["left"].set_bounds(0, 100)
ax.set_xlabel("Concentration (µM)")
ax.set_ylabel("Cell viability (% of DMSO control)")
# the legend sits in the headroom above 100%, clear of every curve
ax.legend(loc="upper right", title="IC$_{50}$ (95% CI)", alignment="left",
          title_fontsize=ms.FS_MATH, bbox_to_anchor=(1.0, 1.0))
ax.text(0.03, 0.05, "n = 3 independent\nexperiments\nMean ± s.d.",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=ms.FS_SMALL, color=ms.GREY_DARK)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig110_dose_response_ci.{ext}")
print("fig110_dose_response_ci: saved png + pdf")
