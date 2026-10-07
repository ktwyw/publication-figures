"""Fig. 180 - Microbial growth curves, Gompertz fits (1.5 column, 120 mm).

A plate-reader growth experiment reduced to the two numbers people
compare: optical density on a log axis for three strains, every
replicate well drawn thin under the fitted modified-Gompertz model, with
the geometric meaning of its parameters shown on one strain (the
steepest tangent has slope µmax and meets the starting level at the lag
time λ); then µmax and λ of every well with mean ± s.d. and a one-way
ANOVA. The self-check is that the maximum slope of ln(OD) of each
fitted curve equals its µmax within 0.5%, that the tangent at the
inflection crosses the baseline at λ, and that the per-well fits
recover the simulated truth within 3 standard errors on average.

Model: ln(OD/OD₀) = A·exp(−exp(µmax·e·(λ − t)/A + 1)) (Zwietering et
al. 1990), least squares on ln(OD). Data: n = 4 wells per strain, read
every 15 min for 18 h, well-to-well variation in every parameter, 4%
multiplicative read noise. Bold lines, fit to the four wells pooled;
bars, mean ± s.d. of the per-well fits. All data are simulated.
"""

import numpy as np
from scipy import optimize, stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(180)

# ------------------------------------------------------------- DATA ----
# strain: (µmax per h, lag λ in h, asymptote A = ln(ODmax/OD₀), colour)
STRAINS = {"Strain A": (0.95, 2.0, 5.0, ms.BLUE),
           "Strain B": (0.60, 2.5, 4.3, ms.ORANGE),
           "Strain C": (0.85, 6.0, 3.7, ms.PINK)}
EXPLAINED = "Strain C"             # strain carrying the λ and µmax geometry
N_WELLS, OD_START, NOISE = 4, 0.01, 0.04
WELL_SD = dict(mu=0.04, lag=0.25, asym=0.08, start=0.08)   # mu is relative
hours = np.arange(0, 18.01, 0.25)


def gompertz(t, start, asym, mu, lag):
    """ln(OD) of the modified Gompertz model (Zwietering form)."""
    return start + asym * np.exp(-np.exp(mu * np.e / asym * (lag - t) + 1))


truth, log_od = {}, {}
for strain, (mu, lag, asym, _colour) in STRAINS.items():
    truth[strain] = np.column_stack([
        np.log(OD_START) + rng.normal(0, WELL_SD["start"], N_WELLS),
        asym + rng.normal(0, WELL_SD["asym"], N_WELLS),
        mu * (1 + rng.normal(0, WELL_SD["mu"], N_WELLS)),
        lag + rng.normal(0, WELL_SD["lag"], N_WELLS)])
    log_od[strain] = np.array([gompertz(hours, *well)
                               + rng.normal(0, NOISE, hours.size)
                               for well in truth[strain]])


# ---------------------------------------------------------- FITTING ----
def fit(t, y):
    """Least-squares Gompertz fit; parameters and their standard errors."""
    slope = np.gradient(y, t)
    guess = [y[t < 1].mean(), np.ptp(y), slope.max(), t[np.argmax(slope)] / 2]
    best, cov = optimize.curve_fit(gompertz, t, y, p0=guess)
    return best, np.sqrt(np.diag(cov))


wells, errors, pooled = {}, {}, {}
for strain in STRAINS:
    fits = [fit(hours, y) for y in log_od[strain]]
    wells[strain] = np.array([best for best, _se in fits])
    errors[strain] = np.array([se for _best, se in fits])
    pooled[strain], _ = fit(np.tile(hours, N_WELLS), log_od[strain].ravel())
p_mu = stats.f_oneway(*[wells[s][:, 2] for s in STRAINS]).pvalue
p_lag = stats.f_oneway(*[wells[s][:, 3] for s in STRAINS]).pvalue

# ------------------------------------------------------- SELF-CHECK ---
fine = np.linspace(0, 18, 18001)
geometry = {}
for strain, best in pooled.items():
    curve = gompertz(fine, *best)
    slope = np.gradient(curve, fine)
    k = int(np.argmax(slope))                        # inflection
    crossing = fine[k] - (curve[k] - best[0]) / slope[k]   # tangent = baseline
    geometry[strain] = (fine[k], curve[k], slope[k], crossing)
    assert abs(slope[k] / best[2] - 1) < 0.005, (strain, slope[k], best[2])
    assert abs(crossing - best[3]) < 0.01, (strain, crossing, best[3])
z = np.concatenate([(wells[s][:, 1:] - truth[s][:, 1:]) / errors[s][:, 1:]
                    for s in STRAINS])               # A, µmax and λ
assert np.abs(z).mean() < 3, np.abs(z).mean()
t_i, _y_i, slope_i, cross_i = geometry[EXPLAINED]
print(f"fig180: self-check passed ({EXPLAINED}: max d ln(OD)/dt = "
      f"{slope_i:.4f} vs fitted µmax {pooled[EXPLAINED][2]:.4f} per h; "
      f"tangent meets baseline at {cross_i:.3f} h vs λ = "
      f"{pooled[EXPLAINED][3]:.3f} h; mean |z| of {z.size} fitted "
      f"parameters = {np.abs(z).mean():.2f}; ANOVA µmax "
      f"{ms.format_p(p_mu)}, lag {ms.format_p(p_lag)})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 82)
ax_a = ms.axes(fig, 15, 11, 54, 63)
ax_mu = ms.axes(fig, 95, 46.5, 21, 27.5)
ax_lag = ms.axes(fig, 95, 11, 21, 27.5)
MU_MAX = r"$\mu_{\mathrm{max}}$"                 # set at ms.FS_MATH

# a, every well (thin) under the pooled fit (bold), on a log OD axis
for strain, (_mu, _lag, _asym, colour) in STRAINS.items():
    for y in log_od[strain]:
        ax_a.plot(hours, np.exp(y), color=colour, lw=0.5, alpha=0.55)
    ax_a.plot(fine, np.exp(gompertz(fine, *pooled[strain])), color=colour,
              lw=1.5)
    ax_a.text(18.4, np.exp(pooled[strain][0] + pooled[strain][1]), strain,
              color=colour, va="center", fontsize=ms.FS_TICK)

# geometry of the parameters on one strain: baseline, steepest tangent,
# and the lag where the two meet
start, asym, mu, lag = pooled[EXPLAINED]
LOW = 0.0058                                     # OD level of the lag arrow
colour = STRAINS[EXPLAINED][3]
t_top = lag + 0.85 * asym / mu                   # tangent drawn to 85% of A
ax_a.plot([lag, t_top], np.exp(start + mu * (np.array([lag, t_top]) - lag)),
          color=ms.INK, lw=0.8, ls=(0, (4, 2)), zorder=4)
ax_a.plot([lag, lag], [LOW, np.exp(start)], color=ms.INK, lw=0.5,
          ls=(0, (1, 1.6)))
ax_a.plot(lag, np.exp(start), "o", ms=3.2, mfc="white", mec=ms.INK, mew=0.7,
          zorder=5)
ax_a.annotate("", xy=(lag, LOW), xytext=(0, LOW),
              arrowprops=dict(arrowstyle="<->", color=ms.INK, lw=0.6,
                              shrinkA=0, shrinkB=0, mutation_scale=5))
ax_a.text(lag + 0.4, LOW, f"Lag λ = {lag:.1f} h", va="center")
t_mid = lag + 0.3 * asym / mu
ax_a.text(t_mid + 0.7, np.exp(start + mu * (t_mid - lag)),
          f"Tangent, slope {MU_MAX}\n= {mu:.2f} h⁻¹", va="top",
          fontsize=ms.FS_MATH, linespacing=1.2)
ax_a.set_yscale("log")
ms.plain_log_ticks(ax_a.yaxis)
ax_a.set_xlim(0, 18)
ax_a.set_ylim(0.004, 3)
ax_a.set_xticks(np.arange(0, 19, 3))
ax_a.set_xlabel("Time (h)")
ax_a.set_ylabel("Optical density, OD₆₀₀")
ax_a.text(0.04, 0.97, f"n = {N_WELLS} wells per strain\n"
          "Thin, wells; bold, Gompertz fit", transform=ax_a.transAxes,
          va="top", fontsize=ms.FS_SMALL, color=ms.GREY_DARK, linespacing=1.3)

# b, the fitted parameters of every well, mean ± s.d. beside the dots
OFFSETS = np.linspace(-0.16, 0.16, N_WELLS)      # fixed spread, not random
for ax, column, p_value in ((ax_mu, 2, p_mu), (ax_lag, 3, p_lag)):
    for k, (strain, spec) in enumerate(STRAINS.items()):
        values = wells[strain][:, column]
        ax.plot(k + OFFSETS, values, "o", ms=2.8, color=spec[3], mec="white",
                mew=0.3)
        ax.errorbar(k + 0.34, values.mean(), yerr=values.std(ddof=1),
                    fmt="_", ms=6, color=ms.INK, mew=0.9, elinewidth=0.6,
                    capsize=0)
    ax.set_xlim(-0.5, 2.75)
    ax.set_xticks(range(3), [name[-1] for name in STRAINS])
    ax.tick_params(axis="x", length=0, pad=3)
    ax.text(0.5, 1.04, f"ANOVA, {ms.format_p(p_value)}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=ms.FS_SMALL)
ax_mu.set_ylim(0.4, 1.1)
ax_mu.set_yticks([0.4, 0.6, 0.8, 1.0])
ax_mu.set_ylabel(f"{MU_MAX} (h⁻¹)", fontsize=ms.FS_MATH)
ax_mu.tick_params(labelbottom=False)
ax_lag.set_ylim(0, 8)
ax_lag.set_yticks([0, 2, 4, 6, 8])
ax_lag.set_ylabel("Lag λ (h)")
ax_lag.text(0.04, 0.97, "Bars, mean ± s.d.", transform=ax_lag.transAxes,
            va="top", fontsize=ms.FS_SMALL, color=ms.GREY_DARK)
ax_lag.set_xlabel("Strain")

ms.panel_label(ax_a, "a", dx_pt=-30, dy_pt=8)
ms.panel_label(ax_mu, "b", dx_pt=-30, dy_pt=8)
ms.assert_aligned([ax_a, ax_mu], edges=("top",))
ms.assert_aligned([ax_a, ax_lag], edges=("bottom",))
ms.assert_aligned([ax_mu, ax_lag], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig180_growth_curves.{ext}")
print("fig180_growth_curves: saved png + pdf")
