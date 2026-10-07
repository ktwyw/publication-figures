"""Fig. 166 - Mass peak with extended ML fit and pulls (single column, 89 mm).

The standard collider "bump" plot: binned events as points with Poisson
error bars, an extended binned maximum-likelihood fit of a Gaussian
signal on a falling exponential background, the fit components, and a
pull panel on the same mass axis that shows whether the model describes
the data. The self-check is that the fitted total yield equals the
observed number of events (a property of the extended likelihood), that
the signal yield is within 3σ of the simulated truth, and that the pulls
have mean within ±0.5 of 0 and standard deviation between 0.6 and 1.4.

Statistics: n = Poisson(500) signal + Poisson(12,000) background events
in 60 bins of 1 GeV; error bars, √n; fit parameters: signal yield, mean
and width, background yield and slope; uncertainties from the inverse
Hessian of −ln L; pulls, (data − fit)/√fit; bands, ±1 and ±2.
All data are simulated.
"""

import numpy as np
from scipy import optimize, stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(166)

# ------------------------------------------------------------- DATA ----
M_LO, M_HI, N_BINS = 100.0, 160.0, 60        # mass window (GeV), 1 GeV bins
TRUE = dict(n_sig=500.0, mean=125.0, width=2.0, n_bkg=12000.0, slope=25.0)

edges = np.linspace(M_LO, M_HI, N_BINS + 1)
centres = 0.5 * (edges[:-1] + edges[1:])
bin_width = edges[1] - edges[0]

signal = rng.normal(TRUE["mean"], TRUE["width"], rng.poisson(TRUE["n_sig"]))
# exponential background by inverse-CDF sampling, truncated to the window
u = rng.uniform(size=rng.poisson(TRUE["n_bkg"]))
span = 1 - np.exp(-(M_HI - M_LO) / TRUE["slope"])
background = M_LO - TRUE["slope"] * np.log(1 - u * span)
counts = np.histogram(np.concatenate([signal, background]), edges)[0]
n_obs = counts.sum()


# --------------------------------------------------------- ESTIMATOR ---
def components(theta, grid):
    """Expected signal and background events between grid points."""
    n_sig, mean, width, n_bkg, slope = theta
    gauss = stats.norm.cdf(grid, mean, width)
    expo = np.exp(-(grid - M_LO) / slope)
    sig = n_sig * np.diff(gauss) / (gauss[-1] - gauss[0])
    bkg = n_bkg * -np.diff(expo) / (expo[0] - expo[-1])
    return sig, bkg


def nll(theta):
    """Extended binned −ln L (Poisson per bin, constant terms dropped)."""
    mu = np.sum(components(theta, edges), axis=0)
    return np.sum(mu - counts * np.log(mu))


def hessian(func, x, steps):
    """Central-difference Hessian with one step size per parameter."""
    k = x.size
    out = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            ei, ej = np.eye(k)[i] * steps[i], np.eye(k)[j] * steps[j]
            out[i, j] = (func(x + ei + ej) - func(x + ei - ej)
                         - func(x - ei + ej) + func(x - ei - ej)) / 4
    return out / np.outer(steps, steps)


# fit in units of SCALE so that every parameter is of order one
SCALE = np.array([100.0, 100.0, 1.0, 1000.0, 10.0])
start = np.array([300.0, 124.0, 3.0, n_obs - 300.0, 30.0]) / SCALE
bounds = [(0, None), (1.1, 1.4), (0.5, 10), (0, None), (0.5, 20)]
fit = optimize.minimize(lambda x: nll(x * SCALE), start, method="L-BFGS-B",
                        bounds=bounds, options=dict(ftol=1e-15, gtol=1e-10,
                                                    maxiter=2000))
theta = fit.x * SCALE
covariance = np.linalg.inv(hessian(nll, theta, 1e-3 * SCALE))
errors = np.sqrt(np.diag(covariance))

fit_sig, fit_bkg = components(theta, edges)
fit_total = fit_sig + fit_bkg
pulls = (counts - fit_total) / np.sqrt(fit_total)
chi2, ndf = np.sum(pulls ** 2), N_BINS - theta.size

# ------------------------------------------------------- SELF-CHECK ---
assert fit.success, fit.message
assert abs(fit_total.sum() - n_obs) < 1e-3 * n_obs, (fit_total.sum(), n_obs)
assert abs(theta[0] - TRUE["n_sig"]) < 3 * errors[0], (theta[0], errors[0])
assert abs(pulls.mean()) < 0.5 and 0.6 < pulls.std(ddof=1) < 1.4
print(f"fig166: self-check passed (fitted total {fit_total.sum():.1f} vs "
      f"{n_obs} observed; signal yield {theta[0]:.0f} ± {errors[0]:.0f}, "
      f"truth {TRUE['n_sig']:.0f}; chi2/ndf = {chi2:.1f}/{ndf}; pulls mean "
      f"{pulls.mean():+.2f}, s.d. {pulls.std(ddof=1):.2f})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 80)
gs = ms.grid(fig, 2, 1, left=13, right=3, top=4, bottom=10, hspace=2.5,
             height_ratios=[3.1, 1])
ax = fig.add_subplot(gs[0])
ax_pull = fig.add_subplot(gs[1], sharex=ax)

# smooth curves: the same bin-integrated model on a fine grid, rescaled
# to events per plotted bin
fine = np.linspace(M_LO, M_HI, 1201)
fine_mid = 0.5 * (fine[:-1] + fine[1:])
per_bin = bin_width / (fine[1] - fine[0])
curve_sig, curve_bkg = (c * per_bin for c in components(theta, fine))

ax.fill_between(fine_mid, curve_sig, color=ms.SKY, alpha=0.55, lw=0,
                label="Signal component")
ax.plot(fine_mid, curve_bkg, color=ms.GREY, lw=0.9, ls=(0, (4, 2)),
        label="Background component")
ax.plot(fine_mid, curve_sig + curve_bkg, color=ms.BLUE, lw=1.3,
        label="Total fit")
ax.errorbar(centres, counts, yerr=np.sqrt(counts), fmt="o", color="black",
            ms=2.0, mew=0, elinewidth=0.6, capsize=0, zorder=4,
            label=f"Simulated data (n = {n_obs:,})")
ax.set_ylim(0, 1.08 * (counts + np.sqrt(counts)).max())
ax.set_ylabel(f"Events / {bin_width:g} GeV")
ax.tick_params(labelbottom=False)
handles, labels = ax.get_legend_handles_labels()
order = [3, 2, 1, 0]                        # data first, as read off the plot
ax.legend([handles[i] for i in order], [labels[i] for i in order],
          loc="upper right", bbox_to_anchor=(1.0, 1.0))
ax.text(0.985, 0.60,
        f"Signal yield {theta[0]:.0f} ± {errors[0]:.0f} "
        f"(truth {TRUE['n_sig']:.0f})\n"
        f"Mean {theta[1]:.2f} ± {errors[1]:.2f} GeV\n"
        f"Width {theta[2]:.2f} ± {errors[2]:.2f} GeV\n"
        f"χ²/ndf = {chi2:.1f}/{ndf}",
        transform=ax.transAxes, ha="right", va="top", fontsize=ms.FS_TICK,
        linespacing=1.35)

# pulls: bands first (no outline), then the zero line and the points
ax_pull.axhspan(-2, 2, color=ms.GREY_LIGHT, alpha=0.45, lw=0)
ax_pull.axhspan(-1, 1, color=ms.GREY_LIGHT, alpha=0.9, lw=0)
ax_pull.axhline(0, color=ms.INK, lw=0.6)
ax_pull.plot(centres, pulls, "o", color="black", ms=2.0, mew=0)
ax_pull.set_xlim(M_LO, M_HI)
ax_pull.set_ylim(-3.6, 3.6)
ax_pull.set_yticks([-2, 0, 2])
ax_pull.set_xlabel("m (GeV)")
ax_pull.set_ylabel("Pull")
fig.align_ylabels([ax, ax_pull])

ms.assert_aligned([ax, ax_pull], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig166_mass_peak_pulls.{ext}")
print("fig166_mass_peak_pulls: saved png + pdf")
