"""Fig. 167 - Upper-limit plot with expected bands (single column, 89 mm).

The "Brazil band" plot of a search: the observed 95% CL upper limit on a
signal cross-section against the hypothesised mass, the median limit
expected without signal and its ±1σ and ±2σ bands from background-only
toy experiments, and a falling theory prediction; masses where the
observed limit lies below the theory are excluded. The two bands are a
darker and a lighter tint of one hue, so they survive greyscale and
colour-vision deficiency. The self-check is that the toy quantiles match
the analytic Gaussian expectation within 5%, that the bands are nested,
and that the printed exclusion boundary is the root (brentq) of the
interpolated difference between observed limit and theory.

Model: Gaussian counting experiment per mass point, n ~ N(b + σLε, √b);
estimate σ̂ = (n − b)/(Lε) with s.d. s = √b/(Lε); CLs upper limit
σ₉₅ = σ̂ + s·Φ⁻¹(1 − 0.05·Φ(σ̂/s)); 2,000 background-only toys per mass
point; bands, central 68.3% and 95.4% of the toy limits; observed, one
background-only dataset. All data are simulated.
"""

import numpy as np
from scipy import optimize, stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(167)

# ------------------------------------------------------------- DATA ----
MASS = np.arange(200.0, 2001.0, 100.0)       # hypothesised masses (GeV)
LUMI = 100.0                                 # integrated luminosity (fb⁻¹)
N_TOYS, ALPHA = 2000, 0.05                   # toys per mass point; 95% CL


def background(m):
    return 2.0e4 * np.exp(-m / 300.0) + 20.0         # expected events


def efficiency(m):
    return 0.5 * (1 - np.exp(-m / 400.0))            # signal acceptance


def theory(m):
    return 400.0 * np.exp(-m / 220.0)                # cross-section (fb)


# --------------------------------------------------------- ESTIMATOR ---
def upper_limit(z, sd):
    """CLs limit for a Gaussian estimate z standard deviations from 0."""
    return sd * (z + stats.norm.ppf(1 - ALPHA * stats.norm.cdf(z)))


sd = np.sqrt(background(MASS)) / (LUMI * efficiency(MASS))   # s.d. of σ̂ (fb)
SIGMAS = np.array([-2, -1, 0, 1, 2])
toys = upper_limit(rng.standard_normal((N_TOYS, MASS.size)), sd)
bands = np.quantile(toys, stats.norm.cdf(SIGMAS), axis=0)    # 5 x n_mass
analytic = upper_limit(SIGMAS[:, None], sd)   # limit is monotonic in z
observed = upper_limit(rng.standard_normal(MASS.size), sd)


def log_gap(m):
    """ln(observed limit / theory), limit interpolated log-linearly."""
    return np.interp(m, MASS, np.log(observed)) - np.log(theory(m))


crossings = np.flatnonzero(np.diff(np.sign(log_gap(MASS))))
i = crossings[0]
m_excluded = optimize.brentq(log_gap, MASS[i], MASS[i + 1], xtol=1e-10)

# ------------------------------------------------------- SELF-CHECK ---
worst = np.max(np.abs(bands / analytic - 1))
assert worst < 0.05, worst
assert np.all(np.diff(bands, axis=0) > 0)                    # nested bands
assert crossings.size == 1 and log_gap(MASS[0]) < 0          # one boundary
limit_there = np.exp(np.interp(m_excluded, MASS, np.log(observed)))
assert abs(limit_there / theory(m_excluded) - 1) < 1e-8
inside = np.mean((observed > bands[0]) & (observed < bands[-1]))
print(f"fig167: self-check passed (toy quantiles within {100 * worst:.1f}% "
      f"of analytic; median limit {analytic[2, 0] / sd[0]:.3f} s.d.; "
      f"excluded below {m_excluded:.0f} GeV, where limit = theory = "
      f"{limit_there:.3f} fb; observed inside ±2σ at "
      f"{100 * inside:.0f}% of {MASS.size} points)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 68)
ax = ms.axes(fig, 13, 10, 72, 54)
Y_LO, Y_HI = 0.05, 400.0
TINT_1, TINT_2 = "#6BAED6", "#C9DDF0"        # darker and lighter blue

ax.fill_between(MASS, bands[0], bands[4], color=TINT_2, lw=0,
                label="Expected ±2σ")
ax.fill_between(MASS, bands[1], bands[3], color=TINT_1, lw=0,
                label="Expected ±1σ")
ax.plot(MASS, bands[2], color=ms.INK, lw=0.9, ls=(0, (4, 2)),
        label="Median expected")
m_fine = np.linspace(MASS[0], MASS[-1], 400)
shown = theory(m_fine) >= Y_LO               # trimmed to the frame
ax.plot(m_fine[shown], theory(m_fine[shown]), color=ms.VERMILLION, lw=1.2,
        label="Theory prediction")
ax.plot(MASS, observed, "o-", color="black", lw=1.0, ms=2.4, mew=0,
        label="Observed")

# excluded range: a bracket under the curves and a drop line from the crossing
y_bracket = 0.075
ax.plot([MASS[0], MASS[0], m_excluded, m_excluded],
        [y_bracket * 1.25, y_bracket, y_bracket, y_bracket * 1.25],
        color=ms.INK, lw=0.6, solid_capstyle="butt")
ax.plot([m_excluded, m_excluded], [y_bracket * 1.25, limit_there],
        color=ms.INK, lw=0.5, ls=(0, (1, 2)))
ax.plot(m_excluded, limit_there, "o", ms=4.2, mfc="none", mec=ms.INK,
        mew=0.7, zorder=5)
ax.text(MASS[0] + 30, y_bracket * 1.3,
        f"Excluded at 95% CL: m < {m_excluded:,.0f} GeV",
        fontsize=ms.FS_TICK, va="bottom")

ax.set_yscale("log")
ax.set_xlim(MASS[0] - 40, MASS[-1] + 40)
ax.set_ylim(Y_LO, Y_HI)
ms.plain_log_ticks(ax.yaxis)
ax.set_xticks(np.arange(200, 2001, 300))
ax.set_xlabel("Hypothesised mass m (GeV)")
ax.set_ylabel("95% CL upper limit on σ (fb)")
handles, labels = ax.get_legend_handles_labels()
order = [4, 2, 1, 0, 3]
ax.legend([handles[k] for k in order], [labels[k] for k in order],
          loc="upper right", bbox_to_anchor=(1.0, 1.0), handlelength=1.8,
          handletextpad=0.8)
ax.text(0.985, 0.60, f"L = {LUMI:.0f} fb⁻¹\n{N_TOYS:,} toys per mass point",
        transform=ax.transAxes, fontsize=ms.FS_SMALL, color=ms.GREY_DARK,
        ha="right", va="top", linespacing=1.3)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig167_limit_bands.{ext}")
print("fig167_limit_bands: saved png + pdf")
