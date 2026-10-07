"""Fig. 152 - Exoplanet transit light curve and fit (single column, 89 mm).

A planet crossing a limb-darkened star: the model integrates the
quadratic limb-darkening law over the annuli of the stellar disc that
the planet covers, so the rounded floor and the gradual ingress come
out of the geometry. Simulated photometry (grey) is binned, and the
radius ratio and impact parameter are fitted by least squares with the
limb darkening held fixed; residuals follow below on the same time
axis. The self-check is that with limb darkening off the mid-transit
depth is (Rp/R*)² to 1e-6, that the flux is exactly 1 out of transit,
that the first-to-fourth-contact duration matches the analytic value
for a circular orbit within the time step, and that the fitted radius
ratio lies within 3 standard errors of the truth.

Model: circular orbit, P = 3.5 d, a/R* = 9, Rp/R* = 0.1, b = 0.35;
I(µ) = 1 − u₁(1 − µ) − u₂(1 − µ)², u₁ = 0.40, u₂ = 0.26. Data:
n = 320 one-minute exposures, Gaussian noise 800 ppm; blue points are
means of 10-minute bins (n = 10 each) ± s.e.m. All data are simulated.
"""

import numpy as np
from scipy import optimize

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(152)

# -------------------------------------------------- GOVERNING MODEL ----
PERIOD_D, A_OVER_RS = 3.5, 9.0             # orbital period (d), a / R*
K_TRUE, B_TRUE = 0.10, 0.35                # radius ratio, impact parameter
U1, U2 = 0.40, 0.26                        # illustrative Sun-like values
NOISE_PPM, CADENCE_MIN, HALF_SPAN_H = 800.0, 1.0, 2.66
BIN = 10                                   # exposures per bin
NODE, WEIGHT = np.polynomial.legendre.leggauss(96)     # on [-1, 1]


def intensity(r, u1, u2):
    """Limb-darkened surface brightness at projected radius r."""
    mu = np.sqrt(np.clip(1 - r ** 2, 0, 1))
    return 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2


def separation(t_hours, b):
    """Sky-projected star-planet distance (stellar radii), circular orbit."""
    phase = 2 * np.pi * np.asarray(t_hours, float) / (24 * PERIOD_D)
    return np.hypot(A_OVER_RS * np.sin(phase), b * np.cos(phase))


# ------------------------------------------------------------ SOLVER ---
def blocked(z, k, u1, u2):
    """Flux hidden by a disc of radius k at separation z (star radius 1).

    Integral over stellar annuli of radius r: surface brightness times
    the arc length 2rφ covered by the planet. Annuli wholly inside the
    planet (r < k − z) contribute their full circumference.
    """
    z = np.atleast_1d(np.asarray(z, float))[:, None]
    # wholly covered annuli, r in [0, k - z]: Gauss-Legendre in r
    top = np.clip(k - z, 0, 1)
    r = 0.5 * top * (NODE + 1)
    full = (0.5 * top * WEIGHT * intensity(r, u1, u2) * 2 * np.pi * r).sum(1)
    # partly covered annuli, r in [|z - k|, min(z + k, 1)]; r = mid - half
    # cos(s) removes the square-root ends of the arc angle φ(r)
    lo, hi = np.abs(z - k), np.minimum(z + k, 1)
    half = 0.5 * np.clip(hi - lo, 0, None)
    s = 0.5 * np.pi * (NODE + 1)
    r = 0.5 * (lo + hi) - half * np.cos(s)
    cos_phi = (r ** 2 + z ** 2 - k ** 2) / np.maximum(2 * r * z, 1e-300)
    phi = np.arccos(np.clip(cos_phi, -1, 1))
    part = (0.5 * np.pi * WEIGHT * half * np.sin(s)
            * intensity(r, u1, u2) * 2 * r * phi).sum(1)
    return full + part


def transit_flux(t_hours, k, b, u1=U1, u2=U2):
    """Relative flux of the star at time t_hours from mid-transit."""
    mu = 0.5 * (NODE + 1)                  # disc flux: exact, I is quadratic
    total = (0.5 * WEIGHT * 2 * np.pi * mu
             * (1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2)).sum()
    return 1 - blocked(separation(t_hours, b), k, u1, u2) / total


# ------------------------------------------------------------- DATA ----
n_obs = int(round(2 * HALF_SPAN_H * 60 / CADENCE_MIN)) + 1
n_obs -= n_obs % BIN
t_obs = (np.arange(n_obs) - (n_obs - 1) / 2) * CADENCE_MIN / 60    # hours
flux_obs = (transit_flux(t_obs, K_TRUE, B_TRUE)
            + rng.normal(0, NOISE_PPM * 1e-6, n_obs))

(k_fit, b_fit), cov = optimize.curve_fit(
    transit_flux, t_obs, flux_obs, p0=(0.08, 0.2),
    sigma=np.full(n_obs, NOISE_PPM * 1e-6), absolute_sigma=True,
    bounds=([0.01, 0.0], [0.3, 0.95]))
k_se, b_se = np.sqrt(np.diag(cov))
resid_ppm = (flux_obs - transit_flux(t_obs, k_fit, b_fit)) * 1e6

t_bin = t_obs.reshape(-1, BIN).mean(1)
flux_bin = flux_obs.reshape(-1, BIN).mean(1)
flux_sem = flux_obs.reshape(-1, BIN).std(1, ddof=1) / np.sqrt(BIN)
resid_bin = resid_ppm.reshape(-1, BIN).mean(1)
resid_sem = resid_ppm.reshape(-1, BIN).std(1, ddof=1) / np.sqrt(BIN)

# ------------------------------------------------------- SELF-CHECK ---
depth_uniform = 1 - transit_flux(0.0, K_TRUE, B_TRUE, 0.0, 0.0)[0]
assert abs(depth_uniform - K_TRUE ** 2) < 1e-6, depth_uniform
DT_H = 1e-3                                # time resolution of the check
t_fine = np.arange(-HALF_SPAN_H, HALF_SPAN_H, DT_H)
model_true = transit_flux(t_fine, K_TRUE, B_TRUE)
in_transit = model_true < 1 - 1e-12
duration_numeric = in_transit.sum() * DT_H
sin_i = np.sqrt(1 - (B_TRUE / A_OVER_RS) ** 2)
duration_exact = 24 * PERIOD_D / np.pi * np.arcsin(
    np.sqrt((1 + K_TRUE) ** 2 - B_TRUE ** 2) / (A_OVER_RS * sin_i))
assert np.all(model_true[np.abs(t_fine) > duration_exact / 2 + DT_H] == 1.0)
assert abs(duration_numeric - duration_exact) <= 2 * DT_H
assert abs(k_fit - K_TRUE) < 3 * k_se, (k_fit, k_se)
print(f"fig152: self-check passed (uniform-disc depth {depth_uniform:.8f} "
      f"vs k² = {K_TRUE ** 2:.8f}; duration {duration_numeric:.3f} h vs "
      f"analytic {duration_exact:.3f} h; fitted Rp/R* = {k_fit:.4f} ± "
      f"{k_se:.4f}, truth {K_TRUE}; b = {b_fit:.2f} ± {b_se:.2f})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 86)
gs = ms.grid(fig, 2, 1, left=17, right=4, top=4, bottom=11, hspace=4,
             height_ratios=[2.6, 1])
ax = fig.add_subplot(gs[0])
ax_res = fig.add_subplot(gs[1], sharex=ax)
DATA_C, FIT_C = ms.BLUE, ms.VERMILLION

model_fit = transit_flux(t_fine, k_fit, b_fit)
depth = 1 - model_fit.min()
ax.plot(t_obs, flux_obs, "o", ms=1.5, color=ms.GREY_LIGHT, mew=0,
        rasterized=True, zorder=1)
ax.plot(t_fine, model_fit, color=FIT_C, lw=1.2, zorder=2)
ax.errorbar(t_bin, flux_bin, yerr=flux_sem, fmt="o", ms=2.4, color=DATA_C,
            mew=0, elinewidth=0.6, capsize=0, zorder=3)

# depth: a dimension line left of ingress; duration: one below the floor
X_DEPTH, Y_DUR = -2.05, 1 - depth - 0.0048
dims = dict(arrowstyle="<->", color=ms.INK, lw=0.5, shrinkA=0, shrinkB=0,
            mutation_scale=5)
ax.annotate("", xy=(X_DEPTH, 1 - depth), xytext=(X_DEPTH, 1.0),
            arrowprops=dims, zorder=4)
ax.plot([X_DEPTH - 0.12, -0.75], [1 - depth] * 2, color=ms.GREY, lw=0.4,
        ls=(0, (2, 2)), zorder=1)
ax.text(X_DEPTH - 0.12, 1 - depth / 2, f"Depth {100 * depth:.2f}%",
        rotation=90, ha="right", va="center", fontsize=ms.FS_TICK)
t14 = np.ptp(t_fine[model_fit < 1 - 1e-12])
ax.annotate("", xy=(-t14 / 2, Y_DUR), xytext=(t14 / 2, Y_DUR),
            arrowprops=dims, zorder=4)
ax.text(0, Y_DUR - 0.0006, f"Duration {t14:.2f} h", ha="center", va="top",
        fontsize=ms.FS_TICK)
ax.text(0, 1 - 0.36 * depth, f"Least-squares fit\nRp/R* = {k_fit:.4f} ± "
        f"{k_se:.4f}\nb = {b_fit:.2f} ± {b_se:.2f}", ha="center",
        va="center", fontsize=ms.FS_TICK, color=FIT_C, linespacing=1.3)
ax.text(0.02, 0.985, f"1-min exposures, n = {n_obs}", transform=ax.transAxes,
        va="top", fontsize=ms.FS_SMALL, color=ms.GREY)
ax.text(1.0, 0.985, f"{BIN}-min means ± s.e.m.", transform=ax.transAxes,
        ha="right", va="top", fontsize=ms.FS_SMALL, color=DATA_C)
ax.set_ylim(1 - depth - 0.0072, 1.0062)
ax.set_ylabel("Relative flux")
ax.tick_params(labelbottom=False)

ax_res.axhline(0, color=FIT_C, lw=0.9, zorder=2)
ax_res.plot(t_obs, resid_ppm, "o", ms=1.5, color=ms.GREY_LIGHT, mew=0,
            rasterized=True, zorder=1)
ax_res.errorbar(t_bin, resid_bin, yerr=resid_sem, fmt="o", ms=2.4,
                color=DATA_C, mew=0, elinewidth=0.6, capsize=0, zorder=3)
ax_res.set_ylim(-3200, 3200)
ax_res.set_yticks([-2000, 0, 2000])
ax_res.set_yticklabels(["−2,000", "0", "2,000"])
ax_res.set_xlim(-HALF_SPAN_H - 0.08, HALF_SPAN_H + 0.08)
ax_res.set_xticks(np.arange(-2, 2.1, 1))
ax_res.set_xlabel("Time from mid-transit (h)")
ax_res.set_ylabel("Residual (ppm)")
fig.align_ylabels([ax, ax_res])

ms.panel_label(ax, "a", dx_pt=-40, dy_pt=0)
ms.panel_label(ax_res, "b", dx_pt=-40, dy_pt=0)
ms.assert_aligned([ax, ax_res], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig152_transit_light_curve.{ext}")
print("fig152_transit_light_curve: saved png + pdf")
