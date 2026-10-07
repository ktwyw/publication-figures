"""Fig. 153 - Galaxy rotation curve decomposition (single column, 89 mm).

Why flat rotation curves call for dark matter: the circular speeds of a
stellar disc (Freeman's formula), a Hernquist bulge and a gas disc add
in quadrature to a baryonic curve that falls beyond a few scale lengths,
while the simulated measurements stay flat to 30 kpc. An NFW halo with
two free parameters, fitted with the baryons held fixed, closes the gap.
The self-check is that total² equals the sum of the squared components
at every radius, that the Freeman disc peaks between 2.15 and 2.2 scale
lengths, and that the bulge is Keplerian within 5% at 40 scale radii.

Model: stellar disc 4 × 10¹⁰ M☉, scale length 3 kpc; gas disc
1.5 × 10¹⁰ M☉, 6 kpc; bulge 5 × 10⁹ M☉, scale radius 0.5 kpc; NFW
halo simulated with ρₛ = 9 × 10⁶ M☉ kpc⁻³, rₛ = 18 kpc. Data: n = 24
radii, Gaussian errors of 5-9 km s⁻¹; bars are ± 1 s.d.; weighted
least-squares fit of (ρₛ, rₛ). All data are simulated.
"""

import numpy as np
from scipy import optimize, special

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(153)

# -------------------------------------------------- GOVERNING MODEL ----
G = 4.30091e-6          # kpc (km/s)² / M☉ (CODATA G with IAU GM☉ and pc)
M_DISC, R_DISC = 4.0e10, 3.0               # stellar disc: M☉, kpc
M_GAS, R_GAS = 1.5e10, 6.0                 # gas: the same law, twice as wide
M_BULGE, A_BULGE = 5.0e9, 0.5              # Hernquist bulge: M☉, kpc
RHO_S_TRUE, R_S_TRUE = 9.0e6, 18.0         # NFW: M☉ kpc⁻³, kpc
R_MAX, N_OBS = 30.0, 24


def v_exp_disc(r, mass, scale):
    """Thin exponential disc (Freeman 1970): Bessel functions at R/2Rd."""
    y = r / (2 * scale)
    bessel = (special.i0(y) * special.k0(y) - special.i1(y) * special.k1(y))
    return np.sqrt(2 * G * mass / scale * y ** 2 * bessel)


def v_hernquist(r, mass, a):
    """Hernquist (1990) sphere: M(<r) = M r² / (r + a)²."""
    return np.sqrt(G * mass * r) / (r + a)


def v_nfw(r, rho_s, r_s):
    """NFW halo: M(<r) = 4π ρs rs³ [ln(1 + x) − x / (1 + x)], x = r/rs."""
    x = r / r_s
    return np.sqrt(4 * np.pi * G * rho_s * r_s ** 3
                   * (np.log1p(x) - x / (1 + x)) / r)


def baryons(r):
    """Stellar disc, gas and bulge speeds at radius r (km/s)."""
    return {"disc": v_exp_disc(r, M_DISC, R_DISC),
            "gas": v_exp_disc(r, M_GAS, R_GAS),
            "bulge": v_hernquist(r, M_BULGE, A_BULGE)}


def v_baryons(r):
    return np.sqrt(sum(v ** 2 for v in baryons(r).values()))


def v_total(r, rho_s, r_s):
    return np.hypot(v_baryons(r), v_nfw(r, rho_s, r_s))


# ------------------------------------------------------------- DATA ----
r_obs = np.linspace(1.5, R_MAX - 1.0, N_OBS)
v_err = rng.uniform(5.0, 9.0, N_OBS)
v_obs = v_total(r_obs, RHO_S_TRUE, R_S_TRUE) + rng.normal(0, v_err)

# --------------------------------------------------------- ESTIMATOR ---
# halo only: fit log10(ρs) and rs so both parameters are of order unity
(log_rho_fit, r_s_fit), cov = optimize.curve_fit(
    lambda r, log_rho, r_s: v_total(r, 10 ** log_rho, r_s), r_obs, v_obs,
    p0=(7.0, 10.0), sigma=v_err, absolute_sigma=True)
rho_s_fit = 10 ** log_rho_fit
chi2 = (((v_obs - v_total(r_obs, rho_s_fit, r_s_fit)) / v_err) ** 2).sum()
dof = N_OBS - 2

r = R_MAX * np.linspace(0, 1, 900)[1:] ** 2     # dense where curves rise
parts = baryons(r)
parts["halo"] = v_nfw(r, rho_s_fit, r_s_fit)
total = v_total(r, rho_s_fit, r_s_fit)
baryonic = v_baryons(r)

# ------------------------------------------------------- SELF-CHECK ---
quadrature = np.abs(total ** 2 - sum(v ** 2 for v in parts.values())).max()
assert quadrature < 1e-8 * total.max() ** 2, quadrature
peak = optimize.minimize_scalar(lambda x: -v_exp_disc(x, M_DISC, R_DISC),
                                bounds=(0.5 * R_DISC, 5 * R_DISC),
                                method="bounded", options={"xatol": 1e-8})
peak_in_rd = peak.x / R_DISC
assert 2.15 <= peak_in_rd <= 2.2, peak_in_rd
r_far = 40 * A_BULGE
kepler_ratio = v_hernquist(r_far, M_BULGE, A_BULGE) / np.sqrt(
    G * M_BULGE / r_far)
assert 0.95 < kepler_ratio < 1.0, kepler_ratio
assert baryonic[-1] < 0.6 * v_obs[-3:].mean()       # baryons fall short
assert 0.3 < chi2 / dof < 2.5, chi2 / dof
print(f"fig153: self-check passed (total² − Σ component² ≤ "
      f"{quadrature:.1e} km² s⁻²; Freeman peak at {peak_in_rd:.3f} Rd; "
      f"bulge/Kepler {kepler_ratio:.3f} at 40 a; NFW fit ρs = "
      f"{rho_s_fit:.2e} M☉ kpc⁻³, rs = {r_s_fit:.1f} kpc, χ² = {chi2:.1f} "
      f"for {dof} d.o.f.)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 70)
ax = ms.axes(fig, 13, 11, 54, 55)

STYLE = {          # key: (label, colour, line style, width)
    "halo": ("Dark halo\n(NFW)", ms.BLUE, (0, (5, 2)), 1.1),
    "disc": ("Stellar disc", ms.ORANGE, (0, (5, 1.5, 1, 1.5)), 1.1),
    "gas": ("Gas", ms.GREEN, (0, (1, 1.2)), 1.2),
    "bulge": ("Bulge", ms.PINK, (0, (3, 1.2)), 1.0),
}
ax.plot(r, baryonic, color=ms.GREY, lw=0.8)
for key, (label, colour, dashes, width) in STYLE.items():
    ax.plot(r, parts[key], color=colour, ls=dashes, lw=width)
    ax.text(R_MAX + 0.8, parts[key][-1], label, color=colour,
            va="center", fontsize=ms.FS_TICK, linespacing=1.1)
ax.plot(r, total, color=ms.INK, lw=1.6, zorder=3)
ax.errorbar(r_obs, v_obs, yerr=v_err, fmt="o", ms=2.6, mfc="white",
            mec=ms.INK, mew=0.6, ecolor=ms.INK, elinewidth=0.6, capsize=0,
            zorder=4)
ax.text(R_MAX + 0.8, total[-1], "Total", va="center", fontsize=ms.FS_TICK,
        fontweight="bold")
ax.text(R_MAX + 0.8, baryonic[-1], "Baryons only", va="center",
        fontsize=ms.FS_TICK, color=ms.GREY)

# the shortfall at the last measured radius: the argument of the figure
R_GAP = 27.3                               # between two measured radii
low, high = v_baryons(R_GAP), v_total(R_GAP, rho_s_fit, r_s_fit)
ax.annotate("", xy=(R_GAP, low + 3), xytext=(R_GAP, high - 7),
            arrowprops=dict(arrowstyle="<->", color=ms.GREY_DARK, lw=0.5,
                            shrinkA=0, shrinkB=0, mutation_scale=5))
ax.text(R_GAP - 0.8, low + 28, "Missing\nmass", ha="right", va="center",
        fontsize=ms.FS_SMALL, color=ms.GREY_DARK, linespacing=1.1)
ax.text(0.03, 0.985, f"n = {N_OBS} radii, bars ± 1 s.d.\nNFW fit: scale "
        f"radius {r_s_fit:.0f} ± {np.sqrt(cov[1, 1]):.0f} kpc",
        transform=ax.transAxes, va="top", fontsize=ms.FS_SMALL,
        color=ms.GREY_DARK, linespacing=1.3)

ax.set_xlim(0, R_MAX)
ax.set_ylim(0, 270)
ax.set_xticks(np.arange(0, R_MAX + 1, 5))
ax.set_yticks(np.arange(0, 251, 50))
ax.set_xlabel("Galactocentric radius (kpc)")
ax.set_ylabel("Circular speed (km s⁻¹)")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig153_galaxy_rotation_curve.{ext}")
print("fig153_galaxy_rotation_curve: saved png + pdf")
