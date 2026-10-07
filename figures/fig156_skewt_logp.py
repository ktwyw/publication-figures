"""Fig. 156 - Skew-T log-P diagram with parcel ascent (1.5 column, 120 mm).

The meteorologist's thermodynamic chart built from its equations: the
sheared coordinate x = T + k ln(p₀/p) against log pressure, with
isotherms, dry adiabats (Poisson's equation), pseudo-adiabats (the
saturated lapse rate integrated with solve_ivp) and saturation
mixing-ratio lines, each family in one quiet colour and labelled in gaps
cut in the lines. Over it, a sounding (temperature and dew point), the
surface parcel lifted dry to its condensation level and then along a
pseudo-adiabat, and the positive area between parcel and environment.
The self-check is that potential temperature is constant along every
drawn dry adiabat (1e-9), that the LCL found by brentq (dry adiabat
meets mixing-ratio line) agrees with Bolton's LCL formula within 1 K,
and that Bolton's θe varies by less than 1% along the parcel's
pseudo-adiabat.

Model: Bolton (1980) saturation vapour pressure, latent heat, LCL
temperature and θe; CAPE = R_d ∫ (T_parcel − T_env) d ln p from the level
of free convection to the equilibrium level, without the virtual
temperature correction. Data: one 61-level sounding built from layer
lapse rates plus autocorrelated noise (s.d. 0.3 K in T, 1.2 K in dew
point). All data are simulated.
"""

import numpy as np
from matplotlib.ticker import NullLocator
from scipy import integrate, optimize

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(156)

# -------------------------------------------------- GOVERNING MODEL ----
# dry-air gas constant and specific heat (J kg⁻¹ K⁻¹) as in Bolton (1980),
# Mon. Weather Rev. 108, 1046-1053; EPS = R_d/R_v; g is standard gravity
RD, CP, EPS, GRAV, T0 = 287.04, 1005.7, 0.622, 9.80665, 273.15
KAPPA = RD / CP
P_BOTTOM, P_TOP, P_REF = 1050.0, 100.0, 1000.0           # hPa
SKEW = 38.0             # K per unit ln p: isotherms at about 45° on the page


def e_sat(t_k):
    """Saturation vapour pressure (hPa): Bolton (1980), eq. 10."""
    t_c = t_k - T0
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))


def r_sat(p, t_k):
    """Saturation mixing ratio (kg/kg)."""
    return EPS * e_sat(t_k) / (p - e_sat(t_k))


def dewpoint(p, r):
    """Temperature (K) at which mixing ratio r saturates: eq. 10 inverted."""
    log_e = np.log(r * p / (EPS + r) / 6.112)
    return 243.5 * log_e / (17.67 - log_e) + T0


def dry_adiabat(p, theta):
    """Poisson's equation, T = θ (p/1000)^κ."""
    return theta * (p / P_REF) ** KAPPA


def moist_lapse(log_p, t_k):
    """dT/d ln p of a saturated parcel (pseudo-adiabatic)."""
    latent = 2.501e6 - 2370.0 * (t_k - T0)               # Bolton, eq. 2
    rs = r_sat(np.exp(log_p), t_k)
    return ((RD * t_k + latent * rs)
            / (CP + latent ** 2 * rs * EPS / (RD * t_k ** 2)))


def pseudo_adiabat(p_start, t_start, p_end):
    return integrate.solve_ivp(moist_lapse, np.log([p_start, p_end]),
                               [t_start], rtol=1e-9, atol=1e-9,
                               dense_output=True).sol


def theta_e(p, t_k):
    """Equivalent potential temperature at saturation: Bolton, eq. 43."""
    r = 1e3 * r_sat(p, t_k)                              # g/kg
    return (t_k * (P_REF / p) ** (0.2854 * (1 - 0.28e-3 * r))
            * np.exp((3.376 / t_k - 0.00254) * r * (1 + 0.81e-3 * r)))


# ------------------------------------------------------------- DATA ----
# Sounding from lapse rates (K/km) by layer: mixed layer, capping
# inversion, troposphere, stratosphere; hydrostatic dT/d ln p = R_d Γ T/g
SURFACE_T, SURFACE_TD = 30.0 + T0, 21.0 + T0             # at 1000 hPa
LAYERS = [(900.0, 9.6), (865.0, -4.0), (190.0, 6.9), (P_TOP, -1.5)]
p_env = np.geomspace(P_REF, P_TOP, 61)


def red_noise(sd, rho=0.85):
    out = np.zeros(p_env.size)                 # zero at the surface
    for i in range(1, out.size):
        out[i] = rho * out[i - 1] + rng.normal(0, sd * np.sqrt(1 - rho ** 2))
    return out


t_env = np.empty(p_env.size)
t_env[0] = SURFACE_T
for i in range(1, p_env.size):
    gamma = next(g for top, g in LAYERS if p_env[i - 1] > top) * 1e-3
    t_env[i] = t_env[i - 1] * (p_env[i] / p_env[i - 1]) ** (RD * gamma / GRAV)
t_env += red_noise(0.3)
# dew point: constant mixing ratio in the mixed layer, drying out above
depression = 4.0 + 26.0 * (1 - np.exp(-np.log(900.0 / p_env) / 0.6))
td_env = np.where(p_env >= 900.0,
                  dewpoint(p_env, r_sat(P_REF, SURFACE_TD)),
                  t_env - np.maximum(depression + red_noise(1.2), 0.5))

# ------------------------------------------------------------ SOLVER ---
theta_parcel = SURFACE_T * (P_REF / p_env[0]) ** KAPPA
r_parcel = r_sat(p_env[0], SURFACE_TD)
p_lcl = optimize.brentq(lambda p: dry_adiabat(p, theta_parcel)
                        - dewpoint(p, r_parcel), 500.0, p_env[0], xtol=1e-10)
t_lcl = dry_adiabat(p_lcl, theta_parcel)
# Bolton (1980), eq. 15: LCL temperature from surface T and dew point
t_lcl_bolton = 1 / (1 / (SURFACE_TD - 56) + np.log(SURFACE_T / SURFACE_TD)
                    / 800) + 56
parcel_moist = pseudo_adiabat(p_lcl, t_lcl, P_TOP)

p_up = np.geomspace(p_lcl, P_TOP, 2000)                  # above the LCL
t_up = parcel_moist(np.log(p_up))[0]
excess = t_up - np.interp(np.log(p_up[::-1]), np.log(p_env[::-1]),
                          t_env[::-1])[::-1]
sign_change = np.flatnonzero(np.diff(np.sign(excess)))
i_lfc, i_el = sign_change[excess[sign_change] < 0][0] + 1, sign_change[-1]
buoyant = slice(i_lfc, i_el + 1)
cape = -RD * integrate.trapezoid(excess[buoyant], np.log(p_up[buoyant]))

THETAS = np.arange(260.0, 461.0, 20.0)                   # dry adiabats, K
THETA_W = np.arange(4.0, 33.0, 4.0) + T0                 # pseudo-adiabats
MIXING = np.array([1.0, 2.0, 4.0, 7.0, 10.0, 16.0, 24.0])    # g/kg
ISOTHERMS = np.arange(-120.0, 41.0, 10.0) + T0
p_line = np.geomspace(P_BOTTOM, P_TOP, 500)

# ------------------------------------------------------- SELF-CHECK ---
worst_theta = max(np.ptp(dry_adiabat(p_line, th) * (P_REF / p_line) ** KAPPA)
                  / th for th in THETAS)
assert worst_theta < 1e-9, worst_theta
assert abs(r_sat(p_lcl, t_lcl) - r_parcel) < 1e-9 * r_parcel
assert abs(t_lcl - t_lcl_bolton) < 1.0, (t_lcl, t_lcl_bolton)
theta_e_up = theta_e(p_up, t_up)
theta_e_spread = np.ptp(theta_e_up) / theta_e_up.mean()
assert theta_e_spread < 0.01, theta_e_spread
assert p_lcl > p_up[i_lfc] > p_up[i_el] and cape > 0
print(f"fig156: self-check passed (theta constant to {worst_theta:.1e}; LCL "
      f"{p_lcl:.1f} hPa, {t_lcl - T0:.2f} °C by brentq vs "
      f"{t_lcl_bolton - T0:.2f} °C Bolton; theta-e spread "
      f"{100 * theta_e_spread:.2f}%; LFC {p_up[i_lfc]:.0f} hPa, EL "
      f"{p_up[i_el]:.0f} hPa; CAPE {cape:.0f} J kg⁻¹)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 112)
ax = ms.axes(fig, 14, 11, 80, 96)
X_LIM = (-38.0, 42.0)                      # °C along the bottom isobar
C_ISO, C_DRY, C_MOIST, C_MIX = ms.GREY, ms.ORANGE, ms.SKY, ms.GREEN


def skew(p, t_k):
    return t_k - T0 + SKEW * np.log(P_BOTTOM / p)


background, labels = [], []


def family(p, t_k, colour, value=None, p_label=None, **style):
    """One background line, ended at the frame, with an optional label."""
    x = skew(p, t_k)
    inside = (x >= X_LIM[0]) & (x <= X_LIM[1])
    background.extend(ax.plot(np.where(inside, x, np.nan), p, color=colour,
                              lw=0.5, zorder=1, **style))
    if value is not None:
        labels.append(ax.text(np.interp(-p_label, -p, x), p_label, value,
                              color=colour, fontsize=ms.FS_SMALL,
                              ha="center", va="center", zorder=2))


for t_iso in ISOTHERMS:                # labelled 7 K inside the left edge
    t_c = t_iso - T0
    name = f"−{-t_c:.0f}" if t_c in (-100, -80, -60, -40) else None
    family(p_line, np.full(p_line.size, t_iso), C_ISO, name,
           P_BOTTOM * np.exp(-(X_LIM[0] + 7 - t_c) / SKEW))
for theta in THETAS:
    family(p_line, dry_adiabat(p_line, theta), C_DRY,
           f"{theta:.0f}" if theta in (300, 340, 380, 420) else None, 118.0)
for theta_w in THETA_W:
    below = pseudo_adiabat(P_REF, theta_w, P_BOTTOM)(np.log(p_line[:12]))[0]
    above = pseudo_adiabat(P_REF, theta_w, 180.0)(np.log(p_line[12:]))[0]
    t_moist = np.concatenate([below, above])
    keep = p_line >= 200.0                 # they merge with dry adiabats above
    family(p_line[keep], t_moist[keep], C_MOIST,
           f"{theta_w - T0:.0f}" if theta_w - T0 in (4, 12, 20, 32) else None,
           215.0)
for r in MIXING:
    keep = p_line >= 600.0
    family(p_line[keep], dewpoint(p_line[keep], r * 1e-3), C_MIX, f"{r:g}",
           1027.0, ls=(0, (4, 2)))

# the sounding and the parcel are the signal: drawn last, never broken
p_dry = np.geomspace(p_env[0], p_lcl, 50)
x_env_up = skew(p_up, t_up - excess)
ax.fill_betweenx(p_up[buoyant], x_env_up[buoyant], skew(p_up, t_up)[buoyant],
                 color="#F3C9AE", lw=0, zorder=3)
ax.plot(np.r_[skew(p_dry, dry_adiabat(p_dry, theta_parcel)),
              skew(p_up, t_up)], np.r_[p_dry, p_up], color=ms.INK, lw=0.9,
        ls=(0, (4, 1.5)), zorder=5)
ax.plot(skew(p_dry, dewpoint(p_dry, r_parcel)), p_dry, color=ms.INK, lw=0.6,
        ls=(0, (1, 1.2)), zorder=5)
ax.plot(skew(p_env, td_env), p_env, color=ms.BLUE, lw=1.3, zorder=6)
ax.plot(skew(p_env, t_env), p_env, color=ms.VERMILLION, lw=1.3, zorder=6)
ax.plot(skew(p_lcl, t_lcl), p_lcl, "o", ms=3.4, mfc="white", mec=ms.INK,
        mew=0.8, zorder=7)

ax.set_yscale("log")
ax.set_ylim(P_BOTTOM, P_TOP)
ax.set_xlim(*X_LIM)
ax.set_yticks([1000, 850, 700, 500, 400, 300, 250, 200, 150, 100])
ms.plain_log_ticks(ax.yaxis)
ax.yaxis.set_minor_locator(NullLocator())
ax.set_xticks(np.arange(-30, 41, 10))
ax.spines[["top", "right"]].set_visible(True)
ax.set_xlabel("Temperature (°C), read along the skewed isotherms")
ax.set_ylabel("Pressure (hPa)")

# levels on the right edge, where the parcel's story is read off
right = ax.get_yaxis_transform()
for p_level, name in ((p_lcl, "LCL"), (p_up[i_lfc], "LFC"),
                      (p_up[i_el], "EL")):
    ax.plot([1.0, 1.02], [p_level, p_level], transform=right, color=ms.INK,
            lw=0.6, clip_on=False)
    ax.text(1.03, p_level, f"{name} {p_level:.0f} hPa", transform=right,
            va="center", fontsize=ms.FS_TICK)
key = [("Temperature", ms.VERMILLION, "bold"), ("Dew point", ms.BLUE, "bold"),
       ("Lifted parcel (dashed)", ms.INK, "normal"),
       (f"CAPE {cape:.0f} J kg⁻¹", ms.VERMILLION, "normal"),
       ("", ms.INK, "normal"),
       ("Isotherms (°C)", C_ISO, "normal"),
       ("Dry adiabats, θ (K)", C_DRY, "normal"),
       ("Pseudo-adiabats,", C_MOIST, "normal"),
       ("  θw (°C)", C_MOIST, "normal"),
       ("Saturation mixing", C_MIX, "normal"),
       ("  ratio (g kg⁻¹)", C_MIX, "normal")]
for row, (text, colour, weight) in enumerate(key):
    ax.text(1.03, 0.60 - 0.034 * row, text, transform=ax.transAxes,
            color=colour, fontweight=weight, fontsize=ms.FS_TICK, va="center")

# cut every background line where a label sits, so no stroke crosses type
fig.canvas.draw()
pad = 2.5 * fig.dpi / 72                                 # 2.5 pt of air
boxes = [t.get_window_extent().padded(pad) for t in labels]
for line in background:
    xy = line.get_xydata()
    px, py = ax.transData.transform(np.nan_to_num(xy, nan=1e6)).T
    hidden = np.any([(px > box.x0) & (px < box.x1) & (py > box.y0)
                     & (py < box.y1) for box in boxes], axis=0)
    line.set_xdata(np.where(hidden, np.nan, xy[:, 0]))

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig156_skewt_logp.{ext}")
print("fig156_skewt_logp: saved png + pdf")
