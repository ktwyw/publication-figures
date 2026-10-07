"""Fig. 189 - I-V and P-V curves of a PV module (double column, 183 mm).

The two standard views of a solar module from the single-diode model.
a, current against voltage at four irradiances with every maximum-power
point (MPP) marked and joined by the MPP locus, short-circuit current
and open-circuit voltage indicated for the top curve, and the MPP
rectangle whose area ratio to V_oc × I_sc is the fill factor. b, power
against voltage for the same irradiances, plus one hot-cell curve that
shows why heat costs voltage, not current. The self-check is that the
explicit Lambert-W current equals a brentq solution of the implicit
equation to 1e-9 A, that each MPP from minimize_scalar is within one
grid step of the grid maximum with dP/dV changing sign across it, that
I_sc is proportional to irradiance within 0.5%, and that V_oc rises
monotonically with irradiance by a·ln(G₂/G₁) within 5%.

Model: I = I_ph − I₀[exp((V + I·R_s)/a) − 1] − (V + I·R_s)/R_sh with
a = n·N_s·kT/q; placeholder 60-cell module, I_ph = 9 A at 1,000 W m⁻²,
I₀ = 2 nA, n = 1.1, R_s = 0.30 Ω, R_sh = 300 Ω; I_ph ∝ irradiance,
I₀ ∝ T³exp(−E_g/kT). No data: every curve is computed from the
equations.
"""

import numpy as np
from matplotlib import colormaps
from scipy import constants, optimize, special

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
N_CELLS = 60                      # cells in series (placeholder module)
I_PH_REF = 9.0                    # photocurrent at G_REF, T_REF (A)
I_0_REF = 2.0e-9                  # diode saturation current at T_REF (A)
N_IDEAL = 1.1                     # diode ideality factor
R_S, R_SH = 0.30, 300.0           # series and shunt resistance (Ω)
ALPHA_I = 5.0e-4                  # relative I_ph temperature coefficient (/K)
E_GAP = 1.121                     # silicon band gap (eV), De Soto et al.,
#                                   Sol. Energy 80, 78 (2006); held constant
G_REF, T_REF = 1000.0, 298.15     # reference irradiance (W m⁻²), cell T (K)
IRRADIANCES = [1000.0, 750.0, 500.0, 250.0]
T_HOT = 338.15                    # the extra curve: 65 °C at G_REF
KB_Q = constants.k / constants.e  # Boltzmann constant / charge (V/K), CODATA


def diode_parameters(g, temp):
    """Photocurrent, saturation current and modified ideality a = nN_s kT/q."""
    i_ph = I_PH_REF * g / G_REF * (1 + ALPHA_I * (temp - T_REF))
    i_0 = I_0_REF * (temp / T_REF) ** 3 * np.exp(
        E_GAP / KB_Q * (1 / T_REF - 1 / temp))
    return i_ph, i_0, N_IDEAL * N_CELLS * KB_Q * temp


# ------------------------------------------------------------ SOLVER ---
def current(v, g, temp=T_REF):
    """Explicit solution of the implicit diode equation (Lambert W)."""
    i_ph, i_0, a = diode_parameters(g, temp)
    scale = a * (R_S + R_SH)
    w = special.lambertw(R_S * i_0 * R_SH / scale
                         * np.exp(R_SH * (R_S * (i_ph + i_0) + v) / scale))
    return (R_SH * (i_ph + i_0) - v) / (R_S + R_SH) - a / R_S * w.real


def current_brentq(v, g, temp=T_REF):
    """The same current by root-finding on the implicit equation."""
    i_ph, i_0, a = diode_parameters(g, temp)

    def residual(i):
        v_d = v + i * R_S
        return i_ph - i_0 * np.expm1(v_d / a) - v_d / R_SH - i

    return optimize.brentq(residual, -20.0, i_ph + 1.0, xtol=1e-14,
                           rtol=1e-15)


def characterise(g, temp=T_REF, n_grid=2001):
    """Curve on a grid from 0 to V_oc with I_sc, V_oc and the MPP."""
    v_oc = optimize.brentq(current, 0, 60, args=(g, temp), xtol=1e-13)
    v = np.linspace(0, v_oc, n_grid)
    i = current(v, g, temp)
    mpp = optimize.minimize_scalar(lambda u: -u * current(u, g, temp),
                                   bounds=(0, v_oc), method="bounded",
                                   options=dict(xatol=1e-10))
    return dict(v=v, i=i, p=v * i, i_sc=i[0], v_oc=v_oc, v_mp=mpp.x,
                p_mp=-mpp.fun, i_mp=-mpp.fun / mpp.x, g=g, temp=temp)


curves = [characterise(g) for g in IRRADIANCES]
hot = characterise(G_REF, T_HOT)
top = curves[0]
fill_factor = top["p_mp"] / (top["v_oc"] * top["i_sc"])
locus = [characterise(g, n_grid=3) for g in np.linspace(150, G_REF, 35)]

# ------------------------------------------------------- SELF-CHECK ---
worst = 0.0
for c in curves + [hot]:
    for v in c["v"][::200]:
        worst = max(worst, abs(current(v, c["g"], c["temp"])
                               - current_brentq(v, c["g"], c["temp"])))
    step = c["v"][1] - c["v"][0]
    assert abs(c["v_mp"] - c["v"][np.argmax(c["p"])]) <= step

    def slope(u, c=c, h=1e-4):
        """dP/dV by central difference."""
        return ((u + h) * current(u + h, c["g"], c["temp"])
                - (u - h) * current(u - h, c["g"], c["temp"])) / (2 * h)

    assert slope(c["v_mp"] - step) > 0 > slope(c["v_mp"] + step)
assert worst < 1e-9, worst
per_irradiance = np.array([c["i_sc"] / c["g"] for c in curves])
assert np.ptp(per_irradiance) < 0.005 * per_irradiance.mean()
v_ocs = np.array([c["v_oc"] for c in curves])
assert np.all(np.diff(v_ocs) < 0)             # IRRADIANCES are descending
a_ref = diode_parameters(G_REF, T_REF)[2]
log_rise = -np.diff(v_ocs) / (a_ref * -np.diff(np.log(IRRADIANCES)))
assert np.all(np.abs(log_rise - 1) < 0.05), log_rise
assert hot["v_oc"] < top["v_oc"] and hot["i_sc"] > top["i_sc"]
print(f"fig189: self-check passed (Lambert W vs brentq within {worst:.1e} "
      f"A; at {G_REF:.0f} W/m2: Isc {top['i_sc']:.3f} A, Voc "
      f"{top['v_oc']:.2f} V, MPP {top['v_mp']:.2f} V x {top['i_mp']:.3f} A "
      f"= {top['p_mp']:.1f} W, FF {fill_factor:.3f}; Isc/G spread "
      f"{100 * np.ptp(per_irradiance) / per_irradiance.mean():.3f}%; "
      f"at {T_HOT - 273.15:.0f} C: Voc {hot['v_oc']:.2f} V, "
      f"{hot['p_mp']:.1f} W)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 72)
gs = ms.grid(fig, 1, 2, left=12, right=4, top=7, bottom=11, wspace=17)
ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
tints = colormaps["Blues"](np.linspace(0.95, 0.5, len(curves)))
HOT, V_MAX = ms.VERMILLION, 48.0
MPP = dict(marker="o", ls="none", ms=3.6, mfc="white", mec=ms.INK, mew=0.8,
           zorder=6)
locus_v = [c["v_mp"] for c in locus]

# a, I–V: the shaded rectangle V_mp × I_mp over V_oc × I_sc is the fill factor
ax_a.fill_between([0, top["v_mp"]], 0, top["i_mp"], color=ms.BLUE,
                  alpha=0.10, lw=0, zorder=0)
for c, tint in zip(curves, tints):
    ax_a.plot(c["v"], c["i"], color=tint, lw=1.3, zorder=3)
    ax_a.text(15, c["i_sc"] + 0.12, f"{c['g']:,.0f} W m⁻²", color=tint,
              fontsize=ms.FS_TICK, va="bottom")
ax_a.plot(locus_v, [c["i_mp"] for c in locus], color=ms.INK, lw=0.6,
          zorder=5)
ax_a.plot([c["v_mp"] for c in curves], [c["i_mp"] for c in curves], **MPP)
ax_a.text(top["v_mp"] + 1.2, top["i_mp"] + 0.55, "Maximum-power points\n"
          "and their locus", fontsize=ms.FS_TICK, va="bottom",
          linespacing=1.2)
ax_a.plot([0, top["v_oc"]], [top["i_sc"], 0], "s", color=tints[0], ms=3.4,
          mec=ms.INK, mew=0.6, clip_on=False, zorder=7)
ax_a.text(1.2, top["i_sc"] + 0.2,
          rf"$I_\mathrm{{sc}}$ = {top['i_sc']:.2f} A", fontsize=ms.FS_MATH,
          va="bottom")
ax_a.text(top["v_oc"] + 1.0, 0.9,
          rf"$V_\mathrm{{oc}}$ = {top['v_oc']:.1f} V", fontsize=ms.FS_MATH,
          va="center")
ax_a.text(13.5, 1.0,
          f"Shaded: {top['v_mp']:.1f} V × {top['i_mp']:.2f} A = "
          f"{top['p_mp']:.0f} W\nFill factor {fill_factor:.3f}",
          fontsize=ms.FS_TICK, ha="center", va="center", linespacing=1.25)
ax_a.set_xlim(0, V_MAX)
ax_a.set_ylim(0, 11)
ax_a.set_yticks(np.arange(0, 11, 2))
ax_a.set_xlabel("Module voltage (V)")
ax_a.set_ylabel("Module current (A)")

# b, P–V: MPP powers read off to the right; the hot curve loses voltage
for c, tint in zip(curves, tints):
    ax_b.plot(c["v"], c["p"], color=tint, lw=1.3, zorder=3)
    ax_b.plot([c["v_mp"], 39.5], [c["p_mp"]] * 2, color=ms.GREY, lw=0.4,
              ls=(0, (1, 1.5)), zorder=2)
    ax_b.text(40.2, c["p_mp"], f"{c['p_mp']:.0f} W", fontsize=ms.FS_TICK,
              va="center")
ax_b.plot(hot["v"], hot["p"], color=HOT, lw=1.1, ls=(0, (3.5, 1.6)),
          zorder=4)
ax_b.plot(locus_v, [c["p_mp"] for c in locus], color=ms.INK, lw=0.6,
          zorder=5)
ax_b.plot([c["v_mp"] for c in curves], [c["p_mp"] for c in curves], **MPP)
ax_b.plot(hot["v_mp"], hot["p_mp"], **{**MPP, "mec": HOT})
Y_DIM = 282.0                               # the voltage-loss dimension line
for c in (hot, top):
    ax_b.plot([c["v_mp"]] * 2, [c["p_mp"] + 9, Y_DIM + 5], color=ms.INK,
              lw=0.5, zorder=2)
ax_b.annotate("", xy=(hot["v_mp"], Y_DIM), xytext=(top["v_mp"], Y_DIM),
              arrowprops=dict(arrowstyle="<|-|>", color=ms.INK, lw=0.5,
                              mutation_scale=5, shrinkA=0, shrinkB=0))
ax_b.text(hot["v_mp"] - 1.0, Y_DIM,
          f"Cell at {T_HOT - 273.15:.0f} °C (dashed): "
          f"−{top['v_mp'] - hot['v_mp']:.1f} V,\n"
          f"−{top['p_mp'] - hot['p_mp']:.0f} W "
          f"(−{100 * (1 - hot['p_mp'] / top['p_mp']):.0f}%) at the MPP",
          color=HOT, fontsize=ms.FS_TICK, ha="right", va="center",
          linespacing=1.25)
ax_b.set_xlim(0, V_MAX)
ax_b.set_ylim(0, 310)
ax_b.set_yticks(np.arange(0, 301, 50))
ax_b.set_xlabel("Module voltage (V)")
ax_b.set_ylabel("Module power (W)")

ms.panel_label(ax_a, "a", dx_pt=-24)
ms.panel_label(ax_b, "b", dx_pt=-28)
ms.assert_aligned([ax_a, ax_b])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig189_pv_module_curves.{ext}")
print("fig189_pv_module_curves: saved png + pdf")
