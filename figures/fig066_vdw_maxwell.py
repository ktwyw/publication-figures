"""Fig. 66 - Van der Waals isotherms with the Maxwell construction.

Everything is computed from the reduced van der Waals equation

    P = 8T/(3v - 1) - 3/v^2        (T, P, v in critical units).

For each subcritical temperature the saturation pressure is found by
solving the Maxwell equal-area condition

    Integral_{v_l}^{v_g} P(v) dv = P_sat (v_g - v_l)

with brentq, using the analytic antiderivative
Integral P dv = (8T/3) ln(3v - 1) + 3/v and the cubic's roots from
np.roots (coefficients [3P, -(P+8T), 9, -3]; check: T = P = 1 gives
3(v-1)^3). Sweeping T builds the binodal dome; the spinodal follows
analytically from dP/dv = 0 as T = (3v-1)^2 / (4 v^3). One isotherm is
highlighted with its tie line and the two equal-area lobes shaded.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy.optimize import brentq

import journal_style as js

js.apply()
HERE = js.HERE


# ------------------------------------------------------------ theory ----
def p_vdw(v, T):
    return 8 * T / (3 * v - 1) - 3 / v**2


def real_roots(P, T):
    r = np.roots([3 * P, -(P + 8 * T), 9.0, -3.0])
    r = r[np.abs(r.imag) < 1e-7].real
    return np.sort(r[r > 1 / 3 + 1e-9])


def area_mismatch(P, T):
    r = real_roots(P, T)
    vl, vg = r[0], r[-1]
    antider = lambda v: (8 * T / 3) * np.log(3 * v - 1) + 3 / v
    return (antider(vg) - antider(vl)) - P * (vg - vl)


def maxwell(T):
    """Saturation pressure and coexisting volumes at reduced T < 1."""
    P_scan = np.linspace(1e-3, 0.999, 500)
    three = np.array([real_roots(P, T).size == 3 for P in P_scan])
    idx = np.flatnonzero(three)
    if idx.size < 6:
        return None
    p_lo, p_hi = P_scan[idx[1]], P_scan[idx[-2]]
    p_sat = brentq(lambda P: area_mismatch(P, T), p_lo, p_hi, xtol=1e-10)
    r = real_roots(p_sat, T)
    return p_sat, r[0], r[1], r[-1]


# binodal dome from a temperature sweep
T_dome = np.concatenate([np.linspace(0.62, 0.84, 8),
                         np.linspace(0.85, 0.98, 18),
                         np.linspace(0.982, 0.995, 6)])
vl_b, vg_b, p_b = [], [], []
for T in T_dome:
    out = maxwell(T)
    if out is None:
        continue
    p_sat, vl, _, vg = out
    vl_b.append(vl)
    vg_b.append(vg)
    p_b.append(p_sat)
vl_b, vg_b, p_b = map(np.array, (vl_b, vg_b, p_b))

# spinodal (analytic)
v_sp = np.linspace(0.40, 4.5, 400)
T_sp = (3 * v_sp - 1) ** 2 / (4 * v_sp**3)
P_sp = p_vdw(v_sp, T_sp)

# highlighted isotherm with tie line and equal-area lobes
T_HL = 0.90
p_sat, vl, vm, vg = maxwell(T_HL)

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.9, 3.1))

dome_v = np.concatenate([vl_b, [1.0], vg_b[::-1]])
dome_p = np.concatenate([p_b, [1.0], p_b[::-1]])
fill_v = np.concatenate([dome_v, [vg_b[0], vl_b[0]]])   # close along P = 0
fill_p = np.concatenate([dome_p, [0.0, 0.0]])
ax.fill(fill_v, fill_p, color="0.95", lw=0, zorder=0)
ax.plot(dome_v, dome_p, color="0.35", lw=1.0, zorder=2)
m = (P_sp > 0) & (T_sp <= 1)
ax.plot(v_sp[m], P_sp[m], ls="--", lw=0.9, color="0.6", zorder=1)

v = np.logspace(np.log10(0.40), np.log10(7.0), 600)
for T in [0.85, 0.90, 1.00, 1.10]:
    hl = np.isclose(T, T_HL)
    ax.plot(v, p_vdw(v, T), lw=1.3 if hl else 0.9,
            color="C0" if hl else ("black" if T == 1.0 else "0.45"),
            zorder=3)

# tie line + equal-area lobes for the highlighted isotherm
v_loop = np.logspace(np.log10(vl), np.log10(vg), 400)
p_loop = p_vdw(v_loop, T_HL)
ax.fill_between(v_loop, p_loop, p_sat, where=v_loop <= vm,
                color="C0", alpha=0.30, lw=0, zorder=2)
ax.fill_between(v_loop, p_sat, p_loop, where=v_loop >= vm,
                color="C1", alpha=0.30, lw=0, zorder=2)
ax.plot([vl, vg], [p_sat, p_sat], color="C1", lw=1.2, zorder=4)
ax.plot([vl, vg], [p_sat, p_sat], "o", ms=3.5, color="C1", zorder=5)
ax.text(np.sqrt(vl * vm), p_sat - 0.075, "$A_1$", ha="center", fontsize=7.5)
ax.text(np.sqrt(vm * vg), p_sat + 0.032, "$A_2$", ha="center", fontsize=7.5)
ax.annotate(f"Maxwell tie line ($A_1 = A_2$), $T_r$ = {T_HL:g}",
            xy=(1.55, p_sat), xytext=(1.78, 0.20), fontsize=6.5,
            color="C1", ha="center",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="C1"))

ax.plot(1, 1, marker="*", ms=10, color="C3", zorder=6, clip_on=False)
ax.text(1.09, 1.035, "critical point", fontsize=7, ha="left")
ax.text(2.72, 0.345, "spinodal", fontsize=6.5, color="0.55", rotation=-33)
ax.text(1.05, 0.28, "L + V", fontsize=7.5, color="0.4", ha="center")
ax.text(6.6, p_vdw(6.6, 1.10) + 0.045, "$T_r$ = 1.10", fontsize=6.5,
        ha="right", color="0.35")
ax.text(6.6, p_vdw(6.6, 0.85) - 0.075, "$T_r$ = 0.85", fontsize=6.5,
        ha="right", color="0.35")

ax.set_xscale("log")
ax.set_xlim(0.4, 7)
ax.set_ylim(0, 1.6)
ax.set_xticks([0.5, 1, 2, 5])
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"reduced volume, $v_r$")
ax.set_ylabel(r"reduced pressure, $P_r$")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig066_vdw_maxwell.{ext}")
print("saved fig066_vdw_maxwell.png / .pdf")
