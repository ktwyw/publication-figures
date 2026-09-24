"""Fig. 73 - Common-tangent construction on Gibbs energy curves.

The thermodynamic engine behind every phase diagram: molar Gibbs
energies of a solid solution and a liquid from the regular-solution
model,

    G(x) = (1-x) G_A + x G_B + Omega x(1-x)
           + RT [x ln x + (1-x) ln(1-x)],

with the common tangent found by solving the two equilibrium
conditions (equal chemical potentials of A and B in both phases) with
fsolve. Tangent points give the coexisting compositions; the tangent's
intercepts at x = 0 and x = 1 are the shared chemical potentials.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import journal_style as js

js.apply()
HERE = js.HERE

# ------------------------------------------------------------ theory ----
RT = 8.314 * 900 / 1000.0                          # kJ/mol at T = 900 K
SOLID = dict(ga=0.0, gb=1.2, omega=14.0)
LIQUID = dict(ga=6.0, gb=0.9, omega=3.0)


def g(x, ga, gb, omega):
    return ((1 - x) * ga + x * gb + omega * x * (1 - x)
            + RT * (x * np.log(x) + (1 - x) * np.log(1 - x)))


def gprime(x, ga, gb, omega):
    return gb - ga + omega * (1 - 2 * x) + RT * np.log(x / (1 - x))


# Both curves are convex (Omega < 2RT), so g' is invertible: parameterise
# the candidate tangent by its slope m and solve one equation in m.
def x_of_slope(m, phase):
    return brentq(lambda x: gprime(x, **phase) - m, 1e-9, 1 - 1e-9)


def intercept_gap(m):
    xa, xb = x_of_slope(m, SOLID), x_of_slope(m, LIQUID)
    return (g(xa, **SOLID) - m * xa) - (g(xb, **LIQUID) - m * xb)


m_scan = np.linspace(-20, 20, 401)
gap = np.array([intercept_gap(m) for m in m_scan])
k = np.flatnonzero(np.sign(gap[:-1]) != np.sign(gap[1:]))[0]
slope = brentq(intercept_gap, m_scan[k], m_scan[k + 1], xtol=1e-12)
x1, x2 = x_of_slope(slope, SOLID), x_of_slope(slope, LIQUID)
mu_a = g(x1, **SOLID) - x1 * slope                 # intercept at x = 0
mu_b = mu_a + slope                                # intercept at x = 1

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.6, 3.0))
x = np.linspace(0.004, 0.996, 500)
gs, gl = g(x, **SOLID), g(x, **LIQUID)
y_lo = min(gs.min(), gl.min()) - 1.1
y_hi = max(SOLID["ga"], SOLID["gb"], LIQUID["ga"], LIQUID["gb"]) + 0.7
ax.plot(x, gs, color="C0", lw=1.3)
ax.plot(x, gl, color="C1", lw=1.3)
ax.plot([0, 1], [mu_a, mu_b], color="black", lw=0.9)

for xt, curve in [(x1, SOLID), (x2, LIQUID)]:
    ax.plot(xt, g(xt, **curve), "o", ms=4, color="black", zorder=5)
    ax.plot([xt, xt], [y_lo, g(xt, **curve)], ls=":", lw=0.7, color="0.5")
for xv, lab in [(x1, r"$x^{\alpha}$"), (x2, r"$x^{L}$")]:
    ax.text(xv, y_lo - 0.16, lab, ha="center", va="top", fontsize=8)

for xi, mu, lab, dx, dy, va in [
        (0, mu_a, r"$\mu_A^{\alpha} = \mu_A^{L}$", 0.015, -0.42, "top"),
        (1, mu_b, r"$\mu_B^{\alpha} = \mu_B^{L}$", -0.02, 0.30, "bottom")]:
    ax.plot(xi, mu, "s", ms=4, mfc="white", mec="black", mew=1.0,
            clip_on=False, zorder=5)
    ax.text(xi + dx, mu + dy, lab, fontsize=7, va=va,
            ha="left" if dx > 0 else "right")

ax.text(0.10, g(0.10, **SOLID) + 0.45, r"$G^{\alpha}$ (solid)",
        color="C0", fontsize=8)
ax.text(0.86, g(0.86, **LIQUID) + 0.45, r"$G^{L}$ (liquid)",
        color="C1", fontsize=8, ha="center")
xm = (x1 + x2) / 2
ax.text(xm, mu_a + slope * xm - 0.42, r"$\alpha$ + L", fontsize=8,
        ha="center", va="top", color="0.3")
ax.text(0.97, y_hi - 0.35, "$T$ = 900 K", fontsize=7, ha="right")

ax.set_xlim(0, 1)
ax.set_ylim(y_lo - 0.5, y_hi)
ax.set_xlabel(r"mole fraction, $x_B$")
ax.set_ylabel(r"molar Gibbs energy, $G$ (kJ mol$^{-1}$)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig073_common_tangent.{ext}")
print(f"saved fig073_common_tangent.png / .pdf  "
      f"(x_alpha = {x1:.3f}, x_L = {x2:.3f})")
