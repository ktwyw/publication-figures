"""Fig. 97 - Gas material balance: the p/Z line (single column).

For a volumetric dry-gas reservoir the real-gas material balance is
exactly linear,

    p/Z = (p_i/Z_i) (1 - G_p/G),

so plotting p/Z against cumulative production and extrapolating to
p/Z = 0 reads off the original gas in place. Synthetic surveys are
generated with a smooth illustrative Z(p), pressure noise added, and a
straight line fitted: the intercept recovers G. The dashed curve is a
water-drive reservoir whose pressure support bends the trend - its
early-time tangent extrapolates to an apparent G ~ 1.5x too large, the
classic interpretation trap.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import brentq

import journal_style as js

js.apply()
HERE = js.HERE

rng = np.random.default_rng(97)

G, P_I = 100.0, 5000.0                             # Bscf, psia


def z_of_p(p):
    return 1.0 - 9.5e-5 * p + 1.55e-8 * p**2       # illustrative


PZ_I = P_I / z_of_p(P_I)

# synthetic pressure surveys along the true depletion line
gp = np.arange(0, 61, 5.0)
p_true = np.array([brentq(lambda p: p / z_of_p(p) - PZ_I * (1 - g / G),
                          30, 5100) for g in gp])
p_meas = p_true + rng.normal(0, 22, p_true.size)
pz_meas = p_meas / z_of_p(p_meas)

fit = stats.linregress(gp, pz_meas)
g_hat = -fit.intercept / fit.slope

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.9, 2.9))

gg = np.linspace(0, g_hat, 50)
ax.plot(gg, fit.intercept + fit.slope * gg, color="C1", lw=1.0)
ax.plot(gp, pz_meas, "o", ms=4, mfc="white", mec="C0", mew=1.0, zorder=5)
ax.plot(gg[gg > 60], fit.intercept + fit.slope * gg[gg > 60], color="C1",
        lw=1.0)
ax.plot(g_hat, 0, "s", ms=4.5, mfc="white", mec="C1", mew=1.2,
        clip_on=False, zorder=6)
ax.annotate(rf"OGIP:  $G$ = {g_hat:.0f} Bscf", (g_hat, 0),
            xytext=(-6, 14), textcoords="offset points", fontsize=7,
            color="C1", ha="right")

# water-drive contrast
gp_wd = np.linspace(0, 78, 100)
pz_wd = PZ_I * (1 - gp_wd / G) * (1 + 0.35 * gp_wd / G)
ax.plot(gp_wd, pz_wd, ls="--", lw=1.0, color="C2")
slope0 = -0.65 * PZ_I / G                          # early-time tangent
g_app = 1 / 0.65 * G
gt = np.linspace(0, g_app, 50)
ax.plot(gt, PZ_I + slope0 * gt, ls=":", lw=0.8, color="C2")
ax.plot(g_app, 0, "s", ms=4, mfc="white", mec="C2", mew=1.0,
        clip_on=False, zorder=6)
ax.annotate(rf"apparent $G$ $\approx$ {g_app:.0f} (wrong)", (g_app, 0),
            xytext=(6, 34), textcoords="offset points", fontsize=6.5,
            color="C2", ha="right")
ax.text(52, 3350, "water drive:\npressure support\nbends the trend",
        fontsize=6.5, color="C2")
ax.text(17, 2720, "volumetric depletion:\nexactly linear", fontsize=6.5,
        color="C0", ha="center")

ax.set_xlim(0, 165)
ax.set_ylim(0, 6000)
ax.set_xlabel(r"cumulative production, $G_p$ (Bscf)")
ax.set_ylabel(r"$p/Z$ (psia)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig097_pz_material_balance.{ext}")
print(f"saved fig097_pz_material_balance.png / .pdf  (G_hat = {g_hat:.1f})")
