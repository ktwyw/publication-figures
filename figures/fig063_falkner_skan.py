"""Fig. 63 - Falkner-Skan boundary layers by shooting (two panels).

(a) Self-similar wedge-flow velocity profiles from the Falkner-Skan
equation

    f''' + f f'' + beta (1 - f'^2) = 0,  f(0) = f'(0) = 0,  f'(inf) = 1,

solved as a boundary-value problem by shooting on the wall shear f''(0)
(solve_ivp + brentq). beta = 0 is the Blasius flat plate; beta < 0 is an
adverse pressure gradient; beta = 1 is plane stagnation flow.

(b) The computed wall shear f''(0) across the whole beta range, with the
classical anchor values marked and the separation point f''(0) -> 0 at
beta ~ -0.1988.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import journal_style as js

js.apply()
HERE = js.HERE
ETA_MAX = 12.0


def shoot(beta, s, dense=False):
    rhs = lambda eta, y: [y[1], y[2], -y[0] * y[2] - beta * (1 - y[1] ** 2)]

    def overshoot(eta, y):          # stop early once the profile diverges;
        return y[1] - 1.8           # keeps the shooting residual's sign

    def undershoot(eta, y):
        return y[1] + 1.0

    overshoot.terminal = undershoot.terminal = True
    return solve_ivp(rhs, [0, ETA_MAX], [0.0, 0.0, s], rtol=1e-7, atol=1e-9,
                     dense_output=dense, events=[overshoot, undershoot])


def wall_shear(beta):
    g = lambda s: shoot(beta, s).y[1, -1] - 1.0
    return brentq(g, 1e-6, 2.6, xtol=1e-10)


# ------------------------------------------------------------- PLOT ----
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                 width_ratios=[1.0, 1.0])

# (a) velocity profiles
betas = [-0.18, -0.10, 0.0, 0.5, 1.0]
eta = np.linspace(0, 6, 300)
for i, beta in enumerate(betas):
    s = wall_shear(beta)
    fp = shoot(beta, s, dense=True).sol(eta)[1]
    ax_a.plot(fp, eta, color=f"C{i}", lw=1.2,
              label=rf"$\beta = {beta:g}$")

ax_a.annotate("", xy=(0.26, 2.55), xytext=(0.60, 1.05),
              arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.4"))
ax_a.text(0.60, 3.35, "increasing adverse\npressure gradient", fontsize=7,
          color="0.3", ha="center")
ax_a.set_xlim(0, 1.04)
ax_a.set_ylim(0, 6)
ax_a.set_xlabel(r"$u/U = f'(\eta)$")
ax_a.set_ylabel(r"similarity variable, $\eta$")
ax_a.legend(loc="upper left", fontsize=7, frameon=False,
            borderaxespad=0.3, handlelength=1.6)

# (b) wall shear vs beta
beta_grid = np.concatenate([np.linspace(-0.195, -0.10, 8),
                            np.linspace(-0.05, 1.0, 12)])
shear = np.array([wall_shear(b) for b in beta_grid])
ax_b.plot(beta_grid, shear, color="black", lw=1.2)
ax_b.axhline(0, lw=0.5, color="0.6")

anchors = [(-0.1988, 0.0, r"separation, $\beta \approx -0.199$", (10, 8)),
           (0.0, 0.4696, r"Blasius, $f''(0) = 0.4696$", (8, -4)),
           (1.0, 1.2326, r"stagnation, $1.2326$", (-8, 6))]
for xb, yb, lab, off in anchors:
    ax_b.plot(xb, yb, "o", ms=4, mfc="white", mec="C3", mew=1.1,
              clip_on=False, zorder=5)
    ha = "right" if off[0] < 0 else "left"
    ax_b.annotate(lab, (xb, yb), xytext=off, textcoords="offset points",
                  fontsize=6.5, color="C3", ha=ha)

ax_b.set_xlim(-0.25, 1.05)
ax_b.set_ylim(-0.08, 1.4)
ax_b.set_xlabel(r"pressure-gradient parameter, $\beta$")
ax_b.set_ylabel(r"wall shear, $f''(0)$")

js.panel_label(ax_a, "a", x=-0.15)
js.panel_label(ax_b, "b", x=-0.15)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig063_falkner_skan.{ext}")
print("saved fig063_falkner_skan.png / .pdf")
