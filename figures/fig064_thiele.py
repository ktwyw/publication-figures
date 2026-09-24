"""Fig. 64 - Effectiveness factor vs generalised Thiele modulus.

The reaction-engineering classic: internal effectiveness factor for a
first-order reaction in a slab, an infinite cylinder, and a sphere,
plotted against the generalised Thiele modulus
Lambda = (V_p/S_p) sqrt(k/D_e), under which all three geometries share
the same asymptotes eta -> 1 (kinetic regime) and eta -> 1/Lambda
(diffusion-limited).

    slab:      eta = tanh(Lambda)/Lambda
    cylinder:  eta = I1(2 Lambda) / (Lambda I0(2 Lambda))
    sphere:    eta = (3 Lambda coth(3 Lambda) - 1) / (3 Lambda^2)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0, i1

import journal_style as js

js.apply()
HERE = js.HERE

# ------------------------------------------------------------ theory ----
lam = np.logspace(-1.2, 1.6, 400)
coth = lambda x: 1.0 / np.tanh(x)
eta = {
    "Slab": np.tanh(lam) / lam,
    "Cylinder": i1(2 * lam) / (lam * i0(2 * lam)),
    "Sphere": (3 * lam * coth(3 * lam) - 1) / (3 * lam**2),
}

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.6, 2.9))
ax.plot(lam, np.ones_like(lam), ls=":", lw=0.9, color="0.55")
ax.plot(lam, 1 / lam, ls="--", lw=0.9, color="0.35")
for i, (name, e) in enumerate(eta.items()):
    ax.plot(lam, e, color=f"C{i}", lw=1.2, label=name)

js.slope_triangle(ax, 7.0, 0.055, 0.35, -1.0, "0.35", "1", "1")
ax.text(0.09, 1.06, "kinetic regime", fontsize=7, color="0.4")
ax.text(2.6, 0.60, r"$\eta = 1/\Lambda$", fontsize=8, color="0.35",
        rotation=-47)
ax.text(11, 0.021, "diffusion-limited", fontsize=7, color="0.4", ha="right")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lam[0], lam[-1])
ax.set_ylim(0.017, 1.6)
ax.set_xlabel(r"generalised Thiele modulus, "
              r"$\Lambda = (V_p/S_p)\sqrt{k/D_e}$")
ax.set_ylabel(r"effectiveness factor, $\eta$")
ax.legend(loc="lower left", frameon=False, borderaxespad=0.4)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig064_thiele.{ext}")
print("saved fig064_thiele.png / .pdf")
