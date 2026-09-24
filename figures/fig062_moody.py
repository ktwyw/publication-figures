"""Fig. 62 - Moody diagram, computed rather than traced.

The most reproduced chart in fluids engineering, regenerated from its
governing relations: the laminar solution f = 64/Re, and the Colebrook
equation for turbulent flow,

    1/sqrt(f) = -2 log10( (eps/D)/3.7 + 2.51/(Re sqrt(f)) ),

solved for a family of relative roughnesses by vectorised fixed-point
iteration. Direct right-edge labels replace a legend; the laminar branch
carries a slope -1 triangle and the transition band is shaded.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

import journal_style as js

js.apply()
HERE = js.HERE


def colebrook(re, rel_rough, iters=60):
    """Darcy friction factor via fixed-point iteration on x = 1/sqrt(f)."""
    x = 2.0 * np.log10(re) - 0.8                    # smooth-pipe first guess
    for _ in range(iters):
        x = -2.0 * np.log10(rel_rough / 3.7 + 2.51 * x / re)
    return 1.0 / x**2


# ------------------------------------------------------------- DATA ----
re_lam = np.logspace(np.log10(600), np.log10(2300), 60)
f_lam = 64.0 / re_lam

re_turb = np.logspace(np.log10(4000), 8, 400)
roughness = [0.0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 3e-2]

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.grid(True, which="both", lw=0.3, color="0.90")
ax.set_axisbelow(True)

ax.axvspan(2300, 4000, color="0.93", lw=0, zorder=0)
ax.text(np.sqrt(2300 * 4000), 0.096, "transition", rotation=90, ha="center",
        va="top", fontsize=7, color="0.4")

ax.plot(re_lam, f_lam, color="black", lw=1.1)
ax.text(1420, 0.055, r"$f = 64/Re$", rotation=-70, fontsize=8,
        ha="center", va="center")

def rr_label(rr):
    e = int(np.floor(np.log10(rr)))
    m = rr / 10**e
    return rf"$10^{{{e}}}$" if np.isclose(m, 1) else rf"${m:g}{{\times}}10^{{{e}}}$"

tform = ax.get_yaxis_transform()
for rr in roughness:
    f = colebrook(re_turb, rr)
    ax.plot(re_turb, f, color="black", lw=0.9)
    if rr == 0:
        ax.text(2.3e6, 0.0104, "smooth pipe", rotation=-8, fontsize=6.5,
                ha="center", va="bottom")
    else:
        ax.text(1.01, f[-1], rr_label(rr), transform=tform, va="center",
                fontsize=6.5)
ax.text(1.14, 0.5, r"relative roughness $\epsilon/D$", transform=ax.transAxes,
        rotation=90, va="center", fontsize=8)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(6e2, 1e8)
ax.set_ylim(8e-3, 0.11)
ax.set_yticks([0.01, 0.02, 0.03, 0.04, 0.06, 0.1])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"Reynolds number, $Re = \rho V D/\mu$")
ax.set_ylabel(r"Darcy friction factor, $f$")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig062_moody.{ext}")
print("saved fig062_moody.png / .pdf")
