"""Fig. 61 - Carreau-fluid channel flow curves with asymptotes (double column).

Reproduces the structure of a classic rheology figure: dimensionless
pressure drop Cu = dp*lambda*h/(eta0*L) vs dimensionless flow rate
Q = lambda*q/(w*h^2) for pressure-driven slit flow of a Carreau fluid,

    eta(g) = eta_inf + (eta0 - eta_inf) * [1 + (lambda*g)^2]^((n-1)/2).

All curves are COMPUTED from the model (wall-shear integral for slit
flow), not digitized from any source:
    Q = (1/(2 s_w^2)) * Integral_0^{s_w} s g(s) ds,   Cu = 2 s_w,
with s = lambda*tau/eta0 and g = lambda*gammadot, evaluated by
parameterising in g so no inversion is needed. Asymptotes are two-term
expansions of the same integral:
    small Cu:  Q = (Cu/12) [1 + 3(1-n)/40 * Cu^2]
    power law: Cu = 2 [2(2n+1) Q / n]^n
    large Cu:  Q = Cu/(12 beta) - (1-beta)(Cu/2)^n / [2(n+2) beta^(n+1)]

Journal-style overrides (boxed axes, inward ticks, STIX serif math,
slope triangles, dual legends) are applied locally on top of the shared
style sheet - a useful template for theory papers.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import cumulative_trapezoid

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": True, "axes.spines.right": True,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "axes.linewidth": 0.9,
})

C_PL, C_LG, C_SM = "red", "deepskyblue", "rebeccapurple"


# ------------------------------------------------------------ theory ----
def flow_curve(n, beta):
    """Exact slit-flow curve, parameterised by dimensionless shear rate g."""
    g = np.logspace(-8, 11, 8000)
    s = (beta + (1 - beta) * (1 + g**2) ** ((n - 1) / 2)) * g
    integral = cumulative_trapezoid(s * g, s, initial=0.0)
    Q, Cu = integral / (2 * s**2), 2 * s
    return Q[1:], Cu[1:]                              # drop the Q = 0 point


def markers_on(Q, Cu, q_targets):
    lg = np.log10(q_targets)
    cu = 10 ** np.interp(lg, np.log10(Q), np.log10(Cu))
    m = cu <= 1.2e4
    return q_targets[m], cu[m]


def power_law_cu(Q, n):
    return 2.0 * (2 * (2 * n + 1) * Q / n) ** n


def small_cu_asym(n, cu_max=6.5):
    cu = np.logspace(-2, np.log10(cu_max), 200)
    return (cu / 12) * (1 + 3 * (1 - n) / 40 * cu**2), cu


def large_cu_asym(n, beta, cu_min):
    cu = np.logspace(np.log10(cu_min), 4, 300)
    q = cu / (12 * beta) - (1 - beta) * (cu / 2) ** n / (2 * (n + 2)
                                                        * beta ** (n + 1))
    m = q > 0
    return q[m], cu[m]


def slope_triangle(ax, x0, y0, dx, slope, color, rise, run):
    x1, y1 = x0 * 10**dx, y0 * 10 ** (dx * slope)
    ax.plot([x0, x1, x1, x0], [y0, y0, y1, y0], color=color, lw=1.1,
            zorder=5, clip_on=False)
    ax.text(np.sqrt(x0 * x1), y0 * 0.82, run, ha="center", va="top",
            fontsize=9, color=color)
    ax.text(x1 * 1.35, np.sqrt(y0 * y1), rise, ha="left", va="center",
            fontsize=9, color=color)


MSTYLES = [dict(marker="o", mfc="black", mec="black", ms=3.6),
           dict(marker="x", mfc="none", mec="0.35", ms=4.2, mew=1.1),
           dict(marker="o", mfc="none", mec="0.70", ms=4.2, mew=1.0)]
Q_TARG = np.logspace(-3, 7, 34)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)

# ------------------------------------------------------- panel (a) ----
BETA_A = 1e-4
for n, st in zip([0.25, 0.5, 0.75], MSTYLES):
    Qf = np.logspace(np.log10(3e-2), 7, 200)
    ax_a.plot(Qf, power_law_cu(Qf, n), ls=(0, (5, 3)), lw=1.0, color=C_PL,
              zorder=2)
    Q, Cu = flow_curve(n, BETA_A)
    qm, cm = markers_on(Q, Cu, Q_TARG)
    ax_a.plot(qm, cm, ls="none", zorder=3, **st, label=f"$n = {n}$")

q_lg, cu_lg = large_cu_asym(0.25, BETA_A, cu_min=68)
ax_a.plot(q_lg, cu_lg, ls=":", lw=1.6, color=C_LG, zorder=2)
q_sm, cu_sm = small_cu_asym(0.25)
ax_a.plot(q_sm, cu_sm, ls="-", lw=1.3, color=C_SM, zorder=4)

slope_triangle(ax_a, 1.2e-2, 2.4e-2, 0.9, 1.0, "black", "1", "1")
slope_triangle(ax_a, 1.0e1, 3.2, 2.3, 0.25, C_PL, "$n$", "1")
slope_triangle(ax_a, 1.2e5, 9.0e1, 1.05, 1.0, C_LG, "1", "1")

leg1 = ax_a.legend(loc="upper left", frameon=False, numpoints=3,
                   handlelength=2.4, handletextpad=0.6, borderaxespad=0.2)
ax_a.add_artist(leg1)
ax_a.text(0.19, 0.66, r"$\beta = 10^{-4}$", transform=ax_a.transAxes)
asym_handles = [Line2D([], [], color=C_SM, ls="-", lw=1.3,
                       label="Small-$Cu$ asymptote"),
                Line2D([], [], color=C_LG, ls=":", lw=1.6,
                       label="Large-$Cu$ asymptote"),
                Line2D([], [], color=C_PL, ls=(0, (5, 3)), lw=1.0,
                       label="Power-law asymptote")]
ax_a.legend(handles=asym_handles, loc="lower right", frameon=False,
            bbox_to_anchor=(0.99, 0.02), handlelength=2.4,
            borderaxespad=0.2)
ax_a.set_ylabel(r"$Cu = \Delta p\,\lambda h/(\eta_0 \ell)$")

# ------------------------------------------------------- panel (b) ----
N_B = 0.25
Qf = np.logspace(np.log10(3e-2), 7, 200)
ax_b.plot(Qf, power_law_cu(Qf, N_B), ls=(0, (5, 3)), lw=1.0, color=C_PL,
          zorder=2)
for beta, st, expo in zip([1e-4, 1e-3, 1e-2], MSTYLES, [-4, -3, -2]):
    Q, Cu = flow_curve(N_B, beta)
    qm, cm = markers_on(Q, Cu, Q_TARG)
    ax_b.plot(qm, cm, ls="none", zorder=3, **st,
              label=rf"$\beta = 10^{{{expo}}}$")
    cu_min = {1e-4: 68, 1e-3: 32, 1e-2: 15}[beta]
    q_lg, cu_lg = large_cu_asym(N_B, beta, cu_min=cu_min)
    ax_b.plot(q_lg, cu_lg, ls=":", lw=1.6, color=C_LG, zorder=2)
q_sm, cu_sm = small_cu_asym(N_B)
ax_b.plot(q_sm, cu_sm, ls="-", lw=1.3, color=C_SM, zorder=4)

slope_triangle(ax_b, 1.2e-2, 2.4e-2, 0.9, 1.0, "black", "1", "1")
slope_triangle(ax_b, 1.0e1, 3.2, 2.3, 0.25, C_PL, "$n = 0.25$", "1")
slope_triangle(ax_b, 1.2e5, 9.0e1, 1.05, 1.0, C_LG, "1", "1")

leg1 = ax_b.legend(loc="upper left", frameon=False, numpoints=3,
                   handlelength=2.4, handletextpad=0.6, borderaxespad=0.2)
ax_b.add_artist(leg1)
ax_b.text(0.19, 0.66, r"$n = 0.25$", transform=ax_b.transAxes)

# ------------------------------------------------------- shared axes ----
for ax, letter in zip((ax_a, ax_b), "ab"):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 1e7)
    ax.set_ylim(1e-2, 1e4)
    ax.set_xticks([1e-3, 1e-1, 1e1, 1e3, 1e5, 1e7])
    ax.set_yticks([1e-2, 1e0, 1e2, 1e4])
    ax.set_xlabel(r"$Q = \lambda q/wh^2$")
    ax.text(-0.16 if letter == "a" else -0.10, 1.01, f"$({letter})$",
            transform=ax.transAxes, fontsize=12, va="bottom")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig061_carreau_flowcurve.{ext}")
print("saved fig061_carreau_flowcurve.png / .pdf")
