"""Fig. 98 - Terzaghi one-dimensional consolidation (two panels).

The founding solution of soil mechanics, from the diffusion series for
a doubly drained layer of thickness 2H:

    u/u0 = sum (2/M) sin(M z/H) exp(-M^2 T_v),   M = (2m+1) pi/2.

(a) Excess pore-pressure isochrones at increasing time factor T_v,
    depth drawn downward, drainage at both boundaries.
(b) Average degree of consolidation U(T_v) from the exact series, with
    the parabolic early-time approximation U = sqrt(4 T_v / pi) and the
    design milestones T_50 = 0.197 and T_90 = 0.848 marked.
"""

import numpy as np
import matplotlib.pyplot as plt

import journal_style as js

js.apply()
HERE = js.HERE

M = (2 * np.arange(0, 120) + 1) * np.pi / 2        # series terms


def u_iso(zeta, tv):
    return np.sum(2.0 / M * np.sin(np.outer(zeta, M))
                  * np.exp(-(M**2) * tv), axis=1)


def u_avg(tv):
    tv = np.atleast_1d(tv)
    return 1 - np.sum(2.0 / M**2 * np.exp(-np.outer(tv, M**2)), axis=1)


fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.0))

# ------------------------------------------------- (a) isochrones ----
zeta = np.linspace(0, 2, 300)
CASES = [(0.05, "C0", 0.32), (0.1, "C1", 0.52), (0.2, "C2", 0.75),
         (0.5, "C3", 1.0), (0.9, "C4", 1.0)]
for tv, color, z_lab in CASES:
    u = u_iso(zeta, tv)
    ax_a.plot(u, zeta, color=color, lw=1.1)
    x_lab = np.interp(z_lab, zeta, u)
    ax_a.text(x_lab + 0.018, z_lab, f"{tv:g}", fontsize=6, color=color,
              va="center")
ax_a.text(0.055, 0.14, "$T_v$ =", fontsize=6, color="0.35")

ax_a.invert_yaxis()
ax_a.set_xlim(0, 1.04)
ax_a.set_ylim(2, 0)
ax_a.set_xlabel(r"excess pore pressure, $u/u_0$")
ax_a.set_ylabel(r"depth, $z/H$")
ax_a.text(0.64, 0.055, "drainage boundary", fontsize=6, color="0.45",
          ha="center", va="top")
ax_a.text(0.64, 1.945, "drainage boundary", fontsize=6, color="0.45",
          ha="center", va="bottom")

# ------------------------------------------------- (b) U vs Tv ----
tv = np.logspace(-3, np.log10(3), 300)
ax_b.semilogx(tv, u_avg(tv), color="C0", lw=1.3)
tv_p = tv[tv < 0.35]
ax_b.semilogx(tv_p, np.sqrt(4 * tv_p / np.pi), ls="--", lw=0.9,
              color="0.45")
ax_b.text(0.0035, 0.115, r"$U = \sqrt{4T_v/\pi}$", fontsize=6.5,
          color="0.45", rotation=33)

for t_ref, u_ref, lab in [(0.197, 0.5, "$T_{50}$ = 0.197"),
                          (0.848, 0.9, "$T_{90}$ = 0.848")]:
    ax_b.plot([t_ref, t_ref, 1e-3], [0, u_ref, u_ref], ls=":", lw=0.7,
              color="C3")
    ax_b.text(t_ref * 1.12, 0.035, lab, fontsize=6.5, color="C3")

ax_b.set_xlim(1e-3, 3)
ax_b.set_ylim(0, 1.02)
ax_b.set_xlabel(r"time factor, $T_v = c_v t / H^2$")
ax_b.set_ylabel(r"average consolidation, $U$")

js.panel_label(ax_a, "a", x=-0.16)
js.panel_label(ax_b, "b", x=-0.18)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig098_terzaghi_consolidation.{ext}")
print("saved fig098_terzaghi_consolidation.png / .pdf")
