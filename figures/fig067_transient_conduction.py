"""Fig. 67 - Transient conduction in a slab: exact eigenseries.

(a) Centre temperature of a convectively cooled slab from the exact
separation-of-variables series

    theta_0(Fo) = sum_n C_n exp(-zeta_n^2 Fo),
    zeta_n tan(zeta_n) = Bi,   C_n = 4 sin(zeta_n)/(2 zeta_n + sin 2 zeta_n),

with eigenvalues found by brentq on each branch of tan. Dashed lines are
the one-term (Heisler) approximation, valid for Fo > 0.2 (shaded region
marks where more terms are required).

(b) The first eigenvalue vs Biot number with its two asymptotes,
zeta_1 ~ sqrt(Bi) (lumped limit) and zeta_1 -> pi/2 (isothermal-surface
limit).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import journal_style as js

js.apply()
HERE = js.HERE


def zeta_roots(bi, n_terms=60):
    """Roots of z tan(z) = Bi, one per branch (k*pi, k*pi + pi/2)."""
    f = lambda z: z * np.tan(z) - bi
    return np.array([brentq(f, k * np.pi + 1e-9,
                            k * np.pi + np.pi / 2 - 1e-9)
                     for k in range(n_terms)])


def theta0(fo, bi, n_terms=60):
    z = zeta_roots(bi, n_terms)
    c = 4 * np.sin(z) / (2 * z + np.sin(2 * z))
    return (c[:, None] * np.exp(-z[:, None] ** 2 * fo[None, :])).sum(0), z, c


# ------------------------------------------------------------- PLOT ----
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                 width_ratios=[1.25, 1])

fo = np.linspace(1e-3, 3, 500)
bi_list = [(0.1, "0.1"), (1.0, "1"), (10.0, "10"), (1e6, r"\infty")]
tform = ax_a.get_yaxis_transform()
for i, (bi, lab) in enumerate(bi_list):
    th, z, c = theta0(fo, bi)
    ax_a.semilogy(fo, th, color=f"C{i}", lw=1.2)
    ax_a.semilogy(fo, c[0] * np.exp(-z[0] ** 2 * fo), ls="--", lw=0.8,
                  color=f"C{i}")
    ax_a.text(1.02, max(th[-1], 6e-4), rf"$Bi = {lab}$", transform=tform,
              va="center", fontsize=7, color=f"C{i}")

ax_a.axvspan(0, 0.2, color="0.94", lw=0, zorder=0)
ax_a.text(0.145, 8e-4, "multi-term (Fo < 0.2)", rotation=90,
          fontsize=6, color="0.45", ha="center", va="bottom")
ax_a.plot([], [], color="0.2", lw=1.2, label="exact series")
ax_a.plot([], [], color="0.2", lw=0.8, ls="--", label="one-term")
ax_a.legend(loc="lower left", bbox_to_anchor=(0.24, 0.03), frameon=False,
            fontsize=7)

ax_a.set_xlim(0, 3)
ax_a.set_ylim(5e-4, 1.7)
ax_a.set_xlabel(r"Fourier number, $Fo = \alpha t/L^2$")
ax_a.set_ylabel(r"centre temperature, $\theta_0$")

# ------------------------------- (b) first eigenvalue vs Biot ----
bi_grid = np.logspace(-2, 3, 200)
z1 = np.array([zeta_roots(b, 1)[0] for b in bi_grid])
ax_b.loglog(bi_grid, z1, color="black", lw=1.2)
ax_b.loglog(bi_grid, np.sqrt(bi_grid), ls="--", lw=0.9, color="C0")
ax_b.axhline(np.pi / 2, ls=":", lw=0.9, color="C3")
ax_b.text(0.30, 0.26, r"$\zeta_1 \simeq \sqrt{Bi}$", rotation=50,
          fontsize=8, color="C0", ha="center")
ax_b.text(600, np.pi / 2 * 1.12, r"$\pi/2$", fontsize=8, color="C3",
          ha="right")
js.slope_triangle(ax_b, 0.030, 0.085, 0.9, 0.5, "0.35", "1", "2")

ax_b.set_xlim(1e-2, 1e3)
ax_b.set_ylim(0.05, 2.6)
ax_b.set_xlabel(r"Biot number, $Bi = hL/k$")
ax_b.set_ylabel(r"first eigenvalue, $\zeta_1$")

js.panel_label(ax_a, "a", x=-0.13)
js.panel_label(ax_b, "b", x=-0.17)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig067_transient_conduction.{ext}")
print("saved fig067_transient_conduction.png / .pdf")
