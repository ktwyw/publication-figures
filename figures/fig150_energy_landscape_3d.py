"""Fig. 150 - Energy landscape: 3-D surface and map (double column, 183 mm).

When a 3-D view earns its place: the surface gives the shape of the
Müller-Brown potential at a glance, and the map beside it, in the same
colour bands, carries everything quantitative: the three minima and the
two saddle points located numerically, their energies, and the
minimum-energy path traced by steepest descent from each saddle. The
self-check is that the analytic gradient vanishes (< 1e-8) at all five
stationary points, that the analytic Hessian has two positive
eigenvalues at each minimum and exactly one negative at each saddle,
that the gradient agrees with finite differences at a random point,
that each descent path ends on a minimum, and that the minimum energies
match the literature (−146.70, −108.17, −80.77) to 0.01.

Model: V(x, y) = Σ Aₖ exp[aₖ(x − xₖ)² + bₖ(x − xₖ)(y − yₖ) + cₖ(y − yₖ)²]
over k = 1..4, with the parameters of Müller & Brown (1979); energies
above 20 are drawn at the ceiling colour; minima by BFGS with a Newton
polish, saddles by a root of the gradient, paths by LSODA on
dx/dt = −∇V. There is no noise: the one random draw is the
finite-difference test point. All data are simulated.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, Normalize
from scipy import integrate, optimize

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(150)

# -------------------------------------------------- GOVERNING MODEL ----
# Müller & Brown, Theor. Chim. Acta 53, 75-93 (1979)
AMP = np.array([-200.0, -100.0, -170.0, 15.0])
A = np.array([-1.0, -1.0, -6.5, 0.7])
B = np.array([0.0, 0.0, 11.0, 0.6])
C = np.array([-10.0, -10.0, -6.5, 0.7])
X0 = np.array([1.0, 0.0, -0.5, -1.0])
Y0 = np.array([0.0, 0.5, 1.5, 1.0])
X_LIM, Y_LIM = (-1.5, 1.2), (-0.3, 2.0)
E_CEIL, E_STEP = 20.0, 10.0                     # colour ceiling, band width
MINIMA_START = {"A": (-0.6, 1.4), "B": (0.6, 0.0), "C": (-0.1, 0.5)}
SADDLE_START = {"S1": (-0.8, 0.6), "S2": (0.2, 0.3)}     # rough guesses
LITERATURE = {"A": -146.6995, "B": -108.1667, "C": -80.7678}


def terms(x, y):
    dx, dy = np.subtract.outer(x, X0), np.subtract.outer(y, Y0)
    return dx, dy, AMP * np.exp(A * dx ** 2 + B * dx * dy + C * dy ** 2)


def energy(point):
    return terms(*point)[2].sum(axis=-1)


def gradient(point):
    dx, dy, e = terms(*point)
    return np.array([(e * (2 * A * dx + B * dy)).sum(axis=-1),
                     (e * (B * dx + 2 * C * dy)).sum(axis=-1)])


def hessian(point):
    dx, dy, e = terms(*point)
    gx, gy = 2 * A * dx + B * dy, B * dx + 2 * C * dy
    hxy = (e * (gx * gy + B)).sum()
    return np.array([[(e * (gx ** 2 + 2 * A)).sum(), hxy],
                     [hxy, (e * (gy ** 2 + 2 * C)).sum()]])


# ------------------------------------------------------------ SOLVER ---
def newton(start):
    """Root of the analytic gradient, with the Hessian as its Jacobian."""
    return optimize.root(gradient, start, jac=hessian, tol=1e-13).x


# minimize finds each basin but stops at its line-search precision; the
# Newton polish takes the gradient on to rounding error
minima = {name: newton(optimize.minimize(energy, start, jac=gradient,
                                         method="BFGS").x)
          for name, start in MINIMA_START.items()}
saddles = {name: newton(start) for name, start in SADDLE_START.items()}

# minimum-energy path: from each saddle, a small step either way along the
# unstable direction, then steepest descent until the gradient has died
paths, path_ends = {}, {}
for name, saddle in saddles.items():
    unstable = np.linalg.eigh(hessian(saddle))[1][:, 0]
    branches = [integrate.solve_ivp(lambda _t, p: -gradient(p), (0, 1.0),
                                    saddle + sign * 1e-4 * unstable,
                                    method="LSODA", rtol=1e-10, atol=1e-12).y
                for sign in (-1, 1)]
    paths[name] = np.hstack([branches[0][:, ::-1], branches[1]])
    path_ends[name] = [branch[:, -1] for branch in branches]

x_grid, y_grid = np.meshgrid(np.linspace(*X_LIM, 160),
                             np.linspace(*Y_LIM, 160))
v_grid = terms(x_grid, y_grid)[2].sum(axis=-1)

# ------------------------------------------------------- SELF-CHECK ---
for point in minima.values():
    assert np.linalg.norm(gradient(point)) < 1e-8
    assert np.all(np.linalg.eigvalsh(hessian(point)) > 0)
for point in saddles.values():
    assert np.linalg.norm(gradient(point)) < 1e-8
    assert np.sum(np.linalg.eigvalsh(hessian(point)) < 0) == 1
probe, h = rng.uniform([X_LIM[0], Y_LIM[0]], [X_LIM[1], Y_LIM[1]]), 1e-6
finite = np.array([(energy(probe + h * e) - energy(probe - h * e)) / (2 * h)
                   for e in np.eye(2)])
assert np.allclose(finite, gradient(probe), rtol=1e-6, atol=1e-6)
for name, value in LITERATURE.items():
    assert abs(energy(minima[name]) - value) < 0.01, (name, value)
connects = {}                # each branch must end on a minimum
for name, ends in path_ends.items():
    reached = [m for end in ends for m, point in minima.items()
               if np.linalg.norm(point - end) < 1e-6]
    connects[name] = "-".join(sorted(reached))
assert connects == {"S1": "A-C", "S2": "B-C"}, connects
fd_error = np.abs(finite - gradient(probe)).max()
print("fig150: self-check passed (minima "
      + ", ".join(f"{energy(p):.2f}" for p in minima.values())
      + "; saddles " + ", ".join(f"{energy(p):.2f}" for p in saddles.values())
      + f"; gradient vs finite difference {fd_error:.1e}; paths S1: A-C, "
      "S2: B-C)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 82)
ax_a = ms.axes(fig, 9, 1, 84, 80, projection="3d")
ax_b = ms.axes(fig, 104, 12, 58, 63)
ax_cbar = ms.axes(fig, 165, 12, 2.6, 63)
levels = np.arange(-150.0, E_CEIL + 1, E_STEP)
cmap = plt.get_cmap("YlGnBu_r")
norm = BoundaryNorm(levels, cmap.N, extend="max")     # bands of the map
smooth = Normalize(levels[0], E_CEIL)                 # same scale, unbanded

# a, the surface, flat above the ceiling so that the basins stay visible;
# every other band edge up to -30 as contour lines on the floor
z_floor = -260.0
ax_a.plot_surface(x_grid, y_grid, np.minimum(v_grid, E_CEIL), cmap=cmap,
                  norm=smooth, rstride=1, cstride=1, linewidth=0,
                  antialiased=False, rasterized=True)
ax_a.contour(x_grid, y_grid, v_grid, levels=levels[:13:2], zdir="z",
             offset=z_floor, cmap=cmap, norm=norm, linewidths=0.5)
ax_a.view_init(elev=30, azim=-125)
ax_a.set_box_aspect((1.15, 1.0, 0.75), zoom=0.98)
ax_a.set_xlim(*X_LIM)
ax_a.set_ylim(*Y_LIM)
ax_a.set_zlim(z_floor, E_CEIL)
ax_a.set_xticks([-1, 0, 1])
ax_a.set_yticks([0, 1, 2])
ax_a.set_zticks([-150, -100, -50, 0])
ax_a.set_xlabel("x", labelpad=-8)
ax_a.set_ylabel("y", labelpad=-8)
ax_a.zaxis.set_rotate_label(False)
ax_a.set_zlabel("Energy V", labelpad=-2, rotation=90)
ax_a.tick_params(pad=-1, labelsize=ms.FS_TICK)
ax_a.grid(False)
for axis in (ax_a.xaxis, ax_a.yaxis, ax_a.zaxis):       # bare, pale panes
    axis.pane.fill = False
    axis.pane.set_edgecolor(ms.GREY_LIGHT)
    axis.line.set_color(ms.GREY)
    axis.line.set_linewidth(0.5)

# b, the map: bands only (no contour strokes under the labels)
bands = ax_b.contourf(x_grid, y_grid, v_grid, levels=levels, cmap=cmap,
                      norm=norm, extend="max", zorder=-1)
ax_b.set_rasterization_zorder(0)          # the bands, and only the bands
for path in paths.values():
    ax_b.plot(*path, color=ms.VERMILLION, lw=1.2, zorder=3)
# labels sit on paler ground beside each point; the leader leaves from the
# corner or edge of the label named by its alignment
LABEL_AT = {"A": (0.0, 1.5, "left", "center"),
            "B": (0.8, 0.55, "center", "bottom"),
            "C": (-0.42, 0.22, "right", "top"),
            "S1": (-0.93, 0.49, "right", "top"),
            "S2": (0.2, -0.03, "center", "top")}
FRACTION = {"left": 0.0, "bottom": 0.0, "center": 0.5, "right": 1.0,
            "top": 1.0}
for kind, points, marker, size in (("Minimum", minima, "o", 4.2),
                                   ("Saddle", saddles, "D", 3.6)):
    for name, point in points.items():
        x_text, y_text, ha, va = LABEL_AT[name]
        ax_b.plot(*point, marker, ms=size, mfc="white", mec=ms.INK, mew=0.7,
                  zorder=4)
        label = f"{kind} {name}\n{energy(point):.2f}".replace("-", "−")
        ax_b.annotate(label, xy=point, xytext=(x_text, y_text), ha=ha, va=va,
                      fontsize=ms.FS_TICK, linespacing=1.15,
                      multialignment="left",
                      arrowprops=dict(arrowstyle="-", color=ms.INK, lw=0.5,
                                      shrinkA=3, shrinkB=3,
                                      relpos=(FRACTION[ha], FRACTION[va])))
ax_b.text(0.975, 0.965, "Minimum-energy path\n(steepest descent)",
          transform=ax_b.transAxes, color=ms.VERMILLION, ha="right", va="top",
          fontsize=ms.FS_TICK)
ax_b.set_xlim(*X_LIM)
ax_b.set_ylim(*Y_LIM)
ax_b.set_xticks(np.arange(-1.5, 1.01, 0.5))
ax_b.set_yticks(np.arange(0, 2.01, 0.5))
ax_b.set_xlabel("x")
ax_b.set_ylabel("y")
for side in ("top", "right"):
    ax_b.spines[side].set_visible(True)

cbar = fig.colorbar(bands, cax=ax_cbar, ticks=np.arange(-150, E_CEIL + 1, 50))
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(length=2, width=0.5)
cbar.set_label("Energy V (arbitrary units)")

ms.panel_label(ax_b, "b", dx_pt=-28)
# a 3-D axes has no plot-area corner: its letter is set level with b
fig.text(4 / 183, 76.4 / 82, "a", fontsize=ms.FS_PANEL, fontweight="bold",
         color="black", ha="left", va="bottom")
ms.assert_aligned([ax_b, ax_cbar], edges=("bottom",))   # top: the arrow tip
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig150_energy_landscape_3d.{ext}")
print("fig150_energy_landscape_3d: saved png + pdf")
