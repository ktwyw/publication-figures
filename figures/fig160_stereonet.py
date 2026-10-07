"""Fig. 160 - Equal-area stereonet with pole density (single column, 89 mm).

A lower-hemisphere Schmidt net drawn from the projection equations:
primitive circle, a 10° graticule of great and small circles, the poles
of two joint sets as points, their density contoured on the sphere in
multiples of a uniform distribution, and for each set the mean pole
(principal eigenvector of the orientation matrix) with the great circle
of the mean plane. The self-check is that the projection is equal-area
(the projected area of the cap within θ of the centre is half its area
on the unit sphere for every θ tested, and the contoured density
averages to 1 over the net), that every point of each great circle is
90° from its pole (1e-9), and that each recovered mean pole lies within
5° of the simulated one.

Model: x = e/√(1 + d), y = n/√(1 + d) for a unit vector (east, north,
down) with d ≥ 0, so the primitive circle has radius 1. Data: n = 35 and
n = 25 poles drawn from Fisher distributions (κ = 35 and 20) about mean
poles 245°/30° and 110°/55° (trend/plunge). Density: exponential kernel
exp[k(|cos ψ| − 1)], k = 30, normalized to a mean of 1 over the
hemisphere. All data are simulated.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(160)

# ------------------------------------------------------------- DATA ----
# name: (trend°, plunge° of the mean pole, Fisher κ, n, colour, marker)
SETS = {"Set 1": (245.0, 30.0, 35.0, 35, ms.BLUE, "o"),
        "Set 2": (110.0, 55.0, 20.0, 25, ms.VERMILLION, "^")}
K_SMOOTH = 30.0                                # kernel concentration
LEVELS = [2.0, 4.0, 6.0, 8.0]                  # multiples of uniform density


def line_vector(trend, plunge):
    """Unit vector (east, north, down) of a line, angles in degrees."""
    tr, pl = np.radians(trend), np.radians(plunge)
    return np.array([np.cos(pl) * np.sin(tr), np.cos(pl) * np.cos(tr),
                     np.sin(pl)])


def fisher_sample(mean, kappa, n):
    """n unit vectors from the Fisher distribution about mean."""
    u = rng.uniform(size=n)
    cos_t = 1 + np.log(u + (1 - u) * np.exp(-2 * kappa)) / kappa
    sin_t, phi = np.sqrt(1 - cos_t ** 2), rng.uniform(0, 2 * np.pi, n)
    a = np.cross(mean, [0.0, 0.0, 1.0])
    a /= np.linalg.norm(a)
    b = np.cross(mean, a)
    return (np.outer(cos_t, mean) + np.outer(sin_t * np.cos(phi), a)
            + np.outer(sin_t * np.sin(phi), b))


poles = {name: fisher_sample(line_vector(trend, plunge), kappa, n)
         for name, (trend, plunge, kappa, n, _c, _m) in SETS.items()}


# -------------------------------------------------- GOVERNING MODEL ----
def project(v):
    """Lower-hemisphere Lambert equal-area projection of unit vectors."""
    v = np.atleast_2d(v)
    v = np.where(v[:, 2:3] < 0, -v, v)         # an axis: use its lower end
    return (v[:, :2] / np.sqrt(1 + v[:, 2:3])).T


def unproject(x, y):
    """Inverse projection: net coordinates to (east, north, down)."""
    r = np.hypot(x, y)
    down = 1 - np.minimum(r, 1.0) ** 2         # beyond the net: its edge
    scale = np.sqrt(1 + down) / np.maximum(r, 1.0)
    return np.stack([x * scale, y * scale, down], -1)


def great_circle(pole, n_points=181):
    """Lower half of the plane normal to pole, from strike to strike."""
    strike = np.cross(pole, [0.0, 0.0, 1.0])
    strike /= np.linalg.norm(strike)
    dip = np.cross(pole, strike)
    dip *= np.sign(dip[2])
    phi = np.linspace(0, np.pi, n_points)[:, None]
    return np.cos(phi) * strike + np.sin(phi) * dip


# -------------------------------------------------------- ESTIMATOR ----
means = {}
for name, v in poles.items():
    orientation = v.T @ v / len(v)             # orientation matrix
    axis = np.linalg.eigh(orientation)[1][:, -1]
    means[name] = axis * np.sign(axis[2])
all_poles = np.vstack(list(poles.values()))
grid = np.linspace(-1, 1, 241)
gx, gy = np.meshgrid(grid, grid)
cos_psi = np.abs(unproject(gx, gy) @ all_poles.T)        # axes, hence |cos|
norm = K_SMOOTH / (1 - np.exp(-K_SMOOTH))      # hemisphere mean of kernel = 1
density = norm * np.exp(K_SMOOTH * (cos_psi - 1)).mean(axis=-1)

# ------------------------------------------------------- SELF-CHECK ---
azimuth = np.radians(np.arange(0, 360, 30))
for theta in np.radians([5, 20, 45, 70, 90]):
    ring = np.stack([np.sin(theta) * np.sin(azimuth),
                     np.sin(theta) * np.cos(azimuth),
                     np.full(azimuth.size, np.cos(theta))], axis=1)
    radius = np.hypot(*project(ring))
    cap_on_sphere = 2 * np.pi * (1 - np.cos(theta))
    assert np.allclose(np.pi * radius ** 2 / cap_on_sphere, 0.5, rtol=1e-12)
inside = gx ** 2 + gy ** 2 <= 1              # equal cells, equal solid angle
mean_density = density[inside].mean()
assert abs(mean_density - 1) < 0.01, mean_density
worst_dot, recovered = 0.0, {}
for name, (trend, plunge, *_rest) in SETS.items():
    worst_dot = max(worst_dot,
                    np.abs(great_circle(means[name]) @ means[name]).max())
    cosine = abs(means[name] @ line_vector(trend, plunge))
    recovered[name] = np.degrees(np.arccos(min(cosine, 1.0)))
    assert recovered[name] < 5.0, (name, recovered[name])
assert worst_dot < 1e-9, worst_dot
assert density.max() > LEVELS[-1], density.max()
print("fig160: self-check passed (cap area ratio 0.5 at five angles; mean "
      f"density over the net {mean_density:.4f}; great circles within "
      f"{worst_dot:.1e} of 90° from their poles; mean poles "
      + ", ".join(f"{a:.1f}°" for a in recovered.values())
      + f" from simulated; peak density {density.max():.1f} x uniform)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 96)
ax = ms.axes(fig, 9.5, 21, 70, 70)
ax_cbar = ms.axes(fig, 56, 9.5, 27, 2.4)
LIM = 1.09                                     # room for the edge ticks
ax.set_xlim(-LIM, LIM)
ax.set_ylim(-LIM, LIM)
ax.set_aspect("equal")
ax.axis("off")

greys = plt.get_cmap("Greys")(np.linspace(0.16, 0.5, len(LEVELS)))
shades = ListedColormap(greys[:-1]).with_extremes(over=greys[-1])
filled = ax.contourf(gx, gy, density, levels=LEVELS, extend="max",
                     cmap=shades, norm=BoundaryNorm(LEVELS, shades.N),
                     zorder=1)
filled.set_clip_path(Circle((0, 0), 1.0, transform=ax.transData))

# graticule: great circles through N and S, small circles about that axis
north, east, down = np.eye(3)[[1, 0, 2]]
phi = np.linspace(0, np.pi, 121)[:, None]
for angle in np.radians(np.arange(10, 180, 10)):
    meridian = great_circle(np.cos(angle) * east + np.sin(angle) * down, 121)
    cone = (np.cos(angle) * north + np.sin(angle)
            * (np.cos(phi) * east + np.sin(phi) * down))
    for curve in (meridian, cone):
        ax.plot(*project(curve), color=ms.GREY_LIGHT, lw=0.35, zorder=2)
circle = np.radians(np.linspace(0, 360, 361))
ax.plot(np.sin(circle), np.cos(circle), color=ms.INK, lw=0.7, zorder=6)
for bearing in np.arange(0, 360, 10):
    length = 0.035 if bearing % 90 == 0 else 0.018
    ux, uy = np.sin(np.radians(bearing)), np.cos(np.radians(bearing))
    ax.plot([ux, ux * (1 + length)], [uy, uy * (1 + length)], color=ms.INK,
            lw=0.6 if bearing % 90 == 0 else 0.4, solid_capstyle="butt")
for label, bearing in zip("NESW", (0, 90, 180, 270)):
    ax.text(1.085 * np.sin(np.radians(bearing)),
            1.085 * np.cos(np.radians(bearing)), label, ha="center",
            va="center", fontsize=ms.FS_BODY, clip_on=False)

handles = []
for name, (_tr, _pl, _kappa, n, colour, marker) in SETS.items():
    ax.plot(*project(poles[name]), marker, ms=3.0, mfc=colour, mec="white",
            mew=0.3, ls="none", zorder=4)
    ax.plot(*project(great_circle(means[name])), color=colour, lw=1.1,
            zorder=3)
    ax.plot(*project(means[name]), marker, ms=6.0, mfc="white", mec=colour,
            mew=1.2, zorder=5)
    mean_trend = np.degrees(np.arctan2(means[name][0], means[name][1])) % 360
    mean_plunge = np.degrees(np.arcsin(means[name][2]))
    handles.append(Line2D([], [], marker=marker, ms=3.4, mfc=colour,
                          mec="white", mew=0.3, ls="none",
                          label=f"{name} poles, n = {n}; mean "
                          f"{mean_trend:03.0f}°/{mean_plunge:02.0f}°"))
handles += [Line2D([], [], marker="o", ms=5.0, mfc="white", mec=ms.INK,
                   mew=1.0, ls="none", label="Mean pole (open symbol)"),
            Line2D([], [], color=ms.INK, lw=1.1, label="Mean plane")]
fig.legend(handles=handles, loc="lower left",
           bbox_to_anchor=(4.5 / 89, 2.5 / 96), handlelength=1.6,
           fontsize=ms.FS_TICK)

cbar = fig.colorbar(filled, cax=ax_cbar, orientation="horizontal",
                    ticks=LEVELS)
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(length=2, width=0.5)
cbar.ax.xaxis.set_major_formatter(lambda v, _pos: f"{v:g}")
cbar.set_label("Pole density (× uniform)", fontsize=ms.FS_TICK)
fig.text(84.5 / 89, 16.5 / 96, "Lower hemisphere, equal area",
         ha="right", va="bottom", fontsize=ms.FS_TICK, color=ms.GREY_DARK)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig160_stereonet.{ext}")
print("fig160_stereonet: saved png + pdf")
