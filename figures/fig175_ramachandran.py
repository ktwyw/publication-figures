"""Fig. 175 - Ramachandran plot with periodic density (single column, 89 mm).

How to draw backbone dihedral angles when both axes are circular: every
residue as a point, over two filled contour levels that enclose 50% and
90% of the residues, from a density estimated on the torus (a histogram
smoothed by a Gaussian that wraps at ±180°) so that a cluster cut by
the plot edge continues on the opposite side. Outliers of the kind
glycine produces get their own marker. The self-check is that the two
contour levels enclose their nominal fractions of the residues within 2
percentage points, that the density has the same values on the −180°
and +180° edges, and that it integrates to 1.

Data: n = 3,000 residues; wrapped bivariate normals for β (46%),
right-handed α (46%) and left-handed Lα (6%) plus 2% outliers uniform
on the torus. Density: 5° bins, wrapped Gaussian smoothing, s.d. 7.5°;
contour levels are the density quantiles over residues. All data are
simulated.
"""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import ndimage

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(175)

# ------------------------------------------------------------- DATA ----
N_RES = 3000
# region: (weight, φ centre, ψ centre, s.d. φ, s.d. ψ, correlation); deg
REGIONS = {"β": (0.46, -115.0, 140.0, 28.0, 24.0, -0.35),
           "α": (0.46, -63.0, -43.0, 11.0, 13.0, -0.55),
           "Lα": (0.06, 58.0, 42.0, 11.0, 14.0, -0.40)}
OUTLIER_FRACTION = 0.02                   # glycine-like, anywhere
BIN_DEG, SMOOTH_DEG = 5.0, 7.5
ENCLOSED = (0.50, 0.90)


def wrap(angle):
    """Map any angle in degrees onto [−180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


weights = [spec[0] for spec in REGIONS.values()] + [OUTLIER_FRACTION]
component = rng.choice(len(weights), N_RES, p=weights)
phi_psi = rng.uniform(-180, 180, (N_RES, 2))            # outliers stay
for index, (_w, phi0, psi0, s_phi, s_psi, rho) in enumerate(
        REGIONS.values()):
    members = component == index
    cov = [[s_phi ** 2, rho * s_phi * s_psi],
           [rho * s_phi * s_psi, s_psi ** 2]]
    phi_psi[members] = wrap(rng.multivariate_normal(
        (phi0, psi0), cov, members.sum(), method="cholesky"))
is_outlier = component == len(REGIONS)
phi, psi = phi_psi.T

# --------------------------------------------------------- ESTIMATOR ---
N_BIN = int(360 / BIN_DEG)
edges = np.linspace(-180, 180, N_BIN + 1)
centres = (edges[:-1] + edges[1:]) / 2
counts = np.histogram2d(phi, psi, bins=[edges, edges])[0]


def smooth(hist):
    """Gaussian smoothing on the torus: the kernel wraps at ±180°."""
    return ndimage.gaussian_filter(hist, SMOOTH_DEG / BIN_DEG, mode="wrap")


density = smooth(counts) / (N_RES * BIN_DEG ** 2)       # per degree²


def density_at(phi_deg, psi_deg):
    """Periodic bilinear interpolation, as the contours are drawn."""
    index = [(np.asarray(a) - centres[0]) / BIN_DEG for a in (phi_deg,
                                                              psi_deg)]
    return ndimage.map_coordinates(density, index, order=1, mode="grid-wrap")


# levels: density quantiles over the residues, so the regions hold 50% and
# 90% of the points (mass under the smoothed density would hold more)
at_residue = density_at(phi, psi)
levels = [np.quantile(at_residue, 1 - f) for f in ENCLOSED]   # 50%, 90%

# the plotted grid adds the two edges, where the periodic estimate is the
# mean of the first and last bin centres
grid = np.concatenate([[-180.0], centres, [180.0]])
shown = density_at(*np.meshgrid(grid, grid, indexing="ij"))

# ------------------------------------------------------- SELF-CHECK ---
# (the enclosed fractions are checked on the drawn contours, below)
assert np.allclose(shown[0], shown[-1], rtol=0, atol=1e-15)      # φ edges
assert np.allclose(shown[:, 0], shown[:, -1], rtol=0, atol=1e-15)  # ψ edges
shift = (17, -9)                          # smoothing commutes with rotation
assert np.allclose(smooth(np.roll(counts, shift, (0, 1))),
                   np.roll(smooth(counts), shift, (0, 1)), atol=1e-12)
integral = density.sum() * BIN_DEG ** 2
assert abs(integral - 1) < 1e-12, integral

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 89)
ax = ms.axes(fig, 15, 11.5, 68, 68)
TINTS = ("#C6DBEF", "#6BAED6")            # Blues: 90% region, 50% region

bands = ax.contourf(grid, grid, shown.T, colors=TINTS, zorder=1,
                    levels=[levels[1], levels[0], shown.max()])
for zero in (ax.axhline, ax.axvline):
    zero(0, color=ms.GREY, lw=0.5, zorder=2)
ax.scatter(phi[~is_outlier], psi[~is_outlier], s=1.6, color=ms.INK,
           alpha=0.55, linewidths=0, rasterized=True, zorder=3)
ax.scatter(phi[is_outlier], psi[is_outlier], s=6, marker="^",
           facecolors="none", edgecolors=ms.VERMILLION, linewidths=0.5,
           rasterized=True, zorder=4)

# region labels in clear ground beside each cluster
for name, (dx, dy) in {"β": (85, 22), "α": (42, 0), "Lα": (37, 0)}.items():
    _w, phi0, psi0, *_ = REGIONS[name]
    ax.text(phi0 + dx, psi0 + dy, name, ha="center", va="center",
            fontsize=ms.FS_PANEL, fontweight="bold", color=ms.BLUE)
handles = [
    Line2D([], [], ls="none", marker="o", ms=1.6, color=ms.INK, mew=0,
           label=f"Residue (n = {N_RES - is_outlier.sum():,})"),
    Line2D([], [], ls="none", marker="^", ms=2.6, mfc="none",
           mec=ms.VERMILLION, mew=0.5,
           label=f"Glycine-like outlier (n = {is_outlier.sum()})"),
    Patch(facecolor=TINTS[1], lw=0, label="50% of residues enclosed"),
    Patch(facecolor=TINTS[0], lw=0, label="90% of residues enclosed")]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.01, 1.0),
          ncol=2, fontsize=ms.FS_SMALL, borderaxespad=0.4,
          handlelength=1.2, columnspacing=1.6)

ticks = np.arange(-180, 181, 60)
ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_aspect("equal")
for side in ("top", "right"):             # a torus map is read as a box
    ax.spines[side].set_visible(True)
ax.set_xlabel("φ (°)")
ax.set_ylabel("ψ (°)")

# SELF-CHECK, continued here because it reads the drawn contour paths:
# the fraction of residues inside the filled regions
outer, inner = (path.contains_points(phi_psi) for path in bands.get_paths())
enclosed = (inner.mean(), (inner | outer).mean())
for fraction, nominal in zip(enclosed, ENCLOSED):
    assert abs(fraction - nominal) < 0.02, (fraction, nominal)
print("fig175: self-check passed (drawn contours enclose "
      f"{100 * enclosed[0]:.1f}% and {100 * enclosed[1]:.1f}% of {N_RES} "
      f"residues; edge mismatch {np.abs(shown[0] - shown[-1]).max():.0e}; "
      f"density integral {integral:.12f})")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig175_ramachandran.{ext}")
print("fig175_ramachandran: saved png + pdf")
