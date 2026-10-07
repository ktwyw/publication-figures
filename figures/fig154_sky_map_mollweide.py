"""Fig. 154 - All-sky source-density map, Mollweide (1.5 column, 120 mm).

How to show a catalogue on the whole sky without distorting density: the
sources are counted on an equal-area grid (uniform in right ascension
and in the sine of declination), so every cell subtends the same solid
angle, and drawn in the equal-area Mollweide projection with east to
the left. A log colour scale carries both the isotropic background and
the plane-like band; the three richest compact overdensities are
circled. The self-check is that every cell subtends the same solid
angle and the cells sum to 4π sr, that the cell counts sum to the
catalogue size, that counts away from the planted structures are
Poisson (variance/mean near 1), and that the excess counted around
each planted overdensity returns the number of sources put there.

Data: n = 50,000 sources: 36,000 isotropic; 11,200 in a band (Gaussian,
s.d. 5°) along a great circle tilted 60° to the equator; 2,800 in four
Gaussian clumps (s.d. 1.5-2.5°). Grid: 72 × 36 cells of 15.9 deg².
Names are placeholders. All data are simulated.
"""

import numpy as np
from matplotlib.colors import LogNorm

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(154)

# ------------------------------------------------------------- DATA ----
N_ISO, N_BAND = 36000, 11200
TILT, NODE = np.radians(60.0), np.radians(90.0)   # band plane: i, RA of node
BAND_SD = np.radians(5.0)
CLUMPS = [      # name, RA (deg), Dec (deg), s.d. (deg), sources
    ("Field A", 200.0, -20.0, 2.0, 1200),
    ("Field B", 150.0, -50.0, 2.5, 800),
    ("Field C", 150.0, 12.0, 1.5, 500),
    ("Field D", 335.0, 38.0, 2.0, 300),
]
N_RA, N_SIN = 72, 36                              # equal-area grid


def unit(ra, dec):
    """Unit vectors (..., 3) from right ascension and declination (rad)."""
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra),
                     np.sin(dec)], axis=-1)


def angles(vec):
    """Right ascension in [0, 2π) and declination from unit vectors."""
    return (np.arctan2(vec[..., 1], vec[..., 0]) % (2 * np.pi),
            np.arcsin(np.clip(vec[..., 2], -1, 1)))


# band frame to sky: tilt about the x axis, then turn the node to NODE
c, s = np.cos(TILT), np.sin(TILT)
ROT_X = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
c, s = np.cos(NODE), np.sin(NODE)
ROT_Z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
BAND_TO_SKY = ROT_Z @ ROT_X
POLE = BAND_TO_SKY @ np.array([0.0, 0.0, 1.0])     # normal of the band plane

iso = unit(rng.uniform(0, 2 * np.pi, N_ISO),
           np.arcsin(rng.uniform(-1, 1, N_ISO)))
band = unit(rng.uniform(0, 2 * np.pi, N_BAND),
            rng.normal(0, BAND_SD, N_BAND)) @ BAND_TO_SKY.T
clumps = []
for _name, ra0, dec0, sd, count in CLUMPS:
    centre = unit(np.radians(ra0), np.radians(dec0))
    kick = rng.normal(0, np.radians(sd), (count, 3))   # isotropic, small
    kick -= np.outer(kick @ centre, centre)            # tangent-plane part
    vec = centre + kick
    clumps.append(vec / np.linalg.norm(vec, axis=1, keepdims=True))
catalogue = np.concatenate([iso, band] + clumps)
ra, dec = angles(catalogue)

# --------------------------------------------------------- ESTIMATOR ---
ra_edges = np.linspace(0, 2 * np.pi, N_RA + 1)
sin_edges = np.linspace(-1, 1, N_SIN + 1)
counts, _, _ = np.histogram2d(ra, np.sin(dec), bins=[ra_edges, sin_edges])
solid_angle = np.outer(np.diff(ra_edges), np.diff(sin_edges))     # sr
cell_deg2 = np.degrees(np.sqrt(solid_angle.mean())) ** 2

# cell centres, and the part of the sky away from every planted structure
centres = unit(*np.meshgrid(0.5 * (ra_edges[:-1] + ra_edges[1:]),
                            np.arcsin(0.5 * (sin_edges[:-1] + sin_edges[1:])),
                            indexing="ij"))
off_band = np.abs(np.arcsin(centres @ POLE)) > 5 * BAND_SD
apertures = [np.arccos(centres @ unit(np.radians(ra0), np.radians(dec0)))
             < np.radians(5 * sd + 5) for _name, ra0, dec0, sd, _n in CLUMPS]
quiet = off_band & ~np.any(apertures, axis=0)

# ------------------------------------------------------- SELF-CHECK ---
assert np.allclose(solid_angle, solid_angle.mean(), rtol=1e-12)
assert abs(solid_angle.sum() - 4 * np.pi) < 1e-10
assert counts.sum() == catalogue.shape[0] == 50000
fano = counts[quiet].var(ddof=1) / counts[quiet].mean()
assert 0.8 < fano < 1.25, fano
expected = N_ISO / (N_RA * N_SIN)
assert abs(counts[quiet].mean() - expected) < 0.05 * expected
excess = []                         # aperture counts minus the background
for aperture, (_name, _ra, _dec, _sd, planted) in zip(apertures, CLUMPS):
    inside = counts[aperture].sum()
    excess.append(inside - aperture.sum() * counts[quiet].mean())
    assert abs(excess[-1] - planted) < 5 * np.sqrt(inside), (excess, planted)
print(f"fig154: self-check passed ({N_RA * N_SIN} cells of "
      f"{cell_deg2:.2f} deg² summing to 4π sr; counts sum to "
      f"{int(counts.sum())}; variance/mean {fano:.3f} over {quiet.sum()} "
      f"quiet cells, mean {counts[quiet].mean():.2f} vs {expected:.2f} "
      "expected; clump excess "
      + ", ".join(f"{e:.0f}" for e in excess) + " vs planted "
      + ", ".join(str(c[4]) for c in CLUMPS) + ")")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 72)
ax = ms.axes(fig, 10, 16, 100, 50, projection="mollweide")
ax_cbar = ms.axes(fig, 35, 9.5, 50, 2.4)


def to_map(ra_rad):
    """Map longitude: RA 180° at the centre, increasing to the left."""
    return np.pi - np.asarray(ra_rad)


# each cell is drawn as SUB × SUB facets of its own count, so that cell
# edges follow the curved projection instead of cutting across the limb
SUB = 4
dec_edges = np.arcsin(sin_edges)
fine_ra = np.linspace(0, 2 * np.pi, N_RA * SUB + 1)
fine_dec = np.concatenate([np.linspace(lo, hi, SUB, endpoint=False)
                           for lo, hi in zip(dec_edges[:-1], dec_edges[1:])]
                          + [dec_edges[-1:]])
mesh = ax.pcolormesh(to_map(fine_ra), fine_dec,
                     np.repeat(np.repeat(counts.T, SUB, axis=0), SUB, axis=1),
                     cmap="magma", norm=LogNorm(4, counts.max()),
                     shading="flat", rasterized=True)

# graticule drawn by hand: every 60° (4 h) in RA and 30° in declination
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
GRID = dict(color="white", lw=0.3, alpha=0.45)
lat_line = np.radians(np.linspace(-89.5, 89.5, 181))
for hour in range(4, 24, 4):
    ax.plot(np.full_like(lat_line, to_map(np.radians(15.0 * hour))),
            lat_line, **GRID)
    # hour labels sit beside their meridian, above the equator
    ax.text(to_map(np.radians(15.0 * hour - 2.5)), np.radians(7.0),
            f"{hour} h", color="white", fontsize=ms.FS_SMALL, ha="left",
            va="center")
for dec_deg in (-60, -30, 0, 30, 60):
    ax.plot(np.linspace(-np.pi, np.pi, 181),
            np.full(181, np.radians(dec_deg)), **GRID)
    ax.annotate(f"{dec_deg:+d}°".replace("-", "−").replace("+0", "0"),
                xy=(-np.pi, np.radians(dec_deg)),
                xytext=(-3 - 0.045 * abs(dec_deg), 0.06 * dec_deg),
                textcoords="offset points", ha="right", va="center",
                fontsize=ms.FS_TICK)

# the three richest planted clumps: 7° circles, names clear of the ring
RING = np.radians(7.0)
turn = np.linspace(0, 2 * np.pi, 91)
for name, ra0, dec0, _sd, _count in sorted(CLUMPS, key=lambda c: -c[4])[:3]:
    centre = unit(np.radians(ra0), np.radians(dec0))
    east = np.cross([0.0, 0.0, 1.0], centre)
    east /= np.linalg.norm(east)
    north = np.cross(centre, east)
    ring = (np.cos(RING) * centre + np.sin(RING)
            * (np.outer(np.cos(turn), east) + np.outer(np.sin(turn), north)))
    ring_ra, ring_dec = angles(ring)
    ax.plot(to_map(ring_ra), ring_dec, color="white", lw=0.7)
    ax.text(to_map(np.radians(ra0)), np.radians(dec0) + RING + 0.035, name,
            color="white", fontsize=ms.FS_TICK, fontweight="bold",
            ha="center", va="bottom")

fig.text(0.5, 69.2 / 72, "Equatorial coordinates; right ascension "
         "increases to the left", ha="center", va="center",
         fontsize=ms.FS_TICK, color=ms.GREY_DARK)
cbar = fig.colorbar(mesh, cax=ax_cbar, orientation="horizontal")
cbar.set_ticks([5, 10, 20, 50, 100, 200])
cbar.set_ticklabels(["5", "10", "20", "50", "100", "200"])
cbar.ax.minorticks_off()
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(length=2, width=0.5)
cbar.set_label(f"Sources per cell ({cell_deg2:.1f} deg² each; "
               f"n = {catalogue.shape[0]:,})")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig154_sky_map_mollweide.{ext}")
print("fig154_sky_map_mollweide: saved png + pdf")
