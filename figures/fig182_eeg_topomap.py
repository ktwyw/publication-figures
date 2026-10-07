"""Fig. 182 - ERP butterfly plot and scalp topography (double column, 183 mm).

The two views an event-related-potential result needs: every electrode's
waveform overlaid in grey with the global field power (the spatial s.d.)
on top, and the scalp map at the latency of the field-power peak. The
map is a thin-plate spline through the electrode values on the sphere,
drawn on an azimuthal-equidistant projection with a head outline whose
nose and ears are geometry, not artwork. The self-check is that the
interpolant returns every electrode's value to 1e-6, that the data are
average-referenced (zero sum over electrodes at every sample, 1e-9) and
that the map's extreme lies within 15° of the simulated generator.

Data: 32 electrodes on rings 0°, 24°, 48° and 72° from the vertex (1, 6,
10, 15 sites); two radial dipoles at 0.45 head radii in a homogeneous
medium, an occipital N1 (−4 µV, 170 ms) and a parietal P3 (+8 µV,
380 ms), plus smooth noise (s.d. 0.25 µV); 500 Hz. All data are simulated.
"""

import numpy as np
from matplotlib.patches import Circle
from scipy import interpolate, ndimage

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(182)

# ------------------------------------------------------------- DATA ----
RINGS = [(0.0, 1), (24.0, 6), (48.0, 10), (72.0, 15)]   # polar angle, sites
DEPTH = 0.45                      # dipole eccentricity (head radius = 1)
# component: polar angle, azimuth (0° = right ear, 90° = nose), µV, ms, ms
SOURCES = {"N1": (62.0, -70.0, -4.0, 170.0, 22.0),
           "P3": (32.0, -100.0, 8.0, 380.0, 65.0)}
MAP_SOURCE = "P3"                 # the generator the map should recover
FS, T0, T1 = 500.0, -100.0, 600.0     # Hz, ms, ms
NOISE_SD, NOISE_SMOOTH = 0.25, 4.0    # µV, samples


def unit(polar_deg, azimuth_deg):
    """Unit vector(s); z through the vertex, y through the nose."""
    polar, azimuth = np.radians(polar_deg), np.radians(azimuth_deg)
    return np.stack([np.sin(polar) * np.cos(azimuth),
                     np.sin(polar) * np.sin(azimuth), np.cos(polar)], -1)


def project(xyz):
    """Azimuthal equidistant: radius = polar angle / 90° (equator = 1)."""
    polar = np.arccos(np.clip(xyz[..., 2], -1, 1)) / (np.pi / 2)
    azimuth = np.arctan2(xyz[..., 1], xyz[..., 0])
    return polar * np.cos(azimuth), polar * np.sin(azimuth)


def unproject(x, y):
    return unit(90.0 * np.hypot(x, y), np.degrees(np.arctan2(y, x)))


# rings start at the nose, so the montage is left-right symmetric
electrodes = np.concatenate([
    unit(np.full(n, polar), 90.0 + 360.0 * np.arange(n) / n)
    for polar, n in RINGS])
N_EL = electrodes.shape[0]
time = np.arange(T0, T1 + 1e-9, 1e3 / FS)

potential = np.zeros((N_EL, time.size))
for polar, azimuth, amplitude, latency, width in SOURCES.values():
    direction = unit(polar, azimuth)
    offset = electrodes - DEPTH * direction
    # radial current dipole in a homogeneous medium: V ∝ p·d / |d|³
    pattern = offset @ direction / np.linalg.norm(offset, axis=1) ** 3
    course = amplitude * np.exp(-0.5 * ((time - latency) / width) ** 2)
    potential += np.outer(pattern / np.abs(pattern).max(), course)
noise = ndimage.gaussian_filter1d(rng.normal(size=potential.shape),
                                  NOISE_SMOOTH, axis=1)
potential += NOISE_SD * noise / noise.std()
potential -= potential.mean(axis=0)            # average reference

# --------------------------------------------------------- ESTIMATOR ---
gfp = potential.std(axis=0)                    # global field power
i_map = int(np.argmax(gfp))
t_map, v_map = time[i_map], potential[:, i_map]

spline = interpolate.RBFInterpolator(electrodes, v_map,
                                     kernel="thin_plate_spline")
N_GRID = 301
gx, gy = np.meshgrid(np.linspace(-1, 1, N_GRID), np.linspace(-1, 1, N_GRID))
inside = np.hypot(gx, gy) <= 1.0
scalp = np.full(gx.shape, np.nan)
scalp[inside] = spline(unproject(gx[inside], gy[inside]))

i_peak = np.nanargmax(np.abs(scalp))
peak_xy = gx.flat[i_peak], gy.flat[i_peak]
source_dir = unit(*SOURCES[MAP_SOURCE][:2])
peak_error = np.degrees(np.arccos(unproject(*peak_xy) @ source_dir))

# ------------------------------------------------------- SELF-CHECK ---
refit = np.abs(spline(electrodes) - v_map).max()
assert refit < 1e-6, refit
reference_sum = np.abs(potential.sum(axis=0)).max()
assert reference_sum < 1e-9, reference_sum
assert peak_error < 15.0, peak_error
assert abs(t_map - SOURCES[MAP_SOURCE][3]) < 30.0, t_map
print(f"fig182: self-check passed (spline residual at electrodes "
      f"{refit:.1e} µV; max |sum over {N_EL} electrodes| "
      f"{reference_sum:.1e} µV; GFP peak {gfp.max():.2f} µV at "
      f"{t_map:.0f} ms; map extreme {scalp.flat[i_peak]:+.2f} µV, "
      f"{peak_error:.1f}° from the {MAP_SOURCE} generator)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 70)
ax_a = ms.axes(fig, 14, 11, 90, 51)
ax_b = ms.axes(fig, 113, 11, 51, 51)
ax_cbar = ms.axes(fig, 166, 18, 2.4, 37)
MAP_COLOUR = ms.VERMILLION

# a, butterfly plot: every electrode in grey, the field power on top
Y_LIM = (-6.0, 9.0)
ax_a.axhline(0, color=ms.GREY, lw=0.5, zorder=1)
ax_a.plot(time, potential.T, color=ms.GREY, lw=0.35, alpha=0.7, zorder=2)
ax_a.plot(time, gfp, color=ms.INK, lw=1.5, zorder=4)
ax_a.plot([0, 0], Y_LIM, color=ms.INK, lw=0.6, zorder=3)
ax_a.plot([t_map, t_map], Y_LIM, color=MAP_COLOUR, lw=0.8,
          ls=(0, (4, 2)), zorder=3)
for x, text, colour in ((0, "Stimulus", ms.INK),
                        (t_map, f"Map in b, {t_map:.0f} ms", MAP_COLOUR)):
    ax_a.annotate(text, xy=(x, Y_LIM[1]), xytext=(0, 2.5), ha="center",
                  textcoords="offset points", va="bottom", color=colour,
                  fontsize=ms.FS_TICK)
ax_a.text(T1, 8.3, "Global field power", fontweight="bold", ha="right",
          va="center", fontsize=ms.FS_TICK)
ax_a.text(T1, 7.3, f"{N_EL} electrodes,\naverage reference", ha="right",
          va="top", color=ms.GREY_DARK, fontsize=ms.FS_TICK)
for name, (_p, _a, amplitude, latency, _w) in SOURCES.items():
    at_peak = potential[:, np.argmin(np.abs(time - latency))]
    y = at_peak.min() - 0.3 if amplitude < 0 else at_peak.max() + 0.3
    ax_a.text(latency - 12, y, name, ha="right", fontsize=ms.FS_TICK,
              va="top" if amplitude < 0 else "bottom")
ax_a.set_xlim(T0, T1)
ax_a.set_ylim(*Y_LIM)
ax_a.set_xticks(np.arange(-100, 601, 100))
ax_a.set_yticks(np.arange(-6, 9.1, 3))
ax_a.set_xlabel("Time from stimulus onset (ms)")
ax_a.set_ylabel("Potential (µV), positive up")

# b, scalp map: rasterized spline surface clipped to the head circle
v_lim = np.ceil(np.nanmax(np.abs(scalp)))
image = ax_b.imshow(scalp, origin="lower", extent=[-1, 1, -1, 1],
                    cmap="RdBu_r", vmin=-v_lim, vmax=v_lim,
                    interpolation="bilinear", rasterized=True, zorder=1)
image.set_clip_path(Circle((0, 0), 1.0, transform=ax_b.transData))
levels = [v for v in np.arange(-v_lim, v_lim + 0.1, 2.0) if v != 0]
ax_b.contour(gx, gy, scalp, levels=levels, colors=ms.INK, linewidths=0.4,
             zorder=2)

# head outline from geometry: unit circle, nose wedge, half-ellipse ears
angle = np.linspace(-np.pi, np.pi, 361)       # 0 points away from the head
outline = dict(color=ms.INK, lw=0.9, solid_joinstyle="round", zorder=4)
ax_b.plot(np.cos(angle), np.sin(angle), **outline)
nose = np.radians([90 - 9, 90, 90 + 9])
ax_b.plot(np.cos(nose) * [1, 1.13, 1], np.sin(nose) * [1, 1.13, 1],
          **outline)
for side in (-1, 1):
    ear_x, ear_y = side * (0.985 + 0.09 * np.cos(angle)), 0.23 * np.sin(angle)
    outside = np.hypot(ear_x, ear_y) >= 1.0
    ax_b.plot(ear_x[outside], ear_y[outside], **outline)
    ax_b.text(side * 1.16, 0, "R" if side > 0 else "L", ha="center",
              va="center", fontsize=ms.FS_TICK)
ax_b.plot(*project(electrodes), "o", ms=2.0, mfc=ms.INK, mec="white",
          mew=0.3, zorder=5)
ax_b.plot(*peak_xy, "+", ms=5.5, mew=0.9, color="black", zorder=6)
ax_b.text(-1.26, 1.26, f"{t_map:.0f} ms", color=MAP_COLOUR, va="top",
          fontsize=ms.FS_BODY)
ax_b.text(1.26, -1.26, "+ map extreme", ha="right",
          va="bottom", fontsize=ms.FS_TICK)
ax_b.set_xlim(-1.28, 1.28)
ax_b.set_ylim(-1.28, 1.28)
ax_b.set_aspect("equal")
ax_b.axis("off")

cbar = fig.colorbar(image, cax=ax_cbar, ticks=np.arange(-v_lim, v_lim + 1, 4))
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(length=2, width=0.5)
cbar.set_label("Potential (µV)")

ms.panel_label(ax_a, "a", dx_pt=-28, dy_pt=8)
ms.panel_label(ax_b, "b", dx_pt=-6, dy_pt=8)
ms.assert_aligned([ax_a, ax_b])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig182_eeg_topomap.{ext}")
print("fig182_eeg_topomap: saved png + pdf")
