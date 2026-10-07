"""Fig. 149 - Image, zoom and fitted line profile (double column, 183 mm).

The three steps by which an image becomes a number: the full field
with a calibrated scale bar and a marked region, that region magnified
pixel for pixel with its own rescaled bar and the profile path, and the
intensity along the path with a Gaussian fit whose full width at half
maximum is the measurement. The field is simulated: sub-resolution
beads and filaments blurred by a Gaussian point-spread function, with
shot noise and a camera offset. The self-check is that the fitted FWHM
equals 2√(2 ln 2) σ of the simulated PSF within 5%, that each scale bar
is exactly its labelled length divided by the pixel size, and that the
zoom shows exactly the pixels inside the marked rectangle.

Image integrity: one linear grey scale (100 to 1,300 counts) for both
images, nearest-neighbour magnification, no gamma or local adjustment.

Model: 50 nm pixels; PSF σ = 100 nm (FWHM 235 nm); 46 beads and 4
filaments; Poisson noise on photons plus a 100-count offset with 2
counts r.m.s. read noise; profile = one pixel row, n = 25 pixels;
least-squares Gaussian plus offset. All data are simulated.
"""

import numpy as np
from matplotlib.patches import Rectangle
from scipy import optimize
from scipy.ndimage import gaussian_filter

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(149)

# ------------------------------------------------------------- DATA ----
SIZE, PIXEL_NM = 320, 50.0                  # px, nm per px (16 µm field)
PSF_SIGMA_NM = 100.0                        # Gaussian PSF, s.d.
OFFSET, READ_NOISE = 100.0, 2.0             # camera counts
LIMITS = (100.0, 1300.0)                    # display range, both images
PROBE = (214, 96)                           # (row, column) of the fitted bead
PROBE_PHOTONS = 1100.0                      # its peak signal
ROI_ORIGIN, ROI_SIZE = (184, 66), 60        # top-left (row, column), px
HALF_PROFILE = 12                           # px each side of the bead
BAR_UM = {"field": 5.0, "zoom": 1.0}        # labelled scale bars

sigma_px = PSF_SIGMA_NM / PIXEL_NM
scene = np.zeros((SIZE, SIZE))              # emitters, in peak photons


def placeable(rows, cols):
    """Keep emitters away from the fitted bead (so it stays isolated) and
    out of the lower right corner, which the field leaves for the bar."""
    isolated = np.hypot(rows - PROBE[0], cols - PROBE[1]) > 16
    in_corner = (rows > 0.8 * SIZE) & (cols > 0.55 * SIZE)
    return np.all(isolated & ~in_corner)


n_beads = 0
while n_beads < 45:                         # beads sit on pixel centres
    row, col = rng.integers(8, SIZE - 8, 2)
    if placeable(row, col):
        scene[row, col] += rng.uniform(350, 1000)
        n_beads += 1
n_filaments = 0
while n_filaments < 4:                      # smooth random-walk curves
    heading = rng.uniform(0, 2 * np.pi) + np.cumsum(rng.normal(0, 0.012, 900))
    rows = rng.uniform(40, SIZE - 40) + np.cumsum(0.3 * np.sin(heading))
    cols = rng.uniform(40, SIZE - 40) + np.cumsum(0.3 * np.cos(heading))
    inside = (rows > 0) & (rows < SIZE - 1) & (cols > 0) & (cols < SIZE - 1)
    if placeable(rows[inside], cols[inside]):
        np.add.at(scene, (np.rint(rows[inside]).astype(int),
                          np.rint(cols[inside]).astype(int)), 22.0)
        n_filaments += 1
scene[PROBE] = PROBE_PHOTONS
# unit-peak PSF: an emitter of a photons on a pixel centre peaks at a
photons = gaussian_filter(scene, sigma_px) * 2 * np.pi * sigma_px ** 2
image = (rng.poisson(photons + 8.0) + OFFSET
         + rng.normal(0, READ_NOISE, scene.shape))

# ------------------------------------------------------- MEASUREMENT ---
r0, c0 = ROI_ORIGIN
roi = image[r0:r0 + ROI_SIZE, c0:c0 + ROI_SIZE]
roi_box = (c0 - 0.5, r0 - 0.5, ROI_SIZE, ROI_SIZE)   # x, y, w, h (px edges)
profile_cols = np.arange(PROBE[1] - HALF_PROFILE, PROBE[1] + HALF_PROFILE + 1)
profile = image[PROBE[0], profile_cols]
distance_nm = (profile_cols - PROBE[1]) * PIXEL_NM


def gaussian(x, amplitude, centre, sigma, offset):
    return amplitude * np.exp(-(x - centre) ** 2 / (2 * sigma ** 2)) + offset


fit, _cov = optimize.curve_fit(gaussian, distance_nm, profile,
                               p0=[profile.max() - profile.min(), 0.0, 150.0,
                                   profile.min()])
TO_FWHM = 2 * np.sqrt(2 * np.log(2))
fwhm_nm, fwhm_true = TO_FWHM * abs(fit[2]), TO_FWHM * PSF_SIGMA_NM
bar_px = {name: 1000 * um / PIXEL_NM for name, um in BAR_UM.items()}

# ------------------------------------------------------- SELF-CHECK ---
assert abs(fwhm_nm - fwhm_true) < 0.05 * fwhm_true, (fwhm_nm, fwhm_true)
for name, um in BAR_UM.items():
    assert bar_px[name] * PIXEL_NM == 1000 * um, name
assert roi.shape == (ROI_SIZE, ROI_SIZE) and np.shares_memory(roi, image)
assert (roi_box[0] + 0.5, roi_box[1] + 0.5) == (c0, r0)
assert r0 <= PROBE[0] < r0 + ROI_SIZE and c0 <= profile_cols[0]
assert profile_cols[-1] < c0 + ROI_SIZE
print(f"fig149: self-check passed (FWHM {fwhm_nm:.0f} nm fitted, "
      f"{fwhm_true:.0f} nm from the PSF; scale bars {bar_px['field']:.0f} px "
      f"= {BAR_UM['field']:g} µm and {bar_px['zoom']:.0f} px = "
      f"{BAR_UM['zoom']:g} µm at {PIXEL_NM:g} nm per px; zoom = "
      f"{ROI_SIZE} x {ROI_SIZE} px ROI)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 65)
EDGE = 55.0                                              # image side, mm
ax_a = ms.axes(fig, 7, 4, EDGE, EDGE)
ax_b = ms.axes(fig, 70, 4, EDGE, EDGE)
ax_c = ms.axes(fig, 142, 13, 36, 46)
show = dict(cmap="gray", vmin=LIMITS[0], vmax=LIMITS[1],
            interpolation="nearest", aspect="auto", rasterized=True)


def scale_bar(ax, extent, length_px, label):
    """White bar in the lower right corner, sized in image pixels."""
    x_left, x_right, y_bottom, y_top = extent
    span = x_right - x_left
    x1 = x_right - 0.06 * span
    y = y_bottom - 0.07 * (y_bottom - y_top)
    ax.add_patch(Rectangle((x1 - length_px, y), length_px, 0.016 * span,
                           color="white", lw=0))
    ax.text(x1 - length_px / 2, y - 0.012 * span, label, color="white",
            ha="center", va="bottom", fontsize=ms.FS_TICK)


# a, the full field with the region of interest outlined
ax_a.set_facecolor("black")
ax_a.imshow(image, **show)
ax_a.add_patch(Rectangle(roi_box[:2], *roi_box[2:], fill=False,
                         edgecolor="white", lw=0.6))
scale_bar(ax_a, (-0.5, SIZE - 0.5, SIZE - 0.5, -0.5), bar_px["field"],
          f"{BAR_UM['field']:g} µm")

# b, exactly those pixels, magnified; the extent keeps full-field pixel
# coordinates so that the profile path is drawn where it was measured
extent_b = (roi_box[0], roi_box[0] + ROI_SIZE, roi_box[1] + ROI_SIZE,
            roi_box[1])
ax_b.set_facecolor("black")
shown_b = ax_b.imshow(roi, extent=extent_b, **show)
assert shown_b.get_array().shape == roi.shape            # nothing resampled
ax_b.plot(profile_cols[[0, -1]], [PROBE[0], PROBE[0]], color=ms.ORANGE,
          lw=0.6)
ax_b.text(profile_cols[-1] + 1.5, PROBE[0], "Profile", color=ms.ORANGE,
          va="center", fontsize=ms.FS_TICK)
scale_bar(ax_b, extent_b, bar_px["zoom"], f"{BAR_UM['zoom']:g} µm")
for ax in (ax_a, ax_b):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# c, the profile, its fit and the width that is reported
x_fine = np.linspace(distance_nm[0], distance_nm[-1], 300)
ax_c.plot(x_fine, gaussian(x_fine, *fit), color=ms.INK, lw=1.0)
ax_c.plot(distance_nm, profile, "o", ms=3, mfc=ms.ORANGE, mec="white",
          mew=0.4, zorder=3)
half_max = fit[3] + fit[0] / 2
ax_c.annotate("", xy=(fit[1] - fwhm_nm / 2, half_max),
              xytext=(fit[1] + fwhm_nm / 2, half_max),
              arrowprops=dict(arrowstyle="<->", color=ms.INK, lw=0.6,
                              shrinkA=0, shrinkB=0, mutation_scale=5))
ax_c.text(fit[1] + fwhm_nm / 2 + 50, half_max, f"FWHM\n{fwhm_nm:.0f} nm",
          va="center", fontsize=ms.FS_TICK)
ax_c.text(0.07, 0.985, "Gaussian + offset fit\nPSF: 2√(2 ln 2) σ = "
          f"{fwhm_true:.0f} nm", transform=ax_c.transAxes, va="top",
          fontsize=ms.FS_SMALL, color=ms.GREY_DARK)
ax_c.set_xlim(-650, 650)
ax_c.set_ylim(0, 1650)
ax_c.set_xticks([-500, 0, 500])
ax_c.set_yticks(np.arange(0, 1501, 500))
ax_c.spines["left"].set_bounds(0, 1500)
ax_c.set_xlabel("Distance along profile (nm)")
ax_c.set_ylabel("Intensity (counts)")

ms.panel_label(ax_a, "a", dx_pt=-12, dy_pt=3)
ms.panel_label(ax_b, "b", dx_pt=-12, dy_pt=3)
ms.panel_label(ax_c, "c", dx_pt=-30, dy_pt=3)
ms.assert_aligned([ax_a, ax_b])
ms.assert_aligned([ax_a, ax_c], edges=("top",))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig149_image_zoom_profile.{ext}")
print("fig149_image_zoom_profile: saved png + pdf")
