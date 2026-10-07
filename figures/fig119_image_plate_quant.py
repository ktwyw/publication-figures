"""Fig. 119 - Two-channel image plate with field quantification (183 mm).

Extends fig011 (one image, one scale bar) to the plate a cell-biology
figure needs: two conditions x three views (nuclei, marker, merge) on a
dark millimetre grid with equal gutters, colour-blind-safe cyan/magenta
pseudo-colour, and a quantification panel whose values are measured from
the same simulated images that are shown (field 1 of each condition is
displayed). The self-check is that the scale bar's pixel length times
the pixel size is the labelled 20 µm, that the value plotted for field 1
is the mean of the marker image on display, that a hand-coded Welch test
reproduces scipy, and that the six image panels share edges and gutters.

Image integrity: identical linear display limits for every image of a
channel; no gamma or local adjustment; scale bar calibrated from the
simulated pixel size (0.2 µm per pixel).

Statistics: n = 8 fields of view per condition from one simulated
experiment; dots, fields; cross, mean ± s.d.; two-sided Welch's t-test.
The images are computer-generated in this script, not microscopy; all
data are simulated.
"""

import numpy as np
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.ndimage import gaussian_filter

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(1919)
SIZE, PIXEL_UM, N_FIELDS = 256, 0.2, 8          # px, µm per px, fields
BAR_UM = 20                                     # labelled scale bar
LIMITS = (0.03, 1.0)                            # display range, all images
PHOTONS = 220.0
CONDITIONS = {"Control": 5, "Treated": 17}      # mean puncta per cell
CYAN, MAGENTA = "#0E8A96", "#C2189E"            # channel labels on white


# ------------------------------------------------------------- DATA ----
def simulate_field(puncta_per_cell):
    """One field: nine nuclei and perinuclear puncta, with shot noise."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    nuclei = np.zeros((SIZE, SIZE))
    marker = np.zeros((SIZE, SIZE))
    centres = []
    while len(centres) < 9:
        candidate = rng.uniform(22, SIZE - 22, 2)
        if all(np.hypot(*(candidate - c)) > 52 for c in centres):
            centres.append(candidate)
    for cx, cy in centres:
        a, b = rng.uniform(14, 19), rng.uniform(10, 14)
        angle = rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
        v = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
        edge = np.minimum(((u / a) ** 2 + (v / b) ** 2 - 1) * 4, 50)
        nuclei += rng.uniform(0.7, 1.0) / (1 + np.exp(edge))
        for _ in range(rng.poisson(puncta_per_cell)):
            radius, theta = rng.uniform(20, 34), rng.uniform(0, 2 * np.pi)
            px, py = cx + radius * np.cos(theta), cy + radius * np.sin(theta)
            marker += rng.uniform(0.6, 1.0) * np.exp(
                -((xx - px) ** 2 + (yy - py) ** 2) / (2 * 1.7 ** 2))
    texture = gaussian_filter(rng.normal(size=nuclei.shape), 2)
    nuclei = gaussian_filter(nuclei, 1.2) * (1 + 0.12 * texture)
    marker = marker + 0.05 * gaussian_filter(nuclei, 9)
    nuclei = rng.poisson(np.clip(nuclei, 0, None) * PHOTONS + 6) / PHOTONS
    marker = rng.poisson(np.clip(marker, 0, None) * PHOTONS + 6) / PHOTONS
    return nuclei, marker


fields = {name: [simulate_field(mean_puncta) for _ in range(N_FIELDS)]
          for name, mean_puncta in CONDITIONS.items()}
shown = {name: images[0] for name, images in fields.items()}   # field 1

# --------------------------------------------------- QUANTIFICATION ----
raw = {name: np.array([marker.mean() for _nuclei, marker in images])
       for name, images in fields.items()}
control_mean = raw["Control"].mean()
relative = {name: values / control_mean for name, values in raw.items()}
test = stats.ttest_ind(relative["Control"], relative["Treated"],
                       equal_var=False)
bar_px = BAR_UM / PIXEL_UM


def welch(x, y):
    """Welch's t, Satterthwaite degrees of freedom, two-sided P."""
    vx, vy = x.var(ddof=1) / x.size, y.var(ddof=1) / y.size
    t = (x.mean() - y.mean()) / np.sqrt(vx + vy)
    dof = (vx + vy) ** 2 / (vx ** 2 / (x.size - 1) + vy ** 2 / (y.size - 1))
    return t, dof, 2 * stats.t.sf(abs(t), dof)


# ------------------------------------------------------- SELF-CHECK ---
assert bar_px == round(bar_px) and bar_px * PIXEL_UM == BAR_UM, bar_px
for name, (_nuclei, marker) in shown.items():
    assert np.isclose(relative[name][0] * control_mean, marker.mean(),
                      rtol=1e-12), name
t_stat, dof, p_hand = welch(relative["Control"], relative["Treated"])
assert np.isclose(t_stat, test.statistic, rtol=1e-10)
assert np.isclose(p_hand, test.pvalue, rtol=1e-8)
print(f"fig119: self-check passed (scale bar {bar_px:.0f} px x {PIXEL_UM} "
      f"µm = {BAR_UM} µm; field-1 values match the displayed images; "
      f"Welch t = {t_stat:.1f}, d.f. = {dof:.1f} matches scipy)")


# ------------------------------------------------------------ FIGURE --
def to_rgb(nuclei=None, marker=None):
    """Linear, identical scaling; cyan = (0, n, n), magenta = (m, 0, m)."""
    rgb = np.zeros((SIZE, SIZE, 3))
    for image, channels in ((nuclei, (1, 2)), (marker, (0, 2))):
        if image is not None:
            scaled = np.clip((image - LIMITS[0]) / (LIMITS[1] - LIMITS[0]),
                             0, 1)
            for channel in channels:
                rgb[..., channel] += scaled
    return np.clip(rgb, 0, 1)


fig = ms.figure(183, 76)
CELL, GAP, X0, Y0 = 31.0, 1.5, 11.0, 5.0        # mm
plate = np.empty((2, 3), dtype=object)
for r, condition in enumerate(CONDITIONS):
    nuclei, marker = shown[condition]
    views = (to_rgb(nuclei=nuclei), to_rgb(marker=marker),
             to_rgb(nuclei, marker))
    for c, view in enumerate(views):
        ax = ms.axes(fig, X0 + c * (CELL + GAP),
                     Y0 + (1 - r) * (CELL + GAP), CELL, CELL)
        ax.imshow(view, interpolation="nearest", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        plate[r, c] = ax
    plate[r, 0].text(-0.035, 0.5, condition, transform=plate[r, 0].transAxes,
                     rotation=90, rotation_mode="anchor", ha="center",
                     va="bottom")
for ax, title, colour in zip(plate[0], ("Nuclei", "Marker", "Merge"),
                             (CYAN, MAGENTA, ms.INK)):
    ax.set_title(title, color=colour, pad=3)

# one calibrated scale bar per merged image, labelled once
for ax in plate[:, 2]:
    ax.add_patch(Rectangle((SIZE - 14 - bar_px, SIZE - 20), bar_px, 5,
                           color="white", lw=0))
plate[0, 2].text(SIZE - 14 - bar_px / 2, SIZE - 27, f"{BAR_UM} µm",
                 color="white", ha="center", va="bottom",
                 fontsize=ms.FS_TICK)

# b, field-level quantification of the marker channel
ax_b = ms.axes(fig, 128, 14, 36, 54.5)
jitter_rng = np.random.default_rng(19)     # horizontal jitter, display only
colours = {"Control": ms.GREY_DARK, "Treated": MAGENTA}
for position, (condition, values) in enumerate(relative.items()):
    ax_b.scatter(position + jitter_rng.uniform(-0.16, 0.16, values.size),
                 values, s=11, color=colours[condition], alpha=0.7,
                 linewidths=0, zorder=2)
    ax_b.errorbar(position + 0.34, values.mean(), yerr=values.std(ddof=1),
                  fmt="_", color=ms.INK, markersize=7, markeredgewidth=1.0,
                  elinewidth=0.8, capsize=0, zorder=3)
top = max(float(values.max()) for values in relative.values())
ms.bracket(ax_b, 0, 1, top * 1.07, ms.format_p(test.pvalue),
           tick=top * 0.025, text_pad=top * 0.012)
ax_b.set_xlim(-0.6, 1.75)
ax_b.set_ylim(0, top * 1.2)
ax_b.set_xticks([0.1, 1.1], list(CONDITIONS))
ax_b.tick_params(axis="x", length=0, pad=3)
ax_b.set_ylabel("Marker intensity per field\n(relative to control mean)")
ax_b.text(1.03, 0.02, "n = 8 fields\nper condition\nMean ± s.d.",
          transform=ax_b.transAxes, fontsize=ms.FS_SMALL,
          color=ms.GREY_DARK, va="bottom")
ms.panel_label(plate[0, 0], "a", dx_pt=-22, dy_pt=4)
ms.panel_label(ax_b, "b", dx_pt=-34, dy_pt=4)

# plate geometry: shared edges per row and column, and equal gutters
for row in plate:
    ms.assert_aligned(list(row))
for column in plate.T:
    ms.assert_aligned(list(column), edges=("left", "right"))
boxes = [[ms.plot_area_pt(ax) for ax in row] for row in plate]
gutters = [boxes[0][1][0] - boxes[0][0][2], boxes[0][2][0] - boxes[0][1][2],
           boxes[0][0][1] - boxes[1][0][3]]
assert max(gutters) - min(gutters) < 0.1, gutters
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig119_image_plate_quant.{ext}")
print("fig119_image_plate_quant: saved png + pdf")
