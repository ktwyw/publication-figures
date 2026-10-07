"""Fig. 141 - Area-proportional Euler diagrams, one scale (1.5 column, 120 mm).

Hand-drawn Venn circles say only that two sets overlap; here every area
is a count. Each circle has area equal to its set size (radius =
sqrt(size / pi)), and the distance between centres is solved with
scipy.optimize.brentq so that the lens has exactly the area of the
intersection. All three panels share one millimetres-per-unit scale, so
circles can be compared across datasets as well as within them, and a
reference disc shows what 100 hits look like. Counts are printed at the
mid-point of each region's chord on the line of centres. The self-check
is that the lens area recomputed from the solved distance reproduces the
intersection to 1e-9 (and agrees with an independent numerical integral),
that every drawn disc has the same area per item on the page, and that
every count fits inside its region.

Data: hits called by two methods among 4,000 candidates in each of three
datasets; counts only, no hypothesis test. All data are simulated.
"""

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle
from scipy import integrate, optimize

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(141)

# ------------------------------------------------------------- DATA ----
# Per dataset: P(hit by A), P(hit by B | hit by A), P(hit by B | not A)
SETS = {"Method A": ms.BLUE, "Method B": ms.VERMILLION}
DATASETS = {"Dataset 1": (0.105, 0.55, 0.040),
            "Dataset 2": (0.130, 0.13, 0.026),
            "Dataset 3": (0.075, 0.16, 0.075)}
N_CANDIDATES = 4000
counts = {}                                   # name -> (|A|, |B|, |A and B|)
for name, (p_a, p_b_in, p_b_out) in DATASETS.items():
    hit_a = rng.random(N_CANDIDATES) < p_a
    hit_b = rng.random(N_CANDIDATES) < np.where(hit_a, p_b_in, p_b_out)
    counts[name] = (int(hit_a.sum()), int(hit_b.sum()),
                    int((hit_a & hit_b).sum()))


# --------------------------------------------------------- GEOMETRY ----
def lens_area(d, r1, r2):
    """Area common to two circles of radii r1, r2 with centres d apart."""
    a1 = np.arccos(np.clip((d * d + r1 * r1 - r2 * r2) / (2 * d * r1), -1, 1))
    a2 = np.arccos(np.clip((d * d + r2 * r2 - r1 * r1) / (2 * d * r2), -1, 1))
    kite = np.sqrt(max((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2)
                       * (d + r1 + r2), 0.0))
    return r1 * r1 * a1 + r2 * r2 * a2 - kite / 2


def lens_area_numeric(d, r1, r2):
    """The same area as the integral of the overlap height across x."""
    def height(x):
        return 2 * min(np.sqrt(max(r1 * r1 - x * x, 0.0)),
                       np.sqrt(max(r2 * r2 - (x - d) ** 2, 0.0)))
    crossing = (d * d + r1 * r1 - r2 * r2) / (2 * d)   # x of the two vertices
    return sum(integrate.quad(height, lo, hi, epsabs=1e-12, epsrel=1e-12)[0]
               for lo, hi in ((d - r2, crossing), (crossing, r1)))


layout = {}
for name, (size_a, size_b, both) in counts.items():
    r_a, r_b = np.sqrt(size_a / np.pi), np.sqrt(size_b / np.pi)  # area = n
    # lens area falls monotonically from the smaller disc (nested) to 0
    # (tangent), so the root is bracketed by those two distances
    d = optimize.brentq(lambda x: lens_area(x, r_a, r_b) - both,
                        abs(r_a - r_b), r_a + r_b, xtol=1e-14, rtol=1e-15)
    shift = (d + r_b - r_a) / 2            # centre the pair on x = 0
    # chord of each region on the line of centres: A only, both, B only
    cuts = np.array([-r_a, d - r_b, r_a, d + r_b]) - shift
    layout[name] = dict(r=(r_a, r_b), d=d, x=(-shift, d - shift), cuts=cuts,
                        values=(size_a - both, both, size_b - both))

# ------------------------------------------------------- SELF-CHECK ---
worst = 0.0
for name, (size_a, size_b, both) in counts.items():
    r_a, r_b = layout[name]["r"]
    d = layout[name]["d"]
    worst = max(worst, abs(lens_area(d, r_a, r_b) - both) / both)
    assert abs(lens_area_numeric(d, r_a, r_b) - both) < 1e-6 * both
    assert abs(r_a - r_b) < d < r_a + r_b            # a proper lens
    assert np.all(np.diff(layout[name]["cuts"]) > 0)  # three regions exist
assert worst < 1e-9, worst

# ------------------------------------------------------------ FIGURE --
SCALE = 0.84                    # millimetres per unit, the same everywhere
PANEL_W, PANEL_H, BASE = 33.0, 33.5, 14.5   # mm; BASE = centre line height
fig = ms.figure(120, 42)
panels = [ms.axes(fig, 3 + 35.5 * k, 3, PANEL_W, PANEL_H) for k in range(3)]
ax_key = ms.axes(fig, 108.5, 3, 10.5, PANEL_H)
for ax, width in zip(panels + [ax_key], [PANEL_W] * 3 + [10.5]):
    ax.set_xlim(-width / 2 / SCALE, width / 2 / SCALE)
    ax.set_ylim(-BASE / SCALE, (PANEL_H - BASE) / SCALE)
    ax.set_aspect("equal")
    ax.axis("off")
renderer = fig.canvas.get_renderer()
r_max = max(max(item["r"]) for item in layout.values())
ALPHA = 0.42


def over(top, bottom):
    """Colour of a translucent fill (alpha = ALPHA) composited on another."""
    return ALPHA * np.array(to_rgb(top)) + (1 - ALPHA) * np.array(bottom)


def ink_for(rgb):
    """White on dark fills, dark on light ones (relative luminance)."""
    return "white" if np.dot(rgb, [0.2126, 0.7152, 0.0722]) < 0.45 else ms.INK


colour_a, colour_b = SETS.values()
fills = [over(colour_a, (1, 1, 1)), over(colour_b, over(colour_a, (1, 1, 1))),
         over(colour_b, (1, 1, 1))]              # A only, both, B only
mm2_per_item = []
for ax, letter, (name, item) in zip(panels, "abc", layout.items()):
    for x, r, colour in zip(item["x"], item["r"], SETS.values()):
        # fill without a stroke, then the outline as a separate thin line
        ax.add_patch(Circle((x, 0), r, facecolor=colour, alpha=ALPHA,
                            linewidth=0))
        ax.add_patch(Circle((x, 0), r, facecolor="none", edgecolor=colour,
                            linewidth=0.7))
        edge = ax.transData.transform([[x - r, 0], [x + r, 0]])[:, 0]
        r_mm = (edge[1] - edge[0]) / 2 / fig.dpi * 25.4
        mm2_per_item.append(np.pi * r_mm ** 2 / (np.pi * r * r))
    mids = (item["cuts"][:-1] + item["cuts"][1:]) / 2
    for x, value, chord, fill in zip(mids, item["values"],
                                     np.diff(item["cuts"]), fills):
        text = ax.text(x, 0, f"{value}", ha="center", va="center",
                       fontsize=ms.FS_TICK, color=ink_for(fill))
        width_mm = text.get_window_extent(renderer).width / fig.dpi * 25.4
        assert width_mm + 1.0 < chord * SCALE, (name, value)   # count fits
    for x, (set_name, colour), size in zip(mids[[0, 2]], SETS.items(),
                                           counts[name][:2]):
        ax.text(x, r_max + 4.2, set_name, ha="center", va="bottom",
                fontsize=ms.FS_TICK, color=colour, fontweight="bold")
        ax.text(x, r_max + 3.6, f"{size} hits", ha="center", va="top",
                fontsize=ms.FS_SMALL, color=ms.GREY_DARK)
    union = sum(item["values"])
    ax.text(0, -r_max - 1.6, f"Jaccard index {item['values'][1] / union:.2f}",
            ha="center", va="top", fontsize=ms.FS_TICK, color=ms.GREY_DARK)
    ms.panel_label(ax, letter, dx_pt=0, dy_pt=2)
    ax.annotate(name, xy=(0, 1), xycoords="axes fraction", xytext=(10, 2),
                textcoords="offset points", va="bottom", fontsize=ms.FS_BODY)

# key: a disc of 100 items at the common scale
r_key = np.sqrt(100 / np.pi)
ax_key.add_patch(Circle((0, 0), r_key, facecolor=ms.GREY_LIGHT, linewidth=0))
ax_key.text(0, 0, "100", ha="center", va="center", fontsize=ms.FS_TICK,
            color=ink_for(to_rgb(ms.GREY_LIGHT)))
ax_key.text(0, -r_key - 1.6, "hits; one\nscale, a–c", ha="center", va="top",
            fontsize=ms.FS_SMALL, color=ms.GREY_DARK, linespacing=1.2)

assert np.ptp(mm2_per_item) < 1e-9 * np.mean(mm2_per_item)   # one scale
assert abs(np.mean(mm2_per_item) - SCALE ** 2) < 1e-9
distances = ", ".join(f"{item['d']:.3f}" for item in layout.values())
print(f"fig141: self-check passed (lens area = intersection to "
      f"{worst:.1e} relative in {len(layout)} panels; all "
      f"{len(mm2_per_item)} discs at {np.mean(mm2_per_item):.4f} mm² per "
      f"item; centre distances {distances})")
ms.assert_aligned(panels + [ax_key])

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig141_euler_proportional.{ext}")
print("fig141_euler_proportional: saved png + pdf")
