"""Fig. 142 - Squarified treemap of a two-level hierarchy (1.5 column, 120 mm).

A treemap gives every leaf an area proportional to its value; what makes
it readable is the shape of those areas. This one codes the squarified
layout of Bruls, Huizing and van Wijk from scratch: values are laid in
rows along the shorter side of the remaining rectangle, and a row is
closed as soon as one more item would worsen its worst aspect ratio. The
layout runs twice, groups on the canvas and then leaves inside each
group; leaves take tints of their group colour (largest darkest), groups
are separated by white gutters cut at drawing time, and a label is drawn
only where the rendered text fits its rectangle. The self-check is that
every rectangle's area is its value times one scale, that the areas sum
to the canvas, that no two rectangles overlap, and that the worst aspect
ratio beats a plain slice-and-dice layout of the same data (both
printed).

Data: 30 leaves in 5 groups, arbitrary positive values; no hypothesis
test. All data are simulated.
"""

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(142)

# ------------------------------------------------------------- DATA ----
# group -> (colour, number of leaves); leaf values are log-normal counts
GROUPS = {"Group A": (ms.BLUE, 8), "Group B": (ms.ORANGE, 7),
          "Group C": (ms.GREEN, 6), "Group D": (ms.PINK, 5),
          "Group E": (ms.GREY, 4)}
leaves = []                                    # (group, name, value)
for group, (_colour, size) in GROUPS.items():
    for value in np.sort(rng.lognormal(3.4, 0.75, size))[::-1]:
        leaves.append((group, f"Item {len(leaves) + 1:02d}",
                       float(np.rint(value))))
group_of = np.array([leaf[0] for leaf in leaves])
value = np.array([leaf[2] for leaf in leaves])
W, H = 114.0, 70.0                             # treemap canvas (mm)


# ----------------------------------------------------------- LAYOUT ----
def worst_ratio(row, side):
    """Worst aspect ratio of a row of areas laid along a side."""
    total = sum(row)
    return max(max(side ** 2 * a / total ** 2, total ** 2 / (side ** 2 * a))
               for a in row)


def squarify(areas, x, y, w, h):
    """Rectangles (x, y, w, h) for areas in descending order that sum to
    w * h (Bruls, Huizing & van Wijk 2000)."""
    areas, rects = list(areas), []
    while areas:
        side = min(w, h)                       # rows run along the short side
        row = [areas.pop(0)]
        while areas and (worst_ratio(row + [areas[0]], side)
                         <= worst_ratio(row, side)):
            row.append(areas.pop(0))
        thick, offset = sum(row) / side, 0.0
        for a in row:
            if w >= h:                         # a column at the left edge
                rects.append((x, y + offset, thick, a / thick))
            else:                              # a row at the top edge
                rects.append((x + offset, y, a / thick, thick))
            offset += a / thick
        if w >= h:
            x, w = x + thick, w - thick
        else:
            y, h = y + thick, h - thick
    return rects


def slice_and_dice(areas, x, y, w, h, vertical):
    """Baseline: cut the rectangle into parallel strips, in order."""
    start = np.concatenate([[0.0], np.cumsum(areas)[:-1]]) / (w * h)
    share = np.asarray(areas) / (w * h)
    if vertical:
        return [(x + s * w, y, f * w, h) for s, f in zip(start, share)]
    return [(x, y + s * h, w, f * h) for s, f in zip(start, share)]


def two_level(outer, inner):
    """Lay out the groups on the canvas, then the leaves in each group."""
    scale = W * H / value.sum()                # mm^2 per unit of value
    totals = {g: value[group_of == g].sum() for g in GROUPS}
    order = sorted(GROUPS, key=totals.get, reverse=True)
    frames = dict(zip(order, outer([totals[g] * scale for g in order],
                                   0.0, 0.0, W, H)))
    rects = np.zeros((value.size, 4))
    for g, frame in frames.items():
        rects[group_of == g] = inner(value[group_of == g] * scale, *frame)
    return rects, frames, scale


rects, frames, scale = two_level(squarify, squarify)
plain = two_level(lambda *args: slice_and_dice(*args, vertical=True),
                  lambda *args: slice_and_dice(*args, vertical=False))[0]


def aspect(r):
    return np.maximum(r[:, 2] / r[:, 3], r[:, 3] / r[:, 2])


# ------------------------------------------------------- SELF-CHECK ---
area = rects[:, 2] * rects[:, 3]
assert np.allclose(area, value * scale, rtol=1e-9, atol=0)
assert abs(area.sum() - W * H) < 1e-9 * W * H
x0, y0, x1, y1 = (rects[:, 0], rects[:, 1], rects[:, 0] + rects[:, 2],
                  rects[:, 1] + rects[:, 3])
assert x0.min() > -1e-9 and y0.min() > -1e-9
assert x1.max() < W + 1e-9 and y1.max() < H + 1e-9
shared_w = np.minimum.outer(x1, x1) - np.maximum.outer(x0, x0)
shared_h = np.minimum.outer(y1, y1) - np.maximum.outer(y0, y0)
overlap = np.clip(shared_w, 0, None) * np.clip(shared_h, 0, None)
np.fill_diagonal(overlap, 0.0)
assert overlap.max() < 1e-9, overlap.max()
assert np.allclose(plain[:, 2] * plain[:, 3], area, rtol=1e-9)
assert aspect(rects).max() < aspect(plain).max()
print(f"fig142: self-check passed ({value.size} leaves, {scale:.4f} mm² per "
      f"unit, areas sum to {area.sum():.1f} = {W * H:.1f} mm², no overlap; "
      f"worst aspect ratio {aspect(rects).max():.2f} squarified vs "
      f"{aspect(plain).max():.1f} slice-and-dice; mean "
      f"{aspect(rects).mean():.2f} vs {aspect(plain).mean():.1f})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 79)
ax = ms.axes(fig, 3, 3, W, H)              # data units are mm, y downward
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
renderer = fig.canvas.get_renderer()
MM_PER_PX = 25.4 / fig.dpi
HAIR, GUTTER, PAD = 0.15, 0.7, 0.9         # mm: leaf gap, group gap, text


def place(x, y, text, room_w, room_h, **style):
    """Text with its top-left corner at (x, y), kept only if it fits the
    room; returns the height it used (mm)."""
    label = ax.text(x, y, text, ha="left", va="top", **style)
    box = label.get_window_extent(renderer)
    if box.width * MM_PER_PX > room_w or box.height * MM_PER_PX > room_h:
        label.remove()
        return 0.0
    return box.height * MM_PER_PX


for (group, name, v), (x, y, w, h) in zip(leaves, rects):
    gx, gy, gw, gh = frames[group]
    # The layout is exact; the gaps are cut from each rectangle only when
    # drawing: a hairline between leaves, a wider gutter on group edges.
    left = GUTTER if np.isclose(x, gx) else HAIR
    top = GUTTER if np.isclose(y, gy) else HAIR
    right = GUTTER if np.isclose(x + w, gx + gw) else HAIR
    bottom = GUTTER if np.isclose(y + h, gy + gh) else HAIR
    first = value[group_of == group].max()
    tint = 0.62 * (1 - v / first) ** 0.6     # 0 = pure group colour
    fill = (1 - tint) * np.array(to_rgb(GROUPS[group][0])) + tint
    ax.add_patch(Rectangle((x + left, y + top), w - left - right,
                           h - top - bottom, facecolor=fill, linewidth=0))
    ink = "white" if fill @ [0.2126, 0.7152, 0.0722] < 0.5 else ms.INK
    tx, ty = x + left + PAD, y + top + PAD
    room_w, room_h = w - left - right - 2 * PAD, h - top - bottom - 2 * PAD
    if v == first:                           # group name on its lead leaf
        used = place(tx, ty, group, room_w, room_h, fontsize=ms.FS_BODY,
                     fontweight="bold", color=ink)
        ty, room_h = ty + used + 0.7, room_h - used - 0.7
    for text in (f"{name}\n{v:.0f}", name, f"{v:.0f}"):
        if place(tx, ty, text, room_w, room_h, fontsize=ms.FS_SMALL,
                 color=ink, linespacing=1.15):
            break

ax.text(0.6, -1.6, "Area is proportional to value", fontsize=ms.FS_TICK,
        va="bottom")
ax.text(W - 0.6, -1.6, f"{value.size} leaves in {len(GROUPS)} groups; total "
        f"{value.sum():,.0f}; worst aspect ratio {aspect(rects).max():.1f}",
        fontsize=ms.FS_TICK, color=ms.GREY_DARK, ha="right", va="bottom")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig142_treemap_squarified.{ext}")
print("fig142_treemap_squarified: saved png + pdf")
