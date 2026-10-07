"""Fig. 139 - Chord diagram of a directed flow matrix (single column, 89 mm).

Where fig026 lays a two-stage flow out as a Sankey, a chord diagram closes
the same bookkeeping into a circle so that every group is both a source
and a target. Each outer arc spans an angle proportional to the group's
total traffic (outflow + inflow); each flow owns a sub-arc of that width
at both of its ends, joined by a cubic Bezier ribbon in the colour of its
source and pulled back from the arc at its target end, so direction can
be read without arrowheads. The three largest flows are saturated, the
rest stay light. The self-check is the bookkeeping itself: the sub-arcs
of every group tile its arc exactly, and the arcs plus the gaps close the
circle at 2 pi.

Data: 6 x 6 matrix of directed flows between placeholder groups (30
non-zero flows, arbitrary units; diagonal empty). All data are simulated.
"""

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(139)

# ------------------------------------------------------------- DATA ----
# FLOW[i, j] = flow from group i to group j (rows: sources).
GROUPS = {"Group A": ms.BLUE, "Group B": ms.ORANGE, "Group C": ms.GREEN,
          "Group D": ms.PINK, "Group E": ms.SKY, "Group F": ms.GREY}
n = len(GROUPS)
FLOW = rng.integers(4, 19, (n, n)).astype(float)
FLOW[0, 3], FLOW[2, 1], FLOW[4, 0] = 64.0, 52.0, 45.0   # the planted story
np.fill_diagonal(FLOW, 0.0)
N_TOP = 3                               # flows to emphasise
TICK = 20.0                             # flow units between tick marks

# ----------------------------------------------------------- LAYOUT ----
# Angles run clockwise from 12 o'clock; lengths are millimetres on the page.
GAP = np.deg2rad(5.0)                   # empty angle between neighbours
R_OUT, R_IN = 27.0, 24.6                # outer arcs (mm)
R_RIBBON, PULL = 24.0, 1.6              # ribbon radius; target-end setback
total = FLOW.sum(axis=1) + FLOW.sum(axis=0)        # out + in per group
unit = (2 * np.pi - n * GAP) / total.sum()         # radians per flow unit
span = total * unit
start = GAP / 2 + np.concatenate([[0.0], np.cumsum(span + GAP)[:-1]])

# Sub-arcs. Walking clockwise along group i, partners are visited from
# the one just behind (i - 1) to the one just ahead (i + 1), outflow
# before inflow, so the two ribbons of a pair never cross each other.
out_arc = np.zeros((n, n, 2))           # [i, j]: at group i, flow i -> j
in_arc = np.zeros((n, n, 2))            # [i, j]: at group j, flow i -> j
for i in range(n):
    cursor = start[i]
    for offset in range(n - 1, 0, -1):
        j = (i + offset) % n
        out_arc[i, j] = cursor, cursor + FLOW[i, j] * unit
        cursor = out_arc[i, j, 1]
        in_arc[j, i] = cursor, cursor + FLOW[j, i] * unit
        cursor = in_arc[j, i, 1]


def on_circle(angle, radius):
    """Points (N, 2) at clockwise-from-top angles on a circle."""
    angle = np.atleast_1d(angle)
    return radius * np.column_stack([np.sin(angle), np.cos(angle)])


def ribbon(source, target):
    """Closed path: source sub-arc, Bezier, target sub-arc, Bezier back."""
    src = on_circle(np.linspace(*source, 24), R_RIBBON)
    dst = on_circle(np.linspace(*target, 24), R_RIBBON - PULL)

    # Control points sit on the four radii, at one depth for both sides.
    # Distant groups are drawn deep towards the centre; neighbours stay
    # shallow (a chord's mid-point radius is cos(half the separation))
    # but must still dip inward, or the ribbon would have no width.
    apart = abs((np.mean(source) - np.mean(target) + np.pi) % (2 * np.pi)
                - np.pi)
    pinch = max(0.2, 0.75 * np.cos(apart / 2))
    verts = np.vstack([src, pinch * src[-1], pinch * dst[0], dst,
                       pinch * dst[-1], pinch * src[0], src[0], src[0]])
    codes = ([Path.MOVETO] + [Path.LINETO] * 23 + [Path.CURVE4] * 3
             + [Path.LINETO] * 23 + [Path.CURVE4] * 3 + [Path.CLOSEPOLY])
    return Path(verts, codes)


pairs = [(i, j) for i in range(n) for j in range(n) if FLOW[i, j] > 0]
pairs.sort(key=lambda ij: FLOW[ij])               # largest drawn last
top = pairs[-N_TOP:][::-1]

# ------------------------------------------------------- SELF-CHECK ---
sub_width = (np.diff(out_arc, axis=2)[..., 0].sum(axis=1)
             + np.diff(in_arc, axis=2)[..., 0].sum(axis=0))
assert np.allclose(sub_width, span, rtol=0, atol=1e-12), sub_width - span
assert abs(span.sum() + n * GAP - 2 * np.pi) < 1e-12
ends = np.sort(np.concatenate([out_arc[i, j] for i, j in pairs]
                              + [in_arc[i, j] for i, j in pairs]))
assert np.all(np.diff(ends.reshape(-1, 2), axis=1) > 0)       # no overlap
assert np.allclose(np.diff(out_arc, axis=2)[..., 0],
                   np.diff(in_arc, axis=2)[..., 0])   # same width both ends
print(f"fig139: self-check passed ({len(pairs)} ribbons; sub-arcs tile all "
      f"{n} arcs to {np.abs(sub_width - span).max():.1e} rad; arcs + gaps = "
      f"{(span.sum() + n * GAP) / np.pi:.12f} pi; total flow "
      f"{FLOW.sum():.0f})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 89)
ax = ms.axes(fig, 0, 0, 89, 89)         # data units are millimetres
ax.set_xlim(-44.5, 44.5)
ax.set_ylim(-48.5, 40.5)                # circle sits high; key goes below
ax.set_aspect("equal")
ax.axis("off")
colours = list(GROUPS.values())

for i, j in pairs:
    hero = (i, j) in top
    ax.add_patch(PathPatch(ribbon(out_arc[i, j], in_arc[i, j]),
                           facecolor=colours[i], alpha=0.88 if hero else 0.26,
                           linewidth=0, zorder=3 if hero else 2))

# Group labels stay horizontal. Each two-line block (name, total) is
# pushed out along its radius until its box just clears the tick ring:
# the support distance of a w x h box in direction u is (w|ux| + h|uy|)/2.
renderer = fig.canvas.get_renderer()
MM_PER_PX = 25.4 / fig.dpi
ticks = []
for i, (name, colour) in enumerate(GROUPS.items()):
    # Wedge angles are counter-clockwise from 3 o'clock, in degrees
    ax.add_patch(Wedge((0, 0), R_OUT, 90 - np.degrees(start[i] + span[i]),
                       90 - np.degrees(start[i]), width=R_OUT - R_IN,
                       facecolor=colour, linewidth=0, zorder=4))
    at = start[i] + unit * np.arange(0, total[i] + 1e-9, TICK)
    ticks += list(zip(on_circle(at, R_OUT + 0.5), on_circle(at, R_OUT + 1.6)))
    label = ax.text(0, 0, name, ha="center", va="bottom", fontsize=ms.FS_BODY)
    value = ax.text(0, 0, f"{total[i]:.0f}", ha="center", va="top",
                    fontsize=ms.FS_TICK, color=ms.GREY_DARK)
    boxes = [t.get_window_extent(renderer) for t in (label, value)]
    width = max(b.width for b in boxes) * MM_PER_PX
    height = sum(b.height for b in boxes) * MM_PER_PX + 0.8
    u = on_circle(start[i] + span[i] / 2, 1.0)[0]
    centre = u * (R_OUT + 2.6 + (width * abs(u[0]) + height * abs(u[1])) / 2)
    seam = centre[1] + height / 2 - boxes[0].height * MM_PER_PX - 0.4
    label.set_position((centre[0], seam + 0.4))
    value.set_position((centre[0], seam - 0.4))
ax.add_collection(LineCollection(ticks, colors=ms.GREY_DARK, linewidths=0.45,
                                 capstyle="butt"))

# key: the emphasised flows, largest first, and how to read the diagram
names = list(GROUPS)
ax.text(-42.5, -37.0, "Largest flows", fontsize=ms.FS_TICK, fontweight="bold",
        va="center")
for row, (i, j) in enumerate(top):
    y = -40.2 - 2.9 * row
    ax.scatter(-41.6, y, marker="s", s=14, color=colours[i], linewidths=0)
    ax.text(-39.6, y, f"{names[i][-1]} → {names[j][-1]}   {FLOW[i, j]:.0f}",
            fontsize=ms.FS_TICK, va="center")
ax.text(42.5, -40.2, "Ribbon colour, source group\n"
        "Recessed end, target group\n"
        f"Label, out + in; tick, {TICK:.0f} units", fontsize=ms.FS_SMALL,
        color=ms.GREY_DARK, ha="right", va="top", linespacing=1.35)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig139_chord_diagram.{ext}")
print("fig139_chord_diagram: saved png + pdf")
