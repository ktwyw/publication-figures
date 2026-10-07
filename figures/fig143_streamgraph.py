"""Fig. 143 - Streamgraph on the wiggle baseline (double column, 183 mm).

A stacked area chart on a flat baseline makes every layer inherit the
bumps of the layers beneath it. A streamgraph spreads that distortion
by moving the baseline: the "wiggle" of Byron and Wattenberg puts it at
g0 = -(1/(n+1)) * sum_i (n - i + 1) f_i, which minimises the summed
squared slopes of the layer boundaries, and the layers are ordered
inside-out so the largest sits in the middle where it is distorted
least. Layers are stroke-free fills, one category carries the story in
a saturated colour while the rest stay muted, each label is set where
its layer is thickest across the label's own width and only if the
rendered text fits (an outer layer that is too thin is named in the
margin beside it), and a scale bar replaces the meaningless y axis.
The self-check is that every layer's thickness equals its data, that
top minus baseline equals the column total at every time, that the
baseline satisfies the wiggle formula term by term, and that the n + 1
boundary curves therefore sum to zero.

Data: 7 categories observed at 121 weekly time steps, weeks 0-120
(arbitrary units); no smoothing, no hypothesis test. All data are simulated.
"""

import numpy as np
from matplotlib.colors import to_rgb

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(143)

# ------------------------------------------------------------- DATA ----
# flow[i, t]: value of category i at week t; each series is a floor plus
# a few smooth pulses, so layers swell and fade like real topics do.
T = 121
week = np.arange(T)
NAMES = [f"Category {c}" for c in "ABCDEFG"]
STORY = "Category C"                       # the one layer drawn in colour
SIZE = [34, 26, 18, 22, 15, 12, 9]         # typical level of each category
flow = np.empty((len(NAMES), T))
for i, level in enumerate(SIZE):
    centres = rng.uniform(-10, T + 10, 5)
    widths = rng.uniform(6, 16, 5)
    heights = rng.uniform(0.3, 1.3, 5)
    pulses = (heights[:, None] * np.exp(-0.5 * ((week - centres[:, None])
                                                / widths[:, None]) ** 2))
    flow[i] = level * (0.3 + pulses.sum(axis=0))
story = NAMES.index(STORY)
flow[story] = 4 + 46 / (1 + np.exp(-(week - 78) / 9.0))    # a late take-off

# ----------------------------------------------------------- LAYOUT ----
# Inside-out order: the largest series in the middle, then alternately
# above and below, so the smallest end up at the two outer edges.
stack = []
for rank, i in enumerate(np.argsort(-flow.sum(axis=1), kind="stable")):
    stack = stack + [i] if rank % 2 == 0 else [i] + stack
stack = np.array(stack)                    # bottom layer first
f = flow[stack]
n = len(stack)

# Wiggle baseline, then every boundary g_0 ... g_n by cumulative sums
weights = n - np.arange(1, n + 1) + 1      # (n - i + 1) for i = 1 ... n
baseline = -(weights[:, None] * f).sum(axis=0) / (n + 1)
bound = baseline + np.vstack([np.zeros(T), np.cumsum(f, axis=0)])

# ------------------------------------------------------- SELF-CHECK ---
assert sorted(stack) == list(range(n))
assert np.allclose(np.diff(bound, axis=0), f, rtol=0, atol=1e-10)
assert np.allclose(bound[-1] - bound[0], flow.sum(axis=0), rtol=0, atol=1e-10)
formula = np.zeros(T)
for i in range(1, n + 1):                  # g0 = -(1/(n+1)) sum (n-i+1) f_i
    formula -= (n - i + 1) * f[i - 1] / (n + 1)
assert np.allclose(bound[0], formula, rtol=0, atol=1e-10)
assert np.abs(bound.sum(axis=0)).max() < 1e-9      # boundaries sum to zero
middle = stack[n // 2]
assert middle == np.argmax(flow.sum(axis=1))       # largest in the middle
print(f"fig143: self-check passed ({n} layers x {T} steps; thickness = data "
      f"and top - baseline = total to 1e-10; wiggle formula holds, "
      f"boundaries sum to {np.abs(bound.sum(axis=0)).max():.1e}; "
      f"{NAMES[middle]} in the middle; peak total "
      f"{flow.sum(axis=0).max():.0f})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 72)
ax = ms.axes(fig, 5, 11, 158, 57)
ax.set_xlim(0, T - 1)
pad = 0.05 * np.ptp(bound)
ax.set_ylim(bound.min() - pad, bound.max() + pad)

# Muted tones alternate dark / light so neighbours separate without strokes
slate = (np.array(to_rgb(ms.BLUE)) + np.array(to_rgb(ms.GREY))) / 2
TINT = [0.42, 0.72, 0.57, 0.80]            # fraction of white mixed in
fills = [np.array(to_rgb(ms.VERMILLION)) if i == story
         else (1 - TINT[k % 4]) * slate + TINT[k % 4]
         for k, i in enumerate(stack)]
for k in range(n):
    ax.fill_between(week, bound[k], bound[k + 1], color=fills[k], linewidth=0)

# Direct labels: slide a box of the label's rendered size along the layer
# and keep the position with the most headroom (min top - max bottom over
# the box width); draw the label only if the box fits with a margin.
renderer = fig.canvas.get_renderer()
x_per_px, y_per_px = 1 / np.abs(np.diag(ax.transData.get_matrix())[:2])
inside = 0
for k, i in enumerate(stack):
    ink = "white" if fills[k] @ [0.2126, 0.7152, 0.0722] < 0.5 else ms.INK
    label = ax.text(0, 0, NAMES[i], ha="center", va="center_baseline",
                    fontsize=ms.FS_TICK, color=ink,
                    fontweight="bold" if i == story else "normal")
    box = label.get_window_extent(renderer)
    half = int(np.ceil(box.width * x_per_px / 2)) + 1      # steps
    height = box.height * y_per_px
    centre = np.arange(half, T - half)
    floor = np.array([bound[k, c - half:c + half + 1].max() for c in centre])
    roof = np.array([bound[k + 1, c - half:c + half + 1].min()
                     for c in centre])
    best = np.argmax(roof - floor)
    if roof[best] - floor[best] >= 1.25 * height:
        label.set_position((centre[best], (roof[best] + floor[best]) / 2))
        inside += 1
    elif k in (0, n - 1):
        # an outer layer too thin for its name: set it just outside, in
        # the empty margin, where that layer is thickest
        c = centre[np.argmax(f[k, centre])]
        span = slice(c - half, c + half + 1)
        outer = (bound[n, span].max() + 0.9 * height if k
                 else bound[0, span].min() - 0.9 * height)
        label.set(position=(c, outer), color=ms.GREY_DARK)
    else:
        label.remove()                     # no room, and no margin beside it

# time axis only; a scale bar says what thickness means
for side in ("left", "right", "top"):
    ax.spines[side].set_visible(False)
ax.set_yticks([])
ax.set_xticks(np.arange(0, T, 20))
ax.spines["bottom"].set_position(("outward", 2))
ax.set_xlabel("Time (weeks)")
BAR = 50.0                                 # units
x_bar = T - 1 + 3.5
ax.plot([x_bar, x_bar], [-BAR / 2, BAR / 2], color=ms.INK, lw=1.0,
        solid_capstyle="butt", clip_on=False)
ax.text(x_bar + 1.4, 0, f"{BAR:.0f} units", fontsize=ms.FS_TICK, va="center")
print(f"fig143: {inside} of {n} labels fit inside their layers")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig143_streamgraph.{ext}")
print("fig143_streamgraph: saved png + pdf")
