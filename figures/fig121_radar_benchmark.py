"""Fig. 121 - Radar benchmark with per-spoke scales (single column, 89 mm).

A radar chart is only honest when each spoke says what its radius means.
Here six tasks in different units, two of them lower-is-better, each get
their own worst-to-best scale, printed at both ends of the spoke, so
"further out" always means "better". The proposed method is the one
filled polygon; baselines are thin outlines. The polygons are built on a
bare polar axes, so the self-check can compare the shoelace area of the
Cartesian vertices with the closed form ½·sin(2π/k)·Σ rᵢrᵢ₊₁, and
confirm that the proposed method encloses the largest area.

Statistics: n = 5 random seeds per method and task; vertices, mean over
seeds; no interval is drawn and no test is made. All data are simulated.
"""

import numpy as np

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(121)
N_SEEDS = 5
HOLE = 0.30          # radius of the worst value; the hole holds its label
METHODS = {"Baseline A": (ms.GREY, (0, (4, 2))),       # (colour, dashes)
           "Baseline B": (ms.BLUE, "-"),
           "Proposed": (ms.VERMILLION, "-")}

# ------------------------------------------------------------- DATA ----
# task: (tick step, higher is better, expected score of A, B, proposed)
TASKS = {"Classification\naccuracy (%)": (5, True, (78.2, 81.5, 85.9)),
         "Detection\nAP": (5, True, (41.0, 46.3, 48.8)),
         "Segmentation\nmIoU": (5, True, (44.1, 43.2, 50.6)),
         "Calibration\nerror (%)": (2, False, (6.9, 4.8, 3.1)),
         "Latency\n(ms)": (20, False, (31.0, 68.0, 44.0)),
         "Robust\naccuracy (%)": (10, True, (48.5, 57.0, 66.2))}
K = len(TASKS)
seeds = np.array([[rng.normal(centre, 0.02 * centre, N_SEEDS)
                   for centre in expected]
                  for _step, _higher, expected in TASKS.values()])
score = seeds.mean(axis=2)                           # (task, method)

# ------------------------------------------------------------ SCALES ---
# Each spoke runs from a round value beyond its worst score (so no polygon
# reaches the hole) to the first round value at or beyond its best score
higher = np.array([h for _step, h, _e in TASKS.values()])
step = np.array([s for s, _h, _e in TASKS.values()])
pad = 0.3 * np.ptp(score, axis=1)
low = step * np.floor((score.min(axis=1) - pad * higher) / step)
high = step * np.ceil((score.max(axis=1) + pad * ~higher) / step)
worst, best = np.where(higher, low, high), np.where(higher, high, low)
radius = HOLE + (1 - HOLE) * ((score - worst[:, None])
                              / (best - worst)[:, None])
theta = 2 * np.pi * np.arange(K) / K                 # clockwise from north


def shoelace(r):
    """Polygon area from the Cartesian vertices."""
    x, y = r * np.sin(theta), r * np.cos(theta)
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


area = np.array([shoelace(radius[:, m]) for m in range(len(METHODS))])

# ------------------------------------------------------- SELF-CHECK ---
closed_form = 0.5 * np.sin(2 * np.pi / K) * np.sum(
    radius * np.roll(radius, -1, axis=0), axis=0)
assert np.allclose(area, closed_form, atol=1e-12), (area, closed_form)
assert np.all((radius >= HOLE) & (radius <= 1))
assert np.argmax(area) == list(METHODS).index("Proposed"), area
print("fig121: self-check passed (shoelace area = ½·sin(2π/k)·Σ rᵢrᵢ₊₁: "
      + ", ".join(f"{name} {a:.3f}" for name, a in zip(METHODS, area))
      + "; proposed largest)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 82)
ax = ms.axes(fig, 20.5, 20.5, 48, 48, projection="polar")
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.spines["polar"].set_visible(False)
loop = np.append(theta, theta[0])

# scaffold: one ring per quarter of every spoke's range, and the spokes
for u in np.linspace(0, 1, 5):
    ax.plot(loop, np.full(K + 1, HOLE + (1 - HOLE) * u), color=ms.GREY_LIGHT,
            lw=0.7 if u == 1 else 0.45, zorder=1)
for angle in theta:
    ax.plot([angle, angle], [HOLE, 1], color=ms.GREY_LIGHT, lw=0.45, zorder=1)

for m, (method, (colour, dashes)) in enumerate(METHODS.items()):
    hero = method == "Proposed"
    r_loop = np.append(radius[:, m], radius[0, m])
    if hero:
        ax.fill(loop, r_loop, color=colour, alpha=0.16, lw=0, zorder=2)
    ax.plot(loop, r_loop, color=colour, ls=dashes, lw=1.5 if hero else 0.9,
            marker="o", markersize=3.0 if hero else 2.0, markeredgewidth=0,
            zorder=4 if hero else 3, clip_on=False, label=method)

# scale values at both ends of each spoke, task names beyond the best end
for k, name in enumerate(TASKS):
    ax.text(theta[k], HOLE - 0.095, f"{worst[k]:g}", ha="center",
            va="center", fontsize=ms.FS_SMALL, color=ms.GREY_DARK)
    ax.text(theta[k], 1.075, f"{best[k]:g}", ha="center", va="center",
            fontsize=ms.FS_SMALL, color=ms.GREY_DARK)
    side = np.sin(theta[k])
    ax.text(theta[k], 1.24 if abs(side) > 0.1 else 1.27, name,
            ha="left" if side > 0.1 else "right" if side < -0.1 else "center",
            va="center", fontsize=ms.FS_TICK, linespacing=1.1)

fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.012), ncols=3,
           handlelength=2.0, columnspacing=1.6)
fig.text(0.5, 0.085, "Outward is better on every spoke; rings mark "
         "quarters of each range", ha="center", fontsize=ms.FS_SMALL,
         color=ms.GREY_DARK)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig121_radar_benchmark.{ext}")
print("fig121_radar_benchmark: saved png + pdf")
