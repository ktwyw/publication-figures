"""Fig. 124 - Parallel coordinates of a sweep (double column, 183 mm).

A hyperparameter sweep, one line per training run across five axes,
each with its own scale in real units: a log axis (learning rate), a
categorical axis (optimizer), a power-of-two axis (batch size) and two
linear ones. Lines are coloured by validation accuracy on a reversed
viridis scale (dark = accurate) and drawn worst first, so the good
region stands out on white; the five best runs are redrawn thick on top.
Every axis carries a clear lane for its tick labels: lines stop at the
lane's left rail and resume, at the same height, from the axis itself,
so no label is ever crossed. Runs that share a level on a discrete
axis are fanned out by a small display-only offset, so the density
there stays visible. The self-check is that each axis maps its
lower limit to 0 and its upper limit to 1, that the inverse map returns
every run's value, and that the labelled best run is the argmax of the
score.

Statistics: n = 150 runs, one random-search draw each (a single training
run, no replicates); colour, validation accuracy; no interval and no
test. All data are simulated.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(124)
N_RUNS, N_TOP = 150, 5
OPTIMIZERS = ["SGD", "Adam", "AdamW"]

# ------------------------------------------------------------- DATA ----
# one random-search draw per run; the optimizer is stored as its index
lr = 10 ** rng.uniform(-4, -1, N_RUNS)
optimizer = rng.integers(0, 3, N_RUNS)
batch = 2 ** rng.integers(4, 9, N_RUNS)              # 16 ... 256
layers = rng.integers(2, 13, N_RUNS)
dropout = rng.uniform(0.0, 0.5, N_RUNS)
best_log_lr = np.array([-1.6, -2.7, -2.5])[optimizer]    # SGD wants more
score = (93.5 + np.array([-1.2, 0.0, 0.5])[optimizer]
         - 3.0 * (np.log10(lr) - best_log_lr) ** 2
         - 0.9 * np.abs(np.log2(batch / 64))
         - 0.22 * (layers - 8) ** 2
         - 70 * (dropout - 0.2) ** 2
         + rng.normal(0, 0.5, N_RUNS))               # validation accuracy (%)


# ------------------------------------------------------------- AXES ----
def make_axis(values, limits, ticks, labels=None, log=False):
    """An axis is its data, a map onto [0, 1], the inverse, and its ticks."""
    to, back = ((np.log10, lambda z: 10.0 ** z) if log
                else (np.asarray, np.asarray))
    low, high = float(to(limits[0])), float(to(limits[1]))
    return dict(
        values=values, limits=limits, ticks=ticks,
        labels=labels or [f"{t:g}" for t in ticks],
        forward=lambda v: (to(v) - low) / (high - low),
        inverse=lambda u: back(low + np.asarray(u) * (high - low)))


AXES = {
    "Learning rate": make_axis(lr, (1e-4, 1e-1), [1e-4, 1e-3, 1e-2, 1e-1],
                               log=True),
    "Optimizer": make_axis(optimizer, (0, 2), [0, 1, 2], OPTIMIZERS),
    "Batch size": make_axis(batch, (16, 256), [16, 32, 64, 128, 256],
                            log=True),
    "Layers": make_axis(layers, (2, 12), [2, 4, 6, 8, 10, 12]),
    "Dropout": make_axis(dropout, (0.0, 0.5), np.linspace(0, 0.5, 6)),
}
unit = np.column_stack([a["forward"](a["values"]) for a in AXES.values()])
# display only: fan out runs that share a level on the three discrete axes
DISCRETE = [title in ("Optimizer", "Batch size", "Layers") for title in AXES]
drawn = unit + np.where(DISCRETE, rng.uniform(-0.022, 0.022, unit.shape), 0)
rank = np.argsort(score)                             # worst first
top, best = rank[-N_TOP:], rank[-1]

# ------------------------------------------------------- SELF-CHECK ---
for title, a in AXES.items():
    assert np.allclose(a["forward"](np.array(a["limits"])), [0, 1]), title
    assert np.allclose(a["inverse"](a["forward"](a["values"])), a["values"],
                       rtol=1e-12), title
assert unit.min() >= 0 and unit.max() <= 1
assert best == np.argmax(score) and set(top) == set(np.argsort(-score)[:5])
print(f"fig124: self-check passed ({len(AXES)} axes map limits to 0 and 1 "
      f"and round-trip {N_RUNS} runs; best run {score[best]:.1f}%: lr = "
      f"{lr[best]:.2g}, {OPTIMIZERS[optimizer[best]]}, batch {batch[best]}, "
      f"{layers[best]} layers, dropout {dropout[best]:.2f})")

# ------------------------------------------------------------ FIGURE --
# x is in millimetres from the first axis; y is the unit interval
WIDTH = 136.0                             # position of the last axis
CMAP = plt.cm.viridis_r                   # dark = high, salient on white
fig = ms.figure(183, 74)
ax = ms.axes(fig, 15, 9, 147, 55)
ax.set_xlim(0, 147)
ax.set_ylim(-0.04, 1.04)
ax.axis("off")
x_axis = np.linspace(0, WIDTH, len(AXES))
# lane width from the longest tick label (about 1.45 mm per character)
lane = np.array([4.6 + 1.45 * max(map(len, a["labels"]))
                 for a in AXES.values()])
norm = plt.Normalize(score.min(), score.max())


def polyline(run):
    """Segments of one run: from each axis to the next axis's lane rail."""
    return [[(x_axis[k], drawn[run, k]),
             (x_axis[k + 1] - lane[k + 1], drawn[run, k + 1])]
            for k in range(len(AXES) - 1)]


for runs, width, alpha in ((rank[:-N_TOP], 0.5, 0.45), (top, 1.5, 1.0)):
    for run in runs:                      # worst first: the best end on top
        ax.add_collection(LineCollection(
            polyline(run), colors=[CMAP(norm(score[run]))],
            linewidths=width, alpha=alpha, capstyle="butt"))

for k, (title, a) in enumerate(AXES.items()):
    x, at = x_axis[k], a["forward"](np.array(a["ticks"], dtype=float))
    ax.plot([x, x], [0, 1], color=ms.INK, lw=0.7, zorder=5)
    if k:                                 # the rail where incoming lines land
        ax.plot([x - lane[k]] * 2, [0, 1], color=ms.GREY_LIGHT, lw=0.5,
                zorder=5)
    ax.hlines(at, x - 1.0, x, color=ms.INK, lw=0.6, zorder=5)
    for y, label in zip(at, a["labels"]):
        ax.text(x - 1.7, y, label, ha="right", va="center",
                fontsize=ms.FS_TICK)
    ax.text(x, 1.07, title, ha="center", va="bottom")
    ax.plot(x, drawn[best, k], "o", color=CMAP(norm(score[best])),
            markersize=3.4, markeredgecolor=ms.INK, markeredgewidth=0.5,
            zorder=6)

ax.text(WIDTH + 2.0, unit[best, -1], f"Best run\n{score[best]:.1f}%",
        ha="left", va="center", fontsize=ms.FS_TICK, linespacing=1.15)
ax.text(0, -0.105, f"n = {N_RUNS} runs; thick lines, the {N_TOP} highest "
        "validation accuracies; lines pause at each axis to keep its "
        "tick labels clear", ha="left", va="center",
        fontsize=ms.FS_TICK, color=ms.GREY_DARK)

cax = ms.axes(fig, 170, 9 + 55 * 0.04 / 1.08, 2.4, 55 / 1.08)
bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
bar.set_label("Validation accuracy (%)")
bar.outline.set_visible(False)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig124_parallel_coordinates.{ext}")
print("fig124_parallel_coordinates: saved png + pdf")
