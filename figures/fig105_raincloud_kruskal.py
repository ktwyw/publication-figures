"""Fig. 105 - Raincloud plot with a bimodal group (single column, 89 mm).

Extends fig006, whose violins come from violinplot and whose groups are
all unimodal: here each cloud is a Gaussian kernel density evaluated in
the script and trimmed where it becomes negligible, so no cloud runs on
to the axis limits, and Class D is built from two populations that its
box alone would hide. A header row gives the Kruskal-Wallis test and
n. The self-check: every
density integrates to 1 over the evaluation grid within 1e-3 before it
is rescaled for display, Class D has exactly two density maxima and
every other class one (prominence >= 10% of the peak), and every
trimmed cloud ends inside the y-limits.

Statistics: n = 45 cells per class; box, median and IQR; whiskers,
1.5 x IQR; half-violin, Gaussian kernel density (Scott bandwidth);
Kruskal-Wallis test across classes. All data are simulated.
"""

import numpy as np
from scipy import signal, stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

CLASSES = {"Class A": ms.BLUE, "Class B": ms.GREEN, "Class C": ms.PINK,
           "Class D": ms.VERMILLION}
BIMODAL = "Class D"
N_CELLS = 45
Y_LIM = (80.0, 320.0)
CLOUD_WIDTH = 0.36       # x extent of the tallest point of each cloud
CLOUD_TRIM = 0.004       # clouds end where they are thinner than this
PROMINENCE = 0.10        # of the peak density, to count a mode

# ------------------------------------------------------------- DATA ----
rng = np.random.default_rng(505)
samples = {
    "Class A": rng.normal(142, 16, N_CELLS),
    "Class B": rng.normal(168, 22, N_CELLS),
    "Class C": rng.lognormal(5.247, 0.16, N_CELLS),
    "Class D": np.concatenate([rng.normal(150, 11, N_CELLS // 3),
                               rng.normal(232, 17,
                                          N_CELLS - N_CELLS // 3)]),
}
kruskal = stats.kruskal(*samples.values())

grid = np.linspace(60, 330, 400)
density = {name: stats.gaussian_kde(values)(grid)
           for name, values in samples.items()}
cloud = {name: d / d.max() * CLOUD_WIDTH for name, d in density.items()}
body = {name: c > CLOUD_TRIM for name, c in cloud.items()}

# ------------------------------------------------------- SELF-CHECK ---
area, modes = {}, {}
for name, d in density.items():
    area[name] = float(np.sum((d[1:] + d[:-1]) / 2 * np.diff(grid)))
    assert abs(area[name] - 1.0) < 1e-3, (name, area[name])
    peaks, _ = signal.find_peaks(d, prominence=PROMINENCE * d.max())
    modes[name] = peaks.size
    assert modes[name] == (2 if name == BIMODAL else 1), (name, modes[name])
    drawn = grid[body[name]]
    assert Y_LIM[0] < drawn.min() and drawn.max() < Y_LIM[1], (name, drawn)
low_mode, high_mode = grid[signal.find_peaks(
    density[BIMODAL], prominence=PROMINENCE * density[BIMODAL].max())[0]]
median_d = float(np.median(samples[BIMODAL]))
print("fig105: self-check passed (KDE areas "
      + ", ".join(f"{a:.5f}" for a in area.values())
      + "; modes " + ", ".join(str(m) for m in modes.values())
      + f"; {BIMODAL} maxima at {low_mode:.0f} and {high_mode:.0f}, "
      f"median {median_d:.0f}; clouds inside the y-limits)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 64)
gs = ms.grid(fig, 1, 1, left=14, right=4, top=8, bottom=10)
ax = fig.add_subplot(gs[0])

jitter = np.random.default_rng(5)    # horizontal jitter: display, not data
for i, (name, colour) in enumerate(CLASSES.items()):
    values = samples[name]
    keep = body[name]                # trim the flat tails of the estimate
    edge = i + 0.10 + cloud[name][keep]
    ax.fill_betweenx(grid[keep], i + 0.10, edge, color=colour, alpha=0.55,
                     linewidth=0)
    ax.plot(edge, grid[keep], color=colour, lw=0.7)
    ax.boxplot(values, positions=[i], widths=0.11, showfliers=False,
               patch_artist=True, manage_ticks=False, zorder=4,
               boxprops=dict(facecolor="white", edgecolor=ms.INK,
                             linewidth=0.6),
               medianprops=dict(color=ms.INK, linewidth=1.0),
               whiskerprops=dict(color=ms.INK, linewidth=0.6),
               capprops=dict(linewidth=0))
    ax.scatter(i - 0.24 + jitter.uniform(-0.09, 0.09, values.size), values,
               s=5, color=colour, alpha=0.75, linewidths=0, zorder=3)

ax.set_xticks(range(len(CLASSES)), list(CLASSES))
ax.set_xlim(-0.5, len(CLASSES) - 0.4)
ax.set_ylim(*Y_LIM)
ax.set_yticks(np.arange(100, 301, 50))
ax.set_ylabel("Soma area (µm²)")
ax.tick_params(axis="x", length=0, pad=4)
ax.text(0, 1.03, f"Kruskal–Wallis H = {kruskal.statistic:.1f}, "
        f"{ms.format_p(kruskal.pvalue)}; n = {N_CELLS} cells per class",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=ms.FS_SMALL,
        color=ms.GREY_DARK)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig105_raincloud_kruskal.{ext}")
print("fig105_raincloud_kruskal: saved png + pdf")
