"""Fig. 144 - Bump chart of rankings over time (1.5 column, 120 mm).

When the question is "who overtook whom", plot ranks, not scores. Ranks
are derived here by sorting simulated scores at every period (rank 1 at
the top), and consecutive ranks are joined by smoothstep curves, which
leave and arrive horizontally so that every crossing happens between
periods and never at one. Three story items carry colour, thicker lines
and their rank inside each marker; the other six stay grey. Lines are
cut where they meet a marker so nothing runs under the numerals, and
names sit at both ends, aligned to the first and last rank, with the net
change in rank at the right. The self-check is that the ranks at every
period are a permutation of 1 ... k and equal scipy.stats.rankdata of
the scores, which have no ties.

Data: 9 items scored at 8 periods (arbitrary units); ranks by
descending score; no hypothesis test. All data are simulated.
"""

import numpy as np
from scipy import stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(144)

# ------------------------------------------------------------- DATA ----
# score[i, t]: score of item i at period t = a start level, a drift per
# period and noise. The story items are given a clear drift.
K, PERIODS = 9, 8
NAMES = [f"Item {i:02d}" for i in range(1, K + 1)]
STORY = {"Item 08": ms.BLUE, "Item 02": ms.VERMILLION, "Item 05": ms.GREEN}
DRIFT = {"Item 08": 5.5, "Item 02": -4.5, "Item 05": 1.2}
period = np.arange(1, PERIODS + 1)
start = np.linspace(72, 40, K) + rng.normal(0, 2.0, K)
drift = np.array([DRIFT.get(name, rng.normal(0, 1.0)) for name in NAMES])
score = (start[:, None] + drift[:, None] * (period - 1)
         + rng.normal(0, 3.0, (K, PERIODS)))

# ------------------------------------------------------------ RANKS ----
# rank 1 = highest score: sort each period, then invert the permutation
order = np.argsort(-score, axis=0, kind="stable")
rank = np.empty_like(order)
np.put_along_axis(rank, order, np.arange(1, K + 1)[:, None], axis=0)
change = rank[:, 0] - rank[:, -1]              # positive = moved up


def connector(r0, r1, samples=61):
    """Smoothstep from rank r0 to r1 over one period: s = 3u^2 - 2u^3 has
    zero slope at both ends, so curves leave and arrive horizontally."""
    u = np.linspace(0, 1, samples)
    return u, r0 + (r1 - r0) * u * u * (3 - 2 * u)


# ------------------------------------------------------- SELF-CHECK ---
assert all(np.unique(column).size == K for column in score.T)   # no ties
assert np.array_equal(np.sort(rank, axis=0),
                      np.tile(np.arange(1, K + 1)[:, None], (1, PERIODS)))
assert np.array_equal(rank, stats.rankdata(-score, axis=0).astype(int))
assert set(STORY) <= set(NAMES)
crossings = int(sum(np.sum((rank[i, :-1] - rank[j, :-1])
                           * (rank[i, 1:] - rank[j, 1:]) < 0)
                    for i in range(K) for j in range(i)))
top = NAMES[int(np.argmax(change))]
print(f"fig144: self-check passed ({K} items x {PERIODS} periods; ranks are "
      f"permutations of 1-{K} and equal scipy.stats.rankdata, no ties; "
      f"{crossings} overtakes; largest rise {top}, "
      f"rank {rank[NAMES.index(top), 0]} to {rank[NAMES.index(top), -1]})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 72)
ax = ms.axes(fig, 26, 10, 73, 55)
ax.set_xlim(0.75, PERIODS + 0.25)
ax.set_ylim(K + 0.55, 0.45)                   # rank 1 at the top
MARKER = 9.0                                  # story marker diameter (pt)
to_px = ax.transData.transform
cut_px = (MARKER / 2 + 0.6) / 72 * fig.dpi    # lines stop here, in pixels

for i in np.argsort([name in STORY for name in NAMES], kind="stable"):
    name = NAMES[i]                           # grey first, story on top
    hero = name in STORY
    colour = STORY.get(name, "#B9B9B9")
    for t in range(PERIODS - 1):
        u, y = connector(rank[i, t], rank[i, t + 1])
        x = period[t] + u
        if hero:   # drop the samples that fall inside either end marker
            px = to_px(np.column_stack([x, y]))
            clear = ((np.hypot(*(px - px[0]).T) > cut_px)
                     & (np.hypot(*(px - px[-1]).T) > cut_px))
            x, y = x[clear], y[clear]
        ax.plot(x, y, color=colour, lw=1.8 if hero else 0.9,
                solid_capstyle="butt", zorder=3 if hero else 2)
    if hero:
        ax.scatter(period, rank[i], s=MARKER ** 2, color=colour, linewidths=0,
                   zorder=4)
        for t in range(PERIODS):
            ax.text(period[t], rank[i, t], str(rank[i, t]), ha="center",
                    va="center_baseline", fontsize=ms.FS_SMALL, color="white",
                    fontweight="bold", zorder=5)
    else:
        ax.scatter(period, rank[i], s=7, color=colour, linewidths=0,
                   zorder=2.5)
    ink = colour if hero else ms.GREY_DARK
    weight = "bold" if hero else "normal"
    ax.text(period[0] - 0.33, rank[i, 0], name, ha="right", va="center",
            fontsize=ms.FS_TICK, color=ink, fontweight=weight)
    delta = f"+{change[i]}" if change[i] > 0 else str(change[i]).replace(
        "-", "−")
    ax.text(period[-1] + 0.33, rank[i, -1], name, ha="left", va="center",
            fontsize=ms.FS_TICK, color=ink, fontweight=weight)
    ax.text(period[-1] + 1.8, rank[i, -1], delta, ha="right", va="center",
            fontsize=ms.FS_TICK, color=ink, fontweight=weight)

# header for the change column; rank scale at the far left; period axis
ax.text(period[-1] + 1.8, 0.45, "Change\nin rank", ha="right", va="bottom",
        fontsize=ms.FS_SMALL, color=ms.GREY_DARK, linespacing=1.15)
for side in ("left", "right", "top"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_bounds(1, PERIODS)
ax.spines["bottom"].set_position(("outward", 1))
ax.set_xticks(period)
ax.set_xlabel("Period")
ax.set_yticks(np.arange(1, K + 1))
ax.tick_params(axis="y", length=0, pad=42)
ax.set_ylabel("Rank")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig144_bump_chart.{ext}")
print("fig144_bump_chart: saved png + pdf")
