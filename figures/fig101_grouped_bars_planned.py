"""Fig. 101 - Grouped bars with planned comparisons (single column, 89 mm).

Extends fig004 from one factor to two: genotype x treatment bars share
a second label row under the axis, every replicate is drawn over its
mean ± s.d., and the three planned comparisons carry exact
Holm-adjusted Welch P values instead of stars, on brackets placed from
the data maximum. The self-check is on the Holm step itself: adjusted P
values never fall below the raw ones or exceed 1, the smallest raw P is
multiplied by exactly the number of comparisons, and every bracket
clears the tallest point it spans.

Statistics: n = 6 biological replicates per group; bars, mean; error
bars, s.d.; two-sided Welch's t-test, Holm-adjusted for three planned
comparisons. All data are simulated.
"""

import numpy as np
from matplotlib.patches import Patch
from scipy import stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

GROUPS = [("WT", "Vehicle"), ("WT", "Compound A"),
          ("KO", "Vehicle"), ("KO", "Compound A")]
CENTRE = {("WT", "Vehicle"): 100, ("WT", "Compound A"): 100,
          ("KO", "Vehicle"): 58, ("KO", "Compound A"): 91}
PAIRS = [(0, 2), (2, 3), (0, 1)]     # planned comparisons, GROUPS indices
X = np.array([0.0, 1.0, 2.6, 3.6])   # a gap separates the genotypes
COLOURS = [ms.GREY_LIGHT, ms.BLUE] * 2

# ------------------------------------------------------------- DATA ----
rng = np.random.default_rng(11)
values = [rng.normal(CENTRE[group], 8, 6) for group in GROUPS]
means = np.array([v.mean() for v in values])
sds = np.array([v.std(ddof=1) for v in values])


# ------------------------------------------------------- STATISTICS ----
def holm(pvalues):
    """Holm step-down adjustment (monotone, capped at 1)."""
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


raw = np.array([stats.ttest_ind(values[i], values[j], equal_var=False).pvalue
                for i, j in PAIRS])
adjusted = holm(raw)

# brackets are placed from the data, not from fixed axis values
top = max(float(v.max()) for v in values)
TICK, TEXT_PAD = 0.025 * top, 0.012 * top
levels = {(0, 1): 1.07 * top, (2, 3): 1.07 * top, (0, 2): 1.21 * top}

# ------------------------------------------------------- SELF-CHECK ---
assert np.all(adjusted >= raw) and np.all(adjusted <= 1.0)
smallest = int(np.argmin(raw))
assert np.isclose(adjusted[smallest], min(len(PAIRS) * raw[smallest], 1.0),
                  rtol=1e-12, atol=0.0)
assert np.all(np.diff(adjusted[np.argsort(raw)]) >= 0)   # order preserved
for (i, j), level in levels.items():
    tallest = max(max(values[k].max(), means[k] + sds[k])
                  for k in range(i, j + 1))
    assert level - TICK > tallest, ((i, j), level, tallest)
print("fig101: self-check passed (Holm P >= raw P, all <= 1; smallest "
      f"raw P {raw[smallest]:.3g} x {len(PAIRS)} = {adjusted[smallest]:.3g}"
      f"; brackets clear the data maximum {top:.1f})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 64)
gs = ms.grid(fig, 1, 1, left=13, right=27, top=5, bottom=12)
ax = fig.add_subplot(gs[0])

ax.bar(X, means, width=0.72, color=COLOURS, edgecolor=ms.INK, linewidth=0.6,
       zorder=2)
ax.errorbar(X, means, yerr=sds, fmt="none", ecolor=ms.INK, elinewidth=0.6,
            capsize=2, capthick=0.6, zorder=4)
jitter = np.random.default_rng(7)    # horizontal jitter: display, not data
for xi, v in zip(X, values):
    ax.scatter(xi + jitter.uniform(-0.2, 0.2, v.size), v, s=7,
               facecolor="white", edgecolor=ms.INK, linewidth=0.5, zorder=5)

for (i, j), p in zip(PAIRS, adjusted):
    ms.bracket(ax, X[i], X[j], levels[(i, j)], ms.format_p(p), TICK,
               TEXT_PAD)

ax.set_xticks(X, ["Vehicle", "Cpd A", "Vehicle", "Cpd A"])
for centre, name in ((0.5, "Wild type"), (3.1, "Knockout")):
    ax.text(centre, -0.135, name, transform=ax.get_xaxis_transform(),
            ha="center", va="top")
ax.set_ylabel("ATP content (% of WT vehicle)")
ax.set_xlim(-0.65, 4.25)
ax.set_ylim(0, 1.32 * top)
ax.set_yticks(np.arange(0, 121, 20))
ax.spines["left"].set_bounds(0, 120)

handles = [Patch(facecolor=ms.GREY_LIGHT, edgecolor=ms.INK, linewidth=0.6,
                 label="Vehicle"),
           Patch(facecolor=ms.BLUE, edgecolor=ms.INK, linewidth=0.6,
                 label="Compound A")]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
          title="Treatment", alignment="left")
ax.text(1.04, 0.58, "n = 6 biological\nreplicates per group\n"
        "Bars: mean ± s.d.\nHolm-adjusted\nWelch's t-test",
        transform=ax.transAxes, fontsize=ms.FS_SMALL, va="top",
        color=ms.GREY_DARK)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig101_grouped_bars_planned.{ext}")
print("fig101_grouped_bars_planned: saved png + pdf")
