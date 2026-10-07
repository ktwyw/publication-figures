"""Fig. 122 - Critical-difference diagram of mean ranks (1.5 column, 120 mm).

The standard way to compare several classifiers over many datasets
(Demšar 2006): rank the methods within each dataset, test the average
ranks with the Friedman test, then join every group of methods whose
average ranks differ by less than the Nemenyi critical difference
CD = q·sqrt(k(k+1)/(6N)). Ranks run right to left so the best method
sits at the right; labels hang on elbow leaders so no name touches the
axis. The self-check is that the average ranks sum to k(k+1)/2 and that
the hand-coded Friedman chi-square reproduces
scipy.stats.friedmanchisquare.

Statistics: n = 24 datasets, k = 6 methods; points, average rank of the
accuracy (1 = best); Friedman test, then two-sided Nemenyi post hoc test
at α = 0.05 (thick bars join methods that do not differ). All data are
simulated.
"""

import numpy as np
from scipy import stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(122)
N_DATASETS = 24
# two-sided Nemenyi critical values q(0.05) / sqrt(2) for k = 2..10 methods
# (Demšar, J. Mach. Learn. Res. 7, 1-30 (2006), Table 5a)
Q_05 = dict(zip(range(2, 11), (1.960, 2.343, 2.569, 2.728, 2.850, 2.949,
                               3.031, 3.102, 3.164)))

# ------------------------------------------------------------- DATA ----
# accuracy = dataset difficulty + method effect + noise, (dataset, method)
EFFECT = {"Proposed": 0.034, "Boosted trees": 0.024,
          "Random forest": 0.015, "Neural network": 0.008,
          "SVM (RBF)": -0.002, "Linear model": -0.016}
METHODS, K = list(EFFECT), len(EFFECT)
accuracy = (rng.uniform(0.62, 0.93, (N_DATASETS, 1))
            + np.array(list(EFFECT.values()))
            + rng.normal(0, 0.016, (N_DATASETS, K)))

# ------------------------------------------------------ RANK TESTS ----
ranks = stats.rankdata(-accuracy, axis=1)            # 1 = most accurate
mean_rank = ranks.mean(axis=0)
chi2 = (12 * N_DATASETS / (K * (K + 1)) * np.sum(mean_rank ** 2)
        - 3 * N_DATASETS * (K + 1))                  # Friedman, no ties
p_value = stats.chi2.sf(chi2, K - 1)
cd = Q_05[K] * np.sqrt(K * (K + 1) / (6 * N_DATASETS))

# Cliques: maximal runs of rank-sorted methods spanning less than one CD
order = np.argsort(mean_rank)
sorted_rank = mean_rank[order]
reach = [int(np.flatnonzero(sorted_rank - r < cd)[-1]) for r in sorted_rank]
cliques = [(i, j) for i, j in enumerate(reach)
           if j > i and (i == 0 or reach[i - 1] < j)]

# ------------------------------------------------------- SELF-CHECK ---
reference = stats.friedmanchisquare(*accuracy.T)
assert abs(mean_rank.sum() - K * (K + 1) / 2) < 1e-12, mean_rank.sum()
assert abs(chi2 - reference.statistic) < 1e-9, (chi2, reference.statistic)
assert abs(p_value - reference.pvalue) <= 1e-9 * reference.pvalue
assert all(sorted_rank[j] - sorted_rank[i] < cd for i, j in cliques)
assert order[0] == METHODS.index("Proposed"), mean_rank
print(f"fig122: self-check passed (mean ranks sum to {mean_rank.sum():.1f} "
      f"= k(k+1)/2; Friedman chi2 = {chi2:.2f} matches scipy, "
      f"P = {p_value:.2g}; CD = {cd:.3f}; {len(cliques)} cliques)")

# ------------------------------------------------------------ FIGURE --
# x is the average rank (reversed); y is millimetres below the rank axis
DEPTH, ROW0, ROW_STEP = 21.0, 9.5, 4.6
fig = ms.figure(120, 47)
ax = ms.axes(fig, 33, 11.5, 54, DEPTH)
ax.set_xlim(K + 0.25, 0.75)
ax.set_ylim(DEPTH, 0)
ax.xaxis.tick_top()
ax.set_xticks(range(1, K + 1))
ax.set_yticks([])
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(True)
ax.spines["top"].set_bounds(1, K)

# the critical difference as a ruler above the axis, at the worst-rank end
CD_Y = -7.5
ax.plot([K, K, K - cd, K - cd], [CD_Y - 0.9, CD_Y, CD_Y, CD_Y - 0.9],
        color=ms.INK, lw=0.8, clip_on=False, solid_joinstyle="miter")
ax.text(K - cd / 2, CD_Y - 1.4, f"CD = {cd:.2f}", ha="center", va="bottom",
        fontsize=ms.FS_TICK)
ax.text(1, CD_Y, "Average rank (1 = best)", ha="right", va="center")

# clique bars, one level per clique so overlapping groups stay distinct
for level, (i, j) in enumerate(cliques):
    ax.plot([sorted_rank[j] + 0.07, sorted_rank[i] - 0.07],
            [2.2 + 1.7 * level] * 2, color=ms.BLUE, lw=2.4,
            solid_capstyle="butt", zorder=3)

# elbow leaders: the worse half hangs to the left, the better half right
half = (K + 1) // 2
for position, m in enumerate(order):
    right = position < half
    row = position if right else K - 1 - position
    y = ROW0 + ROW_STEP * row
    x_end = 0.55 if right else K + 0.45
    colour = ms.VERMILLION if METHODS[m] == "Proposed" else ms.GREY_DARK
    ax.plot([mean_rank[m], mean_rank[m], x_end], [0, y, y], color=colour,
            lw=1.0 if colour == ms.VERMILLION else 0.7, clip_on=False,
            solid_joinstyle="miter", zorder=2)
    ax.plot(mean_rank[m], 0, "o", color=colour, markersize=3.2,
            markeredgewidth=0, clip_on=False, zorder=4)
    label = (f"{mean_rank[m]:.2f}  {METHODS[m]}" if right
             else f"{METHODS[m]}  {mean_rank[m]:.2f}")
    ax.text(x_end + (-0.12 if right else 0.12), y, label, color=colour,
            ha="left" if right else "right", va="center",
            fontweight="bold" if colour == ms.VERMILLION else "normal")

fig.text(0.5, 0.085, f"Friedman χ² = {chi2:.1f}, d.f. = {K - 1}, "
         f"{ms.format_p(p_value)}; n = {N_DATASETS} datasets\nBars join "
         "methods that do not differ (Nemenyi test, α = 0.05)",
         ha="center", va="center", fontsize=ms.FS_TICK, color=ms.GREY_DARK,
         linespacing=1.35)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig122_critical_difference.{ext}")
print("fig122_critical_difference: saved png + pdf")
