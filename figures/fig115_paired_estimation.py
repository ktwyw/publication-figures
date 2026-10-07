"""Fig. 115 - Paired measurements as an estimation plot (120 mm).

Extends fig016, which stops at the slope chart: a second panel plots
the paired differences themselves with their mean and 95% t interval,
so the effect is read off an axis instead of inferred from crossing
lines. Lines and points are coloured by direction of change, with the
counts written in, and the effect size is stated with its interval.
The self-check is that scipy.stats.ttest_rel returns exactly
mean(d) / (s.d.(d) / sqrt(n)), that the 95% interval excludes zero
exactly when P < 0.05, and that the two panels share top and bottom
edges.

Statistics: n = 20 participants measured before and after; diamonds,
means; error bar, 95% confidence interval of the mean difference
(t distribution, 19 d.f.); two-sided paired t-test. All data are
simulated.
"""

import numpy as np
from scipy import stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(1515)
N = 20
DOWN, UP = ms.BLUE, ms.VERMILLION            # direction of change

# ------------------------------------------------------------- DATA ----
before = rng.normal(148, 13, N)              # systolic pressure, mmHg
after = before + rng.normal(-8.5, 6.0, N)
diff = after - before

test = stats.ttest_rel(after, before)
mean_diff = float(diff.mean())
sem = diff.std(ddof=1) / np.sqrt(N)
half = float(stats.t.ppf(0.975, N - 1) * sem)
ci_low, ci_high = mean_diff - half, mean_diff + half
decreased = diff < 0

# ------------------------------------------------------- SELF-CHECK ---
t_by_hand = mean_diff / sem
assert abs(test.statistic - t_by_hand) < 1e-9, (test.statistic, t_by_hand)
assert (ci_low > 0 or ci_high < 0) == (test.pvalue < 0.05)
scipy_ci = test.confidence_interval(0.95)
assert max(abs(scipy_ci.low - ci_low), abs(scipy_ci.high - ci_high)) < 1e-9
print(f"fig115: self-check passed (paired t = {test.statistic:.3f} = "
      f"mean/s.e.m. by hand; mean change {mean_diff:.2f} mmHg, 95% CI "
      f"{ci_low:.2f} to {ci_high:.2f} excludes 0 and P = "
      f"{test.pvalue:.2g} < 0.05)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 62)
gs = ms.grid(fig, 1, 2, left=14, right=5, top=7, bottom=9, wspace=19,
             width_ratios=[1.35, 1])
ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

for b, a, down in zip(before, after, decreased):
    ax_a.plot([0, 1], [b, a], color=DOWN if down else UP, lw=0.6,
              alpha=0.55, marker="o", markersize=2.4, markeredgewidth=0)
ax_a.plot([0, 1], [before.mean(), after.mean()], color=ms.INK, lw=1.4,
          marker="D", markersize=4, zorder=5)
ax_a.set_xlim(-0.35, 1.35)
ax_a.set_xticks([0, 1], ["Before", "After 12 weeks"])
ax_a.set_ylim(105, 185)
ax_a.set_yticks(np.arange(110, 181, 10))
ax_a.set_ylabel("Systolic blood pressure (mmHg)")
ax_a.text(0.03, 0.035, f"Decrease: {int(decreased.sum())} of {N}",
          color=DOWN, transform=ax_a.transAxes, fontsize=ms.FS_TICK,
          va="bottom")
ax_a.text(0.97, 0.035, f"Increase: {int((~decreased).sum())} of {N}",
          color=UP, transform=ax_a.transAxes, fontsize=ms.FS_TICK,
          va="bottom", ha="right")

jitter = np.random.default_rng(15)       # horizontal jitter: display only
ax_b.plot([-0.6, 1.0], [0, 0], color=ms.GREY, lw=0.6, ls=(0, (3, 2)))
ax_b.scatter(jitter.uniform(-0.22, 0.22, N), diff, s=9, linewidths=0,
             alpha=0.8, c=np.where(decreased, DOWN, UP))
ax_b.errorbar([0.62], [mean_diff], yerr=[half], fmt="D", color=ms.INK,
              markersize=4, elinewidth=1.0, capsize=2.5, capthick=1.0)
ax_b.set_xlim(-0.6, 1.0)
ax_b.set_xticks([0, 0.62], ["Individual\ndifferences", "Mean\n(95% CI)"])
ax_b.set_ylim(-26, 10)
ax_b.set_yticks(np.arange(-25, 11, 5))
ax_b.set_ylabel("Change after − before (mmHg)")
estimate = (f"{mean_diff:.1f} mmHg (95% CI {ci_low:.1f} to {ci_high:.1f})"
            .replace("-", "−"))               # true minus signs
ax_b.text(0.5, 1.03, f"{estimate}\nPaired t-test, {ms.format_p(test.pvalue)}",
          transform=ax_b.transAxes, ha="center", va="bottom",
          fontsize=ms.FS_TICK, linespacing=1.15)
ms.panel_label(ax_a, "a")
ms.panel_label(ax_b, "b")

ms.assert_aligned([ax_a, ax_b])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig115_paired_estimation.{ext}")
print("fig115_paired_estimation: saved png + pdf")
