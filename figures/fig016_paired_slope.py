"""Fig. 16 - Paired data slope chart (single column).

Every subject's before/after pair is shown as a thin connecting line -
far more informative than two bars - with group means ± 95% CI overlaid
and a paired t-test with the mean difference and its CI.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(17)

# ------------------------------------------------------------- DATA ----
n = 22
base = rng.normal(148, 9, n)
follow = base - rng.normal(9, 6, n)

diff = follow - base
t_res = stats.ttest_rel(follow, base)
ci_half = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)


def mean_ci(v):
    h = stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
    return v.mean(), h


# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(2.8, 2.9))

for b, f in zip(base, follow):
    ax.plot([0, 1], [b, f], color="0.65", lw=0.7, alpha=0.6, zorder=1)
ax.scatter(np.zeros(n), base, s=9, color="0.45", lw=0, zorder=2)
ax.scatter(np.ones(n), follow, s=9, color="0.45", lw=0, zorder=2)

for xpos, vals in [(-0.14, base), (1.14, follow)]:
    m, h = mean_ci(vals)
    ax.errorbar(xpos, m, yerr=h, fmt="o", ms=5, color="C1",
                capsize=3, lw=1.4, zorder=4)

top = max(base.max(), follow.max())
ax.plot([0, 0, 1, 1], [top + 3, top + 4.5, top + 4.5, top + 3],
        lw=0.8, color="black", clip_on=False)
p_txt = "$P$ < 0.001" if t_res.pvalue < 1e-3 else f"$P$ = {t_res.pvalue:.3f}"
ax.text(0.5, top + 4.8, p_txt, ha="center", va="bottom", fontsize=7)

ax.text(0.03, 0.04,
        rf"$\Delta$ = {diff.mean():.1f} mmHg"
        "\n"
        rf"95% CI [{diff.mean() - ci_half:.1f}, {diff.mean() + ci_half:.1f}]",
        transform=ax.transAxes, fontsize=6.5, va="bottom")

ax.set_xticks([0, 1], ["Baseline", "Week 12"])
ax.set_xlim(-0.4, 1.4)
ax.set_ylim(min(base.min(), follow.min()) - 6, top + 11)
ax.set_ylabel("Systolic blood pressure (mmHg)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig016_paired_slope.{ext}")
print("saved fig016_paired_slope.png / .pdf")
