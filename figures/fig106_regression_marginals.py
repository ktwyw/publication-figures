"""Fig. 106 - Regression band with marginal densities (single column, 89 mm).

Combines fig003 (OLS line with an analytic 95% band) and fig014 (joint
plot with marginals) in one exact-size figure: the band for the mean
response is drawn under the points, the marginal KDEs sit on shared axes
cut from a millimetre GridSpec, and slope ± s.e., r, P and n are reported
in the panel. The self-check is the set of OLS identities: residuals sum
to zero and are orthogonal to x, r² = 1 − SS_res/SS_tot, and the band is
narrowest at the mean of x.

Statistics: n = 140 genes; line, ordinary least-squares fit; band, 95%
confidence interval for the mean response; Pearson r with two-sided
P value. All data are simulated.
"""

import numpy as np
from scipy import stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(606)
X_LIM, Y_LIM = (1.0, 9.0), (1.0, 10.0)

# ------------------------------------------------------------- DATA ----
n = 140
x = rng.normal(5.0, 1.3, n)                       # log2 mRNA abundance
y = 1.1 + 0.78 * x + rng.normal(0, 0.72, n)       # log2 protein abundance

# -------------------------------------------------------- REGRESSION ---
fit = stats.linregress(x, y)
residual = y - (fit.intercept + fit.slope * x)
grid = np.linspace(*X_LIM, 200)
predicted = fit.intercept + fit.slope * grid
residual_sd = np.sqrt(np.sum(residual ** 2) / (n - 2))
sxx = np.sum((x - x.mean()) ** 2)
half_width = stats.t.ppf(0.975, n - 2) * residual_sd * np.sqrt(
    1 / n + (grid - x.mean()) ** 2 / sxx)

# ------------------------------------------------------- SELF-CHECK ---
assert abs(residual.sum()) < 1e-9 and abs(residual @ x) < 1e-9
r2_from_ss = 1 - np.sum(residual ** 2) / np.sum((y - y.mean()) ** 2)
assert abs(fit.rvalue ** 2 - r2_from_ss) < 1e-9, (fit.rvalue ** 2, r2_from_ss)
x_narrowest = grid[np.argmin(half_width)]
assert abs(x_narrowest - x.mean()) <= grid[1] - grid[0]
assert abs(fit.stderr - residual_sd / np.sqrt(sxx)) < 1e-9
print(f"fig106: self-check passed (residual sum {residual.sum():.1e}, "
      f"r² = {fit.rvalue ** 2:.4f} = 1 − SS_res/SS_tot, band narrowest "
      f"at x = {x_narrowest:.2f} vs mean {x.mean():.2f})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 84)
gs = ms.grid(fig, 2, 2, left=13, right=4, top=4, bottom=11, wspace=1.2,
             hspace=1.2, width_ratios=[5.6, 1], height_ratios=[1, 5.6])
ax = fig.add_subplot(gs[1, 0])
ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

ax.fill_between(grid, predicted - half_width, predicted + half_width,
                color=ms.BLUE, alpha=0.18, lw=0, zorder=1)
ax.scatter(x, y, s=8, color=ms.SKY, alpha=0.75, linewidths=0, zorder=2)
ax.plot(grid, predicted, color=ms.BLUE, lw=1.2, zorder=3)
ax.set_xlim(*X_LIM)
ax.set_ylim(*Y_LIM)
ax.set_xticks(np.arange(2, 9, 2))
ax.set_yticks(np.arange(2, 11, 2))
# FS_MATH keeps the mathtext subscript (0.7 of the size) at 5 pt
ax.set_xlabel("mRNA abundance (log$_{2}$ TPM)", fontsize=ms.FS_MATH)
ax.set_ylabel("Protein abundance (log$_{2}$ iBAQ)", fontsize=ms.FS_MATH)
ax.text(0.04, 0.97, f"Pearson r = {fit.rvalue:.2f}\n"
        f"{ms.format_p(fit.pvalue)}\n"
        f"Slope = {fit.slope:.2f} ± {fit.stderr:.2f} (s.e.)\n"
        f"n = {n} genes", transform=ax.transAxes, va="top", ha="left",
        fontsize=ms.FS_TICK)

# marginal densities on the shared axes
support_x, support_y = np.linspace(*X_LIM, 200), np.linspace(*Y_LIM, 200)
density_x = stats.gaussian_kde(x)(support_x)
density_y = stats.gaussian_kde(y)(support_y)
ax_top.fill_between(support_x, 0, density_x, color=ms.BLUE, alpha=0.35, lw=0)
ax_top.set_ylim(0, density_x.max() * 1.05)
ax_right.fill_betweenx(support_y, 0, density_y, color=ms.BLUE, alpha=0.35,
                       lw=0)
ax_right.set_xlim(0, density_y.max() * 1.05)
ax_top.set_axis_off()
ax_right.set_axis_off()

ms.assert_aligned([ax, ax_top], edges=("left", "right"))
ms.assert_aligned([ax, ax_right])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig106_regression_marginals.{ext}")
print("fig106_regression_marginals: saved png + pdf")
