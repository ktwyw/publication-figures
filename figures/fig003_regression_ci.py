"""Fig. 3 - Scatter + linear regression with 95% confidence band (single column).

The confidence band is computed analytically (t-distribution), so the
figure needs only numpy/scipy/matplotlib - no seaborn dependency.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

# ------------------------------------------------------------- DATA ----
rng = np.random.default_rng(11)
x = rng.uniform(0, 10, 60)
y = 1.8 * x + 3.0 + rng.normal(0, 3.0, x.size)

# -------------------------------------------------------- REGRESSION ----
res = stats.linregress(x, y)
n = x.size
x_fit = np.linspace(0, 10, 200)
y_fit = res.intercept + res.slope * x_fit

# 95% confidence band of the regression line
resid = y - (res.intercept + res.slope * x)
s_err = np.sqrt(np.sum(resid**2) / (n - 2))
mean_x = x.mean()
sxx = np.sum((x - mean_x) ** 2)
t_crit = stats.t.ppf(0.975, n - 2)
ci = t_crit * s_err * np.sqrt(1.0 / n + (x_fit - mean_x) ** 2 / sxx)


def sci(p):
    """Format a p-value as LaTeX scientific notation, e.g. 3.1 x 10^-12."""
    mant, exp = f"{p:.1e}".split("e")
    return rf"{float(mant):g} \times 10^{{{int(exp)}}}"


# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.5, 2.8))
ax.scatter(x, y, s=14, color="C0", alpha=0.8, lw=0, zorder=3, label="Samples")
ax.plot(x_fit, y_fit, color="C1", zorder=2, label="Linear fit")
ax.fill_between(x_fit, y_fit - ci, y_fit + ci, color="C1", alpha=0.22, lw=0,
                label="95% CI")

ax.set_xlabel("Dose (mg kg$^{-1}$)")
ax.set_ylabel("Plasma response (a.u.)")

stats_text = (rf"$R^2$ = {res.rvalue**2:.2f}"
              "\n"
              rf"$P = {sci(res.pvalue)}$"
              "\n"
              rf"$n$ = {n}")
ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, va="top", ha="left")
ax.legend(loc="lower right")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig003_regression_ci.{ext}")
print("saved fig003_regression_ci.png / .pdf")
