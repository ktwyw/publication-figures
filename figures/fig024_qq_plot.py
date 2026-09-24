"""Fig. 24 - Normal Q-Q plot with confidence envelope (single column).

The standard residual-normality diagnostic, done properly: a reference
line through the quartiles (as R's qqline), a pointwise 95% envelope
from the Beta distribution of order statistics, and a Shapiro-Wilk test.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(24)

# ------------------------------------------------------------- DATA ----
resid = rng.gamma(4.0, 0.55, 150) - 2.2          # mildly right-skewed
n = resid.size
x_sorted = np.sort(resid)
i = np.arange(1, n + 1)
q_theo = stats.norm.ppf((i - 0.5) / n)

# reference line through the first and third quartiles
q1, q3 = np.percentile(resid, [25, 75])
t1, t3 = stats.norm.ppf([0.25, 0.75])
slope = (q3 - q1) / (t3 - t1)
intercept = q1 - slope * t1

# pointwise 95% envelope: the i-th order statistic's CDF value ~ Beta(i, n-i+1)
lo = intercept + slope * stats.norm.ppf(stats.beta.ppf(0.025, i, n - i + 1))
hi = intercept + slope * stats.norm.ppf(stats.beta.ppf(0.975, i, n - i + 1))

w_stat, p_sw = stats.shapiro(resid)

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.0, 3.0))
ax.fill_between(q_theo, lo, hi, color="0.87", lw=0, label="95% envelope")
ax.plot(q_theo, intercept + slope * q_theo, color="C1", lw=1.1,
        label="Reference line")
ax.plot(q_theo, x_sorted, "o", ms=3.5, mfc="white", color="C0",
        label="Residuals")

ax.set_xlabel("Theoretical quantiles")
ax.set_ylabel("Sample quantiles (residuals)")
p_txt = "$P$ < 0.001" if p_sw < 1e-3 else f"$P$ = {p_sw:.3f}"
ax.text(0.97, 0.04, f"Shapiro\u2013Wilk $W$ = {w_stat:.3f}\n{p_txt}, $n$ = {n}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)
ax.legend(loc="upper left", fontsize=6)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig024_qq_plot.{ext}")
print("saved fig024_qq_plot.png / .pdf")
