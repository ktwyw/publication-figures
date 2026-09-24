"""Fig. 18 - Bland-Altman method-comparison plot (single column).

The standard way to report agreement between two measurement methods:
bias line, 95% limits of agreement, shaded confidence bands for all
three, and right-edge value labels via ax.get_yaxis_transform().
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(18)

# ------------------------------------------------------------- DATA ----
n = 80
true = rng.uniform(60, 180, n)                 # e.g. glucose, mg/dL
method_a = true + rng.normal(0, 5.5, n)
method_b = true + 4.2 + rng.normal(0, 5.5, n)  # small systematic bias

mean_ab = (method_a + method_b) / 2
diff = method_b - method_a

bias = diff.mean()
sd = diff.std(ddof=1)
upper, lower = bias + 1.96 * sd, bias - 1.96 * sd

# 95% CIs of the bias and of each limit of agreement
t_crit = stats.t.ppf(0.975, n - 1)
ci_bias = t_crit * sd / np.sqrt(n)
ci_loa = t_crit * sd * np.sqrt(3.0 / n)

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.5, 2.7))

for center, half in [(bias, ci_bias), (upper, ci_loa), (lower, ci_loa)]:
    ax.axhspan(center - half, center + half, color="0.85", alpha=0.6,
               lw=0, zorder=0)
ax.axhline(0, ls=":", lw=0.7, color="0.6", zorder=1)
ax.axhline(bias, color="C1", lw=1.2, zorder=2)
ax.axhline(upper, color="0.2", ls="--", lw=0.9, zorder=2)
ax.axhline(lower, color="0.2", ls="--", lw=0.9, zorder=2)
ax.scatter(mean_ab, diff, s=12, color="C0", alpha=0.75, lw=0, zorder=3)

tform = ax.get_yaxis_transform()   # x in axes fraction, y in data
ax.text(1.02, bias, f"Bias = {bias:.1f}", transform=tform,
        fontsize=6, va="center", color="C1")
ax.text(1.02, upper, f"+1.96 s.d. = {upper:.1f}", transform=tform,
        fontsize=6, va="center")
ax.text(1.02, lower, f"\u22121.96 s.d. = {lower:.1f}", transform=tform,
        fontsize=6, va="center")

ax.set_xlabel("Mean of methods (mg dL$^{-1}$)")
ax.set_ylabel("Difference, B \u2212 A (mg dL$^{-1}$)")
ax.set_ylim(lower - 3.5 * ci_loa, upper + 3.5 * ci_loa)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig018_bland_altman.{ext}")
print("saved fig018_bland_altman.png / .pdf")
