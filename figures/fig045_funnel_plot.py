"""Fig. 45 - Contour-enhanced funnel plot for meta-analysis (single column).

Effect size vs standard error (inverted, so precise studies sit at the
top), significance contours around zero (grey shades), the pseudo-95%
confidence funnel around the pooled estimate, and Egger's regression
test for small-study effects.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(45)

# ------------------------------------------------------------- DATA ----
n = 28
se = rng.uniform(0.04, 0.45, n)
effect = 0.25 + rng.normal(0, se) + 0.6 * se        # small-study asymmetry

w = 1 / se**2
pooled = np.sum(w * effect) / np.sum(w)

egger = stats.linregress(1 / se, effect / se)       # intercept tests asymmetry
t_egger = egger.intercept / egger.intercept_stderr
p_egger = 2 * stats.t.sf(abs(t_egger), n - 2)

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.4, 3.0))
se_grid = np.linspace(0, se.max() * 1.08, 200)
lim = max(np.abs(effect).max(), 2.576 * se.max()) * 1.1

shades = [(1.645, 1.960, "0.94", "0.05 < $P$ < 0.10"),
          (1.960, 2.576, "0.86", "0.01 < $P$ < 0.05"),
          (2.576, np.inf, "0.76", "$P$ < 0.01")]
for z0, z1, color, _ in shades:
    hi = np.minimum(z1 * se_grid, lim) if np.isfinite(z1) else np.full_like(se_grid, lim)
    for sign in (1, -1):
        ax.fill_betweenx(se_grid, sign * z0 * se_grid, sign * hi,
                         color=color, lw=0, zorder=0)

ax.plot(pooled + 1.96 * se_grid, se_grid, ls=":", lw=0.8, color="C1")
ax.plot(pooled - 1.96 * se_grid, se_grid, ls=":", lw=0.8, color="C1")
ax.axvline(pooled, color="C1", lw=1.0)
ax.axvline(0, color="0.4", lw=0.6)
ax.scatter(effect, se, s=18, facecolor="white", edgecolor="C0", lw=1.0, zorder=3)

ax.set_xlim(-lim, lim)
ax.set_ylim(se.max() * 1.08, 0)                     # inverted: precise on top
ax.set_xlabel("Log odds ratio")
ax.set_ylabel("Standard error")
ax.text(0.03, 0.03, f"Pooled = {pooled:.2f}\nEgger $P$ = {p_egger:.3f}",
        transform=ax.transAxes, va="bottom", fontsize=6.5)
ax.legend(handles=[Patch(color=c, label=l) for _, _, c, l in shades],
          loc="upper left", fontsize=6, title="Significance", title_fontsize=6.5,
          frameon=True, framealpha=0.9, edgecolor="none")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig045_funnel_plot.{ext}")
print("saved fig045_funnel_plot.png / .pdf")
