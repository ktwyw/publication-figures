"""Fig. 12 - Forest plot for meta-analysis (odds ratios, fixed effect).

Techniques shown: log-scaled effect axis with plain-number ticks,
square markers sized by study weight, a pooled-effect diamond, and
text columns aligned outside the axes via ax.get_yaxis_transform().
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(15)

# ------------------------------------------------- SIMULATED STUDIES ----
names = ["Alonso 2016", "Chen 2018", "Dubois 2019", "Kim 2020",
         "Müller 2021", "Okafor 2022", "Rossi 2023", "Sato 2024"]
n_pat = np.array([84, 150, 96, 310, 205, 122, 415, 260])
se = np.sqrt(4.0 / n_pat)                       # approx SE of log-OR
log_or = rng.normal(np.log(0.70), np.hypot(se, 0.08))
or_, lo, hi = np.exp(log_or), np.exp(log_or - 1.96 * se), np.exp(log_or + 1.96 * se)

# fixed-effect pooling (inverse-variance)
w = 1.0 / se**2
pooled = np.sum(w * log_or) / np.sum(w)
se_p = np.sqrt(1.0 / np.sum(w))
p_or, p_lo, p_hi = np.exp(pooled), np.exp(pooled - 1.96 * se_p), np.exp(pooled + 1.96 * se_p)
Q = np.sum(w * (log_or - pooled) ** 2)
i2 = max(0.0, (Q - (len(names) - 1)) / Q) * 100

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(6.4, 3.2), layout="none")
fig.subplots_adjust(left=0.20, right=0.60, top=0.96, bottom=0.34)
k = len(names)
ys = np.arange(k)[::-1] + 2.0                   # studies top to bottom
y_pool = 0.7

ax.axvline(1.0, ls="--", lw=0.8, color="0.45", zorder=1)
ax.errorbar(or_, ys, xerr=[or_ - lo, hi - or_], fmt="none",
            ecolor="black", elinewidth=1.0, capsize=0, zorder=2)
ax.scatter(or_, ys, s=12 + 60 * w / w.max(), marker="s",
           color="C0", zorder=3)
ax.add_patch(Polygon([(p_lo, y_pool), (p_or, y_pool + 0.32),
                      (p_hi, y_pool), (p_or, y_pool - 0.32)],
                     facecolor="C1", edgecolor="black", lw=0.6, zorder=3))
ax.axhline(1.4, color="0.8", lw=0.7)

# aligned text columns to the right of the axes (x in axes fraction,
# y in data coordinates)
tform = ax.get_yaxis_transform()
ax.text(1.05, k + 2.1, "OR (95% CI)", transform=tform, fontsize=6.5)
ax.text(1.78, k + 2.1, "Weight", transform=tform, fontsize=6.5, ha="right")
for y, o, l, h, wi in zip(ys, or_, lo, hi, w):
    ax.text(1.05, y, f"{o:.2f} ({l:.2f}\u2013{h:.2f})", transform=tform,
            fontsize=6.5, va="center")
    ax.text(1.78, y, f"{100 * wi / w.sum():.1f}%", transform=tform,
            fontsize=6.5, va="center", ha="right")
ax.text(1.05, y_pool, f"{p_or:.2f} ({p_lo:.2f}\u2013{p_hi:.2f})",
        transform=tform, fontsize=6.5, va="center", fontweight="bold")

ax.set_yticks(list(ys) + [y_pool], names + ["Pooled (fixed effect)"])
ax.get_yticklabels()[-1].set_fontweight("bold")
ax.tick_params(left=False)
ax.spines["left"].set_visible(False)

ax.set_xscale("log")
ax.set_xlim(min(lo) * 0.8, max(hi) * 1.25)
ax.set_xticks([0.25, 0.5, 1, 2], ["0.25", "0.5", "1", "2"])
ax.minorticks_off()
ax.set_xlabel("Odds ratio")
ax.set_ylim(-0.4, k + 2.7)

xt = ax.get_xaxis_transform()   # x in data coords, y in axes fraction
ax.text(0.93, -0.32, "\u2190 Favours treatment", transform=xt,
        fontsize=5.5, color="0.35", ha="right")
ax.text(1.07, -0.32, "Favours control \u2192", transform=xt,
        fontsize=5.5, color="0.35", ha="left")
ax.text(0.0, -0.46, rf"Heterogeneity: $Q$ = {Q:.1f}, $I^2$ = {i2:.0f}%",
        transform=ax.transAxes, fontsize=6, color="0.3", ha="left")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig012_forest_plot.{ext}")
print("saved fig012_forest_plot.png / .pdf")
