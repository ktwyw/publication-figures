"""Fig. 4 - Bars with raw points, s.e.m. and significance brackets (single column).

Star thresholds: * P<0.05, ** P<0.01, *** P<0.001. Adjust to your
journal's convention, and prefer reporting exact P values in the caption.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

# ------------------------------------------------------------- DATA ----
rng = np.random.default_rng(5)
data = {
    "Control": rng.normal(1.00, 0.13, 12),
    "Drug A": rng.normal(1.28, 0.16, 12),
    "Drug B": rng.normal(1.72, 0.18, 12),
}
labels = list(data)
values = list(data.values())
means = [v.mean() for v in values]
sems = [v.std(ddof=1) / np.sqrt(v.size) for v in values]


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def bracket(ax, x1, x2, y, text, h=0.05):
    """Draw a significance bracket from x1 to x2 at height y."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=0.8, color="black", clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=7)


# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.0, 2.8))
xpos = np.arange(len(labels))
ax.bar(xpos, means, yerr=sems, width=0.6,
       color=[f"C{i}" for i in range(len(labels))],
       alpha=0.85, lw=0, capsize=2, error_kw=dict(lw=0.9))
for i, v in enumerate(values):
    ax.scatter(i + rng.uniform(-0.13, 0.13, v.size), v, s=9,
               color="black", alpha=0.55, lw=0, zorder=3)

ax.set_xticks(xpos, labels)
ax.set_ylabel("Relative expression (fold change)")

p_a = stats.ttest_ind(data["Control"], data["Drug A"]).pvalue
p_b = stats.ttest_ind(data["Control"], data["Drug B"]).pvalue
top = max(v.max() for v in values)
bracket(ax, 0, 1, top + 0.12, stars(p_a))
bracket(ax, 0, 2, top + 0.34, stars(p_b))
ax.set_ylim(0, top + 0.62)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig004_bars_stats.{ext}")
print("saved fig004_bars_stats.png / .pdf")
