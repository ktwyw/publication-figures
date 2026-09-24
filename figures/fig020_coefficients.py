"""Fig. 20 - Regression coefficient plot, two models compared (single column).

Dot-and-whisker plots replace regression tables in talks and increasingly
in papers. Shown: paired series with small vertical offsets, sorting by
effect size, and a zero reference line. Replace the hard-coded estimates
with the output of your own model fit.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

# ------------------------------------- MODEL OUTPUT (replace with yours) ----
labels = np.array(["Age (per 10 yr)", "Male sex", "BMI (per 5 kg m$^{-2}$)",
                   "Current smoker", "Diabetes", "Statin use",
                   "Exercise (h wk$^{-1}$)"])
b_unadj = np.array([0.42, 0.18, 0.35, 0.51, 0.62, -0.05, -0.28])
se_unadj = np.array([0.07, 0.09, 0.08, 0.11, 0.12, 0.10, 0.07])
b_adj = np.array([0.31, 0.10, 0.22, 0.38, 0.45, -0.12, -0.21])
se_adj = np.array([0.08, 0.09, 0.08, 0.11, 0.13, 0.10, 0.07])

order = np.argsort(b_adj)                      # smallest at bottom
labels, b_unadj, se_unadj, b_adj, se_adj = (
    labels[order], b_unadj[order], se_unadj[order],
    b_adj[order], se_adj[order])

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.4, 2.8))
y = np.arange(len(labels))

ax.axvline(0, ls="--", lw=0.8, color="0.5", zorder=1)
ax.errorbar(b_unadj, y + 0.17, xerr=1.96 * se_unadj, fmt="o", ms=4,
            color="0.55", lw=1.1, capsize=0, zorder=2, label="Unadjusted")
ax.errorbar(b_adj, y - 0.17, xerr=1.96 * se_adj, fmt="o", ms=4,
            color="C1", lw=1.1, capsize=0, zorder=3,
            label="Adjusted (full model)")

ax.set_yticks(y, labels)
ax.tick_params(left=False)
ax.spines["left"].set_visible(False)
ax.set_ylim(-0.6, len(labels) - 0.2)
ax.set_xlabel(r"Standardized $\beta$ (95% CI)")
ax.legend(loc="lower right", fontsize=6)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig020_coefficients.{ext}")
print("saved fig020_coefficients.png / .pdf")
