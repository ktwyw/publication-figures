"""Fig. 39 - Empirical CDF comparison with KS tests (single column).

ECDFs show whole distributions without binning choices. Uses the modern
Axes.ecdf (matplotlib >= 3.8), reports two-sample KS tests against the
control, and draws the KS statistic as the maximum vertical gap.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(39)

# ------------------------------------------------------------- DATA ----
samples = {"Control": rng.lognormal(2.30, 0.25, 180),
           "Treatment A": rng.lognormal(2.38, 0.25, 160),
           "Treatment B": rng.lognormal(2.55, 0.30, 170)}
ctrl = samples["Control"]

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.4, 2.7))
for i, (name, d) in enumerate(samples.items()):
    ax.ecdf(d, color=f"C{i}", lw=1.2, label=name)

lines = []
for name in ("Treatment A", "Treatment B"):
    ks = stats.ks_2samp(ctrl, samples[name])
    p_txt = "$P$ < 0.001" if ks.pvalue < 1e-3 else f"$P$ = {ks.pvalue:.3f}"
    lines.append(f"{name.replace('Treatment ', '')} vs control: $D$ = {ks.statistic:.2f}, {p_txt}")
ax.text(0.97, 0.30, "\n".join(lines), transform=ax.transAxes, va="bottom",
        ha="right", fontsize=6)

# mark the KS statistic (maximum gap) for Treatment B
trt = np.sort(samples["Treatment B"])
grid = np.sort(np.concatenate([ctrl, trt]))
f_ctrl = np.searchsorted(np.sort(ctrl), grid, side="right") / ctrl.size
f_trt = np.searchsorted(trt, grid, side="right") / trt.size
k = np.argmax(np.abs(f_ctrl - f_trt))
ax.plot([grid[k], grid[k]], [f_ctrl[k], f_trt[k]], color="0.2", lw=1.3)
ax.annotate("$D$", (grid[k], (f_ctrl[k] + f_trt[k]) / 2), xytext=(5, 0),
            textcoords="offset points", va="center", fontsize=7)

ax.set_xlabel("Cell diameter (µm)")
ax.set_ylabel("Cumulative probability")
ax.set_ylim(0, 1.02)
ax.legend(loc="lower right", fontsize=6)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig039_ecdf_comparison.{ext}")
print("saved fig039_ecdf_comparison.png / .pdf")
