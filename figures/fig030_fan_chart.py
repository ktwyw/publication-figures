"""Fig. 30 - Fan chart: forecast with nested uncertainty bands (single column).

Observed series, then an ensemble forecast summarised as nested 50/80/90%
quantile bands with a median line - the standard way to show projection
uncertainty in climate, epidemiology, economics and hydrology.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(30)

# ------------------------------------------------------------- DATA ----
t_obs = np.arange(2015, 2025, 1 / 12)
obs = 3.4 * (t_obs - 2015) + 2.5 * np.sin(2 * np.pi * t_obs) + rng.normal(0, 2.0, t_obs.size)

t_fc = np.arange(2025, 2031 + 1e-9, 1 / 12)
n_paths = 500
drift = rng.normal(3.8, 0.7, n_paths)[:, None] / 12          # mm per month
shocks = rng.normal(0, 1.4, (n_paths, t_fc.size))
paths = obs[-1] + np.cumsum(drift + shocks, axis=1) \
    + 2.5 * np.sin(2 * np.pi * t_fc)

bands = [(5, 95, 0.15, "90%"), (10, 90, 0.25, "80%"), (25, 75, 0.40, "50%")]
q = {p: np.percentile(paths, p, axis=0) for p in (5, 10, 25, 50, 75, 90, 95)}

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for lo, hi, alpha, _ in bands:
    ax.fill_between(t_fc, q[lo], q[hi], color="C0", alpha=alpha, lw=0)
ax.plot(t_fc, q[50], color="C0", lw=1.2)
ax.plot(t_obs, obs, color="black", lw=0.8)
ax.axvline(t_fc[0], ls="--", lw=0.7, color="0.5")
ax.text(t_fc[0] + 0.15, 0.97, "Forecast", transform=ax.get_xaxis_transform(),
        va="top", fontsize=6.5, color="0.35")

ax.set_xlim(2015, 2031)
ax.set_xlabel("Year")
ax.set_ylabel("Sea-level anomaly (mm)")

handles = [Line2D([], [], color="black", lw=0.8, label="Observed"),
           Line2D([], [], color="C0", lw=1.2, label="Median forecast")]
handles += [Patch(color="C0", alpha=a, label=f"{lab} interval")
            for _, _, a, lab in bands]
ax.legend(handles=handles, loc="upper left", fontsize=6)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig030_fan_chart.{ext}")
print("saved fig030_fan_chart.png / .pdf")
