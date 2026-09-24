"""Fig. 25 - Calibration (reliability) curve for a probabilistic model.

Decile bins of predicted probability vs observed event frequency with
Wilson 95% intervals, the perfect-calibration diagonal, Brier score and
expected calibration error (ECE), plus a histogram of predictions below
so the reader can see where the model actually operates.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(25)


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


# ---------------------------------------------------- SIMULATED MODEL ----
n = 2500
p_pred = rng.beta(2, 2, n)
logit = np.log(p_pred / (1 - p_pred))
p_true = 1 / (1 + np.exp(-(1.35 * logit - 0.15)))   # overconfident model
y = (rng.uniform(size=n) < p_true).astype(int)

edges = np.quantile(p_pred, np.linspace(0, 1, 11))   # decile bins
bin_id = np.clip(np.digitize(p_pred, edges[1:-1]), 0, 9)
mean_pred, obs, lo, hi, counts = [], [], [], [], []
for b in range(10):
    m = bin_id == b
    k, nb = y[m].sum(), m.sum()
    mean_pred.append(p_pred[m].mean())
    obs.append(k / nb)
    l, h = wilson(k, nb)
    lo.append(l)
    hi.append(h)
    counts.append(nb)
mean_pred, obs, lo, hi, counts = map(np.array, (mean_pred, obs, lo, hi, counts))

brier = np.mean((p_pred - y) ** 2)
ece = np.sum(counts / n * np.abs(obs - mean_pred))

# ------------------------------------------------------------- PLOT ----
fig, (ax, ax_h) = plt.subplots(2, 1, figsize=(3.0, 3.6), sharex=True,
                               height_ratios=[3.2, 1])

ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="0.6", label="Perfect calibration")
ax.errorbar(mean_pred, obs, yerr=[obs - lo, hi - obs], fmt="o-", ms=4,
            color="C0", lw=1.0, capsize=2, label="Model (deciles, 95% CI)")
ax.set_ylim(0, 1)
ax.set_ylabel("Observed frequency")
ax.text(0.03, 0.97, f"Brier = {brier:.3f}\nECE = {ece:.3f}",
        transform=ax.transAxes, va="top", fontsize=6.5)
ax.legend(loc="lower right", fontsize=6)

ax_h.hist(p_pred, bins=25, color="0.6", lw=0)
ax_h.set_xlim(0, 1)
ax_h.set_xlabel("Predicted probability")
ax_h.set_ylabel("Count")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig025_calibration.{ext}")
print("saved fig025_calibration.png / .pdf")
