"""Fig. 13 - ROC curves with AUC (single column, square).

ROC and AUC are computed from scratch with numpy (no sklearn needed),
which also makes the underlying definitions explicit.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(13)


def roc_curve(y_true, score):
    """Return FPR, TPR and thresholds, sorted by decreasing threshold."""
    order = np.argsort(-score)
    y = y_true[order]
    tpr = np.concatenate([[0.0], np.cumsum(y) / y.sum()])
    fpr = np.concatenate([[0.0], np.cumsum(1 - y) / (y.size - y.sum())])
    thr = np.concatenate([[np.inf], score[order]])
    return fpr, tpr, thr


# ---------------------------------------------------- SIMULATED DATA ----
n = 600
y = (rng.uniform(size=n) < 0.4).astype(int)
models = {"Clinical score": 0.8, "+ Biomarker": 1.3, "+ Imaging": 1.9}
scores = {name: y * d + rng.normal(0, 1, n) for name, d in models.items()}

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.0, 3.0))
ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="0.6", zorder=1)

for i, (name, s) in enumerate(scores.items()):
    fpr, tpr, thr = roc_curve(y, s)
    auc = np.trapezoid(tpr, fpr)
    ax.plot(fpr, tpr, color=f"C{i}", label=f"{name} (AUC = {auc:.2f})")

# mark the Youden-optimal operating point of the best model
j = np.argmax(tpr - fpr)
ax.plot(fpr[j], tpr[j], "o", ms=5, mfc="white", mec="C2", mew=1.2, zorder=4)
ax.annotate("Youden optimum", (fpr[j], tpr[j]), xytext=(10, -4),
            textcoords="offset points", fontsize=6, color="0.3")

ax.set_xlim(-0.02, 1.0)
ax.set_ylim(0.0, 1.02)
ax.set_aspect("equal")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.legend(loc="lower right", fontsize=6.5)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig013_roc_curves.{ext}")
print("saved fig013_roc_curves.png / .pdf")
