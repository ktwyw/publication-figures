"""Fig. 34 - Training curves with seed variability (double column, two panels).

Mean ± s.d. across random seeds instead of a single lucky run, training
vs validation distinguished by line style, the best validation epoch
marked, and a proxy-artist legend that separates "which model" from
"which split".
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(34)

# ------------------------------------------------------------- DATA ----
epochs = np.arange(1, 101)
SEEDS = 5


def simulate(tau, floor, acc_max):
    tau_s = tau * rng.normal(1, 0.06, (SEEDS, 1))
    train = floor + (2.2 - floor) * np.exp(-epochs / tau_s) + rng.normal(0, 0.015, (SEEDS, 100))
    val = train + 0.08 + 0.0025 * epochs + rng.normal(0, 0.02, (SEEDS, 100))
    acc = acc_max * (1 - np.exp(-epochs / tau_s)) - 0.0004 * epochs + rng.normal(0, 0.008, (SEEDS, 100))
    return train, val, acc


models = {"Baseline": simulate(25, 0.45, 0.88),
          "Proposed": simulate(15, 0.30, 0.94)}


def band(ax, y, color, ls="-", lw=1.2):
    m, s = y.mean(0), y.std(0, ddof=1)
    ax.plot(epochs, m, color=color, ls=ls, lw=lw)
    ax.fill_between(epochs, m - s, m + s, color=color, alpha=0.2, lw=0)


# ------------------------------------------------------------- PLOT ----
fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(7.1, 2.4))

for i, (name, (train, val, acc)) in enumerate(models.items()):
    band(ax_l, train, f"C{i}", ls="--", lw=0.9)
    band(ax_l, val, f"C{i}")
    band(ax_a, acc, f"C{i}")

best = models["Proposed"][2].mean(0).argmax()
best_acc = models["Proposed"][2].mean(0)[best]
ax_a.plot(epochs[best], best_acc, "o", ms=5, mfc="white", mec="C1", mew=1.2, zorder=4)
ax_a.annotate(f"best: epoch {epochs[best]}", (epochs[best], best_acc),
              xytext=(0, -14), textcoords="offset points", ha="center", fontsize=6)

ax_l.set_xlabel("Epoch")
ax_l.set_ylabel("Cross-entropy loss")
ax_a.set_xlabel("Epoch")
ax_a.set_ylabel("Validation accuracy")
ax_a.set_ylim(0.4, 1.0)

handles = [Line2D([], [], color=f"C{i}", lw=1.2, label=n) for i, n in enumerate(models)]
handles += [Line2D([], [], color="0.3", lw=1.2, label="Validation"),
            Line2D([], [], color="0.3", lw=0.9, ls="--", label="Training")]
ax_l.legend(handles=handles, loc="upper right", fontsize=6, ncol=2)

for ax, letter in zip((ax_l, ax_a), "ab"):
    ax.text(-0.18, 1.02, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="right")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig034_training_curves.{ext}")
print("saved fig034_training_curves.png / .pdf")
