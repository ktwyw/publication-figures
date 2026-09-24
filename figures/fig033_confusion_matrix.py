"""Fig. 33 - Confusion matrix, row-normalised with counts (single column).

Colour encodes the fraction of each true class (recall on the diagonal),
while every cell also shows the raw count so nothing is hidden by
class imbalance. Text switches to white on dark cells.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

# ------------------------------------------------------------- DATA ----
classes = ["Water", "Forest", "Urban", "Cropland", "Bare soil"]
cm = np.array([[188, 2, 3, 1, 6],
               [1, 241, 4, 12, 2],
               [4, 3, 176, 6, 11],
               [0, 15, 5, 203, 7],
               [5, 1, 14, 3, 150]])
row_norm = cm / cm.sum(axis=1, keepdims=True)
accuracy = np.trace(cm) / cm.sum()

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.4, 3.1))
im = ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=1)

k = len(classes)
for i in range(k):
    for j in range(k):
        ax.text(j, i, f"{cm[i, j]}\n{100 * row_norm[i, j]:.0f}%",
                ha="center", va="center", fontsize=6,
                color="white" if row_norm[i, j] > 0.5 else "black")

ax.set_xticks(range(k), classes, rotation=30, ha="right", rotation_mode="anchor")
ax.set_yticks(range(k), classes)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_title(f"Overall accuracy = {100 * accuracy:.1f}%", fontsize=8, loc="left")
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Fraction of true class")
cbar.outline.set_linewidth(0.8)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig033_confusion_matrix.{ext}")
print("saved fig033_confusion_matrix.png / .pdf")
