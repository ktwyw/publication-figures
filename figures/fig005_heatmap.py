"""Fig. 5 - Annotated correlation heatmap (single column).

Uses a diverging colormap pinned to [-1, 1] so colour is comparable
across studies, with per-cell annotations.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

# --------------------------------------------- DATA (two latent factors) ----
rng = np.random.default_rng(2)
n = 240
size_f = rng.normal(size=n)   # latent "body size" factor
fit_f = rng.normal(size=n)    # latent "fitness" factor
age = rng.normal(size=n)

height = size_f + 0.40 * rng.normal(size=n)
mass = 0.80 * size_f + 0.50 * rng.normal(size=n)
bmi = mass - 0.55 * height + 0.30 * rng.normal(size=n)
vo2 = fit_f - 0.35 * age + 0.45 * rng.normal(size=n)
rhr = -0.75 * fit_f + 0.50 * rng.normal(size=n)
sbp = 0.50 * age + 0.30 * bmi + 0.60 * rng.normal(size=n)
glu = 0.40 * age + 0.30 * bmi - 0.25 * fit_f + 0.60 * rng.normal(size=n)

X = np.column_stack([height, mass, bmi, age, vo2, rhr, sbp, glu])
names = ["Height", "Body mass", "BMI", "Age", "VO$_2$max",
         "Resting HR", "Systolic BP", "Glucose"]
C = np.corrcoef(X, rowvar=False)

# ------------------------------------------------------------------ PLOT ----
fig, ax = plt.subplots(figsize=(3.7, 3.3))
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)

k = len(names)
ax.set_xticks(range(k), names, rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(k), names)

for i in range(k):
    for j in range(k):
        val = C[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.5,
                color="white" if abs(val) > 0.45 else "black")

for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label("Pearson $r$")
cbar.outline.set_linewidth(0.8)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig005_heatmap.{ext}")
print("saved fig005_heatmap.png / .pdf")
