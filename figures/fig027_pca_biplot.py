"""Fig. 27 - PCA biplot with 95% confidence ellipses (single column).

PCA via numpy SVD (no sklearn), score scatter coloured by group,
covariance-based 95% confidence ellipses (chi-square scaled), and
variable loadings drawn as arrows - the standard ordination figure in
ecology, chemometrics and omics.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(27)


def confidence_ellipse(ax, x, y, color, q=0.95):
    """Add a q-level confidence ellipse of the (x, y) cloud to ax."""
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * np.sqrt(stats.chi2.ppf(q, 2) * vals)
    ax.add_patch(Ellipse((x.mean(), y.mean()), width, height, angle=angle,
                         facecolor=color, alpha=0.12, edgecolor=color, lw=1.0))


# ------------------------------------------------------------- DATA ----
features = ["Leaf area", "Leaf N", "SLA", "Height", "Seed mass", "Wood density"]
groups = ["Forest", "Savanna", "Wetland"]
means = np.array([[1.2, 0.8, -0.5, 1.0, 0.6, 0.3],
                  [-0.8, -0.3, 0.9, -0.9, 0.2, 0.7],
                  [0.1, -0.9, -0.4, 0.2, -1.0, -0.9]])
cov = 0.55 * np.eye(6) + 0.2
X = np.vstack([rng.multivariate_normal(m, cov, 40) for m in means])
labels = np.repeat(np.arange(3), 40)

# ---------------------------------------------------------------- PCA ----
Z = (X - X.mean(0)) / X.std(0, ddof=1)
U, S, Vt = np.linalg.svd(Z, full_matrices=False)
scores = U * S
explained = 100 * S**2 / np.sum(S**2)

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.5, 3.2))
ax.axhline(0, ls=":", lw=0.6, color="0.7", zorder=0)
ax.axvline(0, ls=":", lw=0.6, color="0.7", zorder=0)

for g, name in enumerate(groups):
    m = labels == g
    ax.scatter(scores[m, 0], scores[m, 1], s=12, color=f"C{g}", alpha=0.8,
               lw=0, label=name, zorder=3)
    confidence_ellipse(ax, scores[m, 0], scores[m, 1], f"C{g}")

# loadings as arrows, scaled to the score cloud
arrow_scale = 0.55 * np.abs(scores[:, :2]).max() / np.abs(Vt[:2]).max()
for k, name in enumerate(features):
    vx, vy = Vt[0, k] * arrow_scale, Vt[1, k] * arrow_scale
    ax.annotate("", xy=(vx, vy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.3",
                                mutation_scale=7))
    ax.text(vx * 1.12, vy * 1.12, name, fontsize=6, color="0.25",
            ha="center", va="center")

ax.margins(0.12)
ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
ax.legend(loc="upper left", fontsize=6, markerscale=1.3)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig027_pca_biplot.{ext}")
print("saved fig027_pca_biplot.png / .pdf")
