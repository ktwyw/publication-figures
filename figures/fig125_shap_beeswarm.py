"""Fig. 125 - SHAP summary beeswarm for a linear model (1.5 column, 120 mm).

The summary plot that explains a model feature by feature: one dot per
sample and feature, placed at its Shapley value and coloured by the
feature's own value, with features sorted by mean |φ|. For a linear
model f(x) = b + Σ wⱼxⱼ the Shapley value is exact, φᵢⱼ = wⱼ(xᵢⱼ −
mean xⱼ), so nothing here is approximated. The beeswarm is coded from
scratch: dots are binned along x at one dot diameter and fanned out
symmetrically in y, so each row's thickness follows its local density. The
self-check is additivity: for every sample the Shapley values sum to
f(xᵢ) − mean f.

Statistics: n = 400 samples, 8 features; each dot is one sample; the
column at the right gives mean |φ| over samples; colour runs from the
5th to the 95th percentile of each feature; no test. All data are
simulated.
"""

import matplotlib.pyplot as plt
import numpy as np

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(125)
N = 400

# ------------------------------------------------------------- DATA ----
# feature: (values, effect on the log-odds per standard deviation)
latent = rng.normal(0, 1, N)                         # shared "loyalty"
FEATURES = {
    "Tenure (months)": (rng.gamma(6.0, 5, N) + 5 * np.abs(latent), -0.95),
    "Support calls": (rng.poisson(np.exp(0.5 - 0.3 * latent)), 0.70),
    "Monthly charge": (rng.normal(72, 18, N), 0.52),
    "Late payments": (rng.poisson(0.9, N), 0.40),
    "Data use (GB)": (rng.lognormal(2.6, 0.6, N), -0.30),
    "Premium plan": (rng.random(N) < 0.35, -0.22),
    "Discount (%)": (rng.uniform(0, 30, N), -0.15),
    "Age (years)": (rng.normal(44, 13, N), -0.07),
}
X = np.column_stack([values for values, _beta in FEATURES.values()])
X = X.astype(float)
weight = np.array([beta for _v, beta in FEATURES.values()]) / X.std(axis=0)
INTERCEPT = -1.2

# ---------------------------------------------------- SHAPLEY VALUES ---
f = INTERCEPT + X @ weight                           # model output, log-odds
phi = weight * (X - X.mean(axis=0))                  # exact for a linear model
importance = np.abs(phi).mean(axis=0)
order = np.argsort(-importance)                      # most important first
low, high = np.percentile(X, [5, 95], axis=0)
shade = np.clip((X - low) / (high - low), 0, 1)      # 0 = low, 1 = high


def beeswarm(x, bin_width, step, half_height):
    """y offsets: dots sharing an x bin fan out as 0, +1, -1, +2, -2, ..."""
    level = np.zeros(x.size)
    bins = np.floor(x / bin_width).astype(int)
    for b in np.unique(bins):
        members = np.flatnonzero(bins == b)
        slot = np.arange(members.size)
        level[members] = (slot + 1) // 2 * np.where(slot % 2, 1, -1)
    # a row too tall for its lane is compressed as a whole, so its
    # thickness stays proportional to the local density
    return level * min(step, half_height / max(np.abs(level).max(), 1))


# ------------------------------------------------------- SELF-CHECK ---
residual = np.abs(phi.sum(axis=1) - (f - f.mean())).max()
assert residual < 1e-9, residual
assert np.all(np.diff(importance[order]) <= 0)
print(f"fig125: self-check passed (additivity: max |Σφ − (f − mean f)| = "
      f"{residual:.1e} over {N} samples; base value {f.mean():.3f}; top "
      f"feature {list(FEATURES)[order[0]]}, mean |φ| = "
      f"{importance[order[0]]:.3f})")

# ------------------------------------------------------------ FIGURE --
AX_W, AX_H, DOT = 62.0, 58.0, 0.8                    # millimetres
fig = ms.figure(120, 74)
ax = ms.axes(fig, 28, 11, AX_W, AX_H)
X_LIM = (np.floor(phi.min() - 0.1), np.ceil(phi.max() + 0.1))
ax.set_xlim(*X_LIM)
ax.set_ylim(len(FEATURES) - 0.45, -0.75)
ax.axvline(0, color=ms.GREY, lw=0.6, zorder=1)

bin_width = DOT * (X_LIM[1] - X_LIM[0]) / AX_W       # one dot diameter in x
step = DOT * (len(FEATURES) + 0.3) / AX_H            # ... and in row units
row = ax.get_yaxis_transform()
for y, j in enumerate(order):
    offset = beeswarm(phi[:, j], bin_width, step, half_height=0.40)
    ax.scatter(phi[:, j], y + offset, c=shade[:, j], cmap="viridis", vmin=0,
               vmax=1, s=3.6, linewidths=0, zorder=2)   # 3,200 dots: vector
    ax.text(1.115, y, f"{importance[j]:.2f}", transform=row, ha="right",
            va="center", fontsize=ms.FS_TICK)
ax.text(1.115, -0.72, "Mean |φ|", transform=row, ha="right", va="center",
        fontsize=ms.FS_TICK, color=ms.GREY_DARK)

ax.set_yticks(range(len(FEATURES)), [list(FEATURES)[j] for j in order])
ax.tick_params(axis="y", length=0, pad=4)
ax.spines["left"].set_visible(False)
ax.set_xlabel("SHAP value (change in predicted log-odds)")
ax.text(0.985, 0.02, f"n = {N} samples", transform=ax.transAxes, ha="right",
        fontsize=ms.FS_TICK, color=ms.GREY_DARK)

cax = ms.axes(fig, 104, 11 + AX_H * 0.22, 2.2, AX_H * 0.56)
bar = fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(0, 1), "viridis"),
                   cax=cax, ticks=[0, 1])
bar.ax.set_yticklabels(["Low", "High"])
bar.ax.tick_params(length=0, pad=2)
bar.set_label("Feature value", labelpad=-4)
bar.outline.set_visible(False)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig125_shap_beeswarm.{ext}")
print("fig125_shap_beeswarm: saved png + pdf")
