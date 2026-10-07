"""Fig. 113 - Response surface fitted to a designed experiment (89 mm).

Extends fig017, which draws a known function: here the surface is
estimated. A full quadratic model is fitted by least squares to a
13-run central composite design with five centre replicates, the
optimum is the model's stationary point solved from the Hessian, R² is
reported, and the contour levels are echoed on the colour bar. The
self-check is that the Hessian is negative definite (so the stationary
point is a maximum), that the model gradient vanishes there, that the
optimum lies inside the plotted design space, and that no grid value of
the surface exceeds it.

Statistics: n = 13 runs (circles), ordinary least squares, six
coefficients; star, model-predicted optimum. All data are simulated.
"""

import numpy as np

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(1313)
T_RANGE, TAU_RANGE = (60.0, 140.0), (2.0, 22.0)   # plotted design space

# ------------------------------------------------------------- DATA ----
# Central composite design: 4 factorial, 4 axial and 5 centre runs.
AXIAL = np.sqrt(2)
coded = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1], [-AXIAL, 0],
                  [AXIAL, 0], [0, -AXIAL], [0, AXIAL], [0, 0], [0, 0],
                  [0, 0], [0, 0], [0, 0]])
temperature = 100 + 25 * coded[:, 0]              # °C
residence = 12 + 6 * coded[:, 1]                  # min
truth = (86 - 0.0125 * (temperature - 108) ** 2
         - 0.42 * (residence - 14.5) ** 2
         + 0.035 * (temperature - 108) * (residence - 14.5))
yield_pct = truth + rng.normal(0, 3.0, truth.size)


# ------------------------------------------------------------ MODEL ----
def design_matrix(t, tau):
    """Columns 1, T, tau, T², tau², T·tau of the full quadratic model."""
    return np.column_stack([np.ones_like(t), t, tau, t ** 2, tau ** 2,
                            t * tau])


def predict(t, tau):
    return design_matrix(np.atleast_1d(t), np.atleast_1d(tau)) @ beta


X = design_matrix(temperature, residence)
beta, *_ = np.linalg.lstsq(X, yield_pct, rcond=None)
fitted = X @ beta
r_squared = 1 - (np.sum((yield_pct - fitted) ** 2)
                 / np.sum((yield_pct - yield_pct.mean()) ** 2))

# stationary point: gradient b + H x = 0
hessian = np.array([[2 * beta[3], beta[5]], [beta[5], 2 * beta[4]]])
optimum = np.linalg.solve(hessian, -beta[1:3])
optimum_yield = float(predict(*optimum)[0])

t_grid, tau_grid = np.meshgrid(np.linspace(*T_RANGE, 240),
                               np.linspace(*TAU_RANGE, 240))
surface = predict(t_grid.ravel(), tau_grid.ravel()).reshape(t_grid.shape)

# ------------------------------------------------------- SELF-CHECK ---
eigenvalues = np.linalg.eigvalsh(hessian)
gradient = beta[1:3] + hessian @ optimum
assert np.all(eigenvalues < 0), eigenvalues        # a maximum, not a saddle
assert np.max(np.abs(gradient)) < 1e-9, gradient
assert T_RANGE[0] < optimum[0] < T_RANGE[1]
assert TAU_RANGE[0] < optimum[1] < TAU_RANGE[1]
assert surface.max() <= optimum_yield + 1e-9, (surface.max(), optimum_yield)
print(f"fig113: self-check passed (Hessian eigenvalues "
      f"{eigenvalues[0]:.3f}, {eigenvalues[1]:.4f} < 0; optimum "
      f"{optimum[0]:.1f} °C, {optimum[1]:.1f} min, {optimum_yield:.1f}% "
      f">= grid max {surface.max():.1f}%; R² = {r_squared:.3f})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 72)
ax = ms.axes(fig, 13, 11, 55, 49)
ax_cbar = ms.axes(fig, 71, 11, 2.6, 49)

levels = np.arange(30, 91, 10)
image = ax.imshow(surface, origin="lower", extent=[*T_RANGE, *TAU_RANGE],
                  aspect="auto", cmap="viridis", vmin=30, vmax=90,
                  interpolation="bilinear", rasterized=True)
ax.contour(t_grid, tau_grid, surface, levels=levels, colors="white",
           linewidths=0.5, alpha=0.8)
ax.scatter(temperature, residence, s=14, facecolor="white",
           edgecolor=ms.INK, linewidths=0.6, zorder=3, label="Measured run")
ax.scatter(*optimum, marker="*", s=70, facecolor=ms.VERMILLION,
           edgecolor="white", linewidths=0.5, zorder=4,
           label="Predicted optimum")
ax.set_xlim(*T_RANGE)
ax.set_ylim(*TAU_RANGE)
ax.set_xticks(np.arange(60, 141, 20))
ax.set_yticks(np.arange(2, 23, 4))
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Residence time (min)")
for side in ("top", "right"):
    ax.spines[side].set_visible(True)
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01), ncol=2,
          columnspacing=1.6, handletextpad=0.2, borderaxespad=0)

cbar = fig.colorbar(image, cax=ax_cbar, ticks=levels)
for level in levels[1:-1]:                # the contour levels, echoed
    cbar.ax.plot([0, 1], [level, level], color="white", lw=0.5, alpha=0.8)
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(length=2, width=0.5)
cbar.set_label("Predicted yield (%)")
fig.text(71 / 89, 64.5 / 72, f"R² = {r_squared:.2f}", fontsize=ms.FS_TICK,
         ha="left", va="center", color=ms.GREY_DARK)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig113_response_surface_fit.{ext}")
print("fig113_response_surface_fit: saved png + pdf")
