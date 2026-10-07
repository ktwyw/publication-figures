"""Fig. 126 - GP regression from the kernel equations (double column, 183 mm).

a, a Gaussian-process posterior coded from the textbook equations (Rasmussen &
Williams, Algorithm 2.1): squared-exponential kernel, one Cholesky
factor, posterior mean, 95% band and three sample paths; the band widens
where there are no observations. b, the log marginal likelihood as a
function of the length scale, the quantity that chooses it: too short
fits the noise, too long cannot follow the data. Its maximiser, found by
scipy.optimize.minimize_scalar, is the value used in a. The self-check
is that the Cholesky posterior mean equals the direct
np.linalg.solve formula, that the posterior variance is non-negative and
below the prior variance at every training input, and that the
optimiser agrees with the grid argmax within one grid step.

Statistics: n = 18 noisy observations (noise s.d. 0.25, treated as
known); line, posterior mean; band, pointwise 95% credible interval of
the latent function (mean ± 1.96 s.d.); no test. All data are simulated.
"""

import numpy as np
from scipy import linalg, optimize

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(126)
SIGNAL_SD, NOISE_SD = 1.0, 0.25           # prior s.d. of f; noise s.d.
LENGTH_GRID = np.logspace(-1, 1, 161)     # candidate length scales


def truth(x):
    """The function the observations were drawn around."""
    return 1.1 * np.sin(0.9 * x) + 0.5 * np.cos(2.3 * x)


# ------------------------------------------------------------- DATA ----
# 18 inputs on [0, 10] with a gap, so the band has somewhere to widen
x_train = np.sort(np.concatenate([rng.uniform(0.2, 5.4, 12),
                                  rng.uniform(7.6, 9.8, 6)]))
y_train = truth(x_train) + rng.normal(0, NOISE_SD, x_train.size)
x_test = np.linspace(0, 10, 401)
N = x_train.size


# --------------------------------------------------- GAUSSIAN PROCESS --
def kernel(a, b, length):
    """Squared-exponential (RBF) covariance between two sets of inputs."""
    return SIGNAL_SD ** 2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2
                                   / length ** 2)


def factorise(length):
    """Cholesky factor of K + noise, and alpha = (K + noise)⁻¹ y."""
    k_noisy = kernel(x_train, x_train, length) + NOISE_SD ** 2 * np.eye(N)
    chol = linalg.cholesky(k_noisy, lower=True)
    return chol, linalg.cho_solve((chol, True), y_train)


def log_marginal_likelihood(length):
    """log p(y | X, length) = -½ yᵀα - Σ log Lᵢᵢ - (n/2) log 2π."""
    chol, alpha = factorise(length)
    return (-0.5 * y_train @ alpha - np.log(np.diag(chol)).sum()
            - 0.5 * N * np.log(2 * np.pi))


def posterior(x_new, length):
    """Posterior mean and covariance of the latent function at x_new."""
    chol, alpha = factorise(length)
    k_star = kernel(x_new, x_train, length)
    v = linalg.solve_triangular(chol, k_star.T, lower=True)
    return k_star @ alpha, kernel(x_new, x_new, length) - v.T @ v


lml_grid = np.array([log_marginal_likelihood(s) for s in LENGTH_GRID])
fit = optimize.minimize_scalar(lambda u: -log_marginal_likelihood(np.exp(u)),
                               bounds=np.log(LENGTH_GRID[[0, -1]]),
                               method="bounded")
length_opt, lml_opt = float(np.exp(fit.x)), -fit.fun
mean, cov = posterior(x_test, length_opt)
sd = np.sqrt(np.clip(np.diag(cov), 0, None))
# three functions drawn from the posterior (jitter keeps the factor stable)
paths = mean[:, None] + linalg.cholesky(
    cov + 1e-8 * np.eye(x_test.size), lower=True) @ rng.normal(
        size=(x_test.size, 3))

# ------------------------------------------------------- SELF-CHECK ---
k_noisy = kernel(x_train, x_train, length_opt) + NOISE_SD ** 2 * np.eye(N)
direct = kernel(x_test, x_train, length_opt) @ np.linalg.solve(k_noisy,
                                                                y_train)
var_train = np.diag(posterior(x_train, length_opt)[1])
grid_step = np.log(LENGTH_GRID[1] / LENGTH_GRID[0])
grid_best = LENGTH_GRID[np.argmax(lml_grid)]
assert np.abs(mean - direct).max() < 1e-8
assert np.diag(cov).min() > -1e-10
assert np.all(var_train < SIGNAL_SD ** 2) and var_train.min() >= 0
assert abs(np.log(length_opt / grid_best)) <= grid_step, (length_opt,
                                                          grid_best)
inside = np.mean(np.abs(truth(x_test) - mean) <= 1.96 * sd)
print(f"fig126: self-check passed (Cholesky mean = direct solve to "
      f"{np.abs(mean - direct).max():.1e}; posterior variance at training "
      f"inputs {var_train.min():.3f}–{var_train.max():.3f} < prior "
      f"{SIGNAL_SD ** 2:g}; optimiser ℓ = {length_opt:.3f} vs grid "
      f"{grid_best:.3f}; truth inside the 95% band at {100 * inside:.0f}% "
      f"of x)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 68)
gs = ms.grid(fig, 1, 2, left=13, right=5, top=7, bottom=11, wspace=17,
             width_ratios=[1.75, 1])
ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# a, posterior at the fitted length scale
ax_a.fill_between(x_test, mean - 1.96 * sd, mean + 1.96 * sd, color=ms.BLUE,
                  alpha=0.18, lw=0, label="95% credible band")
ax_a.plot(x_test, paths, color=ms.BLUE, lw=0.5, alpha=0.55,
          label=["Posterior samples", None, None])
ax_a.plot(x_test, truth(x_test), color=ms.GREY_DARK, lw=0.9, ls=(0, (4, 2)),
          label="True function")
ax_a.plot(x_test, mean, color=ms.BLUE, lw=1.5, label="Posterior mean")
ax_a.plot(x_train, y_train, "o", color=ms.INK, markersize=3.0,
          markeredgewidth=0, zorder=5, label=f"Observations (n = {N})")
ax_a.set_xlim(0, 10)
ax_a.set_ylim(-3.2, 4.6)
ax_a.set_yticks(np.arange(-3, 4))
ax_a.spines["left"].set_bounds(-3, 3)
ax_a.set_xlabel("Input x")
ax_a.set_ylabel("Output y")
handles, labels = ax_a.get_legend_handles_labels()
show = [4, 3, 0, 1, 2]                    # data first, then model, then truth
ax_a.legend([handles[i] for i in show], [labels[i] for i in show],
            loc="upper left", ncols=3, handlelength=1.7, columnspacing=1.6,
            bbox_to_anchor=(0.01, 1.0))

# b, the evidence that selects the length scale
y_low = 5 * np.floor(lml_grid[LENGTH_GRID >= 0.3].min() / 5)
ax_b.plot(LENGTH_GRID, lml_grid, color=ms.GREY_DARK, lw=1.2)
ax_b.plot([length_opt] * 2, [y_low, lml_opt], color=ms.BLUE, lw=0.7,
          ls=(0, (3, 2)))
ax_b.plot(length_opt, lml_opt, "o", color=ms.BLUE, markersize=4.2,
          markeredgewidth=0, zorder=4)
ax_b.annotate(f"Maximum, ℓ = {length_opt:.2f}\n(used in a)",
              (length_opt, lml_opt), xytext=(-5, -22),
              textcoords="offset points", ha="right", va="top",
              fontsize=ms.FS_TICK, color=ms.BLUE, linespacing=1.15)
ax_b.set_xscale("log")
ax_b.set_xlim(0.1, 10)
ms.plain_log_ticks(ax_b.xaxis)
ax_b.set_ylim(y_low, 5 * np.ceil((lml_opt + 2) / 5))
ax_b.set_xlabel("Length scale ℓ")
ax_b.set_ylabel("Log marginal likelihood")

ms.panel_label(ax_a, "a", dx_pt=-26)
ms.panel_label(ax_b, "b", dx_pt=-30)
ms.assert_aligned([ax_a, ax_b])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig126_gp_regression.{ext}")
print("fig126_gp_regression: saved png + pdf")
