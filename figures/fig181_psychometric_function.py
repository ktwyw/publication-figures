"""Fig. 181 - Psychometric function, fitted threshold (single column, 89 mm).

The standard summary of a two-alternative forced-choice experiment:
proportion correct against stimulus contrast on a log axis, each point
with its Wilson 95% interval and an area proportional to its number of
trials, a Weibull function with the guess rate fixed at 0.5 and a free
lapse rate fitted by maximum likelihood, and the 75%-correct threshold
dropped to the axis with its parametric-bootstrap 95% interval as a
horizontal bar. The self-check is that the fitted log-likelihood is at
least that of the generating parameters, that the hand-coded Wilson
interval equals scipy.stats.binomtest(...).proportion_ci, and that the
true threshold lies inside the bootstrap interval.

Statistics: ψ(c) = 0.5 + (0.5 − λ)(1 − exp(−(c/α)^β)); truth α = 6%,
β = 2.2, λ = 0.03; 8 contrasts, 40-80 trials each (mean 60, most near
threshold; binomial); error bars, Wilson 95% intervals; bar, 2.5th-97.5th
percentiles of 600 parametric-bootstrap refits. All data are simulated.
"""

import numpy as np
from scipy import optimize, stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(181)

# ------------------------------------------------------------- DATA ----
GUESS = 0.5                                   # two alternatives
TRUE = dict(alpha=6.0, beta=2.2, lapse=0.03)  # contrast in %
P_THRESHOLD = 0.75                            # criterion for the threshold
CONTRAST = np.geomspace(1.5, 24.0, 8)         # % contrast
N_TRIALS = np.array([40, 50, 70, 80, 80, 70, 50, 40])   # mean 60
N_BOOT = 600
LAPSE_MAX = 0.10                              # bound on the lapse rate


def weibull(c, alpha, beta, lapse):
    """Proportion correct at contrast c."""
    return GUESS + (1 - GUESS - lapse) * (1 - np.exp(-(c / alpha) ** beta))


def threshold(alpha, beta, lapse):
    """Contrast at which the function reaches P_THRESHOLD."""
    core = (P_THRESHOLD - GUESS) / (1 - GUESS - lapse)
    return alpha * (-np.log(1 - core)) ** (1 / beta)


n_trials = N_TRIALS
n_correct = rng.binomial(n_trials, weibull(CONTRAST, **TRUE))


# --------------------------------------------------------- ESTIMATOR ---
def log_likelihood(params, k, n):
    p = np.clip(weibull(CONTRAST, *params), 1e-9, 1 - 1e-9)
    return np.sum(k * np.log(p) + (n - k) * np.log(1 - p))


def fit(k, n, starts):
    """Maximum likelihood in (ln α, ln β, λ); best of the given starts."""
    def cost(q):
        return -log_likelihood((np.exp(q[0]), np.exp(q[1]), q[2]), k, n)
    bounds = [np.log([0.5, 60.0]), np.log([0.3, 12.0]), (0.0, LAPSE_MAX)]
    best = min((optimize.minimize(cost, [np.log(a), np.log(b), lam],
                                  method="L-BFGS-B", bounds=bounds)
                for a, b, lam in starts), key=lambda res: res.fun)
    return np.exp(best.x[0]), np.exp(best.x[1]), best.x[2]


def wilson(k, n, level=0.95):
    """Wilson score interval for a binomial proportion."""
    z = stats.norm.ppf(0.5 + level / 2)
    centre = (k + z ** 2 / 2) / (n + z ** 2)
    half = z * np.sqrt(k * (n - k) / n + z ** 2 / 4) / (n + z ** 2)
    return centre - half, centre + half


GRID = [(a, b, 0.02) for a in (3.0, 6.0, 12.0) for b in (1.0, 2.5, 5.0)]
fitted = fit(n_correct, n_trials, GRID)
c_hat = threshold(*fitted)

# parametric bootstrap: resample from the fitted function, refit, re-read
boot = np.empty(N_BOOT)
for i in range(N_BOOT):
    k_star = rng.binomial(n_trials, weibull(CONTRAST, *fitted))
    boot[i] = threshold(*fit(k_star, n_trials, [fitted]))
ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

p_hat = n_correct / n_trials
w_low, w_high = wilson(n_correct, n_trials)

# ------------------------------------------------------- SELF-CHECK ---
ll_fit = log_likelihood(fitted, n_correct, n_trials)
ll_true = log_likelihood(tuple(TRUE.values()), n_correct, n_trials)
assert ll_fit >= ll_true - 1e-9, (ll_fit, ll_true)
for k, n, low, high in zip(n_correct, n_trials, w_low, w_high):
    ref = stats.binomtest(int(k), int(n)).proportion_ci(0.95, method="wilson")
    assert abs(low - ref.low) < 1e-10 and abs(high - ref.high) < 1e-10
c_true = threshold(**TRUE)
assert ci_low < c_true < ci_high, (ci_low, c_true, ci_high)
assert abs(weibull(c_hat, *fitted) - P_THRESHOLD) < 1e-12
print(f"fig181: self-check passed (log-likelihood {ll_fit:.2f} fitted vs "
      f"{ll_true:.2f} at truth; threshold {c_hat:.2f}% [95% CI "
      f"{ci_low:.2f}-{ci_high:.2f}], truth {c_true:.2f}%; slope β = "
      f"{fitted[1]:.2f}; lapse {fitted[2]:.3f}; Wilson matches scipy)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 64)
ax = ms.axes(fig, 14, 11, 70, 48)
X_LIM, Y_LIM = (1.2, 30.0), (0.30, 1.02)
Y_BAR = 0.345                                 # height of the CI bar

ax.axhline(GUESS, color=ms.GREY, lw=0.6, ls=(0, (4, 3)), zorder=1)
ax.text(X_LIM[1], GUESS - 0.014, "Chance", color=ms.GREY_DARK, ha="right",
        va="top", fontsize=ms.FS_TICK)

c_fine = np.geomspace(*X_LIM, 400)
ax.plot(c_fine, weibull(c_fine, *fitted), color=ms.BLUE, lw=1.3, zorder=3)
ax.errorbar(CONTRAST, p_hat, yerr=[p_hat - w_low, w_high - p_hat],
            fmt="none", ecolor=ms.INK, elinewidth=0.6, capsize=0, zorder=4)
ax.scatter(CONTRAST, p_hat, s=0.45 * n_trials, facecolor="white",
           edgecolor=ms.INK, linewidths=0.8, zorder=5)

# threshold: from the criterion on the y axis to the curve, then down
ax.plot([X_LIM[0], c_hat, c_hat], [P_THRESHOLD, P_THRESHOLD, Y_BAR],
        color=ms.VERMILLION, lw=0.7, ls=(0, (2, 2)), zorder=2)
ax.plot([ci_low, ci_high], [Y_BAR, Y_BAR], color=ms.VERMILLION, lw=2.2,
        solid_capstyle="butt", zorder=4)
ax.plot(c_hat, Y_BAR, "o", ms=3.4, mfc="white", mec=ms.VERMILLION,
        mew=0.9, zorder=5)
ax.text(ci_high * 1.1, Y_BAR, "Bootstrap 95% CI", color=ms.VERMILLION,
        va="center", fontsize=ms.FS_TICK)

ax.text(0.985, 0.57, f"Threshold c₇₅ = {c_hat:.1f}%\n"
        f"(95% CI {ci_low:.1f}–{ci_high:.1f}%)\n"
        f"Slope β = {fitted[1]:.1f}\nLapse λ = {fitted[2]:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=ms.FS_TICK,
        linespacing=1.35)
ax.text(0.03, 0.97, "Weibull fit (maximum likelihood)", color=ms.BLUE,
        transform=ax.transAxes, va="top", fontsize=ms.FS_TICK)
ax.text(0.03, 0.895, f"Point area ∝ trials (n = {n_trials.min()}–"
        f"{n_trials.max()});\nbars, Wilson 95% intervals",
        transform=ax.transAxes, va="top", fontsize=ms.FS_SMALL,
        color=ms.GREY_DARK)

ax.set_xscale("log")
ax.set_xlim(*X_LIM)
ax.set_ylim(*Y_LIM)
ax.set_xticks([2, 5, 10, 20])
ms.plain_log_ticks(ax.xaxis)
ax.set_yticks([0.5, 0.75, 1.0])
ax.set_yticks([0.6, 0.7, 0.8, 0.9], minor=True)
ax.spines["left"].set_bounds(0.5, 1.0)
ax.set_xlabel("Stimulus contrast (%)")
ax.set_ylabel("Proportion correct")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig181_psychometric_function.{ext}")
print("fig181_psychometric_function: saved png + pdf")
