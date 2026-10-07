"""Fig. 146 - Corner plot of a sampled posterior (1.5 column, 120 mm).

The standard way to show a Bayesian fit: every one-dimensional marginal
on the diagonal (median and 16th/84th percentiles) and every pairwise
marginal below it, with contours that enclose a stated fraction of the
posterior mass rather than arbitrary density values. A damped
oscillation is fitted to noisy samples by a random-walk
Metropolis-Hastings sampler written out in the script. The self-check
is that the four chains agree (Gelman-Rubin R-hat < 1.05), that each
contour encloses its nominal share of the samples within 2 percentage
points, and that every true value lies in its 95% credible interval.

Model: y = A exp(−t/τ) cos(2πt/P) + Gaussian noise (s.d. 0.25, known);
truth A = 2, τ = 3 s, P = 2 s; n = 60 samples; flat priors on
positive values. Sampler: 4 chains x 50,000 steps, first 5,000 of each
discarded, Gaussian proposal scaled from the least-squares covariance.
Contours enclose 39.3% and 86.5% of the mass (1σ and 2σ of a
two-dimensional Gaussian). All data are simulated.
"""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy import optimize

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(146)

# ------------------------------------------------------------- DATA ----
TRUTH = np.array([2.0, 3.0, 2.0])               # A (a.u.), τ (s), P (s)
LABELS = ["Amplitude A (a.u.)", "Decay time τ (s)", "Period P (s)"]
SYMBOLS = ["A", "τ", "P"]
NOISE_SD, N_OBS = 0.25, 60
N_CHAINS, N_STEPS, BURN_IN = 4, 50_000, 5_000
MASS = (0.393, 0.865)                           # 1σ and 2σ in two dimensions
BINS = 40


def model(t, amplitude, tau, period):
    return amplitude * np.exp(-t / tau) * np.cos(2 * np.pi * t / period)


t_obs = np.linspace(0, 10, N_OBS)
y_obs = model(t_obs, *TRUTH) + rng.normal(0, NOISE_SD, N_OBS)


# ----------------------------------------------------------- SAMPLER ---
def log_posterior(theta):
    """Gaussian log-likelihood with flat priors on positive values;
    theta has shape (chains, 3)."""
    residual = y_obs - model(t_obs, *theta.T[:, :, None])
    logp = -0.5 * np.sum((residual / NOISE_SD) ** 2, axis=1)
    return np.where(np.all(theta > 0, axis=1), logp, -np.inf)


# proposal: the least-squares covariance at the optimal random-walk scale
best, covariance = optimize.curve_fit(model, t_obs, y_obs, p0=[1.5, 2.0, 2.1],
                                      sigma=np.full(N_OBS, NOISE_SD),
                                      absolute_sigma=True)
step = np.linalg.cholesky(covariance * 2.38 ** 2 / 3)
spread = 3 * np.sqrt(np.diag(covariance))      # over-dispersed starts
theta = best + spread * rng.normal(size=(N_CHAINS, 3))
logp = log_posterior(theta)
chains = np.empty((N_STEPS, N_CHAINS, 3))
accepted = 0
for i in range(N_STEPS):                       # all four chains at once
    proposal = theta + rng.normal(size=(N_CHAINS, 3)) @ step.T
    logp_new = log_posterior(proposal)
    accept = np.log(rng.uniform(size=N_CHAINS)) < logp_new - logp
    theta = np.where(accept[:, None], proposal, theta)
    logp = np.where(accept, logp_new, logp)
    accepted += accept.sum()
    chains[i] = theta
acceptance = accepted / (N_STEPS * N_CHAINS)
kept = chains[BURN_IN:]                        # (steps, chains, parameters)
samples = kept.reshape(-1, 3)

# Gelman-Rubin: between-chain against within-chain variance
n_kept = kept.shape[0]
within = kept.var(axis=0, ddof=1).mean(axis=0)
between = n_kept * kept.mean(axis=0).var(axis=0, ddof=1)
r_hat = np.sqrt(((n_kept - 1) / n_kept * within + between / n_kept) / within)

low, median, high = np.percentile(samples, [16, 50, 84], axis=0)
edges = [np.linspace(m - 4.2 * s, m + 4.2 * s, BINS + 1)
         for m, s in zip(samples.mean(axis=0), samples.std(axis=0))]


def density_and_levels(i, j):
    """2-D histogram (x = parameter j, y = parameter i), the count
    levels whose super-level sets hold MASS of the samples (bins sorted
    from fullest down), and the share of samples above each level."""
    counts = np.histogram2d(samples[:, j], samples[:, i],
                            bins=(edges[j], edges[i]))[0].T
    ordered = np.sort(counts.ravel())[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    levels = [ordered[np.searchsorted(cumulative, m)] for m in MASS]
    enclosed = [counts[counts >= lv].sum() / len(samples) for lv in levels]
    return counts, levels, enclosed


pairs = {(i, j): density_and_levels(i, j) for i in range(3) for j in range(i)}

# ------------------------------------------------------- SELF-CHECK ---
assert np.all(r_hat < 1.05), r_hat
worst_mass = max(abs(got - nominal) for _d, _l, enclosed in pairs.values()
                 for got, nominal in zip(enclosed, MASS))
assert worst_mass < 0.02, worst_mass
lo95, hi95 = np.percentile(samples, [2.5, 97.5], axis=0)
assert np.all((lo95 < TRUTH) & (TRUTH < hi95)), (lo95, TRUTH, hi95)
print(f"fig146: self-check passed (acceptance {acceptance:.2f}; R-hat "
      + ", ".join(f"{r:.3f}" for r in r_hat)
      + f"; contour mass within {100 * worst_mass:.1f} points of 39.3% and "
      "86.5%; truth inside every 95% interval)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 116)
gs = ms.grid(fig, 3, 3, left=16, right=5, top=8, bottom=12, wspace=3.5,
             hspace=3.5)
FILL = ("#BBD7EA", ms.BLUE)                    # 86.5% region, 39.3% region
axs = np.empty((3, 3), dtype=object)
for i in range(3):
    for j in range(i + 1):
        ax = axs[i, j] = fig.add_subplot(gs[i, j])
        ax.axvline(TRUTH[j], color=ms.VERMILLION, lw=0.7)
        if i == j:                             # 1-D marginal
            counts, _ = np.histogram(samples[:, i], bins=edges[i])
            ax.stairs(counts, edges[i], color=ms.BLUE, lw=1.0)
            ax.axvline(median[i], color=ms.INK, lw=0.7)
            for q in (low[i], high[i]):
                ax.axvline(q, color=ms.INK, lw=0.7, ls=(0, (3, 2)))
            digits = max(0, 1 - int(np.floor(np.log10(high[i] - low[i]))))
            ax.set_title(f"{SYMBOLS[i]} = {median[i]:.{digits}f} "
                         f"+{high[i] - median[i]:.{digits}f}"
                         f"/−{median[i] - low[i]:.{digits}f}", pad=3,
                         fontsize=ms.FS_TICK)
            ax.set_ylim(0, counts.max() * 1.08)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
        else:                                  # 2-D marginal
            density, levels, _enclosed = pairs[i, j]
            centres = [(e[:-1] + e[1:]) / 2 for e in (edges[j], edges[i])]
            ax.contourf(*centres, density,
                        levels=[levels[1], levels[0], density.max()],
                        colors=FILL)
            ax.axhline(TRUTH[i], color=ms.VERMILLION, lw=0.7)
            ax.set_ylim(edges[i][0], edges[i][-1])
            ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlim(edges[j][0], edges[j][-1])
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelbottom=i == 2, labelleft=j == 0 and i > 0)
for j in range(3):                             # labels on the outer panels
    axs[2, j].set_xlabel(LABELS[j])
for i in (1, 2):
    axs[i, 0].set_ylabel(LABELS[i])
fig.align_ylabels([axs[1, 0], axs[2, 0]])

# key in the empty upper triangle
handles = [Patch(facecolor=FILL[1], lw=0), Patch(facecolor=FILL[0], lw=0),
           Line2D([], [], color=ms.INK, lw=0.7),
           Line2D([], [], color=ms.INK, lw=0.7, ls=(0, (3, 2))),
           Line2D([], [], color=ms.VERMILLION, lw=0.7)]
fig.legend(handles, ["39.3% of posterior mass (1σ in 2-D)",
                     "86.5% of posterior mass (2σ in 2-D)", "Median",
                     "16th and 84th percentiles", "True value"],
           loc="upper right", bbox_to_anchor=(1 - 5 / 120, 1 - 8 / 116),
           borderaxespad=0, handlelength=1.6)
fig.text(1 - 5 / 120, 1 - 25 / 116,
         f"{N_CHAINS} chains × {N_STEPS - BURN_IN:,} steps after burn-in\n"
         f"Acceptance rate {acceptance:.2f}; R-hat ≤ {r_hat.max():.3f}",
         ha="right", va="top", fontsize=ms.FS_SMALL, color=ms.GREY_DARK)

for row in axs:
    ms.assert_aligned([ax for ax in row if ax is not None])
for column in axs.T:
    ms.assert_aligned([ax for ax in column if ax is not None],
                      edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig146_corner_posterior.{ext}")
print("fig146_corner_posterior: saved png + pdf")
