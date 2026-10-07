"""Fig. 179 - Wright-Fisher drift and fixation (double column, 183 mm).

Genetic drift at three population sizes on one frequency axis: twenty
replicate populations each, the same 200 generations, so the only thing
that changes from panel to panel is N. Small populations fix or lose
the allele within tens of generations; large ones barely move. The band
is the expected spread, p₀ ± one s.d., which follows the heterozygosity
decay (1 − 1/(2N))ᵗ. The last panel adds selection: the fraction of
simulated populations in which the allele fixes, against Kimura's
diffusion formula. The self-check is that the neutral fixation fraction
has p₀ inside its 95% interval, that the mean simulated heterozygosity
for N = 20 follows H₀(1 − 1/(2N))ᵗ within 3 standard errors, and that
Kimura's formula tends to p₀ as s tends to 0.

Model: Wright-Fisher, 2N gene copies resampled binomially each
generation; a-c, neutral, p₀ = 0.5; d, genic selection p' = p(1 + s)/
(1 + ps) before sampling, N = 50, p₀ = 0.1, 3,000 populations per point,
error bars 95% Wilson intervals; line, u = (1 − exp(−4Nsp₀))/
(1 − exp(−4Ns)) (Kimura 1962). All data are simulated.
"""

import numpy as np
from scipy import stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(179)

# -------------------------------------------------- GOVERNING MODEL ----
SIZES = [20, 200, 2000]            # diploid population sizes N, panels a-c
GENERATIONS, P_START, N_SHOWN = 200, 0.5, 20
N_CHECK, T_CHECK = 5000, [5, 10, 20, 40]     # heterozygosity check, N = 20
N_SEL, P_SEL, N_RUNS = 50, 0.1, 3000         # panel d
SCALED_S = np.arange(-2.0, 6.5, 1.0)         # 2Ns


def drift(n_diploid, p_start, n_pop, generations):
    """Neutral allele-frequency paths, shape (generations + 1, n_pop)."""
    copies = 2 * n_diploid
    count = np.full(n_pop, round(p_start * copies))
    path = [count]
    for _ in range(generations):
        count = rng.binomial(copies, count / copies)
        path.append(count)
    return np.array(path) / copies


def fixation_fraction(n_diploid, p_start, scaled_s, n_pop):
    """Run every population to loss or fixation; count fixations."""
    copies = 2 * n_diploid
    s = (scaled_s / copies)[:, None]             # s from 2Ns
    count = np.full((scaled_s.size, n_pop), round(p_start * copies))
    while np.any((count > 0) & (count < copies)):
        p = count / copies
        count = rng.binomial(copies, p * (1 + s) / (1 + p * s))
    return (count == copies).sum(axis=1)


def kimura(scaled_s, p_start):
    """Diffusion fixation probability; 2Ns = 0 is the neutral limit."""
    x = np.where(scaled_s == 0, 1.0, scaled_s)
    u = np.expm1(-2 * x * p_start) / np.expm1(-2 * x)
    return np.where(scaled_s == 0, p_start, u)


# -------------------------------------------------------- SIMULATION ---
paths = {n: drift(n, P_START, N_SHOWN, GENERATIONS) for n in SIZES}
many = drift(SIZES[0], P_START, N_CHECK, max(T_CHECK))[T_CHECK]
h_sim = (2 * many * (1 - many)).mean(axis=1)
h_sem = (2 * many * (1 - many)).std(axis=1, ddof=1) / N_CHECK ** 0.5
h_theory = (2 * P_START * (1 - P_START)
            * (1 - 1 / (2 * SIZES[0])) ** np.array(T_CHECK))
fixed = fixation_fraction(N_SEL, P_SEL, SCALED_S, N_RUNS)
interval = np.array([stats.binomtest(int(k), N_RUNS)
                     .proportion_ci(0.95, method="wilson") for k in fixed])

# ------------------------------------------------------- SELF-CHECK ---
neutral = int(np.flatnonzero(SCALED_S == 0)[0])
h_z = np.abs(h_sim - h_theory) / h_sem
inside = int(np.sum((interval[:, 0] <= kimura(SCALED_S, P_SEL))
                    & (kimura(SCALED_S, P_SEL) <= interval[:, 1])))
assert interval[neutral, 0] <= P_SEL <= interval[neutral, 1], interval[neutral]
assert np.all(h_z < 3), h_z
assert abs(kimura(np.array([1e-7]), P_SEL)[0] - P_SEL) < 1e-6
assert abs(kimura(np.array([-1e-7]), P_SEL)[0] - P_SEL) < 1e-6
print(f"fig179: self-check passed (neutral fixation {fixed[neutral]}/"
      f"{N_RUNS} = {fixed[neutral] / N_RUNS:.4f}, 95% CI "
      f"{interval[neutral, 0]:.4f} to {interval[neutral, 1]:.4f} holds p0 = "
      f"{P_SEL}; N = {SIZES[0]} heterozygosity within {h_z.max():.2f} s.e. "
      f"of H0(1-1/2N)^t at t = {T_CHECK}; Kimura -> p0 as s -> 0; "
      f"{inside}/{SCALED_S.size} intervals hold the diffusion value)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 70)
BOTTOM, HEIGHT, WIDTH = 11.0, 42.0, 31.0
drift_axes = [ms.axes(fig, 14 + 35 * k, BOTTOM, WIDTH, HEIGHT)
              for k in range(3)]
ax_d = ms.axes(fig, 132, BOTTOM, 45, HEIGHT)
gens = np.arange(GENERATIONS + 1)

# a-c, trajectories: colour while both alleles survive to generation 200,
# grey once the allele has been fixed or lost
for ax, n in zip(drift_axes, SIZES):
    decay = (1 - 1 / (2 * n)) ** gens                # E[H(t)]/H(0)
    spread = np.sqrt(P_START * (1 - P_START) * (1 - decay))
    ax.fill_between(gens, np.clip(P_START - spread, 0, 1),
                    np.clip(P_START + spread, 0, 1), color=ms.GREY_LIGHT,
                    alpha=0.45, lw=0)
    path = paths[n]
    absorbed = (path[-1] == 0) | (path[-1] == 1)
    for k in np.argsort(~absorbed):              # grey first, colour on top
        ax.plot(gens, path[:, k], lw=0.5,
                color=ms.GREY if absorbed[k] else ms.BLUE)
    ax.set_xlim(0, GENERATIONS)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 100, 200])
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.spines["left"].set_bounds(0, 1)
    ax.text(0.5, 1.17, f"N = {n:,}", transform=ax.transAxes, ha="center",
            va="bottom", fontweight="bold")
    ax.text(0.5, 1.02, f"E[H₂₀₀]/H₀ = {decay[-1]:.3f}\n"
            f"fixed or lost: {absorbed.sum()} of {N_SHOWN}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=ms.FS_SMALL, color=ms.GREY_DARK, linespacing=1.25)
    if ax is not drift_axes[0]:
        ax.tick_params(labelleft=False)
drift_axes[2].text(0.06, 0.97,
                   "Blue, both alleles present\nGrey, fixed or lost",
                   transform=drift_axes[2].transAxes, va="top",
                   fontsize=ms.FS_SMALL, color=ms.GREY_DARK, linespacing=1.3)
drift_axes[2].text(0.06, 0.05, "Band, p₀ ± 1 s.d. (theory)",
                   transform=drift_axes[2].transAxes, va="bottom",
                   fontsize=ms.FS_SMALL, color=ms.GREY_DARK)
drift_axes[0].set_ylabel("Allele frequency p")
drift_axes[1].set_xlabel("Generation")

# d, fixation probability: simulation against the diffusion formula
fine = np.linspace(SCALED_S[0] - 0.3, SCALED_S[-1] + 0.3, 400)
ax_d.plot(fine, kimura(fine, P_SEL), color=ms.INK, lw=1.2)
fraction = fixed / N_RUNS
for k in range(SCALED_S.size):
    colour = ms.VERMILLION if k == neutral else ms.BLUE
    ax_d.errorbar(SCALED_S[k], fraction[k],
                  yerr=[[fraction[k] - interval[k, 0]],
                        [interval[k, 1] - fraction[k]]],
                  fmt="o", ms=4.0 if k == neutral else 3.0, color=colour,
                  mec="white", mew=0.4, ecolor=colour, elinewidth=0.7,
                  capsize=1.6, capthick=0.7, zorder=3)
ax_d.annotate("Neutral: u = p₀", xy=(0, P_SEL), xytext=(7, -7),
              textcoords="offset points", color=ms.VERMILLION, va="center",
              fontsize=ms.FS_TICK)
ax_d.text(0.04, 0.97, "Line, Kimura diffusion formula\n"
          f"Points, {N_RUNS:,} simulated populations;\n"
          "bars, 95% CI\n"
          f"N = {N_SEL}, p₀ = {P_SEL}", transform=ax_d.transAxes, va="top",
          fontsize=ms.FS_SMALL, color=ms.GREY_DARK, linespacing=1.3)
ax_d.set_xlim(fine[0], fine[-1])
ax_d.set_ylim(0, 0.8)
ax_d.set_xticks(np.arange(-2, 7, 2))
ax_d.set_xlabel("Scaled selection coefficient 2Ns")
ax_d.set_ylabel("Fixation probability u")

for ax, letter in zip(drift_axes, "abc"):
    ms.panel_label(ax, letter, dx_pt=-26 if letter == "a" else -6, dy_pt=22)
ms.panel_label(ax_d, "d", dx_pt=-26, dy_pt=22)
ms.assert_aligned(drift_axes + [ax_d])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig179_genetic_drift.{ext}")
print("fig179_genetic_drift: saved png + pdf")
