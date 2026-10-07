"""Fig. 176 - Epidemic curve with reproduction number (1.5 column, 120 mm).

The standard outbreak pair on one time axis: daily reported cases as
bars over the SEIR model's expected incidence, and below them the
effective reproduction number, once from the model (R₀(t)·S(t)/N) and
once estimated from the case counts alone by a renewal-equation
(Cori-type) estimator, which lags the step because it needs a window of
onsets. The self-check is that S + E + I + R = N to 1e-8 (relative),
that the infected pool E + I peaks within one day of the model R_t
crossing 1, and that without the intervention the final size z solves
z = 1 − exp(−R₀z) to 1e-4.

Model: SEIR, N = 200,000, R₀ = 2.5 cut to 0.75 on day 35, latent
period 3 d, infectious period 4 d; cases are symptom onsets (E to I),
40% reported, negative-binomial noise (dispersion 40). Estimator:
7-day windows plotted at the window end, Gamma(1, 5) prior, generation
interval from the model; band, 95% credible interval. All data are
simulated.
"""

import numpy as np
from scipy import integrate, optimize, stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(176)

# -------------------------------------------------- GOVERNING MODEL ----
N_POP, E_START = 200_000.0, 30.0
R0, R0_AFTER = 2.5, 0.75           # basic reproduction number, before/after
SIGMA, GAMMA = 1 / 3.0, 1 / 4.0    # 1/latent and 1/infectious period (per d)
T_INT, T_END = 35, 90              # intervention day; days simulated
REPORTED, DISPERSION = 0.4, 40.0   # reporting fraction; neg.-binomial k
WINDOW = 7                         # estimator window (days)
PRIOR_SHAPE, PRIOR_SCALE = 1.0, 5.0    # Gamma prior on R_t (mean 5, s.d. 5)
STEPS = 20                         # output points per day


def rhs(_t, state, r0):
    """SEIR plus the cumulative number of onsets C (dC/dt = σE)."""
    s, e, i, _r, _c = state
    infection = r0 * GAMMA * s * i / N_POP
    return [-infection, infection - SIGMA * e, SIGMA * e - GAMMA * i,
            GAMMA * i, SIGMA * e]


def run(segments):
    """Integrate through (t0, t1, R₀) segments; restart at each step in R₀."""
    state, times, states = [N_POP - E_START, E_START, 0.0, 0.0, 0.0], [], []
    for t0, t1, r0 in segments:
        t_eval = np.linspace(t0, t1, int(round((t1 - t0) * STEPS)) + 1)
        sol = integrate.solve_ivp(rhs, (t0, t1), state, args=(r0,),
                                  t_eval=t_eval, rtol=1e-10, atol=1e-8)
        state = sol.y[:, -1]
        keep = slice(None) if t1 == segments[-1][1] else slice(None, -1)
        times.append(sol.t[keep])
        states.append(sol.y[:, keep])
    return np.concatenate(times), np.concatenate(states, axis=1)


# ------------------------------------------------- SOLVER, ESTIMATOR ---
t, (s, e, i, r, onsets) = run([(0, T_INT, R0), (T_INT, T_END, R0_AFTER)])
rt_model = np.where(t < T_INT, R0, R0_AFTER) * s / N_POP
days = np.arange(T_END)                              # day d covers [d, d+1)
expected = REPORTED * np.diff(onsets[::STEPS])       # expected cases per day
cases = rng.negative_binomial(DISPERSION, DISPERSION / (DISPERSION + expected))

# generation interval of the model: latent stage, then a uniformly random
# moment of the exponential infectious stage (sum of two exponentials)
lag = np.arange(1, 41)


def interval_cdf(x):
    return 1 - (GAMMA * np.exp(-SIGMA * x)
                - SIGMA * np.exp(-GAMMA * x)) / (GAMMA - SIGMA)


weights = interval_cdf(lag + 0.5) - interval_cdf(np.maximum(lag - 0.5, 0))
weights /= weights.sum()
pressure = np.array([np.sum(weights[:d] * cases[:d][::-1][:lag.size])
                     for d in days])                 # Σ w(u)·cases(d − u)
first = int(np.argmax(np.cumsum(cases) >= 12)) + WINDOW   # enough onsets
ends = np.arange(first, T_END)                       # last day of each window
shape = PRIOR_SHAPE + np.array([cases[d - WINDOW + 1:d + 1].sum()
                                for d in ends])
rate = 1 / PRIOR_SCALE + np.array([pressure[d - WINDOW + 1:d + 1].sum()
                                   for d in ends])
rt_est = shape / rate                                # posterior mean
rt_low, rt_high = stats.gamma.ppf([[0.025], [0.975]], shape, scale=1 / rate)

# the same epidemic left alone, run until it has burnt out
_, free = run([(0, 400, R0)])
final_size = 1 - free[0, -1] / N_POP
final_theory = optimize.brentq(lambda z: z - 1 + np.exp(-R0 * z), 0.01, 1.0,
                               xtol=1e-14)

# ------------------------------------------------------- SELF-CHECK ---
conservation = np.abs(s + e + i + r - N_POP).max() / N_POP
t_peak = t[np.argmax(e + i)]
t_cross = t[np.argmax(rt_model < 1)]
late = ends >= T_INT + 2 * WINDOW                # windows clear of the step
late_gap = np.median(np.abs(rt_est[late]
                          - rt_model[(ends[late] + 1) * STEPS]))
assert conservation < 1e-8, conservation
assert abs(t_peak - t_cross) <= 1.0, (t_peak, t_cross)
assert abs(final_size - final_theory) < 1e-4, (final_size, final_theory)
assert late_gap < 0.1, late_gap                       # estimator finds R_t
print(f"fig176: self-check passed (S+E+I+R = N within {conservation:.1e}; "
      f"E+I peaks at day {t_peak:.2f}, R_t crosses 1 at day {t_cross:.2f}; "
      f"final size without intervention {final_size:.5f} vs "
      f"{final_theory:.5f}; late estimate a median {late_gap:.3f} from model)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 90)
gs = ms.grid(fig, 2, 1, left=16, right=5, top=7, bottom=10, hspace=7,
             height_ratios=[1.2, 1])
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1], sharex=ax_a)
CASE, MODEL, EST = ms.GREY_LIGHT, ms.INK, ms.VERMILLION

# a, epidemic curve: counts as bars, model expectation as a line
ax_a.bar(days, cases, width=1.0, align="edge", color=CASE, edgecolor="white",
         linewidth=0.25)
ax_a.plot(days + 0.5, expected, color=MODEL, lw=1.2)
top = 1.12 * cases.max()
ax_a.set_ylim(0, top)
ax_a.set_ylabel("Reported cases per day")
ax_a.text(T_INT - 1.5, 0.97 * top, f"Intervention, day {T_INT}",
          ha="right", va="top", fontsize=ms.FS_TICK)
ax_a.text(T_END - 1, 0.97 * top,
          f"Bars, reported cases (n = {cases.sum():,})\n"
          "Line, model expectation", ha="right", va="top",
          fontsize=ms.FS_TICK, linespacing=1.3)
ax_a.tick_params(labelbottom=False)

# b, reproduction number: model against the estimate from the bars above
ax_b.plot([0, T_END], [1, 1], color=ms.GREY, lw=0.6, ls=(0, (3, 2)))
ax_b.fill_between(ends + 1, rt_low, rt_high, color=EST, alpha=0.2, lw=0)
ax_b.plot(ends + 1, rt_est, "o", color=EST, ms=1.9, mew=0)
ax_b.plot(t, rt_model, color=MODEL, lw=1.2)
ax_b.set_xlim(0, T_END)
ax_b.set_ylim(0, 3.6)
ax_b.set_yticks(range(4))
ax_b.set_xticks(np.arange(0, T_END + 1, 10))
ax_b.set_xlabel("Time since first infections (days)")
ax_b.set_ylabel("Effective reproduction\nnumber $R_t$", fontsize=ms.FS_MATH)
ax_b.text(T_END - 1, 3.45, "Line, model R₀(t)·S(t)/N\n"
          f"Points, estimate from cases ({WINDOW}-day windows);\n"
          "band, 95% credible interval", ha="right", va="top",
          fontsize=ms.FS_TICK, linespacing=1.3)
ax_b.text(T_END - 1, 1.08, "Epidemic threshold", ha="right", va="bottom",
          fontsize=ms.FS_TICK, color=ms.GREY_DARK)
for ax in (ax_a, ax_b):
    ax.axvline(T_INT, color=ms.BLUE, lw=0.8)

ms.panel_label(ax_a, "a", dx_pt=-34)
ms.panel_label(ax_b, "b", dx_pt=-34)
ms.assert_aligned([ax_a, ax_b], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig176_epidemic_curve_rt.{ext}")
print("fig176_epidemic_curve_rt: saved png + pdf")
