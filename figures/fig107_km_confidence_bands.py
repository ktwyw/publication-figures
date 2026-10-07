"""Fig. 107 - Kaplan-Meier curves with confidence bands (single column, 89 mm).

Extends fig007 with what a clinical journal asks for next: the 95%
Greenwood band on the log(-log S) scale (it cannot leave [0, 1]), a
number-at-risk table whose columns sit on the tick positions, and a
log-rank test coded from the hypergeometric variance. The self-check is
that the hand-coded test reproduces scipy.stats.logrank (scipy >= 1.11).

Statistics: n = 130 patients per arm; ticks, censored observations;
two-sided log-rank test. All data are simulated.
"""

import numpy as np
from scipy import stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(707)
HORIZON = 36.0                               # months of follow-up
RISK_TIMES = np.arange(0, 37, 6)
ARMS = {"Control": ms.GREY_DARK, "Experimental": ms.VERMILLION}

# ------------------------------------------------------------- DATA ----
# Weibull event times (shape 1.25) with uniform administrative censoring.
MEDIAN = {"Control": 11.0, "Experimental": 19.0}
data = {}
for arm in ARMS:
    event_time = rng.weibull(1.25, 130) * MEDIAN[arm] / np.log(2) ** (1 / 1.25)
    censor_time = np.minimum(rng.uniform(14, 60, 130), HORIZON)
    data[arm] = (np.minimum(event_time, censor_time),
                 event_time <= censor_time)             # (time, event)


# -------------------------------------------------------- ESTIMATORS ---
def kaplan_meier(time, event):
    """Step times, S(t) and the 95% log-log (Greenwood) limits."""
    t_event = np.unique(time[event])
    at_risk = np.array([(time >= t).sum() for t in t_event])
    deaths = np.array([((time == t) & event).sum() for t in t_event])
    s = np.cumprod(1 - deaths / at_risk)
    greenwood = np.cumsum(deaths / (at_risk * np.maximum(at_risk - deaths, 1)))
    se = np.sqrt(greenwood) / np.abs(np.log(s))        # s.e. of log(-log S)
    low, high = s ** np.exp(1.96 * se), s ** np.exp(-1.96 * se)
    one = np.ones(1)
    return (np.concatenate([[0.0], t_event]), np.concatenate([one, s]),
            np.concatenate([one, low]), np.concatenate([one, high]))


def logrank(time_a, event_a, time_b, event_b):
    """Two-sample log-rank chi-square (1 d.f.) and its P value."""
    time = np.concatenate([time_a, time_b])
    event = np.concatenate([event_a, event_b])
    in_b = np.arange(time.size) >= time_a.size
    observed = expected = variance = 0.0
    for t in np.unique(time[event]):
        risk = time >= t
        n, n_b = risk.sum(), (risk & in_b).sum()
        d = ((time == t) & event).sum()
        observed += ((time == t) & event & in_b).sum()
        expected += d * n_b / n
        if n > 1:
            variance += d * (n_b / n) * (1 - n_b / n) * (n - d) / (n - 1)
    chi2 = (observed - expected) ** 2 / variance
    return chi2, stats.chi2.sf(chi2, 1)


chi2, p_value = logrank(*data["Control"], *data["Experimental"])
curves = {arm: kaplan_meier(*data[arm]) for arm in ARMS}
medians = {arm: float(t[np.argmax(s <= 0.5)])
           for arm, (t, s, _lo, _hi) in curves.items()}

# ------------------------------------------------------- SELF-CHECK ---
reference = stats.logrank(
    *[stats.CensoredData.right_censored(time, ~event)
      for time, event in data.values()]).pvalue
assert abs(p_value - reference) <= 1e-6 * reference, (p_value, reference)
for t, s, low, high in curves.values():
    assert np.all(np.diff(s) <= 0) and np.all((low <= s) & (s <= high))
print(f"fig107: self-check passed (log-rank chi2 = {chi2:.2f}, "
      f"P = {p_value:.3g} matches scipy; bands bracket S(t))")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 80)
gs = ms.grid(fig, 2, 1, left=24, right=5, top=5, bottom=5, hspace=15,
             height_ratios=[5.2, 1])
ax = fig.add_subplot(gs[0])
ax_risk = fig.add_subplot(gs[1], sharex=ax)

for row, (arm, colour) in enumerate(ARMS.items()):
    time, event = data[arm]
    t, s, low, high = curves[arm]
    t_end = np.append(t, HORIZON)
    ax.fill_between(t_end, np.append(low, low[-1]), np.append(high, high[-1]),
                    step="post", color=colour, alpha=0.15, lw=0)
    ax.step(t_end, np.append(s, s[-1]), where="post", color=colour, lw=1.2,
            label=arm)
    censored = time[~event & (time < HORIZON)]
    ax.scatter(censored, s[np.searchsorted(t, censored, side="right") - 1],
               marker="|", s=9, color=colour, linewidths=0.6, zorder=3)
    for rt in RISK_TIMES:
        ax_risk.text(rt, row, str(int((time >= rt).sum())), ha="center",
                     va="center", fontsize=ms.FS_TICK, color=colour)
    ax_risk.text(-0.03, row, arm, transform=ax_risk.get_yaxis_transform(),
                 ha="right", va="center", fontsize=ms.FS_TICK, color=colour)

ax.set_xlim(-0.8, HORIZON + 0.8)
ax.set_ylim(0, 1.03)
ax.set_xticks(RISK_TIMES)
ax.set_yticks(np.linspace(0, 1, 6))
ax.spines["left"].set_bounds(0, 1)
ax.set_xlabel("Time since randomization (months)")
ax.set_ylabel("Progression-free survival probability")
ax.legend(loc="upper right", handlelength=1.6)
ax.text(0.97, 0.76, f"Log-rank {ms.format_p(p_value)}\n"
        f"Median {medians['Control']:.1f} vs "
        f"{medians['Experimental']:.1f} months", transform=ax.transAxes,
        ha="right", va="top", fontsize=ms.FS_TICK)

ax_risk.set_ylim(1.6, -0.6)
ax_risk.axis("off")
ax_risk.text(-0.03, -1.05, "No. at risk",
             transform=ax_risk.get_yaxis_transform(), ha="right",
             va="center", fontsize=ms.FS_TICK, fontweight="bold")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig107_km_confidence_bands.{ext}")
print("fig107_km_confidence_bands: saved png + pdf")
