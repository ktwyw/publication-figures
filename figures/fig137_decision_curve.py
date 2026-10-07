"""Fig. 137 - Decision curves for two risk models (single column, 89 mm).

Discrimination says whether a model ranks patients; a decision curve
says whether acting on it helps. Net benefit, true positives minus
false positives weighted by the odds of the threshold probability, is
drawn for a basic and an extended logistic model against the two
default strategies, treat all and treat none. The extended model adds a
marker panel and is over-fitted, so it wins only over a middle range of
thresholds; that range is shaded, and a second axis translates each
threshold into the number of false positives one is prepared to accept
per true positive. The self-check
is that the empirical treat-all curve equals π − (1 − π)·p/(1 − p),
crosses zero exactly at the prevalence π, and that both model curves
tend to π as the threshold tends to 0.

Statistics: models fitted by maximum likelihood in a development set
(n = 200), net benefit evaluated in a separate validation set
(n = 2,000); no interval and no test shown. All data are simulated.
"""

import numpy as np
from scipy import optimize

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(137)
THRESHOLDS = np.linspace(0.01, 0.60, 119)        # step 0.005
ODDS_TICKS = np.array([19, 9, 4, 2, 1])          # false per true positive

# ------------------------------------------------------------- DATA ----
# Standardised predictors: a routine score, one informative marker and
# 12 markers that carry nothing. The basic model uses the score, the
# extended one the score plus the whole panel, from only 200 patients.
N_NULL = 12
TRUE_BETA = np.r_[-1.9, 0.8, 1.0, np.zeros(N_NULL)]   # intercept first


def simulate(n):
    design = np.column_stack([np.ones(n),
                              rng.normal(size=(n, TRUE_BETA.size - 1))])
    return design, rng.random(n) < 1 / (1 + np.exp(-design @ TRUE_BETA))


x_dev, y_dev = simulate(200)
x_val, y_val = simulate(2000)


# -------------------------------------------------------- ESTIMATORS ---
def fit_logistic(design, outcome, iterations=25):
    """Maximum-likelihood coefficients by Newton–Raphson (IRLS)."""
    beta = np.zeros(design.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-design @ beta))
        hessian = design.T @ (design * (p * (1 - p))[:, None])
        beta += np.linalg.solve(hessian, design.T @ (outcome - p))
    return beta


def net_benefit(risk, outcome, threshold):
    """TP/n − FP/n × p/(1 − p) when all with risk ≥ p are treated."""
    treated = risk[:, None] >= np.atleast_1d(threshold)[None, :]
    true_pos = (treated & outcome[:, None]).mean(axis=0)
    false_pos = (treated & ~outcome[:, None]).mean(axis=0)
    return true_pos - false_pos * threshold / (1 - threshold)


MODELS = {"Basic model": slice(0, 2), "Extended model": slice(None)}
risk = {}
for name, columns in MODELS.items():
    beta = fit_logistic(x_dev[:, columns], y_dev)
    risk[name] = 1 / (1 + np.exp(-x_val[:, columns] @ beta))
curve = {name: net_benefit(r, y_val, THRESHOLDS) for name, r in risk.items()}
everyone = np.ones(y_val.size)                   # risk 1: always treated
treat_all = net_benefit(everyone, y_val, THRESHOLDS)
prevalence = y_val.mean()

# thresholds where the extended model beats basic, treat all and treat none
rivals = np.maximum(np.maximum(curve["Basic model"], treat_all), 0.0)
best = np.flatnonzero(curve["Extended model"] > rivals)
runs = np.split(best, np.flatnonzero(np.diff(best) > 1) + 1)
span = THRESHOLDS[max(runs, key=len)[[0, -1]]]   # longest unbroken run

# ------------------------------------------------------- SELF-CHECK ---
analytic = prevalence - (1 - prevalence) * THRESHOLDS / (1 - THRESHOLDS)
assert np.allclose(treat_all, analytic, rtol=0, atol=1e-12)
zero = optimize.brentq(lambda p: net_benefit(everyone, y_val, p)[0],
                       0.01, 0.9)
assert abs(zero - prevalence) < 1e-9, (zero, prevalence)
for name, r in risk.items():
    assert abs(net_benefit(r, y_val, 1e-9)[0] - prevalence) < 1e-6, name
gain = (curve["Extended model"] - rivals)[np.argmin(np.abs(THRESHOLDS - 0.2))]
print(f"fig137: self-check passed (treat-all matches the analytic curve and "
      f"crosses zero at prevalence {prevalence:.4f}; both models tend to it "
      f"as p -> 0; extended best for p {span[0]:.3f}-{span[1]:.3f}, "
      f"+{gain * 100:.1f} net true positives per 100 at p = 0.2)")

# ------------------------------------------------------------ FIGURE --
FLOOR = -0.02                                    # axis cut just below zero
fig = ms.figure(89, 70)
ax = fig.add_subplot(ms.grid(fig, 1, 1, left=16, right=5, top=5,
                             bottom=21)[0])

ax.axvspan(*span, color=ms.VERMILLION, alpha=0.09, lw=0, zorder=0)
ax.text(span.mean(), 0.965, "Extended model best\n"
        f"(thresholds {span[0]:.3f}–{span[1]:.3f})",
        transform=ax.get_xaxis_transform(), ha="center", va="top",
        fontsize=ms.FS_TICK, color=ms.VERMILLION, linespacing=1.25)
ax.axhline(0, color=ms.GREY, lw=0.8, label="Treat none")
# treat all from its equation, down to where it leaves the axis
odds_floor = (prevalence - FLOOR) / (1 - prevalence)
p_line = np.linspace(THRESHOLDS[0], odds_floor / (1 + odds_floor), 50)
ax.plot(p_line, prevalence - (1 - prevalence) * p_line / (1 - p_line),
        color=ms.GREY, lw=0.8, ls=(0, (4, 2)), label="Treat all")
for name, colour, width in (("Basic model", ms.GREY_DARK, 1.0),
                            ("Extended model", ms.VERMILLION, 1.5)):
    shown = np.where(curve[name] >= FLOOR, curve[name], np.nan)
    ax.plot(THRESHOLDS, shown, color=colour, lw=width, label=name)

ax.set_xlim(0, 0.6)
ax.set_ylim(FLOOR, 0.24)
ax.set_xticks(np.arange(0, 0.61, 0.1))
ax.set_yticks(np.arange(0, 0.25, 0.05))
ax.spines["left"].set_bounds(0, 0.24)
ax.set_xlabel("Threshold probability")
ax.set_ylabel("Net benefit")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="upper right",
          bbox_to_anchor=(1.0, 0.97), handlelength=1.0, borderaxespad=0)

# the same thresholds as odds: (1 − p)/p false positives per true positive
odds = ax.secondary_xaxis(-0.27)
odds.set_xticks(1 / (1 + ODDS_TICKS), [str(k) for k in ODDS_TICKS])
odds.set_xlabel("False positives accepted per true positive")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig137_decision_curve.{ext}")
print("fig137_decision_curve: saved png + pdf")
