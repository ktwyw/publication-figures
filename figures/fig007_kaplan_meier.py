"""Fig. 7 - Kaplan-Meier survival curves (single column).

Includes censoring tick marks, median-survival guides, a log-rank test
(scipy >= 1.11), and the number-at-risk table journals expect.
The KM estimator is implemented in ~10 lines - no lifelines needed.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(4)
FOLLOW_UP = 60.0  # months


# -------------------------------------------------------- ESTIMATOR ----
def km_curve(time, event):
    """Kaplan-Meier estimator -> (step times starting at 0, S(t))."""
    t = np.asarray(time, float)
    e = np.asarray(event, bool)
    ts, ss, s = [0.0], [1.0], 1.0
    for ti in np.unique(t[e]):
        n_at_risk = np.sum(t >= ti)
        d = np.sum((t == ti) & e)
        s *= 1.0 - d / n_at_risk
        ts.append(ti)
        ss.append(s)
    return np.array(ts), np.array(ss)


def surv_at(ts, ss, tq):
    """S(t) evaluated at arbitrary times tq (right-continuous step)."""
    return ss[np.searchsorted(ts, tq, side="right") - 1]


def median_survival(ts, ss):
    idx = np.where(ss <= 0.5)[0]
    return ts[idx[0]] if idx.size else np.nan


# ------------------------------------------------------------- DATA ----
def simulate_arm(scale, n=90):
    ev = rng.exponential(scale, n)          # true event times
    dropout = rng.uniform(10, FOLLOW_UP, n)  # random loss to follow-up
    obs = np.minimum(np.minimum(ev, dropout), FOLLOW_UP)
    event = ev <= np.minimum(dropout, FOLLOW_UP)
    return obs, event


arms = {
    "Control": (*simulate_arm(24), "C0"),
    "Treatment": (*simulate_arm(46), "C1"),
}

# log-rank test (scipy >= 1.11)
cens = [stats.CensoredData.right_censored(obs, ~ev)
        for obs, ev, _ in arms.values()]
p_lr = stats.logrank(*cens).pvalue
p_txt = "Log-rank $P$ < 0.001" if p_lr < 1e-3 else f"Log-rank $P$ = {p_lr:.3f}"

# ------------------------------------------------------------- PLOT ----
fig, (ax, ax_t) = plt.subplots(2, 1, figsize=(3.6, 3.4),
                               height_ratios=[3.2, 1.0])

medians = []
for label, (obs, ev, color) in arms.items():
    ts, ss = km_curve(obs, ev)
    ax.step(np.append(ts, FOLLOW_UP), np.append(ss, ss[-1]), where="post",
            color=color, lw=1.3, label=f"{label} ($n$ = {obs.size})")
    censored = obs[~ev]
    ax.plot(censored, surv_at(ts, ss, censored), "|", color=color,
            ms=6, mew=1.0)
    medians.append((median_survival(ts, ss), color))

# median-survival guides
valid = [(m, c) for m, c in medians if np.isfinite(m)]
if valid:
    ax.plot([0, max(m for m, _ in valid)], [0.5, 0.5],
            ls=":", lw=0.8, color="0.5")
    for m, color in valid:
        ax.plot([m, m], [0, 0.5], ls=":", lw=0.8, color=color)

ax.set_xlim(0, FOLLOW_UP + 2)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Time (months)")
ax.set_ylabel("Survival probability")
ax.legend(loc="upper right")
ax.text(0.03, 0.06, p_txt, transform=ax.transAxes, va="bottom")

# ------------------------------------------------- NUMBER-AT-RISK ----
ax_t.set_xlim(ax.get_xlim())
ax_t.set_ylim(0, 1.3)
ax_t.axis("off")
table_times = np.arange(0, FOLLOW_UP + 1, 12)
ax_t.text(-5, 1.05, "No. at risk", ha="right", va="center", fontsize=6.5)
for row_y, (label, (obs, ev, color)) in zip((0.62, 0.20), arms.items()):
    ax_t.text(-5, row_y, label, ha="right", va="center",
              fontsize=6.5, color=color)
    for tq in table_times:
        ax_t.text(tq, row_y, str(int(np.sum(obs >= tq))),
                  ha="center", va="center", fontsize=6.5)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig007_kaplan_meier.{ext}")
print("saved fig007_kaplan_meier.png / .pdf")
