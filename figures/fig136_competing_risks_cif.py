"""Fig. 136 - Cumulative incidence with competing risks (single column, 89 mm).

When death without relapse removes patients from the risk of relapse,
1 − Kaplan–Meier treats those deaths as censored and overstates how
many relapse. The Aalen–Johansen estimator, coded here by hand, weights
each cause-specific increment by all-cause survival instead. Relapse is
drawn solid and the competing event dashed in the same group colour;
the naive 1 − KM curve for one group is the thin dotted line above its
solid counterpart. The self-check is the identity CIF(relapse) +
CIF(death) + S(all causes) = 1 at every event time to 1e-12 in each
group, and that naive 1 − KM is never below the cumulative incidence.

Statistics: n = 220 patients per group; lines, Aalen–Johansen
cumulative incidence; uniform administrative censoring from 20 months,
all follow-up ends at 48; no interval and no test shown. All data are
simulated.
"""

import numpy as np
from matplotlib.lines import Line2D

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(136)
N, HORIZON = 220, 48.0                       # per group; months
RISK_TIMES = np.arange(0, 49, 12)
GROUPS = {"Standard": ms.GREY_DARK, "Intensive": ms.VERMILLION}
STYLE = {1: "-", 2: (0, (3.5, 1.8))}         # cause 1 relapse, 2 death

# ------------------------------------------------------------- DATA ----
# Latent cause-specific times: Weibull relapse (shape 0.9) and
# exponential death without relapse; the first of the two is observed.
# Intensive therapy prevents relapses at the price of more early deaths.
RELAPSE_SCALE = {"Standard": 45.0, "Intensive": 120.0}    # months
DEATH_MEAN = {"Standard": 110.0, "Intensive": 80.0}
data = {}
for group in GROUPS:
    t_relapse = RELAPSE_SCALE[group] * rng.weibull(0.9, N)
    t_death = rng.exponential(DEATH_MEAN[group], N)
    t_censor = np.minimum(rng.uniform(20, 80, N), HORIZON)
    time = np.minimum(np.minimum(t_relapse, t_death), t_censor)
    cause = np.where(time == t_censor, 0, np.where(time == t_relapse, 1, 2))
    data[group] = (time, cause)              # cause 0 = censored


# -------------------------------------------------------- ESTIMATORS ---
def aalen_johansen(time, cause):
    """Event times, CIF of causes 1 and 2, and all-cause survival."""
    t_event = np.unique(time[cause > 0])
    at_risk = np.array([(time >= t).sum() for t in t_event])
    d = np.array([[((time == t) & (cause == k)).sum() for t in t_event]
                  for k in (1, 2)])
    surv = np.cumprod(1 - d.sum(axis=0) / at_risk)
    surv_before = np.concatenate([[1.0], surv[:-1]])       # S(t−)
    cif = np.cumsum(surv_before * d / at_risk, axis=1)
    return t_event, cif[0], cif[1], surv


def one_minus_km(time, event, t_grid):
    """Naive 1 − KM for one cause, the other cause censored, on t_grid."""
    t_event = np.unique(time[event])
    at_risk = np.array([(time >= t).sum() for t in t_event])
    deaths = np.array([((time == t) & event).sum() for t in t_event])
    surv = np.concatenate([[1.0], np.cumprod(1 - deaths / at_risk)])
    return 1 - surv[np.searchsorted(t_event, t_grid, side="right")]


curves = {group: aalen_johansen(*data[group]) for group in GROUPS}
NAIVE = "Standard"
t_std, cif_std = curves[NAIVE][0], curves[NAIVE][1]
naive = one_minus_km(data[NAIVE][0], data[NAIVE][1] == 1, t_std)

# ------------------------------------------------------- SELF-CHECK ---
worst = 0.0
for group, (t, cif1, cif2, surv) in curves.items():
    worst = max(worst, np.abs(cif1 + cif2 + surv - 1).max())
    assert np.all(np.diff(cif1) >= 0) and np.all(np.diff(cif2) >= 0)
    relapse_km = one_minus_km(data[group][0], data[group][1] == 1, t)
    assert np.all(relapse_km >= cif1 - 1e-12), group
assert worst < 1e-12, worst
assert naive[-1] > cif_std[-1]
print(f"fig136: self-check passed (CIF1 + CIF2 + S = 1 within {worst:.1e} "
      f"at every event time; naive 1-KM >= CIF everywhere; at "
      f"{HORIZON:.0f} months {NAIVE.lower()} relapse {naive[-1]:.3f} naive "
      f"vs {cif_std[-1]:.3f} Aalen-Johansen)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 82)
gs = ms.grid(fig, 2, 1, left=24, right=5, top=5, bottom=5, hspace=15,
             height_ratios=[5.2, 1])
ax = fig.add_subplot(gs[0])
ax_risk = fig.add_subplot(gs[1], sharex=ax)


def to_horizon(t, y):
    """Step coordinates from (0, 0) to the end of follow-up."""
    return (np.concatenate([[0.0], t, [HORIZON]]),
            np.concatenate([[0.0], y, [y[-1]]]))


for row, (group, colour) in enumerate(GROUPS.items()):
    t, cif1, cif2, _surv = curves[group]
    for k, cif in ((1, cif1), (2, cif2)):
        ax.step(*to_horizon(t, cif), where="post", color=colour,
                lw=1.3 if k == 1 else 1.0, ls=STYLE[k])
    for rt in RISK_TIMES:
        ax_risk.text(rt, row, str(int((data[group][0] >= rt).sum())),
                     ha="center", va="center", fontsize=ms.FS_TICK,
                     color=colour)
    ax_risk.text(-0.07, row, group, transform=ax_risk.get_yaxis_transform(),
                 ha="right", va="center", fontsize=ms.FS_TICK, color=colour)

# the naive curve, labelled where it has pulled clear of the solid line
ax.step(*to_horizon(t_std, naive), where="post", color=ms.GREY, lw=0.8,
        ls=(0, (1, 1.4)))
ax.text(HORIZON, naive[-1] + 0.022,
        f"Naive 1 − KM, {NAIVE.lower()}: {naive[-1]:.2f}\n"
        f"(cumulative incidence {cif_std[-1]:.2f})", ha="right", va="bottom",
        fontsize=ms.FS_TICK, color=ms.GREY_DARK, linespacing=1.25)

ax.set_xlim(-1.0, HORIZON + 1.0)
ax.set_ylim(0, 0.8)
ax.set_xticks(RISK_TIMES)
ax.set_yticks(np.arange(0, 0.81, 0.2))
ax.set_xlabel("Time since randomization (months)")
ax.set_ylabel("Cumulative incidence")
handles = [Line2D([], [], color=colour, lw=1.3, label=group)
           for group, colour in GROUPS.items()]
handles += [Line2D([], [], color=ms.INK, lw=1.0, ls=STYLE[k], label=label)
            for k, label in ((1, "Relapse"), (2, "Death without relapse"))]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 1.0),
          handlelength=2.2)

ax_risk.set_ylim(1.6, -0.6)
ax_risk.axis("off")
ax_risk.text(-0.07, -1.05, "No. at risk",
             transform=ax_risk.get_yaxis_transform(), ha="right",
             va="center", fontsize=ms.FS_TICK, fontweight="bold")

ms.assert_aligned([ax, ax_risk], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig136_competing_risks_cif.{ext}")
print("fig136_competing_risks_cif: saved png + pdf")
