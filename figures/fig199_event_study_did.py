"""Fig. 199 - Difference-in-differences event study (double column, 183 mm).

The two pictures a difference-in-differences paper owes its readers:
(a) raw group means over event time, where a constant gap before the
treatment is the visible case for parallel trends, and (b) event-study
coefficients from a two-way fixed-effects regression with the period
before treatment as the reference, pre-treatment estimates in grey and
post-treatment estimates salient. The regression is solved with
np.linalg.lstsq and the unit-clustered covariance is coded by hand. The
self-check is that, the panel being balanced, each coefficient equals
the difference-in-differences of group means relative to period −1 to
1e-8, that the pre-period coefficients are jointly indistinguishable
from zero at the 1% level, and that the pooled ATT interval covers the
simulated effect.

Statistics: 40 units × 16 periods, 20 treated from event time 0; unit
and period effects, AR(1) noise within unit (ρ = 0.5). Bands in (a),
±1 s.e.m. over the 20 units of a group; bars in (b), 95% CI from
unit-clustered (sandwich) standard errors with a G/(G − 1) correction
and t(G − 1) quantiles; pre-trend test, clustered Wald statistic
referred to F(q, G − 1). All data are simulated.
"""

import numpy as np
from scipy import stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(199)

# ------------------------------------------------------------- DATA ----
N_UNITS, EVENT = 40, np.arange(-8, 8)        # event time; treatment at 0
REFERENCE = -1                               # omitted period
GAP, RHO, NOISE_SD = 1.5, 0.5, 0.6           # level gap; AR(1) noise
TRUE_EFFECT = np.where(EVENT >= 0, 2.0 * (1 - np.exp(-(EVENT + 1) / 1.5)), 0)

treated = np.arange(N_UNITS) < N_UNITS // 2
unit_effect = rng.normal(0, 1.0, N_UNITS) + GAP * treated
period_effect = 0.12 * EVENT + 0.5 * np.sin(EVENT / 2.0)   # common shocks
noise = np.empty((N_UNITS, EVENT.size))
noise[:, 0] = rng.normal(0, NOISE_SD, N_UNITS)
for t in range(1, EVENT.size):               # stationary AR(1) within unit
    noise[:, t] = RHO * noise[:, t - 1] + rng.normal(
        0, NOISE_SD * np.sqrt(1 - RHO ** 2), N_UNITS)
y = (unit_effect[:, None] + period_effect[None, :]
     + treated[:, None] * TRUE_EFFECT[None, :] + noise)    # units × periods


# -------------------------------------------------------- ESTIMATORS ---
def twfe(interactions):
    """Two-way fixed effects by least squares, unit-clustered covariance.

    interactions: (units, periods, k) regressors of interest. Returns
    their coefficients and the k × k cluster-robust covariance.
    """
    n, t = y.shape
    unit_dummies = np.kron(np.eye(n), np.ones((t, 1)))
    period_dummies = np.kron(np.ones((n, 1)), np.eye(t))[:, 1:]
    k = interactions.shape[2]
    design = np.hstack([interactions.reshape(n * t, k), unit_dummies,
                        period_dummies])
    beta = np.linalg.lstsq(design, y.ravel(), rcond=None)[0]
    residual = (y.ravel() - design @ beta).reshape(n, t)
    bread = np.linalg.inv(design.T @ design)
    scores = np.einsum("gt,gtk->gk", residual,
                       design.reshape(n, t, -1))           # one row per unit
    covariance = n / (n - 1) * bread @ (scores.T @ scores) @ bread
    return beta[:k], covariance[:k, :k]


leads_lags = EVENT[EVENT != REFERENCE]
dummies = (treated[:, None, None]
           * (EVENT[None, :, None] == leads_lags[None, None, :])).astype(float)
coef, cov = twfe(dummies)
se = np.sqrt(np.diag(cov))
T_CRIT = stats.t.ppf(0.975, N_UNITS - 1)

post = (treated[:, None] * (EVENT[None, :] >= 0)).astype(float)[:, :, None]
att, att_var = twfe(post)
att, att_se = att[0], np.sqrt(att_var[0, 0])
att_ci = att - T_CRIT * att_se, att + T_CRIT * att_se
att_true = TRUE_EFFECT[EVENT >= 0].mean()

pre = leads_lags < REFERENCE                 # joint test of the leads
wald = coef[pre] @ np.linalg.solve(cov[np.ix_(pre, pre)], coef[pre])
f_stat = wald / pre.sum()
p_pre = stats.f.sf(f_stat, pre.sum(), N_UNITS - 1)

mean = {g: y[rows].mean(axis=0) for g, rows in (("treated", treated),
                                                ("control", ~treated))}
sem = {g: y[rows].std(axis=0, ddof=1) / np.sqrt(rows.sum())
       for g, rows in (("treated", treated), ("control", ~treated))}

# ------------------------------------------------------- SELF-CHECK ---
gap = mean["treated"] - mean["control"]
did = (gap - gap[EVENT == REFERENCE])[EVENT != REFERENCE]
assert np.abs(coef - did).max() < 1e-8, np.abs(coef - did).max()
assert p_pre > 0.01, p_pre                   # leads jointly small
assert att_ci[0] < att_true < att_ci[1], (att_ci, att_true)
assert abs(att - coef[leads_lags >= 0].mean()
           + coef[pre].sum() / (pre.sum() + 1)) < 1e-8     # post − pre mean
print(f"fig199: self-check passed (TWFE = DiD of means within "
      f"{np.abs(coef - did).max():.1e}; pre-trend F({pre.sum()}, "
      f"{N_UNITS - 1}) = {f_stat:.2f}, {ms.format_p(p_pre)}; pooled ATT "
      f"{att:.2f}, 95% CI {att_ci[0]:.2f} to {att_ci[1]:.2f}, simulated "
      f"{att_true:.2f})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 74)
gs = ms.grid(fig, 1, 2, left=13, right=4, top=7, bottom=11, wspace=17)
ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
ONSET = -0.5                                 # between periods −1 and 0
COLOUR = {"treated": ms.VERMILLION, "control": ms.GREY_DARK}

# a, raw means: the gap is constant until the treatment starts
for group, colour in COLOUR.items():
    ax_a.fill_between(EVENT, mean[group] - sem[group],
                      mean[group] + sem[group], color=colour, alpha=0.16,
                      lw=0)
    ax_a.plot(EVENT, mean[group], "-o", color=colour, lw=1.1, ms=2.6)
pre_gap = gap[EVENT < 0].mean()
after = EVENT >= REFERENCE
ax_a.plot(EVENT[after], mean["control"][after] + pre_gap, color=ms.GREY,
          lw=0.7, ls=(0, (1, 1.6)))
y_low = (mean["control"] - sem["control"]).min() - 0.6
y_high = (mean["treated"] + sem["treated"]).max() + 0.3
for ax, (low, high) in ((ax_a, (y_low, y_high)), (ax_b, (-1.3, 3.3))):
    ax.plot([ONSET, ONSET], [low, high], color=ms.INK, lw=0.6,
            ls=(0, (4, 2)))
    ax.set_ylim(low, high)
    ax.set_xlim(EVENT[0] - 0.6, EVENT[-1] + 0.6)
    ax.set_xticks(EVENT[::2])
    ax.set_xlabel("Event time (periods since treatment began)")
x_lab = -7.8
for group, side in (("treated", 1), ("control", -1)):
    edge = (mean[group] + side * sem[group])[EVENT < REFERENCE]
    ax_a.text(x_lab - 0.2, side * (side * edge).max() + side * 0.1,
              f"{group.capitalize()} (n = {N_UNITS // 2} units)",
              va="bottom" if side > 0 else "top", color=COLOUR[group],
              fontweight="bold")
ax_a.text(x_lab - 0.2, y_high, "Bands, ±1 s.e.m. over units\n"
          "Dotted, control + mean\npre-period gap", va="top",
          fontsize=ms.FS_SMALL, color=ms.GREY_DARK, linespacing=1.3)
ax_a.text(ONSET + 0.25, y_low + 0.05, "Treatment begins", va="bottom",
          fontsize=ms.FS_SMALL)
ax_a.set_ylabel("Outcome, group mean (a.u.)")

# b, event-study coefficients relative to the period before treatment
ax_b.plot([EVENT[0] - 0.6, ONSET], [0, 0], color=ms.GREY, lw=0.6)
ax_b.plot([ONSET, EVENT[-1] + 0.6], [0, 0], color=ms.GREY, lw=0.6)
ax_b.plot(EVENT[EVENT >= 0], TRUE_EFFECT[EVENT >= 0], color=ms.GREY,
          lw=0.7, ls=(0, (1, 1.6)))
for chosen, colour in ((leads_lags < 0, ms.GREY_DARK),
                       (leads_lags >= 0, ms.VERMILLION)):
    ax_b.errorbar(leads_lags[chosen], coef[chosen],
                  yerr=T_CRIT * se[chosen], fmt="o", color=colour, ms=3,
                  lw=0.8, capsize=1.5, capthick=0.8)
ax_b.plot(REFERENCE, 0, "o", ms=3.4, mfc="white", mec=ms.GREY_DARK, mew=0.8,
          zorder=3)
ax_b.annotate(f"Pooled post-period ATT {att:.2f} (95% CI {att_ci[0]:.2f} "
              f"to {att_ci[1]:.2f}); simulated {att_true:.2f}", xy=(0, 1),
              xycoords="axes fraction", xytext=(0, 3),
              textcoords="offset points", va="bottom", color=ms.VERMILLION,
              fontsize=ms.FS_TICK)
ax_b.text(x_lab, 3.3, f"Pre-trend test of the {pre.sum()} leads:\n"
          f"F({pre.sum()}, {N_UNITS - 1}) = {f_stat:.2f}, "
          f"{ms.format_p(p_pre)}", va="top", fontsize=ms.FS_SMALL,
          color=ms.GREY_DARK, linespacing=1.3)
ax_b.text(7.4, -1.22, "Bars, 95% CI (unit-clustered s.e.)\n"
          "Open point, reference period −1\n"
          "Dotted, simulated effect by period", va="bottom", ha="right",
          fontsize=ms.FS_SMALL, color=ms.GREY_DARK, linespacing=1.3)
ax_b.set_ylabel("Effect relative to period −1 (a.u.)")

ms.panel_label(ax_a, "a")
ms.panel_label(ax_b, "b")
ms.assert_aligned([ax_a, ax_b])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig199_event_study_did.{ext}")
print("fig199_event_study_did: saved png + pdf")
