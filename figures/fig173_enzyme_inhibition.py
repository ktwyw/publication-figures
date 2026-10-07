"""Fig. 173 - Competitive inhibition, global kinetic fit (1.5 column, 120 mm).

How to show an inhibition experiment so the mechanism can be read from
the plot: initial rate against substrate concentration at three
inhibitor levels, every replicate drawn beside its mean ± s.d., one
global least-squares fit with shared Vₘₐₓ, Kₘ and Kᵢ, and drop lines at
Vₘₐₓ/2 showing that the inhibitor moves the apparent Kₘ but not Vₘₐₓ;
beside it the Lineweaver–Burk transform of the same fit and data, whose
lines meet on the 1/v axis. The self-check is that every true parameter
lies inside its fitted 95% CI, that the three fitted double-reciprocal
lines share the intercept 1/Vₘₐₓ to 1e-9, and that the apparent-Kₘ
ratios read off the fitted curves equal 1 + [I]/Kᵢ.

Statistics: v = Vₘₐₓ[S]/(Kₘ(1 + [I]/Kᵢ) + [S]) with Vₘₐₓ = 100 µM min⁻¹,
Kₘ = 20 µM, Kᵢ = 5 µM; n = 3 replicates at each of 6 substrate and 3
inhibitor concentrations (54 rates), multiplicative log-normal error,
CV 8%; global fit to log v (scipy curve_fit); 95% CI = estimate ±
t₀.₉₇₅,₅₁ × s.e.; error bars, mean ± s.d. All data are simulated.
"""

import numpy as np
from scipy import optimize, stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(173)

# ------------------------------------------------------------- DATA ----
TRUE = dict(vmax=100.0, km=20.0, ki=5.0)      # µM min⁻¹, µM, µM
SUBSTRATE = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0])    # µM
INHIBITOR = (0.0, 5.0, 15.0)                                  # µM
N_REP, CV = 3, 0.08


def rate(s, i, vmax, km, ki):
    """Michaelis–Menten rate with a competitive inhibitor."""
    return vmax * s / (km * (1 + i / ki) + s)


s_all = np.tile(np.repeat(SUBSTRATE, N_REP), len(INHIBITOR))
i_all = np.repeat(INHIBITOR, SUBSTRATE.size * N_REP)
v_all = rate(s_all, i_all, **TRUE) * np.exp(rng.normal(0, CV, s_all.size))

# --------------------------------------------------------- ESTIMATOR ---
# one fit to all 54 rates; log v makes the constant-CV error additive
popt, pcov = optimize.curve_fit(
    lambda x, vmax, km, ki: np.log(rate(x[0], x[1], vmax, km, ki)),
    (s_all, i_all), np.log(v_all), p0=(80.0, 10.0, 10.0))
dof = v_all.size - popt.size
half_ci = stats.t.ppf(0.975, dof) * np.sqrt(np.diag(pcov))
fit = dict(zip(TRUE, popt))
ci = {name: (est - h, est + h) for name, est, h in zip(TRUE, popt, half_ci)}

# apparent Km read off each fitted curve: the [S] at which v = Vmax/2
km_app = [optimize.brentq(lambda s: rate(s, i, **fit) - fit["vmax"] / 2,
                          1e-6, 1e4, xtol=1e-13, rtol=1e-14)
          for i in INHIBITOR]
# double-reciprocal line through two points of each fitted curve
lb_lines = [np.polyfit(1 / SUBSTRATE[[0, -1]],
                       1 / rate(SUBSTRATE[[0, -1]], i, **fit), 1)
            for i in INHIBITOR]                # (slope, intercept)

# ------------------------------------------------------- SELF-CHECK ---
for name in TRUE:
    assert ci[name][0] < TRUE[name] < ci[name][1], (name, ci[name])
for (slope, intercept), i, k in zip(lb_lines, INHIBITOR, km_app):
    assert abs(intercept - 1 / fit["vmax"]) < 1e-9, intercept
    assert abs(k / km_app[0] - (1 + i / fit["ki"])) < 1e-9
    assert abs(slope - k / fit["vmax"]) < 1e-9
print("fig173: self-check passed (" + "; ".join(
    f"{name} = {fit[name]:.1f} [{ci[name][0]:.1f}, {ci[name][1]:.1f}] "
    f"covers {TRUE[name]:g}" for name in TRUE)
    + f"; Lineweaver-Burk intercepts all 1/Vmax = {1 / fit['vmax']:.5f}; "
    f"Km,app ratios {km_app[1] / km_app[0]:.3f}, "
    f"{km_app[2] / km_app[0]:.3f} = 1 + [I]/Ki)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 66)
ax_a = ms.axes(fig, 13, 11, 57, 48)
ax_b = ms.axes(fig, 86, 11, 30, 48)
STYLE = {0.0: (ms.GREY_DARK, "o"), 5.0: (ms.BLUE, "s"),
         15.0: (ms.VERMILLION, "^")}
S_MAX = SUBSTRATE[-1]
JITTER = np.tile([-2.5, 0.0, 2.5], SUBSTRATE.size)     # µM, replicates

# a, rate against substrate: replicates, mean ± s.d., global fit
s_line = np.linspace(0, S_MAX + 4, 400)
for i, k in zip(INHIBITOR, km_app):
    colour, marker = STYLE[i]
    sel = i_all == i
    v_rep = v_all[sel].reshape(SUBSTRATE.size, N_REP)
    ax_a.plot(s_line, rate(s_line, i, **fit), color=colour, lw=1.0, zorder=2)
    ax_a.plot(s_all[sel] + JITTER, v_all[sel], marker, ms=1.8, mfc="none",
              mec=colour, mew=0.4, ls="none", alpha=0.8, zorder=3)
    ax_a.errorbar(SUBSTRATE, v_rep.mean(axis=1),
                  yerr=v_rep.std(axis=1, ddof=1), fmt=marker, ms=3.2,
                  color=colour, mec="white", mew=0.4, elinewidth=0.6,
                  capsize=1.5, capthick=0.6, zorder=4)
    ax_a.plot([k, k], [0, fit["vmax"] / 2], color=colour, lw=0.6,
              ls=(0, (2, 1.5)), zorder=1)
    label = "[I] = 0" if i == 0 else f"{i:g} µM"
    ax_a.text(S_MAX + 9, rate(S_MAX + 4, i, **fit), label, color=colour,
              va="center", fontsize=ms.FS_TICK)
    ax_a.text(k + 2.5, 2.0, "Kₘ" if i == 0 else "Kₘ,app", color=colour,
              va="bottom", fontsize=ms.FS_SMALL)
ax_a.plot([0, km_app[-1]], [fit["vmax"] / 2] * 2, color=ms.GREY, lw=0.6,
          ls=(0, (2, 1.5)), zorder=1)
ax_a.plot([0, 203], [fit["vmax"]] * 2, color=ms.GREY, lw=0.6,
          ls=(0, (2, 1.5)), zorder=1)
ax_a.text(203, fit["vmax"] + 1.5, "Vₘₐₓ", ha="right", va="bottom",
          fontsize=ms.FS_TICK, color=ms.GREY_DARK)
ax_a.text(km_app[-1] + 4, fit["vmax"] / 2 - 1.0, "Vₘₐₓ/2", va="top",
          fontsize=ms.FS_TICK, color=ms.GREY_DARK)
rows = ["Global fit, estimate (95% CI)"] + [
    f"{symbol} = {fit[name]:.{d}f} ({ci[name][0]:.{d}f}–{ci[name][1]:.{d}f})"
    f" {unit}" for name, symbol, unit, d in (
        ("vmax", "Vₘₐₓ", "µM min⁻¹", 0), ("km", "Kₘ", "µM", 1),
        ("ki", "Kᵢ", "µM", 1))]
ax_a.text(203, 12, "\n".join(rows), ha="right", va="bottom",
          fontsize=ms.FS_SMALL, linespacing=1.35)
ax_a.set_xlim(0, 205)
ax_a.set_ylim(0, 112)
ax_a.set_xticks(np.arange(0, S_MAX + 1, 40))
ax_a.spines["bottom"].set_bounds(0, S_MAX)
ax_a.set_yticks(np.arange(0, 101, 25))
ax_a.spines["left"].set_bounds(0, 100)
ax_a.set_xlabel("Substrate concentration [S] (µM)", x=S_MAX / 2 / 205)
ax_a.set_ylabel("Initial rate v (µM min⁻¹)", y=50 / 112)

# b, the same fit and data as reciprocals; lines extend to −1/Km,app
X_LIM = (-0.065, 0.215)
y_top = 1.15 / v_all.min()
ax_b.plot([0, 0], [0, y_top], color=ms.GREY_LIGHT, lw=0.6, zorder=0)
for i, (slope, intercept) in zip(INHIBITOR, lb_lines):
    colour, marker = STYLE[i]
    sel = i_all == i
    x_zero = -intercept / slope
    ax_b.plot([0, X_LIM[1] - 0.008], intercept + slope
              * np.array([0, X_LIM[1] - 0.008]), color=colour, lw=1.0)
    ax_b.plot([x_zero, 0], [0, intercept], color=colour, lw=0.7,
              ls=(0, (2, 1.5)))
    ax_b.plot(1 / s_all[sel], 1 / v_all[sel], marker, ms=2.2, mfc="none",
              mec=colour, mew=0.45, ls="none", zorder=3)
ax_b.plot(0, 1 / fit["vmax"], "o", ms=3.4, mfc="white", mec=ms.INK,
          mew=0.7, zorder=5)
# key for the open circle, in the empty corner above the lines
ax_b.plot(0.014, 0.965 * y_top, "o", ms=3.4, mfc="white", mec=ms.INK,
          mew=0.7)
ax_b.text(0.025, 0.99 * y_top, "common intercept,\n1/v = 1/Vₘₐₓ",
          fontsize=ms.FS_SMALL, va="top", linespacing=1.3)
ax_b.set_xlim(*X_LIM)
ax_b.set_ylim(0, y_top)
ax_b.set_yticks(np.arange(0, y_top, 0.05))
ax_b.set_xticks([0, 0.1, 0.2])
ax_b.set_xlabel("1/[S] (µM⁻¹)")
ax_b.set_ylabel("1/v (min µM⁻¹)")

ms.panel_label(ax_a, "a", dx_pt=-30)
ms.panel_label(ax_b, "b", dx_pt=-30)
ms.assert_aligned([ax_a, ax_b])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig173_enzyme_inhibition.{ext}")
print("fig173_enzyme_inhibition: saved png + pdf")
