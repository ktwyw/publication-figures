"""Fig. 193 - Fatigue S-N curve and Haigh diagram (double column, 183 mm).

From coupon tests to a design check: (a) lives at eight stress
amplitudes on log-log axes, run-outs as open symbols with arrows, the
Basquin line regressed as log N on log S through the failures only
(life, not stress, carries the scatter) and its ±2 s.d. band;
(b) the endurance limit carried into a constant-life (Haigh) diagram
with the Goodman, Gerber and Soderberg mean-stress criteria and the
first-cycle yield (Langer) line, and three load cases with their
Goodman safety factors. The self-check is that the fitted slope and
intercept recover the simulated truth within 3 standard errors, that
all three criteria pass through (0, Se) and their own intercepts on
the mean-stress axis, and that each safety factor n satisfies
σa/Se + σm/Sut = 1/n to 1e-9.

Statistics: n = 5 specimens at each of 8 amplitudes (the lowest lies
below the 10⁷-cycle strength, so it yields the run-outs); life is
log-normal about S = A·N^b (A = 1,300 MPa, b = −0.10, s.d. 0.15 in
log₁₀N); tests stop at 10⁷ cycles; ordinary least squares on failures;
band, ±2 residual s.d. in log₁₀N; Se is the fitted strength at 10⁷
cycles. All data are simulated.
"""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy import stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(193)
SUPER = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

# ------------------------------------------------------------- DATA ----
A_TRUE, B_TRUE = 1300.0, -0.10          # Basquin S = A·N^b (MPa, -)
SD_LOG_N = 0.15                         # scatter in log₁₀ N
N_LIMIT = 1e7                           # run-out (test stopped) cycle count
# seven levels that all fail and one below the 10⁷-cycle strength (MPa)
AMPLITUDES = np.append(np.round(np.geomspace(560, 285, 7)), 240.0)
PER_LEVEL = 5
S_UT, S_Y = 560.0, 420.0                # ultimate and yield strength (MPa)
LOAD_CASES = {"A": (80.0, 110.0), "B": (240.0, 70.0), "C": (140.0, 215.0)}

stress = np.repeat(AMPLITUDES, PER_LEVEL)
log_life = (np.log10(stress / A_TRUE) / B_TRUE
            + rng.normal(0, SD_LOG_N, stress.size))
failed = log_life < np.log10(N_LIMIT)
cycles = np.where(failed, 10 ** log_life, N_LIMIT)

# -------------------------------------------------------- ESTIMATORS ---
# log₁₀N = c0 + c1·log₁₀S on failures; Basquin b = 1/c1, A = 10^(−c0/c1)
fit = stats.linregress(np.log10(stress[failed]), np.log10(cycles[failed]))
c1_true, c0_true = 1 / B_TRUE, -np.log10(A_TRUE) / B_TRUE
b_fit, a_fit = 1 / fit.slope, 10 ** (-fit.intercept / fit.slope)
b_se = fit.stderr / fit.slope ** 2                       # delta method
resid = (np.log10(cycles[failed]) - fit.intercept
         - fit.slope * np.log10(stress[failed]))
s_resid = np.sqrt(np.sum(resid ** 2) / (failed.sum() - 2))
s_e = a_fit * N_LIMIT ** b_fit          # endurance limit: strength at 10⁷


def life(s):
    return 10 ** (fit.intercept + fit.slope * np.log10(s))


def goodman(sm):
    return s_e * (1 - sm / S_UT)


def gerber(sm):
    return s_e * (1 - (sm / S_UT) ** 2)


def soderberg(sm):
    return s_e * (1 - sm / S_Y)


def langer(sm):
    return S_Y - sm


safety = {k: 1 / (sa / s_e + sm / S_UT) for k, (sm, sa) in LOAD_CASES.items()}

# ------------------------------------------------------- SELF-CHECK ---
z_slope = (fit.slope - c1_true) / fit.stderr
z_icpt = (fit.intercept - c0_true) / fit.intercept_stderr
assert abs(z_slope) < 3 and abs(z_icpt) < 3, (z_slope, z_icpt)
for line, intercept in ((goodman, S_UT), (gerber, S_UT), (soderberg, S_Y)):
    assert abs(line(0.0) - s_e) < 1e-9 and abs(line(intercept)) < 1e-9
assert abs(langer(S_Y)) < 1e-12 and langer(0.0) == S_Y
for key, (sm, sa) in LOAD_CASES.items():
    assert abs(sa / s_e + sm / S_UT - 1 / safety[key]) < 1e-9
    n = safety[key]                 # scaling the load by n lands on the line
    assert abs(goodman(n * sm) - n * sa) < 1e-9
print(f"fig193: self-check passed (b = {b_fit:.4f} ± {b_se:.4f} vs "
      f"{B_TRUE}; A = {a_fit:.0f} vs {A_TRUE:.0f} MPa; slope z = "
      f"{z_slope:+.2f}, intercept z = {z_icpt:+.2f}; {failed.sum()} failures,"
      f" {(~failed).sum()} run-outs; Se = {s_e:.1f} MPa; Goodman n = "
      + ", ".join(f"{k} {v:.2f}" for k, v in safety.items()) + ")")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 78)
gs = ms.grid(fig, 1, 2, left=15, right=6, top=7, bottom=11, wspace=21)
ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
DASH = (0, (3.5, 2))

# a, S-N diagram: stress on the ordinate by convention, life the response
s_line = np.geomspace(s_e, 600, 100)
ax_a.fill_betweenx(s_line, life(s_line) * 10 ** (-2 * s_resid),
                   life(s_line) * 10 ** (2 * s_resid), color=ms.BLUE,
                   alpha=0.15, lw=0)
ax_a.plot(life(s_line), s_line, color=ms.BLUE, lw=1.4)
ax_a.plot([1e3, 4e7], [s_e, s_e], color=ms.BLUE, lw=0.8, ls=DASH)
ax_a.plot(cycles[failed], stress[failed], "o", ms=3.2, color=ms.INK, mew=0,
          alpha=0.8)
for level in np.unique(stress[~failed]):
    count = int(np.sum(~failed & (stress == level)))
    ax_a.plot(N_LIMIT, level, "o", ms=3.4, mfc="white", mec=ms.INK, mew=0.7,
              zorder=4)
    ax_a.annotate("", xy=(2.1e7, level), xytext=(1.12e7, level),
                  arrowprops=dict(arrowstyle="-|>", color=ms.INK, lw=0.6,
                                  mutation_scale=5, shrinkA=0, shrinkB=0))
    ax_a.text(2.35e7, level, f"×{count}", va="center", fontsize=ms.FS_SMALL)
ax_a.text(0.90, 0.905, "Basquin fit, S = A·Nᵇ\n"
          f"b = −{-b_fit:.3f} ± {b_se:.3f} (s.e.)\nA = {a_fit:,.0f} MPa\n"
          f"band, ±2 s.d. of log₁₀N (s = {s_resid:.2f})",
          transform=ax_a.transAxes, ha="right", va="top",
          fontsize=ms.FS_TICK, linespacing=1.3)
ax_a.text(2.2e3, s_e * 1.015, f"Endurance limit Sₑ = {s_e:.0f} MPa",
          va="bottom", fontsize=ms.FS_TICK, color=ms.BLUE)
ax_a.text(0.03, 0.04, f"n = {PER_LEVEL} specimens per level; filled, failure "
          f"({failed.sum()});\nopen with arrow, run-out at 10⁷ cycles "
          f"({(~failed).sum()}); Sₑ, fit at 10⁷ cycles",
          transform=ax_a.transAxes, va="bottom", fontsize=ms.FS_SMALL,
          color=ms.GREY_DARK, linespacing=1.3)
ax_a.set_xscale("log")
ax_a.set_yscale("log")
ax_a.set_xlim(1e3, 4e7)
ax_a.set_ylim(200, 650)
ax_a.set_yticks([200, 300, 400, 500, 600])
ms.plain_log_ticks(ax_a.yaxis)
# decades with Unicode exponents: mathtext superscripts would fall below 5 pt
ax_a.xaxis.set_major_formatter(FuncFormatter(
    lambda value, _pos: "10" + f"{np.log10(value):.0f}".translate(SUPER)))
ax_a.set_xlabel("Cycles to failure N")
ax_a.set_ylabel("Stress amplitude S (MPa)")

# b, Haigh diagram at the 10⁷-cycle life
SM_MAX, SA_MAX = 620.0, 330.0
sm = np.linspace(0, S_UT, 300)
sm_y = np.linspace(S_Y - SA_MAX, S_Y, 50)       # yield line inside the frame
ax_b.fill_between(sm, np.clip(np.minimum(goodman(sm), langer(sm)), 0, None),
                  color=ms.BLUE, alpha=0.12, lw=0)
ax_b.plot(sm_y, langer(sm_y), color=ms.GREY, lw=0.9, ls=DASH)
ax_b.plot(sm[sm <= S_Y], soderberg(sm[sm <= S_Y]), color=ms.GREEN, lw=1.0)
ax_b.plot(sm, gerber(sm), color=ms.ORANGE, lw=1.0)
ax_b.plot(sm, goodman(sm), color=ms.BLUE, lw=1.6)
for key, (mean, amp) in LOAD_CASES.items():
    ax_b.plot(mean, amp, "o", ms=4, color=ms.INK, mew=0, zorder=5)
    ax_b.annotate(key, xy=(mean, amp), xytext=(4.5, 0),
                  textcoords="offset points", ha="left", va="center",
                  fontsize=ms.FS_TICK, fontweight="bold")
handles = [Line2D([], [], color=ms.ORANGE, lw=1.0, label="Gerber parabola"),
           Line2D([], [], color=ms.BLUE, lw=1.6, label="Goodman line"),
           Line2D([], [], color=ms.GREEN, lw=1.0, label="Soderberg line"),
           Line2D([], [], color=ms.GREY, lw=0.9, ls=DASH,
                  label="Yield (Langer) line")]
ax_b.legend(handles=handles, loc="upper right", handlelength=2.2)
rows = [f"{k}   n = {safety[k]:.2f}" for k in LOAD_CASES]
ax_b.text(0.985, 0.60, "Goodman safety factor\n" + "\n".join(rows),
          transform=ax_b.transAxes, ha="right", va="top",
          fontsize=ms.FS_TICK, linespacing=1.35)
ax_b.text(30, 22, "Safe by Goodman\nand yield", fontsize=ms.FS_TICK,
          color=ms.BLUE, va="center")
for x, y, label in ((8, s_e + 10, "$S_e$"), (S_Y + 6, 11, "$S_y$"),
                    (S_UT + 6, 11, "$S_{ut}$")):
    ax_b.text(x, y, label, fontsize=ms.FS_MATH, va="center")
ax_b.set_xlim(0, SM_MAX)
ax_b.set_ylim(0, SA_MAX)
ax_b.set_xticks(np.arange(0, 601, 100))
ax_b.set_yticks(np.arange(0, 301, 100))
ax_b.set_xlabel("Mean stress σₘ (MPa)")
ax_b.set_ylabel("Stress amplitude σₐ (MPa)")

ms.panel_label(ax_a, "a", dx_pt=-30)
ms.panel_label(ax_b, "b", dx_pt=-30)
ms.assert_aligned([ax_a, ax_b])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig193_fatigue_sn_goodman.{ext}")
print("fig193_fatigue_sn_goodman: saved png + pdf")
