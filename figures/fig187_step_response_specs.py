"""Fig. 187 - Step-response specifications, second order (1.5 column, 120 mm).

How to dimension a step response like an engineering drawing: unit-step
responses of the standard second-order system for five damping ratios,
one of them (ζ = 0.4) in colour and carrying the classical time-domain
specifications as thin dimension lines: 10–90% rise time, peak time,
percent overshoot and the ±2% settling band with the settling time,
next to the textbook estimate 4/(ζωₙ). The self-check is that the
measured overshoot equals exp(−πζ/√(1 − ζ²)) and the measured peak time
equals π/(ωₙ√(1 − ζ²)) within 0.1%, that the analytic responses match
scipy.signal.step to 1e-6, and that the measured settling time lies in
the last half period before the envelope bound −ln(0.02√(1 − ζ²))/(ζωₙ),
which is the exact form of 4/(ζωₙ) and never exceeds it by more than 1%
for ζ ≤ 0.4.

Model: G(s) = ωₙ² / (s² + 2ζωₙs + ωₙ²), ωₙ = 1, ζ = 0.2, 0.4, 0.7, 1 and
2; closed-form responses for the under-, critically and overdamped
cases; time in units of 1/ωₙ. No data: every curve is computed from the
equations.
"""

import numpy as np
from scipy import optimize, signal

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
OMEGA_N = 1.0                            # natural frequency (sets the unit)
ZETAS = [0.2, 0.4, 0.7, 1.0, 2.0]        # damping ratios
SALIENT = 0.4                            # the dimensioned curve
BAND = 0.02                              # settling tolerance (±2%)
T_END = 14.0


def step_response(t, zeta, wn=OMEGA_N):
    """Unit-step response of wn² / (s² + 2 zeta wn s + wn²), closed form."""
    t = np.asarray(t, dtype=float)
    if zeta < 1:
        wd = wn * np.sqrt(1 - zeta ** 2)
        return 1 - np.exp(-zeta * wn * t) * (
            np.cos(wd * t) + zeta * wn / wd * np.sin(wd * t))
    if zeta == 1:
        return 1 - (1 + wn * t) * np.exp(-wn * t)
    s1, s2 = wn * (-zeta + np.array([1, -1]) * np.sqrt(zeta ** 2 - 1))
    return 1 - (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s2 - s1)


# ---------------------------------------------------- MEASUREMENTS -----
def y(t):
    return step_response(t, SALIENT)


t = np.linspace(0, T_END, 2801)
step = t[1] - t[0]
k_peak = int(np.argmax(y(t)))
peak = optimize.minimize_scalar(lambda u: -y(u), method="bounded",
                                bounds=(t[k_peak - 1], t[k_peak + 1]),
                                options=dict(xatol=1e-12))
t_peak, overshoot = peak.x, -peak.fun - 1
t_10, t_90 = (optimize.brentq(lambda u, level=level: y(u) - level, 0, t_peak,
                              xtol=1e-13) for level in (0.1, 0.9))
t_long = np.arange(0, 60, step)                   # well past any settling
k_last = np.flatnonzero(np.abs(y(t_long) - 1) > BAND)[-1]
t_settle = optimize.brentq(lambda u: abs(y(u) - 1) - BAND, t_long[k_last],
                           t_long[k_last + 1], xtol=1e-13)

# ------------------------------------------------------- SELF-CHECK ---
root = np.sqrt(1 - SALIENT ** 2)
overshoot_theory = np.exp(-np.pi * SALIENT / root)
t_peak_theory = np.pi / (OMEGA_N * root)
assert abs(overshoot - overshoot_theory) < 1e-3 * overshoot_theory
assert abs(t_peak - t_peak_theory) < 1e-3 * t_peak_theory
worst = 0.0
for zeta in ZETAS:
    system = ([OMEGA_N ** 2], [1, 2 * zeta * OMEGA_N, OMEGA_N ** 2])
    _, reference = signal.step(system, T=t)
    worst = max(worst, np.abs(step_response(t, zeta) - reference).max())
assert worst < 1e-6, worst
# |y − 1| = envelope × |sin(ω_d t + φ)|: the response last leaves the band
# within one half period before the envelope enters it
t_envelope = -np.log(BAND * root) / (SALIENT * OMEGA_N)
t_textbook = 4 / (SALIENT * OMEGA_N)
assert t_envelope - t_peak_theory <= t_settle <= t_envelope
assert abs(t_envelope - t_textbook) < 0.01 * t_textbook
print(f"fig187: self-check passed (zeta = {SALIENT}: overshoot "
      f"{100 * overshoot:.3f}% vs {100 * overshoot_theory:.3f}%, peak time "
      f"{t_peak:.4f} vs {t_peak_theory:.4f}; rise time {t_90 - t_10:.3f}; "
      f"settling {t_settle:.3f} <= envelope bound {t_envelope:.3f} "
      f"(textbook {t_textbook:.1f}); max |analytic - scipy| {worst:.1e})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 76)
ax = ms.axes(fig, 13, 11, 102, 60)
KEY = ms.BLUE
GREYS = dict(zip([z for z in ZETAS if z != SALIENT],
                 ["#B4B4B4", "#929292", "#6E6E6E", "#4A4A4A"]))
DIM = dict(color=ms.INK, lw=0.5)                        # dimension lines
ARROW = dict(arrowstyle="<|-|>", color=ms.INK, lw=0.5, mutation_scale=5,
             shrinkA=0, shrinkB=0)

ax.axhspan(1 - BAND, 1 + BAND, color=KEY, alpha=0.16, lw=0, zorder=0)
for zeta, grey in GREYS.items():
    ax.plot(t, step_response(t, zeta), color=grey, lw=0.9, zorder=2)
ax.plot(t, y(t), color=KEY, lw=1.6, zorder=4)

# direct labels in gaps that no curve or dimension line passes through
t_02 = np.pi / np.sqrt(1 - 0.2 ** 2)
ax.text(t_02, step_response(t_02, 0.2) + 0.025, "ζ = 0.2", ha="center",
        va="bottom", fontsize=ms.FS_TICK, color=ms.GREY)
ax.text(t_02, 1 + overshoot + 0.13, f"ζ = {SALIENT}", ha="center",
        va="center", fontsize=ms.FS_TICK, color=KEY, fontweight="bold")
ax.text(5.05, 1.11, "ζ = 0.7", ha="left", va="center", fontsize=ms.FS_TICK,
        color=ms.GREY_DARK)
ax.text(3.65, 0.79, "ζ = 1", ha="left", va="center", fontsize=ms.FS_TICK,
        color=ms.GREY_DARK)
ax.text(8.7, step_response(8.7, 2.0) - 0.04, "ζ = 2", ha="left", va="top",
        fontsize=ms.FS_TICK, color=ms.GREY_DARK)

# rise time: drops from the 10% and 90% points to a dimension near the axis
for t_mark, level in ((t_10, 0.1), (t_90, 0.9)):
    ax.plot([t_mark, t_mark], [0.035, level], **DIM, zorder=3)
    ax.plot(t_mark, level, "o", ms=2.6, color=KEY, mec="white", mew=0.4,
            zorder=5)
ax.annotate("", xy=(t_10, 0.035), xytext=(t_90, 0.035), arrowprops=ARROW)
ax.text(t_90 + 0.12, 0.05, f"Rise time\n10–90%\n{t_90 - t_10:.2f}",
        ha="left", va="bottom", fontsize=ms.FS_SMALL, linespacing=1.2)

# peak time and overshoot
ax.plot([t_peak, t_peak], [0, 1 + overshoot], **DIM, zorder=3)
ax.plot(t_peak, 1 + overshoot, "o", ms=2.6, color=KEY, mec="white", mew=0.4,
        zorder=5)
ax.text(t_peak + 0.2, 0.05, f"Peak time\n{t_peak:.2f}", ha="left",
        va="bottom", fontsize=ms.FS_SMALL, linespacing=1.2)
X_OS = 6.4                                   # where the overshoot is read
ax.plot([t_peak, X_OS + 0.2], [1 + overshoot] * 2, **DIM, zorder=3)
ax.annotate("", xy=(X_OS, 1.0), xytext=(X_OS, 1 + overshoot),
            arrowprops=ARROW)
ax.text(X_OS + 0.3, 1 + BAND + (overshoot - BAND) / 2,
        f"Overshoot\n{100 * overshoot:.1f}%", ha="left", va="center",
        fontsize=ms.FS_SMALL, linespacing=1.2)

# settling: last entry into the band, and the textbook estimate
ax.plot([t_settle, t_settle], [0, y(t_settle)], **DIM, zorder=3)
ax.plot(t_settle, y(t_settle), "o", ms=2.6, color=KEY, mec="white", mew=0.4,
        zorder=5)
ax.text(t_settle - 0.2, 0.05, f"Settling time\n(±2%) {t_settle:.2f}",
        ha="right", va="bottom", fontsize=ms.FS_SMALL, linespacing=1.2)
ax.plot([t_textbook, t_textbook], [0, 1 - BAND], color=ms.GREY, lw=0.5,
        ls=(0, (3, 2)), zorder=3)
ax.text(t_textbook + 0.2, 0.05, f"Estimate\n4/(ζωₙ) = {t_textbook:.0f}",
        ha="left", va="bottom", fontsize=ms.FS_SMALL, color=ms.GREY_DARK,
        linespacing=1.2)
ax.text(T_END - 0.2, 0.88, "±2% settling band", ha="right",
        va="top", fontsize=ms.FS_SMALL, color=KEY)

ax.set_xlim(0, T_END)
ax.set_ylim(0, 1.65)
ax.set_xticks(np.arange(0, T_END + 1, 2))
ax.set_yticks(np.arange(0, 1.61, 0.2))
ax.set_xlabel("Time t (units of 1/ωₙ)")
ax.set_ylabel("Step response y(t)")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig187_step_response_specs.{ext}")
print("fig187_step_response_specs: saved png + pdf")
