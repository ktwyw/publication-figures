"""Fig. 172 - HPLC chromatogram with integrated peaks (1.5 column, 120 mm).

How to report a chromatogram so that every number in the peak table can
be traced on the trace: five tailing (exponentially modified Gaussian)
peaks on a drifting, noisy baseline, the fitted baseline dashed, each
integration window shaded down to it between drop lines, retention
times at the apexes, the resolution of the critical pair, and the area
percentages beside the plot. The self-check is that baseline-subtracted
areas of the noise-free signal recover the simulated amounts within 1%,
that area percentages sum to 100, and that for a Gaussian test peak the
half-height plate number 5.545 (t_R/w½)² equals (t_R/σ)² within 0.1%.

Data: detector signal = Σ areaᵢ · EMG(t; µᵢ, σᵢ, τᵢ) + drift + Gaussian
noise (s.d. 0.15 mAU), sampled at 10 Hz; baseline, cubic least-squares
fit to the points outside the integration windows; areas by the
trapezoid rule with a perpendicular drop between the critical pair;
R_s = 1.18 Δt_R/(w½,₁ + w½,₂). All data are simulated.
"""

import numpy as np
from scipy import integrate, stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(172)

# ------------------------------------------------------------- DATA ----
# analyte: (Gaussian centre µ, σ, exponential tail τ; all in min), area
# in mAU·min
PEAKS = {"A": (1.90, 0.045, 0.030, 3.2), "B": (3.60, 0.055, 0.040, 7.5),
         "C": (5.30, 0.060, 0.035, 5.0), "D": (5.68, 0.062, 0.035, 2.4),
         "E": (8.10, 0.085, 0.060, 4.1)}
T_END, RATE_HZ = 10.0, 10.0               # run length (min), sampling
NOISE_SD = 0.15                           # mAU
PAIR = ("C", "D")                         # the closely eluting pair


def drift(t):
    """Slow baseline drift of a gradient run (mAU)."""
    return 2.0 + 0.55 * t + 1.6 * np.sin(2 * np.pi * t / 26.0)


def emg(t, mu, sigma, tau):
    """Exponentially modified Gaussian of unit area."""
    return stats.exponnorm.pdf(t, tau / sigma, loc=mu, scale=sigma)


t = np.arange(0.0, T_END + 1e-9, 1 / (60 * RATE_HZ))
parts = {name: area * emg(t, mu, sigma, tau)
         for name, (mu, sigma, tau, area) in PEAKS.items()}
clean = sum(parts.values())               # noise-free, drift-free signal
signal = clean + drift(t) + rng.normal(0, NOISE_SD, t.size)

# -------------------------------------------------------- ESTIMATORS ---
# integration windows: µ − 5σ to µ + 5σ + 8τ; the critical pair shares
# a perpendicular drop at the valley of the noise-free signal
edges = {name: [mu - 5 * sigma, mu + 5 * sigma + 8 * tau]
         for name, (mu, sigma, tau, _a) in PEAKS.items()}
between = (t > PEAKS[PAIR[0]][0]) & (t < PEAKS[PAIR[1]][0])
valley = t[between][np.argmin(clean[between])]
edges[PAIR[0]][1] = edges[PAIR[1]][0] = valley
windows = {name: (t >= lo) & (t <= hi) for name, (lo, hi) in edges.items()}

outside = ~np.any(list(windows.values()), axis=0)
baseline = np.polyval(np.polyfit(t[outside], signal[outside], 3), t)


def half_width(time, y):
    """Full width at half maximum by linear interpolation of the flanks."""
    top = np.argmax(y)
    half = y[top] / 2
    left = np.interp(half, y[:top + 1], time[:top + 1])
    right = np.interp(half, y[top:][::-1], time[top:][::-1])
    return right - left


area_true = {name: integrate.trapezoid(clean[w], t[w])
             for name, w in windows.items()}
area_meas = {name: integrate.trapezoid((signal - baseline)[w], t[w])
             for name, w in windows.items()}
percent = {name: 100 * a / sum(area_meas.values())
           for name, a in area_meas.items()}
t_apex = {name: t[np.argmax(part)] for name, part in parts.items()}
w_half = {name: half_width(t, part) for name, part in parts.items()}
FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2))                 # w½ = 2.355 σ
resolution = (FWHM_FACTOR / 2 * (t_apex[PAIR[1]] - t_apex[PAIR[0]])
              / (w_half[PAIR[0]] + w_half[PAIR[1]]))     # 1.18 Δt/Σw½

# ------------------------------------------------------- SELF-CHECK ---
recovery = {name: area_true[name] / PEAKS[name][3] for name in PEAKS}
assert all(abs(r - 1) < 0.01 for r in recovery.values()), recovery
assert abs(sum(percent.values()) - 100) < 1e-9
assert all(abs(area_meas[n] / PEAKS[n][3] - 1) < 0.05 for n in PEAKS)
assert abs(FWHM_FACTOR ** 2 - 5.545) < 1e-3 and abs(FWHM_FACTOR / 2
                                                    - 1.18) < 5e-3
T_TEST, S_TEST = 5.0, 0.06                # pure Gaussian: N = (t_R/σ)²
t_fine = np.linspace(T_TEST - 1, T_TEST + 1, 4001)
w_test = half_width(t_fine, stats.norm.pdf(t_fine, T_TEST, S_TEST))
plates = FWHM_FACTOR ** 2 * (T_TEST / w_test) ** 2
assert abs(plates / (T_TEST / S_TEST) ** 2 - 1) < 1e-3, plates
worst = max(abs(r - 1) for r in recovery.values())
print(f"fig172: self-check passed (noise-free areas within {100 * worst:.2f}%"
      f" of the simulated amounts; area% sums to "
      f"{sum(percent.values()):.1f}; Gaussian test peak N = {plates:.0f} vs "
      f"(t_R/σ)² = {(T_TEST / S_TEST) ** 2:.0f}; R_s({PAIR[0]},{PAIR[1]}) = "
      f"{resolution:.2f})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 70)
ax = ms.axes(fig, 14, 11, 69, 53)
ax_tab = ms.axes(fig, 87.5, 11, 29.5, 53)
FILL = ms.SKY

ax.plot(t, signal, color=ms.INK, lw=0.6, zorder=3)
ax.plot(t, baseline, color=ms.VERMILLION, lw=0.7, ls=(0, (4, 2)), zorder=4)
y_top = 0.0
for name, w in windows.items():
    ax.fill_between(t[w], baseline[w], signal[w], color=FILL, alpha=0.45,
                    lw=0, zorder=1)
    for edge in (np.flatnonzero(w)[0], np.flatnonzero(w)[-1]):
        ax.plot([t[edge]] * 2, [baseline[edge], signal[edge]],
                color=ms.GREY_DARK, lw=0.5, zorder=2)
    apex = np.argmin(np.abs(t - t_apex[name]))
    height = (clean + drift(t))[apex]
    y_top = max(y_top, height)
    ax.text(t_apex[name], height + 2.2, f"{t_apex[name]:.2f}", rotation=90,
            ha="center", va="bottom", fontsize=ms.FS_SMALL)
    ax.text(t_apex[name], height + 9.0, name, ha="center", va="bottom",
            fontsize=ms.FS_TICK, fontweight="bold")

# resolution of the critical pair on a bracket above its two labels
first, second = (t_apex[name] for name in PAIR)
pair_top = max((clean + drift(t))[np.abs(t - x).argmin()]
               for x in (first, second))
ms.bracket(ax, first, second, pair_top + 14.5,
           f"Rₛ = {resolution:.2f}", tick=1.2, text_pad=0.8)
ax.annotate("Fitted\nbaseline", xy=(6.9, np.interp(6.9, t, baseline) + 0.8),
            xytext=(6.9, 15), color=ms.VERMILLION, fontsize=ms.FS_TICK,
            ha="center", va="bottom", linespacing=1.15,
            arrowprops=dict(arrowstyle="-", color=ms.VERMILLION, lw=0.5,
                            shrinkA=1, shrinkB=1))
ax.set_xlim(0, T_END)
ax.set_ylim(0, np.ceil((y_top + 14) / 10) * 10)
ax.set_xticks(np.arange(0, T_END + 0.1, 2))
ax.set_xlabel("Retention time (min)")
ax.set_ylabel("Absorbance (mAU)")

# peak table: text only, every value computed above
ax_tab.axis("off")
ax_tab.set_xlim(0, 1)
ax_tab.set_ylim(0, 1)
COLS = (0.0, 0.65, 1.0)                    # peak | t_R right | area right
ROW0, STEP = 0.90, 0.085
ax_tab.text(COLS[0], ROW0, "Peak", fontsize=ms.FS_TICK, fontweight="bold",
            va="center")
ax_tab.text(COLS[1], ROW0, "RT (min)", fontsize=ms.FS_TICK,
            fontweight="bold", ha="right", va="center")
ax_tab.text(COLS[2], ROW0, "Area %", fontsize=ms.FS_TICK, fontweight="bold",
            ha="right", va="center")
for row, name in enumerate(PEAKS, start=1):
    y_row = ROW0 - row * STEP
    ax_tab.text(COLS[0], y_row, name, fontsize=ms.FS_TICK, va="center")
    ax_tab.text(COLS[1], y_row, f"{t_apex[name]:.2f}", fontsize=ms.FS_TICK,
                ha="right", va="center")
    ax_tab.text(COLS[2], y_row, f"{percent[name]:.1f}", fontsize=ms.FS_TICK,
                ha="right", va="center")
y_sum = ROW0 - (len(PEAKS) + 1) * STEP
ax_tab.text(COLS[0], y_sum, "Total", fontsize=ms.FS_TICK, va="center")
ax_tab.text(COLS[2], y_sum, f"{sum(percent.values()):.1f}",
            fontsize=ms.FS_TICK, ha="right", va="center")
for y_rule in (ROW0 + STEP / 2, ROW0 - STEP / 2, y_sum + STEP / 2,
               y_sum - STEP / 2):
    ax_tab.plot([0, 1], [y_rule] * 2, color=ms.INK, lw=0.5, clip_on=False)
ax_tab.text(0, y_sum - 0.085, "Shaded, integrated area\nabove the fitted "
            "baseline;\nnoise s.d. "
            f"{NOISE_SD:g} mAU", fontsize=ms.FS_SMALL, color=ms.GREY_DARK,
            va="top", linespacing=1.3)

ms.assert_aligned([ax, ax_tab])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig172_chromatogram.{ext}")
print("fig172_chromatogram: saved png + pdf")
