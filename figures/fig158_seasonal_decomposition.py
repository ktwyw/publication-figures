"""Fig. 158 - Classical seasonal decomposition, monthly (1.5 column, 120 mm).

A Keeling-style record split into trend, seasonal cycle and remainder
with the textbook additive method coded in full: a centred 12-month
moving average for the trend, month means of the detrended series for
the seasonal component, and what is left. Each panel has its own y
scale, so a grey bar at the right of every panel marks the same absolute
range (the STL convention) and shows at a glance how small the remainder
is. Because the method assumes a fixed seasonal cycle, the growth of
the simulated cycle leaks into the remainder, which the last panel
shows. The self-check is that trend + seasonal + remainder returns the
series to 1e-10 wherever the trend is defined, that the seasonal
component sums to zero over any 12 consecutive months, and that the
mean growth rate of the estimated trend is within 5% of the simulated.

Data: n = 480 monthly values, January 1985 to December 2024: quadratic
trend, an annual plus semi-annual cycle whose amplitude grows by 1.2%
per year, and AR(1) noise (s.d. 0.35 ppm, lag-one correlation 0.5).
The moving average is undefined for the first and last six months. All
data are simulated.
"""

import matplotlib.dates as mdates
import numpy as np
from matplotlib.patches import Rectangle

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(158)

# ------------------------------------------------------------- DATA ----
dates = np.arange("1985-01", "2025-01", dtype="datetime64[M]")
n = dates.size
month = np.arange(n) % 12
years = np.arange(n) / 12.0                    # time since the first value
LEVEL, RATE, ACCEL = 345.0, 1.35, 0.016        # ppm, ppm/yr, ppm/yr²
AMPLITUDE, AMP_GROWTH = 2.8, 0.012             # ppm; fraction per year
NOISE_SD, NOISE_RHO = 0.35, 0.5

true_trend = LEVEL + RATE * years + ACCEL * years ** 2
cycle = (np.cos(2 * np.pi * (month - 4) / 12)
         + 0.3 * np.cos(4 * np.pi * (month - 2.5) / 12))
true_seasonal = AMPLITUDE * (1 + AMP_GROWTH * years) * cycle
noise = np.empty(n)
noise[0] = rng.normal(0, NOISE_SD)
for i in range(1, n):
    noise[i] = NOISE_RHO * noise[i - 1] + rng.normal(
        0, NOISE_SD * np.sqrt(1 - NOISE_RHO ** 2))
series = true_trend + true_seasonal + noise

# -------------------------------------------------------- ESTIMATOR ----
# centred moving average over an even period: 13 terms, half weight at
# the two ends, so every calendar month counts exactly once
kernel = np.r_[0.5, np.ones(11), 0.5] / 12
trend = np.full(n, np.nan)
trend[6:-6] = np.convolve(series, kernel, mode="valid")
defined = ~np.isnan(trend)
detrended = series - trend
month_mean = np.array([detrended[defined & (month == m)].mean()
                       for m in range(12)])
seasonal = (month_mean - month_mean.mean())[month]       # sums to zero
remainder = series - trend - seasonal

span = years[defined][-1] - years[defined][0]
growth = (trend[defined][-1] - trend[defined][0]) / span
true_growth = (true_trend[defined][-1] - true_trend[defined][0]) / span

# ------------------------------------------------------- SELF-CHECK ---
rebuilt = trend[defined] + seasonal[defined] + remainder[defined]
worst_rebuild = np.abs(rebuilt - series[defined]).max()
assert worst_rebuild < 1e-10, worst_rebuild
window_sums = np.convolve(seasonal, np.ones(12), mode="valid")
assert np.abs(window_sums).max() < 1e-10, np.abs(window_sums).max()
assert abs(growth - true_growth) < 0.05 * true_growth, (growth, true_growth)
assert np.nanstd(remainder) < np.std(seasonal) < np.nanstd(trend)
print(f"fig158: self-check passed (n = {n}; rebuild error "
      f"{worst_rebuild:.1e}; largest 12-month seasonal sum "
      f"{np.abs(window_sums).max():.1e}; trend growth {growth:.3f} vs "
      f"simulated {true_growth:.3f} ppm/yr; remainder s.d. "
      f"{np.nanstd(remainder):.2f} ppm)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 112)
gs = ms.grid(fig, 4, 1, left=17, right=7, top=6, bottom=9, hspace=4.5,
             height_ratios=[1.25, 1.25, 1, 1])
axs = [fig.add_subplot(gs[0])]
axs += [fig.add_subplot(gs[i], sharex=axs[0]) for i in range(1, 4)]
BAR = 4.0                                      # ppm, the same in every panel
panels = [("Observed", series, ms.INK, (338, 432)),
          ("Trend", trend, ms.VERMILLION, (338, 432)),
          ("Seasonal", seasonal, ms.BLUE, (-4.6, 6.4)),
          ("Remainder", remainder, ms.GREY_DARK, (-2.0, 2.8))]
for ax, letter, (name, values, colour, y_lim) in zip(axs, "abcd", panels):
    ax.plot(dates, values, color=colour, lw=0.9 if name == "Trend" else 0.6)
    ax.set_ylim(*y_lim)
    ax.set_ylabel(f"{name}\n(ppm)")
    # the range bar: BAR ppm tall on this panel's own scale
    ax.add_patch(Rectangle((1.012, np.mean(y_lim) - BAR / 2), 0.014, BAR,
                           transform=ax.get_yaxis_transform(), clip_on=False,
                           color=ms.GREY, lw=0))
    ax.tick_params(labelbottom=ax is axs[-1])
    ms.panel_label(ax, letter, dx_pt=-42, dy_pt=-3)
axs[3].plot(dates[[0, -1]], [0, 0], color=ms.GREY_LIGHT, lw=0.5, zorder=0)

note = dict(fontsize=ms.FS_TICK, va="top", color=ms.GREY_DARK)
axs[0].text(0.012, 0.93, f"Monthly means, n = {n}", transform=axs[0].transAxes,
            **note)
axs[1].text(0.012, 0.93, "Centred 12-month moving average: mean growth "
            f"{growth:.2f} ppm yr⁻¹ (simulated {true_growth:.2f})",
            transform=axs[1].transAxes, **note)
axs[2].text(0.012, 0.93, "Month means of the detrended series: one fixed "
            f"cycle, {np.ptp(seasonal):.1f} ppm peak to trough",
            transform=axs[2].transAxes, **note)
axs[3].text(0.012, 0.93, "A fixed cycle cannot follow the growing one: "
            "the misfit is largest at both ends",
            transform=axs[3].transAxes, **note)
axs[0].text(1.0, 1.04, f"Grey bars: {BAR:.0f} ppm in every panel",
            transform=axs[0].transAxes, ha="right", va="bottom",
            fontsize=ms.FS_SMALL, color=ms.GREY_DARK)

axs[0].set_xlim(dates[0], dates[-1])
axs[3].xaxis.set_major_locator(mdates.YearLocator(5))
axs[3].xaxis.set_minor_locator(mdates.YearLocator(1))
axs[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axs[3].set_xlabel("Year")
fig.align_ylabels(axs)

ms.assert_aligned(axs, edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig158_seasonal_decomposition.{ext}")
print("fig158_seasonal_decomposition: saved png + pdf")
