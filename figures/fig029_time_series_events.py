"""Fig. 29 - Time series with a date axis, events and a rolling mean (double column).

Date axes trip everyone up; this shows the modern recipe (AutoDateLocator +
ConciseDateFormatter), raw data de-emphasised under a rolling mean, shaded
intervention periods labelled at the top, and an annotated extreme.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(29)

# ------------------------------------------------------------- DATA ----
dates = np.arange("2019-01-01", "2022-01-01", dtype="datetime64[D]")
t = np.arange(dates.size)
no2 = (28 + 8 * np.cos(2 * np.pi * (t - 15) / 365.25)   # winter maxima
       - 0.004 * t + rng.normal(0, 4.5, t.size))         # slow decline + noise

events = [("Lockdown", "2020-03-15", "2020-06-01", -9),
          ("Low-emission zone", "2021-04-01", "2022-01-01", -4)]
for _, start, end, effect in events:
    m = (dates >= np.datetime64(start)) & (dates < np.datetime64(end))
    no2[m] += effect

WIN = 30
rolling = np.convolve(no2, np.ones(WIN) / WIN, mode="valid")
dates_roll = dates[WIN // 2: WIN // 2 + rolling.size]

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(7.1, 2.4))

for label, start, end, _ in events:
    s, e = np.datetime64(start), np.datetime64(end)
    ax.axvspan(s, e, color="0.92", lw=0, zorder=0)
    ax.text(s + (e - s) / 2, 0.97, label, transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=6.5, color="0.35")

ax.plot(dates, no2, color="C0", lw=0.5, alpha=0.35, label="Daily")
ax.plot(dates_roll, rolling, color="C1", lw=1.4, label=f"{WIN}-day mean")

peak = np.argmax(no2)
ax.annotate("Record daily peak", xy=(dates[peak], no2[peak]),
            xytext=(18, 6), textcoords="offset points", fontsize=6.5,
            arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))

locator = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
ax.set_xlim(dates[0], dates[-1])
ax.set_ylabel("NO$_2$ (\u00b5g m$^{-3}$)")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2,
          fontsize=6.5, borderaxespad=0)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig029_time_series_events.{ext}")
print("saved fig029_time_series_events.png / .pdf")
