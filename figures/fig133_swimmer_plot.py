"""Fig. 133 - Swimmer plot of time on treatment (1.5 column, 120 mm).

One bar per patient of an early-phase dose-escalation cohort, longest
at the top, shaded by dose level (an ordinal ramp, because dose is
ordered). Marker shape carries the events: first response (triangle,
partial; circle, complete), progression (square) and death (cross) at
the bar end, and an arrow head where treatment continues at the data
cut-off. The self-check is bookkeeping a swimmer plot must not get
wrong: every event lies on its own bar, the rows are monotone in
duration, and the arrows are exactly the patients without progression
or death.

Statistics: n = 26 patients (8, 9 and 9 per dose level); bars, months
from first dose to progression, death or data cut-off; descriptive, no
interval and no test. All data are simulated.
"""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(133)
N = 26
DOSES = {"Dose level 1": ms.GREY_LIGHT, "Dose level 2": ms.SKY,
         "Dose level 3": ms.BLUE}
MEDIAN_PFS = np.array([5.0, 9.0, 14.0])          # months, by dose level
RESPONSE_RATE = np.array([0.30, 0.55, 0.80])

# ------------------------------------------------------------- DATA ----
# Per patient: dose level, potential follow-up (staggered entry), latent
# progression and death times, and a first response if it comes in time.
dose = np.repeat([0, 1, 2], [8, 9, 9])
follow_up = rng.uniform(4.0, 24.0, N)
t_progression = rng.exponential(MEDIAN_PFS[dose] / np.log(2))
t_death = rng.exponential(45.0, N)               # death without progression
duration = np.minimum(np.minimum(t_progression, t_death), follow_up)
progressed = t_progression == duration
died = t_death == duration
ongoing = follow_up == duration                  # still on treatment
t_response = rng.uniform(1.4, 4.5, N)            # first restaging window
responder = (rng.random(N) < RESPONSE_RATE[dose]) & (t_response < duration)
complete = responder & (rng.random(N) < 0.35)

order = np.argsort(duration)                     # row 0 = shortest, at bottom
rows = np.arange(N)
patient = np.array([f"Patient {k + 1:02d}" for k in range(N)])

# ------------------------------------------------------- SELF-CHECK ---
assert np.all(np.diff(duration[order]) >= 0)
assert np.all((t_response[responder] > 0)
              & (t_response[responder] < duration[responder]))
end_event = np.where(progressed, t_progression, t_death)[~ongoing]
assert np.array_equal(end_event, duration[~ongoing])
assert np.array_equal(ongoing, ~(progressed | died))
assert not np.any(progressed & died)
print(f"fig133: self-check passed (n = {N}: {responder.sum()} responders "
      f"[{complete.sum()} CR, {(responder & ~complete).sum()} PR], "
      f"{progressed.sum()} progressed, {died.sum()} died, {ongoing.sum()} "
      f"ongoing = arrows; all events on their bars; rows sorted)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 96)
ax = fig.add_subplot(ms.grid(fig, 1, 1, left=19, right=31, top=4,
                             bottom=11)[0])

colours = np.array(list(DOSES.values()))
ax.barh(rows, duration[order], height=0.62, color=colours[dose[order]],
        lw=0, zorder=2)

# events: shape first, colour second; white faces read on every bar shade
EVENTS = {
    "Partial response": (responder & ~complete, t_response,
                         dict(marker="^", s=13, facecolor="white",
                              edgecolor=ms.INK, linewidths=0.6)),
    "Complete response": (complete, t_response,
                          dict(marker="o", s=12, facecolor=ms.INK,
                               edgecolor="white", linewidths=0.5)),
    "Progression": (progressed, duration,
                    dict(marker="s", s=11, facecolor=ms.VERMILLION,
                         edgecolor=ms.INK, linewidths=0.5)),
    "Death": (died, duration,
              dict(marker="x", s=14, color=ms.INK, linewidths=1.0)),
    # arrow head just past the bar end: treatment continues at cut-off
    "Treatment ongoing": (ongoing, duration + 0.55,
                          dict(marker=">", s=13, color=ms.GREY_DARK,
                               linewidths=0)),
}
for mask, when, style in EVENTS.values():
    shown = mask[order]
    ax.scatter(when[order][shown], rows[shown], zorder=4, clip_on=False,
               **style)

ax.set_yticks(rows, patient[order])
ax.tick_params(axis="y", length=0, pad=3, labelsize=ms.FS_SMALL)
ax.set_ylim(-0.7, N - 0.3)
ax.set_xlim(0, 21)
ax.set_xticks(np.arange(0, 22, 3))
ax.spines["left"].set_visible(False)
ax.set_xlabel("Time on treatment (months)")

# two compact keys in the right margin, clear of every bar
dose_n = np.bincount(dose)
key = dict(loc="upper left", fontsize=ms.FS_TICK, borderaxespad=0,
           alignment="left")
first = ax.legend(
    handles=[Patch(facecolor=c, lw=0, label=f"{name} (n = {dose_n[k]})")
             for k, (name, c) in enumerate(DOSES.items())],
    title="Bar: dose cohort", bbox_to_anchor=(1.035, 1.0), handlelength=1.1,
    handleheight=0.7, **key)
ax.add_artist(first)
ax.legend(
    handles=[Line2D([], [], ls="none", marker=style["marker"], label=name,
                    markersize=np.sqrt(style["s"]),
                    markerfacecolor=style.get("facecolor", style.get("color")),
                    markeredgecolor=style.get("edgecolor", style.get("color")),
                    markeredgewidth=style["linewidths"])
             for name, (_mask, _when, style) in EVENTS.items()],
    title="Marker: event", bbox_to_anchor=(1.035, 0.80), handlelength=1.1,
    **key)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig133_swimmer_plot.{ext}")
print("fig133_swimmer_plot: saved png + pdf")
