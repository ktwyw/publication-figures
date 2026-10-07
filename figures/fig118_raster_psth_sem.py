"""Fig. 118 - Two-condition spike raster over a PSTH with s.e.m. (89 mm).

Extends fig037 (one condition, pooled histogram, Gaussian-smoothed rate)
to the comparison a reader actually needs: two stimulus conditions
stacked in one raster with rotated block labels in the trace colours, a
PSTH that is binned, not smoothed, and shows the mean ± s.e.m. across
trials as a band, a stimulus bar above the raster, and both panels
sharing the time axis on a millimetre grid. The self-check is that
binning conserves spikes: for each condition, sum(mean rate x bin width)
x number of trials equals the number of spikes in the histogram range,
and the two plot areas share their left and right edges within 1.5 pt.

Statistics: one example neuron, n = 30 trials per stimulus; a, each tick
is one spike; b, firing rate in 20-ms bins, line, mean across trials;
band, ± s.e.m. across trials; no hypothesis test. All spike trains are
simulated (inhomogeneous Poisson).
"""

import numpy as np

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(1818)
T_MIN, T_MAX, STIM_ON, STIM_OFF = -200, 600, 0, 300      # ms
LATENCY, DECAY = 40, 70.0                                # ms
N_TRIALS, BIN = 30, 20                                   # trials, ms
SHADE = "#EFEDE8"                                        # stimulus window
CONDITIONS = {"Preferred": (48.0, ms.VERMILLION),        # (peak drive, colour)
              "Non-preferred": (11.0, ms.GREY_DARK)}


# ------------------------------------------------------------- DATA ----
def rate(t, peak):
    """6 spikes per s plus a transient that decays to a sustained level."""
    driven = (t >= STIM_ON + LATENCY) & (t < STIM_OFF + LATENCY)
    transient = 0.3 + 0.7 * np.exp(-(t - STIM_ON - LATENCY) / DECAY)
    return 6.0 + peak * np.where(driven, transient, 0.0)


time = np.arange(T_MIN, T_MAX)                           # 1-ms steps
trains = {condition: [time[rng.uniform(size=time.size)
                           < rate(time, peak) / 1000]
                      for _trial in range(N_TRIALS)]
          for condition, (peak, _colour) in CONDITIONS.items()}

# ------------------------------------------------------------- PSTH ----
edges = np.arange(T_MIN, T_MAX + BIN, BIN)
centres = edges[:-1] + BIN / 2
psth = {}
for condition, spikes in trains.items():
    hz = np.array([np.histogram(s, edges)[0] for s in spikes]) / (BIN / 1000)
    psth[condition] = (hz.mean(axis=0),
                       hz.std(axis=0, ddof=1) / np.sqrt(N_TRIALS))

# ------------------------------------------------------- SELF-CHECK ---
n_spikes = {}
for condition, spikes in trains.items():
    pooled = np.concatenate(spikes)
    n_spikes[condition] = int(((pooled >= edges[0])
                               & (pooled <= edges[-1])).sum())
    binned = psth[condition][0].sum() * (BIN / 1000) * N_TRIALS
    assert abs(binned - n_spikes[condition]) < 1e-9, (condition, binned)
assert psth["Preferred"][0].max() > psth["Non-preferred"][0].max()
print("fig118: self-check passed (binning conserves spikes: "
      + ", ".join(f"{c} {n}" for c, n in n_spikes.items())
      + f"; {N_TRIALS} trials, {BIN}-ms bins)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 90)
gs = ms.grid(fig, 2, 1, left=14, right=5, top=9, bottom=11, hspace=7,
             height_ratios=[1.25, 1])
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1], sharex=ax_a)

for block, (condition, (_peak, colour)) in enumerate(CONDITIONS.items()):
    hero = condition == "Preferred"
    base = block * (N_TRIALS + 3)
    ax_a.eventplot(trains[condition], lineoffsets=base + np.arange(N_TRIALS),
                   linelengths=0.8, linewidths=0.5, colors=colour)
    ax_a.text(-0.035, base + (N_TRIALS - 1) / 2, condition,
              transform=ax_a.get_yaxis_transform(), rotation=90,
              rotation_mode="anchor", ha="center", va="bottom", color=colour,
              fontsize=ms.FS_TICK)
    mean, sem = psth[condition]
    ax_b.fill_between(centres, mean - sem, mean + sem, color=colour,
                      alpha=0.22, lw=0)
    ax_b.plot(centres, mean, color=colour, lw=1.3 if hero else 1.0,
              label=condition)

for ax in (ax_a, ax_b):
    ax.axvspan(STIM_ON, STIM_OFF, color=SHADE, lw=0, zorder=0)
ax_a.set_xlim(T_MIN, T_MAX)
ax_a.set_ylim(2 * N_TRIALS + 3.5, -1.5)
ax_a.set_yticks([])
ax_a.spines["left"].set_visible(False)
ax_a.tick_params(axis="x", labelbottom=False)
ax_a.plot([STIM_ON, STIM_OFF], [1.03, 1.03],
          transform=ax_a.get_xaxis_transform(), color=ms.INK, lw=1.6,
          clip_on=False, solid_capstyle="butt")
ax_a.text((STIM_ON + STIM_OFF) / 2, 1.06, "Stimulus",
          transform=ax_a.get_xaxis_transform(), ha="center", va="bottom",
          fontsize=ms.FS_TICK)
ax_a.text(0.99, 1.03, f"{N_TRIALS} trials per stimulus",
          transform=ax_a.transAxes, ha="right", va="bottom",
          fontsize=ms.FS_SMALL, color=ms.GREY_DARK)
ax_b.set_ylim(0, 60)
ax_b.set_yticks(np.arange(0, 61, 20))
ax_b.set_xticks(np.arange(-200, 601, 200))
ax_b.set_xlabel("Time from stimulus onset (ms)")
ax_b.set_ylabel("Firing rate (spikes per s)")
ax_b.legend(loc="upper right")
ms.panel_label(ax_a, "a", dx_pt=-30)
ms.panel_label(ax_b, "b", dx_pt=-30)

ms.assert_aligned([ax_a, ax_b], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig118_raster_psth_sem.{ext}")
print("fig118_raster_psth_sem: saved png + pdf")
