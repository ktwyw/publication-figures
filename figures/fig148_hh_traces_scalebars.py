"""Fig. 148 - Traces with scale bars and f-I curve (double column, 183 mm).

The convention of the patch-clamp literature: voltage traces stacked
without axes, read against an L-shaped scale bar, with the injected
current drawn underneath in the same time base, and the summary curve
(firing rate against current) beside them with the example traces
marked in matching colours. The traces are a Hodgkin-Huxley point
neuron answering five current steps. The self-check is that the resting
state is a fixed point (|dV/dt| < 1e-6 mV/ms without current), that
spike counts never decrease with current across the sweep, and that
each scale bar spans exactly the value printed beside it.

Model: Hodgkin & Huxley (1952) squid-axon membrane, 6.3 °C, rest at
−65 mV; 200 ms current steps; LSODA, rtol 1e-6; spikes are upward
crossings of 0 mV; rate = 1 / mean inter-spike interval when firing
lasts to the end of the step, otherwise 0; sweep in 0.5 µA cm⁻² steps.
All data are simulated.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
# Hodgkin & Huxley, J. Physiol. 117, 500-544 (1952), with voltages
# shifted so that rest is near -65 mV (the modern sign convention).
C_M = 1.0                                   # µF cm⁻²
G_NA, G_K, G_L = 120.0, 36.0, 0.3           # mS cm⁻²
E_NA, E_K, E_L = 50.0, -77.0, -54.387       # mV
T_PRE, T_STEP, T_POST = 20.0, 200.0, 40.0   # ms before, during, after step
EXAMPLES = [2.0, 5.0, 8.0, 12.0, 20.0]      # µA cm⁻², shown as traces
SWEEP = np.arange(0.0, 22.01, 0.5)          # µA cm⁻², for the f-I curve
BAR_MV, BAR_MS, BAR_UA = 20.0, 50.0, 10.0   # scale bars
OFFSET_MV = 125.0                           # vertical spacing of traces


def rates(v):
    """Opening and closing rates (per ms) of the m, h and n gates; plain
    floats, because the solver calls this a few hundred thousand times."""
    alpha = (0.1 * (v + 40) / (1 - math.exp(-(v + 40) / 10)),
             0.07 * math.exp(-(v + 65) / 20),
             0.01 * (v + 55) / (1 - math.exp(-(v + 55) / 10)))
    beta = (4.0 * math.exp(-(v + 65) / 18),
            1 / (1 + math.exp(-(v + 35) / 10)),
            0.125 * math.exp(-(v + 65) / 80))
    return alpha, beta


def rhs(_t, state, current):
    v, m, h, n = state
    (a_m, a_h, a_n), (b_m, b_h, b_n) = rates(v)
    ionic = (G_NA * m ** 3 * h * (v - E_NA) + G_K * n ** 4 * (v - E_K)
             + G_L * (v - E_L))
    return [(current - ionic) / C_M, a_m * (1 - m) - b_m * m,
            a_h * (1 - h) - b_h * h, a_n * (1 - n) - b_n * n]


def steady_gates(v):
    return [a / (a + b) for a, b in zip(*rates(v))]


def spike(_t, state, _current):
    return state[0]                          # upward crossing of 0 mV


spike.direction = 1.0

# ------------------------------------------------------------ SOLVER ---
v_rest = optimize.brentq(
    lambda v: rhs(0.0, [v, *steady_gates(v)], 0.0)[0], -80, -50, xtol=1e-13)
rest = np.array([v_rest, *steady_gates(v_rest)])


def integrate_phase(state, t0, duration, current, sample_ms=None):
    """One constant-current phase; sampled every sample_ms for drawing."""
    t_eval = (None if sample_ms is None else
              np.linspace(t0, t0 + duration, round(duration / sample_ms) + 1))
    return integrate.solve_ivp(rhs, (t0, t0 + duration), state,
                               args=(current,), method="LSODA", rtol=1e-6,
                               atol=1e-8, t_eval=t_eval, events=spike)


def run_step(current):
    """Rest, step, rest: time, voltage and the spike times in the step."""
    phases, state, t0 = [], rest, 0.0
    for duration, amplitude in ((T_PRE, 0.0), (T_STEP, current),
                                (T_POST, 0.0)):
        phases.append(integrate_phase(state, t0, duration, amplitude, 0.05))
        state, t0 = phases[-1].y[:, -1], t0 + duration
    return (np.concatenate([phase.t for phase in phases]),
            np.concatenate([phase.y[0] for phase in phases]),
            phases[1].t_events[0])


def firing_rate(spikes):
    """1 / mean inter-spike interval if firing lasts to the end of the
    step (a train that dies out early is a transient, not a rate)."""
    if spikes.size < 2:
        return 0.0
    interval = np.diff(spikes).mean()
    sustained = T_PRE + T_STEP - spikes[-1] < 2 * interval
    return 1000.0 / interval if sustained else 0.0


# the sweep needs only the step itself: the cell sits at rest before it
sweep_spikes = [integrate_phase(rest, T_PRE, T_STEP, current).t_events[0]
                for current in SWEEP]
counts = np.array([s.size for s in sweep_spikes])
rate = np.array([firing_rate(s) for s in sweep_spikes])
rheobase = SWEEP[np.argmax(counts > 0)]            # first spike
repetitive = SWEEP[np.argmax(rate > 0)]            # sustained firing
traces = {current: run_step(current) for current in EXAMPLES}

# scale bars as data-unit segments, so they can be checked before drawing
T_END = T_PRE + T_STEP + T_POST
bar_x = np.array([T_END - BAR_MS, T_END])                  # ms
bar_y = np.array([v_rest - 55.0, v_rest - 55.0 + BAR_MV])  # mV
bar_i = np.array([0.0, BAR_UA])                            # µA cm⁻²
bar_labels = {"mV": f"{BAR_MV:g} mV", "ms": f"{BAR_MS:g} ms",
              "uA": f"{BAR_UA:g} µA cm⁻²"}

# ------------------------------------------------------- SELF-CHECK ---
drift = np.abs(rhs(0.0, rest, 0.0))
assert drift[0] < 1e-6 and drift[1:].max() < 1e-12, drift
assert np.all(np.diff(rate[rate > 0]) > 0), rate    # monotonic once firing
assert np.all(np.diff(counts) >= 0), counts
for values, key in ((bar_y, "mV"), (bar_x, "ms"), (bar_i, "uA")):
    assert np.ptp(values) == float(bar_labels[key].split()[0]), key
assert all(current in SWEEP for current in EXAMPLES)
print(f"fig148: self-check passed (rest {v_rest:.3f} mV, |dV/dt| = "
      f"{drift[0]:.1e} mV/ms; spike counts {counts.min()}-{counts.max()} "
      f"non-decreasing over {SWEEP.size} currents; rheobase "
      f"{rheobase:g}, sustained firing from {repetitive:g} µA cm⁻²; scale "
      f"bars {BAR_MV:g} mV, {BAR_MS:g} ms, {BAR_UA:g} µA cm⁻²)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 92)
ax_v = ms.axes(fig, 25, 21, 92, 65)                  # traces, no axes
ax_i = ms.axes(fig, 25, 5, 92, 12, sharex=ax_v)      # current protocol
ax_b = ms.axes(fig, 139, 13, 39, 73)
colours = dict(zip(EXAMPLES, plt.cm.plasma(np.linspace(0.0, 0.8, 5))))
label_at = ax_v.get_yaxis_transform()                # x in axes fraction

# a, stacked traces: smallest current at the bottom, 125 mV apart
for k, (current, (t, v, _spikes)) in enumerate(traces.items()):
    ax_v.plot(t, v + k * OFFSET_MV, color=colours[current], lw=0.7)
    ax_v.text(-0.015, v_rest + k * OFFSET_MV, f"{current:g} µA cm⁻²",
              transform=label_at, color=colours[current], ha="right",
              va="center", fontsize=ms.FS_TICK)
    ax_i.plot([0, T_PRE, T_PRE, T_PRE + T_STEP, T_PRE + T_STEP, T_END],
              [0, 0, current, current, 0, 0], color=colours[current], lw=0.7)
ax_v.plot([0, T_END], [-65, -65], color=ms.GREY, lw=0.5, ls=(0, (1, 2)),
          zorder=0)
ax_v.text(1.01, -65, "−65 mV", transform=label_at, color=ms.GREY,
          va="center", fontsize=ms.FS_SMALL)
# L-shaped scale bar under the lowest trace, flush with the record's end
ax_v.plot([bar_x[0], bar_x[1], bar_x[1]], [bar_y[0], bar_y[0], bar_y[1]],
          color=ms.INK, lw=1.0, solid_capstyle="butt",
          solid_joinstyle="miter")
ax_v.text(bar_x.mean(), bar_y[0] - 5, bar_labels["ms"], ha="center",
          va="top", fontsize=ms.FS_TICK)
ax_v.text(1.01, bar_y.mean(), bar_labels["mV"], transform=label_at,
          va="center", fontsize=ms.FS_TICK)
ax_v.set_xlim(0, T_END)
ax_v.set_ylim(v_rest - 78, v_rest + 4 * OFFSET_MV + 118)
ax_v.axis("off")

ax_i.plot([T_END, T_END], bar_i, color=ms.INK, lw=1.0, solid_capstyle="butt",
          clip_on=False)
ax_i.text(1.01, bar_i.mean(), bar_labels["uA"],
          transform=ax_i.get_yaxis_transform(), va="center",
          fontsize=ms.FS_TICK)
ax_i.text(-0.015, 0, "Injected\ncurrent", transform=ax_i.get_yaxis_transform(),
          ha="right", va="bottom", fontsize=ms.FS_TICK, color=ms.GREY_DARK)
ax_i.set_ylim(-1, max(EXAMPLES) + 1)
ax_i.axis("off")

# b, f-I curve: every swept current, the five examples in their colours
ax_b.plot(SWEEP, rate, color=ms.GREY_DARK, lw=0.8, marker="o", ms=1.8,
          mew=0, zorder=2)
for current in EXAMPLES:
    ax_b.plot(current, rate[SWEEP == current], "o", ms=4.5,
              color=colours[current], mec="white", mew=0.5, zorder=3)
# thresholds as flags: a dotted line with its label beside the top, the
# two staggered so that neither label meets the other line or the curve
for threshold, name, top in ((rheobase, "Rheobase", 118),
                             (repetitive, "Sustained firing", 104)):
    ax_b.plot([threshold, threshold], [0, top], color=ms.GREY, lw=0.5,
              ls=(0, (1, 2)), zorder=1)
    ax_b.text(threshold + 0.5, top, f"{name},\n{threshold:g} µA cm⁻²",
              va="top", fontsize=ms.FS_SMALL, color=ms.GREY_DARK)
ax_b.set_xlim(-0.8, 22.8)
ax_b.set_ylim(-3, 119)
ax_b.set_xticks(np.arange(0, 21, 5))
ax_b.set_yticks(np.arange(0, 101, 25))
ax_b.spines["left"].set_bounds(0, 100)
ax_b.spines["bottom"].set_bounds(0, 20)
ax_b.set_xlabel("Injected current (µA cm⁻²)")
ax_b.set_ylabel("Firing rate (Hz)", y=53 / 122)

ms.panel_label(ax_v, "a", dx_pt=-64, dy_pt=-2)
ms.panel_label(ax_b, "b", dx_pt=-28, dy_pt=-2)
ms.assert_aligned([ax_v, ax_i], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig148_hh_traces_scalebars.{ext}")
print("fig148_hh_traces_scalebars: saved png + pdf")
