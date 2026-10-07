"""Fig. 159 - Seismic refraction record section, wiggles (1.5 column, 120 mm).

How a crustal refraction profile is read: 30 seismograms drawn as wiggle
traces at their source-receiver offsets, positive lobes filled, on a
reduced time axis that runs downward and flattens the Moho head wave;
over them the travel-time curves of the direct wave, the head wave and
the Moho reflection from one two-layer model, each labelled where it
leaves the section, with the critical and crossover distances marked.
The self-check is that the crossover found by brentq equals
2h√((v₂ + v₁)/(v₂ − v₁)), that the head wave starts at 2h tan θc and is
tangent there to the reflection hyperbola (equal time and slope, and
never later than it), and that the first peak of each noise-free trace
lies within one sample of the earlier theoretical arrival wherever the
two earliest arrivals are more than one wavelet breadth apart.

Model: flat layer of thickness h = 20 km and v₁ = 6.0 km s⁻¹ over a
half-space of v₂ = 8.0 km s⁻¹; reduced time t − x/v₂. Data: 3 Hz Ricker
wavelets at the three arrival times with illustrative amplitudes plus
Gaussian noise (s.d. 0.05 of the direct-wave amplitude), sampled at
50 Hz; n = 30 traces. All data are simulated.
"""

import numpy as np
from scipy import optimize, signal

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(159)

# -------------------------------------------------- GOVERNING MODEL ----
THICKNESS, V1, V2 = 20.0, 6.0, 8.0             # km, km/s, km/s
V_REDUCE = V2                                  # reduction velocity
OFFSETS = np.arange(6.0, 181.0, 6.0)           # km, 30 receivers
DT, T_MIN, T_MAX = 0.02, -0.6, 8.8             # s (reduced time window)
F_RICKER, NOISE_SD = 3.0, 0.05
AMP_DIRECT, AMP_HEAD = 1.0, 0.45               # illustrative amplitudes

theta_c = np.arcsin(V1 / V2)                   # critical angle
x_critical = 2 * THICKNESS * np.tan(theta_c)
x_cross_analytic = 2 * THICKNESS * np.sqrt((V2 + V1) / (V2 - V1))


def t_direct(x):
    return x / V1


def t_head(x):
    """Head wave: down and up at the critical angle, along layer 2's top."""
    return x / V2 + 2 * THICKNESS * np.cos(theta_c) / V1


def t_reflection(x):
    return np.hypot(x, 2 * THICKNESS) / V1


def amp_reflection(x):
    """Weak before the critical distance, strong beyond it."""
    return 0.35 + 0.5 / (1 + np.exp(-(x - x_critical) / 8.0))


def ricker(tau):
    arg = (np.pi * F_RICKER * tau) ** 2
    return (1 - 2 * arg) * np.exp(-arg)


# ------------------------------------------------------------- DATA ----
t_reduced = np.arange(T_MIN, T_MAX + DT / 2, DT)
arrivals, clean = [], np.zeros((OFFSETS.size, t_reduced.size))
for i, x in enumerate(OFFSETS):
    phases = [(t_direct(x), AMP_DIRECT), (t_reflection(x), amp_reflection(x))]
    if x >= x_critical:                        # no head wave before this
        phases.append((t_head(x), AMP_HEAD))
    arrivals.append(sorted(t - x / V_REDUCE for t, _amp in phases))
    for t_arrival, amplitude in phases:
        clean[i] += amplitude * ricker(t_reduced - (t_arrival - x / V_REDUCE))
traces = clean + rng.normal(0, NOISE_SD, clean.shape)

# ----------------------------------------------------------- SOLVER ---
x_cross = optimize.brentq(lambda x: t_direct(x) - t_head(x), x_critical,
                          20 * THICKNESS, xtol=1e-12)
# first-arrival pick: the first peak above a tenth of the trace maximum
picks = np.array([t_reduced[signal.find_peaks(row, height=0.1 * row.max())[0]
                            [0]] for row in clean])
first_theory = np.array([a[0] for a in arrivals])
breadth = np.sqrt(6) / (np.pi * F_RICKER)      # between the Ricker side lobes
separated = np.array([a[1] - a[0] > breadth for a in arrivals])

# ------------------------------------------------------- SELF-CHECK ---
assert abs(x_cross - x_cross_analytic) < 1e-9, (x_cross, x_cross_analytic)
assert abs(t_reflection(x_critical) - t_head(x_critical)) < 1e-12
slope_reflection = x_critical / (V1 * np.hypot(x_critical, 2 * THICKNESS))
assert abs(slope_reflection - 1 / V2) < 1e-12          # tangent, not crossing
x_test = np.linspace(0, 2 * OFFSETS[-1], 2001)
assert np.all(t_reflection(x_test) - t_head(x_test) > -1e-12)
has_head = np.array([len(a) == 3 for a in arrivals])
assert np.array_equal(has_head, OFFSETS >= x_critical)
pick_error = np.abs(picks - first_theory)[separated]
assert pick_error.max() <= DT + 1e-12, pick_error.max()
head_first = first_theory[separated] < (OFFSETS / V1 - OFFSETS / V_REDUCE)[
    separated] - 1e-9
assert np.array_equal(head_first, OFFSETS[separated] > x_cross)
print(f"fig159: self-check passed (critical distance {x_critical:.2f} km; "
      f"crossover {x_cross:.3f} km vs analytic {x_cross_analytic:.3f} km; "
      f"reflection tangent to head wave at t = {t_head(x_critical):.3f} s; "
      f"first-arrival picks within {pick_error.max() / DT:.2f} sample at "
      f"{separated.sum()} of {OFFSETS.size} traces)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 100)
ax = ms.axes(fig, 14, 10.5, 82, 81)
X_END = OFFSETS[-1]
gain = 1.1 * (OFFSETS[1] - OFFSETS[0])         # km of deflection per unit

for x, trace in zip(OFFSETS, traces):
    ax.plot(x + gain * trace, t_reduced, color=ms.INK, lw=0.3, zorder=3)
    ax.fill_betweenx(t_reduced, x, x + gain * trace, where=trace > 0,
                     color=ms.INK, lw=0, zorder=3)

curves = [("Direct wave", t_direct, 0.0, ms.BLUE),
          ("Moho head wave", t_head, x_critical, ms.VERMILLION),
          ("Moho reflection", t_reflection, 0.0, ms.GREEN)]
for name, travel_time, x_start, colour in curves:
    x_line = np.linspace(x_start, X_END, 300)
    reduced = travel_time(x_line) - x_line / V_REDUCE
    ax.plot(x_line, reduced, color=colour, lw=1.2, zorder=2)   # under traces
    ax.plot([X_END, X_END + 8], [reduced[-1]] * 2, color=colour, lw=1.2,
            clip_on=False, solid_capstyle="butt", zorder=2)
    ax.text(X_END + 10, reduced[-1], name, color=colour, va="center",
            fontsize=ms.FS_TICK)

# landmarks: a dotted drop from the top edge to the point on the curves
for x_mark, name in ((x_critical, "Critical distance"),
                     (x_cross, "Crossover")):
    t_mark = t_head(x_mark) - x_mark / V_REDUCE
    ax.plot([x_mark, x_mark], [T_MIN, t_mark], color=ms.VERMILLION, lw=0.7,
            ls=(0, (1.5, 1.5)))
    ax.plot(x_mark, t_mark, "o", ms=3.6, mfc="white", mec=ms.VERMILLION,
            mew=0.9, zorder=5)
    ax.annotate(f"{name}\n{x_mark:.1f} km", (x_mark, T_MIN), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=ms.FS_TICK, linespacing=1.15)

ax.set_xlim(0, X_END + 8)
ax.set_ylim(T_MAX, T_MIN)                      # time increases downward
ax.set_xticks(np.arange(0, X_END + 1, 50))
ax.set_yticks(np.arange(0, T_MAX, 2))
ax.set_xlabel("Source–receiver offset x (km)")
ax.set_ylabel(f"Reduced time t − x/{V_REDUCE:.1f} (s)")
ax.text(1.02, 1.0, f"h = {THICKNESS:.0f} km\nv₁ = {V1:.1f} km s⁻¹\n"
        f"v₂ = {V2:.1f} km s⁻¹", transform=ax.transAxes, va="top",
        fontsize=ms.FS_TICK, color=ms.GREY_DARK, linespacing=1.3)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig159_seismic_record_section.{ext}")
print("fig159_seismic_record_section: saved png + pdf")
