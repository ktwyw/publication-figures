"""Fig. 177 - Lotka-Volterra predator-prey cycles (double column, 183 mm).

The classical predator-prey model shown the two ways that explain each
other: prey and predator against time for one initial condition, the
predator peak trailing the prey peak by a quarter cycle; and the phase
plane, where every initial condition lies on its own closed orbit, a
level set of the conserved quantity H, circulating anticlockwise around
the coexistence equilibrium at the crossing of the nullclines. The
self-check is that H = δx − γ·ln x + βy − α·ln y is constant along each
trajectory to 1e-7 (relative), that the time-averaged populations over
one period equal the equilibrium (γ/δ, α/β) within 0.5%, and that the
period of the smallest orbit is within 2% of 2π/√(αγ).

Model: dx/dt = αx − βxy, dy/dt = δxy − γy with α = 1.0, β = 0.1,
γ = 1.5, δ = 0.075 (per year; densities in arbitrary units), integrated
by solve_ivp (DOP853, rtol 1e-11); periods from event detection. No
data: every curve is computed from the equations.
"""

import numpy as np
from scipy import integrate

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
ALPHA, BETA, GAMMA, DELTA = 1.0, 0.1, 1.5, 0.075
X_EQ, Y_EQ = GAMMA / DELTA, ALPHA / BETA             # coexistence equilibrium
STARTS = [1.2, 1.6, 2.1, 2.7]     # orbits start at the prey maximum (k·x*, y*)
SHOWN = 2                         # index of the orbit followed in panel a
CONTOURS = [1.4, 1.85, 2.4, 3.0]      # extra level sets of H, same convention
T_RUN = 20.0                      # years integrated (three periods or more)
X_LIM, Y_LIM = (0.0, 80.0), (0.0, 48.0)


def conserved(x, y):
    return DELTA * x - GAMMA * np.log(x) + BETA * y - ALPHA * np.log(y)


def rhs(_t, state):
    """Lotka-Volterra plus the running integrals of x and y."""
    x, y = state[:2]
    return [ALPHA * x - BETA * x * y, DELTA * x * y - GAMMA * y, x, y]


def back_at_start(_t, state):
    return state[1] - Y_EQ        # predator rising through y*: prey maximum


back_at_start.direction = 1.0

# ------------------------------------------------------------ SOLVER ---
orbits = []
for k in STARTS:
    sol = integrate.solve_ivp(rhs, (0, T_RUN), [k * X_EQ, Y_EQ, 0.0, 0.0],
                              method="DOP853", rtol=1e-11, atol=1e-12,
                              dense_output=True, events=back_at_start)
    returns = sol.t_events[0][sol.t_events[0] > 0.1]     # skip the start
    period = returns[0]
    t_cycle = np.linspace(0, period, 1201)
    x, y, _ix, _iy = sol.sol(t_cycle)
    mean_x, mean_y = sol.sol(period)[2:] / period
    h = conserved(*sol.sol(np.linspace(0, T_RUN, 4001))[:2])
    orbits.append(dict(k=k, sol=sol, period=period, x=x, y=y,
                       means=(mean_x, mean_y), returns=returns,
                       drift=np.abs(h / h[0] - 1).max()))

# ------------------------------------------------------- SELF-CHECK ---
worst_drift = max(o["drift"] for o in orbits)
worst_mean = max(abs(m / eq - 1) for o in orbits
                 for m, eq in zip(o["means"], (X_EQ, Y_EQ)))
period_linear = 2 * np.pi / np.sqrt(ALPHA * GAMMA)
periods = [o["period"] for o in orbits]
assert worst_drift < 1e-7, worst_drift
assert worst_mean < 0.005, worst_mean
assert abs(periods[0] / period_linear - 1) < 0.02, periods[0]
assert np.all(np.diff(periods) > 0)              # larger orbits are slower
print(f"fig177: self-check passed (H constant within {worst_drift:.1e}; "
      f"time averages equal ({X_EQ:g}, {Y_EQ:g}) within {worst_mean:.1e}; "
      f"smallest-orbit period {periods[0]:.4f} vs 2π/√(αγ) = "
      f"{period_linear:.4f}; periods {periods[0]:.2f} to {periods[-1]:.2f})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 74)
ax_a = ms.axes(fig, 14, 11, 82, 54)
ax_b = ms.axes(fig, 121, 11, 56, 54)
PREY, PREDATOR = ms.BLUE, ms.VERMILLION
main = orbits[SHOWN]
T_MAIN = main["period"]

# a, time series of the salient orbit; dotted lines are the equilibrium
# values, which the self-check shows are also the averages over a cycle
t_end = 2.85 * T_MAIN             # ends where the two curves are far apart
t = np.linspace(0, t_end, 2000)
x_t, y_t = main["sol"].sol(t)[:2]
for level, colour in ((X_EQ, PREY), (Y_EQ, PREDATOR)):
    ax_a.plot([0, t_end], [level, level], color=colour, lw=0.6,
              ls=(0, (1, 1.6)))
ax_a.plot(t, x_t, color=PREY, lw=1.3)
ax_a.plot(t, y_t, color=PREDATOR, lw=1.3)
x_top = x_t.max()
ms.bracket(ax_a, T_MAIN, 2 * T_MAIN, 1.09 * x_top,
           f"Period T = {T_MAIN:.2f} years", tick=0.03 * x_top,
           text_pad=0.015 * x_top, fontsize=ms.FS_TICK)
for value, name, colour in ((x_t[-1], "Prey x", PREY),
                            (y_t[-1], "Predator y", PREDATOR)):
    ax_a.text(1.015, value, name, color=colour, va="center",
              transform=ax_a.get_yaxis_transform())
ax_a.text(0.025, 0.985, "Dotted, cycle means =\nequilibrium (γ/δ, α/β)",
          transform=ax_a.transAxes, fontsize=ms.FS_SMALL, color=ms.GREY_DARK,
          va="top", linespacing=1.3)
ax_a.set_xlim(0, t_end)
ax_a.set_ylim(0, 1.22 * x_top)
ax_a.set_xlabel("Time (years)")
ax_a.set_ylabel("Population density (a.u.)")

# b, phase plane: level sets of H (light), nullclines (dashed), the four
# orbits with arrowheads for the direction of circulation
gx, gy = np.meshgrid(np.linspace(1, X_LIM[1], 300),
                     np.linspace(0.5, Y_LIM[1], 300))
levels = sorted(conserved(k * X_EQ, Y_EQ) for k in CONTOURS)
ax_b.contour(gx, gy, conserved(gx, gy), levels=levels, colors=ms.GREY_LIGHT,
             linewidths=0.5, linestyles="solid")
dash = dict(color=ms.GREY, lw=0.6, ls=(0, (3, 2)))
ax_b.plot([X_EQ, X_EQ], Y_LIM, **dash)                # dy/dt = 0
ax_b.plot(X_LIM, [Y_EQ, Y_EQ], **dash)                # dx/dt = 0
for o in orbits:
    salient = o is main
    colour = ms.INK if salient else ms.GREY_DARK
    ax_b.plot(o["x"], o["y"], color=colour, lw=1.5 if salient else 0.8)
    # arrowheads on the falling flank (prey declining as predators grow)
    # and at the prey minimum (predators declining)
    for i_mid in (int(np.argmax(o["x"] + 0.8 * o["y"])),
                  int(np.argmin(o["x"]))):
        i_tail, i_tip = (i_mid - 8) % 1200, i_mid + 8    # arrays are periodic
        ax_b.annotate("", xy=(o["x"][i_tip], o["y"][i_tip]),
                      xytext=(o["x"][i_tail], o["y"][i_tail]),
                      arrowprops=dict(arrowstyle="-|>", color=colour, lw=0,
                                      mutation_scale=9, shrinkA=0, shrinkB=0))
ax_b.plot(X_EQ, Y_EQ, "o", ms=3.6, color=ms.GREEN, mec="white", mew=0.5,
          zorder=5)
ax_b.annotate(f"Equilibrium (γ/δ, α/β)\n= ({X_EQ:g}, {Y_EQ:g})",
              xy=(X_EQ, Y_EQ), xytext=(0.97, 0.76), textcoords="axes fraction",
              ha="right", va="top", fontsize=ms.FS_TICK, color=ms.GREEN,
              arrowprops=dict(arrowstyle="-", color=ms.GREEN, lw=0.5,
                              shrinkA=1, shrinkB=3, relpos=(0.0, 0.3)))
ax_b.text(0.97, 0.97, "Bold, orbit of panel a\nLight, H = constant",
          transform=ax_b.transAxes, ha="right", va="top",
          fontsize=ms.FS_SMALL, color=ms.GREY_DARK, linespacing=1.3)
ax_b.annotate("dy/dt = 0", xy=(X_EQ, Y_LIM[1]), xytext=(0, 3),
              textcoords="offset points", ha="center", va="bottom",
              fontsize=ms.FS_TICK, color=ms.GREY_DARK)
ax_b.annotate("dx/dt = 0", xy=(X_LIM[1], Y_EQ), xytext=(-2, 3),
              textcoords="offset points", ha="right", va="bottom",
              fontsize=ms.FS_TICK, color=ms.GREY_DARK)
ax_b.set_xlim(*X_LIM)
ax_b.set_ylim(*Y_LIM)
ax_b.set_xlabel("Prey density x (a.u.)")
ax_b.set_ylabel("Predator density y (a.u.)")

ms.panel_label(ax_a, "a", dx_pt=-28, dy_pt=8)
ms.panel_label(ax_b, "b", dx_pt=-26, dy_pt=8)
ms.assert_aligned([ax_a, ax_b])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig177_predator_prey.{ext}")
print("fig177_predator_prey: saved png + pdf")
