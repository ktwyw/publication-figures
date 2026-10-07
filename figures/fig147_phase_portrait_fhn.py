"""Fig. 147 - Phase portrait of an oscillator (double column, 183 mm).

How to show a planar dynamical system so that the geometry explains the
time series: the FitzHugh-Nagumo model with its two nullclines, a
direction field, trajectories from four initial conditions that all end
on the same limit cycle, and the unstable fixed point at the nullcline
crossing; beside it, v(t) and w(t) of the trajectory that starts next
to the fixed point, with one period bracketed. The self-check is that
the fixed point satisfies both nullcline equations to 1e-10, that its
Jacobian eigenvalues have positive real part, and that the period from
upward zero-crossings of v matches the autocorrelation peak within 1%.

Model: dv/dt = v − v³/3 − w + I, dw/dt = ε(v + a − bw) with a = 0.7,
b = 0.8, ε = 0.08, I = 0.5 (dimensionless); fixed point by brentq,
trajectories by solve_ivp (RK45, rtol 1e-10). Arrows show direction
only (unit length on the page). All data are simulated.
"""

import numpy as np
from scipy import integrate, optimize, signal

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
A, B, EPS, CURRENT = 0.7, 0.8, 0.08, 0.5
STARTS = [(None, None), (2.4, 0.2), (-2.4, 0.9), (-0.4, 1.85)]  # (v, w)
NUDGE = (0.04, 0.02)               # first start: this far from the fixed point
T_LONG, T_SETTLE, T_SHOWN = 1000.0, 200.0, 200.0
V_LIM, W_LIM = (-2.6, 2.6), (-0.9, 2.0)


def v_nullcline(v):
    return v - v ** 3 / 3 + CURRENT              # dv/dt = 0


def w_nullcline(v):
    return (v + A) / B                           # dw/dt = 0


def rhs(_t, state):
    v, w = state
    return [v - v ** 3 / 3 - w + CURRENT, EPS * (v + A - B * w)]


def upward_crossing(_t, state):
    return state[0]


upward_crossing.direction = 1.0

# ------------------------------------------------------------ SOLVER ---
v_star = optimize.brentq(lambda v: v_nullcline(v) - w_nullcline(v), -3, 3,
                         xtol=1e-14, rtol=1e-14)
w_star = w_nullcline(v_star)
jacobian = np.array([[1 - v_star ** 2, -1.0], [EPS, -EPS * B]])
eigenvalues = np.linalg.eigvals(jacobian)

STARTS[0] = (v_star + NUDGE[0], w_star + NUDGE[1])
solve = dict(method="RK45", rtol=1e-10, atol=1e-12, dense_output=True)
long_run = integrate.solve_ivp(rhs, (0, T_LONG), STARTS[0],
                               events=upward_crossing, **solve)
transients = [integrate.solve_ivp(rhs, (0, 120), start, **solve)
              for start in STARTS]

# period 1: spacing of upward zero-crossings of v once on the cycle
crossings = long_run.t_events[0]
settled = crossings[crossings > T_SETTLE]
period_crossing = np.diff(settled).mean()
# period 2: first peak of the autocorrelation of v on a uniform grid
DT = 0.02
v_uniform = long_run.sol(np.arange(T_SETTLE, T_LONG, DT))[0]
centred = v_uniform - v_uniform.mean()
acf = signal.correlate(centred, centred, mode="full",
                       method="fft")[centred.size - 1:]
acf /= centred.size - np.arange(centred.size)    # unbiased at every lag
lag = signal.find_peaks(acf[:centred.size // 2], prominence=1.0)[0][0]
period_acf = lag * DT

# ------------------------------------------------------- SELF-CHECK ---
residual = np.abs(rhs(0.0, (v_star, w_star)))
assert residual[0] < 1e-10 and residual[1] < 1e-10, residual
assert abs(w_star - v_nullcline(v_star)) < 1e-10
assert np.all(eigenvalues.real > 0), eigenvalues          # unstable
assert abs(eigenvalues[0].imag) > 0                       # a focus
assert abs(period_crossing - period_acf) < 0.01 * period_crossing
assert np.ptp(np.diff(settled)) < 1e-6 * period_crossing  # truly periodic
print(f"fig147: self-check passed (fixed point ({v_star:.4f}, "
      f"{w_star:.4f}), residual {residual.max():.1e}; eigenvalues "
      f"{eigenvalues[0].real:.3f} ± {abs(eigenvalues[0].imag):.3f}i; period "
      f"{period_crossing:.3f} from crossings, {period_acf:.2f} from "
      "autocorrelation)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 80)
A_W, A_H = 76.0, 61.0                                    # panel a, mm
ax_a = ms.axes(fig, 14, 11, A_W, A_H)
ax_b = ms.axes(fig, 110, 11, 64, A_H)
V_COLOUR, W_COLOUR = ms.BLUE, ms.ORANGE          # one colour per variable

# a, direction field: unit arrows in page space, so only direction is read;
# labels sit on rows of the arrow grid, and the arrows under them are left
# out (row 1 from v = -0.45 to 2.15; rows 13 and 14 right of v = 0.9)
rows = np.linspace(*W_LIM, 17)[1:-1]
gv, gw = np.meshgrid(np.linspace(*V_LIM, 23)[1:-1], rows)
keep = ~(((gw == rows[1]) & (gv > -0.45) & (gv < 2.15))
         | ((gw >= rows[13]) & (gv > 0.9)))
dv, dw = rhs(0.0, (gv[keep], gw[keep]))
page_x = dv / np.ptp(V_LIM) * A_W
page_y = dw / np.ptp(W_LIM) * A_H
norm = np.hypot(page_x, page_y)
ax_a.quiver(gv[keep], gw[keep], page_x / norm, page_y / norm, angles="uv",
            pivot="mid", scale=34, width=0.0028, headwidth=4, headlength=4.5,
            headaxislength=4, color=ms.GREY_LIGHT, zorder=1)

# nullclines end at the frame (not merely clipped by it), so that the
# labels above the top edge are clear of the paths in the PDF as well
v_line = np.linspace(*V_LIM, 400)
for nullcline, colour in ((v_nullcline, V_COLOUR), (w_nullcline, W_COLOUR)):
    w_line = nullcline(v_line)
    inside = (w_line >= W_LIM[0]) & (w_line <= W_LIM[1])
    ax_a.plot(v_line[inside], w_line[inside], color=colour, lw=0.9, zorder=2)
for run, start in zip(transients, STARTS):
    path = run.sol(np.linspace(0, 120, 3000))
    ax_a.plot(*path, color=ms.GREY, lw=0.6, zorder=3)
    ax_a.plot(*start, "o", color=ms.GREY, ms=2.4, mew=0, zorder=5)
cycle = long_run.sol(np.linspace(settled[-2], settled[-1], 800))
ax_a.plot(*cycle, color=ms.INK, lw=1.6, zorder=4)
ax_a.plot(v_star, w_star, "o", ms=4, mfc="white", mec=ms.INK, mew=0.8,
          zorder=6)
ax_a.set_xlim(*V_LIM)
ax_a.set_ylim(*W_LIM)
ax_a.set_xlabel("Fast variable v")
ax_a.set_ylabel("Recovery variable w")

# direct labels: nullclines where they leave the top edge, the rest in
# clear regions of the plane
v_exit = optimize.brentq(lambda v: v_nullcline(v) - W_LIM[1], -3, -1)
w_exit = B * W_LIM[1] - A
for v_top, name, colour in ((v_exit, "v-nullcline", V_COLOUR),
                            (w_exit, "w-nullcline", W_COLOUR)):
    ax_a.annotate(name, xy=(v_top, W_LIM[1]), xytext=(0, 3),
                  textcoords="offset points", color=colour, ha="center",
                  va="bottom", fontsize=ms.FS_TICK)
ax_a.text(2.5, rows[13], "Limit cycle", fontsize=ms.FS_TICK,
          fontweight="bold", ha="right", va="center")
ax_a.annotate("Unstable focus, λ = "
              f"{eigenvalues[0].real:.2f} ± {abs(eigenvalues[0].imag):.2f}i",
              xy=(v_star, w_star), xytext=(-0.35, rows[1]),
              fontsize=ms.FS_TICK, va="center",
              arrowprops=dict(arrowstyle="-", color=ms.INK, lw=0.5,
                              shrinkA=1, shrinkB=3, relpos=(0.0, 0.5)))
ax_a.text(2.5, rows[14], "Dots, initial conditions", ha="right",
          va="center", fontsize=ms.FS_SMALL, color=ms.GREY_DARK)

# b, the same dynamics against time, for the start beside the fixed point
t_shown = np.linspace(0, T_SHOWN, 4000)
v_t, w_t = long_run.sol(t_shown)
ax_b.plot(t_shown, v_t, color=V_COLOUR, lw=1.1)
ax_b.plot(t_shown, w_t, color=W_COLOUR, lw=1.1)
ax_b.text(T_SHOWN + 3, v_t[-1], "v", color=V_COLOUR, va="center")
ax_b.text(T_SHOWN + 3, w_t[-1], "w", color=W_COLOUR, va="center")
t1, t2 = crossings[(crossings > 100) & (crossings < T_SHOWN)][:2]
ms.bracket(ax_b, t1, t2, 2.45, f"Period T = {t2 - t1:.1f}", tick=0.12,
           text_pad=0.06, fontsize=ms.FS_TICK)
ax_b.set_xlim(0, T_SHOWN)
ax_b.set_ylim(-2.3, 2.9)
ax_b.set_yticks(np.arange(-2, 2.1, 1))
ax_b.spines["left"].set_bounds(-2, 2)
ax_b.set_xlabel("Time t (dimensionless)")
ax_b.set_ylabel("v, w")

ms.panel_label(ax_a, "a", dx_pt=-30, dy_pt=8)
ms.panel_label(ax_b, "b", dx_pt=-26, dy_pt=8)
ms.assert_aligned([ax_a, ax_b])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig147_phase_portrait_fhn.{ext}")
print("fig147_phase_portrait_fhn: saved png + pdf")
