"""Fig. 195 - Beam shear-force and bending-moment diagram (1.5 column, 120 mm).

The three drawings of a beam calculation stacked on one x axis: the
free-body sketch to scale (pin, roller, distributed load, point load,
applied couple and the computed reactions), the shear force with its
jumps at every point load, and the bending moment, whose maximum sits
where the shear crosses zero and whose zero crossings are the points
of contraflexure. Shear and moment come from Macaulay brackets, so the
diagrams follow from the loads alone. The self-check is that the
reactions satisfy ΣF = 0 and ΣM = 0 to 1e-9, that dM/dx = V away from
the point loads (error below 1e-3 of the peak shear), that M jumps by
exactly the applied couple, that M = 0 at the pinned end and at the
free end, and that the maximum M lies at a zero of V.

Model: Euler-Bernoulli statics of a simply supported beam with an
overhang; sagging moment positive, shear positive when the left part
pushes up, so dM/dx = V; a clockwise couple raises M. No data: every
curve is computed from the equations.
"""

import numpy as np
from matplotlib.patches import Ellipse, Polygon
from scipy import optimize

import manuscript as ms

ms.apply()
HERE = ms.HERE

# ------------------------------------------------------------- DATA ----
LENGTH = 8.0                    # beam length (m); free end at x = LENGTH
X_PIN, X_ROLLER = 0.0, 6.0      # supports A and B (m)
W, X_W1, X_W2 = 12.0, 0.0, 3.0  # distributed load (kN m⁻¹) from X_W1 to X_W2
P, X_P = 6.0, 8.0               # point load (kN, downward) at the free end
M0, X_M = 18.0, 4.5             # applied couple (kN m, clockwise)

# ---------------------------------------------------------- STATICS ----
# ΣF = 0 and ΣM about x = 0 = 0 (anticlockwise positive) for the reactions
load, centroid = W * (X_W2 - X_W1), (X_W1 + X_W2) / 2
r_a, r_b = np.linalg.solve([[1.0, 1.0], [X_PIN, X_ROLLER]],
                           [load + P, load * centroid + P * X_P + M0])


def ramp(x, a):
    return np.clip(x - a, 0.0, None)             # Macaulay bracket <x − a>¹


def shear_moment(x, x_ref=None):
    """V and M at x; steps switch on where x_ref (default x) has passed."""
    on = (lambda a: np.asarray(x if x_ref is None else x_ref) >= a)
    v = (r_a * on(X_PIN) + r_b * on(X_ROLLER) - P * on(X_P)
         - W * (ramp(x, X_W1) - ramp(x, X_W2)))
    m = (r_a * ramp(x, X_PIN) + r_b * ramp(x, X_ROLLER) - P * ramp(x, X_P)
         - W / 2 * (ramp(x, X_W1) ** 2 - ramp(x, X_W2) ** 2) + M0 * on(X_M))
    return v, m


# sample each load-free stretch separately, so both sides of a jump exist
breaks = np.unique([0.0, X_PIN, X_ROLLER, X_W1, X_W2, X_P, X_M, LENGTH])
pieces = [np.linspace(a, b, 401) for a, b in zip(breaks[:-1], breaks[1:])]
values = [shear_moment(x, np.full_like(x, x.mean())) for x in pieces]
x_all = np.concatenate(pieces)
v_all = np.concatenate([v for v, _m in values])
m_all = np.concatenate([m for _v, m in values])

# V = 0 inside a stretch (the moment extremum) and M = 0 (contraflexure)
x_v0, x_m0 = [], []
for x, (v, m) in zip(pieces, values):
    ref = x.mean()
    for which, found in ((0, x_v0), (1, x_m0)):
        y = (v, m)[which]
        if y[1] * y[-2] < 0:
            found.append(optimize.brentq(
                lambda s: shear_moment(s, ref)[which], x[1], x[-2],
                xtol=1e-13))
x_peak = x_v0[int(np.argmax([shear_moment(s)[1] for s in x_v0]))]
m_peak = float(shear_moment(x_peak)[1])

# ------------------------------------------------------- SELF-CHECK ---
assert abs(r_a + r_b - load - P) < 1e-9                          # ΣF = 0
for pivot in (X_ROLLER, LENGTH):                 # ΣM = 0 about other points
    moment = (r_a * (X_PIN - pivot) + r_b * (X_ROLLER - pivot)
              - load * (centroid - pivot) - P * (X_P - pivot) - M0)
    assert abs(moment) < 1e-9, moment
slope_error = max(np.abs(np.gradient(m, x)[1:-1] - v[1:-1]).max()
                  for x, (v, m) in zip(pieces, values))
assert slope_error < 1e-3 * np.abs(v_all).max(), slope_error     # dM/dx = V
k = int(np.flatnonzero(breaks == X_M)[0])
jump = values[k][1][0] - values[k - 1][1][-1]
assert abs(jump - M0) < 1e-9
assert abs(m_all[0]) < 1e-9 and abs(m_all[-1]) < 1e-9   # pin end, free end
assert abs(shear_moment(x_peak)[0]) < 1e-9 and m_peak >= m_all.max() - 1e-9
assert abs(m_peak - m_all.max()) < 1e-3 * m_peak
print(f"fig195: self-check passed (RA = {r_a:.3f}, RB = {r_b:.3f} kN; "
      f"max |dM/dx − V| = {slope_error:.1e} kN; couple jump {jump:.3f} kN m;"
      f" Mmax = {m_peak:.3f} kN m at x = {x_peak:.4f} m where V = 0; "
      "contraflexure at x = " + ", ".join(f"{s:.3f}" for s in x_m0) + " m)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 112)
LEFT, RIGHT, X_LIM = 17.0, 6.0, (-0.45, 8.45)
gs = ms.grid(fig, 3, 1, left=LEFT, right=RIGHT, top=4, bottom=11, hspace=6,
             height_ratios=[25, 33, 33])
ax_s = fig.add_subplot(gs[0])
ax_v = fig.add_subplot(gs[1], sharex=ax_s)
ax_m = fig.add_subplot(gs[2], sharex=ax_s)
LOAD_C, REACT_C = ms.VERMILLION, ms.BLUE
MM = np.ptp(X_LIM) / (120 - LEFT - RIGHT)        # metres of x per page mm


def arrow(ax, tail, head, colour, lw=0.9):
    ax.annotate("", xy=head, xytext=tail, arrowprops=dict(
        arrowstyle="-|>", color=colour, lw=lw, mutation_scale=6, shrinkA=0,
        shrinkB=0))


# sketch: x to scale in metres, y in millimetres on the page (ylim spans
# the 25 mm panel), so supports and the couple keep their shape
ax_s.set_ylim(-13, 12)
ax_s.axis("off")
ax_s.plot([0, LENGTH], [0, 0], color=ms.INK, lw=3.0, solid_capstyle="butt")
BASE = -4.2                                      # ground level under supports
ax_s.add_patch(Polygon([(X_PIN, -0.8), (X_PIN - 2.0 * MM, BASE),
                        (X_PIN + 2.0 * MM, BASE)], closed=True, fc="white",
                       ec=ms.INK, lw=0.8))
radius = (-0.8 - BASE) / 2
ax_s.add_patch(Ellipse((X_ROLLER, BASE + radius), 2 * radius * MM, 2 * radius,
                       fc="white", ec=ms.INK, lw=0.8))
for x_sup in (X_PIN, X_ROLLER):
    ax_s.plot([x_sup - 3.2 * MM, x_sup + 3.2 * MM], [BASE, BASE],
              color=ms.INK, lw=0.8)
for x_sup, reaction, name in ((X_PIN, r_a, "A"), (X_ROLLER, r_b, "B")):
    arrow(ax_s, (x_sup, -12.5), (x_sup, BASE - 0.6), REACT_C, lw=1.1)
    ax_s.text(x_sup + 1.8 * MM, -9.3, f"$R_{name}$ = {reaction:.1f} kN",
              color=REACT_C, fontsize=ms.FS_MATH, va="center")
ax_s.plot([X_W1, X_W2], [6.5, 6.5], color=LOAD_C, lw=0.9)
for x_arrow in np.linspace(X_W1, X_W2, 7):
    arrow(ax_s, (x_arrow, 6.5), (x_arrow, 1.6), LOAD_C, lw=0.7)
ax_s.text((X_W1 + X_W2) / 2, 8.0, f"w = {W:g} kN m⁻¹", color=LOAD_C,
          ha="center", va="bottom", fontsize=ms.FS_TICK)
arrow(ax_s, (X_P, 9.0), (X_P, 1.6), LOAD_C, lw=1.1)
ax_s.text(X_P - 1.8 * MM, 7.0, f"P = {P:g} kN", color=LOAD_C, ha="right",
          va="center", fontsize=ms.FS_TICK)
angle = np.radians(np.linspace(215, -35, 80))    # clockwise, over the top
arc_x, arc_y = X_M + 4.2 * MM * np.cos(angle), 4.2 * np.sin(angle)
ax_s.plot(arc_x[:-3], arc_y[:-3], color=LOAD_C, lw=0.9)
arrow(ax_s, (arc_x[-8], arc_y[-8]), (arc_x[-1], arc_y[-1]), LOAD_C)
ax_s.text(X_M, 6.2, f"M₀ = {M0:g} kN m", color=LOAD_C, ha="center",
          va="bottom", fontsize=ms.FS_TICK)

# shear force, with the closing jumps at the two ends
x_v = np.concatenate([[x_all[0]], x_all, [x_all[-1]]])
v_plot = np.concatenate([[0.0], v_all, [0.0]])
ax_v.fill_between(x_v, v_plot, where=v_plot >= 0, color=ms.BLUE, alpha=0.2,
                  lw=0)
ax_v.fill_between(x_v, v_plot, where=v_plot <= 0, color=ms.VERMILLION,
                  alpha=0.2, lw=0)
ax_v.plot(x_v, v_plot, color=ms.INK, lw=1.1)
jumps = ((X_PIN, r_a, 0.08, r_a + 3.6, "left"),
         (X_ROLLER, r_b, X_ROLLER + 0.1, -7.0, "left"),
         (X_P, -P, X_P - 0.1, 2.9, "right"))
for _x, size, x_text, y_text, ha in jumps:
    ax_v.text(x_text, y_text, f"{size:+.1f} kN".replace("-", "−"), ha=ha,
              va="center", fontsize=ms.FS_TICK)
ax_v.set_yticks([-10, 0, 10, 20])
ax_v.set_ylabel("Shear force V (kN)")

# bending moment, sagging positive upward
ax_m.plot(x_all, m_all, color=ms.INK, lw=1.1)
ax_m.plot(x_peak, m_peak, "o", ms=3.4, color=ms.INK, mew=0)
ax_m.text(x_peak + 0.2, m_peak + 2.0, f"Maximum {m_peak:.1f} kN m at x = "
          f"{x_peak:.2f} m, where V = 0", va="bottom", fontsize=ms.FS_TICK)
ax_m.plot(x_m0, np.zeros(len(x_m0)), "o", ms=3.6, mfc="white", mec=ms.INK,
          mew=0.8, zorder=5)
ax_m.text(X_M + 0.7, 11.5, "Points of contraflexure (M = 0)\nat x = "
          + " and ".join(f"{s:.2f}" for s in x_m0) + " m", va="center",
          fontsize=ms.FS_TICK, linespacing=1.25)
ax_m.text(X_M - 0.1, -11.0, f"Couple: +{M0:g} kN m", ha="right", va="center",
          fontsize=ms.FS_TICK)
V_LIM, M_LIM = (-19.0, 29.0), (-17.0, 29.0)
# guide from the zero of V down to the maximum of M in the panel below
for ax, span in ((ax_v, (V_LIM[0], 0.0)), (ax_m, (m_peak, M_LIM[1]))):
    ax.plot([0, LENGTH], [0, 0], color=ms.GREY, lw=0.5, zorder=0)
    ax.plot([x_peak, x_peak], span, color=ms.GREY, lw=0.5, ls=(0, (2, 1.6)),
            zorder=0)
ax_v.set_ylim(*V_LIM)
ax_m.set_ylim(*M_LIM)
ax_v.tick_params(labelbottom=False)
ax_m.set_xlim(*X_LIM)
ax_m.set_yticks([-10, 0, 10, 20])
ax_m.spines["bottom"].set_bounds(0, LENGTH)
ax_v.spines["bottom"].set_bounds(0, LENGTH)
ax_m.set_xlabel("Position along the beam x (m)")
ax_m.set_ylabel("Bending moment M\n(kN m, sagging +)")

for ax, letter in ((ax_s, "a"), (ax_v, "b"), (ax_m, "c")):
    ms.panel_label(ax, letter, dx_pt=-40, dy_pt=-4)
ms.assert_aligned([ax_s, ax_v, ax_m], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig195_shear_moment_diagram.{ext}")
print("fig195_shear_moment_diagram: saved png + pdf")
