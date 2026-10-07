"""Fig. 191 - Aircraft V-n manoeuvre and gust envelope (single column, 89 mm).

The flight envelope a light aircraft's structure is sized for: the
stall parabolas n = ±(V/Vs)² bound the low-speed side, the limit load
factors cap it, and the dive speed closes it. Discrete-gust lines fan
out from n = 1 at V = 0; where the gust envelope lies outside the
manoeuvre envelope it is the gust, not the pilot, that sizes the
structure. The self-check is that the corner speed found by root
finding equals Vs·√n_max to 1e-9, that the stall curve passes through
n = 1 at Vs, that the envelope polygon is closed with every vertex on
its generating curves, and that the gust load factors are linear in V.

Model: placeholder light aircraft, W/S = 700 N m⁻², C_L,max = +1.5 and
−1.0, limit load factors +3.8 and −1.52, V_C = 70 and V_D = 98 m s⁻¹
EAS; gust n = 1 ± K·ρ₀·U·V·a/(2·W/S) with the FAR 23.341 alleviation
factor K and derived gusts of 50 and 25 ft s⁻¹ at V_C and V_D. No
data: every curve is computed from the equations.
"""

import numpy as np
from scipy import optimize

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
RHO0, G0 = 1.225, 9.80665        # ISA sea-level density (kg m⁻³), g (m s⁻²)
WING_LOADING = 700.0             # W/S (N m⁻²), placeholder
CL_MAX, CL_MIN = 1.5, -1.0       # clean, positive and negative
N_MAX = 3.8                      # normal category, FAR 23.337(a)
N_MIN = -0.4 * N_MAX             # FAR 23.337(b)
N_MIN_AT_VD = 0.0                # negative limit relieved to zero at V_D
V_C, V_D = 70.0, 98.0            # design cruise and dive speeds (m s⁻¹ EAS)
LIFT_SLOPE, CHORD = 5.0, 1.5     # a (rad⁻¹), mean geometric chord (m)
FT = 0.3048
GUST = {"C": 50.0 * FT, "D": 25.0 * FT}    # U_de (m s⁻¹), FAR 23.333(c)
V_LIM, N_LIM = (0.0, 102.0), (-2.9, 4.45)


def stall_n(v, cl):
    """Load factor at the stall boundary: n = ρ₀V²C_L / (2 W/S)."""
    return RHO0 * v ** 2 * cl / (2 * WING_LOADING)


# alleviation factor, FAR 23.341(c): K = 0.88µ/(5.3 + µ), µ = 2(W/S)/(ρ c a g)
mass_ratio = 2 * WING_LOADING / (RHO0 * CHORD * LIFT_SLOPE * G0)
K_GUST = 0.88 * mass_ratio / (5.3 + mass_ratio)


def gust_n(v, u, sign):
    return 1 + sign * K_GUST * RHO0 * u * v * LIFT_SLOPE / (2 * WING_LOADING)


# --------------------------------------------------------- ENVELOPE ----
v_s = np.sqrt(2 * WING_LOADING / (RHO0 * CL_MAX))          # 1 g stall speed
v_s_neg = np.sqrt(2 * WING_LOADING / (RHO0 * -CL_MIN))
v_a = optimize.brentq(lambda v: stall_n(v, CL_MAX) - N_MAX, v_s, V_D,
                      xtol=1e-13)                          # corner speed
v_g = optimize.brentq(lambda v: stall_n(v, CL_MIN) - N_MIN, v_s_neg, V_D,
                      xtol=1e-13)                          # negative corner

v_up, v_dn = np.linspace(0, v_a, 200), np.linspace(v_g, 0, 200)
poly_v = np.concatenate([v_up, [V_D, V_D, V_C], v_dn, [0.0]])
poly_n = np.concatenate([stall_n(v_up, CL_MAX), [N_MAX, N_MIN_AT_VD, N_MIN],
                         stall_n(v_dn, CL_MIN), [0.0]])

v = np.linspace(0, V_D, 4001)
man_up = np.minimum(stall_n(v, CL_MAX), N_MAX)
man_lo = np.maximum(stall_n(v, CL_MIN), np.interp(
    v, [0, V_C, V_D], [N_MIN, N_MIN, N_MIN_AT_VD]))
gust_pts = {s: [gust_n(V_C, GUST["C"], s), gust_n(V_D, GUST["D"], s)]
            for s in (1, -1)}
# gust envelope: the V_C gust line, then straight on to the V_D gust point
gust_up = np.minimum(stall_n(v, CL_MAX), np.interp(
    v, [0, V_C, V_D], [1, *gust_pts[1]]))
gust_lo = np.maximum(stall_n(v, CL_MIN), np.interp(
    v, [0, V_C, V_D], [1, *gust_pts[-1]]))

# ------------------------------------------------------- SELF-CHECK ---
assert abs(v_a - v_s * np.sqrt(N_MAX)) < 1e-9
assert abs(stall_n(v_s, CL_MAX) - 1) < 1e-12
assert abs(stall_n(v_s_neg, CL_MIN) + 1) < 1e-12
assert poly_v[0] == poly_v[-1] and poly_n[0] == poly_n[-1]       # closed
on_stall = np.isclose(poly_n, stall_n(poly_v, CL_MAX), atol=1e-9) | \
    np.isclose(poly_n, stall_n(poly_v, CL_MIN), atol=1e-9)
on_limit = np.isclose(poly_n, N_MAX) | np.isclose(poly_n, np.interp(
    poly_v, [0, V_C, V_D], [N_MIN, N_MIN, N_MIN_AT_VD]))
assert np.all(on_stall | on_limit)
for u in GUST.values():
    line = gust_n(v, u, 1)
    assert np.abs(np.diff(line, 2)).max() < 1e-12 and line[0] == 1.0
assert gust_pts[1][0] > N_MAX and gust_pts[-1][0] < N_MIN   # gust-critical
print(f"fig191: self-check passed (Vs = {v_s:.2f}, VA = Vs·√{N_MAX:g} = "
      f"{v_a:.2f} m/s; K = {K_GUST:.3f}; gust n at VC = "
      f"{gust_pts[1][0]:+.2f}/{gust_pts[-1][0]:+.2f}, at VD = "
      f"{gust_pts[1][1]:+.2f}/{gust_pts[-1][1]:+.2f}; polygon closed, "
      f"{poly_v.size} vertices on their curves)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 72)
ax = ms.axes(fig, 12, 10.5, 66, 55)
GUST_COLOUR, DASH = ms.VERMILLION, (0, (3.5, 2))

ax.fill(poly_v, poly_n, color=ms.BLUE, alpha=0.12, lw=0)
ax.plot([0, V_D], [1, 1], color=ms.GREY, lw=0.5, ls=(0, (1, 1.6)))
# design speeds: each vertical spans the envelope at that speed
spans = {v_s: (stall_n(v_s, CL_MIN), 1.0), v_a: (N_MIN, N_MAX),
         V_C: (gust_pts[-1][0], gust_pts[1][0]),
         V_D: (gust_pts[-1][1], N_MAX)}
for speed, (lo, hi) in spans.items():
    ax.plot([speed, speed], [lo, hi], color=ms.GREY, lw=0.5)
for sign in (1, -1):
    for key, speed in (("C", V_C), ("D", V_D)):
        ax.plot([0, speed], [1, gust_n(speed, GUST[key], sign)],
                color=GUST_COLOUR, lw=0.6, ls=DASH)
for outer, inner in ((gust_up, man_up), (gust_lo, man_lo)):
    beyond = np.abs(outer - 1) > np.abs(inner - 1) + 1e-9
    ax.plot(v, np.where(beyond, outer, np.nan), color=GUST_COLOUR, lw=1.3)
ax.plot([V_D, V_D], [gust_pts[-1][1], N_MIN_AT_VD], color=GUST_COLOUR, lw=1.3)
ax.plot([V_C, V_D], gust_pts[1], color=GUST_COLOUR, lw=0.6, ls=DASH)
ax.plot(poly_v, poly_n, color=ms.BLUE, lw=1.5, solid_joinstyle="round")
ax.plot(v_s, 1, "o", ms=3.2, mfc="white", mec=ms.INK, mew=0.7, zorder=5)

top = ax.secondary_xaxis("top")
top.set_xticks([v_s, v_a, V_C, V_D], ["$V_S$", "$V_A$", "$V_C$", "$V_D$"],
               fontsize=ms.FS_MATH)
top.spines["top"].set_visible(False)

small = dict(fontsize=ms.FS_SMALL, color=ms.GREY_DARK)
ax.text(v_a - 8, 3.5, "Positive stall\n$n = (V/V_S)$²", ha="right",
        va="center", fontsize=ms.FS_MATH, color=ms.BLUE)
ax.text(v_g - 12, -1.05, "Negative stall", ha="right", va="center",
        fontsize=ms.FS_TICK, color=ms.BLUE)
ax.text(v_g - 3, N_MIN - 0.1, f"−{-N_MIN:g}", ha="right", va="top", **small)
ax.text(V_LIM[1] + 1, N_MAX, f"+{N_MAX:g}", va="center", **small)
ax.text(V_LIM[1] + 1, 1, "1 g", va="center", **small)
ax.text(84, 0.42, "Manoeuvre\nenvelope", ha="center", va="center",
        fontsize=ms.FS_TICK, color=ms.BLUE, fontweight="bold")
ax.text(84.5, -2.3, "Gust envelope", ha="center", va="center",
        fontsize=ms.FS_TICK, color=GUST_COLOUR, fontweight="bold")
ax.text(2, -2.35, f"Dashed: discrete gusts of {GUST['C']:.1f} m s⁻¹\n"
        f"at cruise and {GUST['D']:.1f} m s⁻¹ at dive speed", ha="left",
        va="center", fontsize=ms.FS_SMALL, color=GUST_COLOUR)
ax.annotate(f"1 g stall, {v_s:.1f} m s⁻¹", xy=(v_s, 1), xytext=(3, 2.75),
            va="center", **small,
            arrowprops=dict(arrowstyle="-", color=ms.GREY_DARK, lw=0.5,
                            shrinkA=1, shrinkB=3, relpos=(0.5, 0.0)))

ax.set_xlim(*V_LIM)
ax.set_ylim(*N_LIM)
ax.set_xticks(np.arange(0, 101, 20))
ax.set_yticks(np.arange(-2, 4.1, 1))
ax.spines["left"].set_bounds(-2, 4)
ax.spines["bottom"].set_bounds(0, 100)
ax.set_xlabel("Equivalent airspeed V (m s⁻¹)")
ax.set_ylabel("Load factor n")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig191_vn_diagram.{ext}")
print("fig191_vn_diagram: saved png + pdf")
