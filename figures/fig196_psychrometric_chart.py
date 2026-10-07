"""Fig. 196 - Psychrometric chart, cooling process (double column, 183 mm).

A psychrometric chart built from three equations instead of a scanned
template: the saturation curve, relative-humidity curves, and the two
oblique families (constant enthalpy, constant wet-bulb temperature)
that practitioners read loads from, with humidity ratio on the
right-hand axis as convention has it. On it, summer air conditioning as
three numbered states: outdoor air is cooled and dehumidified at the
coil, then reheated to the supply condition; a comfort zone is filled
for reference. The self-check is that 25 °C and 50% RH give 9.9 g kg⁻¹
and 50.3 kJ kg⁻¹, that wet-bulb equals dry-bulb on the saturation
curve, that every enthalpy line has constant h to 1e-9, and that the
dew point of state 1 (brentq) lies on the drawn saturation curve.

Model: sea level, p = 101.325 kPa; saturation pressure over liquid water
(also below 0 °C) from the Magnus form; W = 0.622·p_w/(p − p_w);
h = 1.006·T + W·(2501 + 1.86·T) kJ per kg dry air; wet-bulb lines from
the ASHRAE psychrometric equation. The comfort zone (20-26 °C, 30-60%
RH) and the three states are illustrative. No data: every curve is
computed from the equations.
"""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import optimize

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
P_ATM = 101.325                              # kPa, standard atmosphere
T_LIM, W_MAX = (-10.0, 50.0), 30.0           # °C; g per kg dry air
RH_LEVELS = np.arange(0.1, 0.95, 0.1)
H_LEVELS = np.arange(0, 121, 10)             # kJ per kg dry air
WB_LEVELS = np.arange(0, 36, 5)              # °C
H_LABELLED, WB_LABELLED = (10, 30, 50, 70, 90), (5, 15, 20, 25, 30)
LABEL_LANE = 85.0                            # RH labels sit near this h
COMFORT_T, COMFORT_RH = (20.0, 26.0), (0.30, 0.60)
STATES = {1: ("Outdoor air", 34.0, 0.50),    # name, dry-bulb °C, RH
          2: ("Leaving the coil", 13.0, 0.95)}
T_SUPPLY = 19.0                              # state 3: reheated at constant W


def p_sat(t):
    """kPa; Magnus form, coefficients of Alduchov & Eskridge (1996)."""
    return 0.61094 * np.exp(17.625 * t / (t + 243.04))


def w_from_pw(p_w):
    return 622.0 * p_w / (P_ATM - p_w)       # g/kg; 0.622 = M_water/M_air


def w_rh(t, rh):
    return w_from_pw(rh * p_sat(t))


def rel_humidity(t, w):
    return w * P_ATM / (622.0 + w) / p_sat(t)


def enthalpy(t, w):
    return 1.006 * t + w / 1000 * (2501 + 1.86 * t)


def w_enthalpy(t, h):
    return 1000 * (h - 1.006 * t) / (2501 + 1.86 * t)


def w_wet_bulb(t, t_wb):
    """ASHRAE Fundamentals psychrometric equation (SI), g/kg."""
    return ((2501 - 2.326 * t_wb) * w_rh(t_wb, 1.0) - 1006 * (t - t_wb)) / (
        2501 + 1.86 * t - 4.186 * t_wb)


# ------------------------------------------------------------ SOLVER ---
def wet_bulb(t, w):
    # the bracket reaches past t so that saturated air is a sign change
    return optimize.brentq(lambda t_wb: w_wet_bulb(t, t_wb) - w, -40, t + 1,
                           xtol=1e-12)


def dew_point(w):
    return optimize.brentq(lambda t: w_rh(t, 1.0) - w, -60, 60, xtol=1e-12)


def span(w_of_t):
    """Dry-bulb range over which an oblique line lies inside the chart."""
    def ceiling(t):
        return w_of_t(t) - min(w_rh(t, 1.0), W_MAX)
    start = (T_LIM[0] if ceiling(T_LIM[0]) < 0 else
             optimize.brentq(ceiling, *T_LIM, xtol=1e-12))
    end = (T_LIM[1] if w_of_t(T_LIM[1]) > 0 else
           optimize.brentq(w_of_t, start, T_LIM[1], xtol=1e-12))
    return np.linspace(start, end, 400)


states = {k: (name, t, w_rh(t, rh)) for k, (name, t, rh) in STATES.items()}
states[3] = ("Supply air, reheated", T_SUPPLY, states[2][2])
table = {k: (t, rel_humidity(t, w), w, enthalpy(t, w), wet_bulb(t, w))
         for k, (_name, t, w) in states.items()}
t_dew = dew_point(states[1][2])
t_top = dew_point(W_MAX)                     # saturation leaves the frame
t_sat = np.linspace(T_LIM[0], t_top, 600)
h_lines = {h: span(lambda t, h=h: w_enthalpy(t, h)) for h in H_LEVELS}
wb_lines = {t_wb: span(lambda t, t_wb=t_wb: w_wet_bulb(t, t_wb))
            for t_wb in WB_LEVELS}

# ------------------------------------------------------- SELF-CHECK ---
w_ref = w_rh(25.0, 0.5)
assert abs(w_ref - 9.9) < 0.1 and abs(enthalpy(25.0, w_ref) - 50.3) < 0.5
assert abs(rel_humidity(25.0, w_ref) - 0.5) < 1e-12
for t in (0.0, 10.0, 20.0, 30.0):
    assert abs(wet_bulb(t, w_rh(t, 1.0)) - t) < 1e-8       # saturated air
    assert abs(w_wet_bulb(t, t) - w_rh(t, 1.0)) < 1e-12
h_spread = max(np.ptp(enthalpy(t, w_enthalpy(t, h)))
               for h, t in h_lines.items())
assert h_spread < 1e-9, h_spread
assert abs(w_rh(t_dew, 1.0) - states[1][2]) < 1e-9
assert abs(np.interp(t_dew, t_sat, w_rh(t_sat, 1.0)) - states[1][2]) < 1e-3
assert table[1][4] > t_dew and table[3][3] > table[2][3]   # T_wb > T_dew
print(f"fig196: self-check passed (25 °C, 50% RH: W = {w_ref:.2f} g/kg, "
      f"h = {enthalpy(25.0, w_ref):.2f} kJ/kg; wet-bulb = dry-bulb at "
      f"saturation; enthalpy lines constant within {h_spread:.1e}; dew "
      f"point of state 1 = {t_dew:.2f} °C on the saturation curve; coil "
      f"load {table[1][3] - table[2][3]:.1f} kJ/kg)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 118)
AX_W, AX_H = 160.0, 101.0                    # plot area, mm
ax = ms.axes(fig, 8, 11, AX_W, AX_H)
SCALE = np.array([AX_W / np.ptp(T_LIM), AX_H / W_MAX])     # mm per unit
obstacles = []                               # drawn strokes, in mm


def stroke(t, w, **style):
    """Plot a line and remember where it runs, for label placement."""
    ax.plot(t, w, **style)
    obstacles.append(resample(np.column_stack([t, w])))


def resample(path, n=400):
    """n points along a polyline, in millimetres on the page."""
    s_old, s_new = np.linspace(0, 1, len(path)), np.linspace(0, 1, n)
    return np.column_stack([np.interp(s_new, s_old, column)
                            for column in path.T]) * SCALE


def clearance(centres, half, cloud):
    """Gap (mm) between label boxes at centres and the nearest stroke."""
    offset = np.abs(cloud[None] - centres[:, None] * SCALE)
    return (offset - half).max(axis=2).min(axis=1)


for h, t in h_lines.items():
    stroke(t, w_enthalpy(t, h), color=ms.GREY_LIGHT, lw=0.5)
    if h in H_LABELLED:
        ax.annotate(f"{h}", xy=(t[0], w_enthalpy(t[0], h)), xytext=(-3, 1),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=ms.FS_SMALL, color=ms.GREY_DARK)
for t_wb, t in wb_lines.items():
    stroke(t, w_wet_bulb(t, t_wb), color=ms.BLUE, lw=0.4, ls=(0, (4, 3)))
    if t_wb in WB_LABELLED:
        ax.annotate(f"{t_wb} °C", xy=(t[0], w_wet_bulb(t[0], t_wb)),
                    xytext=(-3, 1), textcoords="offset points", ha="right",
                    va="bottom", fontsize=ms.FS_SMALL, color=ms.BLUE)
stroke(t_sat, w_rh(t_sat, 1.0), color=ms.INK, lw=1.4, zorder=4)

# comfort zone between two RH curves and two dry-bulb temperatures
t_zone = np.linspace(*COMFORT_T, 50)
zone = np.column_stack([np.r_[t_zone, t_zone[::-1]],
                        np.r_[w_rh(t_zone, COMFORT_RH[0]),
                              w_rh(t_zone[::-1], COMFORT_RH[1])]])
ax.fill(*zone.T, color=ms.GREEN, alpha=0.22, lw=0)
# labels keep clear of the bounding box of the fill, not only its outline
corners = [(t, w) for t in COMFORT_T for w in (zone[:, 1].min(),
                                               zone[:, 1].max())]
obstacles.append(resample(np.array(corners)[[0, 1, 3, 2, 0]], 1200))

# process 1 -> 2 -> 3 and the dew point of state 1
points = np.array([states[k][1:] for k in (1, 2, 3)])
stroke([t_dew, points[0, 0]], [points[0, 1]] * 2, color=ms.VERMILLION,
       lw=0.6, ls=(0, (1, 1.6)))
for a, b in ((0, 1), (1, 2)):
    stroke(points[[a, b], 0], points[[a, b], 1], color=ms.VERMILLION, lw=1.4,
           zorder=5)
    ax.annotate("", xy=0.42 * points[a] + 0.58 * points[b], xytext=points[a],
                zorder=5, arrowprops=dict(
                    arrowstyle="-|>,head_length=0.55,head_width=0.28",
                    color=ms.VERMILLION, lw=0, shrinkA=0, shrinkB=0,
                    mutation_scale=12))      # head only, part-way along
ax.plot(*points.T, "o", ms=4, color=ms.VERMILLION, mec="white", mew=0.5,
        zorder=6)
ax.plot(t_dew, points[0, 1], "o", ms=3.4, mfc="white", mec=ms.VERMILLION,
        mew=0.8, zorder=6)

# RH curves are computed before they are drawn, so that every label can
# take the clearest spot among all strokes; rotated text is avoided
# because its bounding box would cut across the neighbouring lines
rh_curves = {}
for rh in RH_LEVELS:
    t_end = (T_LIM[1] if w_rh(T_LIM[1], rh) < W_MAX else optimize.brentq(
        lambda t, rh=rh: w_rh(t, rh) - W_MAX, *T_LIM, xtol=1e-12))
    t = np.linspace(T_LIM[0], t_end, 500)
    rh_curves[rh] = np.column_stack([t, w_rh(t, rh)])
rh_cloud = {rh: resample(curve) for rh, curve in rh_curves.items()}
ring = 3.4 * np.exp(1j * np.linspace(0, 2 * np.pi, 48, endpoint=False))
NUMBER_HALF, RH_HALF = np.array([1.1, 1.5]), np.array([2.5, 1.3])    # mm
box = np.array([[i, j] for i in np.linspace(-1, 1, 5)
                for j in np.linspace(-1, 1, 5)])
for k, point in zip((1, 2, 3), points):
    spots = point + np.column_stack([ring.real, ring.imag]) / SCALE
    cloud = np.vstack(obstacles + list(rh_cloud.values()))
    best = spots[np.argmax(clearance(spots, NUMBER_HALF, cloud))]
    ax.text(*best, str(k), ha="center", va="center", fontweight="bold",
            color=ms.VERMILLION)
    obstacles.append(best * SCALE + box * NUMBER_HALF)
worst = np.inf
for rh, curve in rh_curves.items():
    cloud = np.vstack(obstacles + [c for r, c in rh_cloud.items() if r != rh])
    inside = (curve[:, 0] > 26) & (curve[:, 0] < 47) & (curve[:, 1] < 27)
    gap = clearance(curve[inside], RH_HALF, cloud)
    # among clear spots, prefer the corridor between two enthalpy lines
    off_lane = np.abs(enthalpy(*curve[inside].T) - LABEL_LANE)
    choice = np.argmin(np.where(gap >= 0.8, off_lane, np.inf))
    spot = curve[inside][choice]
    worst = min(worst, gap[choice])
    obstacles.append(spot * SCALE + box * RH_HALF)
    near = np.all(np.abs((curve - spot) * SCALE) < RH_HALF + 0.5, axis=1)
    ax.plot(*np.where(near[:, None], np.nan, curve).T, color=ms.GREY, lw=0.6,
            zorder=3)
    ax.text(*spot, f"{rh:.0%}", ha="center", va="center",
            fontsize=ms.FS_SMALL, color=ms.GREY_DARK)
assert worst >= 0.8, worst                   # every RH label has clear space

# key and state table in the empty region left of the saturation curve
handles = [
    Line2D([], [], color=ms.INK, lw=1.4, label="Saturation (100% RH)"),
    Line2D([], [], color=ms.GREY, lw=0.6, label="Relative humidity"),
    Line2D([], [], color=ms.GREY_LIGHT, lw=0.8,
           label="Enthalpy h (kJ kg⁻¹ dry air)"),
    Line2D([], [], color=ms.BLUE, lw=0.5, ls=(0, (4, 3)),
           label="Wet-bulb temperature"),
    Patch(color=ms.GREEN, alpha=0.22, lw=0,
          label=f"Comfort zone, {COMFORT_T[0]:.0f}–{COMFORT_T[1]:.0f} °C, "
          f"{100 * COMFORT_RH[0]:.0f}–{COMFORT_RH[1]:.0%} RH"),
    Line2D([], [], color=ms.VERMILLION, lw=1.4, marker="o", ms=3,
           label="Cooling and dehumidifying, then reheat")]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.70),
          handlelength=2.4, fontsize=ms.FS_TICK)
COLS = (-9.5, 3.4, 8.2, 14.2, 20.0, 25.5)    # column edges, on the °C axis
header = ("State", "T (°C)", "RH (%)", "W (g kg⁻¹)", "h (kJ kg⁻¹)",
          "T wb (°C)")
rows = [header] + [(f"{k}  {states[k][0]}", f"{t:.1f}", f"{100 * rh:.0f}",
                    f"{w:.2f}", f"{h:.1f}", f"{t_wb:.1f}")
                   for k, (t, rh, w, h, t_wb) in table.items()]
for i, cells in enumerate(rows):
    for x, cell in zip(COLS, cells):
        ax.text(x, 29.6 - 1.0 * i, cell,
                ha="left" if x == COLS[0] else "right", va="top",
                fontsize=ms.FS_TICK, fontweight="bold" if i == 0 else None)
ax.text(COLS[0], 29.6 - 1.0 * 4.4,
        f"Dew point of state 1 (open circle): {t_dew:.1f} °C\n"
        f"Coil load h₁ − h₂ = {table[1][3] - table[2][3]:.1f} kJ kg⁻¹; "
        f"reheat h₃ − h₂ = {table[3][3] - table[2][3]:.1f} kJ kg⁻¹\n"
        f"Moisture removed W₁ − W₂ = {table[1][2] - table[2][2]:.2f} g kg⁻¹; "
        f"p = {P_ATM} kPa", va="top", fontsize=ms.FS_SMALL,
        color=ms.GREY_DARK, linespacing=1.35)

ax.set_xlim(*T_LIM)
ax.set_ylim(0, W_MAX)
ax.set_xticks(np.arange(-10, 51, 5))
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.spines["right"].set_visible(True)
ax.spines["left"].set_visible(False)
ax.set_xlabel("Dry-bulb temperature (°C)")
ax.set_ylabel("Humidity ratio W (g kg⁻¹ dry air)")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig196_psychrometric_chart.{ext}")
print("fig196_psychrometric_chart: saved png + pdf")
