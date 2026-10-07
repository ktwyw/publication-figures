"""Fig. 155 - Hohmann and bi-elliptic orbit transfers (double column, 183 mm).

Everything here follows from the vis-viva equation v² = GM(2/r − 1/a).
(a) Earth-to-Mars Hohmann transfer to scale: two tangential burns join
circular orbits by half an ellipse, and Mars must lead Earth by the
phase angle at departure to be at the far apsis on arrival. (b) Total
Δv against the radius ratio for Hohmann and for bi-elliptic transfers
through an intermediate apoapsis: above a ratio of 11.94 a bi-elliptic
transfer can be cheaper, above 15.58 (the Hohmann maximum) every one
is. The self-check is that energy and angular momentum agree at both
apsides of the transfer ellipse, that the Earth-Mars Δv is 5.5-5.7
km/s over 255-262 days with a phase angle near 44°, and that brentq
finds the bi-parabolic crossover at 11.94 ± 0.01.

Model: coplanar circular orbits, impulsive burns; arrows in (a) are
proportional to Δv; filled markers show the planet the spacecraft is
at, open markers the other planet at that moment. No data: every curve
is computed from the equations.
"""

import numpy as np
from scipy import optimize

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
GM_SUN = 1.32712440018e20      # m³ s⁻², heliocentric constant (IAU / DE4xx)
AU = 1.495978707e11            # m, astronomical unit (IAU 2012, exact)
A_MARS = 1.524                 # au, Mars semi-major axis (rounded)
DAY = 86400.0                  # s
C_RATIOS = (2.0, 5.0)          # bi-elliptic apoapsis in units of r₂
R_GRID = np.geomspace(1.0, 100.0, 600)


def vis_viva(r, a, gm=1.0):
    """Orbital speed at radius r on an orbit of semi-major axis a."""
    return np.sqrt(gm * (2.0 / r - 1.0 / a))


def hohmann(ratio):
    """Total Δv / v₁ for r₁ = 1 to r₂ = ratio (GM = 1)."""
    a = 0.5 * (1 + ratio)
    return ((vis_viva(1, a) - vis_viva(1, 1))
            + (vis_viva(ratio, ratio) - vis_viva(ratio, a)))


def bi_elliptic(ratio, c):
    """Total Δv / v₁ via an apoapsis at c r₂ (c = inf: bi-parabolic)."""
    r_b = c * ratio
    a_out, a_back = 0.5 * (1 + r_b), 0.5 * (ratio + r_b)
    at_apoapsis = (0.0 if np.isinf(c) else
                   vis_viva(r_b, a_back) - vis_viva(r_b, a_out))
    return ((vis_viva(1, a_out) - vis_viva(1, 1)) + at_apoapsis
            + (vis_viva(ratio, a_back) - vis_viva(ratio, ratio)))


# ------------------------------------------------------------ SOLVER ---
# (a) Earth to Mars, SI units
r1, r2 = AU, A_MARS * AU
a_t = 0.5 * (r1 + r2)
v_peri, v_apo = vis_viva(r1, a_t, GM_SUN), vis_viva(r2, a_t, GM_SUN)
dv1 = v_peri - vis_viva(r1, r1, GM_SUN)
dv2 = vis_viva(r2, r2, GM_SUN) - v_apo
t_transfer = np.pi * np.sqrt(a_t ** 3 / GM_SUN)             # half a period
n_earth, n_mars = np.sqrt(GM_SUN / r1 ** 3), np.sqrt(GM_SUN / r2 ** 3)
phase = np.pi - n_mars * t_transfer            # Mars ahead of Earth at start
earth_end = n_earth * t_transfer               # Earth's angle at arrival

# (b) dimensionless curves and their landmarks
curves = {c: bi_elliptic(R_GRID, c) for c in C_RATIOS + (np.inf,)}
crossover = optimize.brentq(lambda x: hohmann(x) - bi_elliptic(x, np.inf),
                            5, 15, xtol=1e-12)
peak = optimize.minimize_scalar(lambda x: -hohmann(x), bounds=(5, 30),
                                method="bounded", options={"xatol": 1e-9}).x

# ------------------------------------------------------- SELF-CHECK ---
energy = [0.5 * v ** 2 - GM_SUN / r for v, r in ((v_peri, r1), (v_apo, r2))]
assert abs(energy[0] / energy[1] - 1) < 1e-12               # same orbit
assert abs(energy[0] / (-GM_SUN / (2 * a_t)) - 1) < 1e-12
assert abs(r1 * v_peri / (r2 * v_apo) - 1) < 1e-12          # r × v at apsides
assert 5.5e3 < dv1 + dv2 < 5.7e3, dv1 + dv2
assert 255 < t_transfer / DAY < 262, t_transfer / DAY
assert abs(np.degrees(phase) - 44.0) < 1.0, np.degrees(phase)
assert abs(crossover - 11.94) < 0.01, crossover
assert abs(peak - 15.58) < 0.01, peak
assert np.allclose(bi_elliptic(R_GRID, 1.0), hohmann(R_GRID), atol=1e-12)
assert np.allclose(bi_elliptic(R_GRID, 1e12), curves[np.inf], atol=1e-5)
print(f"fig155: self-check passed (Earth-Mars Δv {dv1 / 1e3:.3f} + "
      f"{dv2 / 1e3:.3f} = {(dv1 + dv2) / 1e3:.3f} km/s, "
      f"{t_transfer / DAY:.1f} days, phase angle {np.degrees(phase):.2f}°; "
      f"energy and r × v equal at both apsides; bi-parabolic crossover "
      f"{crossover:.3f}, Hohmann maximum at {peak:.3f})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 78)
X_LIM, Y_LIM = (-2.4, 2.0), (-1.8, 1.8)                    # au, panel a
A_H = 60.0
ax_a = ms.axes(fig, 12, 11, A_H * np.ptp(X_LIM) / np.ptp(Y_LIM), A_H)
ax_b = ms.axes(fig, 101, 11, 75, A_H)
ax_in = ms.axes(fig, 136, 21, 36, 18)
EARTH, MARS, BURN = ms.BLUE, ms.VERMILLION, ms.GREEN
COLOUR = {2.0: ms.ORANGE, 5.0: ms.PINK, np.inf: ms.GREY_DARK}
DASH = {2.0: "-", 5.0: "-", np.inf: (0, (4, 1.6))}


def on_circle(radius, start, stop):
    angle = np.linspace(start, stop, 361)
    return radius * np.cos(angle), radius * np.sin(angle)


def at(radius, angle):
    return radius * np.cos(angle), radius * np.sin(angle)


# a, geometry in au; perihelion of the transfer (departure) on the +x axis
ecc = (A_MARS - 1) / (A_MARS + 1)
theta = np.linspace(0, np.pi, 300)
r_ellipse = 0.5 * (1 + A_MARS) * (1 - ecc ** 2) / (1 + ecc * np.cos(theta))
for radius, colour, start, stop in ((1.0, EARTH, 0.0, earth_end),
                                    (A_MARS, MARS, phase, np.pi)):
    ax_a.plot(*on_circle(radius, 0, 2 * np.pi), color=colour, lw=0.5,
              alpha=0.45)
    ax_a.plot(*on_circle(radius, start, stop), color=colour, lw=1.3)
ax_a.plot(r_ellipse * np.cos(theta), -r_ellipse * np.sin(theta),
          color=ms.GREY, lw=0.7, ls=(0, (3, 2)))           # unused half
ax_a.plot(r_ellipse * np.cos(theta), r_ellipse * np.sin(theta),
          color=ms.INK, lw=1.7, zorder=3)
for radius, angle in ((1.0, 0.0), (A_MARS, phase)):        # phase-angle rays
    ax_a.plot([0, at(radius, angle)[0]], [0, at(radius, angle)[1]],
              color=ms.GREY, lw=0.5)
ax_a.plot(*on_circle(0.38, 0, phase), color=ms.GREY_DARK, lw=0.6)
ax_a.text(*at(0.74, 0.44 * phase), f"{np.degrees(phase):.1f}°",
          ha="center", va="center", fontsize=ms.FS_TICK)

ax_a.plot(0, 0, "o", ms=5.5, color=ms.ORANGE, mec=ms.INK, mew=0.5, zorder=5)
planet = dict(marker="o", ms=4.2, mew=0.9, zorder=6, ls="none")
ax_a.plot(1, 0, mfc=EARTH, mec=EARTH, **planet)
ax_a.plot(-A_MARS, 0, mfc=MARS, mec=MARS, **planet)
ax_a.plot(*at(1.0, earth_end), mfc="white", mec=EARTH, **planet)
ax_a.plot(*at(A_MARS, phase), mfc="white", mec=MARS, **planet)

AU_PER_KMS = 0.15                              # arrow length per km/s of Δv
burn = dict(arrowstyle="-|>", color=BURN, lw=1.5, shrinkA=0, shrinkB=0,
            mutation_scale=7)
ax_a.annotate("", xy=(1, AU_PER_KMS * dv1 / 1e3), xytext=(1, 0),
              arrowprops=burn, zorder=7)
ax_a.annotate("", xy=(-A_MARS, -AU_PER_KMS * dv2 / 1e3),
              xytext=(-A_MARS, 0), arrowprops=burn, zorder=7)

label = dict(fontsize=ms.FS_TICK, linespacing=1.2)
ax_a.text(0.86, -0.10, f"Departure\nΔv₁ = {dv1 / 1e3:.2f} km/s", ha="right",
          va="top", color=BURN, **label)
ax_a.text(-1.65, -0.20, f"Arrival\nΔv₂ =\n{dv2 / 1e3:.2f} km/s", ha="right",
          va="center", color=BURN, **label)
ax_a.text(-0.40, 0.38, f"Hohmann\ntransfer\n{t_transfer / DAY:.0f} days",
          ha="center", va="center", fontweight="bold", **label)
ax_a.text(1.2, 1.12, "Mars at\ndeparture,\n"
          f"{np.degrees(phase):.1f}° ahead", color=MARS, va="bottom", **label)
ax_a.text(-0.33, -0.55, "Earth at\narrival", color=EARTH, ha="center",
          va="center", **label)
ax_a.text(-0.08, -0.08, "Sun", ha="right", va="top", **label)
ax_a.set_xlim(*X_LIM)
ax_a.set_ylim(*Y_LIM)
ax_a.set_aspect("equal")
ax_a.set_xticks([-2, -1, 0, 1])
ax_a.set_yticks([-1, 0, 1])
ax_a.set_xlabel("x (au)")
ax_a.set_ylabel("y (au)")

# b, the same curves in the main axes and in the zoom on the crossover
ZOOM_X, ZOOM_Y = (9.0, 24.0), (0.505, 0.556)
for axis, width in ((ax_b, 1.0), (ax_in, 0.9)):
    for c, cost in curves.items():
        shown = np.ones(R_GRID.size, bool) if axis is ax_b else (
            (R_GRID >= ZOOM_X[0]) & (R_GRID <= ZOOM_X[1])
            & (cost >= ZOOM_Y[0]) & (cost <= ZOOM_Y[1]))   # trim to frame
        axis.plot(R_GRID[shown], cost[shown], color=COLOUR[c], ls=DASH[c],
                  lw=width)
    in_x = np.ones(R_GRID.size, bool) if axis is ax_b else (
        (R_GRID >= ZOOM_X[0]) & (R_GRID <= ZOOM_X[1]))
    axis.plot(R_GRID[in_x], hohmann(R_GRID[in_x]), color=ms.INK,
              lw=1.7 * width, zorder=3)
    axis.plot([crossover, peak], [hohmann(crossover), hohmann(peak)], "o",
              ms=3.0, mfc="white", mec=ms.INK, mew=0.7, zorder=4)
    axis.set_xscale("log")

ax_b.text(1.9, 0.18, "Hohmann", fontweight="bold", va="center", **label)
ax_b.text(1.08, 0.862, "Bi-elliptic via ∞ (bi-parabolic)", va="center",
          color=COLOUR[np.inf], **label)
ax_b.text(1.08, 0.637, "via 5 r₂", va="center", color=COLOUR[5.0], **label)
ax_b.text(1.08, 0.533, "via 2 r₂", va="center", color=COLOUR[2.0], **label)
ax_b.text(97, 0.80, f"Above r₂/r₁ = {crossover:.2f} a bi-elliptic\n"
          f"transfer can beat Hohmann; above\n{peak:.2f} every "
          "bi-elliptic does.", ha="right", va="top", fontsize=ms.FS_SMALL,
          color=ms.GREY_DARK, linespacing=1.25)
ax_b.plot([ZOOM_X[0], ZOOM_X[1], ZOOM_X[1], ZOOM_X[0], ZOOM_X[0]],
          [ZOOM_Y[0], ZOOM_Y[0], ZOOM_Y[1], ZOOM_Y[1], ZOOM_Y[0]],
          color=ms.GREY, lw=0.5)                           # the zoomed region
ax_b.set_xlim(1, 100)
ax_b.set_ylim(0, 0.9)
ms.plain_log_ticks(ax_b.xaxis)
ax_b.set_xlabel("Radius ratio r₂/r₁")
ax_b.set_ylabel("Total Δv / initial circular speed")

# zoom: the two landmark ratios, each tied to its point by a thin rule
ax_in.plot([crossover] * 2, [ZOOM_Y[0], hohmann(crossover) - 0.002],
           color=ms.GREY, lw=0.5)
ax_in.plot([peak] * 2, [hohmann(peak) + 0.002, ZOOM_Y[1]], color=ms.GREY,
           lw=0.5)
ax_in.text(crossover / 1.02, 0.5115, f"{crossover:.2f}", ha="right",
           va="center", fontsize=ms.FS_SMALL)
ax_in.text(peak * 1.02, 0.5495, f"{peak:.2f}", ha="left", va="center",
           fontsize=ms.FS_SMALL)
ax_in.set_xlim(*ZOOM_X)
ax_in.set_ylim(*ZOOM_Y)
ax_in.set_xticks([10, 15, 20])
ax_in.set_xticks([], minor=True)
ax_in.set_xticklabels(["10", "15", "20"])
ax_in.set_yticks([0.51, 0.53, 0.55])
ax_in.tick_params(labelsize=ms.FS_SMALL, length=2)
for side in ("top", "right"):
    ax_in.spines[side].set_visible(True)
for spine in ax_in.spines.values():
    spine.set_color(ms.GREY)
    spine.set_linewidth(0.5)

ms.panel_label(ax_a, "a", dx_pt=-26, dy_pt=6)
ms.panel_label(ax_b, "b", dx_pt=-30, dy_pt=6)
ms.assert_aligned([ax_a, ax_b])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig155_hohmann_transfer.{ext}")
print("fig155_hohmann_transfer: saved png + pdf")
