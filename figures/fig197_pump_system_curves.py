"""Fig. 197 - Pump selection chart with a system curve (1.5 column, 120 mm).

How a variable-speed centrifugal pump is matched to a pipe system: one
reference head-flow curve is carried to four speeds by the affinity
laws (Q ∝ N, H ∝ N², P ∝ N³), an efficiency model that depends only on
Q/N gives iso-efficiency lines and the best-efficiency locus as
parabolas through the origin, and the system curve (static head plus
friction) picks one operating point per speed. Because of the static
head the operating points slide off the best-efficiency locus as the
pump slows down. The self-check is that every operating point found by
brentq has pump head = system head to 1e-9, that similar points obey
Q ∝ N, H ∝ N² and P ∝ N³ to 1e-9, and that the efficiency at the
best-efficiency point is the maximum of the efficiency curve.

Model: H = N²·H₀ − a·Q² (N relative to 1,450 rpm; H₀ = 50 m, 40 m at
200 m³ h⁻¹), η = η_max·x(2 − x) with x = Q/(N·Q_BEP), η_max = 82%;
system H = 18 m + k·Q²; shaft power P = ρgQH/η for water at 20 °C. The
pump and the system are placeholders. No data: every curve is computed
from the equations.
"""

import numpy as np
from scipy import optimize

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
RPM_DESIGN = 1450.0
SPEEDS = np.array([0.7, 0.8, 0.9, 1.0])      # N / N_design
H_SHUTOFF, Q_BEP, H_BEP = 50.0, 200.0, 40.0  # m, m³/h, m at design speed
ETA_MAX = 0.82
H_STATIC, K_SYSTEM = 18.0, 4.8e-4            # m, m per (m³/h)²
RHO, G = 998.2, 9.80665   # water at 20 °C, kg/m³; standard gravity, m/s²
X_END = 1.55                                 # curves drawn to Q = 1.55 Q_BEP
ETA_LEVELS = (0.60, 0.70, 0.80)
A_PUMP = (H_SHUTOFF - H_BEP) / Q_BEP ** 2


def pump_head(q, n):
    """Affinity laws: H/N² is one function of Q/N."""
    return n ** 2 * (H_SHUTOFF - A_PUMP * (q / n) ** 2)


def efficiency(q, n):
    x = q / (n * Q_BEP)
    return ETA_MAX * x * (2 - x)


def system_head(q):
    return H_STATIC + K_SYSTEM * q ** 2


def shaft_power(q, n):
    """kW; ρgQH/η with Q/η = N·Q_BEP/(η_max(2 − x)), finite at Q = 0."""
    x = q / (n * Q_BEP)
    return (RHO * G * n * Q_BEP / 3600 * pump_head(q, n)
            / (ETA_MAX * (2 - x)) / 1000)


# ------------------------------------------------------------ SOLVER ---
q_op = np.array([optimize.brentq(
    lambda q: pump_head(q, n) - system_head(q), 0.0, 2 * n * Q_BEP,
    xtol=1e-13, rtol=1e-14) for n in SPEEDS])
h_op = system_head(q_op)
eta_op = efficiency(q_op, SPEEDS)
power_op = shaft_power(q_op, SPEEDS)
q_bep, h_bep = SPEEDS * Q_BEP, pump_head(SPEEDS * Q_BEP, SPEEDS)
power_bep = shaft_power(q_bep, SPEEDS)

# ------------------------------------------------------- SELF-CHECK ---
assert np.abs(pump_head(q_op, SPEEDS) - h_op).max() < 1e-9
assert np.abs(h_bep / (SPEEDS ** 2 * H_BEP) - 1).max() < 1e-9      # H ∝ N²
assert np.abs(h_bep / q_bep ** 2 - H_BEP / Q_BEP ** 2).max() < 1e-12
assert np.abs(power_bep / (SPEEDS ** 3 * power_bep[-1]) - 1).max() < 1e-9
for n, qb in zip(SPEEDS, q_bep):
    q_scan = np.linspace(0, X_END * n * Q_BEP, 2001)
    assert efficiency(q_scan, n).max() <= efficiency(qb, n) + 1e-12
    assert abs(efficiency(qb, n) - ETA_MAX) < 1e-12
hydraulic_kw = RHO * G * q_op / 3600 * h_op / 1000
assert np.abs(power_op * eta_op / hydraulic_kw - 1).max() < 1e-12  # P=ρgQH/η
x_op = q_op / q_bep            # static head: slower means further left
assert np.all(np.diff(x_op) > 0) and x_op[0] < 1 < x_op[-1]
print(f"fig197: self-check passed (design point Q = {q_op[-1]:.1f} m3/h, "
      f"H = {h_op[-1]:.2f} m, eta = {eta_op[-1]:.3f}, P = {power_op[-1]:.2f} "
      f"kW; head residual {np.abs(pump_head(q_op, SPEEDS) - h_op).max():.1e}"
      f" m; BEP power {power_bep[0]:.2f} to {power_bep[-1]:.2f} kW ∝ N³; "
      f"efficiency at the lowest speed {eta_op[0]:.3f})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 102)
gs = ms.grid(fig, 2, 1, left=14, right=4, top=6, bottom=11, hspace=5,
             height_ratios=[3.3, 1])
ax = fig.add_subplot(gs[0])
ax_p = fig.add_subplot(gs[1], sharex=ax)
SALIENT = ms.VERMILLION
n_span = np.linspace(SPEEDS[0], SPEEDS[-1], 60)

# iso-efficiency lines: all points with the same Q/N, between the lowest
# and the highest speed; both roots x = 1 ± √(1 − η/η_max) are drawn
for level in ETA_LEVELS:
    for sign in (-1, 1):
        x = 1 + sign * np.sqrt(1 - level / ETA_MAX)
        ax.plot(x * n_span * Q_BEP, pump_head(x * n_span * Q_BEP, n_span),
                color=ms.GREY, lw=0.5)
        ax.annotate(f"{level:.0%}", xy=(x * Q_BEP, pump_head(x * Q_BEP, 1.0)),
                    xytext=(0, 5), textcoords="offset points", ha="center",
                    va="bottom", fontsize=ms.FS_SMALL, color=ms.GREY_DARK)
q_locus = np.linspace(0, Q_BEP, 100)
ax.plot(q_locus, H_BEP * (q_locus / Q_BEP) ** 2, color=ms.GREY_DARK, lw=0.7,
        ls=(0, (4, 2)))
ax.annotate(f"{ETA_MAX:.0%}", xy=(Q_BEP, H_BEP), xytext=(0, 5),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=ms.FS_SMALL, fontweight="bold")
ax.text(72, 2.2, "Best-efficiency locus, H ∝ Q²", fontsize=ms.FS_TICK,
        color=ms.GREY_DARK, va="center")
ax.text(Q_BEP * 0.4, H_SHUTOFF * 1.075, "Iso-efficiency lines (constant Q/N)",
        fontsize=ms.FS_TICK, color=ms.GREY_DARK, va="bottom")

for n in SPEEDS:
    q = np.linspace(0, X_END * n * Q_BEP, 200)
    design = n == SPEEDS[-1]
    ax.plot(q, pump_head(q, n), color=ms.INK, lw=1.3 if design else 0.9)
    ax.annotate(f"{n * RPM_DESIGN:,.0f} rpm", xy=(q[-1], pump_head(q[-1], n)),
                xytext=(3, 0), textcoords="offset points", va="center",
                fontsize=ms.FS_TICK,
                fontweight="bold" if design else "normal")
ax.plot(q_bep, h_bep, "o", ms=3, mfc="white", mec=ms.INK, mew=0.7, zorder=4)

# the system curve stops at the design-speed operating point: no speed
# on this chart can push more flow through the system
q_sys = np.linspace(0, q_op[-1], 200)
ax.plot(q_sys, system_head(q_sys), color=SALIENT, lw=1.6, zorder=3)
ax.plot(q_op, h_op, "o", ms=4.2, color=SALIENT, mec="white", mew=0.6,
        zorder=5)
ax.text(5, H_STATIC - 1.6, "System curve\n"
        f"H = {H_STATIC:.0f} m + k·Q²", color=SALIENT, fontweight="bold",
        fontsize=ms.FS_TICK, va="top", linespacing=1.3)
ax.text(258, 57.5, f"Operating point, {RPM_DESIGN:,.0f} rpm", va="top",
        fontsize=ms.FS_TICK, fontweight="bold")
rows = [("Flow Q", f"{q_op[-1]:.0f} m³ h⁻¹", ms.INK),
        ("Head H", f"{h_op[-1]:.1f} m", ms.INK),
        ("Efficiency η", f"{eta_op[-1]:.1%}", ms.INK),
        ("Shaft power P", f"{power_op[-1]:.1f} kW", ms.INK),
        (f"η at {SPEEDS[0] * RPM_DESIGN:,.0f} rpm", f"{eta_op[0]:.1%}",
         ms.GREY_DARK)]
for i, (name, value, colour) in enumerate(rows):
    for x, cell, align in ((258, name, "left"), (386, value, "right")):
        ax.text(x, 54.1 - 3.3 * i, cell, fontsize=ms.FS_TICK, va="top",
                ha=align, color=colour)

ax.set_xlim(0, 390)
ax.set_ylim(0, 58)
ax.set_yticks(np.arange(0, 51, 10))
ax.spines["left"].set_bounds(0, 50)
ax.tick_params(labelbottom=False)
ax.set_ylabel("Head H (m)")

# shaft power at design speed on the same flow axis
q = np.linspace(0, X_END * Q_BEP, 200)
ax_p.plot(q, shaft_power(q, 1.0), color=ms.INK, lw=1.3)
ax_p.plot(Q_BEP, power_bep[-1], "o", ms=3, mfc="white", mec=ms.INK, mew=0.7,
          zorder=4)
ax_p.plot(q_op[-1], power_op[-1], "o", ms=4.2, color=SALIENT, mec="white",
          mew=0.6, zorder=5)
ax_p.annotate(f"{RPM_DESIGN:,.0f} rpm", xy=(q[-1], shaft_power(q[-1], 1.0)),
              xytext=(3, 0), textcoords="offset points", va="center",
              fontsize=ms.FS_TICK, fontweight="bold")
ax_p.text(5, 39.5, "Filled, operating point; open, best-efficiency point",
          fontsize=ms.FS_SMALL, va="top", color=ms.GREY_DARK)
ax_p.set_ylim(10, 40)
ax_p.set_yticks([10, 20, 30, 40])
ax_p.set_xlabel("Flow rate Q (m³ h⁻¹)")
ax_p.set_ylabel("Shaft power\nP (kW)")

ms.assert_aligned([ax, ax_p], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig197_pump_system_curves.{ext}")
print("fig197_pump_system_curves: saved png + pdf")
