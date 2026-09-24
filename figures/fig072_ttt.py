"""Fig. 72 - Time-temperature-transformation (TTT) diagram from kinetics.

The C-curve emerges from the physics rather than being sketched: the
transformation time combines a diffusion term (slow at low T) with a
nucleation-barrier term (diverging as undercooling vanishes),

    ln t(T) = a/T + b/(T (Tm - T)^2) + const,

which is minimised at the "nose". Start (1%) and finish (99%) curves,
the eutectoid and martensite lines, and two cooling paths - a quench
that misses the nose (-> martensite) and a slow cool that intersects
the curves (-> pearlite) - complete the classic diagram. Illustrative
kinetic constants, tuned to a plain-carbon-steel-like nose.
"""

import numpy as np
import matplotlib.pyplot as plt

import journal_style as js

js.apply()
HERE = js.HERE

# ------------------------------------------------------------ theory ----
TM = 1000.0                                       # eutectoid, K (727 C)
A_DIFF, B_NUCL = 25_000.0, 3.0e8                  # K, K^3 (illustrative)
T_NOSE_TIME = 1.2                                 # s at the nose

T_k = np.linspace(560, 992, 500)
f = A_DIFF / T_k + B_NUCL / (T_k * (TM - T_k) ** 2)
t_start = T_NOSE_TIME * np.exp(np.minimum(f - f.min(), 50.0))
t_finish = 25.0 * t_start
T_c = T_k - 273.15

MS, MF = 350.0, 200.0                             # martensite start/finish, C

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(4.4, 3.2))

for t_curve, color, lab, x_lab in [(t_start, "black", "1%", 0.62),
                                   (t_finish, "0.45", "99%", 0.62)]:
    m = t_curve < 2e5
    ax.plot(t_curve[m], T_c[m], color=color, lw=1.2)
nose_i = np.argmin(t_start)
ax.text(0.80, T_c[nose_i] + 8, "1%", fontsize=7, ha="center")
ax.text(t_finish[nose_i] * 1.5, T_c[nose_i] + 6, "99%", fontsize=7,
        color="0.45")

ax.axhline(TM - 273.15, ls="--", lw=0.8, color="0.5")
ax.text(0.6, TM - 273.15 + 8, "$A_1$ (eutectoid, 727 \u00b0C)", fontsize=6.5,
        color="0.5")
for T_m, lab in [(MS, "$M_s$"), (MF, "$M_f$")]:
    ax.axhline(T_m, ls="--", lw=0.8, color="C3")
    ax.text(0.6, T_m + 6, lab, fontsize=7, color="C3")

# cooling paths
t_path = np.logspace(np.log10(0.5), np.log10(400), 300)
for rate, color, lab, t_lab, dy in [(200.0, "C0",
                                     "200 \u00b0C s$^{-1}$ \u2192 martensite",
                                     2.4, 0),
                                    (2.0, "C1",
                                     "2 \u00b0C s$^{-1}$ \u2192 pearlite",
                                     60.0, 14)]:
    T_path = 780.0 - rate * t_path
    m = (T_path > 120) & (T_path < 762)
    ax.plot(t_path[m], T_path[m], ls="--", lw=1.0, color=color)
    ax.annotate(lab, (t_lab, 780.0 - rate * t_lab), xytext=(6, dy),
                textcoords="offset points", fontsize=6.5, color=color)

ax.text(1.7, 648, "austenite\n(metastable)", fontsize=7, color="0.3")
ax.text(2.5e3, 645, "pearlite", fontsize=7, color="0.3")
ax.text(2.5e3, 420, "bainite", fontsize=7, color="0.3")
ax.text(0.62, 158, "martensite", fontsize=7, color="C3")

ax.set_xscale("log")
ax.set_xlim(0.5, 2e5)
ax.set_ylim(120, 780)
ax.set_xlabel("time, $t$ (s)")
ax.set_ylabel("temperature (\u00b0C)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig072_ttt.{ext}")
print("saved fig072_ttt.png / .pdf")
