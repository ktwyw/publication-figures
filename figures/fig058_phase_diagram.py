"""Fig. 58 - Binary eutectic phase diagram, drawn in code (single column).

A schematic A-B eutectic system: liquidus and solidus branches, solvus
lines, the eutectic isotherm and point, labelled phase fields, and one
example tie line - the figure every materials thermodynamics paper and
textbook needs, fully parameterised so it is easy to reshape.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

# ----------------------------------------------------- parameters ----
TM_A, TM_B = 600.0, 450.0            # melting points of pure A, B (deg C)
XE, TE = 62.0, 300.0                 # eutectic composition (%B), temperature
XAE, XBE = 18.0, 90.0                # solid solubility limits at TE
T_MIN = 100.0

x_l1 = np.linspace(0, XE, 120)                                   # liquidus A
liq1 = TM_A + (TE - TM_A) * (x_l1 / XE) ** 1.15
x_l2 = np.linspace(XE, 100, 120)                                 # liquidus B
liq2 = TM_B + (TE - TM_B) * ((100 - x_l2) / (100 - XE)) ** 1.15
x_s1 = np.linspace(0, XAE, 80)                                   # solidus A
sol1 = TM_A + (TE - TM_A) * (x_s1 / XAE) ** 0.8
x_s2 = np.linspace(XBE, 100, 80)                                 # solidus B
sol2 = TM_B + (TE - TM_B) * ((100 - x_s2) / (100 - XBE)) ** 0.8
t_sv = np.linspace(TE, T_MIN, 80)                                # solvus lines
svA = XAE - 12.0 * ((TE - t_sv) / (TE - T_MIN)) ** 0.7
svB = XBE + 7.0 * ((TE - t_sv) / (TE - T_MIN)) ** 0.7

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.5, 2.9))
for xx, tt in [(x_l1, liq1), (x_l2, liq2), (x_s1, sol1), (x_s2, sol2),
               (svA, t_sv), (svB, t_sv)]:
    ax.plot(xx, tt, color="black", lw=1.2)
ax.plot([XAE, XBE], [TE, TE], color="black", lw=1.2)              # eutectic line

ax.plot(XE, TE, "o", ms=4, color="C1", zorder=4)
ax.annotate(rf"E ({XE:.0f}%, {TE:.0f} °C)", (XE, TE), xytext=(4, -11),
            textcoords="offset points", fontsize=6, color="C1")

# example tie line at T = 430 C in the L + alpha field
T_TIE = 430.0
x_liq = XE * ((TM_A - T_TIE) / (TM_A - TE)) ** (1 / 1.15)
x_sol = XAE * ((TM_A - T_TIE) / (TM_A - TE)) ** (1 / 0.8)
ax.plot([x_sol, x_liq], [T_TIE, T_TIE], ls="--", lw=0.8, color="C0")
ax.plot([x_sol, x_liq], [T_TIE, T_TIE], "o", ms=3, color="C0")
ax.annotate("tie line", ((x_sol + x_liq) / 2, T_TIE), xytext=(0, 4),
            textcoords="offset points", ha="center", fontsize=5.5,
            color="C0")

labels = [("L", 46, 560), (r"L + $\alpha$", 26, 400),
          (r"L + $\beta$", 84, 365), (r"$\alpha$", 5, 220),
          (r"$\beta$", 96.5, 200), (r"$\alpha$ + $\beta$", 54, 190)]
for text, xx, tt in labels:
    ax.text(xx, tt, text, ha="center", va="center", fontsize=8)

ax.text(0, TM_A + 12, r"$T_{m,A}$", ha="center", fontsize=6)
ax.text(100, TM_B + 12, r"$T_{m,B}$", ha="center", fontsize=6)
ax.set_xlim(0, 100)
ax.set_ylim(T_MIN - 10, 660)
ax.set_xlabel("Composition, $x_B$ (at.% B)")
ax.set_ylabel("Temperature (\u00b0C)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig058_phase_diagram.{ext}")
print("saved fig058_phase_diagram.png / .pdf")
