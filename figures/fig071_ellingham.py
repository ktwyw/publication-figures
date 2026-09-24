"""Fig. 71 - Ellingham diagram from Delta_H and Delta_S (single column, tall).

The extractive-metallurgy master chart: standard Gibbs energy of oxide
formation per mole of O2, Delta_G = Delta_H - T Delta_S, using
approximate textbook thermodynamic values. Metal-oxide lines slope
upward (gas O2 consumed); the 2C + O2 = 2CO line slopes DOWN (net gas
produced), so carbon eventually undercuts every metal line - the
marked crossing with FeO at ~1000 K is the thermodynamic basis of the
blast furnace. The Mg line kinks at the metal's boiling point.
"""

import numpy as np
import matplotlib.pyplot as plt

import journal_style as js

js.apply()
HERE = js.HERE

T = np.linspace(300, 2100, 400)

# reaction, dH (kJ / mol O2), dS (J K^-1 / mol O2), colour, label T position
LINES = [
    ("2Mg + O$_2$ = 2MgO", -1204, -216, "0.15", 700, 12),
    ("$\\frac{4}{3}$Al + O$_2$ = $\\frac{2}{3}$Al$_2$O$_3$",
     -1118, -210, "0.15", 1520, 12),
    ("Ti + O$_2$ = TiO$_2$", -944, -185, "0.15", 620, 12),
    ("2Zn + O$_2$ = 2ZnO", -696, -201, "0.15", None, 12),
    ("2Fe + O$_2$ = 2FeO", -530, -130, "0.15", None, 12),
    ("2H$_2$ + O$_2$ = 2H$_2$O", -484, -89, "C0", None, 12),
    ("2CO + O$_2$ = 2CO$_2$", -566, -173, "C1", 480, -14),
    ("C + O$_2$ = CO$_2$", -394, -0.8, "C1", 620, 12),
    ("2C + O$_2$ = 2CO", -221, 179, "C1", 1480, 12),
]
T_BP_MG, DS_MG_GAS = 1363.0, -412.0               # Mg boils; slope steepens


def gibbs(dh, ds, label):
    g = dh - T * ds / 1000.0
    if label.startswith("2Mg"):
        g_bp = dh - T_BP_MG * ds / 1000.0
        hot = T > T_BP_MG
        g[hot] = g_bp - (T[hot] - T_BP_MG) * DS_MG_GAS / 1000.0
    return g


# visual rotation of a line label (accounts for the axes aspect)
X_IN, Y_IN = 3.9 / 1800.0, 3.0 / 1140.0           # inches per K, per kJ


def rot(slope_kj_per_k):
    return np.degrees(np.arctan(slope_kj_per_k * X_IN / Y_IN))


# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(4.8, 3.7))

edge = []                                          # right-margin labels
for label, dh, ds, color, t_lab, dy in LINES:
    g = gibbs(dh, ds, label)
    ax.plot(T, g, color=color, lw=1.0)
    if t_lab is None:
        edge.append([label, color, g[-1], g[-1]])   # [.., end, label y]
        continue
    k = np.argmin(np.abs(T - t_lab))
    slope = -ds / 1000.0
    if label.startswith("2Mg") and t_lab > T_BP_MG:
        slope = -DS_MG_GAS / 1000.0
    ax.text(t_lab, g[k] + dy, label, fontsize=6, color=color,
            rotation=rot(slope), ha="center",
            va="bottom" if dy > 0 else "top", rotation_mode="anchor")

edge.sort(key=lambda e: e[2])                       # nudge upward, 40 kJ gaps
for i in range(1, len(edge)):
    edge[i][3] = max(edge[i][3], edge[i - 1][3] + 40)
for label, color, g_end, y_lab in edge:
    ax.plot([2100, 2140], [g_end, y_lab], lw=0.5, color="0.6",
            clip_on=False)
    ax.text(2155, y_lab, label, fontsize=6, color=color, va="center")

# Mg boiling-point kink
g_bp = -1204 - T_BP_MG * (-216) / 1000.0
ax.plot(T_BP_MG, g_bp, marker="|", ms=6, color="0.15", mew=1.2)
ax.text(T_BP_MG + 20, g_bp - 14, "b.p. Mg", fontsize=5.5, color="0.35",
        va="top")

# carbothermic crossing: 2C+O2=2CO undercuts 2Fe+O2=2FeO
T_X = (530 - 221) / (0.130 + 0.179)
G_X = -530 + 0.130 * T_X
ax.plot(T_X, G_X, "o", ms=4.5, mfc="white", mec="C1", mew=1.2, zorder=5)
ax.plot([T_X, T_X], [G_X, -1250], ls=":", lw=0.7, color="C1")
ax.annotate(f"C reduces FeO\nfor $T$ > {T_X:.0f} K", (T_X, G_X),
            xytext=(14, 16), textcoords="offset points", fontsize=6.5,
            color="C1")

secax = ax.secondary_xaxis("top", functions=(lambda k: k - 273.15,
                                             lambda c: c + 273.15))
secax.set_xticks([200, 600, 1000, 1400, 1800])
secax.set_xlabel("temperature (\u00b0C)", fontsize=9)
secax.tick_params(labelsize=8)

ax.set_xlim(300, 2400)
ax.set_ylim(-1250, -110)
ax.set_xlabel("temperature, $T$ (K)")
ax.set_ylabel(r"$\Delta G^\circ$ (kJ per mol O$_2$)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig071_ellingham.{ext}")
print("saved fig071_ellingham.png / .pdf")
