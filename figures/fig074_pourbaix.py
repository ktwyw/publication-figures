"""Fig. 74 - Pourbaix (E-pH) diagram for iron, computed line by line.

A simplified corrosion map of the Fe-H2O system at 25 C with dissolved
species at 1e-6 M. Every boundary is a Nernst line E = E0 - 0.0592
(nH/ne) pH (shifted for the ion activity); the water-stability window
is the dashed pair (a) H+/H2 and (b) O2/H2O. A quiet self-check is
built in: three independently written boundaries (Fe2+/Fe2O3,
Fe2+/Fe3O4, Fe3O4/Fe2O3) meet at one triple point (pH 7.28), as they
must. Regions: immunity (Fe), corrosion (Fe2+, Fe3+), passivation
(Fe2O3, Fe3O4). Approximate textbook thermodynamic values.
"""

import numpy as np
import matplotlib.pyplot as plt

import journal_style as js

js.apply()
HERE = js.HERE

TOP, BOT = 1.75, -1.25

# ----------------------------------------------- boundary lines (V) ----
E_FE_FE2 = -0.617                                  # Fe/Fe2+  (at 1e-6 M)
E_FE2_FE3 = 0.771                                  # Fe2+/Fe3+
PH_FE3_OX = 1.76                                   # Fe3+/Fe2O3 (vertical)


def e_fe2_fe2o3(ph):                               # Fe2O3 + 6H+ + 2e = 2Fe2+
    return 1.083 - 0.1775 * ph


def e_fe3o4_fe2o3(ph):                             # 3Fe2O3 + 2H+ + 2e = 2Fe3O4
    return 0.221 - 0.0592 * ph


def e_fe2_fe3o4(ph):                               # Fe3O4 + 8H+ + 2e = 3Fe2+
    return 1.513 - 0.2368 * ph


def e_fe_fe3o4(ph):                                # Fe3O4 + 8H+ + 8e = 3Fe
    return -0.085 - 0.0592 * ph


# triple points (consistency: all pairs give the same pH)
PH_T = (1.083 - 0.221) / (0.1775 - 0.0592)         # 7.28
E_T = e_fe2_fe2o3(PH_T)
PH_9 = (1.513 - E_FE_FE2) / 0.2368                 # 9.0: Fe2+/Fe3O4 hits Fe line

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(4.3, 3.5))

regions = [  # polygon, facecolor, alpha
    ([(0, BOT), (14, BOT), (14, e_fe_fe3o4(14)), (PH_9, E_FE_FE2),
      (0, E_FE_FE2)], "0.90", 1.0),
    ([(0, E_FE_FE2), (PH_9, E_FE_FE2), (PH_T, E_T),
      (PH_FE3_OX, E_FE2_FE3), (0, E_FE2_FE3)], "C1", 0.20),
    ([(0, E_FE2_FE3), (PH_FE3_OX, E_FE2_FE3), (PH_FE3_OX, TOP),
      (0, TOP)], "C1", 0.34),
    ([(PH_FE3_OX, TOP), (PH_FE3_OX, E_FE2_FE3), (PH_T, E_T),
      (14, e_fe3o4_fe2o3(14)), (14, TOP)], "C2", 0.18),
    ([(PH_T, E_T), (PH_9, E_FE_FE2), (14, e_fe_fe3o4(14)),
      (14, e_fe3o4_fe2o3(14))], "C2", 0.32),
]
for poly, fc, alpha in regions:
    ax.fill(*zip(*poly), facecolor=fc, alpha=alpha, lw=0, zorder=0)

# solid boundaries
ax.plot([0, PH_9], [E_FE_FE2, E_FE_FE2], color="black", lw=1.0)
ax.plot([0, PH_FE3_OX], [E_FE2_FE3, E_FE2_FE3], color="black", lw=1.0)
ax.plot([PH_FE3_OX, PH_FE3_OX], [E_FE2_FE3, TOP], color="black", lw=1.0)
ph = np.linspace(PH_FE3_OX, PH_T, 20)
ax.plot(ph, e_fe2_fe2o3(ph), color="black", lw=1.0)
ph = np.linspace(PH_T, 14, 20)
ax.plot(ph, e_fe3o4_fe2o3(ph), color="black", lw=1.0)
ph = np.linspace(PH_T, PH_9, 20)
ax.plot(ph, e_fe2_fe3o4(ph), color="black", lw=1.0)
ph = np.linspace(PH_9, 14, 20)
ax.plot(ph, e_fe_fe3o4(ph), color="black", lw=1.0)

# water stability window (dashed)
ph = np.linspace(0, 14, 20)
ax.plot(ph, -0.0592 * ph, ls="--", lw=0.9, color="0.35")
ax.plot(ph, 1.229 - 0.0592 * ph, ls="--", lw=0.9, color="0.35")
ax.text(5.0, -0.0592 * 5.0 + 0.05, "(a) H$^+$/H$_2$", fontsize=6.5,
        color="0.35", rotation=-12)
ax.text(2.6, 1.229 - 0.0592 * 2.6 + 0.05, "(b) O$_2$/H$_2$O", fontsize=6.5,
        color="0.35", rotation=-12)

labels = [(6.8, -0.95, "immunity\n(Fe)", "0.25"),
          (3.1, 0.02, "corrosion\n(Fe$^{2+}$)", "C1"),
          (0.85, 1.32, "Fe$^{3+}$", "C1"),
          (9.3, 0.72, "passivation\n(Fe$_2$O$_3$)", "C2"),
          (11.5, -0.60, "Fe$_3$O$_4$", "C2")]
for x, y, txt, c in labels:
    ax.text(x, y, txt, ha="center", fontsize=7.5, color=c)
ax.text(13.7, -1.17, "[Fe species] = 10$^{-6}$ M, 25 \u00b0C", fontsize=6,
        ha="right", color="0.35")

ax.set_xlim(0, 14)
ax.set_ylim(BOT, TOP)
ax.set_xlabel("pH")
ax.set_ylabel("potential, $E$ vs. SHE (V)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig074_pourbaix.{ext}")
print("saved fig074_pourbaix.png / .pdf")
