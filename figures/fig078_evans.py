"""Fig. 78 - Evans diagram: mixed-potential theory (single column).

Corrosion as the intersection of half-reaction polarisation lines on
E vs log|i| axes: the anodic metal-dissolution Tafel line meets each
cathodic line at a mixed potential, fixing (E_corr, i_corr). Two
environments are compared: an acid where hydrogen evolution is
activation-controlled, and an aerated neutral solution where oxygen
reduction hits its diffusion limit - there the anodic line intersects
the vertical i_L segment, so i_corr = i_L regardless of kinetics.
Illustrative kinetic parameters.
"""

import numpy as np
import matplotlib.pyplot as plt

import journal_style as js

js.apply()
HERE = js.HERE

# ---------------------------------------------------- half-reactions ----
E_A0, I0_A, B_A = -0.44, 1e-7, 0.060               # Fe -> Fe2+ (anodic)
E_H0, I0_H, B_H = 0.00, 1e-6, 0.120                # 2H+ + 2e -> H2
E_O0, I0_O, B_O = 0.80, 1e-7, 0.120                # O2 reduction
I_L = 5e-5                                         # O2 diffusion limit

log_i = np.linspace(-8, -2, 200)


def e_anodic(li):
    return E_A0 + B_A * (li - np.log10(I0_A))


def e_cath(li, e0, i0, b):
    return e0 - b * (li - np.log10(i0))


# intersections
li_h = (E_H0 - E_A0 + B_A * np.log10(I0_A) + B_H * np.log10(I0_H)) \
    / (B_A + B_H)
e_h = e_anodic(li_h)
e_o = e_anodic(np.log10(I_L))                      # diffusion-controlled case

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.6, 3.1))

ax.plot(log_i, e_anodic(log_i), color="C3", lw=1.2)
ax.plot(log_i, e_cath(log_i, E_H0, I0_H, B_H), color="C0", lw=1.2)

li_o = np.linspace(-8, np.log10(I_L), 100)         # O2: Tafel then vertical
ax.plot(li_o, e_cath(li_o, E_O0, I0_O, B_O), color="C2", lw=1.2)
ax.plot([np.log10(I_L)] * 2,
        [e_cath(np.log10(I_L), E_O0, I0_O, B_O), -0.62],
        color="C2", lw=1.2)

for li, e, lab, dx in [(li_h, e_h, "acid:\n$E_{corr},\\ i_{corr}$", 0.15),
                       (np.log10(I_L), e_o,
                        "aerated:\n$i_{corr} = i_L$", -0.15)]:
    ax.plot(li, e, "o", ms=5, mfc="white", mec="black", mew=1.2, zorder=5)
    ax.plot([li, li], [-0.75, e], ls=":", lw=0.7, color="0.5", zorder=1)
    ax.text(li + dx, -0.73, lab, fontsize=6.5, va="bottom",
            ha="left" if dx > 0 else "right")

ax.text(-6.5, -0.372, "Fe $\\rightarrow$ Fe$^{2+}$ + 2e$^-$", rotation=10,
        fontsize=6.5, color="C3", ha="center")
ax.text(-6.2, 0.085, "2H$^+$ + 2e$^-$ $\\rightarrow$ H$_2$", rotation=-20,
        fontsize=6.5, color="C0", ha="center")
ax.text(-6.6, 0.60, "O$_2$ + 4e$^-$ $\\rightarrow$ ...", rotation=-20,
        fontsize=6.5, color="C2", ha="center")
ax.text(np.log10(I_L) + 0.07, -0.05, "diffusion\nlimit, $i_L$", fontsize=6,
        color="C2", va="center")

ax.set_xlim(-8, -2)
ax.set_ylim(-0.78, 0.85)
ax.set_xlabel(r"$\log_{10}|i|$  ($i$ in A cm$^{-2}$)")
ax.set_ylabel(r"potential, $E$ vs. SHE (V)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig078_evans.{ext}")
print("saved fig078_evans.png / .pdf")
