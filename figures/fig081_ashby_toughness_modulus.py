"""Fig. 81 - Ashby chart: fracture toughness vs Young's modulus.

The damage-tolerance selection chart: family bubbles on K_IC vs E with
dotted contours of the strain-energy release rate

    G_c = K_IC^2 / E     (kJ m^-2, with K in MPa sqrt(m), E in GPa),

which are straight lines of slope 1/2 on log-log axes. Metals sit
orders of magnitude above ceramics in G_c even where K_IC looks only
tens of times larger - the chart makes the square visible. Approximate,
generic property ranges.
"""

import numpy as np
import matplotlib.pyplot as plt

import ashby
import journal_style as js

js.apply()
HERE = js.HERE

# (name, log10 E_GPa, log10 K_MPa_sqrt_m, thick, length, tilt, colour, dx, dy)
FAMILIES = [
    ("Foams", -2.10, -1.90, 0.55, 2.20, 42, "C0", 0.00, 0.00),
    ("Elastomers", -2.00, -0.75, 0.35, 1.10, 60, "C1", 0.55, 0.10),
    ("Polymers", 0.25, 0.35, 0.30, 0.85, 45, "C2", 0.52, -0.10),
    ("Woods", 0.95, 0.02, 0.28, 0.55, 40, "C3", 0.00, -0.42),
    ("Glasses", 1.85, -0.15, 0.16, 0.30, 25, "0.40", 0.00, -0.36),
    ("Composites", 1.75, 1.30, 0.28, 0.70, 42, "C4", -0.56, 0.20),
    ("Metals", 2.05, 1.65, 0.50, 1.00, 35, "C5", 0.15, 0.55),
    ("Ceramics", 2.60, 0.55, 0.32, 0.55, 30, "C6", 0.00, -0.42),
]

fig, ax = plt.subplots(figsize=(4.6, 3.7))
ashby.setup(ax, (1e-4, 1e3), (3e-3, 3e2),
            r"Young's modulus, $E$ (GPa)",
            r"fracture toughness, $K_{IC}$ (MPa m$^{1/2}$)")

# G_c contours: K = sqrt(G E)  ->  slope 1/2 through (E=1, K=sqrt(G))
for g in [1e-3, 1e-2, 1e-1, 1, 10, 100]:
    ashby.guideline(ax, 0.5, 1.0, np.sqrt(g))
    x_lab = 90.0 if g == 1e-2 else 380.0            # dodge the ceramics blob
    ax.text(x_lab, np.sqrt(g * x_lab) * 1.22, f"{g:g}", fontsize=5.5,
            color="0.45", ha="center")
ax.text(1.6e-4, 120, r"$G_c = K_{IC}^2/E$  (kJ m$^{-2}$)", fontsize=6.5,
        color="0.45")

for name, cx, cy, thick, length, tilt, color, dx, dy in FAMILIES:
    ashby.bubble(ax, cx, cy, thick, length, tilt, color, name, dx, dy)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig081_ashby_toughness_modulus.{ext}")
print("saved fig081_ashby_toughness_modulus.png / .pdf")
