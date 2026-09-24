"""Fig. 80 - Ashby chart: strength vs density (uses ashby.py).

Second chart of the selection-chart family, now built on the shared
module: material-family bubbles in log space and minimum-mass guide
lines for ties (sigma/rho, slope 1), beams (sigma^(2/3)/rho, slope 3/2)
and panels (sigma^(1/2)/rho, slope 2), each label rotated to the axes'
true decade aspect by ashby.slope_angle. Strength means yield for
metals/polymers, tear for elastomers, flexural/crush for ceramics and
foams; all ranges are approximate, generic textbook values.
"""

import matplotlib.pyplot as plt

import ashby
import journal_style as js

js.apply()
HERE = js.HERE

# (name, log10 rho, log10 sigma_MPa, thick dec, length dec, tilt, colour,
#  label dx, dy)
FAMILIES = [
    ("Foams", -1.05, -0.20, 0.85, 2.60, 55, "C0", 0.00, 0.00),
    ("Elastomers", 0.02, 0.85, 0.28, 1.35, 72, "C1", 0.36, 0.00),
    ("Polymers", 0.06, 1.55, 0.26, 0.95, 60, "C2", -0.42, -0.30),
    ("Woods", -0.27, 1.80, 0.30, 0.85, 55, "C3", -0.50, 0.05),
    ("Composites", 0.23, 2.55, 0.22, 0.95, 58, "C4", -0.62, 0.15),
    ("Metals", 0.72, 2.35, 0.60, 1.70, 42, "C5", 0.44, -0.48),
    ("Ceramics", 0.55, 3.10, 0.32, 0.95, 45, "C6", -0.30, 0.42),
]

fig, ax = plt.subplots(figsize=(4.6, 3.7))
ashby.setup(ax, (0.01, 30), (0.01, 1e4),
            r"density, $\rho$ (Mg m$^{-3}$)",
            r"strength, $\sigma_f$ (MPa)",
            xticks=[0.01, 0.1, 1, 10])

ashby.guideline(ax, 1.0, 10, 3.0, label=r"$\sigma_f/\rho$ (ties)",
                lab_x=0.35)
ashby.guideline(ax, 1.5, 10, 30.0,
                label=r"$\sigma_f^{2/3}/\rho$ (beams)", lab_x=0.10)
ashby.guideline(ax, 2.0, 10, 200.0,
                label=r"$\sigma_f^{1/2}/\rho$ (panels)", lab_x=0.55)

for name, cx, cy, thick, length, tilt, color, dx, dy in FAMILIES:
    ashby.bubble(ax, cx, cy, thick, length, tilt, color, name, dx, dy)

ax.text(0.011, 6800,
        "guide lines: minimum-mass design\n"
        r"$\sigma_f$: yield (metals, polymers); crush/flexural"
        " (ceramics, foams)",
        fontsize=5.5, color="0.45", va="top")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig080_ashby_strength_density.{ext}")
print("saved fig080_ashby_strength_density.png / .pdf")
