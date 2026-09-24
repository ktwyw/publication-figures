"""Fig. 75 - Ashby material-property chart: Young's modulus vs density.

The materials-selection classic: material families as ellipses on
log-log axes (drawn parametrically in log space, so they stay elliptical
on the chart), spanning approximate, generic property ranges, with the
minimum-mass design guidelines E/rho, E^(1/2)/rho and E^(1/3)/rho as
dashed lines of slope 1, 2 and 3. Label rotations are computed from the
axes' decade aspect so the guideline labels lie along their lines.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

import journal_style as js

js.apply()
HERE = js.HERE

# family: (log10 rho center, log10 E center, thickness dec, length dec,
#          tilt deg of the long axis in log space, colour, label dx, dy)
FAMILIES = [
    ("Foams", -1.05, -2.10, 1.05, 3.10, 62, "C0", 0.0, 0.0),
    ("Elastomers", 0.02, -2.00, 0.26, 1.80, 75, "C1", 0.30, 0.0),
    ("Polymers", 0.06, 0.25, 0.26, 0.85, 60, "C2", 0.42, -0.1),
    ("Woods", -0.27, 0.95, 0.33, 0.75, 55, "C3", -0.52, 0.0),
    ("Composites", 0.23, 1.75, 0.20, 0.80, 55, "C4", -0.72, 0.1),
    ("Metals", 0.72, 2.02, 0.72, 0.90, 40, "C5", 0.30, -0.55),
    ("Ceramics", 0.55, 2.62, 0.30, 0.55, 45, "C6", -0.25, 0.42),
]


def log_ellipse(ax, cx, cy, w, h, ang, color, label, dx, dy):
    t = np.linspace(0, 2 * np.pi, 200)
    a, b = h / 2, w / 2                            # major axis along the tilt
    ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    lx = cx + a * np.cos(t) * ca - b * np.sin(t) * sa
    ly = cy + a * np.cos(t) * sa + b * np.sin(t) * ca
    ax.fill(10**lx, 10**ly, facecolor=color, alpha=0.30, lw=0, zorder=2)
    ax.plot(10**lx, 10**ly, color=color, lw=0.9, zorder=3)
    ax.text(10 ** (cx + dx), 10 ** (cy + dy), label, ha="center",
            va="center", fontsize=7, color=color, zorder=4,
            fontweight="bold")


# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(4.6, 3.7))

# guide lines of slope n through (rho, E) = (10, 1e-3)
rho = np.logspace(-2, np.log10(30), 100)
X_IN, Y_IN = 3.7 / 3.48, 2.9 / 7.0                 # inches per decade
for n in (1, 2, 3):
    ax.plot(rho, 1e-3 * (rho / 10.0) ** n, ls="--", lw=0.8, color="0.45",
            zorder=1)
    ang = np.degrees(np.arctan(n * Y_IN / X_IN))
    sup = "" if n == 1 else rf"^{{1/{n}}}"
    rho_lab = {1: 2.2, 2: 24.0, 3: 26.0}[n]
    e_lab = {1: 3.1e-4, 2: 6.5e-3, 3: 1.7e-2}[n]
    ax.text(rho_lab, e_lab, rf"$E{sup}/\rho$", rotation=ang, fontsize=6.5,
            color="0.35", rotation_mode="anchor", ha="center")
ax.text(0.011, 1.6e-4, "minimum-mass\ndesign guides", fontsize=6,
        color="0.45")

for name, cx, cy, w, h, ang, color, dx, dy in FAMILIES:
    log_ellipse(ax, cx, cy, w, h, ang, color, name, dx, dy)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.01, 30)
ax.set_ylim(1e-4, 2e3)
ax.set_xticks([0.01, 0.1, 1, 10])
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"density, $\rho$ (Mg m$^{-3}$)")
ax.set_ylabel(r"Young's modulus, $E$ (GPa)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig075_ashby.{ext}")
print("saved fig075_ashby.png / .pdf")
