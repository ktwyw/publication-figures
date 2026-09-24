"""Fig. 82 - Smith chart, computed from the mapping (single column, square).

The RF engineer's canvas, generated from first principles: the bilinear
map Gamma = (z - 1)/(z + 1) sends constant-resistance loci to circles
centred at (r/(1+r), 0) with radius 1/(1+r), and constant-reactance
loci to circles centred at (1, 1/x) with radius 1/|x|, clipped to the
unit disc. A worked two-step match is overlaid: a series line rotates
the load z_L = 0.3 - j0.4 at constant |Gamma| (toward the generator,
clockwise) onto the r = 1 circle, then a series capacitor walks it down
that circle to the centre.
"""

import numpy as np
import matplotlib.pyplot as plt

import journal_style as js

js.apply()
HERE = js.HERE


def gamma(z):
    return (z - 1) / (z + 1)


def circle(cx, cy, rad, n=400):
    t = np.linspace(0, 2 * np.pi, n)
    return cx + rad * np.cos(t), cy + rad * np.sin(t)


fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.set_aspect("equal")
ax.axis("off")

# ------------------------------------------------- chart grid ----
ax.plot(*circle(0, 0, 1), color="black", lw=1.1)
ax.plot([-1, 1], [0, 0], color="0.55", lw=0.6)

for r in [0.2, 0.5, 1, 2, 5]:
    gx, gy = circle(r / (1 + r), 0, 1 / (1 + r))
    ax.plot(gx, gy, color="0.55" if r != 1 else "0.30",
            lw=0.6 if r != 1 else 0.9)
    ax.text((r - 1) / (r + 1), 0.045, f"{r:g}", fontsize=6, color="0.35",
            ha="center")
ax.text(-1.08, 0, "0", fontsize=6, color="0.35", ha="right", va="center")
ax.text(1.08, 0, r"$\infty$", fontsize=6, color="0.35", ha="left",
        va="center")

for x in [0.2, 0.5, 1, 2, 5]:
    for sgn in (+1, -1):
        gx, gy = circle(1, sgn / x, 1 / x, n=2000)
        inside = gx**2 + gy**2 <= 1.0001
        ax.plot(gx[inside], gy[inside], color="0.55", lw=0.6)
    rim = gamma(1j * x)
    for sgn, lab in ((+1, f"j{x:g}"), (-1, f"\u2212j{x:g}")):
        ax.text(1.10 * rim.real, 1.10 * sgn * rim.imag, lab, fontsize=6,
                color="0.35", ha="center", va="center")

# --------------------------------------------- worked match ----
Z_L = 0.3 - 0.4j
g_l = gamma(Z_L)
mag = abs(g_l)

# step 1: series line, constant |Gamma|, clockwise onto the r = 1 circle
gx, gy = circle(0, 0, mag)
ax.plot(gx, gy, ls="--", lw=0.7, color="C0")                # VSWR circle
th_l = np.degrees(np.angle(g_l)) % 360
g_1 = 0.35 + 0.477j                                          # on r = 1
th_1 = np.degrees(np.angle(g_1))
th = np.radians(np.linspace(th_l, th_1, 120))                # decreasing
ax.plot(mag * np.cos(th), mag * np.sin(th), color="C0", lw=1.6, zorder=5)
k = 60
ax.annotate("", xy=(mag * np.cos(th[k + 3]), mag * np.sin(th[k + 3])),
            xytext=(mag * np.cos(th[k]), mag * np.sin(th[k])),
            arrowprops=dict(arrowstyle="-|>", lw=1.2, color="C0"))

# step 2: series capacitor, along r = 1 from x = +1.468 to 0
x_arr = np.linspace(1.468, 0, 120)
g_c = gamma(1 + 1j * x_arr)
ax.plot(g_c.real, g_c.imag, color="C1", lw=1.6, zorder=5)
ax.annotate("", xy=(g_c.real[88], g_c.imag[88]),
            xytext=(g_c.real[85], g_c.imag[85]),
            arrowprops=dict(arrowstyle="-|>", lw=1.2, color="C1"))

for pt, mfc in ((g_l, "C0"), (g_1, "white"), (0 + 0j, "C1")):
    ax.plot(pt.real, pt.imag, "o", ms=5, mfc=mfc, mec="black", mew=1.0,
            zorder=6)

ax.annotate(r"$z_L = 0.3 - j0.4$", (g_l.real, g_l.imag), xytext=(-8, -12),
            textcoords="offset points", fontsize=6.5, ha="right", color="C0")
ax.annotate(r"line, $\ell \approx 0.24\lambda$", xy=(-0.47, 0.36),
            xytext=(-1.18, 0.72), fontsize=6.5, color="C0",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="C0"))
ax.annotate(r"series C, $x = -1.47$", xy=(0.265, 0.441),
            xytext=(0.57, 0.57), fontsize=6.5, color="C1",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="C1"))
ax.text(0.035, -0.085, "match, $z = 1$", fontsize=6.5, ha="left",
        color="C1")

th_g = np.radians(np.linspace(118, 96, 40))
ax.plot(1.16 * np.cos(th_g), 1.16 * np.sin(th_g), lw=0.7, color="0.4")
ax.annotate("", xy=(1.16 * np.cos(th_g[-1]), 1.16 * np.sin(th_g[-1])),
            xytext=(1.16 * np.cos(th_g[-3]), 1.16 * np.sin(th_g[-3])),
            arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.4"))
ax.text(-0.30, 1.24, "toward generator", fontsize=6, color="0.4",
        ha="center")

ax.text(0, -1.28, r"$\Gamma = (z-1)/(z+1)$,   $z = r + jx$ (normalised)",
        fontsize=6.5, ha="center", color="0.35")
ax.set_xlim(-1.32, 1.32)
ax.set_ylim(-1.38, 1.34)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig082_smith_chart.{ext}")
print("saved fig082_smith_chart.png / .pdf")
