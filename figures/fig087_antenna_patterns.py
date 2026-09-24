"""Fig. 87 - Antenna patterns: element and array factor (two polar panels).

Radiation patterns computed from first principles and drawn on polar
axes with a dB radial scale (floor at -40 dB).

(a) Half-wave dipole element pattern F(theta) =
    cos((pi/2) cos theta)/sin theta - the classic donut cross-section
    with nulls along the dipole axis.

(b) Uniform linear array factor for N = 8 elements at d = lambda/2,
    AF = sin(N psi/2)/(N sin(psi/2)), psi = k d cos theta + beta:
    broadside (beta = 0) and electronically steered to 60 deg from the
    axis (beta = -pi cos 60 deg), with the -13 dB first side lobe of
    the uniform taper annotated.
"""

import numpy as np
import matplotlib.pyplot as plt

import journal_style as js

js.apply()
HERE = js.HERE

FLOOR = 40.0                                       # dB below peak at origin


def to_r(db):
    return np.maximum(db, -FLOOR) + FLOOR


th = np.linspace(0, 2 * np.pi, 1441)

# ------------------------------------------------ element pattern ----
sin_t = np.maximum(np.abs(np.sin(th)), 1e-9)
F = np.abs(np.cos(np.pi / 2 * np.cos(th)) / sin_t)
F_db = 20 * np.log10(np.maximum(F / F.max(), 1e-9))

# ------------------------------------------------ array factor ----
N, KD = 8, np.pi                                   # d = lambda/2


def af_db(beta):
    psi = KD * np.cos(th) + beta
    num = np.sin(N * psi / 2)
    den = N * np.sin(psi / 2)
    af = np.abs(np.where(np.abs(den) < 1e-9, 1.0,
                         num / np.where(np.abs(den) < 1e-9, 1.0, den)))
    return 20 * np.log10(np.maximum(af, 1e-9))


fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(7.0, 3.4), subplot_kw=dict(projection="polar"))

for ax in (ax_a, ax_b):
    ax.set_theta_zero_location("N")
    ax.set_rlim(0, FLOOR + 2)
    ax.set_rgrids([10, 20, 30, 40], ["\u221230", "\u221220", "\u221210",
                                     "0 dB"], fontsize=6, color="0.4")
    ax.set_thetagrids(np.arange(0, 360, 30), fontsize=6.5)
    ax.set_rlabel_position(12)                     # quiet notch in both
    ax.grid(lw=0.4, color="0.85")

ax_a.plot(th, to_r(F_db), color="C0", lw=1.3)
ax_a.set_title("half-wave dipole element", fontsize=8, pad=14)
ax_a.annotate("nulls along\nthe dipole axis", xy=(0, 3),
              xytext=(0.84, 0.96), textcoords="axes fraction",
              fontsize=6.5, color="0.35", ha="left",
              arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.4"))

ax_b.plot(th, to_r(af_db(0.0)), color="C0", lw=1.2, label="broadside")
ax_b.plot(th, to_r(af_db(-KD * np.cos(np.radians(60)))), color="C1",
          lw=1.1, ls="--", label="steered to 60\u00b0")
ax_b.set_title("array factor, $N$ = 8, $d = \\lambda/2$", fontsize=8,
               pad=14)
th_sl = np.arccos(3.0 / (2 * N))                   # first side-lobe peak
ax_b.annotate("first side lobe\n\u221213 dB", xy=(th_sl, FLOOR - 13.3),
              xytext=(-0.04, 0.94), textcoords="axes fraction",
              fontsize=6.5, color="0.35", ha="left",
              arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.4"))
ax_b.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=2,
            fontsize=6.5, frameon=False)

js.panel_label(ax_a, "a", x=-0.02)
js.panel_label(ax_b, "b", x=-0.02)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig087_antenna_patterns.{ext}")
print("saved fig087_antenna_patterns.png / .pdf")
