"""Fig. 85 - Mohr-Coulomb failure: circles tangent by construction.

Geotechnical engineering's central diagram, generated so that tangency
is exact rather than drawn: for each confining stress sigma_3 the
failure stress follows

    sigma_1 = sigma_3 tan^2(45 + phi/2) + 2c tan(45 + phi/2),

and the resulting Mohr circles are tangent to the envelope
tau = c + sigma tan(phi) analytically (the distance from each centre to
the line equals its radius - a built-in self-check). Tangency points
sit at (sigma_c - R sin phi, R cos phi): the stress state on the
failure plane. Equal aspect so angles read true.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

import journal_style as js

js.apply()
HERE = js.HERE

PHI = np.radians(32.0)                             # friction angle
C = 15.0                                           # cohesion, kPa
N_PHI = np.tan(np.pi / 4 + PHI / 2) ** 2
SIGMA3 = np.array([50.0, 100.0, 200.0])
SIGMA1 = SIGMA3 * N_PHI + 2 * C * np.sqrt(N_PHI)

fig, ax = plt.subplots(figsize=(5.6, 3.0))
ax.set_aspect("equal")

# envelope
sig = np.linspace(-C / np.tan(PHI), 800, 50)
ax.plot(sig, C + sig * np.tan(PHI), color="C3", lw=1.2)
ax.text(505, 362, r"$\tau_f = c + \sigma\,\tan\phi$", rotation=32,
        fontsize=7.5, color="C3")

# Mohr circles at failure + tangency points
theta = np.linspace(0, np.pi, 200)
for s3, s1 in zip(SIGMA3, SIGMA1):
    sc, rad = (s1 + s3) / 2, (s1 - s3) / 2
    ax.plot(sc + rad * np.cos(theta), rad * np.sin(theta), color="C0",
            lw=1.1)
    ax.plot(sc - rad * np.sin(PHI), rad * np.cos(PHI), "o", ms=3.5,
            color="C3", zorder=5)

# annotate the middle circle
s3, s1 = SIGMA3[1], SIGMA1[1]
sc, rad = (s1 + s3) / 2, (s1 - s3) / 2
for sv, lab, dx, ha in [(s3, r"$\sigma_3$", -9, "right"),
                        (s1, r"$\sigma_1$", 9, "left")]:
    ax.plot([sv, sv], [0, 0], marker="|", ms=6, color="C0")
    ax.text(sv + dx, 11, lab, ha=ha, va="bottom", fontsize=7.5, color="C0")
ax.annotate("failure state\n(tangency)",
            xy=(sc - rad * np.sin(PHI), rad * np.cos(PHI)),
            xytext=(62, 235), fontsize=6.5, color="C3",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="C3"))

# friction angle marker
ax.plot([430, 585], [283.7, 283.7], ls=":", lw=0.6, color="0.5")
ax.add_patch(Arc((430, 283.7), 190, 190, theta1=0, theta2=32,
                 color="0.35", lw=0.8))
ax.text(548, 305, r"$\phi = 32°$", fontsize=7.5, color="0.35")

# cohesion intercept
ax.plot(0, C, "s", ms=3.5, mfc="white", mec="C3", mew=1.0, zorder=5)
ax.annotate("cohesion, $c$", xy=(0, C), xytext=(-52, 90), fontsize=6.5,
            color="C3", ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="C3"))

ax.set_xlim(-60, 800)
ax.set_ylim(0, 400)
ax.set_xlabel(r"normal stress, $\sigma$ (kPa)")
ax.set_ylabel(r"shear stress, $\tau$ (kPa)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig085_mohr_coulomb.{ext}")
print("saved fig085_mohr_coulomb.png / .pdf")
