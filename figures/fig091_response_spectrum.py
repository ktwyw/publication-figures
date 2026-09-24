"""Fig. 91 - Earthquake response spectrum, computed the honest way.

Not a code-shape sketch: (a) a seeded synthetic ground motion (band-
passed noise with a build-and-decay envelope, scaled to PGA = 0.30 g),
and (b) its elastic pseudo-acceleration spectrum obtained by actually
integrating the SDOF equation

    u'' + 2 zeta omega u' + omega^2 u = -a_g(t)

with a vectorised central-difference scheme over 90 periods and three
damping ratios, then S_a = omega^2 max|u|. The spectrum approaches the
PGA at short periods, as it must.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import journal_style as js

js.apply()
HERE = js.HERE

rng = np.random.default_rng(91)
G = 9.81

# ------------------------------------------- synthetic ground motion ----
DT, DUR = 0.005, 25.0
t = np.arange(0, DUR + DT, DT)
b, a = signal.butter(4, [0.4, 8.0], btype="band", fs=1 / DT)
ag = signal.filtfilt(b, a, rng.standard_normal(t.size))
env = np.minimum(t / 2.5, 1.0) * np.exp(-0.18 * np.maximum(t - 9.0, 0.0))
ag *= env
ag *= 0.30 * G / np.abs(ag).max()                  # PGA = 0.30 g

# ------------------------------------------------- SDOF sweep ----
T = np.geomspace(0.05, 5.0, 90)
om = 2 * np.pi / T
ZETAS = [(0.02, "C3"), (0.05, "C0"), (0.10, "C2")]

spectra = {}
for zeta, _ in ZETAS:
    c1 = 1.0 - zeta * om * DT
    c2 = 1.0 + zeta * om * DT
    k = 2.0 - (om * DT) ** 2
    u0 = np.zeros_like(om)
    u1 = np.zeros_like(om)
    umax = np.zeros_like(om)
    for i in range(1, t.size - 1):
        u2 = (DT * DT * (-ag[i]) + k * u1 - c1 * u0) / c2
        umax = np.maximum(umax, np.abs(u2))
        u0, u1 = u1, u2
    spectra[zeta] = om**2 * umax / G               # S_a in g

# ------------------------------------------------------------- PLOT ----
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                                 width_ratios=[1.15, 1])

ax_a.plot(t, ag / G, color="C0", lw=0.5)
i_pk = np.argmax(np.abs(ag))
ax_a.plot(t[i_pk], ag[i_pk] / G, "o", ms=3.5, mfc="white", mec="C3",
          mew=1.0, zorder=5)
ax_a.annotate("PGA = 0.30 g", (t[i_pk], ag[i_pk] / G), xytext=(8, -2),
              textcoords="offset points", fontsize=6.5, color="C3")
ax_a.set_xlim(0, DUR)
ax_a.set_ylim(-0.36, 0.36)
ax_a.set_xlabel("time (s)")
ax_a.set_ylabel("ground acceleration (g)")

for zeta, color in ZETAS:
    ax_b.semilogx(T, spectra[zeta], color=color, lw=1.1,
                  label=rf"$\zeta$ = {zeta:g}")
ax_b.plot([0.05, 0.085], [0.30, 0.30], ls=":", lw=0.8, color="0.4")
ax_b.text(0.09, 0.30, "PGA", fontsize=6, color="0.4", va="center")
ax_b.legend(loc="upper right", fontsize=6.5, frameon=False)

ax_b.set_xlim(0.05, 5)
ax_b.set_ylim(0, 1.12 * spectra[0.02].max())
ax_b.set_xlabel("period, $T$ (s)")
ax_b.set_ylabel(r"pseudo-acceleration, $S_a$ (g)")

js.panel_label(ax_a, "a", x=-0.14)
js.panel_label(ax_b, "b", x=-0.20)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig091_response_spectrum.{ext}")
print(f"saved fig091_response_spectrum.png / .pdf  "
      f"(peak S_a = {spectra[0.02].max():.2f} g)")
