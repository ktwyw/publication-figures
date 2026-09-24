"""Fig. 88 - Computer architecture twins: roofline and Amdahl (2 panels).

(a) The roofline model: attainable performance
    P = min(P_peak, BW x I) on log-log axes, with the ridge point at
    I* = P_peak/BW, a lower no-vectorisation ceiling, and example
    kernels placed under their governing roof. Reuses ashby.setup and
    ashby.slope_angle for the decade axes and the slope-1 label.

(b) Amdahl's law S = 1/((1-p) + p/N) for three parallel fractions,
    each saturating at 1/(1-p), against Gustafson's scaled speedup
    S = N - (1-p)(N-1) - the fixed-size vs scaled-size views of the
    same machine. Illustrative parameters.
"""

import numpy as np
import matplotlib.pyplot as plt

import ashby
import journal_style as js

js.apply()
HERE = js.HERE

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.9))

# ------------------------------------------------- (a) roofline ----
BW, P_PEAK, P_SCALAR = 200.0, 1000.0, 125.0        # GB/s, GFLOP/s
I = np.logspace(np.log10(1 / 16), np.log10(128), 200)

ashby.setup(ax_a, (1 / 16, 128), (4, 2500),
            "arithmetic intensity, $I$ (FLOP/byte)",
            "performance (GFLOP/s)",
            xticks=[0.0625, 0.25, 1, 4, 16, 64])
ax_a.plot(I, np.minimum(P_PEAK, BW * I), color="black", lw=1.3)
ax_a.plot([P_SCALAR / BW, 128], [P_SCALAR, P_SCALAR], ls="--", lw=0.9,
          color="0.45")
ax_a.text(20, P_SCALAR * 1.25, "no vectorisation", fontsize=6,
          color="0.45")
ridge = P_PEAK / BW
ax_a.plot(ridge, P_PEAK, "o", ms=4.5, mfc="white", mec="black", mew=1.1,
          zorder=5)
ax_a.plot([ridge, ridge], [4, P_PEAK], ls=":", lw=0.7, color="0.6")
ax_a.text(ridge, 5.6, f"ridge\n$I^*$ = {ridge:g}", fontsize=6,
          color="0.35", ha="center")
ang = ashby.slope_angle(ax_a, 1)
ax_a.text(0.16, BW * 0.16 * 1.45, "BW = 200 GB s$^{-1}$", rotation=ang,
          fontsize=6, color="0.35", rotation_mode="anchor", ha="center")
ax_a.text(0.40, 900, "$P_{peak}$ = 1 TFLOP s$^{-1}$", fontsize=6,
          color="0.35")

KERNELS = [("SpMV", 0.25, 0.65), ("stencil", 0.85, 0.62),
           ("FFT", 4.0, 0.55), ("GEMM", 32.0, 0.85)]
for name, ii, eff in KERNELS:
    y = eff * min(P_PEAK, BW * ii)
    ax_a.plot(ii, y, "o", ms=4, color="C1", zorder=5)
    ax_a.annotate(name, (ii, y), xytext=(0, -11),
                  textcoords="offset points", fontsize=6, color="C1",
                  ha="center")
ax_a.text(0.085, 6.5, "memory-\nbound", fontsize=6, color="0.5")
ax_a.text(40, 320, "compute-\nbound", fontsize=6, color="0.5",
          ha="center")

# ------------------------------------------- (b) Amdahl/Gustafson ----
Np = np.logspace(0, np.log10(1024), 200)
for p, color in [(0.5, "C0"), (0.9, "C1"), (0.99, "C2")]:
    ax_b.plot(Np, 1 / ((1 - p) + p / Np), color=color, lw=1.2)
    ax_b.axhline(1 / (1 - p), ls=":", lw=0.6, color=color)
    ax_b.text(1080, 1 / (1 - p), f"$p$ = {p:g}", fontsize=6.5,
              color=color, va="center")
ax_b.plot(Np, Np - 0.1 * (Np - 1), color="C1", lw=1.0, ls="--")
ax_b.text(3.2, 60, "Amdahl:\n$S \\to 1/(1-p)$", fontsize=6.5,
          color="0.35")

ax_b.set_xscale("log", base=2)
ax_b.set_yscale("log")
ax_b.set_xlim(1, 1024)
ax_b.set_ylim(1, 1200)
ax_b.set_xlabel("processors, $N$")
ax_b.set_ylabel("speedup, $S$")
ang_g = ashby.slope_angle(ax_b, 1)                 # after log scales are set
ax_b.text(40, (40 - 0.1 * 39) * 1.5, "Gustafson, $p$ = 0.9",
          rotation=ang_g, fontsize=6.5, color="C1", rotation_mode="anchor",
          ha="center")

js.panel_label(ax_a, "a", x=-0.20)
js.panel_label(ax_b, "b", x=-0.17)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig088_roofline_amdahl.{ext}")
print("saved fig088_roofline_amdahl.png / .pdf")
