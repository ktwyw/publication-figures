"""Fig. 55 - Nyquist plot (EIS) with equivalent-circuit fit (single column).

Electrochemical impedance: -Z'' vs Z' on equal-aspect axes, measured
points with the Randles-model curve, selected frequencies annotated,
and the equivalent circuit itself drawn in code as an inset - no
external drawing tool needed.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(55)

# --------------------------------------------- Randles circuit model ----
RS, RCT, CDL, SIGMA = 12.0, 85.0, 2.0e-5, 26.0


def z_model(omega):
    zw = SIGMA * omega**-0.5 * (1 - 1j)                 # Warburg
    z_par = 1.0 / (1j * omega * CDL + 1.0 / (RCT + zw))
    return RS + z_par


f_meas = np.logspace(-1, 5, 45)                          # Hz
w = 2 * np.pi * f_meas
z = z_model(w) * (1 + rng.normal(0, 0.012, w.size)
                  + 1j * rng.normal(0, 0.012, w.size))
f_fit = np.logspace(-1.2, 5.2, 400)
z_fit = z_model(2 * np.pi * f_fit)

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.6, 3.0))
ax.plot(z_fit.real, -z_fit.imag, color="C1", lw=1.1, label="Randles fit")
ax.plot(z.real, -z.imag, "o", ms=3.5, mfc="white", color="C0",
        label="Measured")

offsets = {1e4: (7, 2), 1e2: (0, 7), 1e0: (9, -10)}
for f_tag in (1e4, 1e2, 1e0):                            # annotate frequencies
    k = np.argmin(np.abs(f_meas - f_tag))
    label = f"{f_tag:g} Hz" if f_tag < 1e3 else f"{f_tag/1e3:g} kHz"
    ax.annotate(label, (z.real[k], -z.imag[k]), xytext=offsets[f_tag],
                textcoords="offset points", fontsize=6, color="0.3")

ax.set_aspect("equal")
ax.set_xlim(0, 165)
ax.set_ylim(0, 148)
ax.set_xlabel(r"$Z'$ ($\Omega$)")
ax.set_ylabel(r"$-Z''$ ($\Omega$)")
ax.legend(loc="lower right", fontsize=6)
ax.text(0.05, 0.585,
        rf"$R_s$ = {RS:.0f} $\Omega$, $R_{{ct}}$ = {RCT:.0f} $\Omega$,"
        rf" $C_{{dl}}$ = {CDL*1e6:.0f} µF",
        transform=ax.transAxes, fontsize=6, va="top")

# -------------------------- equivalent-circuit schematic (inset) ----------
axc = ax.inset_axes([0.03, 0.62, 0.55, 0.34])
axc.set_xlim(0, 10)
axc.set_ylim(0, 2.2)
axc.axis("off")
LW, COL = 0.9, "0.15"


def wire(x0, y0, x1, y1):
    axc.plot([x0, x1], [y0, y1], color=COL, lw=LW)


def resistor(x0, x1, y, label):
    n, amp = 6, 0.14
    xs = np.linspace(x0, x1, 2 * n + 1)
    ys = y + amp * np.array([0] + [(-1) ** i for i in range(2 * n - 1)] + [0])
    axc.plot(xs, ys, color=COL, lw=LW)
    axc.text((x0 + x1) / 2, y + 0.30, label, ha="center", fontsize=5.5)


def capacitor(x, y, label):
    for dx in (-0.09, 0.09):
        axc.plot([x + dx, x + dx], [y - 0.22, y + 0.22], color=COL, lw=1.3)
    axc.text(x, y + 0.36, label, ha="center", fontsize=5.5)


wire(0.2, 1.1, 1.0, 1.1)
resistor(1.0, 2.6, 1.1, "$R_s$")
wire(2.6, 1.1, 3.4, 1.1)
wire(3.4, 0.55, 3.4, 1.65)                                # split node
wire(3.4, 1.65, 5.2, 1.65)                                # top: Cdl
capacitor(5.5, 1.65, "$C_{dl}$")
wire(5.8, 1.65, 7.6, 1.65)
wire(3.4, 0.55, 4.0, 0.55)                                # bottom: Rct + W
resistor(4.0, 5.4, 0.55, "$R_{ct}$")
wire(5.4, 0.55, 5.9, 0.55)
axc.add_patch(plt.Rectangle((5.9, 0.35), 0.9, 0.4, fill=False,
                            edgecolor=COL, lw=LW))
axc.text(6.35, 0.55, "$Z_w$", ha="center", va="center", fontsize=5.5)
wire(6.8, 0.55, 7.6, 0.55)
wire(7.6, 0.55, 7.6, 1.65)                                # rejoin node
wire(7.6, 1.1, 9.4, 1.1)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig055_nyquist_eis.{ext}")
print("saved fig055_nyquist_eis.png / .pdf")
