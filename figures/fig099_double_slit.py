"""Fig. 99 - Double-slit interference under its single-slit envelope.

The wave-optics classic, computed from the Fraunhofer result

    I/I_0 = cos^2(pi d sin(theta)/lambda) * sinc^2(pi a sin(theta)/lambda),

with slit separation d = 4a. The cos^2 fringes (spacing lambda/d) ride
under the sinc^2 diffraction envelope (nulls at multiples of lambda/a),
so every fourth interference order coincides with an envelope zero and
vanishes - the "missing orders" at m = +/-4, +/-8, a built-in check that
the two length scales were wired correctly. A rendered screen strip
sits above the curve, the way the pattern is actually seen.
"""

import numpy as np
import matplotlib.pyplot as plt

import journal_style as js

js.apply()
HERE = js.HERE

D_OVER_A = 4                                       # slit separation / width
x = np.linspace(-3, 3, 3000)                       # x = (a/lambda) sin(theta)
beta = np.pi * x
envelope = np.sinc(x) ** 2                         # sinc^2(beta), numpy sinc
fringes = np.cos(D_OVER_A * beta) ** 2
I = fringes * envelope

fig, (ax_s, ax) = plt.subplots(2, 1, figsize=(4.2, 3.1),
                               height_ratios=[0.16, 1.0], sharex=True)

# ---------------------------------------------------- screen strip ----
ax_s.imshow(I[np.newaxis, :], aspect="auto", cmap="Greys_r",
            extent=[x[0], x[-1], 0, 1], vmin=0, vmax=1)
ax_s.set_yticks([])
ax_s.tick_params(bottom=False)
ax_s.text(-2.97, 0.5, "screen", fontsize=6, color="white", va="center")

# --------------------------------------------------------- curves ----
ax.plot(x, I, color="C0", lw=1.0)
ax.plot(x, envelope, ls="--", lw=0.9, color="C1")
ax.text(0.62, 0.80, "single-slit envelope\n" r"sinc$^2(\pi a\sin\theta/\lambda)$",
        fontsize=6.5, color="C1")

for m in range(1, 4):                              # label a few orders
    y_env = envelope[np.argmin(np.abs(x - m / D_OVER_A))]
    ax.text(m / D_OVER_A, y_env + 0.045, f"{m}", fontsize=6,
            color="0.4", ha="center")
ax.annotate("missing order\n($m = \\pm 4$: fringe at\nenvelope null)",
            xy=(1.0, 0.012), xytext=(1.62, 0.42), fontsize=6.5,
            color="C3",
            arrowprops=dict(arrowstyle="-|>", lw=0.7, color="C3"))
ax.plot(1.0, 0.0, "o", ms=3.5, mfc="white", mec="C3", mew=1.0, zorder=5)
ax.plot(-1.0, 0.0, "o", ms=3.5, mfc="white", mec="C3", mew=1.0, zorder=5)

ax.text(-2.95, 0.90, "$d = 4a$", fontsize=7, color="0.3")
ax.set_xlim(-3, 3)
ax.set_ylim(0, 1.06)
ax.set_xlabel(r"$(a/\lambda)\,\sin\theta$")
ax.set_ylabel(r"intensity, $I/I_0$")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig099_double_slit.{ext}")
print("saved fig099_double_slit.png / .pdf")
