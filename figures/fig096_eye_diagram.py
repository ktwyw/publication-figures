"""Fig. 96 - Eye diagrams: how a channel closes the eye (two panels).

The digital-link health chart, built from scratch: random binary
symbols shaped with a raised-cosine pulse (beta = 0.35, singularities
handled analytically), then folded modulo two symbol periods and
overplotted with translucent traces so trace density forms the eye.

(a) At the transmitter the RC pulse is Nyquist: every trace passes
    exactly through +/-1 at the sampling instant - the eye is fully
    open there by construction.
(b) After a band-limited channel (zero-phase Butterworth) plus AWGN,
    ISI and noise thicken the rails and shrink the margin.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import journal_style as js

js.apply()
HERE = js.HERE

rng = np.random.default_rng(96)

SPS, BETA, SPAN, NSYM = 16, 0.35, 10, 600

# ---------------------------------------------- raised-cosine pulse ----
t = np.arange(-SPAN / 2, SPAN / 2 + 1 / SPS, 1 / SPS)
h = np.sinc(t) * np.cos(np.pi * BETA * t)
den = 1 - (2 * BETA * t) ** 2
sing = np.isclose(den, 0.0)
h = np.where(sing, np.pi / 4 * np.sinc(1 / (2 * BETA)), h / np.where(
    sing, 1.0, den))

bits = rng.choice([-1.0, 1.0], NSYM)
x = np.zeros(NSYM * SPS)
x[::SPS] = bits
tx = np.convolve(x, h, mode="same")

# ------------------------------------------------- channel + noise ----
b, a = signal.butter(3, 0.45 / (SPS / 2))
rx = signal.filtfilt(b, a, tx) + rng.normal(0, 0.08, tx.size)


def eye(ax, sig, color):
    t_eye = np.arange(2 * SPS + 1) / SPS - 1.0
    for k in range(SPAN, NSYM - SPAN, 2):
        seg = sig[k * SPS:(k + 2) * SPS + 1]
        ax.plot(t_eye, seg, color=color, lw=0.5, alpha=0.13)
    ax.axvline(0, ls=":", lw=0.8, color="0.35")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.75, 1.75)
    ax.set_xlabel("time (symbol periods)")


fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)

eye(ax_a, tx, "C0")
ax_a.set_ylabel("amplitude")
ax_a.set_title(r"transmit: raised cosine, $\beta$ = 0.35", fontsize=8)
ax_a.annotate("", xy=(0.055, 1.0), xytext=(0.055, -1.0),
              arrowprops=dict(arrowstyle="<->", lw=0.9, color="C3"))
ax_a.text(0.10, 0.0, "eye\nheight", fontsize=6.5, color="C3",
          va="center")
ax_a.text(0.03, -1.62, "sampling instant", fontsize=6, color="0.35",
          ha="left")

eye(ax_b, rx, "C1")
ax_b.set_title("after band-limited channel + noise", fontsize=8)
ax_b.annotate("ISI + noise\nclose the eye", xy=(0.0, 0.62),
              xytext=(0.34, 1.32), fontsize=6.5, color="0.3",
              arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.4"))

js.panel_label(ax_a, "a", x=-0.16)
js.panel_label(ax_b, "b", x=-0.05)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig096_eye_diagram.{ext}")
print("saved fig096_eye_diagram.png / .pdf")
