"""Fig. 83 - Digital-communication BER waterfalls (single column).

Bit-error probability vs E_b/N_0 for coherent modulations on the AWGN
channel, computed from the Q-function (Q(x) = erfc(x/sqrt(2))/2):

    BPSK/QPSK : P_b = Q( sqrt(2 gamma) )
    16-QAM    : P_b = (3/4)  Q( sqrt(0.8 gamma) )
    64-QAM    : P_b = (7/12) Q( sqrt(0.2857 gamma) )   (Gray, nearest
                                                        -neighbour approx.)

with gamma = E_b/N_0. The dashed curve is an illustrative rate-1/2
coded system drawn as a 6 dB shift of BPSK; the ultimate Shannon limit
E_b/N_0 -> ln 2 = -1.59 dB is the vertical asymptote no code can cross.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

import journal_style as js

js.apply()
HERE = js.HERE


def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2.0))


snr_db = np.linspace(-3, 20, 500)
g = 10 ** (snr_db / 10)

curves = [
    ("BPSK / QPSK", qfunc(np.sqrt(2 * g)), "C0", 6.9, "right"),
    ("16-QAM", 0.75 * qfunc(np.sqrt(0.8 * g)), "C1", 11.4, "left"),
    ("64-QAM", (7 / 12) * qfunc(np.sqrt(18 / 63 * g)), "C2", 15.6, "left"),
]
coded = qfunc(np.sqrt(2 * 10 ** ((snr_db + 6) / 10)))

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.7, 3.0))
ax.grid(True, which="both", lw=0.3, color="0.90")
ax.set_axisbelow(True)

for name, pb, color, x_lab, ha in curves:
    ax.semilogy(snr_db, pb, color=color, lw=1.2)
    ax.text(x_lab, 1.3e-3, name, fontsize=6.5, color=color, ha=ha)
ax.semilogy(snr_db, coded, color="C0", lw=1.0, ls="--")
ax.text(1.1, 1.1e-2, "coded\n(6 dB, illustrative)", fontsize=6,
        color="C0", ha="left")

# coding-gain arrow at P_b = 1e-5
x_unc, x_cod = 9.59, 3.59
ax.annotate("", xy=(x_cod, 1e-5), xytext=(x_unc, 1e-5),
            arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.35"))
ax.text((x_unc + x_cod) / 2, 1.7e-5, "coding gain", fontsize=6,
        color="0.35", ha="center")

ax.axvline(10 * np.log10(np.log(2)), ls="--", lw=0.9, color="C3")
ax.text(10 * np.log10(np.log(2)) - 0.38, 2.5e-7,
        r"Shannon limit ($-1.59$ dB)", rotation=90, fontsize=6,
        color="C3", va="bottom")

ax.set_xlim(-3, 20)
ax.set_ylim(1e-7, 0.7)
ax.set_xlabel(r"$E_b/N_0$ (dB)")
ax.set_ylabel(r"bit error probability, $P_b$")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig083_ber_waterfall.{ext}")
print("saved fig083_ber_waterfall.png / .pdf")
