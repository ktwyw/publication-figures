"""Fig. 95 - Titration curves, solved exactly (single column).

pH during titration of 50 mL of 0.1 M acid with 0.1 M NaOH, computed
with no approximations: at every added volume the proton condition

    [H+] + [Na+] = [OH-] + [A-],
    [A-] = C_a' K_a/(K_a + [H+])   (weak),   [A-] = C_a'   (strong),

is solved for [H+] with brentq (dilution included). One strong acid and
four weak acids (pKa = 3, 5, 7, 9) share the same equivalence volume;
the half-equivalence points sit at pH = pKa, and the equivalence jump
shrinks and the equivalence pH climbs as the acid weakens - why
indicator choice depends on pKa.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import journal_style as js

js.apply()
HERE = js.HERE

VA, CA, CB, KW = 50.0, 0.1, 0.1, 1e-14
V_EQ = CA * VA / CB

vb = np.linspace(0.02, 100.0, 700)


def ph_curve(pka):
    """pka = None for a strong acid."""
    ph = np.empty_like(vb)
    for i, v in enumerate(vb):
        ca = CA * VA / (VA + v)
        cb = CB * v / (VA + v)

        def f(h):
            a_minus = ca if pka is None else ca * 10**-pka / (10**-pka + h)
            return h + cb - KW / h - a_minus

        ph[i] = -np.log10(brentq(f, 1e-14, 1.0, xtol=1e-16))
    return ph


CASES = [(None, "0.15", "strong"), (3, "C0", None), (5, "C1", None),
         (7, "C2", None), (9, "C3", None)]

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.8, 3.1))

ax.axvline(V_EQ, ls="--", lw=0.8, color="0.6")
ax.text(V_EQ + 1.2, 0.55, "equivalence", fontsize=6, color="0.5",
        rotation=90, va="bottom")

for pka, color, tag in CASES:
    ph = ph_curve(pka)
    ax.plot(vb, ph, color=color, lw=1.1)
    if pka is None:
        ax.text(30, 1.08, "strong acid", fontsize=6.5, color=color,
                ha="center")
    else:
        ax.plot(V_EQ / 2, pka, "o", ms=3.2, mfc="white", mec=color,
                mew=1.0, zorder=5)
        ax.text(13, pka - 0.92, f"p$K_a$ = {pka}", fontsize=6.5,
                color=color)

ax.annotate("half-equivalence:\npH = p$K_a$", xy=(V_EQ / 2, 5),
            xytext=(37.5, 3.15), fontsize=6.5, color="C1",
            arrowprops=dict(arrowstyle="-|>", lw=0.7, color="C1"))

ax.set_xlim(0, 100)
ax.set_ylim(0, 14)
ax.set_yticks(range(0, 15, 2))
ax.set_xlabel("base added, $V_b$ (mL)")
ax.set_ylabel("pH")
ax.text(0.98, 0.03, "50 mL of 0.1 M acid + 0.1 M NaOH",
        transform=ax.transAxes, fontsize=5.8, color="0.4", ha="right")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig095_titration_curves.{ext}")
print("saved fig095_titration_curves.png / .pdf")
