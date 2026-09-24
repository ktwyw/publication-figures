"""Fig. 94 - Quantum harmonic oscillator: levels and wavefunctions.

The first real eigenvalue problem most students meet, drawn from the
exact solutions: the potential V = xi^2/2 (in units of h_bar omega,
xi = x sqrt(m omega / h_bar)), the evenly spaced levels
E_n = (n + 1/2), and the Hermite-Gaussian eigenfunctions

    psi_n(xi)  proportional to  H_n(xi) exp(-xi^2/2)

riding on their levels. Each wavefunction extends past its classical
turning point (dots, where E_n = V) into the forbidden region - the
tunnelling tail - and the zero-point energy E_0 = 1/2 is bracketed.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermval

import journal_style as js

js.apply()
HERE = js.HERE

xi = np.linspace(-4.6, 4.6, 800)
V = xi**2 / 2
N_MAX = 5
AMP = 0.42                                         # display amplitude

fig, ax = plt.subplots(figsize=(3.7, 3.4))

vis = V <= 6.9
ax.plot(xi[vis], V[vis], color="0.15", lw=1.3)

for n in range(N_MAX + 1):
    e_n = n + 0.5
    xt = np.sqrt(2 * e_n)                          # classical turning point
    coeff = np.zeros(n + 1)
    coeff[n] = 1.0
    psi = hermval(xi, coeff) * np.exp(-(xi**2) / 2)
    psi = AMP * psi / np.abs(psi).max()
    ax.axhline(e_n, lw=0.5, color="0.85", zorder=0)
    ax.plot(xi, e_n + psi, color="C0", lw=1.0)
    ax.plot([-xt, xt], [e_n, e_n], "o", ms=2.6, color="C3", zorder=5)
    ax.text(4.05, e_n + 0.06, f"$n$ = {n}", fontsize=6.5, color="0.35")

ax.annotate("", xy=(0.16, 0.5), xytext=(0.16, 0.012),
            arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.35"))
ax.text(0.30, 0.235, "$E_0 = \\frac{1}{2}\\hbar\\omega$", fontsize=6.5,
        color="0.35", ha="left", va="center")
ax.annotate("classical turning points\n($E_n = V$); tails tunnel beyond",
            xy=(np.sqrt(2 * 5.5), 5.53), xytext=(2.02, 6.42), fontsize=6.5,
            color="C3",
            arrowprops=dict(arrowstyle="-|>", lw=0.7, color="C3"))
ax.text(-4.45, 6.35, "$E_n = (n+\\frac{1}{2})\\,\\hbar\\omega$",
        fontsize=7.5)
ax.text(1.92, 1.22, "$V(\\xi) = \\frac{1}{2}\\xi^2$", fontsize=7,
        color="0.15", ha="left")

ax.set_xlim(-4.6, 4.9)
ax.set_ylim(-0.25, 6.9)
ax.set_xlabel(r"$\xi = x\,\sqrt{m\omega/\hbar}$")
ax.set_ylabel(r"energy ($\hbar\omega$)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig094_qho_wavefunctions.{ext}")
print("saved fig094_qho_wavefunctions.png / .pdf")
