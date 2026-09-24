"""Fig. 93 - Blackbody radiation: Planck, Wien, and the catastrophe.

Spectral radiance computed from Planck's law,

    B(lambda, T) = (2 h c^2 / lambda^5) / (exp(hc/(lambda k T)) - 1),

for four temperatures, with the Wien displacement locus
lambda_max T = 2898 um K threading the peaks (drawn by evaluating
B(lambda_max(T), T) along a continuous T sweep - it lands on every
peak, a built-in check), the Rayleigh-Jeans classical limit diverging
at short wavelength for the hottest curve, and the visible band shaded.
"""

import numpy as np
import matplotlib.pyplot as plt

import journal_style as js

js.apply()
HERE = js.HERE

H, C, KB = 6.62607e-34, 2.99792e8, 1.380649e-23
B_WIEN = 2.8978e-3                                 # m K


def planck(lam_um, T):
    """Spectral radiance in kW m^-2 nm^-1 sr^-1."""
    lam = lam_um * 1e-6
    return (2 * H * C**2 / lam**5
            / np.expm1(H * C / (lam * KB * T))) * 1e-12


lam = np.linspace(0.05, 3.0, 800)
TEMPS = [(3000, "C1", 0.05, 0.4), (4000, "C2", 0.05, 0.4),
         (5000, "C0", 0.05, 0.4), (5800, "C3", 0.33, -8.6)]

fig, ax = plt.subplots(figsize=(3.9, 3.0))

ax.axvspan(0.38, 0.75, color="gold", alpha=0.15, lw=0)
ax.text(0.615, 28.3, "visible", fontsize=6, color="0.45", ha="center")

for T, color, dx, dy in TEMPS:
    ax.plot(lam, planck(lam, T), color=color, lw=1.2)
    lm = B_WIEN / T * 1e6
    ax.text(lm + dx, planck(lm, T) + dy, f"{T} K", fontsize=7,
            color=color)

# Wien displacement locus through the peaks
T_sweep = np.linspace(2600, 6400, 100)
lam_max = B_WIEN / T_sweep * 1e6
ax.plot(lam_max, planck(lam_max, T_sweep), ls="--", lw=0.9, color="0.3")
ax.text(1.06, 3.4, "Wien:  $\\lambda_{max}T$ = 2898 µm K", fontsize=6.5,
        color="0.3", rotation=-38)

# Rayleigh-Jeans for the hottest curve
b_rj = 2 * C * KB * 5800 / (lam * 1e-6) ** 4 * 1e-12
ax.plot(lam, b_rj, ls=":", lw=1.0, color="C3")
ax.annotate("Rayleigh\u2013Jeans (classical):\nultraviolet catastrophe",
            xy=(1.16, 26.5), xytext=(1.60, 22.3), fontsize=6.5,
            color="C3",
            arrowprops=dict(arrowstyle="-|>", lw=0.7, color="C3"))

ax.set_xlim(0, 3)
ax.set_ylim(0, 29)
ax.set_xlabel(r"wavelength, $\lambda$ (µm)")
ax.set_ylabel(r"spectral radiance (kW m$^{-2}$ nm$^{-1}$ sr$^{-1}$)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig093_planck_blackbody.{ext}")
print("saved fig093_planck_blackbody.png / .pdf")
