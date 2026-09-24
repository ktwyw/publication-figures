"""Fig. 69 - Tafel plot from Butler-Volmer kinetics (single column).

Electrode polarisation computed from the Butler-Volmer equation with a
mass-transfer limit on the cathodic branch,

    i = i0 [exp(aa f eta) - exp(-ac f eta)],   f = F/RT,
    cathodic branch combined with i_L via 1/i = 1/i_k + 1/i_L,

then plotted the way electrochemists measure it: overpotential vs
log|i|. The dashed Tafel extrapolations intersect at the exchange
current density i0; the vertical asymptote marks i_L. Synthetic
"measured" points are sampled from the same model.
"""

import numpy as np
import matplotlib.pyplot as plt

import journal_style as js

js.apply()
HERE = js.HERE

rng = np.random.default_rng(69)

# ------------------------------------------------------------ theory ----
F_RT = 96485.0 / (8.314 * 298.15)                 # 1/V
I0, IL = 1e-6, 2e-3                               # A/cm^2
AA = AC = 0.5
B_TAFEL = np.log(10) / (AA * F_RT)                # 118 mV per decade

eta = np.linspace(-0.40, 0.40, 1200)
i_a = I0 * np.exp(AA * F_RT * eta)
i_ck = I0 * np.exp(-AC * F_RT * eta)              # kinetic cathodic magnitude
i_c = 1.0 / (1.0 / i_ck + 1.0 / IL)               # mass-transfer combined
i_net = i_a - i_c

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.5, 3.0))
ax.plot(np.log10(np.abs(i_net)), eta, color="C0", lw=1.2, zorder=3)

# synthetic measured points
eta_m = np.concatenate([np.linspace(-0.38, -0.03, 14),
                        np.linspace(0.03, 0.38, 14)])
i_m = (I0 * np.exp(AA * F_RT * eta_m)
       - 1 / (1 / (I0 * np.exp(-AC * F_RT * eta_m)) + 1 / IL))
i_m *= 1 + rng.normal(0, 0.06, i_m.size)
ax.plot(np.log10(np.abs(i_m)), eta_m, "o", ms=3.2, mfc="white", mec="0.3",
        mew=0.8, zorder=4)

# Tafel extrapolations, intersecting at (log10 i0, 0)
eta_t = np.linspace(0, 0.40, 50)
ax.plot(np.log10(I0) + eta_t / B_TAFEL, eta_t, ls="--", lw=0.9, color="C1")
ax.plot(np.log10(I0) + eta_t / B_TAFEL, -eta_t, ls="--", lw=0.9, color="C1")
ax.plot(np.log10(I0), 0, "o", ms=4, color="C1", zorder=5)
ax.annotate("exchange\ncurrent, $i_0$", (np.log10(I0), 0),
            xytext=(-6.52, -0.115), fontsize=7, color="C1", ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-|>", lw=0.7, color="C1"))
ax.axvline(np.log10(IL), ls=":", lw=0.9, color="0.45")
ax.text(np.log10(IL) - 0.08, -0.385,
        r"$i_L$ (mass-transfer limit)", rotation=90, fontsize=6.5,
        color="0.45", ha="right", va="bottom")

ax.text(-4.85, 0.205, rf"$b_a$ = {1000*B_TAFEL:.0f} mV dec$^{{-1}}$",
        rotation=28, fontsize=7, color="C1", ha="center")
ax.text(-4.30, -0.088, rf"$b_c$ = −{1000*B_TAFEL:.0f} mV dec$^{{-1}}$",
        rotation=-28, fontsize=7, color="C1", ha="center")
ax.axhline(0, lw=0.5, color="0.75", zorder=0)

ax.set_xlim(-6.6, -2.2)
ax.set_ylim(-0.42, 0.42)
ax.set_xlabel(r"$\log_{10}|i|$  ($i$ in A cm$^{-2}$)")
ax.set_ylabel(r"overpotential, $\eta$ (V)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig069_tafel.{ext}")
print("saved fig069_tafel.png / .pdf")
