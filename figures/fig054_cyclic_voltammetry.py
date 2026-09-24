"""Fig. 54 - Cyclic voltammetry at several scan rates (single column).

The electrochemist's duck: forward/reverse sweeps forming closed loops,
scan rates encoded by a sequential colormap, a scan-direction arrow, and
the diagnostic inset - anodic peak current vs sqrt(scan rate), whose
linearity (Randles-Sevcik) indicates diffusion control.
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(54)

E_PA, E_PC = 0.25, 0.19                       # peak potentials (V)


def branch(E, Ep, ip, sign):
    """Empirical diffusion-peak shape: sigmoidal rise, t^-1/2-like tail."""
    x = sign * (E - Ep)
    rise = 1.0 / (1.0 + np.exp(-x / 0.018))
    tail = 1.0 / np.sqrt(1.0 + np.clip(x, 0, None) * 14)
    return sign * ip * 1.32 * rise * tail


# ------------------------------------------------------------- DATA ----
rates = [10, 25, 50, 100, 200]                # mV/s
E_fwd = np.linspace(-0.20, 0.60, 400)
E_rev = E_fwd[::-1]
cmap = mpl.colormaps["viridis"]

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.5, 3.0))
ip_a = []
for i, v in enumerate(rates):
    ip = 6.5 * np.sqrt(v)                     # Randles-Sevcik: ip ~ v^1/2
    cap = 0.05 * v                            # capacitive envelope
    i_f = branch(E_fwd, E_PA, ip, +1) + cap + rng.normal(0, 0.6, E_fwd.size)
    i_r = branch(E_rev, E_PC, ip * 0.97, -1) - cap + rng.normal(0, 0.6, E_rev.size)
    color = cmap(0.15 + 0.7 * i / (len(rates) - 1))
    ax.plot(np.concatenate([E_fwd, E_rev]), np.concatenate([i_f, i_r]),
            color=color, lw=1.0, label=f"{v}")
    ip_a.append(i_f.max())

ax.annotate("", xy=(0.52, 18), xytext=(0.38, 14),
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.35"))
ax.set_xlabel("Potential vs. Ag/AgCl (V)")
ax.set_ylabel("Current (\u00b5A)")
ax.legend(title="$v$ (mV s$^{-1}$)", fontsize=6, title_fontsize=6.5,
          loc="upper left")

# inset: Randles-Sevcik linearity
axins = ax.inset_axes([0.66, 0.10, 0.32, 0.30])
sqv = np.sqrt(rates)
res = stats.linregress(sqv, ip_a)
axins.plot(sqv, res.intercept + res.slope * np.array(sqv), color="C1", lw=0.9)
axins.plot(sqv, ip_a, "o", ms=3, color="C0")
axins.set_xlabel(r"$v^{1/2}$", fontsize=6, labelpad=1)
axins.set_ylabel(r"$i_{p,a}$ (µA)", fontsize=6, labelpad=1)
axins.tick_params(labelsize=5.5)
axins.text(0.06, 0.94, f"$R^2$ = {res.rvalue**2:.3f}",
           transform=axins.transAxes, fontsize=5.5, va="top")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig054_cyclic_voltammetry.{ext}")
print("saved fig054_cyclic_voltammetry.png / .pdf")
