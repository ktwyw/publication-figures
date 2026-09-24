"""Fig. 60 - Tauc plot with band-gap extrapolation (single column).

Optical band gaps from UV-vis absorption: (alpha*h*nu)^2 vs photon
energy for a direct allowed transition, a linear fit over the marked
Tauc region, dashed extrapolation to the abscissa, and the intercept
reported as E_g for each sample.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(60)

# ------------------------------------------------------------- DATA ----
hv = np.linspace(2.4, 4.0, 260)                          # photon energy, eV


def tauc_signal(eg, amp, urbach=0.06):
    """(alpha*h*nu)^2-like signal: linear above Eg, Urbach tail below."""
    core = amp * np.clip(hv - eg, 0, None)
    tail = amp * urbach * np.exp((hv - eg) / urbach)
    y = np.where(hv < eg, tail, core + amp * urbach)
    return y * (1 + rng.normal(0, 0.02, hv.size))


samples = {"TiO$_2$": (3.20, 1.0), "N-doped TiO$_2$": (2.88, 0.85)}

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.4, 2.8))

for i, (name, (eg_true, amp)) in enumerate(samples.items()):
    y = tauc_signal(eg_true, amp)
    ax.plot(hv, y, color=f"C{i}", lw=1.1, label=name)

    fit_mask = (hv > eg_true + 0.10) & (hv < eg_true + 0.45)  # Tauc region
    res = stats.linregress(hv[fit_mask], y[fit_mask])
    eg_fit = -res.intercept / res.slope
    xx = np.array([eg_fit, eg_true + 0.55])
    ax.plot(xx, res.intercept + res.slope * xx, ls="--", lw=0.8,
            color=f"C{i}")
    ax.plot(eg_fit, 0, marker="o", ms=4, mfc="white", mec=f"C{i}", mew=1.1,
            clip_on=False, zorder=4)
    place = [dict(xytext=(10, 26), ha="left"),
             dict(xytext=(-8, 30), ha="right")][i]
    ax.annotate(rf"$E_g$ = {eg_fit:.2f} eV", (eg_fit, 0),
                textcoords="offset points", fontsize=6.5, color=f"C{i}",
                **place)

ax.set_xlim(2.4, 4.0)
ax.set_ylim(0, 0.95)
ax.set_xlabel(r"Photon energy, $h\nu$ (eV)")
ax.set_ylabel(r"$(\alpha h\nu)^2$ (a.u.)")
ax.legend(loc="upper left", fontsize=6)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig060_tauc_bandgap.{ext}")
print("saved fig060_tauc_bandgap.png / .pdf")
