"""Fig. 65 - Residence-time distributions: tanks-in-series model.

The reactor-engineering canon: exit-age distribution for N ideal CSTRs
in series,

    E(theta) = N (N theta)^(N-1) exp(-N theta) / Gamma(N),

sweeping N from a single stirred tank (exponential) toward plug flow
(delta function at theta = 1). The inset shows the model's signature
scaling sigma_theta^2 = 1/N on log-log axes.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import gamma

import journal_style as js

js.apply()
HERE = js.HERE

# ------------------------------------------------------------ theory ----
theta = np.linspace(0, 3, 700)
N_list = [1, 2, 5, 10, 50]
cmap = mpl.colormaps["viridis"]


def E(theta, N):
    return N * (N * theta) ** (N - 1) * np.exp(-N * theta) / gamma(N)


# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.7, 2.9))

label_pos = {1: (0.10, 1.04, "left"), 2: (0.50, 0.80, "center"),
             5: (0.80, 1.04, "center"), 10: (0.90, 1.40, "center"),
             50: (1.13, 2.72, "left")}
for i, N in enumerate(N_list):
    color = cmap(0.08 + 0.80 * i / (len(N_list) - 1))
    ax.plot(theta, E(theta, N), color=color, lw=1.2)
    x, y, ha = label_pos[N]
    ax.text(x, y, f"$N = {N}$", color=color, fontsize=7.5, ha=ha)

ax.axvline(1.0, ls=":", lw=0.8, color="0.5")
ax.text(1.0, 3.02, r"plug flow, $N \to \infty$", fontsize=7,
        color="0.45", ha="center")

ax.set_xlim(0, 3)
ax.set_ylim(0, 3.25)
ax.set_xlabel(r"dimensionless time, $\theta = t/\bar{t}$")
ax.set_ylabel(r"exit-age distribution, $E(\theta)$")

# ------------------------------------- inset: variance scaling ----
axins = ax.inset_axes([0.60, 0.55, 0.37, 0.40])
Ns = np.array([1, 2, 5, 10, 20, 50, 100])
axins.plot(Ns, 1 / Ns, "o-", ms=3, lw=0.9, color="black")
axins.text(5.0, 0.33, r"$\sigma_\theta^2 = 1/N$", rotation=-38, fontsize=6.5)
axins.set_xscale("log")
axins.set_yscale("log")
axins.set_xlim(0.8, 130)
axins.set_ylim(7e-3, 1.6)
axins.set_xlabel(r"$N$", fontsize=7, labelpad=1)
axins.set_ylabel(r"$\sigma_\theta^2$", fontsize=7, labelpad=1)
axins.tick_params(labelsize=6)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig065_rtd_tanks.{ext}")
print("saved fig065_rtd_tanks.png / .pdf")
