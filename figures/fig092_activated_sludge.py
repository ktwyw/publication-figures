"""Fig. 92 - Chemostat washout diagram (Monod kinetics, single column).

The bioreactor/wastewater design classic: steady states of a CSTR with
Monod growth mu = mu_m S/(K_s + S) at dilution rate D give

    S* = D K_s/(mu_m - D),      X* = Y (S_0 - S*),

valid until washout at D_c = mu_m S_0/(K_s + S_0), beyond which X = 0
and S = S_0. Productivity D X* peaks at the analytic optimum
D_opt = mu_m (1 - sqrt(K_s/(K_s + S_0))) - marked and cross-checked
against the numerical maximum. Illustrative parameters.
"""

import numpy as np
import matplotlib.pyplot as plt

import journal_style as js

js.apply()
HERE = js.HERE

MU, KS, Y, S0 = 4.0, 50.0, 0.5, 250.0              # 1/d, mg/L, -, mg/L
D_C = MU * S0 / (KS + S0)
D_OPT = MU * (1 - np.sqrt(KS / (KS + S0)))

D = np.linspace(0.001, 3.6, 600)
S = np.where(D < D_C, D * KS / (MU - D), S0)
X = np.where(D < D_C, Y * (S0 - D * KS / (MU - D)), 0.0)
prod = D * X
assert abs(D[np.argmax(prod)] - D_OPT) < 0.02      # analytic == numeric

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.9, 2.9))
ax2 = ax.twinx()

ax.plot(D, S, color="C1", lw=1.2)
ax.plot(D, X, color="C0", lw=1.2)
ax2.plot(D, prod, color="C2", lw=1.0)

ax.axvline(D_C, ls="--", lw=0.9, color="0.4")
ax.text(D_C + 0.06, 8, "washout,  $D_c = \\mu_m S_0/(K_s+S_0)$",
        rotation=90, fontsize=6, color="0.4", va="bottom")

P_OPT = D_OPT * Y * (S0 - D_OPT * KS / (MU - D_OPT))
ax2.plot(D_OPT, P_OPT, "o", ms=4.5, mfc="white", mec="C2", mew=1.2,
         zorder=5)
ax2.plot([D_OPT, D_OPT], [0, P_OPT], ls=":", lw=0.7, color="C2")
ax2.annotate("$D_{opt} = \\mu_m\\,(1-\\sqrt{K_s/(K_s+S_0)})$",
             (D_OPT, P_OPT), xytext=(-8, 10), textcoords="offset points",
             fontsize=6.5, color="C2", ha="right")

ax.text(0.28, 133, "biomass, $X$", fontsize=7, color="C0")
ax.text(2.05, 40, "effluent substrate, $S$", fontsize=7, color="C1",
        ha="center", va="top")
ax2.text(1.05, 160, "productivity, $D\\,X$", fontsize=7, color="C2")

ax.text(0.03, 0.075, f"$\\mu_m$ = {MU:g} d$^{{-1}}$, $K_s$ = {KS:g}, "
        f"$Y$ = {Y:g}, $S_0$ = {S0:g} mg/L",
        transform=ax.transAxes, fontsize=5.8, color="0.4")

ax.set_xlim(0, 3.6)
ax.set_ylim(0, 265)
ax2.set_ylim(0, 265)
ax.set_xlabel("dilution rate, $D$ (d$^{-1}$)")
ax.set_ylabel(r"concentration (mg L$^{-1}$)")
ax2.set_ylabel(r"productivity, $D\,X$ (mg L$^{-1}$ d$^{-1}$)",
               color="C2")
ax2.tick_params(axis="y", colors="C2")
ax2.spines["right"].set_color("C2")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig092_activated_sludge.{ext}")
print(f"saved fig092_activated_sludge.png / .pdf  "
      f"(D_c = {D_C:.2f}, D_opt = {D_OPT:.2f} d^-1)")
