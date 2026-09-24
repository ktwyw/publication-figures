"""Fig. 70 - Fixed-bed adsorption breakthrough curves (single column).

Column outlet concentration from the logistic (Thomas-type) solution,

    C/C0 = 1 / (1 + exp((tau - t)/w)),

where tau is the stoichiometric time (inversely proportional to the
superficial velocity u) and w the mass-transfer-zone time scale
(broadening with u). Breakthrough (C/C0 = 0.05) and exhaustion (0.95)
are marked, and the mass-transfer zone is shaded for one curve.
"""

import numpy as np
import matplotlib.pyplot as plt

import journal_style as js

js.apply()
HERE = js.HERE

# ------------------------------------------------------------ theory ----
t = np.linspace(0, 170, 800)
cases = [(1.0, "C0"), (2.0, "C1"), (4.0, "C2")]     # u in cm/min


def curve(u):
    tau = 130.0 / u                                  # stoichiometric time
    w = 3.2 * np.sqrt(u)                             # MTZ time scale
    return 1.0 / (1.0 + np.exp((tau - t) / w)), tau, w


CB, CE = 0.05, 0.95                                  # breakthrough/exhaustion

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.6, 2.8))
for lo in (CB, CE):
    ax.axhline(lo, ls=":", lw=0.6, color="0.7", zorder=0)
ax.text(168, CB + 0.015, f"$C/C_0$ = {CB}", ha="right", va="bottom",
        fontsize=6, color="0.5")
ax.text(168, CE + 0.015, f"$C/C_0$ = {CE}", ha="right", va="bottom",
        fontsize=6, color="0.5")

for u, color in cases:
    c, tau, w = curve(u)
    ax.plot(t, c, color=color, lw=1.2, label=f"$u$ = {u:g} cm min$^{{-1}}$")
    tb, te = tau - w * np.log((1 - CB) / CB), tau + w * np.log((1 - CB) / CB)
    ax.plot(tb, CB, "o", ms=3.5, color=color, zorder=4)
    ax.plot(te, CE, "s", ms=3.2, color=color, zorder=4)
    if u == 2.0:                                    # shade MTZ for one case
        ax.axvspan(tb, te, color=color, alpha=0.10, lw=0)
        ax.annotate("mass-transfer\nzone", ((tb + te) / 2, 0.5),
                    xytext=(26, -6), textcoords="offset points", fontsize=7,
                    color=color, ha="left",
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=color))
        ax.annotate("", xy=(tb, CB), xytext=(tb, -0.075),
                    arrowprops=dict(arrowstyle="-", ls=":", lw=0.7,
                                    color=color))
        ax.text(tb, -0.09, r"$t_b$", ha="center", va="top", fontsize=7,
                color=color)

ax.annotate("increasing $u$", xy=(31, 0.70), xytext=(52, 0.90), fontsize=7,
            color="0.35", ha="left", va="center",
            arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.4"))

ax.set_xlim(0, 170)
ax.set_ylim(0, 1.06)
ax.set_xlabel(r"time, $t$ (min)")
ax.set_ylabel(r"outlet concentration, $C/C_0$")
ax.legend(loc="upper left", bbox_to_anchor=(0.50, 0.84),
          fontsize=6.5, frameon=False)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig070_breakthrough.{ext}")
print("saved fig070_breakthrough.png / .pdf")
