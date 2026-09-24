"""Fig. 86 - Streeter-Phelps dissolved-oxygen sag (single column).

Environmental engineering's founding model: downstream of a waste
discharge the oxygen deficit obeys

    D(t) = k_d L0/(k_a - k_d) (e^{-k_d t} - e^{-k_a t}) + D0 e^{-k_a t},

the balance of deoxygenation (k_d, BOD L0) against reaeration (k_a).
Three reaeration rates span a sluggish to a turbulent stream; each
curve's critical point is placed from the analytic t_c and
D_c = (k_d/k_a) L0 e^{-k_d t_c}, and the 5 mg/L aquatic-life threshold
shows which streams violate it. Illustrative parameters.
"""

import numpy as np
import matplotlib.pyplot as plt

import journal_style as js

js.apply()
HERE = js.HERE

KD, L0, D0, DO_SAT = 0.35, 18.0, 1.5, 9.1          # 1/d, mg/L, mg/L, mg/L
KA_LIST = [(0.5, "C3", "sluggish", 8.40, 6.55),
           (0.9, "C1", "moderate", 5.45, 6.42),
           (1.6, "C0", "turbulent", 3.20, 6.98)]

t = np.linspace(0, 14, 600)


def deficit(t, ka):
    return (KD * L0 / (ka - KD) * (np.exp(-KD * t) - np.exp(-ka * t))
            + D0 * np.exp(-ka * t))


def t_crit(ka):
    return (1.0 / (ka - KD)
            * np.log(ka / KD * (1 - D0 * (ka - KD) / (KD * L0))))


# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.8, 2.9))

ax.axhline(DO_SAT, ls=":", lw=0.7, color="0.55")
ax.text(13.8, DO_SAT + 0.09, r"saturation, DO$_{sat}$", fontsize=6,
        color="0.5", ha="right")
ax.axhline(5.0, ls="--", lw=0.8, color="0.45")
ax.text(13.8, 5.0 + 0.09, r"5 mg/L minimum (aquatic life)", fontsize=6,
        color="0.45", ha="right")

for ka, color, tag, lx, ly in KA_LIST:
    do = DO_SAT - deficit(t, ka)
    ax.plot(t, do, color=color, lw=1.2)
    tc = t_crit(ka)
    dc = KD / ka * L0 * np.exp(-KD * tc)
    ax.plot(tc, DO_SAT - dc, "o", ms=3.5, mfc="white", mec=color, mew=1.1,
            zorder=5)
    ax.text(lx, ly, rf"$k_a$ = {ka:g} d$^{{-1}}$ ({tag})", fontsize=6.5,
            color=color, ha="left")

tc_m = t_crit(0.5)
ax.plot([tc_m, tc_m], [0, DO_SAT - KD / 0.5 * L0 * np.exp(-KD * tc_m)],
        ls=":", lw=0.6, color="C3")
ax.text(tc_m, 2.72, r"$t_c$", ha="center", va="top", fontsize=7,
        color="C3")
ax.annotate("critical point:\n$k_a D_c = k_d L(t_c)$",
            xy=(tc_m, DO_SAT - KD / 0.5 * L0 * np.exp(-KD * tc_m)),
            xytext=(6.8, 2.92), fontsize=6.5, color="C3",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="C3"))

ax.annotate("waste discharge:\n$L_0$ = 18, $D_0$ = 1.5 mg/L",
            xy=(0, DO_SAT - D0), xytext=(1.1, 10.05), fontsize=6.5,
            color="0.3",
            arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.4"))
ax.text(0.98, 0.05, rf"$k_d$ = {KD:g} d$^{{-1}}$",
        transform=ax.transAxes, fontsize=6.5, ha="right", color="0.35")

ax.set_xlim(0, 14)
ax.set_ylim(2.5, 10.7)
ax.set_xlabel("travel time downstream, $t$ (d)")
ax.set_ylabel(r"dissolved oxygen (mg L$^{-1}$)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig086_streeter_phelps.{ext}")
print("saved fig086_streeter_phelps.png / .pdf")
