"""Fig. 84 - Buckley-Leverett waterflood: Welge tangent + profile.

Petroleum reservoir engineering's signature construction, computed end
to end. (a) The fractional-flow curve from Corey relative
permeabilities,

    f_w = (k_rw/mu_w) / (k_rw/mu_w + k_ro/mu_o),

with the Welge tangent drawn from the connate point (S_wc, 0): its
tangency f'(S*) = f(S*)/(S* - S_wc) fixes the shock-front saturation,
and its extension to f_w = 1 reads off the average saturation behind
the front at breakthrough. (b) The saturation profile x_D = t_D f'(S_w)
- a rarefaction ending in the shock - shown at two pore-volume times.
"""

import numpy as np
import matplotlib.pyplot as plt

import journal_style as js

js.apply()
HERE = js.HERE

# ---------------------------------------------------- Corey model ----
SWC, SOR = 0.20, 0.20
KRW0, KRO0, NW, NO = 0.4, 0.9, 3.0, 2.0
MUW, MUO = 1.0, 5.0                                # cp

S = np.linspace(SWC, 1 - SOR, 4001)
Se = (S - SWC) / (1 - SWC - SOR)
krw = KRW0 * Se**NW
kro = KRO0 * (1 - Se) ** NO
fw = (krw / MUW) / (krw / MUW + kro / MUO)
dfw = np.gradient(fw, S)

# Welge tangent from (SWC, 0)
mask = S > SWC + 0.05
g = dfw * (S - SWC) - fw
i0 = np.flatnonzero(mask & (np.sign(g) != np.sign(np.roll(g, -1))))[0]
w = -g[i0] / (g[i0 + 1] - g[i0])
S_F = S[i0] + w * (S[i0 + 1] - S[i0])              # shock saturation
F_F = np.interp(S_F, S, fw)
M_F = F_F / (S_F - SWC)                            # tangent slope
S_BAR = SWC + 1.0 / M_F                            # avg S_w at breakthrough

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.9))

# ------------------------------------------------- (a) tangent ----
ax_a.plot(S, fw, color="C0", lw=1.3)
s_t = np.array([SWC, SWC + 1.06 / M_F])
ax_a.plot(s_t, M_F * (s_t - SWC), color="C1", lw=0.9)
ax_a.plot(S_F, F_F, "o", ms=4.5, color="C1", zorder=5)
ax_a.plot(S_BAR, 1.0, "s", ms=4.5, mfc="white", mec="C1", mew=1.1,
          zorder=5)
ax_a.plot([S_F, S_F], [0, F_F], ls=":", lw=0.7, color="0.5")

ang = np.degrees(np.arctan(M_F * 0.543))           # axes decade aspect
ax_a.text(0.325, 0.415, "Welge tangent", rotation=ang,
          fontsize=6.5, color="C1", ha="center")
ax_a.annotate(rf"$\bar S_w$ = {S_BAR:.2f} at breakthrough", (S_BAR, 1.0),
              xytext=(10, -22), textcoords="offset points", fontsize=6.5,
              color="C1")
ax_a.annotate(rf"shock: $S_{{wf}}$ = {S_F:.2f}, $f_w$ = {F_F:.2f}",
              (S_F, F_F), xytext=(10, -4), textcoords="offset points",
              fontsize=6.5, color="C1")
ax_a.text(0.79, 0.06, f"$\\mu_o/\\mu_w$ = {MUO:g}", fontsize=6.5,
          ha="right", color="0.35")

ax_a.set_xlim(0.15, 0.85)
ax_a.set_ylim(0, 1.05)
ax_a.set_xlabel(r"water saturation, $S_w$")
ax_a.set_ylabel(r"fractional flow of water, $f_w$")

# ------------------------------------------------- (b) profile ----
for t_d, color, ls in [(0.15, "0.5", "--"), (0.30, "C0", "-")]:
    sel = S >= S_F
    x_rar = t_d * dfw[sel]
    x_s = t_d * M_F
    xx = np.concatenate([x_rar[::-1], [x_s, x_s, 1.0]])
    ss = np.concatenate([S[sel][::-1], [S_F, SWC, SWC]])
    ax_b.plot(xx, ss, color=color, lw=1.2, ls=ls)
    ax_b.text(t_d * M_F, 0.163, f"$t_D$ = {t_d:g}", fontsize=6.5,
              color=color, ha="center")

ax_b.axhline(SWC, ls=":", lw=0.6, color="0.6")
ax_b.axhline(1 - SOR, ls=":", lw=0.6, color="0.6")
ax_b.text(0.985, SWC + 0.012, "$S_{wc}$", fontsize=6.5, ha="right",
          color="0.5")
ax_b.text(0.985, 1 - SOR + 0.012, "$1 - S_{or}$", fontsize=6.5,
          ha="right", color="0.5")
ax_b.annotate("shock front", xy=(0.30 * M_F + 0.005, 0.38),
              xytext=(0.78, 0.44), fontsize=6.5, color="0.35",
              arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.4"))

ax_b.set_xlim(0, 1)
ax_b.set_ylim(0.12, 0.86)
ax_b.set_xlabel(r"dimensionless distance, $x_D$")
ax_b.set_ylabel(r"$S_w$")

js.panel_label(ax_a, "a", x=-0.17)
js.panel_label(ax_b, "b", x=-0.15)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig084_buckley_leverett.{ext}")
print(f"saved fig084_buckley_leverett.png / .pdf  "
      f"(S_wf = {S_F:.3f}, slope = {M_F:.2f})")
