"""Fig. 100 - The Carnot cycle, told in both planes (two panels).

For the hundredth figure, the engine that started thermodynamics,
computed for one mole of diatomic ideal gas between T_h = 600 K and
T_c = 300 K with an isothermal volume ratio of 2. Every state point
follows from pV = nRT and the adiabats TV^(gamma-1) = const.

(a) p-V: the four computed branches enclose the net work (shaded).
(b) T-s: the same cycle is a rectangle, s built from
    ds = c_v dT/T + R dV/V, so the enclosed area is (T_h - T_c) dS.

The script integrates the p-V loop numerically and asserts it equals
the T-s rectangle area - the same 1.73 kJ from two different planes,
which is the whole point of a state function.
"""

import numpy as np
import matplotlib.pyplot as plt

import journal_style as js

js.apply()
HERE = js.HERE

R, GAMMA, N = 8.314, 1.4, 1.0
TH, TC = 600.0, 300.0
V1 = 5.0e-3                                        # m^3
V2 = 2 * V1
V3 = V2 * (TH / TC) ** (1 / (GAMMA - 1))
V4 = V3 / 2                                        # closes the cycle


def p_iso(v, T):
    return N * R * T / v


def p_adi(v, v0, T0):
    return N * R * T0 / v0 * (v0 / v) ** GAMMA


seg = []                                           # (V, p) four branches
v = np.linspace(V1, V2, 100)
seg.append((v, p_iso(v, TH)))
v = np.linspace(V2, V3, 200)
seg.append((v, p_adi(v, V2, TH)))
v = np.linspace(V3, V4, 200)
seg.append((v, p_iso(v, TC)))
v = np.linspace(V4, V1, 200)
seg.append((v, p_adi(v, V4, TC)))

V_loop = np.concatenate([s[0] for s in seg])
P_loop = np.concatenate([s[1] for s in seg])
W_PV = np.trapezoid(P_loop, V_loop)                # closed-loop area
DS = N * R * np.log(V2 / V1)
W_TS = (TH - TC) * DS
assert abs(W_PV - W_TS) < 1.0                      # same work, both planes

# ------------------------------------------------------------- PLOT ----
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.0))

# ---- (a) p-V
ax_a.fill(V_loop * 1e3, P_loop / 1e3, color="C0", alpha=0.18, lw=0)
for (v, p), color, k in zip(seg, ["C3", "0.35", "C0", "0.35"],
                            [55, 60, 100, 170]):
    ax_a.plot(v * 1e3, p / 1e3, color=color, lw=1.2)
    ax_a.annotate("", xy=(v[k + 4] * 1e3, p[k + 4] / 1e3),
                  xytext=(v[k] * 1e3, p[k] / 1e3),
                  arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                  color=color))
for vv, TT, num, dx, dy in [(V1, TH, "1", -1.5, 30), (V2, TH, "2", 2, 30),
                            (V3, TC, "3", 1.5, 12), (V4, TC, "4", -2.5, 8)]:
    ax_a.plot(vv * 1e3, p_iso(vv, TT) / 1e3, "o", ms=3.5, color="black",
              zorder=5)
    ax_a.text(vv * 1e3 + dx, p_iso(vv, TT) / 1e3 + dy, num, fontsize=7.5)

ax_a.text(8.4, 800, "$T_h$ = 600 K", fontsize=6.5, color="C3")
ax_a.text(36, 112, "$T_c$ = 300 K", fontsize=6.5, color="C0")
ax_a.annotate("adiabats\n($q$ = 0)", xy=(7.8, 545),
              xytext=(12.6, 650), fontsize=6.5, color="0.35",
              arrowprops=dict(arrowstyle="-", lw=0.6, color="0.45"))
ax_a.annotate(f"$w_{{net}}$ = {W_PV:.0f} J", xy=(13, 320),
              xytext=(31, 620), fontsize=7, color="C2",
              arrowprops=dict(arrowstyle="-|>", lw=0.7, color="C2"))

ax_a.set_xlim(0, 60)
ax_a.set_ylim(0, 1080)
ax_a.set_xlabel("volume, $V$ (L)")
ax_a.set_ylabel("pressure, $p$ (kPa)")

# ---- (b) T-s
s_pts = np.array([0, DS, DS, 0, 0])
T_pts = np.array([TH, TH, TC, TC, TH])
ax_b.fill(s_pts, T_pts, color="C0", alpha=0.18, lw=0)
ax_b.plot(s_pts, T_pts, color="0.15", lw=1.2)
for i, (num, dx, dy) in enumerate([("1", -0.22, 12), ("2", 0.12, 12),
                                   ("3", 0.12, -26), ("4", -0.22, -26)]):
    ax_b.plot(s_pts[i], T_pts[i], "o", ms=3.5, color="black", zorder=5)
    ax_b.text(s_pts[i] + dx, T_pts[i] + dy, num, fontsize=7.5)
for x0, x1, y, dy in [(0, DS, TH, 14), (DS, 0, TC, -22)]:
    k = 0.5 * (x0 + x1)
    ax_b.annotate("", xy=(k + 0.35 * (x1 - x0) / abs(x1 - x0), y),
                  xytext=(k, y),
                  arrowprops=dict(arrowstyle="-|>", lw=1.0, color="0.15"))

ax_b.text(DS / 2, TH + 16, r"$q_h = T_h\,\Delta S$", fontsize=6.5,
          ha="center", color="C3")
ax_b.text(DS / 2, TC - 20, r"$q_c = T_c\,\Delta S$", fontsize=6.5,
          ha="center", va="top", color="C0")
ax_b.text(DS / 2, 450,
          f"$w_{{net}} = (T_h - T_c)\\,\\Delta S$ = {W_TS:.0f} J",
          fontsize=7, ha="center", color="C2")
ax_b.text(DS / 2, 395, r"$\eta = 1 - T_c/T_h$ = 50%", fontsize=7.5,
          ha="center")
ax_b.text(DS + 0.25, 450, "same area,\nsame work:\nstate function",
          fontsize=6, color="0.45")

ax_b.set_xlim(-0.9, 7.6)
ax_b.set_ylim(220, 680)
ax_b.set_xlabel(r"entropy, $s - s_1$ (J K$^{-1}$)")
ax_b.set_ylabel(r"temperature, $T$ (K)")

js.panel_label(ax_a, "a", x=-0.17)
js.panel_label(ax_b, "b", x=-0.15)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig100_carnot_cycle.{ext}")
print(f"saved fig100_carnot_cycle.png / .pdf  "
      f"(W_pv = {W_PV:.1f} J, W_ts = {W_TS:.1f} J)")
