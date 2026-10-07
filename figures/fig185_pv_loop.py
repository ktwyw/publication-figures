"""Fig. 185 - Left-ventricular pressure-volume loops (1.5 column, 120 mm).

How the pressure-volume plane summarises a heartbeat: a time-varying
elastance ventricle ejecting into a windkessel gives the baseline loop
with its four phases, valve events and stroke work (the enclosed area);
two loops at reduced filling pressure show that the end-systolic
corners fall on one straight line of slope E_es through V₀ while the
end-diastolic corners follow the passive curve. Pressure and volume
against time over the same beat sit beside it. The self-check is that
stroke work from the time integral of P dV equals the shoelace area
within 0.5%, that the integrated aortic flow equals EDV − ESV, that the
slope fitted to the three end-systolic points is within 5% of the
model's E_es, and that the last beat is periodic to 1e-3.

Model: P = e(t) E_es (V − V₀) + (1 − e(t)) A (exp(B (V − V₀)) − 1),
half-cosine activation e(t); ideal-diode mitral and aortic valves with
series resistance; constant filling pressure; two-element windkessel.
Parameters are in the GOVERNING MODEL block (generic adult values, not
a fit to data). No data: every curve is computed from the equations.
"""

import numpy as np
from matplotlib.lines import Line2D
from scipy import integrate

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
T_BEAT = 0.8                       # s (75 beats per minute)
T_RISE, T_FALL = 0.28, 0.14        # s, activation up and down
E_ES, V0 = 2.5, 10.0               # mmHg/mL, mL: end-systolic elastance
A_ED, B_ED = 0.35, 0.03            # mmHg, 1/mL: passive (diastolic) curve
R_MV, R_AV = 0.02, 0.02            # mmHg s/mL: open-valve resistances
R_SYS, C_AO = 1.0, 1.5             # mmHg s/mL, mL/mmHg: windkessel
P_FILL = (10.0, 6.5, 4.5)          # mmHg: baseline and two reduced preloads
N_BEATS = 30                       # beats run before the one that is drawn
MMHG_ML_TO_J = 133.322e-6          # 1 mmHg = 133.322 Pa (conventional mmHg)


def activation(t):
    """e(t) in [0, 1]: half-cosine rise, half-cosine fall, then rest."""
    t = np.mod(t, T_BEAT)
    rise = 0.5 * (1 - np.cos(np.pi * t / T_RISE))
    fall = 0.5 * (1 + np.cos(np.pi * (t - T_RISE) / T_FALL))
    return np.where(t < T_RISE, rise, np.where(t < T_RISE + T_FALL, fall, 0))


def passive(volume):
    """End-diastolic pressure-volume relation (mmHg)."""
    return A_ED * (np.exp(B_ED * (volume - V0)) - 1)


def lv_pressure(t, volume):
    e = activation(t)
    return e * E_ES * (volume - V0) + (1 - e) * passive(volume)


def flows(t, volume, p_ao, p_fill):
    """Mitral inflow and aortic outflow (mL/s) through ideal diodes."""
    p_lv = lv_pressure(t, volume)
    return (np.maximum(p_fill - p_lv, 0) / R_MV,
            np.maximum(p_lv - p_ao, 0) / R_AV)


def rhs(t, state, p_fill):
    q_mv, q_av = flows(t, *state, p_fill)
    return [q_mv - q_av, (q_av - state[1] / R_SYS) / C_AO]


# ------------------------------------------------------------ SOLVER ---
def aortic_gradient(t, state, _p_fill):
    return lv_pressure(t, state[0]) - state[1]


def mitral_gradient(t, state, p_fill):
    return lv_pressure(t, state[0]) - p_fill


def simulate(p_fill):
    """Run to a periodic state; return the last beat on a dense grid."""
    solve = dict(method="RK45", rtol=1e-9, atol=1e-10, max_step=T_BEAT / 200,
                 args=(p_fill,))
    state = [120.0, 80.0]                        # mL, mmHg
    for _ in range(N_BEATS):
        state = integrate.solve_ivp(rhs, (0, T_BEAT), state, **solve).y[:, -1]
    beat = integrate.solve_ivp(rhs, (0, T_BEAT), state, dense_output=True,
                               events=(aortic_gradient, mitral_gradient),
                               **solve)
    t = np.linspace(0, T_BEAT, 4001)
    volume, p_ao = beat.sol(t)
    p_lv = lv_pressure(t, volume)
    q_mv, q_av = flows(t, volume, p_ao, p_fill)
    (t_avo, t_avc), (t_mvc, t_mvo) = beat.t_events      # in time order
    i_es = np.argmax(p_lv / (volume - V0))       # maximal elastance
    return dict(
        t=t, v=volume, p=p_lv, p_ao=p_ao, start=np.array(state),
        end=beat.y[:, -1], events=dict(MVC=t_mvc, AVO=t_avo, AVC=t_avc,
                                       MVO=t_mvo),
        edv=beat.sol(t_mvc)[0], esv=beat.sol(t_avc)[0],
        sv_flow=integrate.trapezoid(q_av, t),
        work_time=-integrate.trapezoid(p_lv * (q_mv - q_av), t),
        work_shoelace=0.5 * np.sum(volume * np.roll(p_lv, -1)
                                   - np.roll(volume, -1) * p_lv),
        es=(volume[i_es], p_lv[i_es]))


loops = [simulate(p_fill) for p_fill in P_FILL]
base = loops[0]
es_v, es_p = np.array([loop["es"] for loop in loops]).T
slope_fit, intercept = np.polyfit(es_v, es_p, 1)
v0_fit = -intercept / slope_fit
stroke_volume = base["edv"] - base["esv"]
ejection_fraction = stroke_volume / base["edv"]

# ------------------------------------------------------- SELF-CHECK ---
for loop in loops:
    assert abs(loop["work_time"] / loop["work_shoelace"] - 1) < 0.005
    assert abs(loop["sv_flow"] / (loop["edv"] - loop["esv"]) - 1) < 1e-3
    assert np.abs(loop["end"] - loop["start"]).max() < 1e-3
    assert abs(loop["p"][0] - passive(loop["v"][0])) < 1e-9   # ED on EDPVR
assert abs(slope_fit / E_ES - 1) < 0.05, slope_fit
assert abs(v0_fit - V0) < 2.0, v0_fit
drift = np.abs(base["end"] - base["start"]).max()
print(f"fig185: self-check passed (stroke work {base['work_time']:.1f} vs "
      f"shoelace {base['work_shoelace']:.1f} mmHg mL = "
      f"{base['work_time'] * MMHG_ML_TO_J:.3f} J; SV {stroke_volume:.2f} mL "
      f"= integrated outflow {base['sv_flow']:.2f} mL; EF "
      f"{100 * ejection_fraction:.1f}%; ESPVR slope {slope_fit:.3f} vs "
      f"E_es {E_ES}, V0 {v0_fit:.2f} mL; beat-to-beat drift {drift:.1e})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 84)
ax_a = ms.axes(fig, 13, 11, 61, 66)
ax_p = ms.axes(fig, 89, 47, 27, 30)
ax_v = ms.axes(fig, 89, 11, 27, 30, sharex=ax_p)
LOOP = ms.BLUE
AORTIC = dict(marker="o", ms=3.6, mfc=LOOP, mec="white", mew=0.5, ls="none")
MITRAL = dict(marker="o", ms=3.3, mfc="white", mec=LOOP, mew=0.9, ls="none")
STYLE = dict(MVC=MITRAL, MVO=MITRAL, AVO=AORTIC, AVC=AORTIC)

# a, the loops; the relations are drawn only as far as the labels allow
P_TOP = 118.0
v_ed = np.linspace(V0, 136, 200)
ax_a.plot(v_ed, passive(v_ed), color=ms.INK, lw=0.7)
ax_a.plot([v0_fit, (P_TOP - intercept) / slope_fit], [0, P_TOP],
          color=ms.INK, lw=0.7, ls=(0, (4, 2)))
for loop in loops[1:]:
    ax_a.plot(loop["v"], loop["p"], color=ms.GREY, lw=0.8)
ax_a.fill(base["v"], base["p"], color=LOOP, alpha=0.12, lw=0)
ax_a.plot(base["v"], base["p"], color=LOOP, lw=1.6)
ax_a.plot(es_v, es_p, "s", ms=2.4, mfc=ms.INK, mew=0, zorder=7)
for name, t_event in base["events"].items():
    i = np.argmin(np.abs(base["t"] - t_event))
    ax_a.plot(base["v"][i], base["p"][i], zorder=6, **STYLE[name])

mid_v, peak_p = (base["edv"] + base["esv"]) / 2, base["p"].max()
ax_a.text(mid_v, peak_p + 2, "Ejection", ha="center", va="bottom",
          color=LOOP, fontsize=ms.FS_TICK)
ax_a.text(base["edv"] + 3, 50, "Isovolumic\ncontraction", va="center",
          color=LOOP, fontsize=ms.FS_TICK)
# inside the loop only the pocket left of the reduced-preload loops is free
pocket = (base["esv"] + loops[-1]["edv"]) / 2
ax_a.text(base["esv"] + 3, 36, "Isovolumic\nrelaxation", va="center",
          color=LOOP, fontsize=ms.FS_TICK)
ax_a.text(pocket, 11, "Filling", ha="center", va="bottom", color=LOOP,
          fontsize=ms.FS_TICK)
ax_a.text(2, 163, "ESPVR (dashed)\n"
          f"slope {slope_fit:.2f} mmHg mL⁻¹", va="top", fontsize=ms.FS_TICK,
          linespacing=1.3)
ax_a.text(v0_fit - 2, 5, "V₀", ha="right", va="center", fontsize=ms.FS_TICK)
ax_a.text(138, passive(136), "EDPVR", va="center", fontsize=ms.FS_TICK)
ax_a.text(158, 163, f"SV = {stroke_volume:.0f} mL, "
          f"EF = {100 * ejection_fraction:.0f}%\nStroke work (shaded) = "
          f"{base['work_time'] * MMHG_ML_TO_J:.2f} J", ha="right", va="top",
          fontsize=ms.FS_TICK, linespacing=1.3)
ES = dict(marker="s", ms=2.4, mfc=ms.INK, mew=0, ls="none")
ax_a.legend(handles=[Line2D([], [], **AORTIC), Line2D([], [], **MITRAL),
                     Line2D([], [], **ES),
                     Line2D([], [], color=ms.GREY, lw=0.8)],
            labels=["Aortic valve opens, closes",
                    "Mitral valve closes, opens",
                    "End systole (maximal elastance)", "Reduced preload"],
            loc="upper right", bbox_to_anchor=(1.0, 0.905),
            fontsize=ms.FS_SMALL, handlelength=1.2)
ax_a.set_xlim(0, 160)
ax_a.set_ylim(0, 165)
ax_a.set_xticks(np.arange(0, 161, 40))
ax_a.set_yticks(np.arange(0, 161, 40))
ax_a.set_xlabel("Left-ventricular volume (mL)")
ax_a.set_ylabel("Left-ventricular pressure (mmHg)")

# b, the same beat against time; the ejection period is shaded in both
t = base["t"]
for ax in (ax_p, ax_v):
    ax.axvspan(base["events"]["AVO"], base["events"]["AVC"],
               color=ms.GREY_LIGHT, alpha=0.45, lw=0)
ax_p.plot(t, base["p_ao"], color=ms.GREY_DARK, lw=0.9)
ax_p.plot(t, base["p"], color=LOOP, lw=1.3)
ax_v.plot(t, base["v"], color=ms.INK, lw=1.3)
for name, t_event in base["events"].items():
    i = np.argmin(np.abs(t - t_event))
    ax_p.plot(t[i], base["p"][i], zorder=5, clip_on=False, **STYLE[name])
ax_p.text(0.5, np.interp(0.5, t, base["p_ao"]) + 5, "Aorta",
          color=ms.GREY_DARK, va="bottom", fontsize=ms.FS_TICK)
ax_p.text(0.47, 30, "Left\nventricle", color=LOOP, va="center",
          fontsize=ms.FS_TICK)
ax_p.annotate("Ejection", xy=((base["events"]["AVO"]
                               + base["events"]["AVC"]) / 2, 1),
              xycoords=("data", "axes fraction"), xytext=(0, 2.5),
              textcoords="offset points", ha="center", va="bottom",
              color=ms.GREY_DARK, fontsize=ms.FS_TICK)
ax_p.set_xlim(0, T_BEAT)
ax_p.set_ylim(0, 125)
ax_p.set_yticks([0, 40, 80, 120])
ax_p.tick_params(labelbottom=False)
ax_p.set_ylabel("Pressure (mmHg)")
ax_v.set_ylim(40, 130)
ax_v.set_yticks([40, 80, 120])
ax_v.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
ax_v.set_xlabel("Time from end diastole (s)")
ax_v.set_ylabel("Volume (mL)")

ms.panel_label(ax_a, "a", dx_pt=-28, dy_pt=6)
ms.panel_label(ax_p, "b", dx_pt=-28, dy_pt=6)
ms.assert_aligned([ax_a, ax_v], edges=("bottom",))
ms.assert_aligned([ax_a, ax_p], edges=("top",))
ms.assert_aligned([ax_p, ax_v], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig185_pv_loop.{ext}")
print("fig185_pv_loop: saved png + pdf")
