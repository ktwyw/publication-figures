"""Fig. 184 - Bolus kinetics and repeated oral dosing (double column, 183 mm).

Two textbook pharmacokinetic pictures, each computed from its model:
the bi-exponential plasma curve after an intravenous bolus in a
two-compartment model on a semilog axis, stripped into its distribution
and terminal phases with their half-lives; and repeated oral dosing in
a one-compartment model with first-order absorption, built by
superposition, with the therapeutic window, the steady-state peak and
trough and the time to 90% of steady state. The self-check is that
A/α + B/β and a quadrature of the curve both give Dose/CL to 1e-6, that
the bi-exponential equals a solve_ivp solution of the two-compartment
equations to 1e-6, and that the trough accumulation ratio of the
superposed curve equals 1/(1 − exp(−kτ)) within 0.5%.

Model: a, dose 100 mg, V₁ = 10 L, V₂ = 30 L, CL = 5 L/h, Q = 15 L/h;
samples with 12% log-normal error (n = 13); squares, early samples minus
the terminal line. b, 250 mg every 8 h, F = 0.8, V = 40 L, k = ln 2 / 6 h,
kₐ = 1.2 h⁻¹; window 3-10 mg/L. The drug and its parameters are
placeholders. All data are simulated.
"""

import numpy as np
from scipy import integrate

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(184)

# -------------------------------------------------- GOVERNING MODEL ----
# a, two-compartment model, intravenous bolus into the central compartment
DOSE_IV, V1, V2, CL, Q = 100.0, 10.0, 30.0, 5.0, 15.0    # mg, L, L, L/h, L/h
T_SAMPLE = np.array([0.08, 0.17, 0.33, 0.5, 0.75, 1, 1.5, 2, 4, 8, 12, 18, 24])
CV = 0.12                                    # log-normal assay error
# b, one-compartment model, first-order absorption, repeated oral doses
DOSE_PO, BIOAVAIL, V_PO, KA = 250.0, 0.8, 40.0, 1.2      # mg, -, L, 1/h
K_EL = np.log(2) / 6.0                       # 1/h (half-life 6 h)
TAU, N_DOSES = 8.0, 8                        # dosing interval (h), doses
MEC, TOXIC = 3.0, 10.0                       # therapeutic window (mg/L)

# ------------------------------------------------------------ SOLVER ---
k10, k12, k21 = CL / V1, Q / V1, Q / V2
total = k10 + k12 + k21
root = np.sqrt(total ** 2 - 4 * k10 * k21)
alpha, beta = (total + root) / 2, (total - root) / 2     # hybrid rates
coef_a = DOSE_IV / V1 * (alpha - k21) / (alpha - beta)
coef_b = DOSE_IV / V1 * (k21 - beta) / (alpha - beta)


def bolus(t):
    """Central concentration (mg/L) after the bolus."""
    return coef_a * np.exp(-alpha * t) + coef_b * np.exp(-beta * t)


def two_compartment(_t, amount):
    central, peripheral = amount
    return [-(k10 + k12) * central + k21 * peripheral,
            k12 * central - k21 * peripheral]


t_check = np.linspace(0, 24, 97)
ode = integrate.solve_ivp(two_compartment, (0, 24), [DOSE_IV, 0.0],
                          t_eval=t_check, method="LSODA", rtol=1e-11,
                          atol=1e-12)
samples = bolus(T_SAMPLE) * np.exp(rng.normal(0, CV, T_SAMPLE.size))
# curve stripping: early samples minus the terminal line fall on A exp(−αt)
early = T_SAMPLE <= 1.0
stripped = samples[early] - coef_b * np.exp(-beta * T_SAMPLE[early])


def oral_single(t):
    """Concentration after one oral dose at t = 0 (zero before it)."""
    t = np.maximum(t, 0.0)
    scale = BIOAVAIL * DOSE_PO * KA / (V_PO * (KA - K_EL))
    return scale * (np.exp(-K_EL * t) - np.exp(-KA * t))


def oral_repeated(t):
    """Superposition of N_DOSES doses given every TAU hours."""
    return sum(oral_single(t - i * TAU) for i in range(N_DOSES))


def steady_state(t):
    """Analytic steady-state concentration at time t after a dose."""
    scale = BIOAVAIL * DOSE_PO * KA / (V_PO * (KA - K_EL))
    return scale * (np.exp(-K_EL * t) / (1 - np.exp(-K_EL * TAU))
                    - np.exp(-KA * t) / (1 - np.exp(-KA * TAU)))


t_peak_ss = np.log(KA * (1 - np.exp(-K_EL * TAU))
                   / (K_EL * (1 - np.exp(-KA * TAU)))) / (KA - K_EL)
peak_ss, trough_ss = steady_state(t_peak_ss), steady_state(TAU)
t_last = (N_DOSES - 1) * TAU                 # time of the last dose
t_90 = np.log(10) / K_EL                     # 1 − exp(−k t) = 0.9

# ------------------------------------------------------- SELF-CHECK ---
auc_sum = coef_a / alpha + coef_b / beta
auc_quad = (integrate.quad(bolus, 0, 2.0, epsabs=1e-12, epsrel=1e-12)[0]
            + integrate.quad(bolus, 2.0, np.inf, epsabs=1e-12,
                             epsrel=1e-12)[0])
assert abs(auc_sum - DOSE_IV / CL) < 1e-6 * DOSE_IV / CL
assert abs(auc_quad - DOSE_IV / CL) < 1e-6 * DOSE_IV / CL, auc_quad
ode_error = np.abs(ode.y[0] / V1 - bolus(t_check)).max()
assert ode_error < 1e-6, ode_error
# elimination phase: absorption of every earlier dose is over at a trough
ratio = oral_repeated(N_DOSES * TAU) / oral_single(TAU)
ratio_theory = 1 / (1 - np.exp(-K_EL * TAU))
assert abs(ratio / ratio_theory - 1) < 0.005, (ratio, ratio_theory)
assert abs(oral_repeated(t_last + t_peak_ss) / peak_ss - 1) < 0.005
print(f"fig184: self-check passed (AUC {auc_quad:.6f} = Dose/CL "
      f"{DOSE_IV / CL:.6f} mg h/L; bi-exponential vs solve_ivp "
      f"{ode_error:.1e} mg/L; half-lives {np.log(2) / alpha:.2f} and "
      f"{np.log(2) / beta:.2f} h; accumulation ratio {ratio:.4f} vs "
      f"{ratio_theory:.4f}; steady state {trough_ss:.2f}-{peak_ss:.2f} mg/L)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 68)
gs = ms.grid(fig, 1, 2, left=14, right=13, top=7, bottom=11, wspace=17,
             width_ratios=[1, 1.35])
ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
DIST, TERM = ms.ORANGE, ms.BLUE              # one colour per phase

# a, semilog plot: the two phases are straight lines, trimmed to the frame
Y_LIM = (0.1, 20.0)
t_fine = np.linspace(0, 24, 600)
t_floor = np.log(coef_a / Y_LIM[0]) / alpha  # where A exp(−αt) leaves
t_dist = np.linspace(0, min(t_floor, 24.0), 50)
ax_a.plot(t_fine, coef_b * np.exp(-beta * t_fine), color=TERM, lw=0.9,
          ls=(0, (4, 2)))
ax_a.plot(t_dist, coef_a * np.exp(-alpha * t_dist), color=DIST, lw=0.9,
          ls=(0, (4, 2)))
ax_a.plot(t_fine, bolus(t_fine), color=ms.INK, lw=1.3)
ax_a.plot(T_SAMPLE, samples, "o", ms=3.2, mfc="white", mec=ms.INK, mew=0.8,
          zorder=4)
ax_a.plot(T_SAMPLE[early], stripped, "s", ms=2.8, mfc="white", mec=DIST,
          mew=0.8, zorder=4)
ax_a.text(2.6, 0.13, "Distribution phase,\n"
          f"A exp(−αt), t½ = {np.log(2) / alpha:.2f} h\n"
          "(squares, samples − terminal line)", color=DIST, va="bottom",
          fontsize=ms.FS_TICK, linespacing=1.3)
ax_a.text(11, coef_b * np.exp(-beta * 11) * 1.5, "Terminal phase,\n"
          f"B exp(−βt), t½ = {np.log(2) / beta:.1f} h", color=TERM,
          va="bottom", fontsize=ms.FS_TICK, linespacing=1.3)
ax_a.text(0.97, 0.96, "Line, C(t) = A exp(−αt) + B exp(−βt)\n"
          f"Points, samples (n = {T_SAMPLE.size})\n"
          f"AUC = Dose/CL = {DOSE_IV / CL:.0f} mg h L⁻¹",
          transform=ax_a.transAxes, ha="right", va="top",
          fontsize=ms.FS_TICK, linespacing=1.35)
ax_a.set_yscale("log")
ax_a.set_xlim(-0.6, 24.6)
ax_a.set_ylim(*Y_LIM)
ax_a.set_xticks(np.arange(0, 25, 4))
ax_a.spines["bottom"].set_bounds(0, 24)
ms.plain_log_ticks(ax_a.yaxis)
ax_a.set_xlabel("Time after intravenous bolus (h)")
ax_a.set_ylabel("Plasma concentration (mg L⁻¹)")

# b, repeated dosing: accumulation into the therapeutic window
T_END = N_DOSES * TAU
t_fine = np.linspace(0, T_END, 1600)
ax_b.axhspan(MEC, TOXIC, color=ms.GREEN, alpha=0.12, lw=0)
ax_b.plot(t_fine, oral_repeated(t_fine), color=ms.INK, lw=1.2, zorder=3)
ax_b.plot(TAU * np.arange(N_DOSES), np.full(N_DOSES, 11.2), "v", ms=3.2,
          mfc=ms.GREY_DARK, mew=0, clip_on=False, zorder=5)
ax_b.plot([t_last + t_peak_ss, T_END],
          [oral_repeated(t_last + t_peak_ss), oral_repeated(T_END)], "o",
          ms=3.6, mfc=ms.VERMILLION, mec="white", mew=0.5, clip_on=False,
          zorder=5)
ax_b.plot([t_90, t_90], [0, peak_ss + 0.6], color=ms.GREY_DARK, lw=0.6,
          ls=(0, (2, 2)), zorder=2)
ax_b.text(t_90 + 0.8, 1.0, "90% of steady state\n"
          f"at {t_90:.0f} h (3.3 half-lives)", color=ms.GREY_DARK,
          va="center", fontsize=ms.FS_TICK)
ax_b.text(T_END + 1.2, peak_ss, f"Peak\n{peak_ss:.1f}", va="center",
          color=ms.VERMILLION, fontsize=ms.FS_TICK)
ax_b.text(T_END + 1.2, trough_ss, f"Trough\n{trough_ss:.1f}", va="center",
          color=ms.VERMILLION, fontsize=ms.FS_TICK)
ax_b.text(1.0, TOXIC - 0.3, "Therapeutic window", va="top", color=ms.GREEN,
          fontweight="bold", fontsize=ms.FS_TICK)
ax_b.text(1.0, TOXIC - 1.15, f"minimum effective {MEC:.0f} mg L⁻¹ to toxic "
          f"{TOXIC:.0f} mg L⁻¹", va="top", color=ms.GREEN,
          fontsize=ms.FS_TICK)
ax_b.text(1.0, 12.1, f"Triangles, {DOSE_PO:.0f} mg orally every {TAU:.0f} h",
          va="center", fontsize=ms.FS_TICK)
ax_b.set_xlim(0, T_END)
ax_b.set_ylim(0, 12.5)
ax_b.set_xticks(TAU * np.arange(N_DOSES + 1))
ax_b.set_yticks(np.arange(0, 12.1, 2))
ax_b.set_xlabel("Time after first dose (h)")
ax_b.set_ylabel("Plasma concentration (mg L⁻¹)")

ms.panel_label(ax_a, "a", dx_pt=-30, dy_pt=6)
ms.panel_label(ax_b, "b", dx_pt=-24, dy_pt=6)
ms.assert_aligned([ax_a, ax_b])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig184_pharmacokinetics.{ext}")
print("fig184_pharmacokinetics: saved png + pdf")
