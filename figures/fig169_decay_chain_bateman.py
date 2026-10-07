"""Fig. 169 - Decay chain: Bateman solution, ingrowth (single column, 89 mm).

A four-member radioactive chain A → B → C → D (stable) on log axes: the
parent decays, the daughters grow in, peak and then follow the parent in
transient equilibrium, and the stable end member collects everything.
Lines are the analytic Bateman solution; open markers are an independent
numerical integration of the rate equations. The self-check is that the
two agree to 1e-6 (relative) wherever a population exceeds 1e-9 of the
initial amount, that the total number of nuclei is conserved to 1e-10,
and that the daughter peaks at t = ln(λ_B/λ_A)/(λ_B − λ_A).

Model: dN₁/dt = −λ₁N₁, dNᵢ/dt = λᵢ₋₁Nᵢ₋₁ − λᵢNᵢ, with N_A(0) = N₀ and
the rest zero; half-lives 10 d (A), 2 d (B), 0.25 d (C), D stable;
λ = ln 2/T½; numerical solution by solve_ivp (DOP853, rtol 1e-12).
No data: every curve is computed from the equations.
"""

import numpy as np
from scipy import integrate

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
NAMES = ("A", "B", "C", "D (stable)")
HALF_LIFE = np.array([10.0, 2.0, 0.25, np.inf])      # days
LAMBDA = np.log(2) / HALF_LIFE                       # decay constants (d⁻¹)
T_MIN, T_MAX = 0.01, 300.0                           # plotted times (d)
N_FLOOR = 1e-6                                       # lower axis limit, N/N₀


def bateman(t):
    """N_i(t)/N₀ for a chain with distinct decay constants (Bateman 1910)."""
    t = np.atleast_1d(t)[:, None]
    out = []
    for n in range(LAMBDA.size):
        lam = LAMBDA[:n + 1]
        gaps = lam[None, :] - lam[:, None]           # gaps[j, k] = λ_k − λ_j
        np.fill_diagonal(gaps, 1.0)
        out.append(np.prod(lam[:-1])
                   * np.sum(np.exp(-lam * t) / np.prod(gaps, axis=1), axis=1))
    return np.array(out)


def rates(_t, n):
    decay = LAMBDA * n
    return np.concatenate([[0.0], decay[:-1]]) - decay


def daughter_peak(_t, n):
    return LAMBDA[0] * n[0] - LAMBDA[1] * n[1]       # dN_B/dt


# ------------------------------------------------------------ SOLVER ---
t_line = np.geomspace(T_MIN, T_MAX, 500)
t_mark = np.geomspace(T_MIN, T_MAX, 19)[1:-1]        # sparse markers
t_check = np.geomspace(T_MIN, T_MAX, 120)
n_line = bateman(t_line)
numeric = integrate.solve_ivp(
    rates, (0, T_MAX), [1.0, 0, 0, 0], method="DOP853", rtol=1e-12,
    atol=1e-20, dense_output=True, events=daughter_peak)
t_peak_numeric = numeric.t_events[0][0]
t_peak = np.log(LAMBDA[1] / LAMBDA[0]) / (LAMBDA[1] - LAMBDA[0])
equilibrium = LAMBDA[0] / (LAMBDA[1] - LAMBDA[0])    # N_B/N_A at late times

# ------------------------------------------------------- SELF-CHECK ---
exact, approx = bateman(t_check), numeric.sol(t_check)
populated = exact > 1e-9
mismatch = np.max(np.abs(approx[populated] / exact[populated] - 1))
assert mismatch < 1e-6, mismatch
drift = np.max(np.abs(exact.sum(axis=0) - 1))
assert drift < 1e-10, drift
assert abs(t_peak_numeric / t_peak - 1) < 1e-6, (t_peak_numeric, t_peak)
late = bateman(100.0)[:, 0]
assert abs(late[1] / late[0] / equilibrium - 1) < 1e-6
print(f"fig169: self-check passed (Bateman vs solve_ivp: max relative "
      f"difference {mismatch:.1e} over {populated.sum()} values; total "
      f"conserved to {drift:.1e}; daughter peak at {t_peak_numeric:.5f} d, "
      f"analytic {t_peak:.5f} d; N_B/N_A -> {equilibrium:.3f})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 78)
ax = ms.axes(fig, 14, 10, 71, 57)
COLOURS = (ms.INK, ms.VERMILLION, ms.BLUE, ms.GREY)

n_mark = numeric.sol(t_mark)
for values, marks, colour in zip(n_line, n_mark, COLOURS):
    shown = values >= N_FLOOR                        # trimmed to the frame
    ax.plot(t_line[shown], values[shown], color=colour, lw=1.2)
    keep = marks >= N_FLOOR
    ax.plot(t_mark[keep], marks[keep], "o", ms=3.2, mfc="white", mec=colour,
            mew=0.7, zorder=4)

# the daughter's maximum, from the analytic formula
n_peak = bateman(t_peak)[1, 0]
ax.plot([t_peak, t_peak], [N_FLOOR, n_peak], color=ms.VERMILLION, lw=0.5,
        ls=(0, (1, 2)), zorder=1)
ax.plot(t_peak, n_peak, "D", ms=2.6, color=ms.VERMILLION, mew=0, zorder=5)
ax.text(t_peak * 1.15, N_FLOOR * 1.5, f"t = {t_peak:.2f} d",
        color=ms.VERMILLION, fontsize=ms.FS_TICK, va="bottom")

# direct labels, each in a gap beside its own curve
labels = ((0.012, 1.35, "left"), (0.012, 1.3e-2, "left"),
          (0.012, 2.4e-4, "left"), (270.0, 1.35, "right"))
for (x, y, ha), name, half_life, colour in zip(labels, NAMES, HALF_LIFE,
                                               COLOURS):
    detail = f", T½ = {half_life:g} d" if np.isfinite(half_life) else ""
    ax.text(x, y, name + detail, color=colour, ha=ha, va="bottom",
            fontsize=ms.FS_TICK)
proxies = [ax.plot([], [], color=ms.GREY_DARK, lw=1.2)[0],
           ax.plot([], [], "o", ms=3.2, mfc="white", mec=ms.GREY_DARK,
                   mew=0.7)[0]]
ax.legend(proxies, ["Bateman", "solve_ivp"], loc="lower right",
          bbox_to_anchor=(4.6, 1.3e-6), bbox_transform=ax.transData,
          borderaxespad=0)

# header: the two results the chain is chosen to show (subscripts in
# mathtext, hence FS_MATH)
lam_a, lam_b = r"$\lambda_\mathrm{A}$", r"$\lambda_\mathrm{B}$"
n_a, n_b = r"$N_\mathrm{A}$", r"$N_\mathrm{B}$"
fig.text(14 / 89, 1 - 4.2 / 78,
         f"B peaks at t = ln({lam_b}/{lam_a})/({lam_b} − {lam_a}) = "
         f"{t_peak:.2f} d,", fontsize=ms.FS_MATH, va="center")
fig.text(14 / 89, 1 - 8.2 / 78,
         f"and {n_b}/{n_a} → {lam_a}/({lam_b} − {lam_a}) = "
         f"{equilibrium:.2f} (transient equilibrium)",
         fontsize=ms.FS_MATH, va="center")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(T_MIN, T_MAX)
ax.set_ylim(N_FLOOR, 3.0)
ms.plain_log_ticks(ax.xaxis)
ax.set_yticks(10.0 ** np.arange(-6, 1))
ax.tick_params(labelsize=ms.FS_MATH)          # mathtext exponents ≥ 5 pt
ax.set_xlabel("Time t (days)")
ax.set_ylabel("Number of nuclei N/N₀")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig169_decay_chain_bateman.{ext}")
print("fig169_decay_chain_bateman: saved png + pdf")
