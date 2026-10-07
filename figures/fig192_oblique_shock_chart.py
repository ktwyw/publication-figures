"""Fig. 192 - Oblique-shock θ-β-M chart for γ = 1.4 (single column, 89 mm).

The working chart of supersonic aerodynamics: for each upstream Mach
number the shock angle β against the flow deflection θ is a closed
loop from the Mach angle to a normal shock, with a weak (solid) and a
strong (dashed) solution for every deflection below the maximum and
none above it, where the shock detaches. The locus of maximum
deflection and the sonic locus (M₂ = 1) run side by side through the
noses. The self-check is that every curve starts at arcsin(1/M) and
ends at 90° with θ = 0, that the M → ∞ maximum deflection is 45.58°,
that the numerical maximum of each curve matches the closed-form β at
maximum deflection, that M₂ = 1 on the sonic locus, and that the weak
and strong roots from brentq satisfy the relation to 1e-10.

Model: tan θ = 2 cot β (M² sin²β − 1) / [M²(γ + cos 2β) + 2], perfect
gas, γ = 1.4; loci from NACA Report 1135, eqs 167-168. No data: every
curve is computed from the equations.
"""

import numpy as np
from matplotlib.lines import Line2D
from scipy import optimize

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
GAMMA = 1.4
MACH = [1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0, np.inf]
CHECK_M, CHECK_THETA = 2.0, np.radians(10.0)   # brentq spot check
THETA_MAX_INF = 45.58           # degrees, γ = 1.4 (NACA Report 1135 charts)


def theta(beta, q):
    """Deflection from the θ-β-M relation, in q = 1/M² so that M = ∞ works."""
    return np.arctan2(2 * np.cos(beta) * (np.sin(beta) ** 2 - q),
                      np.sin(beta) * (GAMMA + np.cos(2 * beta) + 2 * q))


def beta_max_deflection(q):
    """Shock angle at maximum deflection (NACA 1135, eq. 168)."""
    g1 = GAMMA + 1
    root = np.sqrt(g1 * (q ** 2 + (GAMMA - 1) * q / 2 + g1 / 16))
    return np.arcsin(np.sqrt((g1 / 4 - q + root) / GAMMA))


def beta_sonic(q):
    """Shock angle for sonic downstream flow (NACA 1135, eq. 167)."""
    g1 = GAMMA + 1
    root = np.sqrt(g1 * (g1 - 2 * (3 - GAMMA) * q + (GAMMA + 9) * q ** 2))
    return np.arcsin(np.sqrt((g1 - (3 - GAMMA) * q + root) / (4 * GAMMA)))


def mach_downstream(beta, q):
    """M₂ behind the shock from the normal-shock relations."""
    mn2 = np.sin(beta) ** 2 / q
    mn2_down = (1 + (GAMMA - 1) / 2 * mn2) / (GAMMA * mn2 - (GAMMA - 1) / 2)
    return np.sqrt(mn2_down) / np.sin(beta - theta(beta, q))


# ------------------------------------------------------------ SOLVER ---
curves = {}
for m in MACH:
    q = 1 / m ** 2
    mu = np.arcsin(1 / m)                        # Mach angle; 0 for M = ∞
    b_max = beta_max_deflection(q)
    found = optimize.minimize_scalar(lambda b: -theta(b, q), method="bounded",
                                     bounds=(mu, np.pi / 2),
                                     options=dict(xatol=1e-12))
    curves[m] = dict(q=q, mu=mu, b_max=b_max, b_num=found.x, t_num=-found.fun,
                     weak=np.linspace(mu, b_max, 300),
                     strong=np.linspace(b_max, np.pi / 2, 200))

q_locus = np.linspace(1, 0, 400)                 # M from 1 to ∞
locus_max = (theta(beta_max_deflection(q_locus), q_locus),
             beta_max_deflection(q_locus))
locus_sonic = (theta(beta_sonic(q_locus), q_locus), beta_sonic(q_locus))

CHECK_Q = 1 / CHECK_M ** 2
b_peak = beta_max_deflection(CHECK_Q)


def residual(beta):
    return theta(beta, CHECK_Q) - CHECK_THETA


b_weak = optimize.brentq(residual, np.arcsin(1 / CHECK_M), b_peak, xtol=1e-14)
b_strong = optimize.brentq(residual, b_peak, np.pi / 2, xtol=1e-14)

# ------------------------------------------------------- SELF-CHECK ---
for m, c in curves.items():
    assert abs(np.sin(c["mu"]) - 1 / m) < 1e-15            # Mach angle
    assert abs(theta(c["weak"][0], c["q"])) < 1e-12        # θ = 0 there
    assert c["strong"][-1] == np.pi / 2
    assert abs(theta(np.pi / 2, c["q"])) < 1e-12           # normal shock
    assert abs(c["b_num"] - c["b_max"]) < np.radians(1e-3), m
    assert abs(c["t_num"] - theta(c["b_max"], c["q"])) < 1e-10, m
theta_inf = np.degrees(theta(curves[np.inf]["b_max"], 0.0))
assert abs(theta_inf - THETA_MAX_INF) < 0.01, theta_inf
inner = slice(1, -1)                             # 1 < M < ∞
m2 = mach_downstream(beta_sonic(q_locus[inner]), q_locus[inner])
assert np.abs(m2 - 1).max() < 1e-9, np.abs(m2 - 1).max()
assert np.all(locus_sonic[1][inner] < locus_max[1][inner])     # weak side
assert max(abs(residual(b_weak)), abs(residual(b_strong))) < 1e-10
print(f"fig192: self-check passed (θmax(M→∞) = {theta_inf:.3f}° at β = "
      f"{np.degrees(curves[np.inf]['b_max']):.2f}°; numerical and analytic "
      f"maxima agree for {len(MACH)} curves; M = {CHECK_M:g}, θ = 10°: weak "
      f"β = {np.degrees(b_weak):.3f}°, strong β = {np.degrees(b_strong):.3f}°;"
      f" |M₂ − 1| on sonic locus < {np.abs(m2 - 1).max():.0e})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 84)
ax = ms.axes(fig, 12, 10.5, 72, 69)
GUTTER = 5.5                # degrees of θ left of the origin for the labels
DASH = (0, (3, 1.6))

for m, c in curves.items():
    ax.plot(np.degrees(theta(c["weak"], c["q"])), np.degrees(c["weak"]),
            color=ms.INK, lw=0.8)
    ax.plot(np.degrees(theta(c["strong"], c["q"])), np.degrees(c["strong"]),
            color=ms.GREY, lw=0.7, ls=DASH)
    ax.text(-0.7, np.degrees(c["mu"]), "∞" if np.isinf(m) else f"{m:g}",
            ha="right", va="center", fontsize=ms.FS_SMALL, clip_on=False)
ax.text(-0.7, np.degrees(curves[MACH[0]]["mu"]) + 4.5, "M₁ =", ha="right",
        va="center", fontsize=ms.FS_SMALL)
ax.plot(*np.degrees(locus_sonic), color=ms.BLUE, lw=1.0)
ax.plot(*np.degrees(locus_max), color=ms.VERMILLION, lw=1.3)

leader = dict(arrowstyle="-", lw=0.5, shrinkA=1, shrinkB=1)
k = int(np.argmin(np.abs(q_locus - 1 / 3.4 ** 2)))
ax.annotate("Maximum deflection", xy=np.degrees([locus_max[0][k],
                                                 locus_max[1][k]]),
            xytext=(32, 84), color=ms.VERMILLION, fontsize=ms.FS_TICK,
            va="center", arrowprops=dict(color=ms.VERMILLION,
                                         relpos=(0.5, 0.0), **leader))
ax.annotate("Sonic line, M₂ = 1", xy=np.degrees([locus_sonic[0][k],
                                                 locus_sonic[1][k]]),
            xytext=(36, 37), color=ms.BLUE, fontsize=ms.FS_TICK,
            va="center", arrowprops=dict(color=ms.BLUE, relpos=(0.3, 1.0),
                                         **leader))
handles = [Line2D([], [], color=ms.INK, lw=0.8, label="Weak shock"),
           Line2D([], [], color=ms.GREY, lw=0.7, ls=DASH,
                  label="Strong shock")]
ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.03),
          handlelength=2.4, title="γ = 1.4", alignment="left")

ax.set_xlim(-GUTTER, 50)
ax.set_ylim(0, 90)
ax.set_xticks(np.arange(0, 51, 10))
ax.set_yticks(np.arange(0, 91, 10))
ax.spines["bottom"].set_bounds(0, 50)
ax.set_xlabel("Flow deflection angle θ (°)")
ax.set_ylabel("Shock angle β (°)")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig192_oblique_shock_chart.{ext}")
print("fig192_oblique_shock_chart: saved png + pdf")
