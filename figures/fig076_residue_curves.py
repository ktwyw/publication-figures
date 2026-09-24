"""Fig. 76 - Residue curve map for a ternary mixture (single column).

Simple-distillation residue curves from the defining ODE

    dx_i/dxi = x_i - y_i,     y_i = alpha_i x_i / sum_j alpha_j x_j,

integrated forward and backward from seed compositions with solve_ivp
(constant relative volatilities alpha = 4 : 2 : 1). All curves leave the
light vertex (unstable node), sweep past the intermediate (saddle) and
terminate at the heavy vertex (stable node) - the topology that governs
feasible distillation sequences. Plotted in barycentric coordinates.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import journal_style as js

js.apply()
HERE = js.HERE

ALPHA = np.array([4.0, 2.0, 1.0])                  # light, intermediate, heavy
V_L = np.array([0.5, np.sqrt(3) / 2])              # triangle vertices
V_I = np.array([1.0, 0.0])
V_H = np.array([0.0, 0.0])


def rhs(_, x):
    y = ALPHA * x / np.dot(ALPHA, x)
    return x - y


def to_xy(x):
    """Barycentric (xL, xI, xH) -> plane; x has shape (3, n)."""
    return (np.outer(V_L, x[0]) + np.outer(V_I, x[1])
            + np.outer(V_H, x[2]))


def edge(_, x):
    return min(x) - 1e-4


edge.terminal = True

SEEDS = [(0.80, 0.10), (0.60, 0.30), (0.40, 0.50), (0.20, 0.70),
         (0.10, 0.45), (0.30, 0.15), (0.55, 0.08), (0.08, 0.15)]

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.8, 3.4))
ax.set_aspect("equal")
ax.axis("off")

tri = np.array([V_H, V_I, V_L, V_H]).T
ax.plot(tri[0], tri[1], color="black", lw=1.1)

for xl, xi in SEEDS:
    x0 = np.array([xl, xi, 1 - xl - xi])
    branches = []
    for sign in (+1, -1):
        sol = solve_ivp(lambda t, x: sign * np.asarray(rhs(t, x)),
                        [0, 15], x0, events=edge, max_step=0.1,
                        dense_output=True)
        branches.append(sol.y[:, ::(1 if sign > 0 else -1)])
    x_full = np.hstack([branches[1], x0[:, None], branches[0]])
    P = to_xy(x_full)
    ax.plot(P[0], P[1], color="C0", lw=0.9)
    # direction arrow anchored mid-descent (x_L ~ 0.5), finite length:
    # near the nodes trajectories crawl, so index-based arrows collapse
    k = int(np.argmin(np.abs(x_full[0] - 0.5)))
    j = k
    while (j < P.shape[1] - 1
           and np.hypot(P[0, j] - P[0, k], P[1, j] - P[1, k]) < 0.05):
        j += 1
    ax.annotate("", xy=(P[0, j], P[1, j]), xytext=(P[0, k], P[1, k]),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="C0"))

# nodes and vertex labels
ax.plot(*V_L, "o", ms=6, mfc="white", mec="C3", mew=1.3, zorder=5)
ax.plot(*V_H, "o", ms=6, color="C3", zorder=5)
ax.plot(*V_I, "s", ms=5.5, mfc="white", mec="0.3", mew=1.2, zorder=5)
ax.annotate("L (light)\nunstable node", V_L, xytext=(0, 10),
            textcoords="offset points", ha="center", fontsize=7)
ax.annotate("H (heavy)\nstable node", V_H, xytext=(-4, -12),
            textcoords="offset points", ha="left", va="top", fontsize=7)
ax.annotate("I (intermediate)\nsaddle", V_I, xytext=(4, -12),
            textcoords="offset points", ha="right", va="top", fontsize=7)
ax.text(0.5, -0.155, r"$\alpha_L : \alpha_I : \alpha_H$ = 4 : 2 : 1",
        ha="center", fontsize=7, color="0.35")

ax.set_xlim(-0.12, 1.12)
ax.set_ylim(-0.20, 1.02)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig076_residue_curves.{ext}")
print("saved fig076_residue_curves.png / .pdf")
