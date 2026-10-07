"""Fig. 186 - Root locus of a third-order loop (single column, 89 mm).

How to draw a root locus that can be read like a design chart: the
closed-loop poles of a unity-feedback loop are solved from the
characteristic polynomial on a fine logarithmic gain grid and joined
into three continuous branches, with the asymptotes and their centroid,
the breakaway point, the imaginary-axis crossing and its critical gain,
and the ζ = 0.5 ray with the poles at the gain that puts them on it.
The self-check is that the numerically found crossing gain equals the
Routh–Hurwitz value within 0.1%, the centroid equals (Σpoles − Σzeros)
/ (n − m), and the breakaway point is a root of dK/ds to 1e-8.

Model: G(s) = K / (s(s + 2)(s + 5)), unity negative feedback, so the
closed-loop poles solve s³ + 7s² + 10s + K = 0; K from 1e-3 to the gain
at which the complex pair reaches |Im s| = 6. No data: every curve is
computed from the equations.
"""

import numpy as np
from matplotlib.lines import Line2D
from scipy import optimize

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
POLES = np.array([0.0, -2.0, -5.0])     # open-loop poles of G(s) (s⁻¹)
ZEROS = np.array([])                    # no finite zeros
ZETA = 0.5                              # design damping ratio
X_LIM, Y_LIM, IM_END = (-11.0, 4.0), (-6.5, 6.5), 6.0
RAY_END = 5.0                           # the ζ ray is drawn to this Im s
DEN = np.poly(POLES)                    # s³ + 7s² + 10s
N_EXCESS = POLES.size - ZEROS.size      # n − m asymptotes


def closed_loop_poles(gain):
    """Roots of 1 + K G(s) = 0, sorted by imaginary part."""
    roots = np.roots(DEN + np.r_[np.zeros(DEN.size - 1), gain])
    return roots[np.argsort(roots.imag)]


# ------------------------------------------------------------ SOLVER ---
# landmarks first, so the gain grid can pass exactly through them
dk_ds = -np.polyder(DEN)                              # K(s) = −den(s)
candidates = np.roots(dk_ds).real
breakaway = candidates[(candidates > POLES[1]) & (candidates < POLES[0])][0]
k_break = -np.polyval(DEN, breakaway)
k_cross = optimize.brentq(lambda k: closed_loop_poles(k)[-1].real, 1, 500,
                          xtol=1e-12, rtol=1e-14)
w_cross = closed_loop_poles(k_cross)[-1].imag


def damping(gain):
    upper = closed_loop_poles(gain)[-1]
    return -upper.real / abs(upper)


k_zeta = optimize.brentq(lambda k: damping(k) - ZETA, 1.001 * k_break,
                         k_cross, xtol=1e-12)
design_poles = closed_loop_poles(k_zeta)
k_end = optimize.brentq(lambda k: closed_loop_poles(k)[-1].imag - IM_END,
                        k_cross, 1e4)

gains = np.unique(np.r_[np.logspace(-3, np.log10(k_end), 3000),
                        k_break, k_zeta, k_cross])
# consistent branches: each new root set is matched to the previous one
branches = np.empty((gains.size, POLES.size), dtype=complex)
branches[0] = POLES[np.argmin(np.abs(POLES[:, None]
                                     - closed_loop_poles(gains[0])), axis=0)]
for i, gain in enumerate(gains[1:], start=1):
    cost = np.abs(branches[i - 1][:, None] - closed_loop_poles(gain)[None, :])
    rows, cols = optimize.linear_sum_assignment(cost)
    branches[i, rows] = closed_loop_poles(gain)[cols]

centroid = (POLES.sum() - ZEROS.sum()) / N_EXCESS
angles = (2 * np.arange(N_EXCESS) + 1) * np.pi / N_EXCESS    # 60°, 180°, 300°

# ------------------------------------------------------- SELF-CHECK ---
# Routh–Hurwitz for s³ + a₂s² + a₁s + K: stable iff 0 < K < a₂a₁
k_routh = DEN[1] * DEN[2]
assert abs(k_cross - k_routh) < 1e-3 * k_routh, (k_cross, k_routh)
assert abs(w_cross - np.sqrt(DEN[2])) < 1e-3 * w_cross        # ω² = a₁
assert abs(centroid - (-DEN[1] / N_EXCESS)) < 1e-12           # Σpoles = −a₂
assert abs(np.polyval(dk_ds, breakaway)) < 1e-8
assert np.abs(np.diff(branches, axis=0)).max() < 0.2          # no jumps
assert np.allclose(np.polyval(DEN, design_poles) + k_zeta, 0, atol=1e-9)
print(f"fig186: self-check passed (critical gain {k_cross:.4f} vs Routh "
      f"{k_routh:.0f}, crossing at ±{w_cross:.4f}j; centroid "
      f"{centroid:.4f}; breakaway {breakaway:.4f} at K = {k_break:.4f}; "
      f"zeta = {ZETA} at K = {k_zeta:.3f})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 76)
ax = ms.axes(fig, 14, 11, 70, 70 * np.ptp(Y_LIM) / np.ptp(X_LIM))
LOCUS, DESIGN = ms.BLUE, ms.VERMILLION

ax.axvspan(X_LIM[0], 0, color=ms.GREY_LIGHT, alpha=0.3, lw=0, zorder=0)
ax.plot(X_LIM, [0, 0], color=ms.GREY, lw=0.4, zorder=1)

# asymptotes from the centroid, trimmed to |Im s| = IM_END
reach = IM_END / np.sin(angles[0])
for angle in (angles[0], angles[-1]):
    ax.plot(centroid + np.array([0, reach * np.cos(angle)]),
            [0, reach * np.sin(angle)], color=ms.GREY_DARK, lw=0.6,
            ls=(0, (4, 2.5)), zorder=2)
ax.plot(centroid, 0, "o", ms=3, mfc="white", mec=ms.GREY_DARK, mew=0.7,
        zorder=6)

# constant-damping ray: cos θ = ζ, measured from the negative real axis
ray = RAY_END / np.sqrt(1 - ZETA ** 2) * np.array([0, 1])
ax.plot(-ZETA * ray, np.sqrt(1 - ZETA ** 2) * ray, color=DESIGN, lw=0.7,
        ls=(0, (1.5, 1.5)), zorder=2)
ax.text(-ZETA * ray[1] + 0.5, RAY_END, f"ζ = {ZETA}", color=DESIGN,
        fontsize=ms.FS_TICK, ha="left", va="center")

for k in range(POLES.size):
    ax.plot(branches[:, k].real, branches[:, k].imag, color=LOCUS, lw=1.3,
            zorder=3)
    tip, tail = branches[-1, k], branches[-40, k]        # towards K → ∞
    ax.annotate("", xy=(tip.real, tip.imag), xytext=(tail.real, tail.imag),
                arrowprops=dict(arrowstyle="-|>", color=LOCUS, lw=0,
                                mutation_scale=7, shrinkA=0, shrinkB=0),
                zorder=3)
ax.plot(POLES.real, POLES.imag, "x", color=ms.INK, ms=5, mew=1.0, zorder=7)

# landmarks, each labelled where no branch, ray or asymptote passes
ax.plot(breakaway, 0, "D", ms=3.2, mfc="white", mec=ms.INK, mew=0.8, zorder=8)
ax.annotate(f"Breakaway\ns = −{-breakaway:.2f}\nK = {k_break:.2f}",
            xy=(breakaway, 0), xytext=(-2.75, 0.45), ha="right", va="bottom",
            fontsize=ms.FS_SMALL, linespacing=1.2,
            arrowprops=dict(arrowstyle="-", color=ms.INK, lw=0.5, shrinkA=2,
                            shrinkB=3, relpos=(1.0, 0.3)))
ax.text(centroid - 0.4, -0.4, f"Centroid\n−{-centroid:.2f}", ha="right",
        va="top", fontsize=ms.FS_SMALL, color=ms.GREY_DARK, linespacing=1.2)
ax.plot([0, 0], [-w_cross, w_cross], "s", ms=3.2, mfc="white", mec=ms.INK,
        mew=0.8, zorder=8)
ax.text(0.45, w_cross - 0.1, f"K = {k_cross:.1f}\nω = {w_cross:.2f} rad s⁻¹",
        ha="left", va="top", fontsize=ms.FS_SMALL, linespacing=1.2)
ax.plot(design_poles.real, design_poles.imag, "o", ms=3.6, color=DESIGN,
        mec="white", mew=0.5, zorder=9)
upper = design_poles[-1]
ax.annotate(f"K = {k_zeta:.2f}", xy=(upper.real, upper.imag),
            xytext=(0.45, upper.imag), ha="left", va="center", color=DESIGN,
            fontsize=ms.FS_SMALL,
            arrowprops=dict(arrowstyle="-", color=DESIGN, lw=0.5, shrinkA=1.5,
                            shrinkB=3))
factors = "".join("s" if p == 0 else f"(s + {-p:g})" for p in POLES)
ax.text(X_LIM[0] + 0.5, Y_LIM[1] - 0.5, f"G(s) = K / ({factors})",
        ha="left", va="top", fontsize=ms.FS_BODY)
ax.text(X_LIM[0] + 0.5, Y_LIM[1] - 1.5, "Stable half-plane shaded",
        ha="left", va="top", fontsize=ms.FS_SMALL, color=ms.GREY_DARK)

handles = [
    Line2D([], [], color=LOCUS, lw=1.3, label="Root locus (arrows: K → ∞)"),
    Line2D([], [], color=ms.GREY_DARK, lw=0.6, ls=(0, (4, 2.5)),
           label=f"Asymptotes (±{np.degrees(angles[0]):.0f}°, 180°)"),
    Line2D([], [], color=ms.INK, ls="none", marker="x", ms=4.5, mew=1.0,
           label="Open-loop poles"),
    Line2D([], [], color=DESIGN, ls="none", marker="o", ms=3.6,
           label=f"Poles at ζ = {ZETA} gain")]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.01),
          fontsize=ms.FS_SMALL, handlelength=2.2)

ax.set_xlim(*X_LIM)
ax.set_ylim(*Y_LIM)
ax.set_aspect("equal")
ax.set_xticks(np.arange(-10, 5, 2))
ax.set_yticks(np.arange(-6, 7, 2))
ax.set_xlabel("Real part, Re s (s⁻¹)")
ax.set_ylabel("Imaginary part, Im s (rad s⁻¹)")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig186_root_locus.{ext}")
print("fig186_root_locus: saved png + pdf")
