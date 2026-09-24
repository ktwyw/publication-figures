"""Fig. 90 - Influence lines for a simply supported beam (stacked panels).

The structural engineer's moving-load tool, computed directly from
statics for a unit load at position xi on a span L with a section C at
x = a:

    R_A(xi) = 1 - xi/L
    V_C(xi) = -xi/L (xi < a),  1 - xi/L (xi > a)   [jump of 1 at C]
    M_C(xi) = xi (L-a)/L (xi < a),  a (L-xi)/L (xi > a)

Positive and negative regions are shaded (load positions that increase
or decrease the response), the shear jump equals unity at the section,
and the moment influence line peaks at a(L-a)/L.
"""

import numpy as np
import matplotlib.pyplot as plt

import journal_style as js

js.apply()
HERE = js.HERE

L, A = 10.0, 4.0
xi = np.linspace(0, L, 400)

fig, axes = plt.subplots(4, 1, figsize=(3.7, 4.4), sharex=True,
                         height_ratios=[0.55, 1, 1, 1])
ax_s, ax_r, ax_v, ax_m = axes

# ----------------------------------------------------- beam sketch ----
ax_s.axis("off")
ax_s.plot([0, L], [0, 0], color="0.15", lw=2.5, solid_capstyle="butt")
ax_s.plot([0, L], [-0.28, -0.28], marker="^", ms=8, mfc="white",
          mec="0.15", mew=1.2, lw=0)
ax_s.plot([A, A], [-0.12, 0.12], color="C3", lw=1.2)
ax_s.text(A, -0.30, "C", ha="center", va="top", fontsize=8, color="C3")
ax_s.annotate("", xy=(2.5, 0.06), xytext=(2.5, 0.95),
              arrowprops=dict(arrowstyle="-|>", lw=1.1, color="0.3"))
ax_s.text(2.5, 1.05, "moving unit load", ha="center", fontsize=6.5,
          color="0.3")
ax_s.text(0, -0.62, "A", ha="center", va="top", fontsize=7, color="0.3")
ax_s.text(L, -0.62, "B", ha="center", va="top", fontsize=7, color="0.3")
ax_s.set_ylim(-1.0, 1.3)

# --------------------------------------------------- influence lines ----
def shade(ax, x, y):
    ax.fill_between(x, y, 0, where=y >= 0, color="C0", alpha=0.22, lw=0)
    ax.fill_between(x, y, 0, where=y < 0, color="C3", alpha=0.22, lw=0)


r_a = 1 - xi / L
ax_r.plot(xi, r_a, color="C0", lw=1.2)
shade(ax_r, xi, r_a)
ax_r.text(0.18, 0.88, "1", fontsize=7, color="C0")
ax_r.set_ylabel("$R_A$")
ax_r.set_ylim(-0.15, 1.18)

x1, x2 = xi[xi <= A], xi[xi >= A]
v1, v2 = -x1 / L, 1 - x2 / L
ax_v.plot(x1, v1, color="C0", lw=1.2)
ax_v.plot(x2, v2, color="C0", lw=1.2)
ax_v.plot([A, A], [-A / L, 1 - A / L], ls=":", lw=0.8, color="C0")
shade(ax_v, x1, v1)
shade(ax_v, x2, v2)
ax_v.text(A + 0.25, 1 - A / L, f"+{1 - A/L:g}", fontsize=6.5, color="C0",
          va="bottom")
ax_v.text(A - 0.25, -A / L - 0.02, f"\u2212{A/L:g}", fontsize=6.5,
          color="C3", ha="right", va="top")
ax_v.text(9.8, 0.42, "jump = 1 at C", fontsize=6, color="0.4",
          ha="right")
ax_v.set_ylabel("$V_C$")
ax_v.set_ylim(-0.62, 0.82)

m = np.where(xi < A, xi * (L - A) / L, A * (L - xi) / L)
ax_m.plot(xi, m, color="C0", lw=1.2)
shade(ax_m, xi, m)
ax_m.text(A, A * (L - A) / L + 0.12, f"$a(L-a)/L$ = {A*(L-A)/L:g}",
          fontsize=6.5, color="C0", ha="center", va="bottom")
ax_m.set_ylabel("$M_C$ (m)")
ax_m.set_ylim(-0.35, 3.15)
ax_m.set_xlabel(r"position of unit load, $\xi$ (m)")

for ax in (ax_r, ax_v, ax_m):
    ax.axhline(0, lw=0.6, color="0.6")
    ax.axvline(A, ls=":", lw=0.6, color="0.7")
ax_m.set_xlim(0, L)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig090_influence_lines.{ext}")
print("saved fig090_influence_lines.png / .pdf")
