"""Fig. 170 - Liquid-drop nuclear binding energies (double column, 183 mm).

The semi-empirical mass formula (Bethe-Weizsäcker) explains the shape of
the binding-energy curve term by term. Panel a builds B/A along the
valley of stability from the constant volume term by subtracting the
surface, Coulomb and asymmetry terms in turn, and marks the maximum that
separates energy release by fusion from release by fission. Panel b maps
B/A over the N-Z plane inside the drip lines, with the line of stability
from ∂B/∂Z = 0 bending away from N = Z. The self-check is that the
analytic most-stable Z(A) agrees with the numerical arg-max over integer
Z within ±1 for every A, that B/A along the valley peaks for A between
55 and 65, and that B/A of A = 56, Z = 26 lies between 8.5 and 9.0 MeV.

Model: B = a_V·A − a_S·A^(2/3) − a_C·Z²/A^(1/3) − a_A·(A − 2Z)²/A ± δ,
δ = a_P/√A (+ even-even, − odd-odd, 0 odd A); coefficients of Rohlf
(1994): 15.75, 17.8, 0.711, 23.7 and 11.18 MeV. The drawn curves and the
map omit the pairing term (it is used in the self-check); "bound" means
positive one-neutron and one-proton separation energies.
No data: every curve is computed from the equations.
"""

import numpy as np

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
# Coefficients (MeV): J. W. Rohlf, Modern Physics from α to Z⁰ (Wiley,
# 1994), the set with a Z² Coulomb term and a_P/√A pairing.
A_V, A_S, A_C, A_A, A_P = 15.75, 17.8, 0.711, 23.7, 11.18
A_GRID = np.arange(2, 271)                   # mass numbers of panel a
Z_MAX, N_MAX = 126, 180                      # extent of the map in panel b
MAGIC = (20, 28, 50, 82, 126)                # shell closures (guides only)


def terms(a, z):
    """Volume, surface, Coulomb and asymmetry terms (MeV, all positive)."""
    return (A_V * a, A_S * a ** (2 / 3), A_C * z ** 2 / a ** (1 / 3),
            A_A * (a - 2 * z) ** 2 / a)


def binding_energy(a, z, pairing=False):
    volume, surface, coulomb, asymmetry = terms(a, z)
    smooth = volume - surface - coulomb - asymmetry
    if not pairing:
        return smooth
    sign = np.where(a % 2 == 1, 0, np.where(z % 2 == 0, 1, -1))
    return smooth + sign * A_P / np.sqrt(a)


def stable_z(a):
    """Z that maximizes B at fixed A: ∂B/∂Z = 0."""
    return a / 2 / (1 + A_C * a ** (2 / 3) / (4 * A_A))


# ------------------------------------------------------------ SOLVER ---
z_valley = stable_z(A_GRID)
volume, surface, coulomb, asymmetry = (t / A_GRID
                                       for t in terms(A_GRID, z_valley))
steps = np.cumsum([volume, -surface, -coulomb, -asymmetry], axis=0)
per_nucleon = steps[-1]                      # B/A along the valley
a_peak = A_GRID[np.argmax(per_nucleon)]

# numerical arg-max over integer Z, pairing included
z_best = np.array([np.argmax(binding_energy(a, np.arange(1, a), True)) + 1
                   for a in A_GRID])

# N-Z plane: B/A where the last neutron and the last proton are bound
z_map, n_map = np.meshgrid(np.arange(1, Z_MAX + 1), np.arange(1, N_MAX + 1),
                           indexing="ij")
b_map = binding_energy(z_map + n_map, z_map)
s_neutron = b_map - binding_energy(z_map + n_map - 1, z_map)
s_proton = b_map - np.where(
    z_map > 1, binding_energy(z_map + n_map - 1, np.maximum(z_map - 1, 1)), 0)
bound = (s_neutron > 0) & (s_proton > 0)
ba_map = np.where(bound, b_map / (z_map + n_map), np.nan)

# ------------------------------------------------------- SELF-CHECK ---
z_gap = np.max(np.abs(z_best - z_valley))
assert z_gap <= 1.0, z_gap
assert 55 <= a_peak <= 65, a_peak
iron = binding_energy(np.array(56), np.array(26), True) / 56
assert 8.5 < iron < 9.0, iron
assert np.all(np.diff(steps, axis=0) < 0)    # every correction lowers B/A
print(f"fig170: self-check passed (analytic Z(A) within {z_gap:.2f} of the "
      f"integer arg-max for A = {A_GRID[0]}-{A_GRID[-1]}; B/A peaks at "
      f"A = {a_peak} with {per_nucleon.max():.3f} MeV; B/A(A=56, Z=26) = "
      f"{iron:.3f} MeV; {bound.sum()} bound nuclides mapped)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 74)
ax_a = ms.axes(fig, 13, 11, 68, 56)
ax_b = ms.axes(fig, 98, 11, 80, 56)          # 80:56 = 180:126, equal scale
ax_cbar = ms.axes(fig, 104, 59.5, 30, 2.2)
CURVE = ms.BLUE

# a, the curve built up term by term: bands are the three corrections
bands = (("Surface", ms.SKY, (150, 14.1)), ("Coulomb", ms.ORANGE, (215, 10.6)),
         ("Asymmetry", ms.GREEN, None))
for (name, colour, spot), upper, lower in zip(bands, steps[:-1], steps[1:]):
    ax_a.fill_between(A_GRID, lower, upper, color=colour, alpha=0.38, lw=0)
    if spot:
        ax_a.text(*spot, name, ha="center", va="center", fontsize=ms.FS_TICK)
ax_a.plot(A_GRID, steps[0], color=ms.GREY_DARK, lw=0.8)
ax_a.plot(A_GRID, per_nucleon, color=CURVE, lw=1.5)
ax_a.plot(a_peak, per_nucleon.max(), "o", ms=3.6, mfc="white", mec=CURVE,
          mew=0.9, zorder=5)
ax_a.text(4, A_V + 0.35, f"Volume term, {A_V:g} MeV per nucleon",
          fontsize=ms.FS_TICK, color=ms.GREY_DARK, va="bottom")
ax_a.annotate(f"B/A: maximum {per_nucleon.max():.2f} MeV at A = {a_peak}",
              xy=(a_peak, per_nucleon.max()), xytext=(a_peak - 8, 7.0),
              color=CURVE, fontsize=ms.FS_TICK, va="top",
              arrowprops=dict(arrowstyle="-", color=CURVE, lw=0.5,
                              shrinkA=1, shrinkB=3, relpos=(0.06, 1.0)))
a_tag = 236                                  # the asymmetry band is thin
mid = (steps[2] + steps[3])[A_GRID == a_tag][0] / 2
ax_a.annotate("Asymmetry", xy=(a_tag, mid), xytext=(a_tag + 18, 5.9),
              fontsize=ms.FS_TICK, ha="right", va="top",
              arrowprops=dict(arrowstyle="-", color=ms.INK, lw=0.5,
                              shrinkA=1, shrinkB=0, relpos=(0.78, 1.0)))
arrow = dict(arrowstyle="-|>", color=ms.INK, lw=0.8, shrinkA=0, shrinkB=0,
             mutation_scale=6)
for start, name, ha in ((6, "Fusion", "left"), (262, "Fission", "right")):
    end = a_peak - 7 if start < a_peak else a_peak + 7
    ax_a.annotate("", xy=(end, 3.3), xytext=(start, 3.3), arrowprops=arrow)
    ax_a.text(start, 2.6, f"{name} releases energy", ha=ha, va="top",
              fontsize=ms.FS_TICK)
ax_a.set_xlim(0, 270)
ax_a.set_ylim(0, 17.6)
ax_a.set_xticks(np.arange(0, 251, 50))
ax_a.set_yticks(np.arange(0, 17, 4))
ax_a.set_xlabel("Mass number A")
ax_a.set_ylabel("Energy per nucleon (MeV)")

# b, B/A over the N-Z plane (one pixel per nuclide, rasterized)
mesh = ax_b.imshow(ba_map, origin="lower", cmap="viridis", vmin=6.0,
                   vmax=8.8, interpolation="nearest", rasterized=True,
                   aspect="auto", extent=[0.5, N_MAX + 0.5, 0.5, Z_MAX + 0.5])
ax_b.set_xlim(0, N_MAX)                      # set now: the guides and line
ax_b.set_ylim(0, Z_MAX)                      # below are trimmed to the frame
for magic in MAGIC:                          # guides span the bound region
    n_ok = np.flatnonzero(bound[magic - 1]) + 1
    if n_ok.size:                            # Z = 126 is unbound on this map
        ax_b.plot(n_ok[[0, -1]], [magic, magic], color="white", lw=0.4,
                  alpha=0.75)
    z_ok = np.flatnonzero(bound[:, magic - 1]) + 1
    ax_b.plot([magic, magic], z_ok[[0, -1]], color="white", lw=0.4,
              alpha=0.75)
    ax_b.text(magic, z_ok[-1] + 4, str(magic), fontsize=ms.FS_SMALL,
              color=ms.GREY_DARK, ha="center", va="bottom")
a_line = np.arange(2.0, 330.0)               # the line runs to the frame
a_line = a_line[a_line - stable_z(a_line) <= N_MAX]
z_line, n_line = stable_z(a_line), a_line - stable_z(a_line)
# white on a slightly wider black line: visible on every colour of the map
ax_b.plot(n_line, z_line, color="black", lw=1.6)
ax_b.plot(n_line, z_line, color="white", lw=0.8)
ax_b.plot([0, 100], [0, 100], color=ms.INK, lw=0.6, ls=(0, (4, 3)))
ax_b.text(100, 103, "N = Z", fontsize=ms.FS_TICK, ha="center", va="bottom")
ax_b.text(176, 5, "Line: most stable Z(A), from ∂B/∂Z = 0\n"
          "Thin guides and numbers: magic N and Z\n"
          "Mapped: neutron and proton both bound",
          fontsize=ms.FS_SMALL, ha="right", va="bottom", linespacing=1.4)
ax_b.set_xticks(np.arange(0, 181, 30))
ax_b.set_yticks(np.arange(0, 121, 20))
ax_b.set_xlabel("Neutron number N")
ax_b.set_ylabel("Proton number Z")
cbar = fig.colorbar(mesh, cax=ax_cbar, orientation="horizontal",
                    ticks=[6, 7, 8, 8.8], extend="min")
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(length=2, width=0.5, labelsize=ms.FS_SMALL)
cbar.ax.set_title("B/A (MeV)", fontsize=ms.FS_TICK, pad=2.5)

ms.panel_label(ax_a, "a", dx_pt=-26, dy_pt=5)
ms.panel_label(ax_b, "b", dx_pt=-26, dy_pt=5)
ms.assert_aligned([ax_a, ax_b])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig170_binding_energy.{ext}")
print("fig170_binding_energy: saved png + pdf")
