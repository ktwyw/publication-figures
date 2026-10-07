"""Fig. 161 - Graphene π bands and density of states (1.5 column, 120 mm).

The standard pairing of a band structure with its density of states on
a shared energy axis: the nearest-neighbour tight-binding π bands of
graphene along Γ–K–M–Γ touch at the Dirac point K, and the histogram of
band energies over the whole Brillouin zone beside them vanishes
linearly there and peaks at the M-point saddles (van Hove
singularities at ±t). The self-check is that E(K) = 0 to 1e-12,
E(Γ) = ±3t and E(M) = ±t, that the bands are symmetric about zero, that
each band's DOS integrates to one state per cell within 0.5%, that the
histogram peaks within one bin of ±t, and that its slope at the Dirac
point matches the Dirac-cone value 2/(√3·π·t²) within 5%.

Model: E±(k) = ±t|1 + exp(ik·a₁) + exp(ik·a₂)|, a₁ = a(1, 0),
a₂ = a(1/2, √3/2), energies in units of t; DOS (per spin, per unit
cell) from a 2,000 × 2,000 midpoint grid over the reciprocal cell in
bins of 0.025t. The one random draw is the set of wavevectors at which
the 2 × 2 Bloch Hamiltonian is diagonalised to test the closed form.
No data: every curve is computed from the equations.
"""

import numpy as np
from matplotlib.colors import to_rgb

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
T_HOP = 1.0                                # hopping t: the energy unit
A1, A2 = np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2])   # a = 1
B1 = 2 * np.pi * np.array([1.0, -1 / np.sqrt(3)])      # bᵢ·aⱼ = 2π δᵢⱼ
B2 = 2 * np.pi * np.array([0.0, 2 / np.sqrt(3)])
POINTS = {"Γ": np.zeros(2), "K": (2 * B1 + B2) / 3, "M": B1 / 2}
PATH = ["Γ", "K", "M", "Γ"]
N_PATH, N_BZ, BIN = 300, 2000, 0.025       # per segment, per axis, bin (t)
E_MAX = 3.3


def conduction(k):
    """Upper band E₊(k) = t|f(k)|; the lower band is its mirror image."""
    return T_HOP * np.abs(1 + np.exp(1j * (k @ A1)) + np.exp(1j * (k @ A2)))


# ------------------------------------------------------------ SOLVER ---
# bands along the path, x = cumulative length in units of 1/a
path_x, path_e, nodes = [], [], [0.0]
for start, end in zip(PATH[:-1], PATH[1:]):
    s = np.linspace(0, 1, N_PATH)
    k = POINTS[start] + s[:, None] * (POINTS[end] - POINTS[start])
    length = np.linalg.norm(POINTS[end] - POINTS[start])
    path_x.append(nodes[-1] + s * length)
    path_e.append(conduction(k))
    nodes.append(nodes[-1] + length)
path_x, path_e = np.concatenate(path_x), np.concatenate(path_e)

# density of states: histogram of E₊ over a uniform midpoint grid of the
# reciprocal unit cell (same area as the hexagonal zone)
u = (np.arange(N_BZ) + 0.5) / N_BZ
e_bz = conduction((u[:, None, None] * B1 + u[None, :, None] * B2)
                  .reshape(-1, 2))
edges = np.linspace(0, 3.2 * T_HOP, int(round(3.2 * T_HOP / BIN)) + 1)
dos_half = np.histogram(e_bz, bins=edges)[0] / (e_bz.size * BIN)
centres = (edges[:-1] + edges[1:]) / 2
# the valence band is the exact mirror image: g(−E) = g(E)

# ------------------------------------------------------- SELF-CHECK ---
assert conduction(POINTS["K"]) < 1e-12                  # Dirac point
assert abs(conduction(POINTS["Γ"]) - 3 * T_HOP) < 1e-12
assert abs(conduction(POINTS["M"]) - T_HOP) < 1e-12
# the closed form against the 2 × 2 Bloch Hamiltonian [[0, f], [f*, 0]] at
# random wavevectors: eigenvalues ±t|f|, symmetric about zero
probe = np.random.default_rng(161).uniform(-10, 10, (1000, 2))
f_k = T_HOP * (1 + np.exp(1j * (probe @ A1)) + np.exp(1j * (probe @ A2)))
bloch = np.zeros((probe.shape[0], 2, 2), dtype=complex)
bloch[:, 0, 1], bloch[:, 1, 0] = f_k, f_k.conj()
levels = np.linalg.eigvalsh(bloch)
assert np.abs(levels.sum(axis=1)).max() < 1e-12
assert np.abs(levels[:, 1] - conduction(probe)).max() < 1e-12
states = dos_half.sum() * BIN                           # per band, per cell
assert abs(states - 1) < 0.005, states
e_peak = centres[np.argmax(dos_half)]
assert abs(e_peak - T_HOP) <= BIN, e_peak               # van Hove at ±t
DIRAC_SLOPE = 2 / (np.sqrt(3) * np.pi * T_HOP ** 2)     # cone, v = √3·t·a/2
low = centres < 0.2 * T_HOP
slope = (dos_half[low] @ centres[low]) / (centres[low] @ centres[low])
assert abs(slope / DIRAC_SLOPE - 1) < 0.05, slope
print(f"fig161: self-check passed (E(K) = {conduction(POINTS['K']):.1e}, "
      f"E(Γ) = ±{conduction(POINTS['Γ']):.3f}t, E(M) = "
      f"±{conduction(POINTS['M']):.3f}t; {states:.4f} states per band; DOS "
      f"peak at ±{e_peak:.4f}t; Dirac slope {slope:.4f} vs "
      f"{DIRAC_SLOPE:.4f} per t² per cell)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 72)
gs = ms.grid(fig, 1, 2, left=13, right=4, top=7, bottom=10, wspace=4,
             width_ratios=[1.75, 1])
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1], sharey=ax_a)
VALENCE, CONDUCTION = ms.BLUE, ms.VERMILLION

# a, the bands; symmetry lines stop short of the labels above the frame
for x_node in nodes[1:-1]:
    ax_a.plot([x_node, x_node], [-E_MAX, E_MAX], color=ms.GREY_LIGHT, lw=0.6,
              zorder=1)
ax_a.plot(nodes[0::3], [0, 0], color=ms.GREY, lw=0.6, ls=(0, (4, 3)),
          zorder=2)
ax_a.plot(path_x, path_e, color=CONDUCTION, lw=1.3, zorder=3)
ax_a.plot(path_x, -path_e, color=VALENCE, lw=1.3, zorder=3)
ax_a.plot(nodes[1], 0, "o", ms=4, mfc="white", mec=ms.INK, mew=0.8, zorder=4)
ax_a.text(nodes[1] - 0.75, 0.13, "Dirac point", ha="right", va="bottom",
          fontsize=ms.FS_TICK)
ax_a.text(nodes[2] + 0.3, 0.8, "Saddle, E = +t", va="top",
          fontsize=ms.FS_TICK, color=ms.GREY_DARK)
ax_a.text(nodes[2] + 0.3, -0.8, "Saddle, E = −t", va="bottom",
          fontsize=ms.FS_TICK, color=ms.GREY_DARK)
ax_a.text(0.35, 1.0, "Conduction\nband π*", color=CONDUCTION,
          fontsize=ms.FS_TICK, va="center", linespacing=1.15)
ax_a.text(0.35, -1.0, "Valence\nband π", color=VALENCE, fontsize=ms.FS_TICK,
          va="center", linespacing=1.15)
ax_a.set_xlim(nodes[0], nodes[-1])
ax_a.set_ylim(-E_MAX, E_MAX)
ax_a.set_xticks(nodes, PATH)
ax_a.tick_params(axis="x", length=0, labelsize=ms.FS_BODY)
ax_a.set_yticks(np.arange(-3, 3.1, 1))
ax_a.set_xlabel("Wavevector along Γ–K–M–Γ")
ax_a.set_ylabel("Energy E/t")
for side in ("top", "right"):
    ax_a.spines[side].set_visible(True)

# b, the histogram on its side: one edgeless pale bar per bin (opaque and
# slightly overlapping, so neighbours leave no seams), one outline on top
outline_e, outline_g = np.repeat(edges, 2)[1:-1], np.repeat(dos_half, 2)
for sign, colour in ((-1, VALENCE), (1, CONDUCTION)):
    pale = 0.25 * np.array(to_rgb(colour)) + 0.75
    ax_b.barh(sign * centres, dos_half, height=1.05 * BIN, color=pale,
              linewidth=0)
    ax_b.plot(outline_g, sign * outline_e, color=colour, lw=0.8)
e_cone = np.array([-0.55, 0.0, 0.55]) * T_HOP
ax_b.plot(DIRAC_SLOPE * np.abs(e_cone), e_cone, color=ms.INK, lw=0.6)
dos_max = dos_half.max()
x_max = 1.08 * dos_max
ax_b.plot([0.45 * dos_max, x_max], [0, 0], color=ms.GREY, lw=0.6,
          ls=(0, (4, 3)))
ax_b.text(0.97 * x_max, 0.13, "Fermi level", ha="right", va="bottom",
          fontsize=ms.FS_TICK, color=ms.GREY_DARK)
ax_b.text(0.26 * dos_max, -0.1, "g ∝ |E|", fontsize=ms.FS_TICK, va="top")
for sign, va in ((1, "bottom"), (-1, "top")):
    ax_b.text(0.97 * x_max, sign * 1.5, "van Hove\nsingularity\nE = "
              + ("+t" if sign > 0 else "−t"), ha="right", va=va,
              fontsize=ms.FS_TICK, linespacing=1.15)
ax_b.set_xlim(0, x_max)
ax_b.set_xticks([0, 0.5, 1.0])
ax_b.tick_params(labelleft=False)
ax_b.set_xlabel("DOS g (states per t per cell)")

fig.align_xlabels([ax_a, ax_b])
ms.panel_label(ax_a, "a", dx_pt=-26)
ms.panel_label(ax_b, "b", dx_pt=-3)
ms.assert_aligned([ax_a, ax_b])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig161_band_structure_dos.{ext}")
print("fig161_band_structure_dos: saved png + pdf")
