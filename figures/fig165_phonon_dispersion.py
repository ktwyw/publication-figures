"""Fig. 165 - Phonons of a diatomic chain (single column, 89 mm).

The textbook dispersion figure with its checks built in: acoustic and
optical branches of a chain of alternating masses across the first
Brillouin zone, the band gap between them shaded and its edges given by
their formulas, the sound line as the tangent at k → 0, and below it
the group velocity of both branches, which vanishes at the zone
boundary. The eigenfrequencies of a finite periodic chain, found by
diagonalising its dynamical matrix, sit on the curves. The self-check is
that all 2N numerical modes lie on the analytic branches (ω to 1e-8;
the zero mode in ω² to 1e-12), that the gap edges are √(2C/M₁) and
√(2C/M₂), the optical frequency at k = 0 is √(2C(1/M₁ + 1/M₂)), the
acoustic slope at k → 0 is c = a·√(C/(2(M₁ + M₂))), and that dω/dk
agrees with a finite difference and is zero at k = ±π/a.

Model: ω² = C(1/M₁ + 1/M₂) ± C·√[(1/M₁ + 1/M₂)² − 4 sin²(ka/2)/(M₁M₂)]
with M₁ = 2, M₂ = 1, C = 1, cell length a = 1 (dimensionless); finite
chain of N = 12 cells, wavevector of each mode from the Fourier
transform of its eigenvector. No data: every curve is computed from the
equations.
"""

import numpy as np

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
M1, M2 = 2.0, 1.0                 # heavy and light mass
SPRING, CELL = 1.0, 1.0           # force constant C, cell length a
N_CELLS = 12                      # finite periodic chain: 2N modes
S = 1 / M1 + 1 / M2


def root(k):
    return np.sqrt(S ** 2 - 4 * np.sin(k * CELL / 2) ** 2 / (M1 * M2))


def omega(k, branch):
    """branch = −1: acoustic, +1: optical.

    The acoustic root is taken from the product ω₊²ω₋² = 4C² sin²(ka/2)
    /(M₁M₂), which avoids the cancellation in S − root at small k.
    """
    optical = np.sqrt(SPRING * (S + root(k)))
    acoustic = 2 * SPRING * np.abs(np.sin(k * CELL / 2)) / (
        np.sqrt(M1 * M2) * optical)
    return np.where(np.asarray(branch) > 0, optical, acoustic)


def group_velocity(k, branch):
    """dω/dk, differentiated by hand from the dispersion relation."""
    return (-branch * SPRING * CELL * np.sin(k * CELL)
            / (2 * M1 * M2 * root(k) * omega(k, branch)))


SOUND = CELL * np.sqrt(SPRING / (2 * (M1 + M2)))      # long-wavelength limit
GAP = np.sqrt(2 * SPRING / M1), np.sqrt(2 * SPRING / M2)
OPTICAL_TOP = np.sqrt(2 * SPRING * S)

# ------------------------------------------------------------ SOLVER ---
# dynamical matrix of the ring: D = M^(-1/2) K M^(-1/2), eigenvalues ω²
masses = np.tile([M1, M2], N_CELLS)
n_atoms = masses.size
stiffness = 2 * SPRING * np.eye(n_atoms)
for i in range(n_atoms):
    stiffness[i, (i + 1) % n_atoms] = stiffness[(i + 1) % n_atoms, i] = -SPRING
omega2_chain, modes = np.linalg.eigh(stiffness / np.sqrt(np.outer(masses,
                                                                  masses)))
omega_chain = np.sqrt(np.clip(omega2_chain, 0, None))

# |k| of each mode: the peak of its cell-to-cell Fourier spectrum (both
# sublattices, because one of them is at rest at the zone boundary)
spectrum = (np.abs(np.fft.fft(modes[0::2], axis=0)) ** 2
            + np.abs(np.fft.fft(modes[1::2], axis=0)) ** 2)
index = np.argmax(spectrum[:N_CELLS // 2 + 1], axis=0)
k_chain = 2 * np.pi * index / (N_CELLS * CELL)
# a degenerate ±k pair is two standing waves: one dot goes to each sign
for j in range(1, 2 * N_CELLS):
    if index[j] == index[j - 1] and abs(omega_chain[j] - omega_chain[j - 1]) \
            < 1e-9 and k_chain[j - 1] > 0:
        k_chain[j] = -k_chain[j]
branch_chain = np.where(omega_chain > np.mean(GAP), 1, -1)

k_half = np.linspace(1e-6, np.pi / CELL, 400)         # k > 0; ω is even in k
k_full = np.concatenate([-k_half[::-1], k_half])

# ------------------------------------------------------- SELF-CHECK ---
analytic = omega(k_chain, branch_chain)
nonzero = omega_chain > 1e-3
assert nonzero.sum() == 2 * N_CELLS - 1                # one zero mode
mode_error = np.abs(omega_chain - analytic)[nonzero].max()
assert mode_error < 1e-8, mode_error
assert np.abs(omega2_chain - analytic ** 2).max() < 1e-12
assert (branch_chain == 1).sum() == (branch_chain == -1).sum() == N_CELLS
edge = np.pi / CELL
assert abs(omega(edge, -1) - GAP[0]) < 1e-12           # heavy atoms move
assert abs(omega(edge, 1) - GAP[1]) < 1e-12            # light atoms move
assert abs(omega(0.0, 1) - OPTICAL_TOP) < 1e-12
slope = omega(1e-5, -1) / 1e-5
assert abs(slope / SOUND - 1) < 1e-8, slope
assert abs(group_velocity(1e-5, -1) / SOUND - 1) < 1e-8
finite = {b: np.gradient(omega(k_half, b), k_half) for b in (-1, 1)}
fd_error = max(np.abs(finite[b] - group_velocity(k_half, b))[1:-1].max()
               for b in (-1, 1))
assert fd_error < 1e-4, fd_error
assert all(abs(group_velocity(edge, b)) < 1e-12 for b in (-1, 1))
print(f"fig165: self-check passed ({2 * N_CELLS} chain modes on the "
      f"branches, max |Δω| = {mode_error:.1e}; gap {GAP[0]:.4f} to "
      f"{GAP[1]:.4f}; optical ω(0) = {OPTICAL_TOP:.4f}; sound velocity "
      f"{slope:.6f} vs {SOUND:.6f}; dω/dk vs finite difference "
      f"{fd_error:.1e})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 100)
gs = ms.grid(fig, 2, 1, left=14, right=8, top=5, bottom=10, hspace=6,
             height_ratios=[1.75, 1])
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1], sharex=ax_a)
COLOUR = {-1: ms.BLUE, 1: ms.VERMILLION}
x_full, x_half = k_full * CELL / np.pi, k_half * CELL / np.pi

# a, dispersion: gap as an edgeless band, labels inside it
ax_a.axhspan(*GAP, color=ms.GREY_LIGHT, alpha=0.45, linewidth=0, zorder=0)
ax_a.text(0, GAP[1] - 0.035, f"√(2C/M₂) = {GAP[1]:.3f}", ha="center",
          va="top", fontsize=ms.FS_SMALL)
ax_a.text(0, np.mean(GAP), "Band gap", ha="center", va="center",
          fontsize=ms.FS_TICK, fontweight="bold")
ax_a.text(0, GAP[0] + 0.035, f"√(2C/M₁) = {GAP[0]:.3f}", ha="center",
          va="bottom", fontsize=ms.FS_SMALL)
tangent = np.array([-0.72, 0.0, 0.72])                 # ends below the gap
ax_a.plot(tangent, SOUND * np.abs(tangent) * np.pi / CELL, color=ms.GREY,
          lw=0.6, ls=(0, (4, 2.5)), zorder=1)
for branch in (-1, 1):
    ax_a.plot(x_full, omega(k_full, branch), color=COLOUR[branch], lw=1.3,
              zorder=2)
ax_a.plot(k_chain * CELL / np.pi, omega_chain, "o", ms=3, mfc=ms.INK,
          mec="white", mew=0.5, zorder=3, clip_on=False)
ax_a.text(0, OPTICAL_TOP + 0.07, f"√(2C(1/M₁ + 1/M₂)) = {OPTICAL_TOP:.3f}",
          ha="center", va="bottom", fontsize=ms.FS_SMALL)
ax_a.text(0, 1.56, "Optical branch", color=COLOUR[1], ha="center",
          va="center", fontsize=ms.FS_TICK)
ax_a.text(0.96, 0.06, "Acoustic branch", color=COLOUR[-1], ha="right",
          va="bottom", fontsize=ms.FS_TICK)
ax_a.text(0, 0.66, "Sound line\nω = c|k|", color=ms.GREY_DARK, ha="center",
          va="center", fontsize=ms.FS_TICK, linespacing=1.15)
ax_a.text(-0.96, 0.06, f"Dots: N = {N_CELLS} chain", ha="left", va="bottom",
          fontsize=ms.FS_SMALL)
ax_a.set_ylim(0, 1.98)
ax_a.set_yticks(np.arange(0, 1.6, 0.5))
ax_a.set_ylabel("Frequency ω / √(C/M₂)")
ax_a.tick_params(labelbottom=False)

# b, group velocity: the acoustic branch jumps from −c to +c through k = 0
for branch in (-1, 1):
    for sign in (-1, 1):
        ax_b.plot(sign * x_half, sign * group_velocity(k_half, branch),
                  color=COLOUR[branch], lw=1.3)
for sign, name in ((1, "+c"), (-1, "−c")):
    ax_b.plot([-1, 1], [sign * SOUND, sign * SOUND], color=ms.GREY, lw=0.6,
              ls=(0, (4, 2.5)), zorder=0)
    ax_b.text(1.04, sign * SOUND, name, va="center", fontsize=ms.FS_TICK,
              color=ms.GREY_DARK)
ax_b.plot([-1, 1], [0, 0], "o", ms=3.2, mfc="white", mec=ms.INK, mew=0.7,
          zorder=4, clip_on=False)
ax_b.text(-0.96, 0.27, f"c = a√(C/(2(M₁ + M₂))) = {SOUND:.3f}",
          fontsize=ms.FS_SMALL, va="center")
ax_b.text(0.96, -0.3, "Zero at the zone\nboundary (circles)", ha="right",
          va="center", fontsize=ms.FS_SMALL, linespacing=1.15)
ax_b.set_xlim(-1, 1)
ax_b.set_xticks(np.arange(-1, 1.01, 0.5))
ax_b.set_ylim(-0.52, 0.52)
ax_b.set_yticks([-0.4, 0, 0.4])
ax_b.set_xlabel("Wavevector ka/π")
ax_b.set_ylabel("dω/dk / (a√(C/M₂))")

fig.align_ylabels([ax_a, ax_b])
ms.panel_label(ax_a, "a", dx_pt=-30, dy_pt=2)
ms.panel_label(ax_b, "b", dx_pt=-30, dy_pt=2)
ms.assert_aligned([ax_a, ax_b], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig165_phonon_dispersion.{ext}")
print("fig165_phonon_dispersion: saved png + pdf")
