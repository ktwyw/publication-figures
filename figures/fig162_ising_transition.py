"""Fig. 162 - 2-D Ising transition by Monte Carlo (double column, 183 mm).

How to show a phase transition from a simulation: the order parameter
against temperature for several lattice sizes converging on the exact
infinite-lattice curve, the susceptibility peak that sharpens and moves
towards T_c as the lattice grows, and spin snapshots below, at and above
T_c that show what the numbers mean. The self-check is that the exact
T_c = 2/ln(1 + √2) equals 2.269185 to 1e-6, that at the lowest
temperature the magnetisation of every lattice agrees with Onsager's
formula within 2% and the energy per spin lies within 0.1J of −2J, that
at the highest temperature |m| falls with lattice size, and that the
susceptibility peak grows with lattice size.

Model: H = −J Σ sᵢsⱼ over nearest neighbours, periodic L × L lattices
(L = 8, 16, 32), J = k_B = 1; checkerboard Metropolis, all temperatures
updated at once, cold start, 1,000 sweeps discarded and 5,000 measured
in n = 20 blocks of 250. Error bars: s.e.m. over blocks for ⟨|m|⟩,
block jackknife for χ = N(⟨m²⟩ − ⟨|m|⟩²)/T. Snapshots: L = 64 after
4,000 sweeps. All data are simulated.
"""

import numpy as np

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(162)

# ------------------------------------------------------------- DATA ----
SIZES = [8, 16, 32]
# temperatures in units of J/k_B: steps of 0.05 through the transition
TEMPS = np.concatenate([np.linspace(1.5, 1.9, 5), np.linspace(2.0, 2.75, 16),
                        np.linspace(2.8, 3.5, 8)])
N_EQUIL, N_BLOCKS, BLOCK = 1000, 20, 250                   # sweeps
T_C = 2 / np.log(1 + np.sqrt(2))       # Kramers & Wannier 1941; Onsager 1944
SNAP_L, SNAP_SWEEPS = 64, 4000
SNAP_TEMPS = np.array([2.0, T_C, 3.0])


def onsager(temp):
    """Spontaneous magnetisation of the infinite lattice (Yang 1952)."""
    temp = np.asarray(temp, dtype=float)
    below = np.clip(1 - np.sinh(2 / np.minimum(temp, T_C)) ** -4.0, 0, None)
    return np.where(temp < T_C, below ** 0.125, 0.0)


# --------------------------------------------------------- SIMULATION --
def neighbours(spins):
    """Sum of the four periodic neighbours (wrapped slices beat np.roll)."""
    rows = np.concatenate([spins[:, -1:], spins, spins[:, :1]], axis=1)
    cols = np.concatenate([spins[:, :, -1:], spins, spins[:, :, :1]], axis=2)
    return rows[:, :-2] + rows[:, 2:] + cols[:, :, :-2] + cols[:, :, 2:]


def metropolis(size, temps, n_discard, n_measure):
    """Checkerboard Metropolis for all temperatures at once.

    One colour of the checkerboard has all its neighbours on the other, so
    a whole colour is updated in a single vectorised step. Returns m and
    the energy per spin after every measured sweep, and the final spins.
    """
    spins = np.ones((temps.size, size, size), dtype=np.int8)
    row, col = np.indices((size, size))
    colours = [(row + col) % 2 == parity for parity in (0, 1)]
    # single precision halves the cost of the exponential and the draws
    two_beta = (2 / temps).astype(np.float32)[:, None, None]
    m = np.empty((n_measure, temps.size))
    e = np.empty_like(m)
    for sweep in range(n_discard + n_measure):
        for colour in colours:
            half_cost = spins * neighbours(spins)          # ΔE of a flip / 2J
            accept = (rng.random(spins.shape, dtype=np.float32)
                      < np.exp(-two_beta * half_cost))
            spins = np.where(accept & colour, -spins, spins)
        if sweep >= n_discard:
            m[sweep - n_discard] = spins.mean(axis=(1, 2))
            # each bond is counted from both of its ends
            e[sweep - n_discard] = -(spins * neighbours(spins)).mean(
                axis=(1, 2)) / 2
    return m, e, spins


results = {}
for size in SIZES:
    m, e, _ = metropolis(size, TEMPS, N_EQUIL, N_BLOCKS * BLOCK)
    abs_blocks = np.abs(m).reshape(N_BLOCKS, BLOCK, -1).mean(axis=1)
    sq_blocks = (m ** 2).reshape(N_BLOCKS, BLOCK, -1).mean(axis=1)
    # jackknife: χ with each block left out in turn
    abs_out = (abs_blocks.sum(axis=0) - abs_blocks) / (N_BLOCKS - 1)
    sq_out = (sq_blocks.sum(axis=0) - sq_blocks) / (N_BLOCKS - 1)
    chi_out = size ** 2 * (sq_out - abs_out ** 2) / TEMPS
    results[size] = dict(
        m=abs_blocks.mean(axis=0),
        m_sem=abs_blocks.std(axis=0, ddof=1) / np.sqrt(N_BLOCKS),
        chi=size ** 2 * (sq_blocks.mean(axis=0)
                         - abs_blocks.mean(axis=0) ** 2) / TEMPS,
        chi_err=np.sqrt((N_BLOCKS - 1) * chi_out.var(axis=0)),
        energy=e.mean(axis=0))
snapshots = metropolis(SNAP_L, SNAP_TEMPS, SNAP_SWEEPS, 0)[2]

# ------------------------------------------------------- SELF-CHECK ---
assert abs(T_C - 2.269185) < 1e-6, T_C
assert abs(np.sinh(2 / T_C) - 1) < 1e-12        # self-dual point, sinh = 1
for size, res in results.items():
    assert abs(res["m"][0] / onsager(TEMPS[0]) - 1) < 0.02, (size, res["m"][0])
    assert -2.0 <= res["energy"][0] < -1.9, (size, res["energy"][0])
tails = [results[size]["m"][-1] for size in SIZES]
peaks = [results[size]["chi"].max() for size in SIZES]
assert tails[0] > tails[1] > tails[2], tails    # |m| ~ N^(-1/2) above T_c
assert peaks[0] < peaks[1] < peaks[2], peaks
big = results[SIZES[-1]]
print(f"fig162: self-check passed (T_c = {T_C:.6f}; at T = {TEMPS[0]:.1f}, "
      f"L = {SIZES[-1]}: |m| = {big['m'][0]:.4f} vs Onsager "
      f"{onsager(TEMPS[0]):.4f}, E = {big['energy'][0]:.3f}J per spin; at "
      f"T = {TEMPS[-1]:.1f}: |m| = "
      + ", ".join(f"{v:.3f}" for v in tails) + "; χ peaks "
      + ", ".join(f"{v:.1f}" for v in peaks) + ")")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 84)
PANEL_H, SNAP, GAP = 67.5, 20.0, 3.75          # mm; 3 SNAP + 2 GAP = PANEL_H
ax_a = ms.axes(fig, 13, 10.5, 56, PANEL_H)
ax_b = ms.axes(fig, 85, 10.5, 56, PANEL_H)
ax_c = [ms.axes(fig, 157, 10.5 + (2 - i) * (SNAP + GAP), SNAP, SNAP)
        for i in range(3)]
STYLE = {8: (ms.SKY, "o"), 16: (ms.BLUE, "s"), 32: (ms.VERMILLION, "^")}
T_LIM = (1.4, 3.6)

# a, order parameter; the exact curve is the reference, so it is drawn in ink
t_exact = np.concatenate([np.linspace(T_LIM[0], T_C, 400), [T_LIM[1]]])
ax_a.plot(t_exact, onsager(t_exact), color=ms.INK, lw=1.0,
          label="Onsager, L → ∞")
for size, res in results.items():
    colour, marker = STYLE[size]
    ax_a.errorbar(TEMPS, res["m"], yerr=res["m_sem"], fmt=marker,
                  color=colour, ms=2.6, mew=0, elinewidth=0.6, capsize=0,
                  label=f"L = {size}")
ax_a.set_ylim(0, 1.05)
ax_a.set_ylabel("Magnetisation per spin ⟨|m|⟩")
ax_a.legend(loc="upper right", frameon=False, handlelength=1.6)

# b, susceptibility: thin lines only guide the eye between temperatures
for size, res in results.items():
    colour, marker = STYLE[size]
    ax_b.plot(TEMPS, res["chi"], color=colour, lw=0.5, alpha=0.6)
    ax_b.errorbar(TEMPS, res["chi"], yerr=res["chi_err"], fmt=marker,
                  color=colour, ms=2.6, mew=0, elinewidth=0.6, capsize=0)
# headroom for the tallest error bar, so that it ends inside the frame
chi_top = 1.06 * max((res["chi"] + res["chi_err"]).max()
                     for res in results.values())
ax_b.set_ylim(0, chi_top)
ax_b.set_yticks(np.arange(0, chi_top, 5))
ax_b.text(0.98, 0.97, "Symbols as in a", transform=ax_b.transAxes,
          ha="right", va="top", fontsize=ms.FS_TICK, color=ms.GREY_DARK)
ax_b.set_ylabel("Susceptibility χ (per spin, units of 1/J)")

for ax, top in ((ax_a, 1.05), (ax_b, chi_top)):
    ax.plot([T_C, T_C], [0, top], color=ms.GREY, lw=0.6, ls=(0, (4, 3)),
            zorder=0)
    ax.annotate(f"Exact T$_{{\\mathrm{{c}}}}$ = {T_C:.3f}", xy=(T_C, top),
                xytext=(0, 3), textcoords="offset points", ha="center",
                va="bottom", fontsize=ms.FS_MATH, color=ms.GREY_DARK)
    ax.set_xlim(*T_LIM)
    ax.set_xticks(np.arange(1.5, 3.51, 0.5))
    ax.set_xlabel("Temperature k$_{\\mathrm{B}}$T/J", fontsize=ms.FS_MATH)

# c, one configuration per regime: every spin is a pixel (up = white)
for ax, spins, temp, regime in zip(ax_c, snapshots, SNAP_TEMPS,
                                   ("ordered", "critical", "disordered")):
    ax.imshow(spins, cmap="gray", vmin=-1, vmax=1, interpolation="nearest",
              rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    ax.text(0, 1.03, f"T = {temp:.2f}", transform=ax.transAxes,
            fontsize=ms.FS_SMALL, va="bottom")
    ax.text(1, 1.03, regime, transform=ax.transAxes, fontsize=ms.FS_SMALL,
            color=ms.GREY_DARK, ha="right", va="bottom")
ax_c[2].set_xlabel(f"L = {SNAP_L}, spin up white", fontsize=ms.FS_SMALL)

ms.panel_label(ax_a, "a", dx_pt=-28, dy_pt=8)
ms.panel_label(ax_b, "b", dx_pt=-28, dy_pt=8)
ms.panel_label(ax_c[0], "c", dx_pt=-12, dy_pt=8)
ms.assert_aligned([ax_a, ax_b])
ms.assert_aligned([ax_a, ax_c[0]], edges=("top",))
ms.assert_aligned([ax_a, ax_c[2]], edges=("bottom",))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig162_ising_transition.{ext}")
print("fig162_ising_transition: saved png + pdf")
