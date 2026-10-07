"""Fig. 164 - Wigner function of a Schrödinger cat state (1.5 column, 120 mm).

A phase-space map that takes both signs needs a diverging colour scale
centred on zero: the Wigner function of a superposition of two coherent
states shows two positive blobs and, between them, interference fringes
that go negative, the signature of a non-classical state. Marginal axes
show that integrating W over p or x returns the position and momentum
probability densities. The self-check is that W is real and integrates
to 1 within 1e-3, that both marginals match |ψ(x)|² and |φ(p)|²
computed directly (within 1e-3 of the peak), that W(0, 0) = 1/π (the
parity of the even cat), and that the minimum of W is negative.

Model: ψ(x) ∝ exp[−(x − x₀)²/2] + exp[−(x + x₀)²/2], x₀ = √2·α, α = 2
(ħ = m = ω = 1); W(x, p) = (1/π) ∫ ψ*(x + y) ψ(x − y) exp(2ipy) dy by
direct quadrature (step 0.04) on a 241 × 141 grid; φ(p) by quadrature
of the Fourier integral. No data: every curve is computed from the
equations.
"""

import numpy as np
from matplotlib.lines import Line2D

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
ALPHA = 2.0                                   # coherent amplitude (real)
X0 = np.sqrt(2) * ALPHA                       # ⟨x⟩ of |α⟩ for ħ = m = ω = 1
X_MAX, P_MAX, STEP = 6.0, 3.5, 0.05           # phase-space grid
Y_MAX, Y_STEP = 8.0, 0.04                     # quadrature variable


def psi(x):
    """Even cat state (|α⟩ + |−α⟩)/norm in the position representation."""
    lobes = np.exp(-(x - X0) ** 2 / 2) + np.exp(-(x + X0) ** 2 / 2)
    norm2 = 2 * np.sqrt(np.pi) * (1 + np.exp(-X0 ** 2))   # ∫ lobes² dx
    return lobes / np.sqrt(norm2)


# ------------------------------------------------------------ SOLVER ---
x = np.linspace(-X_MAX, X_MAX, int(round(2 * X_MAX / STEP)) + 1)
p = np.linspace(-P_MAX, P_MAX, int(round(2 * P_MAX / STEP)) + 1)
y = np.linspace(-Y_MAX, Y_MAX, int(round(2 * Y_MAX / Y_STEP)) + 1)

# W(x, p) = (1/π) ∫ ψ*(x + y) ψ(x − y) exp(2ipy) dy: rows x, columns p
overlap = np.conj(psi(x[:, None] + y)) * psi(x[:, None] - y)
wigner_complex = overlap @ np.exp(2j * np.outer(y, p)) * Y_STEP / np.pi
wigner = wigner_complex.real

from_w_x = wigner.sum(axis=1) * STEP          # ∫ W dp
from_w_p = wigner.sum(axis=0) * STEP          # ∫ W dx
direct_x = np.abs(psi(x)) ** 2
x_fine = np.linspace(-12, 12, 4801)           # independent of the W grid
phi = (psi(x_fine) * np.exp(-1j * np.outer(p, x_fine))).sum(axis=1) \
    * (x_fine[1] - x_fine[0]) / np.sqrt(2 * np.pi)
direct_p = np.abs(phi) ** 2

# ------------------------------------------------------- SELF-CHECK ---
imaginary = np.abs(wigner_complex.imag).max()
total = wigner.sum() * STEP ** 2
error_x = np.abs(from_w_x - direct_x).max() / direct_x.max()
error_p = np.abs(from_w_p - direct_p).max() / direct_p.max()
negative_volume = (np.abs(wigner).sum() * STEP ** 2 - total) / 2
assert imaginary < 1e-12, imaginary           # W is real
assert abs(total - 1) < 1e-3, total
assert error_x < 1e-3 and error_p < 1e-3, (error_x, error_p)
assert wigner.min() < -0.1                    # fringes go negative
ix, ip = np.unravel_index(np.argmax(wigner[x > 1]), wigner[x > 1].shape)
assert abs(x[x > 1][ix] - X0) <= STEP and abs(p[ip]) <= STEP   # blob centre
# W(0, 0) = ⟨parity⟩/π, and the even cat state has parity +1
origin = wigner[x.size // 2, p.size // 2]
assert abs(origin - 1 / np.pi) < 1e-6, origin
print(f"fig164: self-check passed (∫W = {total:.6f}; max |Im W| = "
      f"{imaginary:.1e}; marginal errors {error_x:.1e} (x), {error_p:.1e} "
      f"(p) of peak; W(0, 0)·π = {origin * np.pi:.6f}; min W = "
      f"{wigner.min():.4f}; negative volume "
      f"{negative_volume:.4f})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 80)
X_MM, Y_MM, W_MM, H_MM, SIDE = 14.0, 11.0, 68.0, 49.0, 13.0
ax = ms.axes(fig, X_MM, Y_MM, W_MM, H_MM)
ax_top = ms.axes(fig, X_MM, Y_MM + H_MM + 2.5, W_MM, SIDE, sharex=ax)
ax_right = ms.axes(fig, X_MM + W_MM + 2.5, Y_MM, SIDE, H_MM, sharey=ax)
ax_cbar = ms.axes(fig, X_MM + W_MM + SIDE + 7.5, Y_MM, 2.6, H_MM)

# symmetric limits put zero on the white centre of the diverging map
limit = np.abs(wigner).max()
image = ax.imshow(wigner.T, origin="lower", aspect="auto", cmap="RdBu_r",
                  vmin=-limit, vmax=limit, interpolation="nearest",
                  rasterized=True,
                  extent=[-X_MAX - STEP / 2, X_MAX + STEP / 2,
                          -P_MAX - STEP / 2, P_MAX + STEP / 2])
ax.annotate("Interference fringes\n(W below zero: non-classical)",
            xy=(0.0, 1.75), xytext=(0.0, 2.45), ha="center", va="bottom",
            fontsize=ms.FS_TICK, linespacing=1.15,
            arrowprops=dict(arrowstyle="-", color=ms.INK, lw=0.5, shrinkA=1.5,
                            shrinkB=0))
for sign, name in ((-1, "|−α⟩"), (1, "|α⟩")):
    ax.text(sign * X0, -1.75, name, ha="center", va="top")
ax.set_xlim(-X_MAX, X_MAX)
ax.set_ylim(-P_MAX, P_MAX)
ax.set_xticks(np.arange(-4, 4.1, 2))
ax.set_yticks(np.arange(-3, 3.1, 1))
ax.set_xlabel("Position x")
ax.set_ylabel("Momentum p")
for side in ("top", "right"):
    ax.spines[side].set_visible(True)

# marginals: solid from W, dashed computed directly from the wavefunction
ax_top.plot(x, from_w_x, color=ms.INK, lw=1.2)
ax_top.plot(x, direct_x, color=ms.ORANGE, lw=0.9, ls=(0, (2.5, 2)))
ax_top.set_ylim(0, 1.15 * direct_x.max())
ax_top.set_yticks([0, 0.2])
ax_top.set_ylabel("P(x)")
ax_top.tick_params(labelbottom=False)
ax_right.plot(from_w_p, p, color=ms.INK, lw=1.2)
ax_right.plot(direct_p, p, color=ms.ORANGE, lw=0.9, ls=(0, (2.5, 2)))
ax_right.set_xlim(0, 1.15 * direct_p.max())
ax_right.set_xticks([0, 1])
ax_right.set_xlabel("P(p)")
ax_right.tick_params(labelleft=False)
fig.legend(handles=[Line2D([], [], color=ms.INK, lw=1.2, label="∫ W"),
                    Line2D([], [], color=ms.ORANGE, lw=0.9,
                           ls=(0, (2.5, 2)), label="Direct")],
           loc="center", frameon=False, fontsize=ms.FS_TICK,
           bbox_to_anchor=((X_MM + W_MM + 2.5 + SIDE / 2 + 1) / 120,
                           (Y_MM + H_MM + 2.5 + SIDE / 2) / 80))

cbar = fig.colorbar(image, cax=ax_cbar, ticks=np.arange(-0.3, 0.31, 0.1))
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(length=2, width=0.5)
cbar.set_label("Wigner function W(x, p)")

ms.assert_aligned([ax, ax_top], edges=("left", "right"))
ms.assert_aligned([ax, ax_right, ax_cbar])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig164_wigner_cat_state.{ext}")
print("fig164_wigner_cat_state: saved png + pdf")
