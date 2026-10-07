"""Fig. 128 - Sequence logo with exact letter geometry (single column, 89 mm).

A logo is a stacked bar chart whose bars are letters: the stack at each
position is its information content in bits, split among the bases in
proportion to their frequency, the most frequent on top. Setting a
letter by font size cannot hit a height, so each glyph is a TextPath
outline stretched to its exact box. The self-check: every stack equals
2 - H - e(n) from the entropy of the counts with the Schneider
small-sample correction, the drawn glyph heights (read back from the
patches) sum to that stack, no stack exceeds 2 bits, and the top letters
spell the planted motif at the conserved positions.

Statistics: n = 200 aligned sites of 12 positions; stack height,
information content R = 2 - H - e(n) bits with e(n) = 3 / (2 ln 2 n);
letter height, frequency x R; no test. All data are simulated.
"""

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(128)
BASES = "ACGT"
COLOUR = {"A": ms.GREEN, "C": ms.BLUE, "G": ms.ORANGE, "T": ms.VERMILLION}
N_SITES = 200

# ------------------------------------------------------------- DATA ----
# Planted motif: the preferred base at each position and how strongly it
# is preferred (the other three bases share the rest unevenly).
MOTIF = "TGACCGATTCAG"
PREFERENCE = [0.50, 0.72, 0.94, 0.97, 0.90, 0.55, 0.42, 0.95, 0.98, 0.85,
              0.60, 0.45]
CONSERVED = np.array(PREFERENCE) >= 0.85
sites = np.empty((N_SITES, len(MOTIF)), dtype=int)   # aligned sites, 0-3
for j, (base, p_top) in enumerate(zip(MOTIF, PREFERENCE)):
    p = np.empty(4)
    rest = rng.dirichlet([2.0, 2.0, 2.0]) * (1 - p_top)
    p[BASES.index(base)] = p_top
    p[np.arange(4) != BASES.index(base)] = rest
    sites[:, j] = rng.choice(4, N_SITES, p=p)

# ------------------------------------------------ INFORMATION CONTENT --
counts = np.stack([(sites == b).sum(axis=0) for b in range(4)])   # (4, L)
freq = counts / N_SITES
with np.errstate(divide="ignore", invalid="ignore"):
    entropy = -np.nansum(freq * np.log2(freq), axis=0)    # 0 log 0 = 0
small_sample = (4 - 1) / (2 * np.log(2) * N_SITES)        # Schneider e(n)
info = 2 - entropy - small_sample                          # bits
height = freq * info                                       # (4, L)

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 52)
ax = fig.add_subplot(ms.grid(fig, 1, 1, left=13, right=4, top=7,
                             bottom=11)[0])
FONT = FontProperties(family="sans-serif", weight="bold")
LETTER_WIDTH = 0.86                                        # of one position


def draw_letter(base, x_centre, y0, tall):
    """Stretch the glyph outline to fill its box exactly; return the patch."""
    outline = TextPath((0, 0), base, size=1, prop=FONT)
    box = outline.get_extents()
    fit = (Affine2D().translate(-box.x0, -box.y0)
           .scale(LETTER_WIDTH / box.width, tall / box.height)
           .translate(x_centre - LETTER_WIDTH / 2, y0))
    return ax.add_patch(PathPatch(fit.transform_path(outline),
                                  facecolor=COLOUR[base], lw=0))


drawn = np.zeros(len(MOTIF))         # stack heights read back from patches
consensus = ""
for j in range(len(MOTIF)):
    y = 0.0
    for b in np.argsort(height[:, j]):                     # tallest on top
        if height[b, j] > 0:
            patch = draw_letter(BASES[b], j + 1, y, height[b, j])
            extent = patch.get_path().get_extents()
            assert np.isclose(extent.y0, y) and np.isclose(
                extent.width, LETTER_WIDTH)
            drawn[j] += extent.height
            y += height[b, j]
    consensus += BASES[b]                                  # the top letter

ax.set_xlim(0.45, len(MOTIF) + 0.55)
ax.set_ylim(0, 2)
ax.set_xticks(range(1, len(MOTIF) + 1))
ax.set_yticks([0, 0.5, 1, 1.5, 2], ["0", "0.5", "1.0", "1.5", "2.0"])
ax.tick_params(axis="x", length=0, pad=3)
ax.spines["bottom"].set_visible(False)
ax.set_xlabel("Position in motif")
ax.set_ylabel("Information (bits)")
ax.text(1.0, 1.03, f"n = {N_SITES} sites", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=ms.FS_TICK, color=ms.GREY_DARK)

# ------------------------------------------------------- SELF-CHECK ---
# (after drawing, because the heights are read back from the glyph patches)
assert np.allclose(counts.sum(axis=0), N_SITES)
assert np.allclose(drawn, 2 - entropy - small_sample, atol=1e-12)
assert np.allclose(height.sum(axis=0), info) and np.all(info > 0)
assert info.max() <= 2 - small_sample < 2
assert all(consensus[j] == MOTIF[j] for j in np.flatnonzero(CONSERVED))
print(f"fig128: self-check passed (stacks = 2 - H - e(n), e(n) = "
      f"{small_sample:.4f} bits; glyph heights sum to each stack; tallest "
      f"{info.max():.2f} bits <= 2; consensus {consensus} matches the "
      f"planted motif at all {CONSERVED.sum()} conserved positions)")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig128_sequence_logo.{ext}")
print("fig128_sequence_logo: saved png + pdf")
