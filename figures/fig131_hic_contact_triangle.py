"""Fig. 131 - Hi-C triangle over its insulation score (1.5 column, 120 mm).

A contact matrix is symmetric, so half of a square heatmap is wasted
ink and the diagonal, which carries the genome, runs at 45 degrees to
every other track. Rotating the upper triangle puts the diagonal on the
x axis: domains become triangles standing on the genome and a 1D track
can sit directly underneath. The matrix is simulated as power-law
distance decay times an enrichment inside five planted domains, with
Poisson counting noise; the insulation score is the mean contact
frequency in a square window sliding along the diagonal, and its minima
are the boundary calls. The self-check: the matrix is symmetric, the
rotated axes are exactly equal-aspect (so the triangle really is 45
degrees), and every planted boundary has a detected minimum within two
bins, with no other minimum called.

Statistics: 200 bins of 25 kb, one simulated map; colour, contacts per
bin pair (log scale); insulation, log₂ of the mean contacts in a
10 x 10-bin window across each bin edge over its chromosome-wide mean;
minima by prominence >= 0.3. All data are simulated.
"""

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy import signal

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(131)
N_BINS, BIN_MB = 200, 0.025
MAX_DISTANCE_MB = 3.0                        # top of the drawn triangle
WINDOW, PROMINENCE, TOLERANCE = 10, 0.3, 2   # insulation window (bins)

# ------------------------------------------------------------- DATA ----
# Expected contacts: A (d + 1)^-1 between bins d apart, times ENRICHMENT
# when both bins lie in the same planted domain; observed = Poisson.
BOUNDARIES = np.array([36, 79, 114, 163])    # first bin of each new domain
AMPLITUDE, DECAY, ENRICHMENT = 420.0, 1.0, 2.6
i, j = np.indices((N_BINS, N_BINS))
domain = np.searchsorted(BOUNDARIES, np.arange(N_BINS), side="right")
expected = (AMPLITUDE * (np.abs(i - j) + 1.0) ** -DECAY
            * np.where(domain[i] == domain[j], ENRICHMENT, 1.0))
upper = np.triu(rng.poisson(expected))
contacts = upper + np.triu(upper, 1).T       # each pair is counted once

# ------------------------------------------------------- INSULATION ---
# edge k separates bins k - 1 and k; the window is the square of contacts
# between the WINDOW bins on its left and the WINDOW bins on its right
edges = np.arange(WINDOW, N_BINS - WINDOW + 1)
window_mean = np.array([contacts[k - WINDOW:k, k:k + WINDOW].mean()
                        for k in edges])
insulation = np.log2(window_mean / window_mean.mean())
minima = edges[signal.find_peaks(-insulation, prominence=PROMINENCE)[0]]

# ------------------------------------------------------- SELF-CHECK ---
assert np.array_equal(contacts, contacts.T)
nearest = np.abs(minima[:, None] - BOUNDARIES[None, :]).min(axis=0)
assert np.all(nearest <= TOLERANCE), nearest
assert minima.size == BOUNDARIES.size, minima
print(f"fig131: self-check passed (matrix symmetric, {N_BINS} x {N_BINS}; "
      f"insulation minima at bins {minima.tolist()} for planted boundaries "
      f"{BOUNDARIES.tolist()}, worst offset {nearest.max()} bin)")

# ------------------------------------------------------------ FIGURE --
# The map height follows from its width so that one Mb is the same length
# on both axes; y is half the genomic distance after the 45° rotation.
REGION_MB = N_BINS * BIN_MB
LEFT, BOTTOM, AX_W, TRACK_H, GAP = 17.0, 11.0, 96.0, 13.0, 4.0
MAP_H = AX_W * (MAX_DISTANCE_MB / 2) / REGION_MB
fig = ms.figure(120, BOTTOM + TRACK_H + GAP + MAP_H + 6.0)
ax_ins = ms.axes(fig, LEFT, BOTTOM, AX_W, TRACK_H)
ax = ms.axes(fig, LEFT, BOTTOM + TRACK_H + GAP, AX_W, MAP_H, sharex=ax_ins)

# a, cell (i, j) spans x = (i + j) / 2, y = (j - i) / 2 at its corners
vi, vj = np.indices((N_BINS + 1, N_BINS + 1)) * BIN_MB
mesh = ax.pcolormesh((vi + vj) / 2, (vj - vi) / 2,
                     np.ma.masked_where((i > j) | (contacts == 0), contacts),
                     cmap="Reds", norm=LogNorm(2, AMPLITUDE * ENRICHMENT),
                     shading="flat", rasterized=True)
ax.set_xlim(0, REGION_MB)
ax.set_ylim(0, MAX_DISTANCE_MB / 2)
ax.set_yticks(np.arange(0, MAX_DISTANCE_MB + 0.1, 1.0) / 2,
              [f"{d:g}" for d in np.arange(0, MAX_DISTANCE_MB + 0.1, 1.0)])
ax.set_ylabel("Distance (Mb)")
ax.spines["bottom"].set_visible(False)
ax.tick_params(bottom=False, labelbottom=False)
x_scale = ax.get_window_extent().width / REGION_MB
y_scale = ax.get_window_extent().height / (MAX_DISTANCE_MB / 2)
assert np.isclose(x_scale, y_scale), (x_scale, y_scale)    # true 45°

# compact colour bar in the empty corner left of the triangle
ax_cbar = ms.axes(fig, LEFT + 1.5, BOTTOM + TRACK_H + GAP + MAP_H - 6.0,
                  13.0, 1.6)
cbar = fig.colorbar(mesh, cax=ax_cbar, orientation="horizontal",
                    ticks=[3, 30, 300])
cbar.ax.xaxis.set_major_formatter("{x:g}")
cbar.ax.minorticks_off()
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(length=2, width=0.5, labelsize=ms.FS_SMALL)
cbar.ax.set_title("Contacts", fontsize=ms.FS_TICK, pad=2.5, loc="left")

# b, insulation score; planted boundaries as ticks on the shared genome
ax_ins.axhline(0, color=ms.GREY_LIGHT, lw=0.5)
ax_ins.plot(edges * BIN_MB, insulation, color=ms.INK, lw=0.9)
found = np.isin(edges, minima)
ax_ins.plot(edges[found] * BIN_MB, insulation[found], ls="", marker="o",
            markersize=3.6, markerfacecolor="white",
            markeredgecolor=ms.VERMILLION, markeredgewidth=0.9, zorder=3)
top = ax_ins.get_xaxis_transform()
ax_ins.vlines(BOUNDARIES * BIN_MB, 1.0, 1.0 + 0.6 * GAP / TRACK_H,
              transform=top, color=ms.BLUE, lw=1.1, clip_on=False)
ax_ins.set_ylim(-1.5, 0.75)
ax_ins.set_yticks([-1, 0])
ax_ins.yaxis.set_major_formatter(lambda v, _pos: f"{v:g}".replace("-", "−"))
ax_ins.set_xticks(np.arange(0, REGION_MB + 0.1, 1.0))
ax_ins.set_xlabel("Genomic position (Mb)")
ax_ins.set_ylabel("Insulation\n(log₂)")

fig.legend(handles=[
    Line2D([], [], ls="", marker="|", markersize=5, markeredgewidth=1.1,
           color=ms.BLUE, label="Planted boundary"),
    Line2D([], [], ls="", marker="o", markersize=3.6, markerfacecolor="white",
           markeredgecolor=ms.VERMILLION, markeredgewidth=0.9,
           label="Insulation minimum")],
    loc="upper right", bbox_to_anchor=(ax.get_position().x1,
                                       ax.get_position().y1),
    borderaxespad=0, handletextpad=0.3)

ms.panel_label(ax, "a", dx_pt=-34)
ms.panel_label(ax_ins, "b", dx_pt=-34, dy_pt=1)
ms.assert_aligned([ax, ax_ins], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig131_hic_contact_triangle.{ext}")
print("fig131_hic_contact_triangle: saved png + pdf")
