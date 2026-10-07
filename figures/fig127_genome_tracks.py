"""Fig. 127 - Genome tracks from simulated reads (double column, 183 mm).

A browser panel is a stack of tracks on one coordinate axis, and its
argument is vertical: a peak in one condition, its absence in the other,
flat input beneath, and the gene models that give the position meaning.
Here the coverage is not drawn but piled up: fragment starts are sampled
from a rate profile (flat background plus planted enriched regions) and
every fragment adds one to each base it covers. The three coverage
tracks share one y scale, printed once per track, peaks are called per
1-kb bin against the input, and the shaded enhancer is bound only in
condition A. The self-check: the integral of each coverage track equals
n_reads x fragment length exactly (base-level and as drawn in 200-bp
bins), and every called peak overlaps a planted enriched region.

Statistics: n = 4,000 fragments of 200 bp per track; coverage, mean
per 200-bp bin; peaks, 1-kb bins with one-sided Poisson P below a
Bonferroni threshold (0.01 / 200 bins) against the input rate, adjacent
bins merged. All data are simulated.
"""

import numpy as np
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from scipy import stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(127)
ORIGIN, WINDOW = 31_400_000, 200_000         # window start and length (bp)
N_READS, FRAGMENT = 4000, 200
DRAW_BIN, CALL_BIN, ALPHA = 200, 1000, 0.01
TRACKS = {"Condition A": ms.VERMILLION, "Condition B": ms.BLUE,
          "Input": ms.GREY}

# ------------------------------------------------------------- DATA ----
# Gene models: name, start, end (bp from the window start), strand, exons
GENES = [("GENE1", 12_000, 61_000, "+", 6), ("GENE2", 82_000, 118_000, "-", 5),
         ("GENE3", 150_000, 189_000, "+", 7)]
exons = {}
for name, start, end, _strand, n_exons in GENES:
    inner = np.sort(rng.choice(np.arange(start + 3000, end - 3000, 1500),
                               n_exons - 2, replace=False))
    left = np.concatenate([[start], inner, [end - 1400]])
    exons[name] = np.column_stack([left, left + rng.integers(500, 1300,
                                                             n_exons)])
    exons[name][-1, 1] = end                 # the last exon closes the gene

# Planted enriched regions: centre (bp), width s.d. (bp) and the share of
# all fragments they attract in conditions A and B. The enhancer is A-only.
PLANTED = {"GENE1 promoter": (12_000, 550, 0.085, 0.080),
           "GENE2 promoter": (118_000, 600, 0.070, 0.075),
           "Enhancer": (135_000, 800, 0.110, 0.000),
           "GENE3 promoter": (150_000, 550, 0.075, 0.070)}
SHARE = {"Condition A": 2, "Condition B": 3}          # column of PLANTED
position = np.arange(WINDOW - FRAGMENT + 1)           # admissible starts


def simulate_starts(track):
    """Fragment starts drawn from the track's rate profile."""
    rate = np.ones(position.size) / position.size      # flat background
    if track in SHARE:
        signal = np.zeros(position.size)
        for row in PLANTED.values():
            centre, sd, share = row[0], row[1], row[SHARE[track]]
            signal += share * stats.norm.pdf(position + FRAGMENT / 2,
                                             centre, sd)
        rate = (1 - signal.sum()) * rate + signal
    return rng.choice(position, N_READS, p=rate / rate.sum())


starts = {track: simulate_starts(track) for track in TRACKS}


# ------------------------------------------------- COVERAGE AND PEAKS --
def pile_up(start):
    """Per-base coverage: +1 at each start, -1 one fragment later."""
    step = np.zeros(WINDOW + 1, dtype=np.int64)
    np.add.at(step, start, 1)
    np.add.at(step, start + FRAGMENT, -1)
    return np.cumsum(step)[:WINDOW]


def call_peaks(start, control):
    """Merged 1-kb bins enriched over the input (Poisson, Bonferroni)."""
    edges = np.arange(0, WINDOW + 1, CALL_BIN)
    count = np.histogram(start + FRAGMENT // 2, edges)[0]
    background = np.histogram(control + FRAGMENT // 2, edges)[0]
    # local input rate over 5 bins, never below the window-wide mean
    local = np.convolve(background, np.ones(5) / 5, mode="same")
    p = stats.poisson.sf(count - 1, np.maximum(local, background.mean()))
    hit = np.concatenate([[0], p < ALPHA / count.size, [0]]).astype(int)
    change = np.diff(hit)                    # +1 opens a peak, -1 closes it
    return np.column_stack([edges[change == 1], edges[change == -1]])


coverage = {track: pile_up(start) for track, start in starts.items()}
binned = {track: c.reshape(-1, DRAW_BIN).mean(axis=1)
          for track, c in coverage.items()}
peaks = {track: call_peaks(starts[track], starts["Input"]) for track in SHARE}
regions = {name: (row[0] - 3 * row[1], row[0] + 3 * row[1])
           for name, row in PLANTED.items()}
y_top = 10 * int(np.ceil(max(b.max() for b in binned.values()) / 10))

# ------------------------------------------------------- SELF-CHECK ---
for track in TRACKS:
    assert coverage[track].sum() == N_READS * FRAGMENT, track
    assert np.isclose(binned[track].sum() * DRAW_BIN, N_READS * FRAGMENT)
for track, column in SHARE.items():
    planted = [regions[name] for name, row in PLANTED.items() if row[column]]
    for lo, hi in peaks[track]:
        assert any(lo < r_hi and hi > r_lo for r_lo, r_hi in planted), (
            track, lo, hi)
    assert len(peaks[track]) == len(planted), (track, peaks[track])
print(f"fig127: self-check passed (each track integrates to "
      f"{N_READS} x {FRAGMENT} = {N_READS * FRAGMENT} bp; peaks called "
      f"A = {len(peaks['Condition A'])}, B = {len(peaks['Condition B'])}, "
      "all on planted regions)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 78)
gs = ms.grid(fig, 5, 1, left=25, right=6, top=9, bottom=11, hspace=2.2,
             height_ratios=[11, 11, 11, 5.5, 9])
axs = [fig.add_subplot(gs[k]) for k in range(5)]
ax_peak, ax_gene = axs[3], axs[4]


def mb(offset):
    """Offset from the window start (bp) -> chromosome position (Mb)."""
    return (ORIGIN + np.asarray(offset)) / 1e6


def name_track(ax, text, colour, y=0.5):
    ax.text(-0.012, y, text, transform=ax.transAxes, color=colour,
            ha="right", va="center")


# the enhancer shading is one patch behind every track, gaps included
box = [ax.get_position() for ax in (axs[0], ax_gene)]
ax_back = fig.add_axes([box[0].x0, box[1].y0, box[0].width,
                        box[0].y1 - box[1].y0], zorder=-1)
ax_back.axvspan(*mb(regions["Enhancer"]), color="#F6EBC8", lw=0)
ax_back.set_xlim(*mb([0, WINDOW]))
ax_back.set_axis_off()
axs[0].text(mb(PLANTED["Enhancer"][0]), 1.06, "Enhancer",
            transform=axs[0].get_xaxis_transform(), ha="center", va="bottom",
            fontsize=ms.FS_TICK)

# coverage: mean per 200-bp bin as a filled step, one y scale for all
edges_mb = mb(np.arange(0, WINDOW + 1, DRAW_BIN))
for ax, (track, colour) in zip(axs, TRACKS.items()):
    ax.fill_between(edges_mb, np.append(binned[track], 0), step="post",
                    color=colour, lw=0)
    ax.set_ylim(0, y_top)
    name_track(ax, track, colour, y=0.62)
    ax.text(-0.012, 0.24, f"[0–{y_top}]", transform=ax.transAxes, ha="right",
            va="center", fontsize=ms.FS_SMALL, color=ms.GREY_DARK)

# peak calls, one row per condition
for row, (track, called) in enumerate(peaks.items()):
    ax_peak.broken_barh([(mb(lo), (hi - lo) / 1e6) for lo, hi in called],
                        (row - 0.32, 0.64), color=TRACKS[track], lw=0)
ax_peak.set_ylim(1.5, -0.5)
name_track(ax_peak, "Peaks", ms.INK)

# gene models: intron line with strand chevrons under the exon boxes
forward = Path([(-0.5, -1.0), (0.5, 0.0), (-0.5, 1.0)])
CHEVRON = {"+": forward, "-": forward.transformed(Affine2D().scale(-1, 1))}
for name, start, end, strand, _n in GENES:
    ax_gene.plot(mb([start, end]), [0, 0], color=ms.INK, lw=0.6)
    marks = np.arange(start + 2000, end, 3000)
    in_exon = np.any((marks[:, None] > exons[name][:, 0] - 1200)
                     & (marks[:, None] < exons[name][:, 1] + 1200), axis=1)
    ax_gene.plot(mb(marks[~in_exon]), np.zeros((~in_exon).sum()), ls="",
                 marker=CHEVRON[strand], markersize=2.6,
                 markerfacecolor="none", markeredgecolor=ms.INK,
                 markeredgewidth=0.5)
    ax_gene.broken_barh([(mb(a), (b - a) / 1e6) for a, b in exons[name]],
                        (-0.3, 0.6), color=ms.INK, lw=0, zorder=3)
    label = f"{name} →" if strand == "+" else f"← {name}"
    ax_gene.text(mb((start + end) / 2), -0.62, label, ha="center", va="top",
                 fontsize=ms.FS_TICK, fontstyle="italic")
ax_gene.set_ylim(-1.75, 0.6)
name_track(ax_gene, "Genes", ms.INK)

for ax in axs:
    ax.set_xlim(*mb([0, WINDOW]))
    ax.set_yticks([])
    ax.patch.set_alpha(0)                 # let the shared shading through
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(ax is ax_gene and side == "bottom")
    if ax is not ax_gene:
        ax.tick_params(bottom=False, labelbottom=False)
ax_gene.spines["bottom"].set_position(("outward", 3))
ax_gene.set_xticks(mb(np.arange(0, WINDOW + 1, 25_000)))
ax_gene.xaxis.set_major_formatter("{x:.3f}")
ax_gene.set_xlabel("Position on chromosome (Mb)")

# scale bar: 20 kb in data units, above the right end of the first track
bar = mb([WINDOW - 20_000, WINDOW])
axs[0].plot(bar, [1.12, 1.12], transform=axs[0].get_xaxis_transform(),
            color=ms.INK, lw=1.0, solid_capstyle="butt", clip_on=False)
axs[0].text(bar[0] - 0.001, 1.12, "20 kb",
            transform=axs[0].get_xaxis_transform(), ha="right", va="center",
            fontsize=ms.FS_TICK)

ms.assert_aligned(axs + [ax_back], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig127_genome_tracks.{ext}")
print("fig127_genome_tracks: saved png + pdf")
