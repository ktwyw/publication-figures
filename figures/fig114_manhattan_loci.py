"""Fig. 114 - Manhattan plot at realistic density (double column, 183 mm).

Extends fig035: about 90,000 variants spread over the real autosome
length proportions and drawn as one rasterised layer under vector text,
the two thresholds named in the right margin instead of a legend, each
locus labelled above its lead variant, and the x-axis spine bounded to
the genome. The self-check is that the null is calibrated (genomic
inflation factor, the median chi-square of variants outside the locus
windows over 0.4549, within 0.95-1.05), that every genome-wide
significant variant lies within ±5 Mb of a designated locus, and that
exactly five chromosomes carry one.

Statistics: n = 90,002 variants (90,000 shared out by chromosome length
and rounded), all drawn; dashed line, genome-wide significance
(P = 5e-8); dotted line, suggestive (P = 1e-5). All data are simulated
and the locus names are placeholders.
"""

import numpy as np
from scipy import stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(1414)
N_TARGET = 90000                             # before per-chromosome rounding
CHROM_MB = np.array([248, 242, 198, 190, 182, 171, 159, 145, 138, 134, 135,
                     133, 114, 107, 102, 90, 83, 80, 59, 64, 47, 51])
# (name, chromosome, lead position in Mb, peak -log10 P)
LOCI = [("Locus 1", 2, 121.0, 14.5), ("Locus 2", 6, 32.5, 23.0),
        ("Locus 3", 9, 22.0, 10.8), ("Locus 4", 11, 68.0, 17.2),
        ("Locus 5", 16, 53.8, 12.4)]
GENOME_WIDE, SUGGESTIVE = 5e-8, 1e-5
WINDOW_MB = 5.0                              # half-width of a locus window

# ------------------------------------------------------------- DATA ----
# Null: P ~ Uniform(0, 1), so -log10 P is exponential with rate ln 10.
# Each locus adds a Gaussian-shaped tower (s.d. 0.9 Mb) of linked variants.
counts = np.round(CHROM_MB / CHROM_MB.sum() * N_TARGET).astype(int)
chrom, position, neg_log_p = [], [], []
for c, (length, count) in enumerate(zip(CHROM_MB, counts), start=1):
    pos = np.sort(rng.uniform(0, length, count))
    nlp = rng.exponential(1 / np.log(10), count)
    for _name, locus_chrom, centre, peak in LOCI:
        if locus_chrom == c:
            decay = np.exp(-0.5 * ((pos - centre) / 0.9) ** 2)
            nlp = np.maximum(nlp, peak * decay * rng.uniform(0.45, 1.0, count))
            nlp[np.argmin(np.abs(pos - centre))] = peak
    chrom.append(np.full(count, c))
    position.append(pos)
    neg_log_p.append(nlp)
chrom, position, neg_log_p = map(np.concatenate, (chrom, position, neg_log_p))
p_value = 10.0 ** (-neg_log_p)

offsets = np.concatenate([[0], np.cumsum(CHROM_MB)[:-1]])
genome_mb = position + offsets[chrom - 1]            # cumulative coordinate
genome_end = float(CHROM_MB.sum())
threshold = -np.log10(GENOME_WIDE)
significant = neg_log_p >= threshold

# ------------------------------------------------------- SELF-CHECK ---
in_window = np.zeros(chrom.size, bool)
for _name, locus_chrom, centre, _peak in LOCI:
    in_window |= (chrom == locus_chrom) & (np.abs(position - centre)
                                           <= WINDOW_MB)
chi2_null = stats.chi2.isf(p_value[~in_window], df=1)
inflation = float(np.median(chi2_null) / stats.chi2.ppf(0.5, df=1))
hit_chroms = np.unique(chrom[significant])
assert np.all(p_value > 0) and abs(chrom.size - N_TARGET) <= len(CHROM_MB)
assert 0.95 < inflation < 1.05, inflation            # the null is calibrated
assert np.all(in_window[significant])                # no stray hits
assert hit_chroms.tolist() == sorted(c for _n, c, _p, _k in LOCI)
print(f"fig114: self-check passed ({chrom.size:,} variants; genomic "
      f"inflation = {inflation:.3f}; {int(significant.sum())} genome-wide "
      f"significant, all within ±{WINDOW_MB:g} Mb of a locus, on "
      f"chromosomes {', '.join(map(str, hit_chroms))})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 62)
ax = fig.add_subplot(ms.grid(fig, 1, 1, left=12, right=24, top=6,
                             bottom=11)[0])

colours = np.where(chrom % 2 == 1, ms.BLUE, ms.SKY)
ax.scatter(genome_mb[~significant], neg_log_p[~significant], s=1.5,
           c=colours[~significant], linewidths=0, rasterized=True, zorder=2)
ax.scatter(genome_mb[significant], neg_log_p[significant], s=3.5,
           color=ms.VERMILLION, linewidths=0, rasterized=True, zorder=3)
ax.plot([0, genome_end], [threshold] * 2, color=ms.GREY_DARK, lw=0.6,
        ls=(0, (4, 2)), zorder=1)
ax.plot([0, genome_end], [-np.log10(SUGGESTIVE)] * 2, color=ms.GREY, lw=0.6,
        ls=(0, (1, 2)), zorder=1)

# thresholds named in the right margin, at the height of their lines
margin = ax.get_yaxis_transform()
ax.text(1.012, threshold, "Genome-wide\nsignificance", transform=margin,
        va="center", ha="left", fontsize=ms.FS_SMALL, color=ms.GREY_DARK)
ax.text(1.012, -np.log10(SUGGESTIVE) - 0.3, "Suggestive", transform=margin,
        va="center", ha="left", fontsize=ms.FS_SMALL, color=ms.GREY)
for name, locus_chrom, centre, peak in LOCI:
    ax.annotate(name, xy=(offsets[locus_chrom - 1] + centre, peak),
                xytext=(0, 3), textcoords="offset points", ha="center",
                va="bottom", fontsize=ms.FS_TICK, fontstyle="italic")

centres = offsets + CHROM_MB / 2
shown = [c for c in range(1, 23) if c <= 16 or c % 2 == 0]
ax.set_xticks([centres[c - 1] for c in shown], [str(c) for c in shown])
ax.tick_params(axis="x", length=0, pad=3)
ax.set_xlim(-15, genome_end + 15)
ax.set_ylim(0, 27)
ax.set_yticks(np.arange(0, 26, 5))
ax.set_xlabel("Chromosome", fontsize=ms.FS_MATH)
ax.set_ylabel("−log$_{10}$ P", fontsize=ms.FS_MATH)
ax.spines["bottom"].set_bounds(0, genome_end)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig114_manhattan_loci.{ext}")
print("fig114_manhattan_loci: saved png + pdf")
