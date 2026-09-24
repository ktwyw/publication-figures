"""Fig. 35 - Manhattan plot for genome-wide association results (double column).

Cumulative genomic coordinate with alternating chromosome colours,
genome-wide (5e-8) and suggestive (1e-5) thresholds, rasterised points
so the PDF stays small with 10^5-10^6 SNPs, and labelled lead loci.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(35)

# ------------------------------------------------------------- DATA ----
chr_len = np.array([248, 242, 198, 190, 181, 171, 159, 145, 138, 134, 135,
                    133, 114, 107, 102, 90, 83, 80, 59, 64, 47, 51], float)  # Mb
offsets = np.concatenate([[0], np.cumsum(chr_len)[:-1]])
centers = offsets + chr_len / 2

pos, chrom, neglogp = [], [], []
for c, (length, off) in enumerate(zip(chr_len, offsets), start=1):
    n = int(length * 25)
    pos.append(off + rng.uniform(0, length, n))
    chrom.append(np.full(n, c))
    neglogp.append(-np.log10(rng.uniform(size=n)))

# three associated loci: (chromosome, position Mb, gene, peak -log10 P)
loci = [(1, 55.5, "PCSK9", 12), (19, 11.1, "LDLR", 9.5), (19, 45.4, "APOE", 21)]
for c, mb, _, peak in loci:
    d = rng.normal(0, 0.25, 40)
    pos.append(offsets[c - 1] + mb + d)
    chrom.append(np.full(40, c))
    neglogp.append(peak * np.exp(-d**2 / (2 * 0.25**2)) * rng.uniform(0.6, 1, 40))
pos, chrom, neglogp = map(np.concatenate, (pos, chrom, neglogp))

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(7.1, 2.3))
colors = np.where(chrom % 2 == 1, "#1f4e79", "#7fa7cc")
ax.scatter(pos, neglogp, s=3, c=colors, lw=0, rasterized=True)

ax.axhline(-np.log10(5e-8), color="C1", ls="--", lw=0.8)
ax.axhline(-np.log10(1e-5), color="0.55", ls=":", lw=0.7)
for c, mb, gene, peak in loci:
    xg = offsets[c - 1] + mb
    ax.annotate(gene, (xg, peak), xytext=(0, 5), textcoords="offset points",
                ha="left" if xg < 0.03 * chr_len.sum() else "center",
                fontsize=6, style="italic")

ax.set_xticks(centers, [str(c) for c in range(1, 23)], fontsize=5.5)
ax.set_xlim(0, chr_len.sum())
ax.set_ylim(0, neglogp.max() + 3)
ax.set_xlabel("Chromosome")
ax.set_ylabel(r"$-\log_{10}(P)$")
ax.tick_params(axis="x", length=0)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig035_manhattan.{ext}")
print("saved fig035_manhattan.png / .pdf")
