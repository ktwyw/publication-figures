"""Fig. 103 - Volcano plot with a coded BH adjustment (single column, 89 mm).

Extends fig008, which thresholds raw P values: here each gene has its
own standard error, the two-sided P values are adjusted by a
Benjamini-Hochberg step written out in the script, and the y axis and
thresholds use the adjusted values. All 6,000 genes are drawn as a
rasterised point layer under vector axes and text, and a header row
gives the count in each class. Eight hand-placed placeholder hits
(GENE-A to GENE-H) are appended after the adjustment so the labels have
fixed positions. The self-check runs on the 5,992 simulated genes only:
the coded adjustment must equal scipy.stats.false_discovery_control to
1e-12, never fall below the raw P value, and reject exactly the genes
the BH step-up rule rejects; the three classes must add up to the
number of genes.

Statistics: 6,000 genes; two-sided z-test per gene; thresholds
|log2 fold change| >= 1 and Benjamini-Hochberg adjusted P < 0.05.
All data are simulated and the gene names are placeholders.
"""

import numpy as np
from scipy import stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

N_GENES = 6000
LFC_CUT, PADJ_CUT = 1.0, 0.05
X_LIM = 5.6
# placeholder top hits: (name, log2 fold change, -log10 adjusted P)
HITS = [("GENE-A", 3.6, 27.5), ("GENE-B", 2.9, 21.0), ("GENE-C", 4.3, 15.5),
        ("GENE-D", 2.2, 12.5), ("GENE-E", -3.3, 24.0), ("GENE-F", -2.5, 18.0),
        ("GENE-G", -4.1, 13.5), ("GENE-H", -1.9, 10.5)]


def benjamini_hochberg(p):
    """BH adjusted P: p(i) n / i, made monotone from the largest P down."""
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted = np.empty(n)
    adjusted[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1],
                                 1.0)
    return adjusted


# ------------------------------------------------------------- DATA ----
# 7% of genes respond; the standard error grows with the effect size.
rng = np.random.default_rng(303)
n_sim = N_GENES - len(HITS)
responsive = rng.uniform(size=n_sim) < 0.07
effect = responsive * rng.normal(0, 1.25, n_sim)     # true log2 fold change
se = (0.30 * rng.lognormal(0.0, 0.25, n_sim)
      * (1 + 0.45 * np.abs(effect)))                 # per-gene s.e.
lfc_sim = effect + rng.normal(0, se)
p_sim = 2 * stats.norm.sf(np.abs(lfc_sim) / se)
padj_sim = benjamini_hochberg(p_sim)

lfc = np.concatenate([lfc_sim, [h[1] for h in HITS]])
padj = np.concatenate([padj_sim, [10.0 ** -h[2] for h in HITS]])
assert np.all(padj > 0), "adjusted P must be positive before the log"
neg_log = -np.log10(padj)
up = (lfc >= LFC_CUT) & (padj < PADJ_CUT)
down = (lfc <= -LFC_CUT) & (padj < PADJ_CUT)
ns = ~(up | down)

# ------------------------------------------------------- SELF-CHECK ---
# simulated genes only: the placeholder hits never went through BH
reference = stats.false_discovery_control(p_sim, method="bh")
max_dev = float(np.max(np.abs(padj_sim - reference)))
assert max_dev < 1e-12, max_dev
assert np.all(padj_sim >= p_sim) and np.all(padj_sim <= 1.0)
# step-up rule: reject the k smallest P, k the largest i with
# p(i) <= i alpha / n; the adjusted values must give the same k
p_sorted = np.sort(p_sim)
passing = np.nonzero(p_sorted
                     <= np.arange(1, n_sim + 1) * PADJ_CUT / n_sim)[0]
k_step_up = int(passing[-1]) + 1 if passing.size else 0
assert int((padj_sim <= PADJ_CUT).sum()) == k_step_up
assert not np.any(up & down)
assert int(up.sum()) + int(down.sum()) + int(ns.sum()) == N_GENES
print(f"fig103: self-check passed (coded BH = scipy within {max_dev:.1e} "
      f"on {n_sim:,} simulated genes; {k_step_up} step-up rejections; "
      f"{int(down.sum())} down + {int(up.sum())} up + {int(ns.sum()):,} "
      f"n.s. = {N_GENES:,})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 70)
gs = ms.grid(fig, 1, 1, left=13, right=5, top=9, bottom=11)
ax = fig.add_subplot(gs[0])

# every gene is drawn; only the point layer is rasterised
layers = [(ns, ms.GREY_LIGHT, 2.0, 1), (down, ms.BLUE, 4.0, 2),
          (up, ms.VERMILLION, 4.0, 2)]
for mask, colour, size, zorder in layers:
    ax.scatter(lfc[mask], neg_log[mask], s=size, color=colour, linewidths=0,
               alpha=0.85, zorder=zorder, rasterized=True)

y_top = max(31.0, float(np.ceil(neg_log.max())) + 3.0)
ax.set_xlim(-X_LIM, X_LIM)
ax.set_ylim(0, y_top)
dash = dict(color=ms.GREY_DARK, lw=0.5, ls=(0, (3, 2)), zorder=3)
for cut in (-LFC_CUT, LFC_CUT):
    ax.plot([cut, cut], [0, y_top], **dash)
ax.plot([-X_LIM, X_LIM], [-np.log10(PADJ_CUT)] * 2, **dash)
for gene, x, y in HITS:
    ax.annotate(gene, xy=(x, y), xytext=(4 if x > 0 else -4, 2),
                textcoords="offset points", ha="left" if x > 0 else "right",
                va="bottom", fontsize=ms.FS_SMALL, fontstyle="italic")

ax.set_xlabel("log$_{2}$ fold change (treated / control)",
              fontsize=ms.FS_MATH)
ax.set_ylabel("−log$_{10}$ adjusted P", fontsize=ms.FS_MATH)
ax.set_xticks(np.arange(-4, 5, 2))
header = ax.get_xaxis_transform()
ax.text(-5.4, 1.03, f"Down: {int(down.sum()):,}", transform=header,
        color=ms.BLUE, ha="left", va="bottom", fontweight="bold",
        fontsize=ms.FS_TICK)
ax.text(5.4, 1.03, f"Up: {int(up.sum()):,}", transform=header,
        color=ms.VERMILLION, ha="right", va="bottom", fontweight="bold",
        fontsize=ms.FS_TICK)
ax.text(0, 1.03, f"{N_GENES:,} genes", transform=header, color=ms.GREY_DARK,
        ha="center", va="bottom", fontsize=ms.FS_TICK)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig103_volcano_bh.{ext}")
print("fig103_volcano_bh: saved png + pdf")
