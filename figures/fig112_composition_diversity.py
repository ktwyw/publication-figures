"""Fig. 112 - Per-sample community composition and diversity (183 mm).

Extends fig021, a stacked area over time, to the other common case:
one stacked bar per sample for two groups, with group brackets under
the axis and a legend ordered like the stack. A second panel reduces
each bar to its Shannon index and compares the groups. The self-check
is that every bar closes at exactly 100%, that every index lies in
[0, ln 7], that an even seven-taxon community scores ln 7 through the
same function, and that the two panels share top and bottom edges.

Statistics: n = 8 individuals per group; box, median and IQR; whiskers,
1.5 x IQR; every point shown; two-sided Mann-Whitney U test.
All data are simulated.
"""

import numpy as np
from scipy import stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(1212)
N_PER_GROUP = 8
TAXA = {"Bacteroidota": ms.BLUE, "Firmicutes": ms.SKY,
        "Proteobacteria": ms.VERMILLION, "Actinobacteriota": ms.ORANGE,
        "Verrucomicrobiota": ms.GREEN, "Fusobacteriota": ms.PINK,
        "Other": ms.GREY_LIGHT}
GROUPS = {"Healthy": ms.GREY_DARK, "Disease": ms.VERMILLION}

# ------------------------------------------------------------- DATA ----
# Dirichlet shares; a small concentration makes the disease group uneven.
CONCENTRATION = {"Healthy": np.array([34, 28, 6, 12, 9, 4, 7]) * 0.9,
                 "Disease": np.array([13, 17, 52, 4, 3, 5, 6]) * 0.45}
shares = np.vstack([rng.dirichlet(CONCENTRATION[group])
                    for group in GROUPS for _ in range(N_PER_GROUP)])
samples = [f"{group[0]}{k}" for group in GROUPS
           for k in range(1, N_PER_GROUP + 1)]
group_of = np.repeat(list(GROUPS), N_PER_GROUP)


def shannon(p):
    """Shannon index H = -sum p ln p, with 0 ln 0 taken as 0."""
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


diversity = np.array([shannon(row) for row in shares])
h_healthy = diversity[group_of == "Healthy"]
h_disease = diversity[group_of == "Disease"]
test = stats.mannwhitneyu(h_healthy, h_disease, alternative="two-sided")

# ------------------------------------------------------- SELF-CHECK ---
H_MAX = np.log(len(TAXA))
assert np.allclose(shares.sum(axis=1), 1.0, atol=1e-12)
assert np.all((diversity >= 0) & (diversity <= H_MAX))
assert abs(shannon(np.full(len(TAXA), 1 / len(TAXA))) - H_MAX) < 1e-12
print(f"fig112: self-check passed (16 bars sum to 100%; Shannon "
      f"{diversity.min():.2f}-{diversity.max():.2f} within "
      f"[0, ln 7 = {H_MAX:.3f}]; medians {np.median(h_healthy):.2f} vs "
      f"{np.median(h_disease):.2f}, U = {test.statistic:.0f}, "
      f"P = {test.pvalue:.2g})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 68)
ax_a = ms.axes(fig, 14, 16, 96, 45)
ax_b = ms.axes(fig, 153, 16, 26, 45)

x = np.concatenate([np.arange(N_PER_GROUP),
                    np.arange(N_PER_GROUP) + N_PER_GROUP + 0.8])
bottom = np.zeros(len(samples))
for k, (taxon, colour) in enumerate(TAXA.items()):
    ax_a.bar(x, shares[:, k] * 100, bottom=bottom, width=0.82, color=colour,
             edgecolor="white", linewidth=0.3, label=taxon)
    bottom += shares[:, k] * 100
assert np.allclose(bottom, 100.0, atol=1e-9)       # every bar tops out at 100
ax_a.set_xlim(-0.7, 17.5)
ax_a.set_ylim(0, 100)
ax_a.set_xticks(x, samples)
ax_a.tick_params(axis="x", length=0, pad=2)
ax_a.set_ylabel("Relative abundance (%)")
ax_a.spines["bottom"].set_visible(False)

below = ax_a.get_xaxis_transform()
for name, colour in GROUPS.items():
    xs = x[group_of == name]
    ax_a.plot([xs[0] - 0.41, xs[-1] + 0.41], [-0.135, -0.135],
              transform=below, clip_on=False, color=colour, lw=1.2,
              solid_capstyle="butt")
    ax_a.text(xs.mean(), -0.17, f"{name} (n = {N_PER_GROUP})",
              transform=below, ha="center", va="top", color=colour)

# legend top-to-bottom in the order the bars stack
handles, labels = ax_a.get_legend_handles_labels()
ax_a.legend(handles[::-1], labels[::-1], loc="center left",
            bbox_to_anchor=(1.01, 0.5), title="Phylum", alignment="left",
            handlelength=1.0, handleheight=1.0)

jitter = np.random.default_rng(12)       # horizontal jitter: display only
for position, (name, colour) in enumerate(GROUPS.items()):
    values = diversity[group_of == name]
    ax_b.boxplot(values, positions=[position], widths=0.5, showfliers=False,
                 patch_artist=True, manage_ticks=False,
                 boxprops=dict(facecolor="white", edgecolor=colour,
                               linewidth=0.8),
                 medianprops=dict(color=colour, linewidth=1.2),
                 whiskerprops=dict(color=colour, linewidth=0.8),
                 capprops=dict(linewidth=0))
    ax_b.scatter(position + jitter.uniform(-0.14, 0.14, values.size), values,
                 s=8, color=colour, alpha=0.8, linewidths=0, zorder=3)
low, top = diversity.min(), diversity.max()
span = top - low
ms.bracket(ax_b, 0, 1, top + 0.12 * span, ms.format_p(test.pvalue),
           tick=0.04 * span, text_pad=0.025 * span)
ax_b.set_xlim(-0.6, 1.6)
ax_b.set_ylim(low - 0.12 * span, top + 0.32 * span)
ax_b.set_xticks([0, 1], list(GROUPS))
ax_b.set_ylabel("Shannon diversity index")
ms.panel_label(ax_a, "a")
ms.panel_label(ax_b, "b")

ms.assert_aligned([ax_a, ax_b])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig112_composition_diversity.{ext}")
print("fig112_composition_diversity: saved png + pdf")
