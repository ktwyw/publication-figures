"""Fig. 130 - Protein lollipop of mutation recurrence (1.5 column, 120 mm).

A lollipop plot puts mutation recurrence on the protein it hits: one stem
per mutated residue, head area proportional to the number of samples,
colour for the consequence class, over a domain bar that says what the
residue belongs to. Only hotspots earn a label, and "hotspot" is a
computed statement: the background rate per residue is estimated from
the fraction of residues with no mutation (which planted hotspots cannot
inflate), and a residue is labelled when its count exceeds the Poisson
quantile at a Bonferroni-corrected level. The self-check: the plotted
counts sum to the number of simulated mutations, and the labels read
back from the figure are exactly the residues above that threshold,
which are exactly the planted hotspots.

Statistics: n = 300 mutations on a 420-residue protein; head area,
number of mutations at the residue; colour, majority class at the
residue; hotspot, count above the upper 0.01 / 420 Poisson quantile of
the background rate. All data are simulated.
"""

import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(130)
LENGTH, N_BACKGROUND, ALPHA = 420, 170, 0.01
CLASSES = {"Missense": ms.BLUE, "Truncating": ms.VERMILLION,
           "In-frame": ms.GREEN}
AREA_PER_MUTATION = 2.6                      # head area, pt² per mutation

# ------------------------------------------------------------- DATA ----
# Domain architecture: name, first and last residue, colour
DOMAINS = [("Domain A", 28, 118, ms.SKY), ("Motif B", 142, 153, ms.GREY_DARK),
           ("Domain C", 164, 262, ms.YELLOW), ("Domain D", 306, 384, ms.PINK)]
# Planted hotspots: residue -> (number of mutations, class index)
HOTSPOTS = {72: (34, 0), 175: (52, 0), 248: (17, 2), 341: (27, 1)}
residue = rng.integers(1, LENGTH + 1, N_BACKGROUND)          # background
klass = rng.choice(3, N_BACKGROUND, p=[0.68, 0.24, 0.08])
for site, (n_site, k_site) in HOTSPOTS.items():
    residue = np.append(residue, np.full(n_site, site))
    klass = np.append(klass, np.full(n_site, k_site))
N_MUTATIONS = residue.size

# ------------------------------------------------- COUNTS AND HOTSPOTS --
by_class = np.zeros((3, LENGTH + 1), dtype=int)     # index 0 is unused
np.add.at(by_class, (klass, residue), 1)
count = by_class.sum(axis=0)
mutated = np.flatnonzero(count)
majority = by_class[:, mutated].argmax(axis=0)
# Poisson background: P(count = 0) = exp(-rate), so the empty residues
# give the rate without being told which residues are hotspots
rate = -np.log(np.mean(count[1:] == 0))
threshold = int(stats.poisson.isf(ALPHA / LENGTH, rate))
called = np.flatnonzero(count > threshold)

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 65.5)
gs = ms.grid(fig, 2, 1, left=14, right=5, top=6, bottom=11, hspace=0,
             height_ratios=[40, 8.5])
ax = fig.add_subplot(gs[0])
ax_bar = fig.add_subplot(gs[1], sharex=ax)

# stems first, then heads from the largest down so small ones stay visible
order = np.argsort(-count[mutated])
ax.vlines(mutated, 0, count[mutated], color=ms.GREY, lw=0.5, zorder=2)
ax.scatter(mutated[order], count[mutated][order],
           s=AREA_PER_MUTATION * count[mutated][order],
           c=[list(CLASSES.values())[k] for k in majority[order]],
           edgecolors="white", linewidths=0.3, zorder=3, clip_on=False)
labels = []
for site in called:
    radius = np.sqrt(AREA_PER_MUTATION * count[site] / np.pi)   # points
    labels.append(ax.annotate(f"p.X{site}", (site, count[site]),
                              xytext=(0, radius + 1.5),
                              textcoords="offset points", ha="center",
                              va="bottom", fontsize=ms.FS_TICK))
ax.set_xlim(0, LENGTH + 1)
ax.set_ylim(0, 60)
ax.set_yticks(range(0, 61, 20))
ax.spines["bottom"].set_visible(False)
ax.tick_params(bottom=False, labelbottom=False)
ax.set_ylabel("No. of mutations")

# protein: grey backbone, domains named inside when the name fits
# (the backbone is drawn only in the linkers, so no label sits on two fills)
linker_start = [1] + [last for _n, _first, last, _c in DOMAINS]
linker_end = [first for _n, first, _last, _c in DOMAINS] + [LENGTH]
ax_bar.broken_barh([(a, b - a) for a, b in zip(linker_start, linker_end)],
                   (-0.16, 0.32), color=ms.GREY_LIGHT, lw=0)
renderer = fig.canvas.get_renderer()
for name, first, last, colour in DOMAINS:
    ax_bar.broken_barh([(first, last - first)], (-0.5, 1.0), color=colour,
                       lw=0)
    text = ax_bar.text((first + last) / 2, 0, name, ha="center", va="center",
                       fontsize=ms.FS_TICK,
                       color="white" if colour == ms.GREY_DARK else ms.INK)
    box = text.get_window_extent(renderer).transformed(
        ax_bar.transData.inverted())
    if box.width > 0.9 * (last - first):           # too long: set it above
        text.set(y=0.62, va="bottom", color=ms.INK)
ax_bar.set_ylim(-0.62, 1.55)
ax_bar.set_yticks([])
ax_bar.spines["left"].set_visible(False)
ax_bar.spines["bottom"].set_position(("outward", 2))
ax_bar.set_xticks([1, 100, 200, 300, LENGTH])
ax_bar.set_xlabel("Amino-acid position")
ax_bar.patch.set_alpha(0)

class_key = ax.legend(
    handles=[Line2D([], [], ls="", marker="o", markersize=4.2,
                    markeredgewidth=0, color=colour, label=name)
             for name, colour in CLASSES.items()],
    title="Mutation class", alignment="left", loc="upper right",
    bbox_to_anchor=(0.80, 1.0), handletextpad=0.2)
ax.add_artist(class_key)
ax.legend(
    handles=[Line2D([], [], ls="", marker="o", color=ms.GREY,
                    markersize=np.sqrt(AREA_PER_MUTATION * n),   # s = size²
                    markeredgewidth=0, label=str(n)) for n in (1, 10, 40)],
    title="Mutations", alignment="left", loc="upper right",
    bbox_to_anchor=(1.0, 1.0), handletextpad=0.6, labelspacing=0.75)
ax.text(0.012, 0.97, f"n = {N_MUTATIONS} mutations", transform=ax.transAxes,
        va="top", fontsize=ms.FS_TICK, color=ms.GREY_DARK)

# ------------------------------------------------------- SELF-CHECK ---
# (after drawing, because the hotspot labels are read back from the figure)
assert count[mutated].sum() == N_MUTATIONS == N_BACKGROUND + sum(
    n for n, _k in HOTSPOTS.values())
labelled = sorted(int(t.get_text().removeprefix("p.X")) for t in labels)
assert labelled == sorted(np.flatnonzero(count > threshold).tolist())
assert labelled == sorted(HOTSPOTS)
background_max = int(np.delete(count, list(HOTSPOTS)).max())
print(f"fig130: self-check passed (counts sum to {N_MUTATIONS} mutations; "
      f"background rate {rate:.3f} per residue -> Poisson threshold > "
      f"{threshold}; labelled hotspots {labelled} = planted; largest "
      f"background count {background_max})")

ms.assert_aligned([ax, ax_bar], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig130_lollipop_mutations.{ext}")
print("fig130_lollipop_mutations: saved png + pdf")
