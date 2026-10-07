"""Fig. 129 - Oncoprint with memo-sorted samples (double column, 183 mm).

An oncoprint answers two questions at once: how often is each gene
altered, and do the alterations avoid each other? Genes are ranked by
frequency and samples by the "memo sort" (each sample scored as a binary
number whose most significant bit is the most frequent gene), which
turns mutually exclusive drivers into a staircase. Copy-number events
fill the cell and mutations are an inner band, so two events can share
one cell; the legend swatches are drawn by the same function as the
cells. The self-check: every printed percentage equals the row mean of
the binary matrix, the column order is non-increasing in the memo-sort
score, and the top bars add up to the number of events in the matrix.

Statistics: n = 60 samples x 12 genes; right bars, % of samples with
any alteration in the gene; top bars, alterations per sample by type;
no hypothesis test. All data are simulated.
"""

import numpy as np

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(129)
N_GENES, N_SAMPLES = 12, 60
# type: (colour, cell width, cell height) - copy number fills the cell
KINDS = {"Amplification": (ms.VERMILLION, 0.84, 0.86),
         "Deep deletion": (ms.BLUE, 0.84, 0.86),
         "Truncating": (ms.INK, 0.84, 0.34),
         "Missense": (ms.GREEN, 0.84, 0.34)}
EMPTY = ("#E4E3E1", 0.84, 0.86)
SUBTYPES = {"Subtype A": ms.PINK, "Subtype B": ms.SKY, "Subtype C": ms.YELLOW}

# ------------------------------------------------------------- DATA ----
# events[k, g, s] is True if sample s carries alteration type k in gene g.
# Four drivers are (almost) mutually exclusive and track the subtype; the
# other eight genes are altered independently at falling rates.
genes = [f"GENE{g}" for g in range(1, N_GENES + 1)]
subtype = rng.choice(3, N_SAMPLES, p=[0.45, 0.35, 0.20])
DRIVER_GIVEN_SUBTYPE = np.array([[0.62, 0.14, 0.08, 0.04],     # + none
                                 [0.10, 0.52, 0.18, 0.06],
                                 [0.06, 0.10, 0.30, 0.38]])
KIND_MIX = rng.dirichlet([0.7] * 4, N_GENES)        # each gene's own habit
altered_true = np.zeros((N_GENES, N_SAMPLES), dtype=bool)
for s in range(N_SAMPLES):
    p = DRIVER_GIVEN_SUBTYPE[subtype[s]]
    driver = rng.choice(5, p=np.append(p, 1 - p.sum()))
    if driver < 4:
        altered_true[driver, s] = True
altered_true[4:] = (rng.random((N_GENES - 4, N_SAMPLES))
                    < np.linspace(0.26, 0.08, N_GENES - 4)[:, None])
events = np.zeros((4, N_GENES, N_SAMPLES), dtype=bool)
for g, s in zip(*np.nonzero(altered_true)):
    kind = rng.choice(4, p=KIND_MIX[g])
    events[kind, g, s] = True
    if kind < 2 and rng.random() < 0.25:            # copy number + mutation
        events[rng.choice([2, 3]), g, s] = True

# ---------------------------------------------------------- MEMO SORT --
altered = events.any(axis=0)                         # binary matrix (g, s)
gene_order = np.argsort(-altered.mean(axis=1), kind="stable")
weights = 2 ** np.arange(N_GENES - 1, -1, -1)        # top gene = top bit
score = weights @ altered[gene_order]
sample_order = np.argsort(-score, kind="stable")
shown = events[:, gene_order][:, :, sample_order]    # as drawn
percent = 100 * shown.any(axis=0).mean(axis=1)
per_sample = shown.sum(axis=1)                       # (kind, sample)

# ------------------------------------------------------------ FIGURE --
CELL_W, CELL_H = 2.2, 4.0                            # millimetres
LEFT, BOTTOM = 17.0, 12.0
MAT_W, MAT_H = N_SAMPLES * CELL_W, N_GENES * CELL_H
fig = ms.figure(183, 83)
ax = ms.axes(fig, LEFT, BOTTOM, MAT_W, MAT_H)
ax_sub = ms.axes(fig, LEFT, BOTTOM + MAT_H + 1.2, MAT_W, 2.4)
ax_top = ms.axes(fig, LEFT, BOTTOM + MAT_H + 5.2, MAT_W, 12.0)
ax_pct = ms.axes(fig, LEFT + MAT_W + 2.0, BOTTOM, 17.0, MAT_H)


def draw_cells(x, y, style, **kwargs):
    """Cells centred on (x, y) in matrix units; also draws the legend."""
    colour, width, tall = style
    ax.bar(x, tall, width, bottom=np.asarray(y) - tall / 2, color=colour,
           lw=0, clip_on=False, **kwargs)


rows, cols = np.mgrid[0:N_GENES, 0:N_SAMPLES]
draw_cells(cols.ravel(), rows.ravel(), EMPTY)
for k, style in enumerate(KINDS.values()):
    g, s = np.nonzero(shown[k])
    draw_cells(s, g, style, zorder=3 + (k >= 2))     # mutations on top
ax.set_xlim(-0.5, N_SAMPLES - 0.5)
ax.set_ylim(N_GENES - 0.5, -0.5)
ax.set_yticks(range(N_GENES), [genes[g] for g in gene_order],
              fontstyle="italic")
ax.set_xticks([])
ax.tick_params(length=0, pad=3)
for spine in ax.spines.values():
    spine.set_visible(False)

# legend row under the matrix, in the same units as the cells
y_key = N_GENES + 0.75
x_key = 0.0
for name, style in list(KINDS.items()) + [("No alteration", EMPTY)]:
    draw_cells([x_key], [y_key], EMPTY)
    draw_cells([x_key], [y_key], style, zorder=3)
    ax.text(x_key + 0.9, y_key, name, va="center", fontsize=ms.FS_TICK)
    x_key += 2.2 + 0.55 * len(name)           # advance by the label length
ax.text(N_SAMPLES - 0.5, y_key, f"n = {N_SAMPLES} samples, memo-sorted",
        ha="right", va="center", fontsize=ms.FS_TICK, color=ms.GREY_DARK)

# clinical annotation row with its own key in the right margin
for code, colour in enumerate(SUBTYPES.values()):
    here = np.flatnonzero(subtype[sample_order] == code)
    ax_sub.bar(here, 1, 0.84, color=colour, lw=0)
ax_sub.set_xlim(*ax.get_xlim())
ax_sub.set_ylim(0, 1)
ax_sub.set_axis_off()
ax_sub.text(-0.012, 0.5, "Subtype", transform=ax_sub.transAxes, ha="right",
            va="center", fontsize=ms.FS_TICK)
for code, (name, colour) in enumerate(SUBTYPES.items()):
    y_sw = 0.5 + 1.75 * (2 - code)                    # in row heights
    ax_sub.bar(N_SAMPLES + 1.0, 0.9, 0.84, bottom=y_sw - 0.45, color=colour,
               lw=0, clip_on=False)
    ax_sub.text(N_SAMPLES + 1.9, y_sw, name, va="center",
                fontsize=ms.FS_TICK)

# top: alterations per sample, stacked in the legend order
base = np.zeros(N_SAMPLES)
for k, (colour, _w, _h) in enumerate(KINDS.values()):
    ax_top.bar(range(N_SAMPLES), per_sample[k], 0.84, bottom=base,
               color=colour, lw=0)
    base += per_sample[k]
ax_top.set_xlim(*ax.get_xlim())
ax_top.set_ylim(0, base.max())
ax_top.set_yticks(range(0, int(base.max()) + 1, 2))
ax_top.set_xticks([])
ax_top.spines["bottom"].set_visible(False)
ax_top.set_ylabel("Alterations\nper sample")

# right: % of samples altered, the number printed at the bar end
ax_pct.barh(range(N_GENES), percent, 0.62, color=ms.GREY_DARK, lw=0)
printed = [ax_pct.text(value + 1.5, g, f"{value:.0f}%", va="center",
                       fontsize=ms.FS_TICK)
           for g, value in enumerate(percent)]
ax_pct.set_xlim(0, 45)
ax_pct.set_ylim(*ax.get_ylim())
ax_pct.set_xticks([0, 20, 40])
ax_pct.set_yticks([])
ax_pct.spines["left"].set_visible(False)
ax_pct.set_xlabel("Altered (%)")

# ------------------------------------------------------- SELF-CHECK ---
# (after drawing, because the percentages are read back from the figure)
row_mean = 100 * altered[gene_order].mean(axis=1)
assert [t.get_text() for t in printed] == [f"{v:.0f}%" for v in row_mean]
assert np.all(np.diff(row_mean) <= 0)
assert np.all(np.diff(score[sample_order]) <= 0)
assert base.sum() == events.sum() and not (events[0] & events[1]).any()
print(f"fig129: self-check passed (printed % = row means, "
      f"{row_mean[0]:.0f}% down to {row_mean[-1]:.0f}%; memo-sort score "
      f"non-increasing over {N_SAMPLES} columns; top bars sum to "
      f"{int(events.sum())} events, {int((events.sum(axis=0) == 2).sum())} "
      "cells carry two)")

ms.assert_aligned([ax, ax_sub, ax_top], edges=("left", "right"))
ms.assert_aligned([ax, ax_pct])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig129_oncoprint.{ext}")
print("fig129_oncoprint: saved png + pdf")
