"""Fig. 111 - Cell embedding beside a marker dot plot (double column, 183 mm).

A figure type new to the library: a 6,000-point embedding drawn as a
rasterised point layer (vector axes and text) with each population
labelled directly, beside a marker-gene dot plot in which dot area is
the percentage of cells expressing and colour is the scaled mean
expression, with a hand-built size legend and a colour bar placed in
millimetres. The self-check is that scaled expression peaks at exactly 1
for every gene, that each population's two highest-scoring genes are its
two designated markers, that the size-legend handles use the same area
per percentage point as the plotted dots, and that the two panels share
top and bottom edges.

Statistics: a, n = 6,000 cells, all drawn; b, dot area, percentage of
cells with non-zero expression; colour, mean expression scaled to each
gene's maximum across populations. All data are simulated; the embedding
is a simulated two-dimensional layout, not the output of UMAP or t-SNE.
"""

import numpy as np

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(1111)
POPULATIONS = [f"Population {k}" for k in range(1, 7)]
COLOURS = np.array([ms.BLUE, ms.VERMILLION, ms.GREEN, ms.PINK, ms.ORANGE,
                    ms.SKY])
GENES = [f"Marker {k}" for k in range(1, 13)]     # two per population
N_POP, N_GENE = len(POPULATIONS), len(GENES)
AREA_PER_PCT = 0.42                 # dot area, points² per percentage point
LEGEND_PCT = (25, 50, 75, 100)
# side of each cloud that carries its label, as (dx, dy) unit steps
LABEL_SIDE = [(0, 1), (-1, 0), (0, 1), (1, 0), (0, -1), (0, -1)]

# ------------------------------------------------------------- DATA ----
# a: one rotated, anisotropic Gaussian cloud per population
CENTRES = np.array([[-6.0, 4.0], [-2.5, -4.5], [1.5, 5.5], [5.5, 0.5],
                    [0.5, -0.5], [6.5, -5.5]])
SIZES = [1500, 1200, 1100, 900, 800, 500]
clouds = []
for centre, size in zip(CENTRES, SIZES):
    angle = rng.uniform(0, np.pi)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    clouds.append(rng.normal(0, 1, (size, 2)) * np.array([1.5, 0.8])
                  @ rotation.T + centre)
cells = np.vstack(clouds)                         # (n_cells, 2)
population = np.repeat(np.arange(N_POP), SIZES)   # population index per cell

# b: per population and gene, % of cells expressing and mean expression
is_marker = np.arange(N_GENE) // 2 == np.arange(N_POP)[:, None]
pct = np.empty((N_POP, N_GENE))
mean_expr = np.empty((N_POP, N_GENE))
for p in range(N_POP):
    for g in range(N_GENE):
        if is_marker[p, g]:
            pct[p, g] = rng.uniform(72, 96)
            mean_expr[p, g] = rng.uniform(2.2, 3.4)
        else:
            pct[p, g] = rng.uniform(2, 28)
            mean_expr[p, g] = rng.uniform(0.05, 0.9)
scaled = mean_expr / mean_expr.max(axis=0)        # each gene's maximum = 1

# ------------------------------------------------------- SELF-CHECK ---
assert np.all(scaled.max(axis=0) == 1.0) and scaled.min() > 0
top_two = np.sort(np.argsort(scaled, axis=1)[:, -2:], axis=1)
assert np.array_equal(top_two, np.arange(N_GENE).reshape(N_POP, 2)), top_two
margin = scaled[is_marker].min() - scaled[~is_marker].max()
assert margin > 0 and pct[is_marker].min() > pct[~is_marker].max()

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 80)
ax_a = ms.axes(fig, 12, 14, 60, 60)
ax_b = ms.axes(fig, 104, 14, 56, 60)

# a: every cell drawn in random order (no population hides another);
# the point layer is rasterised, axes and text stay vector
order = np.random.default_rng(1).permutation(len(cells))
ax_a.scatter(cells[order, 0], cells[order, 1], s=1.6, linewidths=0,
             c=COLOURS[population[order]], alpha=0.8, rasterized=True)
for p, (dx, dy) in enumerate(LABEL_SIDE):
    # direct label just beyond the cloud (98th percentile of its extent)
    # on the side that faces empty space, not a neighbouring population
    mine = cells[population == p]
    centre = np.median(mine, axis=0)
    reach = np.quantile((mine - centre) @ [dx, dy], 0.98)
    ax_a.annotate(str(p + 1), xy=centre + reach * np.array([dx, dy]),
                  xytext=(3 * dx, 3 * dy), textcoords="offset points",
                  ha=("right", "center", "left")[dx + 1],
                  va=("top", "center", "bottom")[dy + 1],
                  fontsize=ms.FS_BODY, fontweight="bold", color=COLOURS[p])
ax_a.set_xlim(-11, 11)
ax_a.set_ylim(-11, 11)
ax_a.set_xticks([])
ax_a.set_yticks([])
ax_a.set_xlabel("Embedding dimension 1")
ax_a.set_ylabel("Embedding dimension 2")
ax_a.text(0.02, 0.02, f"n = {len(cells):,} cells", transform=ax_a.transAxes,
          fontsize=ms.FS_TICK, color=ms.GREY_DARK, va="bottom")

# b: dot plot, populations in rows and genes in columns
gene_x, pop_y = np.meshgrid(np.arange(N_GENE), np.arange(N_POP))
dots = ax_b.scatter(gene_x.ravel(), pop_y.ravel(),
                    s=pct.ravel() * AREA_PER_PCT, c=scaled.ravel(),
                    cmap="Reds", vmin=0, vmax=1, edgecolor=ms.GREY_DARK,
                    linewidths=0.3)
ax_b.set_xlim(-0.7, N_GENE - 0.3)
ax_b.set_ylim(N_POP - 0.4, -0.6)
ax_b.set_xticks(range(N_GENE), GENES, rotation=90, ha="right", va="center",
                rotation_mode="anchor", fontstyle="italic")
ax_b.set_yticks(range(N_POP), POPULATIONS)
ax_b.tick_params(length=0, pad=3)
ax_b.tick_params(axis="y", pad=9)
ax_b.scatter(np.full(N_POP, -0.032), np.arange(N_POP), s=16, marker="s",
             c=COLOURS, linewidths=0, transform=ax_b.get_yaxis_transform(),
             clip_on=False)                       # colour key to panel a
for side in ("top", "right"):
    ax_b.spines[side].set_visible(True)

ax_cbar = ms.axes(fig, 166, 50, 2.4, 18)
cbar = fig.colorbar(dots, cax=ax_cbar, ticks=[0, 0.5, 1])
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(length=2, width=0.5)
cbar.ax.set_title("Scaled mean\nexpression", fontsize=ms.FS_TICK,
                  loc="left", pad=4)
handles = [ax_b.scatter([], [], s=level * AREA_PER_PCT, color="white",
                        edgecolor=ms.GREY_DARK, linewidths=0.5,
                        label=f"{level}") for level in LEGEND_PCT]
fig.legend(handles=handles, title="Expressing\ncells (%)", alignment="left",
           loc="upper left", bbox_to_anchor=(164.5 / 183, 40 / 80),
           labelspacing=0.55, handletextpad=0.8)
ms.panel_label(ax_a, "a", dx_pt=-20)
ms.panel_label(ax_b, "b", dx_pt=-20)

# the legend must encode area exactly as the plotted dots do
legend_scale = np.array([h.get_sizes()[0] for h in handles]) / LEGEND_PCT
assert np.allclose(legend_scale, dots.get_sizes()[0] / pct[0, 0])
worst = ms.assert_aligned([ax_a, ax_b])
print(f"fig111: self-check passed (every gene peaks at 1; top-2 genes = "
      f"designated markers for all {N_POP} populations, margin "
      f"{margin:.2f}; legend area {legend_scale[0]:.2f} pt² per % as "
      f"plotted; panel edges within {worst:.2f} pt)")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig111_embedding_dotplot.{ext}")
print("fig111_embedding_dotplot: saved png + pdf")
