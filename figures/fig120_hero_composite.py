"""Fig. 120 - Asymmetric composite: a hero panel and two supports (183 mm).

Where fig002 lays four equal panels on a 2 x 2 grid, this figure gives
the main result the space: one hero panel spanning two rows beside two
stacked supporting panels, on a millimetre grid. One role per panel
(a, main effect: capacity retention over 500 cycles; b, mechanism:
charge-transfer resistance growth; c, robustness: capacity across
discharge rates), one uncertainty definition throughout, direct labels
on the hero curves, and panel letters at fixed point offsets. The
self-check is that retention is exactly 100% at cycle 0 for every cell,
the coated group retains more at cycle 500, the hero's top and bottom
edges match the small panels' within 1.5 pt, and the small panels share
left and right edges.

Statistics: n = 4 cells per group in every panel; lines, symbols and
bars, mean; bands and error bars, ± s.d.; b also shows every cell; no
hypothesis test. All data are simulated.
"""

import numpy as np

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(2020)
GROUPS = {"Uncoated": ms.GREY_DARK, "Coated": ms.BLUE}
N_CELLS = 4
CYCLES = np.arange(0, 501, 10)
CHECKPOINTS = [1, 250, 500]
C_RATES = ["0.1C", "0.5C", "1C", "2C", "5C"]

# ------------------------------------------------------------- DATA ----
FADE = {"Uncoated": (0.00036, 0.0085), "Coated": (0.00012, 0.0030)}
RESISTANCE = {"Uncoated": [42, 96, 158], "Coated": [45, 58, 71]}  # Ω cm²
CAPACITY = {"Uncoated": [198, 186, 174, 151, 104],
            "Coated": [196, 187, 177, 160, 122]}                  # mAh/g
retention = np.empty((len(GROUPS), N_CELLS, CYCLES.size))   # (group, cell, .)
rct = np.empty((len(GROUPS), N_CELLS, len(CHECKPOINTS)))
capacity = np.empty((len(GROUPS), N_CELLS, len(C_RATES)))
for g, group in enumerate(GROUPS):
    linear, root = FADE[group]
    for cell in range(N_CELLS):
        scale = rng.normal(1, 0.12)
        loss = scale * (linear * CYCLES + root * np.sqrt(CYCLES) / 2)
        retention[g, cell] = (100 * (1 - loss)
                              + rng.normal(0, 0.25, CYCLES.size))
        retention[g, cell, 0] = 100.0           # normalised to cycle 0
        rct[g, cell] = [rng.normal(r, 0.07 * r) for r in RESISTANCE[group]]
        capacity[g, cell] = [rng.normal(q, 3.0) for q in CAPACITY[group]]


def mean_sd(values):
    """The one uncertainty definition of the figure: across cells."""
    return values.mean(axis=1), values.std(axis=1, ddof=1)


ret_mean, ret_sd = mean_sd(retention)
rct_mean, rct_sd = mean_sd(rct)
cap_mean, cap_sd = mean_sd(capacity)

# ------------------------------------------------------- SELF-CHECK ---
assert np.all(retention[:, :, 0] == 100.0) and np.all(ret_sd[:, 0] == 0)
assert CYCLES[-1] == 500 and ret_mean[1, -1] > ret_mean[0, -1]
assert np.all(rct_mean[0, 1:] > rct_mean[1, 1:])     # mechanism, cycles 250+
print("fig120: self-check passed (retention 100% at cycle 0 for all "
      f"{retention.shape[0] * N_CELLS} cells; at cycle 500 coated "
      f"{ret_mean[1, -1]:.1f} ± {ret_sd[1, -1]:.1f}% > uncoated "
      f"{ret_mean[0, -1]:.1f} ± {ret_sd[0, -1]:.1f}%)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 98)
gs = ms.grid(fig, 2, 2, left=14, right=5, top=7, bottom=11, wspace=19,
             hspace=17, width_ratios=[1.75, 1])
ax_a = fig.add_subplot(gs[:, 0])
ax_b, ax_c = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

# a, hero: capacity retention, labelled directly in empty regions
# (above the upper curve, below-left of the lower one)
LABEL_AT = {"Uncoated": 300, "Coated": 240}           # anchor cycle
for g, (group, colour) in enumerate(GROUPS.items()):
    hero = group == "Coated"
    ax_a.fill_between(CYCLES, ret_mean[g] - ret_sd[g], ret_mean[g] + ret_sd[g],
                      color=colour, alpha=0.2, lw=0)
    ax_a.plot(CYCLES, ret_mean[g], color=colour, lw=1.5 if hero else 1.1)
    side = 1 if hero else -1
    k = int(np.flatnonzero(CYCLES == LABEL_AT[group])[0])
    ax_a.annotate(f"{group}: {ret_mean[g, -1]:.1f} ± {ret_sd[g, -1]:.1f}% "
                  "at cycle 500",
                  xy=(CYCLES[k], ret_mean[g, k] + side * ret_sd[g, k]),
                  xytext=(0, 5 * side), textcoords="offset points",
                  ha="left" if hero else "right",
                  va="bottom" if hero else "top", color=colour,
                  fontsize=ms.FS_TICK, fontweight="bold" if hero else "normal")
ax_a.plot([0, 500], [80, 80], color=ms.GREY, lw=0.6, ls=(0, (3, 2)))
ax_a.text(494, 80.7, "80% end-of-life criterion", fontsize=ms.FS_SMALL,
          color=ms.GREY, va="bottom", ha="right")
ax_a.set_xlim(0, 505)
ax_a.set_ylim(60, 101.5)
ax_a.set_yticks(np.arange(60, 101, 10))
ax_a.set_xlabel("Cycle number")
ax_a.set_ylabel("Capacity retention (%)")
ax_a.text(0.03, 0.04, "n = 4 cells per group; mean ± s.d.\n"
          "1C charge / 1C discharge, 25 °C", transform=ax_a.transAxes,
          fontsize=ms.FS_SMALL, color=ms.GREY_DARK, va="bottom")

# b, mechanism: interfacial resistance at three checkpoints
jitter_rng = np.random.default_rng(20)     # horizontal jitter, display only
WIDTH = 0.36
for g, (group, colour) in enumerate(GROUPS.items()):
    x = np.arange(len(CHECKPOINTS)) + (g - 0.5) * WIDTH
    ax_b.bar(x, rct_mean[g], width=WIDTH * 0.92, color=colour,
             alpha=0.85 if group == "Coated" else 0.45, lw=0, label=group)
    ax_b.errorbar(x, rct_mean[g], yerr=rct_sd[g], fmt="none", ecolor=ms.INK,
                  elinewidth=0.6, capsize=1.8, capthick=0.6)
    for k, xk in enumerate(x):
        ax_b.scatter(xk + jitter_rng.uniform(-0.08, 0.08, N_CELLS),
                     rct[g, :, k], s=5, facecolor="white", edgecolor=ms.INK,
                     linewidths=0.4, zorder=3)
ax_b.set_xticks(range(len(CHECKPOINTS)), [f"Cycle {c}" for c in CHECKPOINTS])
ax_b.tick_params(axis="x", length=0, pad=3)
ax_b.set_ylim(0, 200)
ax_b.set_yticks(np.arange(0, 201, 50))
ax_b.set_ylabel("Charge-transfer resistance\n(Ω cm²)")
ax_b.legend(loc="upper left", handlelength=1.0, handleheight=0.9)

# c, robustness: rate capability
for g, (group, colour) in enumerate(GROUPS.items()):
    ax_c.errorbar(range(len(C_RATES)), cap_mean[g], yerr=cap_sd[g],
                  color=colour, marker="o", markersize=3.2, markeredgewidth=0,
                  lw=1.3 if group == "Coated" else 1.0, elinewidth=0.6,
                  capsize=1.8, capthick=0.6, label=group)
ax_c.set_xticks(range(len(C_RATES)), C_RATES)
ax_c.set_xlim(-0.4, len(C_RATES) - 0.6)
ax_c.set_ylim(80, 210)
ax_c.set_yticks(np.arange(80, 211, 40))
ax_c.set_xlabel("Discharge rate")
ax_c.set_ylabel("Discharge capacity\n(mAh per g)")
ax_c.legend(loc="lower left")

ms.panel_label(ax_a, "a", dx_pt=-26)
ms.panel_label(ax_b, "b", dx_pt=-34)
ms.panel_label(ax_c, "c", dx_pt=-34)

# the hero spans exactly the two stacked panels
box_a, box_b, box_c = (ms.plot_area_pt(ax) for ax in (ax_a, ax_b, ax_c))
assert abs(box_a[3] - box_b[3]) <= 1.5 and abs(box_a[1] - box_c[1]) <= 1.5
ms.assert_aligned([ax_b, ax_c], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig120_hero_composite.{ext}")
print("fig120_hero_composite: saved png + pdf")
