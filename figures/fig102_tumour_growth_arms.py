"""Fig. 102 - Tumour growth by treatment arm (single column, 89 mm).

Extends panel (a) of fig002, where the s.e.m. band is a formula around
one noisy curve: here every animal is simulated longitudinally and the
line and band are the mean ± s.d. across animals at each visit. Arms
are labelled at the right edge in their own colour instead of a legend,
and a bar above the axes marks the dosing period. The self-check uses
the design: before dosing all arms follow one growth law, so the arm
means at days <= 8 must agree within 4 standard errors of their
difference; every band must lie inside the y-limits, and the
end-point means must be far enough apart for the direct labels.

Statistics: n = 8 animals per arm; line, mean; band, ± s.d.; day-28
comparison by two-sided Welch's t-test (printed, not drawn). All data
are simulated.
"""

import numpy as np
from scipy import stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

ARMS = {"Vehicle": ms.GREY, "Monotherapy": ms.GREEN,
        "Combination": ms.VERMILLION}
RATE = {"Vehicle": 0.094, "Monotherapy": 0.078, "Combination": 0.034}
BASE_RATE = 0.094                    # per day, all arms before dosing
DAYS = np.arange(0, 29, 4)
TREATMENT_START = 8
N_ANIMALS = 8
Y_MAX = 1800.0
HEIGHT_MM, TOP_MM, BOTTOM_MM = 62.0, 7.0, 11.0

# ------------------------------------------------------------- DATA ----
# Exponential growth per animal: a common rate until dosing starts, then
# the arm's rate plus an animal-specific deviation; 6% measurement noise.
rng = np.random.default_rng(202)
volume = {}
for arm in ARMS:
    animals = np.empty((N_ANIMALS, DAYS.size))
    for a in range(N_ANIMALS):
        v0 = rng.normal(95, 10)
        wobble = rng.normal(0, 0.008)
        for d, day in enumerate(DAYS):
            growth = (BASE_RATE * min(day, TREATMENT_START)
                      + (RATE[arm] + wobble) * max(day - TREATMENT_START, 0))
            animals[a, d] = v0 * np.exp(growth) * rng.normal(1, 0.06)
    volume[arm] = animals
mean = {arm: v.mean(axis=0) for arm, v in volume.items()}
sd = {arm: v.std(axis=0, ddof=1) for arm, v in volume.items()}

welch = {arm: stats.ttest_ind(volume["Combination"][:, -1],
                              volume[arm][:, -1], equal_var=False).pvalue
         for arm in ("Vehicle", "Monotherapy")}

# ------------------------------------------------------- SELF-CHECK ---
# 1. one growth law before dosing: arm means agree within 4 s.e. of
#    their difference at every pre-dosing visit
pre = DAYS <= TREATMENT_START
names = list(ARMS)
worst_z = 0.0
for i, first in enumerate(names):
    for second in names[i + 1:]:
        diff = np.abs(mean[first][pre] - mean[second][pre])
        se = np.sqrt((sd[first][pre] ** 2 + sd[second][pre] ** 2) / N_ANIMALS)
        worst_z = max(worst_z, float(np.max(diff / se)))
assert worst_z < 4.0, worst_z
# 2. nothing clipped: every band inside the y-limits
band_top = max(float((mean[arm] + sd[arm]).max()) for arm in ARMS)
band_low = min(float((mean[arm] - sd[arm]).min()) for arm in ARMS)
assert 0.0 < band_low and band_top < Y_MAX, (band_low, band_top)
# 3. direct labels cannot overlap: end-point means at least 1.5 label
#    heights apart on the page
pt_per_unit = (HEIGHT_MM - TOP_MM - BOTTOM_MM) / 25.4 * 72 / Y_MAX
ends = np.sort([mean[arm][-1] for arm in ARMS])
min_gap_pt = float(np.min(np.diff(ends))) * pt_per_unit
assert min_gap_pt >= 1.5 * ms.FS_TICK, min_gap_pt
print(f"fig102: self-check passed (pre-dosing arm means within "
      f"{worst_z:.2f} s.e.; bands span {band_low:.0f}-{band_top:.0f} of "
      f"0-{Y_MAX:.0f} mm³; labels >= {min_gap_pt:.0f} pt apart; day-28 "
      f"Welch P = {welch['Vehicle']:.1e}, {welch['Monotherapy']:.1e})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, HEIGHT_MM)
gs = ms.grid(fig, 1, 1, left=14, right=24, top=TOP_MM, bottom=BOTTOM_MM)
ax = fig.add_subplot(gs[0])

for arm, colour in ARMS.items():
    hero = arm == "Combination"
    ax.fill_between(DAYS, mean[arm] - sd[arm], mean[arm] + sd[arm],
                    color=colour, alpha=0.18, linewidth=0)
    ax.plot(DAYS, mean[arm], color=colour, marker="o", markersize=2.6,
            lw=1.4 if hero else 1.0, zorder=3 if hero else 2)
    ax.annotate(arm, xy=(DAYS[-1], mean[arm][-1]), xytext=(5, 0),
                textcoords="offset points", color=colour, va="center",
                ha="left", fontsize=ms.FS_TICK, annotation_clip=False,
                fontweight="bold" if hero else "normal")

ax.set_xlim(-0.8, 28.8)
ax.set_ylim(0, Y_MAX)
ax.set_xticks(DAYS)
ax.set_yticks(np.arange(0, Y_MAX + 1, 300))
ax.set_xlabel("Time after implantation (days)")
ax.set_ylabel("Tumour volume (mm³)")
# dosing-period bar above the plot area, so no line crosses any text
above = ax.get_xaxis_transform()
ax.plot([TREATMENT_START, DAYS[-1]], [1.035, 1.035], transform=above,
        color=ms.INK, lw=1.2, clip_on=False, solid_capstyle="butt")
ax.text(TREATMENT_START, 1.06, "Dosing period", transform=above, ha="left",
        va="bottom", fontsize=ms.FS_TICK)
ax.text(0.03, 0.95, "n = 8 animals per arm\nMean ± s.d.",
        transform=ax.transAxes, va="top", fontsize=ms.FS_SMALL,
        color=ms.GREY_DARK)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig102_tumour_growth_arms.{ext}")
print("fig102_tumour_growth_arms: saved png + pdf")
