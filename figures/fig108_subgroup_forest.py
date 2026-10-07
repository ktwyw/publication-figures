"""Fig. 108 - Subgroup forest plot of hazard ratios for one trial (120 mm).

Where fig012 pools independent studies, this is the forest a trial report
needs: prespecified subgroups as grouped rows under header lines with
alternating bands, events/patients for both arms as text columns placed
in millimetres, squares sized by inverse-variance weight, an overall
diamond, and a log hazard-ratio axis with plain decimal ticks and
"favours" captions. The self-check is conservation: every factor
partitions the same patients, so events and patients must sum to the
same totals (129/400 and 176/400) for each factor in each arm; the
"All patients" row is built from those sums. The pooled estimate must
also lie inside the range of its strata with a narrower interval, and
pooling over any other factor must agree within one standard error.

Statistics: n = 400 patients per arm; squares, hazard ratio (area
proportional to inverse-variance weight); lines, 95% Wald interval on
the log scale; diamond, fixed-effect estimate pooled over the age
strata. All values are simulated summary data.
"""

import numpy as np

import manuscript as ms

ms.apply()
HERE = ms.HERE

# ------------------------------------------------------------- DATA ----
# factor, subgroup, events/patients (treatment), events/patients
# (control), log hazard ratio, standard error
ROWS = [
    ("Age", "< 65 years", "58/212", "84/208", -0.46, 0.171),
    ("Age", "≥ 65 years", "71/188", "92/192", -0.33, 0.158),
    ("Sex", "Female", "49/176", "70/171", -0.42, 0.186),
    ("Sex", "Male", "80/224", "106/229", -0.37, 0.148),
    ("Disease stage", "Stage II", "31/141", "49/138", -0.53, 0.229),
    ("Disease stage", "Stage III", "98/259", "127/262", -0.34, 0.134),
    ("Biomarker", "Positive", "44/163", "81/160", -0.71, 0.187),
    ("Biomarker", "Negative", "85/237", "95/240", -0.14, 0.149),
]
factor = np.array([row[0] for row in ROWS])
FACTORS = list(dict.fromkeys(factor))             # in order of appearance
log_hr = np.array([row[4] for row in ROWS])
se = np.array([row[5] for row in ROWS])
counts = np.array([[int(v) for v in f"{row[2]}/{row[3]}".split("/")]
                   for row in ROWS])   # events, patients: treatment, control

hr = np.exp(log_hr)
ci_low, ci_high = np.exp(log_hr - 1.96 * se), np.exp(log_hr + 1.96 * se)
weight = 1 / se ** 2


def pool(mask):
    """Fixed-effect (inverse-variance) log hazard ratio and its s.e."""
    return (float(np.sum(log_hr[mask] * weight[mask]) / weight[mask].sum()),
            float(np.sqrt(1 / weight[mask].sum())))


# The overall estimate pools one factor's strata, so each patient counts once
overall_log, overall_se = pool(factor == "Age")
overall = np.exp(overall_log + np.array([0.0, -1.96, 1.96]) * overall_se)
totals = counts[factor == "Age"].sum(axis=0)

# ------------------------------------------------------- SELF-CHECK ---
for name in FACTORS:
    assert np.array_equal(counts[factor == name].sum(axis=0), totals), name
    pooled_log, _pooled_se = pool(factor == name)
    assert abs(pooled_log - overall_log) < overall_se, (name, pooled_log)
assert tuple(totals) == (129, 400, 176, 400), totals
assert hr.min() < overall[0] < hr.max()
assert np.all(overall[2] / overall[1] < ci_high / ci_low)
print(f"fig108: self-check passed (each of {len(FACTORS)} factors sums to "
      f"{totals[0]}/{totals[1]} vs {totals[2]}/{totals[3]}; pooled HR "
      f"{overall[0]:.2f} ({overall[1]:.2f}–{overall[2]:.2f}) inside strata "
      f"range {hr.min():.2f}–{hr.max():.2f})")

# ------------------------------------------------------------ FIGURE --
# Rows: a header line per factor, then its subgroups, then a small gap
y_row, headers, y = np.empty(len(ROWS)), {}, 0.0
for name in FACTORS:
    headers[name] = y
    members = np.flatnonzero(factor == name)
    y_row[members] = y + 1 + np.arange(members.size)
    y += 1 + members.size + 0.35
y_overall = y + 0.2

WIDTH, LEFT, RIGHT = 120.0, 59.0, 30.5            # millimetres
fig = ms.figure(WIDTH, 82)
ax = fig.add_subplot(ms.grid(fig, 1, 1, left=LEFT, right=RIGHT, top=9,
                             bottom=11)[0])


def col(x_mm):
    """Axes fraction of a position x_mm from the figure's left edge."""
    return (x_mm - LEFT) / (WIDTH - LEFT - RIGHT)


COL = {"label": col(2.5), "treatment": col(33.5), "control": col(49.5),
       "estimate": col(92.0)}

for k, (name, y0) in enumerate(headers.items()):
    if k % 2 == 0:
        ax.axhspan(y0 - 0.5, y0 + (factor == name).sum() + 0.6,
                   xmin=col(1.0), xmax=col(WIDTH - 1.0), color="#F3F1EC",
                   lw=0, clip_on=False, zorder=0)
ax.plot([1, 1], [-0.5, y_overall + 0.8], color=ms.GREY, lw=0.6,
        ls=(0, (3, 2)), zorder=1)
ax.hlines(y_row, ci_low, ci_high, color=ms.BLUE, lw=0.9, zorder=2)
ax.scatter(hr, y_row, s=weight / weight.max() * 34, marker="s",
           color=ms.BLUE, linewidths=0, zorder=3)
ax.fill([overall[1], overall[0], overall[2], overall[0]],
        [y_overall, y_overall - 0.32, y_overall, y_overall + 0.32],
        color=ms.VERMILLION, lw=0, zorder=3)

ax.set_xscale("log")
ax.set_xlim(0.3, 1.6)
ax.set_xticks([0.3, 0.5, 0.7, 1.0, 1.5])
ax.minorticks_off()
ms.plain_log_ticks(ax.xaxis)
ax.set_ylim(y_overall + 0.9, -0.6)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Hazard ratio (95% CI)", labelpad=10)

# text columns: x in axes fractions, y in row units
row = ax.get_yaxis_transform()
bold = dict(transform=row, va="center", fontweight="bold")
plain = dict(transform=row, va="center")
head_y = -1.25
ax.text(COL["label"], head_y, "Subgroup", **bold)
ax.text(COL["treatment"], head_y, "Treatment", ha="center", **bold)
ax.text(COL["control"], head_y, "Control", ha="center", **bold)
ax.text(COL["estimate"], head_y, "HR (95% CI)", **bold)
ax.text((COL["treatment"] + COL["control"]) / 2, head_y - 0.95,
        "Events / patients", transform=row, va="center", ha="center",
        fontsize=ms.FS_TICK, color=ms.GREY_DARK)
for name, y0 in headers.items():
    ax.text(COL["label"], y0, name, **bold)
for k, (_name, subgroup, treatment, control, _log_hr, _se) in enumerate(ROWS):
    ax.text(COL["label"] + col(2.0) - col(0.0), y_row[k], subgroup, **plain)
    ax.text(COL["treatment"], y_row[k], treatment, ha="center", **plain)
    ax.text(COL["control"], y_row[k], control, ha="center", **plain)
    ax.text(COL["estimate"], y_row[k],
            f"{hr[k]:.2f} ({ci_low[k]:.2f}–{ci_high[k]:.2f})", **plain)
ax.text(COL["label"], y_overall, "All patients", **bold)
ax.text(COL["treatment"], y_overall, f"{totals[0]}/{totals[1]}",
        ha="center", **plain)
ax.text(COL["control"], y_overall, f"{totals[2]}/{totals[3]}", ha="center",
        **plain)
ax.text(COL["estimate"], y_overall,
        f"{overall[0]:.2f} ({overall[1]:.2f}–{overall[2]:.2f})", **bold)

below = ax.get_xaxis_transform()
caption = dict(transform=below, va="top", fontsize=ms.FS_SMALL,
               color=ms.GREY_DARK)
ax.text(0.97, -0.066, "← Favours treatment", ha="right", **caption)
ax.text(1.03, -0.066, "Favours control →", ha="left", **caption)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig108_subgroup_forest.{ext}")
print("fig108_subgroup_forest: saved png + pdf")
