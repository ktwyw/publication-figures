"""Fig. 135 - CONSORT flow diagram drawn in code (1.5 column, 120 mm).

A participant-flow diagram is usually assembled by hand in a drawing
program, which is where its numbers drift from the analysis. Here every
count is tallied from the enrolment table, every box is sized from the
rendered extent of its own text, and boxes, gaps and right-angle
connectors are placed in millimetres, so the diagram redraws itself
when the data change. The self-check is conservation at every node
(assessed = excluded + randomised; randomised = arm A + arm B;
allocated = lost + discontinued + analysed in each arm) and, measured
with the renderer after drawing, that every text lies inside its box.

Statistics: n = 486 people assessed for eligibility; 1:1 allocation;
counts only, no interval and no test. All data are simulated.
"""

import numpy as np
from matplotlib.patches import FancyBboxPatch

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(135)
ARMS = ["Treatment", "Control"]

# ------------------------------------------------------------- DATA ----
# Enrolment table, one entry per person: screening outcome, then for the
# randomised the arm, whether the intervention was received, and the
# status at the 12-month visit.
SCREENING = {"Randomised": 0.62, "Not meeting inclusion criteria": 0.22,
             "Declined to participate": 0.11, "Other reasons": 0.05}
FOLLOW_UP = {"Completed": (0.84, 0.86), "Lost to follow-up": (0.05, 0.07),
             "Adverse event": (0.07, 0.02), "Withdrew consent": (0.04, 0.05)}
screening = rng.choice(list(SCREENING), 486, p=list(SCREENING.values()))
n_randomised = int((screening == "Randomised").sum())
arm = rng.permutation(np.resize(ARMS, n_randomised))         # 1:1
received = rng.random(n_randomised) > 0.03
status = np.array([rng.choice(list(FOLLOW_UP), p=[p[ARMS.index(a)]
                                                  for p in FOLLOW_UP.values()])
                   for a in arm])

# ------------------------------------------------------------ COUNTS ---
excluded = {reason: int((screening == reason).sum())
            for reason in list(SCREENING)[1:]}
flow = {}
for name in ARMS:
    here = arm == name
    flow[name] = {"allocated": int(here.sum()),
                  "received": int((here & received).sum()),
                  **{state: int((here & (status == state)).sum())
                     for state in FOLLOW_UP}}
    flow[name]["discontinued"] = (flow[name]["Adverse event"]
                                  + flow[name]["Withdrew consent"])

# ------------------------------------------------------- SELF-CHECK ---
assert screening.size == sum(excluded.values()) + n_randomised
assert n_randomised == sum(f["allocated"] for f in flow.values())
for f in flow.values():
    assert f["allocated"] == (f["Lost to follow-up"] + f["discontinued"]
                              + f["Completed"])
    assert 0 <= f["allocated"] - f["received"] <= f["allocated"]
assert abs(flow["Treatment"]["allocated"] - flow["Control"]["allocated"]) <= 1
print(f"fig135: self-check passed ({screening.size} assessed = "
      f"{sum(excluded.values())} excluded + {n_randomised} randomised; "
      + "; ".join(f"{name} {f['allocated']} = {f['Lost to follow-up']} lost "
                  f"+ {f['discontinued']} discontinued + {f['Completed']} "
                  f"analysed" for name, f in flow.items()) + ")")

# ------------------------------------------------------------ FIGURE --
W, H = 120.0, 99.0                      # page, mm; the axes is the page
EDGE, PAD, CLEAR, RADIUS = 4.0, 1.7, 2.6, 0.8
STAGE_X, LEFT = 4.5, 10.0               # stage captions, first column
fig = ms.figure(W, H)
ax = ms.axes(fig, 0, 0, W, H)
ax.set(xlim=(0, W), ylim=(0, H))
ax.axis("off")
renderer = fig.canvas.get_renderer()
boxes = []                              # (text, patch), for the last check


def n(count):
    return f"(n = {count})"


def measure(label):
    """Text artist and the size (mm) of a box that holds it with padding."""
    text = ax.text(0, 0, label, fontsize=ms.FS_TICK, linespacing=1.4,
                   va="center")
    extent = text.get_window_extent(renderer)
    return (text, extent.width / fig.dpi * 25.4 + 2 * PAD,
            extent.height / fig.dpi * 25.4 + 2 * PAD)


def place(item, x_left, y_top, width=None, height=None, fill="white"):
    """Draw the box; return its centre x, top y and bottom y (mm)."""
    text, own_width, own_height = item
    width, height = width or own_width, height or own_height
    patch = FancyBboxPatch(
        (x_left, y_top - height), width, height, facecolor=fill,
        edgecolor=ms.INK, lw=0.6,
        boxstyle=f"round,pad=0,rounding_size={RADIUS}")
    ax.add_patch(patch)
    text.set_position((x_left + PAD, y_top - height / 2))
    boxes.append((text, patch))
    return x_left + width / 2, y_top, y_top - height


def connect(*points):
    """Right-angle connector through the points, arrow head at the end."""
    assert all(a[0] == b[0] or a[1] == b[1]
               for a, b in zip(points, points[1:])), "not a right angle"
    ax.plot(*zip(*points[:-1]), color=ms.INK, lw=0.6,
            solid_capstyle="projecting", solid_joinstyle="miter")
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>", color=ms.INK, lw=0.6,
                                shrinkA=0, shrinkB=0, mutation_scale=5.5))


assessed = measure(f"Assessed for eligibility {n(screening.size)}")
randomised = measure(f"Randomised {n(n_randomised)}")
exclusion = measure(f"Excluded {n(sum(excluded.values()))}\n" + "\n".join(
    f"   {reason} {n(count)}" for reason, count in excluded.items()))
allocation, follow_up, analysis = [], [], []
for name, f in flow.items():
    allocation.append(measure(
        f"Allocated to {name.lower()} {n(f['allocated'])}\n"
        f"   Received intervention {n(f['received'])}\n"
        f"   Did not receive it {n(f['allocated'] - f['received'])}"))
    follow_up.append(measure(
        f"Lost to follow-up {n(f['Lost to follow-up'])}\n"
        f"Discontinued intervention {n(f['discontinued'])}\n"
        f"   Adverse event {n(f['Adverse event'])}\n"
        f"   Withdrew consent {n(f['Withdrew consent'])}"))
    analysis.append(measure(f"Analysed {n(f['Completed'])}\n"
                            f"   Completed 12-month visit"))

# columns share one width and each row one height, so edges line up
rows = (allocation, follow_up, analysis)
col_w = max(item[1] for row in rows for item in row)
row_h = [max(item[2] for item in row) for row in rows]
col_x = [LEFT, W - EDGE - col_w]
x_mid = (col_x[0] + col_x[1] + col_w) / 2
side_gap = exclusion[2] + 2 * CLEAR     # room for the exclusion box
gap = (H - 2 * EDGE - assessed[2] - side_gap - randomised[2]
       - sum(row_h)) / 3                # what is left, shared equally
assert col_x[1] - col_x[0] - col_w >= 4 and gap >= 5, (col_w, gap)

_x, y_top, y = place(assessed, x_mid - assessed[1] / 2, H - EDGE)
stage_y = {"Enrolment": y - side_gap / 2}
side = place(exclusion, col_x[1] + col_w - exclusion[1],
             y - CLEAR, fill="#F3F1EC")
_x, y_rand, y_split = place(randomised, x_mid - randomised[1] / 2,
                            y - side_gap)
connect((x_mid, y), (x_mid, y_rand))
connect((x_mid, y - side_gap / 2),
        (col_x[1] + col_w - exclusion[1], y - side_gap / 2))
for x_left, *column in zip(col_x, allocation, follow_up, analysis):
    y_from = y_split
    for k, (item, stage) in enumerate(zip(column, ("Allocation",
                                                   "Follow-up", "Analysis"))):
        cx, top, bottom = place(item, x_left, y_from - gap, col_w, row_h[k])
        if k == 0:                      # fan out from the randomised box
            connect((x_mid, y_split), (x_mid, y_split - gap / 2),
                    (cx, y_split - gap / 2), (cx, top))
        else:
            connect((cx, y_from), (cx, top))
        stage_y[stage] = (top + bottom) / 2
        y_from = bottom
for stage, y_stage in stage_y.items():
    ax.text(STAGE_X, y_stage, stage, rotation=90, ha="center", va="center",
            fontsize=ms.FS_TICK, fontweight="bold", color=ms.GREY_DARK)

# measured after drawing: every text inside its own box, boxes on the page
fig.canvas.draw()
for text, patch in boxes:
    inner, outer = text.get_window_extent(), patch.get_window_extent()
    assert (outer.x0 < inner.x0 and inner.x1 < outer.x1
            and outer.y0 < inner.y0 and inner.y1 < outer.y1), text.get_text()
    assert fig.bbox.contains(outer.x0, outer.y0)
    assert fig.bbox.contains(outer.x1, outer.y1)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig135_consort_flow.{ext}")
print("fig135_consort_flow: saved png + pdf")
