"""Fig. 134 - Waterfall of best tumour response (1.5 column, 120 mm).

Best percentage change in the sum of target-lesion diameters, one bar
per patient from worst to best, with the two thresholds that define the
categories drawn as dashed lines (+20%, progression; −30%, response).
Category is derived from the value, never typed in; complete responses
carry a white diamond as well as a colour, bars above +100% are cut at
the axis and flagged, and a row of markers under the bars shows a
binary biomarker. The self-check is that each bar's category agrees with the
thresholds and that the Clopper–Pearson interval coded from beta
quantiles equals scipy.stats.binomtest(...).proportion_ci("exact").

Statistics: n = 40 patients; objective response rate = (CR + PR) / n
with the exact (Clopper–Pearson) 95% CI; response rates by biomarker
compared with a two-sided Fisher exact test. All data are simulated.
"""

import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(134)
N = 40
PD_LINE, PR_LINE, CAP = 20.0, -30.0, 100.0       # % change thresholds
COLOUR = {"PD": ms.VERMILLION, "SD": ms.GREY, "PR": ms.SKY, "CR": ms.BLUE}
CR_MARK = dict(marker="D", color="white", markersize=2.3, ls="none",
               markeredgewidth=0)                # second cue besides colour
NAMES = {"PD": "Progressive disease", "SD": "Stable disease",
         "PR": "Partial response", "CR": "Complete response"}

# ------------------------------------------------------------- DATA ----
# Biomarker-positive tumours shrink more; −100% is the floor (no lesion).
biomarker = rng.random(N) < 0.45
change = np.where(biomarker, rng.normal(-55, 38, N), rng.normal(4, 42, N))
change[rng.choice(np.flatnonzero(~biomarker), 2, replace=False)] += 115
change = np.round(np.maximum(change, -100.0), 1)


# -------------------------------------------------------- ESTIMATORS ---
def categorise(pct):
    """Best-response category from the percentage change alone."""
    return np.select([pct >= PD_LINE, pct <= -100, pct <= PR_LINE],
                     ["PD", "CR", "PR"], default="SD")


def clopper_pearson(k, n, level=0.95):
    """Exact binomial interval from beta quantiles."""
    a = (1 - level) / 2
    low = stats.beta.ppf(a, k, n - k + 1) if k > 0 else 0.0
    high = stats.beta.ppf(1 - a, k + 1, n - k) if k < n else 1.0
    return float(low), float(high)


order = np.argsort(-change, kind="stable")       # worst (growth) first
change, biomarker = change[order], biomarker[order]
category = categorise(change)
responder = np.isin(category, ["PR", "CR"])
n_resp = int(responder.sum())
orr, (orr_low, orr_high) = n_resp / N, clopper_pearson(n_resp, N)
by_marker = [(int(responder[m].sum()), int(m.sum()))
             for m in (biomarker, ~biomarker)]   # (responders, patients)
p_marker = stats.fisher_exact([[k, n - k] for k, n in by_marker]).pvalue

# ------------------------------------------------------- SELF-CHECK ---
assert np.all(np.diff(change) <= 0)
assert np.all(change[category == "PD"] >= PD_LINE)
assert np.all((change[category == "SD"] < PD_LINE)
              & (change[category == "SD"] > PR_LINE))
assert np.all((change[category == "PR"] <= PR_LINE)
              & (change[category == "PR"] > -100))
assert np.all(change[category == "CR"] == -100)
reference = stats.binomtest(n_resp, N).proportion_ci(0.95, method="exact")
assert np.allclose([orr_low, orr_high], [reference.low, reference.high],
                   rtol=0, atol=1e-12)
counts = {name: int((category == name).sum()) for name in COLOUR}
print(f"fig134: self-check passed (categories match thresholds: {counts}; "
      f"ORR {n_resp}/{N} = {orr:.1%}, Clopper-Pearson 95% CI "
      f"{orr_low:.1%}-{orr_high:.1%} equals scipy exact)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 78)
gs = ms.grid(fig, 2, 1, left=15, right=22, top=5, bottom=8, hspace=2.5,
             height_ratios=[15, 1])
ax = fig.add_subplot(gs[0])
ax_marker = fig.add_subplot(gs[1], sharex=ax)

x = np.arange(N)
for name, colour in COLOUR.items():
    member = category == name
    ax.bar(x[member], np.minimum(change[member], CAP), width=0.78, lw=0,
           color=colour, zorder=2)
is_cr = category == "CR"
ax.plot(x[is_cr], np.full(is_cr.sum(), -7.0), zorder=3, **CR_MARK)
ax.axhline(0, color=ms.INK, lw=0.6, zorder=3)
for level, label in ((PD_LINE, "+20%\nprogression"),
                     (PR_LINE, "−30%\nresponse")):
    ax.axhline(level, color=ms.GREY_DARK, lw=0.6, ls=(0, (4, 2.5)), zorder=1)
    ax.text(1.015, level, label, transform=ax.get_yaxis_transform(),
            va="center", fontsize=ms.FS_TICK, color=ms.GREY_DARK,
            linespacing=1.1)

# bars cut at the axis limit: an arrow head above each, true values beside
cut = np.flatnonzero(change > CAP)
ax.scatter(cut, np.full(cut.size, CAP + 4.5), marker="^", s=9, color=ms.INK,
           linewidths=0, clip_on=False, zorder=4)
ax.text(cut.max() + 0.9, CAP + 4.5, "cut at +100%; actual "
        + ", ".join(f"+{v:.0f}%" for v in change[cut]), va="center",
        fontsize=ms.FS_TICK, color=ms.GREY_DARK)

ax.set_xlim(-0.8, N - 0.2)
ax.set_ylim(-105, 104)
ax.set_yticks(np.arange(-100, 101, 20))
ax.spines["left"].set_bounds(-100, 100)
ax.spines["bottom"].set_visible(False)
ax.tick_params(axis="x", bottom=False, labelbottom=False)
ax.set_ylabel("Best change from baseline in\nsum of target lesions (%)")

handles = [Patch(facecolor=colour, lw=0) for colour in COLOUR.values()]
handles[-1] = (handles[-1], Line2D([], [], **CR_MARK))
ax.legend(handles, [f"{NAMES[name]} ({counts[name]})" for name in COLOUR],
          handler_map={tuple: HandlerTuple(ndivide=1)}, loc="upper right",
          bbox_to_anchor=(1.0, 0.88), handlelength=1.3, handleheight=0.8)
ax.text(0.03, 0.06, f"Objective response rate {n_resp}/{N} = "
        f"{orr * 100:.1f}%\n(exact 95% CI {orr_low * 100:.1f}–"
        f"{orr_high * 100:.1f}%)\nBiomarker +: {by_marker[0][0]}/"
        f"{by_marker[0][1]};  −: {by_marker[1][0]}/{by_marker[1][1]};  "
        f"Fisher {ms.format_p(p_marker)}", transform=ax.transAxes,
        va="bottom", fontsize=ms.FS_TICK, linespacing=1.35)

# biomarker strip: filled = positive, open = negative (shape of the fill,
# not colour, carries it)
ax_marker.scatter(x[biomarker], np.zeros(biomarker.sum()), marker="o", s=9,
                  color=ms.INK, linewidths=0, label="Positive")
ax_marker.scatter(x[~biomarker], np.zeros((~biomarker).sum()), marker="o",
                  s=7, facecolor="none", edgecolor=ms.GREY, linewidths=0.5,
                  label="Negative")
ax_marker.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
                 handletextpad=0.1, labelspacing=0.15, borderaxespad=0.2)
ax_marker.set_ylim(-1, 1)
ax_marker.axis("off")
ax_marker.text(-0.012, 0.5, "Biomarker", transform=ax_marker.transAxes,
               ha="right", va="center", fontsize=ms.FS_TICK)
ax_marker.text(0.5, -0.7, f"Patients (n = {N}), ordered by best change",
               transform=ax_marker.transAxes, ha="center", va="top",
               fontsize=ms.FS_BODY)

ms.assert_aligned([ax, ax_marker], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig134_response_waterfall.{ext}")
print("fig134_response_waterfall: saved png + pdf")
