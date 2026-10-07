"""Fig. 109 - ROC beside precision-recall, with bootstrap intervals (120 mm).

Extends fig013: at 20% prevalence the ROC curve flatters every model, so
the precision-recall curve sits beside it on equal square panels. Each
AUC carries a 95% percentile-bootstrap interval (1,000 resamples of the
whole test set), and both panels are labelled directly in the curves'
own colours, in the corner the curves leave empty. The self-check is the
Mann-Whitney identity: the rank-sum AUC must equal the trapezoid area
under the plotted ROC curve.

Statistics: held-out test set, n = 2,000 (400 positive); dashed lines,
chance (the diagonal; the prevalence). All scores are simulated.
"""

import numpy as np
from scipy import stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(909)
N_TEST, PREVALENCE, N_BOOT = 2000, 0.20, 1000
MODELS = {"Clinical": (1.0, ms.GREY), "Imaging": (1.4, ms.GREEN),
          "Combined": (2.0, ms.VERMILLION)}       # (class separation, colour)

# ------------------------------------------------------------- DATA ----
label = np.arange(N_TEST) < N_TEST * PREVALENCE
scores = {name: rng.normal(0, 1, N_TEST) + separation * label
          for name, (separation, _colour) in MODELS.items()}


# ---------------------------------------------------------- METRICS ----
def auc_rank(y, score):
    """AUC as the Mann-Whitney U statistic (ties get mid-ranks)."""
    n_pos = y.sum()
    rank_sum = stats.rankdata(score)[y].sum()
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * (y.size - n_pos))


def curves(y, score):
    """ROC and precision-recall coordinates, and the average precision."""
    order = np.argsort(-score, kind="stable")
    hits = np.cumsum(y[order])
    misses = np.cumsum(~y[order])
    tpr, fpr = hits / y.sum(), misses / (~y).sum()
    precision = hits / (hits + misses)
    ap = float(np.sum(np.diff(tpr, prepend=0.0) * precision))
    return (np.concatenate([[0.0], fpr]), np.concatenate([[0.0], tpr]),
            np.concatenate([[1.0], precision]), ap)


boot_rng = np.random.default_rng(99)
results = {}
for name in MODELS:
    fpr, tpr, precision, ap = curves(label, scores[name])
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):                 # resample the whole test set
        pick = boot_rng.integers(0, N_TEST, N_TEST)
        boot[b] = auc_rank(label[pick], scores[name][pick])
    results[name] = dict(fpr=fpr, tpr=tpr, precision=precision, ap=ap,
                         auc=auc_rank(label, scores[name]),
                         ci=np.percentile(boot, [2.5, 97.5]))

# ------------------------------------------------------- SELF-CHECK ---
for name, r in results.items():
    mid_tpr = (r["tpr"][1:] + r["tpr"][:-1]) / 2
    area = float(np.sum(np.diff(r["fpr"]) * mid_tpr))
    assert abs(area - r["auc"]) < 1e-9, (name, area, r["auc"])
    # at recall 1 every case is called positive: precision = prevalence
    assert abs(r["precision"][-1] - PREVALENCE) < 1e-12
print("fig109: self-check passed (rank-sum AUC = trapezoid area: "
      + ", ".join(f"{n} {r['auc']:.4f}" for n, r in results.items()) + ")")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 60)
gs = ms.grid(fig, 1, 2, left=13, right=4, top=6, bottom=10.5, wspace=16)
ax_roc, ax_pr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

dash = dict(color=ms.GREY_LIGHT, lw=0.7, ls=(0, (3, 2)))
ax_roc.plot([0, 1], [0, 1], **dash)
ax_pr.plot([0, 1], [PREVALENCE, PREVALENCE], **dash)
for k, (name, (_separation, colour)) in enumerate(MODELS.items()):
    r = results[name]
    hero = name == "Combined"
    weight = "bold" if hero else "normal"
    ax_roc.plot(r["fpr"], r["tpr"], color=colour, lw=1.4 if hero else 1.0)
    ax_pr.plot(r["tpr"], r["precision"], color=colour, lw=1.4 if hero else 1.0)
    # direct labels in each panel's empty corner, in the curve's colour
    ax_roc.text(0.98, 0.05 + 0.06 * k, f"{name}  {r['auc']:.2f} "
                f"({r['ci'][0]:.2f}\u2013{r['ci'][1]:.2f})",
                transform=ax_roc.transAxes, color=colour, ha="right",
                va="center", fontsize=ms.FS_TICK, fontweight=weight)
    ax_pr.text(0.03, 0.05 + 0.06 * k, f"{name}  AP = {r['ap']:.2f}",
               transform=ax_pr.transAxes, color=colour, va="center",
               fontsize=ms.FS_TICK, fontweight=weight)
ax_roc.text(0.98, 0.05 + 0.06 * 3, "AUC (95% CI)", transform=ax_roc.transAxes,
            ha="right", va="center", fontsize=ms.FS_TICK)

for ax in (ax_roc, ax_pr):
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.spines["left"].set_bounds(0, 1)
    ax.spines["bottom"].set_bounds(0, 1)
ax_roc.set_xlabel("False-positive rate (1 − specificity)")
ax_roc.set_ylabel("True-positive rate (sensitivity)")
ax_pr.set_xlabel("Recall (sensitivity)")
ax_pr.set_ylabel("Precision (positive predictive value)")
ax_pr.text(0.99, PREVALENCE - 0.025, "Chance", ha="right", va="top",
           fontsize=ms.FS_SMALL, color=ms.GREY)
ms.panel_label(ax_roc, "a")
ms.panel_label(ax_pr, "b")

ms.assert_aligned([ax_roc, ax_pr])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig109_roc_pr_bootstrap.{ext}")
print("fig109_roc_pr_bootstrap: saved png + pdf")
