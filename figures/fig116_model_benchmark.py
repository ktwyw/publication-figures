"""Fig. 116 - Four-panel model benchmark as one argument (double column).

Joins fig034 (training curves), fig025 (calibration) and fig033 (confusion
matrix), which stand alone in the library, into one 2 x 2 argument on a
millimetre grid with one role per panel: a, main effect (validation
accuracy during training); b, mechanism (component ablation, every seed
as a dot, the drop versus the full model written at the right);
c, reliability diagram with the expected calibration error (ECE);
d, row-normalised confusion matrix of the proposed model. Panels a-c use
the same uncertainty definition. The self-check is that every confusion
row sums to 1, the proposed model has the smaller ECE, the full model
beats every ablation, and the four plot areas line up within 1.5 pt.

Statistics: n = 5 random seeds per configuration; a-c, mean ± s.d.
across seeds (band or error bar); d, mean over seeds, no interval; no
hypothesis test. All data are simulated.
"""

import numpy as np

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(1616)
N_SEEDS, EPOCHS = 5, np.arange(1, 61)
METHODS = {"Baseline A": (84.6, 9.0, ms.GREY),       # (plateau, tau, colour)
           "Baseline B": (88.1, 11.0, ms.GREEN),
           "Proposed": (92.0, 8.0, ms.VERMILLION)}
ABLATIONS = {"Full model": 91.6, "w/o pretraining": 88.9,
             "w/o augmentation": 90.2, "w/o attention": 89.7,
             "w/o all three": 85.0}                  # test accuracy (%)
ROW_COLOURS = [ms.VERMILLION] + [ms.GREY_DARK] * 3 + [ms.GREY]
CALIBRATED = {"Baseline B": (0.115, "s"), "Proposed": (0.025, "o")}
CLASSES = [f"Class {k}" for k in range(1, 6)]

# ------------------------------------------------------------- DATA ----
# a: saturating learning curves, array (method, seed, epoch)
curves = np.empty((len(METHODS), N_SEEDS, EPOCHS.size))
for m, (plateau, tau, _colour) in enumerate(METHODS.values()):
    for seed in range(N_SEEDS):
        offset = rng.normal(0, 1.3)
        curves[m, seed] = ((plateau + offset)
                           * (1 - 0.62 * np.exp(-EPOCHS / tau))
                           + rng.normal(0, 0.35, EPOCHS.size))

# b: test accuracy of each configuration, array (configuration, seed)
ablation = np.array([[rng.normal(centre, 0.45) for _seed in range(N_SEEDS)]
                     for centre in ABLATIONS.values()])

# c: observed accuracy per confidence bin, array (method, seed, bin)
conf_bins = np.linspace(0.55, 0.95, 5)
calibration = np.array(
    [[conf_bins - gap * (conf_bins - 0.45) / 0.5
      + rng.normal(0, 0.012, conf_bins.size) for _seed in range(N_SEEDS)]
     for gap, _marker in CALIBRATED.values()])

# d: row-normalised confusion matrix of the proposed model, seed mean
confusion = np.zeros((5, 5))
for _seed in range(N_SEEDS):
    matrix = rng.dirichlet(np.ones(5) * 0.6, 5) * 0.085
    matrix[1, 2] += 0.06
    matrix[2, 1] += 0.045
    np.fill_diagonal(matrix, 0)
    np.fill_diagonal(matrix, 1 - matrix.sum(axis=1))
    confusion += matrix / N_SEEDS


# ---------------------------------------------------------- SUMMARY ----
def mean_sd(values, axis):
    """The one uncertainty definition used in panels a-c."""
    return values.mean(axis=axis), values.std(axis=axis, ddof=1)


curve_mean, curve_sd = mean_sd(curves, 1)
abl_mean, abl_sd = mean_sd(ablation, 1)
cal_mean, cal_sd = mean_sd(calibration, 1)
drop = abl_mean - abl_mean[0]
ece = dict(zip(CALIBRATED, np.abs(cal_mean - conf_bins).mean(axis=1)))

# ------------------------------------------------------- SELF-CHECK ---
assert np.allclose(confusion.sum(axis=1), 1, atol=1e-12)
assert confusion.min() >= 0
assert ece["Proposed"] < ece["Baseline B"], ece
assert np.all(drop[1:] < 0), drop
print(f"fig116: self-check passed (confusion rows sum to 1; ECE "
      f"{ece['Proposed']:.3f} proposed < {ece['Baseline B']:.3f} baseline; "
      f"full model {abl_mean[0]:.1f}% > best ablation "
      f"{abl_mean[1:].max():.1f}%)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 128)
gs = ms.grid(fig, 2, 2, left=28, right=6, top=7, bottom=12, wspace=40,
             hspace=19)
ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

# a, validation accuracy during training, labelled directly
label_y = {"Baseline A": 83.6, "Baseline B": 88.0, "Proposed": 92.4}
for m, (method, (_plateau, _tau, colour)) in enumerate(METHODS.items()):
    hero = method == "Proposed"
    ax_a.fill_between(EPOCHS, curve_mean[m] - curve_sd[m],
                      curve_mean[m] + curve_sd[m], color=colour, alpha=0.2,
                      lw=0)
    ax_a.plot(EPOCHS, curve_mean[m], color=colour, lw=1.4 if hero else 1.0)
    ax_a.text(61.5, label_y[method], method, color=colour, va="center",
              fontsize=ms.FS_TICK, fontweight="bold" if hero else "normal")
ax_a.text(0.97, 0.06, "Mean ± s.d., n = 5 seeds", transform=ax_a.transAxes,
          ha="right", fontsize=ms.FS_TICK, color=ms.GREY_DARK)
ax_a.set_xlim(0, 61)
ax_a.set_ylim(30, 96)
ax_a.set_xticks(np.arange(0, 61, 10))
ax_a.set_xlabel("Training epoch")
ax_a.set_ylabel("Validation accuracy (%)")

# b, ablation: seed-level dots with mean ± s.d. (no bars, because the
# axis does not start at zero)
jitter_rng = np.random.default_rng(16)      # vertical jitter, display only
for i, colour in enumerate(ROW_COLOURS):
    ax_b.plot([80, 92.4], [i, i], color=ms.GREY_LIGHT, lw=0.4, zorder=1)
    ax_b.scatter(ablation[i], i + jitter_rng.uniform(-0.2, 0.2, N_SEEDS),
                 s=7, color=colour, alpha=0.45, linewidths=0, zorder=3)
    ax_b.errorbar(abl_mean[i], i, xerr=abl_sd[i], fmt="D", color=colour,
                  markersize=3.6, markeredgewidth=0, elinewidth=0.9,
                  capsize=2.2, capthick=0.9, zorder=4)
    if i:
        ax_b.text(93.9, i, f"{drop[i]:.1f}".replace("-", "−"), ha="right",
                  va="center", fontsize=ms.FS_TICK, color=ms.GREY_DARK)
ax_b.text(93.9, -0.62, "Δ vs full", ha="right", va="center",
          fontsize=ms.FS_TICK, color=ms.GREY_DARK)
ax_b.set_yticks(range(len(ABLATIONS)), list(ABLATIONS))
ax_b.set_ylim(len(ABLATIONS) - 0.45, -0.95)
ax_b.set_xlim(80, 94)
ax_b.set_xticks(np.arange(80, 95, 2))
ax_b.tick_params(axis="y", length=0, pad=3)
ax_b.set_xlabel("Test accuracy (%)")

# c, reliability diagram
ax_c.plot([0.5, 1.0], [0.5, 1.0], color=ms.GREY, lw=0.7, ls=(0, (3, 2)),
          label="Perfect calibration")
for k, (method, (_gap, marker)) in enumerate(CALIBRATED.items()):
    hero = method == "Proposed"
    ax_c.errorbar(conf_bins, cal_mean[k], yerr=cal_sd[k],
                  color=METHODS[method][2], marker=marker, markersize=3.2,
                  markeredgewidth=0, lw=1.4 if hero else 1.0, elinewidth=0.6,
                  capsize=1.8, capthick=0.6,
                  label=f"{method} (ECE {ece[method]:.3f})")
ax_c.set_xlim(0.5, 1.0)
ax_c.set_ylim(0.5, 1.0)
ax_c.set_xticks(np.linspace(0.5, 1.0, 6))
ax_c.set_yticks(np.linspace(0.5, 1.0, 6))
ax_c.set_xlabel("Mean predicted confidence")
ax_c.set_ylabel("Observed accuracy")
ax_c.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))

# d, confusion matrix of the proposed model
ax_d.imshow(confusion * 100, cmap="Blues", vmin=0, vmax=100, aspect="auto")
for (i, j), value in np.ndenumerate(confusion * 100):
    ax_d.text(j, i, f"{value:.1f}", ha="center", va="center",
              fontsize=ms.FS_TICK, color="white" if value > 55 else ms.INK)
ax_d.set_xticks(range(5), CLASSES)
ax_d.set_yticks(range(5), CLASSES)
ax_d.tick_params(length=0, pad=3)
ax_d.set_xlabel("Predicted class")
ax_d.set_ylabel("True class")
for spine in ax_d.spines.values():
    spine.set_visible(False)
ax_d.set_title("Proposed model, row-normalised (%)", fontsize=ms.FS_TICK,
               loc="right", pad=3)

for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")):
    ms.panel_label(ax, letter, dx_pt=-52 if letter in "bd" else -26)

ms.assert_aligned([ax_a, ax_b])
ms.assert_aligned([ax_c, ax_d])
ms.assert_aligned([ax_a, ax_c], edges=("left", "right"))
ms.assert_aligned([ax_b, ax_d], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig116_model_benchmark.{ext}")
print("fig116_model_benchmark: saved png + pdf")
