"""Fig. 138 - Specification curve of one estimate (double column, 183 mm).

One coefficient is rarely one number: it depends on defensible choices
about covariates, outliers, the outcome scale and the analysis sample.
Every combination of five such choices is fitted to one simulated
cohort. Panel a ranks the 72 exposure coefficients with 95% confidence
intervals, filled where the interval excludes zero, with the median
marked and the main (pre-registered) specification highlighted; panel b
shows which options produced each estimate, so the baseline covariate,
not chance, is seen to separate the large estimates from the small. The
self-check is that the number of specifications is the product of the
option counts, that the hand-coded OLS slope and standard error equal
scipy.stats.linregress for the unadjusted specification and (through
the Frisch–Waugh residual regression and np.linalg.lstsq) for the main
one, and that the plotted estimates are sorted.

Statistics: n = 400 participants (fewer after per-protocol or outlier
exclusion); points, OLS coefficient of the standardised exposure on the
standardised outcome; whiskers, 95% t interval; no multiplicity
adjustment. All data are simulated.
"""

from itertools import product

import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(138)
N = 400
CHOICES = {
    "Covariates": ["None", "Age, sex", "Age, sex, baseline"],
    "Site": ["Ignored", "Fixed effects"],
    "Outliers": ["Kept", "Excluded (|z| > 2.5)", "Winsorised (±2.5 s.d.)"],
    "Outcome": ["Raw scale", "Log scale"],
    "Sample": ["All participants", "Per protocol"],
}
MAIN = ("Age, sex, baseline", "Fixed effects", "Kept", "Log scale",
        "All participants")

# ------------------------------------------------------------- DATA ----
# The baseline score drives both exposure and outcome (a confounder);
# the outcome is log-normal, so the raw scale has a long right tail.
site = rng.integers(0, 4, N)
age, baseline = rng.normal(size=N), rng.normal(size=N)
sex = rng.random(N) < 0.5
raw = 0.55 * baseline + 0.25 * age + 0.20 * (site - 1.5) + rng.normal(size=N)
exposure = (raw - raw.mean()) / raw.std(ddof=1)
outcome = np.exp(0.5 * (0.24 * exposure + 0.50 * baseline + 0.20 * age
                        + 0.30 * sex + 0.15 * (site - 1.5)
                        + rng.normal(size=N)))
per_protocol = rng.random(N) < 0.85
COVARIATES = {"None": [], "Age, sex": [age, sex],
              "Age, sex, baseline": [age, sex, baseline]}


# -------------------------------------------------------- ESTIMATORS ---
def prepare(covariates, site_rule, outliers, scale, sample):
    """Design matrix (intercept, exposure, adjusters) and outcome z-score."""
    keep = np.ones(N, bool)
    if sample == "Per protocol":
        keep &= per_protocol
    y = np.log(outcome) if scale == "Log scale" else outcome
    z = (y - y[keep].mean()) / y[keep].std(ddof=1)
    if outliers.startswith("Excluded"):
        keep &= np.abs(z) <= 2.5
    elif outliers.startswith("Winsorised"):
        z = np.clip(z, -2.5, 2.5)
    columns = [np.ones(N), exposure, *COVARIATES[covariates]]
    if site_rule == "Fixed effects":
        columns += [site == k for k in (1, 2, 3)]
    return np.column_stack(columns)[keep].astype(float), z[keep]


def ols(design, y):
    """Coefficients, their standard errors and the residual d.f."""
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    residual = y - design @ beta
    dof = y.size - design.shape[1]
    return beta, np.sqrt(residual @ residual / dof * np.diag(xtx_inv)), dof


specs = list(product(*CHOICES.values()))
fits = [ols(*prepare(*spec)) for spec in specs]
estimate = np.array([beta[1] for beta, _se, _dof in fits])
half = np.array([stats.t.ppf(0.975, dof) * se[1] for _beta, se, dof in fits])
order = np.argsort(estimate, kind="stable")
estimate, half = estimate[order], half[order]
specs = [specs[k] for k in order]
excludes_zero = np.abs(estimate) > half
main = specs.index(MAIN)
median = float(np.median(estimate))

# ------------------------------------------------------- SELF-CHECK ---
assert len(specs) == np.prod([len(v) for v in CHOICES.values()])
assert len(set(specs)) == len(specs) and np.all(np.diff(estimate) >= 0)
design, z = prepare("None", "Ignored", "Kept", "Raw scale", "All participants")
beta, se, _dof = ols(design, z)
simple = stats.linregress(design[:, 1], z)
assert np.allclose([beta[1], se[1]], [simple.slope, simple.stderr], rtol=1e-10)
design, z = prepare(*MAIN)
beta, se, dof = ols(design, z)
assert np.allclose(beta, np.linalg.lstsq(design, z, rcond=None)[0], atol=1e-10)
others = np.delete(design, 1, axis=1)            # Frisch–Waugh: partial out
partial = [v - others @ np.linalg.lstsq(others, v, rcond=None)[0]
           for v in (design[:, 1], z)]
fwl = stats.linregress(*partial)                 # its d.f. are n − 2
assert np.allclose([beta[1], se[1]],
                   [fwl.slope, fwl.stderr * np.sqrt((z.size - 2) / dof)],
                   rtol=1e-9)
sizes = " x ".join(str(len(options)) for options in CHOICES.values())
print(f"fig138: self-check passed ({len(specs)} specifications = {sizes}; "
      f"OLS slope and s.e. match linregress and lstsq; "
      f"estimates sorted, {estimate[0]:.3f} to {estimate[-1]:.3f}, median "
      f"{median:.3f}; main {estimate[main]:.3f} ± {half[main]:.3f}; "
      f"{excludes_zero.sum()} of {len(specs)} CIs exclude zero)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 118)
gs = ms.grid(fig, 2, 1, left=43, right=21, top=6, bottom=10, hspace=5,
             height_ratios=[1, 1.12])
ax = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1], sharex=ax)
rank = np.arange(len(specs))

# a, ranked estimates; fill as well as colour says whether zero is excluded
for axis in (ax, ax_b):                          # main specification column
    axis.axvspan(main - 0.5, main + 0.5, color=ms.VERMILLION, alpha=0.14,
                 lw=0, zorder=0)
ax.axhline(0, color=ms.INK, lw=0.6, zorder=1)
ax.axhline(median, color=ms.GREY_DARK, lw=0.6, ls=(0, (4, 2.5)), zorder=1)
ax.text(1.012, median, f"Median {median:.2f}",
        transform=ax.get_yaxis_transform(), va="center",
        fontsize=ms.FS_TICK, color=ms.GREY_DARK)
STYLES = {True: dict(color=ms.BLUE, mfc=ms.BLUE, label="excludes"),
          False: dict(color=ms.GREY, mfc="white", label="includes")}
for flag, style in STYLES.items():
    member = (excludes_zero == flag) & (rank != main)
    ax.vlines(rank[member], (estimate - half)[member],
              (estimate + half)[member], color=style["color"], lw=0.7)
    ax.plot(rank[member], estimate[member], ls="none", marker="o",
            markersize=2.6, markeredgewidth=0.6, zorder=3,
            markeredgecolor=style["color"], markerfacecolor=style["mfc"])
ax.vlines(main, estimate[main] - half[main], estimate[main] + half[main],
          color=ms.VERMILLION, lw=1.1)
ax.plot(main, estimate[main], marker="D", markersize=3.6, zorder=4,
        color=ms.VERMILLION, markeredgewidth=0)
ax.text(main + 1.0, 0.985, (
        f"Main specification: {estimate[main]:.2f} (95% CI "
        f"{estimate[main] - half[main]:.2f} to "
        f"{estimate[main] + half[main]:.2f})").replace("-", "−"),
        transform=ax.get_xaxis_transform(), va="top", fontsize=ms.FS_TICK,
        color=ms.VERMILLION)
ax.legend(handles=[Line2D([], [], marker="o", markersize=2.6, lw=0.7,
                          markeredgewidth=0.6, color=style["color"],
                          markerfacecolor=style["mfc"],
                          label=f"95% CI {style['label']} 0 "
                                f"(n = {(excludes_zero == flag).sum()})")
                   for flag, style in STYLES.items()],
          loc="lower right", handlelength=1.2)
ax.set_xlim(-1.2, len(specs) + 0.2)
ax.set_ylim(-0.12, 0.56)
ax.spines["bottom"].set_visible(False)
ax.tick_params(axis="x", bottom=False, labelbottom=False)
ax.set_ylabel("Exposure coefficient\n(s.d. of outcome per s.d.)")

# b, which option each column used: a header line per family, then rows
y, y_ticks = 0.0, []
for family, (name, options) in enumerate(CHOICES.items()):
    ax_b.text(-0.012, y, name, transform=ax_b.get_yaxis_transform(),
              ha="right", va="center", fontweight="bold", fontsize=ms.FS_TICK)
    for option in options:
        y += 1
        used = np.array([spec[family] == option for spec in specs])
        ax_b.plot([-0.7, len(specs) - 0.3], [y, y], color=ms.GREY_LIGHT,
                  lw=0.4, zorder=1)
        ax_b.scatter(rank[used], np.full(used.sum(), y), s=5.5, linewidths=0,
                     color=np.where(rank[used] == main, ms.VERMILLION,
                                    ms.GREY_DARK), zorder=3)
        y_ticks.append(y)
    y += 1.5
ax_b.set_yticks(y_ticks, [o for options in CHOICES.values() for o in options])
ax_b.set_ylim(y - 0.8, -0.7)
ax_b.tick_params(axis="y", length=0, pad=4)
ax_b.spines["left"].set_visible(False)
ax_b.set_xticks([0, 17, 35, 53, 71], [1, 18, 36, 54, 72])
ax_b.set_xlabel("Specification, ranked by estimate")

for axis, letter in ((ax, "a"), (ax_b, "b")):
    ms.panel_label(axis, letter, dx_pt=-112)
ms.assert_aligned([ax, ax_b], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig138_specification_curve.{ext}")
print("fig138_specification_curve: saved png + pdf")
