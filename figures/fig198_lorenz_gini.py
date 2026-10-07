"""Fig. 198 - Lorenz curves and the Gini coefficient (single column, 89 mm).

How to draw income inequality so that the summary number can be read
off the picture: Lorenz curves of three populations on a square plot of
cumulative shares, each named beside its line and tabulated with its
Gini coefficient, the closed form for log-normal incomes dashed on top
of each sample curve, the area whose double is the Gini coefficient
shaded for one curve, and the share of the top 10% read off the most
unequal curve. The self-check is that the sorted-sample Gini formula
equals 1 − 2∫L dp by trapezoid within 1e-3 and the log-normal value
2Φ(σ/√2) − 1 within 0.02, and that every curve is convex and runs from
(0, 0) to (1, 1).

Data: n = 5,000 log-normal incomes per population, σ = 0.4, 0.8 and 1.2
(placeholder populations A-C); solid lines, sample Lorenz curves;
dashed, L(p) = Φ(Φ⁻¹(p) − σ). No interval shown. All data are simulated.
"""

import numpy as np
from scipy import stats

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(198)

# ------------------------------------------------------------- DATA ----
N = 5000
SIGMA = {"A": 0.4, "B": 0.8, "C": 1.2}       # log-income dispersions
COLOUR = {"A": ms.GREY, "B": ms.BLUE, "C": ms.VERMILLION}
SHADED, TOP = "C", 0.10                      # shaded curve; top share shown
incomes = {name: np.exp(s * rng.standard_normal(N))
           for name, s in SIGMA.items()}


# -------------------------------------------------------- ESTIMATORS ---
def lorenz(x):
    """Cumulative population and income shares, starting at (0, 0)."""
    x = np.sort(x)
    return (np.arange(x.size + 1) / x.size,
            np.concatenate([[0.0], np.cumsum(x) / x.sum()]))


def gini_sorted(x):
    """Sorted-sample formula: G = 2 Σ i·x(i) / (n Σ x) − (n + 1)/n."""
    x = np.sort(x)
    rank = np.arange(1, x.size + 1)
    return 2 * (rank * x).sum() / (x.size * x.sum()) - (x.size + 1) / x.size


def lorenz_lognormal(p, sigma):
    return stats.norm.cdf(stats.norm.ppf(p) - sigma)


curves = {name: lorenz(x) for name, x in incomes.items()}
gini = {name: gini_sorted(x) for name, x in incomes.items()}
gini_exact = {name: 2 * stats.norm.cdf(s / np.sqrt(2)) - 1
              for name, s in SIGMA.items()}
p_fine = np.linspace(0, 1, 801)
p_top, share_c = curves[SHADED]
top_share = 1 - share_c[round((1 - TOP) * N)]

# ------------------------------------------------------- SELF-CHECK ---
for name, (p, share) in curves.items():
    area = ((share[1:] + share[:-1]) / 2 * np.diff(p)).sum()   # trapezoid
    assert abs(gini[name] - (1 - 2 * area)) < 1e-3, name
    assert abs(gini[name] - gini_exact[name]) < 0.02, name
    assert share[0] == 0 and abs(share[-1] - 1) < 1e-12
    assert np.all(np.diff(share, 2) > -1e-12), name            # convex
    exact = lorenz_lognormal(p_fine, SIGMA[name])
    assert exact[0] == 0 and exact[-1] == 1
    assert np.all(np.diff(exact, 2) > -1e-12), name
assert gini["A"] < gini["B"] < gini["C"]
print("fig198: self-check passed (Gini sample vs log-normal: "
      + ", ".join(f"{name} {gini[name]:.3f} vs {gini_exact[name]:.3f}"
                  for name in SIGMA)
      + f"; sorted formula = 1 - 2*area; all curves convex; top "
      f"{TOP:.0%} of {SHADED} hold {top_share:.1%})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 78)
SIDE = 62.0                                  # square plot area, mm
ax = ms.axes(fig, 14, 11, SIDE, SIDE)

ax.fill_between(p_top, share_c, p_top, color=COLOUR[SHADED], alpha=0.10,
                lw=0)
ax.plot([0, 1], [0, 1], color=ms.GREY, lw=0.8)
for name, (p, share) in curves.items():
    ax.plot(p, share, color=COLOUR[name], lw=1.5)
    ax.plot(p_fine, lorenz_lognormal(p_fine, SIGMA[name]), color=ms.INK,
            lw=0.5, ls=(0, (3, 2.5)))


# letters sit in the gaps under each curve; rotated text along a curve
# would put its bounding box across the neighbouring lines
for name, p0 in (("A", 0.60), ("B", 0.62), ("C", 0.64)):
    ax.annotate(name, xy=(p0, np.interp(p0, *curves[name])),
                xytext=(4.5, -4.5), textcoords="offset points",
                ha="center", va="center", color=COLOUR[name],
                fontweight="bold")
ax.text(0.03, 0.365, "Line of equality", ha="left", va="bottom",
        fontsize=ms.FS_TICK, color=ms.GREY_DARK)

# aligned table of the three populations, then how to read the figure
COLS = (0.04, 0.17, 0.30)                    # name, σ, Gini (axes fraction)
rows = [("", "σ", "Gini, sample (log-normal)", ms.INK)]
rows += [(name, f"{SIGMA[name]:.1f}",
          f"{gini[name]:.3f} ({gini_exact[name]:.3f})", COLOUR[name])
         for name in SIGMA]
for i, (*cells, colour) in enumerate(rows):
    for x, cell, weight in zip(COLS, cells, ("bold", "normal", "normal")):
        ax.text(x, 0.965 - 0.047 * i, cell, ha="left", va="top",
                fontsize=ms.FS_TICK, color=colour, fontweight=weight)
ax.text(COLS[0], 0.745,
        f"Solid, samples (n = {N:,} each)\n"
        "Dashed, Φ(Φ⁻¹(p) − σ)\n"
        f"Shaded area S: Gini of {SHADED} = 2S", ha="left", va="top",
        fontsize=ms.FS_TICK, linespacing=1.3)

# share of the top 10% in the most unequal population
p_cut = 1 - TOP
ax.plot([p_cut, p_cut, 1], [0, 1 - top_share, 1 - top_share],
        color=ms.INK, lw=0.5, ls=(0, (1, 1.6)))
ax.plot([1.035, 1.055, 1.055, 1.035],
        [1 - top_share, 1 - top_share, 1, 1], color=ms.INK, lw=0.6,
        clip_on=False, solid_capstyle="butt")
ax.text(1.085, 1 - top_share / 2,
        f"Top {TOP:.0%} of {SHADED} hold {top_share:.0%}", rotation=90,
        ha="left", va="center", fontsize=ms.FS_TICK)

ticks = np.linspace(0, 1, 6)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks(ticks, [f"{t:.0%}" for t in ticks])
ax.set_yticks(ticks, [f"{t:.0%}" for t in ticks])
ax.set_xlabel("Cumulative share of population, poorest first")
ax.set_ylabel("Cumulative share of income")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig198_lorenz_gini.{ext}")
print("fig198_lorenz_gini: saved png + pdf")
