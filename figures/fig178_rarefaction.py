"""Fig. 178 - Rank-abundance and rarefaction curves (double column, 183 mm).

Why species counts from samples of different size cannot be compared
directly. Three communities of different evenness are shown as
rank-abundance (Whittaker) curves, then as individual-based rarefaction
curves: the expected richness in a random subsample of n individuals,
from the hypergeometric formula, with the spread of actual random
subsamples around it. Read at a common n, the largest sample is no
longer the richest. The self-check is that the analytic curve equals
the subsampling mean within 3 Monte-Carlo standard errors at four
sample sizes per community, equals the observed richness at the full
sample size, and equals 1 at n = 1.

Data: individuals drawn from lognormal species-abundance distributions
(species pool, log-s.d., individuals: 110, 2.2, 1,600; 60, 0.6, 800;
32, 1.1, 400). Lines, E[S(n)] = Σ[1 − C(N − Nᵢ, n)/C(N, n)]; bands,
central 95% of 200 random subsamples without replacement; points,
observed richness. All data are simulated.
"""

import numpy as np
from scipy.special import gammaln

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(178)

# ------------------------------------------------------------- DATA ----
# name: (species pool, s.d. of ln abundance, individuals counted, colour)
SITES = {"Site A": (110, 2.2, 1600, ms.BLUE),
         "Site B": (60, 0.6, 800, ms.VERMILLION),
         "Site C": (32, 1.1, 400, ms.GREEN)}
MARKERS = {"Site A": "o", "Site B": "s", "Site C": "^"}
N_SUBSAMPLES = 200
CHECK_SIZES = [5, 20, 80, 300]     # sample sizes compared in the self-check
counts = {}
for site, (pool, log_sd, n_ind, _colour) in SITES.items():
    relative = np.exp(rng.normal(0.0, log_sd, pool))
    drawn = rng.multinomial(n_ind, relative / relative.sum())
    counts[site] = np.sort(drawn[drawn > 0])[::-1]       # ranked abundances
N_COMMON = min(c.sum() for c in counts.values())         # smallest sample


# -------------------------------------------------------- ESTIMATORS ---
def log_choose(a, k):
    return gammaln(a + 1) - gammaln(k + 1) - gammaln(a - k + 1)


def rarefy(abundance, n):
    """Expected richness in n individuals drawn without replacement."""
    n = np.atleast_1d(n)[:, None]
    total, others = abundance.sum(), abundance.sum() - abundance[None, :]
    possible = others >= n             # can species i be missed altogether?
    log_miss = (log_choose(np.where(possible, others, n), n)
                - log_choose(total, n))
    return np.sum(1 - np.where(possible, np.exp(log_miss), 0.0), axis=1)


def subsample_curves(abundance, sizes):
    """Richness at each size along N_SUBSAMPLES random orderings."""
    labels = np.repeat(np.arange(abundance.size), abundance)
    curves = np.empty((N_SUBSAMPLES, sizes.size))
    for row in range(N_SUBSAMPLES):
        _species, first_seen = np.unique(rng.permutation(labels),
                                         return_index=True)
        curves[row] = np.searchsorted(np.sort(first_seen), sizes)
    return curves


results = {}
for site, abundance in counts.items():
    total = int(abundance.sum())
    sizes = np.unique(np.concatenate([np.arange(1, 40),
                                      np.linspace(40, total, 120).astype(int),
                                      CHECK_SIZES, [N_COMMON]]))
    curves = subsample_curves(abundance, sizes)
    results[site] = dict(sizes=sizes, analytic=rarefy(abundance, sizes),
                         mean=curves.mean(axis=0),
                         sem=curves.std(axis=0, ddof=1) / N_SUBSAMPLES ** 0.5,
                         band=np.percentile(curves, [2.5, 97.5], axis=0),
                         observed=abundance.size, total=total,
                         common=rarefy(abundance, N_COMMON)[0])

# ------------------------------------------------------- SELF-CHECK ---
worst_z = 0.0
for site, res in results.items():
    at = np.isin(res["sizes"], CHECK_SIZES)
    z = np.abs(res["analytic"][at] - res["mean"][at]) / res["sem"][at]
    worst_z = max(worst_z, z.max())
    assert np.all(z < 3), (site, z)
    assert abs(res["analytic"][-1] - res["observed"]) < 1e-9, site
    assert abs(res["analytic"][0] - 1.0) < 1e-9, site
    assert np.all(np.diff(res["analytic"]) > 0)          # more is never fewer
observed = {site: res["observed"] for site, res in results.items()}
rarefied = {site: res["common"] for site, res in results.items()}
assert observed["Site A"] > observed["Site B"]           # the raw ranking ...
assert rarefied["Site A"] < rarefied["Site B"]           # ... reverses
print("fig178: self-check passed (analytic = subsampling mean within "
      f"{worst_z:.2f} s.e. at n = {CHECK_SIZES}; = observed S at full N; "
      f"= 1 at n = 1; rarefied to n = {N_COMMON}: "
      + ", ".join(f"{site[-1]} {observed[site]} -> {rarefied[site]:.1f}"
                  for site in SITES) + ")")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 72)
gs = ms.grid(fig, 1, 2, left=15, right=6, top=8, bottom=11, wspace=20,
             width_ratios=[1, 1.25])
ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# a, Whittaker plot: steep curves are uneven, shallow ones even
for site, abundance in counts.items():
    colour = SITES[site][3]
    rank = np.arange(1, abundance.size + 1)
    share = abundance / abundance.sum()
    ax_a.plot(rank, share, color=colour, lw=0.9, marker=MARKERS[site],
              ms=2.0, mew=0,
              label=f"{site}: N = {abundance.sum():,}, S = {abundance.size}")
ax_a.set_yscale("log")
ms.plain_log_ticks(ax_a.yaxis)
ax_a.set_xlim(0, 80)
ax_a.set_ylim(4e-4, 0.5)
ax_a.set_xlabel("Species rank")
ax_a.set_ylabel("Relative abundance")
ax_a.legend(loc="upper right", title="N individuals, S species observed",
            handlelength=1.8)

# b, rarefaction: analytic expectation over the subsampling band
ax_b.axvline(N_COMMON, color=ms.GREY, lw=0.6, ls=(0, (3, 2)))
LABEL_SHIFT = {"Site A": (-2, 7), "Site B": (6, 7), "Site C": (7, -3)}
for site, res in results.items():
    colour = SITES[site][3]
    ax_b.fill_between(res["sizes"], *res["band"], color=colour, alpha=0.2,
                      lw=0)
    ax_b.plot(res["sizes"], res["analytic"], color=colour, lw=1.2)
    ax_b.plot(res["total"], res["observed"], MARKERS[site], color=colour,
              ms=3.6, mec="white", mew=0.5, clip_on=False, zorder=4)
    if res["total"] > N_COMMON:        # rarefied richness, open point
        ax_b.plot(N_COMMON, res["common"], "o", ms=3.0, mfc="white",
                  mec=colour, mew=0.7, zorder=5)
    ax_b.annotate(f"{site}, S = {res['observed']}",
                  xy=(res["total"], res["observed"]), xytext=LABEL_SHIFT[site],
                  textcoords="offset points", color=colour,
                  ha="left" if site == "Site C" else "right",
                  va="bottom", fontsize=ms.FS_TICK)
ax_b.set_xlim(0, 1680)
ax_b.set_ylim(0, 84)
ax_b.set_xlabel("Individuals sampled, n")
ax_b.set_ylabel("Expected species richness")
lines = [f"Rarefied to n = {N_COMMON}:"] + [
    f"{site}  {rarefied[site]:.1f}"
    for site in sorted(SITES, key=rarefied.get, reverse=True)]
colours = [ms.INK] + [SITES[site][3] for site in
                      sorted(SITES, key=rarefied.get, reverse=True)]
for row, (line, colour) in enumerate(zip(lines, colours)):
    ax_b.annotate(line, xy=(860, 44), xytext=(0, -8.5 * row),
                  textcoords="offset points", color=colour, va="center",
                  fontsize=ms.FS_TICK)
ax_b.text(860, 3, "Lines, hypergeometric expectation\n"
          f"Bands, 95% of {N_SUBSAMPLES} random subsamples\n"
          "Filled points, observed; open, rarefied", va="bottom",
          fontsize=ms.FS_SMALL, color=ms.GREY_DARK, linespacing=1.3)

ms.panel_label(ax_a, "a", dx_pt=-30, dy_pt=8)
ms.panel_label(ax_b, "b", dx_pt=-26, dy_pt=8)
ms.assert_aligned([ax_a, ax_b])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig178_rarefaction.{ext}")
print("fig178_rarefaction: saved png + pdf")
