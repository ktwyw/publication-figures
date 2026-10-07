"""Fig. 123 - Pareto frontier of accuracy against cost (single column, 89 mm).

Forty models from three families trade accuracy against inference cost.
Only the non-dominated ones matter: no other model is both at least as
cheap and at least as accurate. The frontier is drawn as a staircase
(the best accuracy available at or below each cost), its models are
coloured by family and labelled directly, and everything dominated is
grey. The self-check is that a sorted sweep (by cost, keeping each new
accuracy record) returns exactly the brute-force O(n²) non-dominated
set.

Statistics: n = 40 models (14, 13 and 13 per family); each point is one
model, a single evaluation; no interval and no test. All data are
simulated.
"""

import numpy as np

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(141)
# family: (prefix, n, ceiling %, deficit at 1 ms, decay, colour, marker)
FAMILIES = {"ConvNet": ("C", 14, 84.0, 14.0, 0.55, ms.BLUE, "o"),
            "Transformer": ("T", 13, 90.5, 26.0, 0.42, ms.VERMILLION, "s"),
            "Hybrid": ("H", 13, 88.0, 19.0, 0.50, ms.GREEN, "^")}

# ------------------------------------------------------------- DATA ----
# cost: latency per image (ms); accuracy: a saturating family curve minus
# a positive shortfall, so most models sit below their family's best
cost, accuracy, family, name = [], [], [], []
for fam, (prefix, n, ceiling, deficit, decay, _col, _mk) in FAMILIES.items():
    c = np.sort(10 ** rng.uniform(-0.15, 2.25, n))
    cost.extend(c)
    accuracy.extend(ceiling - deficit * c ** -decay - rng.exponential(3.0, n))
    family.extend([fam] * n)
    name.extend(f"{prefix}{i + 1}" for i in range(n))
cost, accuracy, family = map(np.array, (cost, accuracy, family))
N = cost.size


# --------------------------------------------------------- FRONTIER ---
def pareto_sweep(cost, accuracy):
    """Indices on the frontier: sort by cost, keep every accuracy record."""
    keep, record = [], -np.inf
    for i in np.lexsort((-accuracy, cost)):      # cheapest first
        if accuracy[i] > record:
            keep.append(i)
            record = accuracy[i]
    return np.array(keep)


def pareto_brute(cost, accuracy):
    """Indices that no other point dominates, by all n² comparisons."""
    no_worse = ((cost[None, :] <= cost[:, None])
                & (accuracy[None, :] >= accuracy[:, None]))
    better = ((cost[None, :] < cost[:, None])
              | (accuracy[None, :] > accuracy[:, None]))
    return np.flatnonzero(~np.any(no_worse & better, axis=1))


front = pareto_sweep(cost, accuracy)
on_front = np.isin(np.arange(N), front)

# ------------------------------------------------------- SELF-CHECK ---
assert np.array_equal(np.sort(front), pareto_brute(cost, accuracy))
assert np.all(np.diff(cost[front]) > 0)
assert np.all(np.diff(accuracy[front]) > 0)
print(f"fig123: self-check passed (sweep = brute force: {front.size} of "
      f"{N} models non-dominated; frontier from {accuracy[front[0]]:.1f}% "
      f"at {cost[front[0]]:.2g} ms to {accuracy[front[-1]]:.1f}% at "
      f"{cost[front[-1]]:.3g} ms)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 68)
ax = fig.add_subplot(ms.grid(fig, 1, 1, left=13, right=4, top=4, bottom=11)[0])
X_MAX = 300.0
GREY_MID = "#B4B4B4"          # dominated models: visible, never salient

# staircase: accuracy stays at each frontier model until the next one
ax.step(np.append(cost[front], X_MAX), np.append(accuracy[front],
                                                 accuracy[front[-1]]),
        where="post", color=ms.INK, lw=0.9, zorder=2)
for fam, (_p, _n, _c, _d, _k, colour, marker) in FAMILIES.items():
    member = family == fam
    ax.scatter(cost[member & ~on_front], accuracy[member & ~on_front], s=11,
               marker=marker, color=GREY_MID, linewidths=0, zorder=1)
    ax.scatter(cost[member & on_front], accuracy[member & on_front], s=17,
               marker=marker, color=colour, linewidths=0, zorder=3,
               label=fam)
# nothing lies above and to the left of a frontier point, so labels go there
for i in front:
    ax.annotate(name[i], (cost[i], accuracy[i]), xytext=(-2.5, 2.0),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=ms.FS_SMALL,
                color=FAMILIES[family[i]][5])

ax.set_xscale("log")
ax.set_xlim(0.5, X_MAX)
ms.plain_log_ticks(ax.xaxis)
ax.set_ylim(58, 90)
ax.set_yticks(np.arange(60, 91, 10))
ax.set_xlabel("Inference cost (ms per image)")
ax.set_ylabel("Top-1 accuracy (%)")

ax.annotate("", xy=(0.05, 0.95), xytext=(0.17, 0.80),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=ms.GREY_DARK, lw=0.8,
                            mutation_scale=7, shrinkA=0, shrinkB=0))
ax.text(0.185, 0.78, "Better", transform=ax.transAxes, ha="left", va="top",
        fontsize=ms.FS_TICK, color=ms.GREY_DARK)
ax.legend(loc="lower right", title="On the frontier",
          handletextpad=0.2, alignment="left")
ax.text(0.97, 0.275, f"Grey, dominated\n(n = {N - front.size} of {N})",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=ms.FS_TICK, color=ms.GREY_DARK)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig123_pareto_front.{ext}")
print("fig123_pareto_front: saved png + pdf")
