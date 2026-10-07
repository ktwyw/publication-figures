"""Fig. 200 - Comparison counts of sorting algorithms (1.5 column, 120 mm).

How to show empirical complexity without a stopwatch: insertion sort,
top-down merge sort and random-pivot quicksort are implemented here with
a comparison counter and run on seeded random permutations, so every
point is reproducible to the last comparison. Means with min-max
whiskers sit on the exact expected counts, slope triangles give the
fitted log-log exponents, and the information-theoretic bound log₂(n!)
closes off the region no comparison sort can average in; the strip
below resolves how far each algorithm stays above that bound. The
self-check is that every run returns the sorted array, every merge-sort
count respects the worst case n⌈log₂n⌉ − 2^⌈log₂n⌉ + 1, every mean lies
on or above log₂(n!), the insertion-sort mean is within 5% of
n(n − 1)/4 at the largest n and its fitted exponent is 2.0 ± 0.05.

Model: thin lines are exact expectations for distinct keys: insertion
n(n − 1)/4 + n − Hₙ; merge M(n) = M(⌈n/2⌉) + M(⌊n/2⌋) + n − a/(b + 1)
− b/(a + 1) for halves a and b, close to n·log₂n − 1.25n; quicksort
2(n + 1)Hₙ − 4n, leading term 1.39·n·log₂n. Data: 8 seeded random
permutations per n, n = 2⁴ to 2¹²; points, means; whiskers, minimum to
maximum. All data are simulated.
"""

import functools
import math

import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy import special

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(200)

# ------------------------------------------------------------- DATA ----
SIZES = 2 ** np.arange(4, 13)                # n = 16 ... 4,096
REPS = 8                                     # seeded permutations per n


# -------------------------------------------------------- ALGORITHMS ---
def insertion_sort(keys):
    a, count = list(keys), 0
    for i in range(1, len(a)):
        x, j = a[i], i - 1
        while j >= 0:
            count += 1                       # compare a[j] with x
            if a[j] <= x:
                break
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = x
    return a, count


def merge_sort(keys):
    if len(keys) < 2:
        return list(keys), 0
    half = (len(keys) + 1) // 2              # sizes ⌈n/2⌉ and ⌊n/2⌋
    (left, c_left), (right, c_right) = merge_sort(keys[:half]), merge_sort(
        keys[half:])
    out, i, j, count = [], 0, 0, c_left + c_right
    while i < len(left) and j < len(right):
        count += 1                           # compare the two heads
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    return out + left[i:] + right[j:], count


def quicksort(keys):
    """In place, pivot drawn uniformly with the seeded generator."""
    a, count, stack = list(keys), 0, [(0, len(keys) - 1)]
    while stack:
        low, high = stack.pop()
        if low >= high:
            continue
        pick = int(rng.integers(low, high + 1))
        a[pick], a[high] = a[high], a[pick]
        pivot, store = a[high], low
        for j in range(low, high):
            count += 1                       # compare a[j] with the pivot
            if a[j] < pivot:
                a[store], a[j] = a[j], a[store]
                store += 1
        a[store], a[high] = a[high], a[store]
        stack += [(low, store - 1), (store + 1, high)]
    return a, count


def harmonic(n):
    return special.digamma(n + 1.0) + np.euler_gamma


def log2_factorial(n):
    return special.gammaln(n + 1.0) / np.log(2)


@functools.lru_cache(maxsize=None)
def merge_mean_one(n):
    """Exact mean: merging runs of a and b keys costs n − a/(b+1) − b/(a+1)."""
    if n < 2:
        return 0.0
    a, b = (n + 1) // 2, n // 2
    return (merge_mean_one(a) + merge_mean_one(b) + n - a / (b + 1)
            - b / (a + 1))


def merge_mean(n):
    return np.array([merge_mean_one(int(v)) for v in np.atleast_1d(n)])


ALGORITHMS = {     # name: (function, exact expected comparisons, colour)
    "Insertion sort": (insertion_sort,
                       lambda n: n * (n - 1) / 4 + n - harmonic(n),
                       ms.GREY_DARK),
    "Quicksort": (quicksort, lambda n: 2 * (n + 1) * harmonic(n) - 4 * n,
                  ms.BLUE),
    "Merge sort": (merge_sort, merge_mean, ms.VERMILLION),
}

counts = {name: np.empty((SIZES.size, REPS), int) for name in ALGORITHMS}
all_sorted = True
for row, n in enumerate(SIZES):
    for rep in range(REPS):
        keys = rng.permutation(int(n)).tolist()
        for name, (sort, _theory, _colour) in ALGORITHMS.items():
            result, counts[name][row, rep] = sort(keys)
            all_sorted &= result == list(range(n))
mean = {name: c.mean(axis=1) for name, c in counts.items()}
slope = {name: np.polyfit(np.log(SIZES), np.log(m), 1)[0]
         for name, m in mean.items()}
bound = log2_factorial(SIZES)

# ------------------------------------------------------- SELF-CHECK ---
assert all_sorted
ceil_log = np.ceil(np.log2(SIZES)).astype(int)
merge_worst = SIZES * ceil_log - 2 ** ceil_log + 1
assert np.all(counts["Merge sort"] <= merge_worst[:, None])
# ⌈log₂(n!)⌉ bounds the worst case and log₂(n!) the average over all
# inputs; a single lucky permutation may need fewer (three runs here do)
assert np.all(merge_worst >= [math.ceil(math.log2(math.factorial(n)))
                              for n in SIZES.tolist()])
for name, (_sort, theory, _colour) in ALGORITHMS.items():
    assert np.all(mean[name] >= bound), name
    assert abs(mean[name][-1] / theory(SIZES[-1:])[0] - 1) < 0.05, name
n_max = int(SIZES[-1])
quadratic = n_max * (n_max - 1) / 4
assert abs(mean["Insertion sort"][-1] / quadratic - 1) < 0.05
assert abs(slope["Insertion sort"] - 2.0) < 0.05, slope
assert np.abs(mean["Merge sort"] / merge_mean(SIZES) - 1).max() < 0.03
print(f"fig200: self-check passed ({3 * SIZES.size * REPS} runs sorted; "
      f"at n = {n_max}: insertion {mean['Insertion sort'][-1]:.0f} vs "
      f"n(n-1)/4 = {quadratic:.0f}, merge {mean['Merge sort'][-1]:.0f} "
      f"<= worst case {merge_worst[-1]}, quicksort "
      f"{mean['Quicksort'][-1]:.0f}, log2(n!) = {bound[-1]:.0f}; fitted "
      "exponents " + ", ".join(f"{v:.3f}" for v in slope.values()) + ")")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 104)
gs = ms.grid(fig, 2, 1, left=15, right=5, top=5, bottom=11, hspace=5,
             height_ratios=[2.6, 1])
ax = fig.add_subplot(gs[0])
ax_gap = fig.add_subplot(gs[1], sharex=ax)
n_line = np.unique(np.round(np.geomspace(SIZES[0], SIZES[-1], 300)))
bound_line = log2_factorial(n_line)
Y_LOW, Y_HIGH, GAP_LIM = 20.0, 2e7, (-0.6, 5.0)

ax.fill_between(n_line, Y_LOW, bound_line, color=ms.GREY_LIGHT, alpha=0.45,
                lw=0)
ax.plot(n_line, bound_line, color=ms.INK, lw=0.8, ls=(0, (4, 2)))
for name, (_sort, theory, colour) in ALGORITHMS.items():
    low, high = counts[name].min(axis=1), counts[name].max(axis=1)
    ax.plot(n_line, theory(n_line), color=colour, lw=0.6)
    ax.errorbar(SIZES, mean[name], yerr=[mean[name] - low, high - mean[name]],
                fmt="o", color=colour, ms=2.8, lw=0.7, capsize=1.5,
                capthick=0.7)
    # strip: comparisons per key above the bound, trimmed to its frame
    excess = (theory(n_line) - bound_line) / n_line
    ax_gap.plot(n_line[excess <= GAP_LIM[1]], excess[excess <= GAP_LIM[1]],
                color=colour, lw=0.6)
    shown = (high - bound) / SIZES <= GAP_LIM[1]
    ax_gap.errorbar(SIZES[shown], ((mean[name] - bound) / SIZES)[shown],
                    yerr=[((mean[name] - low) / SIZES)[shown],
                          ((high - mean[name]) / SIZES)[shown]], fmt="o",
                    color=colour, ms=2.8, lw=0.7, capsize=1.5, capthick=0.7)
ax_gap.fill_between(SIZES[[0, -1]], GAP_LIM[0], 0, color=ms.GREY_LIGHT,
                    alpha=0.45, lw=0)
ax_gap.plot(SIZES[[0, -1]], [0, 0], color=ms.INK, lw=0.8, ls=(0, (4, 2)))


def slope_triangle(x0, x1, name, lift):
    """Triangle above a fitted line (as fig064), labelled with its slope."""
    theory, colour = ALGORITHMS[name][1:]
    y0 = lift * theory(np.array([x0]))[0]
    y1 = y0 * (x1 / x0) ** slope[name]
    ax.plot([x0, x0, x1, x0], [y0, y1, y1, y0], color=colour, lw=0.6)
    ax.text(np.sqrt(x0 * x1), y1 * 1.25, "1", ha="center", va="bottom",
            fontsize=ms.FS_TICK, color=colour)
    ax.text(x0 / 1.12, np.sqrt(y0 * y1), f"{slope[name]:.2f}", ha="right",
            va="center", fontsize=ms.FS_TICK, color=colour)


slope_triangle(2 ** 6.5, 2 ** 8.5, "Insertion sort", 2.6)
slope_triangle(2 ** 10, 2 ** 12, "Quicksort", 2.0)

# direct labels beyond the last points; formulas in an aligned block
for name, nudge in (("Insertion sort", 1.0), ("Quicksort", 1.5),
                    ("Merge sort", 1 / 1.5)):
    ax.text(SIZES[-1] * 1.22, mean[name][-1] * nudge, name, va="center",
            color=ALGORITHMS[name][2], fontweight="bold")
ax.text(2 ** 11.8, 60, "No comparison sort averages\n"
        "fewer than log₂(n!) (dashed)", ha="right", va="bottom",
        fontsize=ms.FS_TICK)
for i, (label, formula, colour) in enumerate((
        ("Thin lines, expected comparisons", "", ms.INK),
        ("insertion", "n(n − 1)/4 + n − Hₙ", ms.GREY_DARK),
        ("quicksort", "2(n + 1)Hₙ − 4n ≈ 1.39 n log₂n", ms.BLUE),
        ("merge", "recurrence ≈ n log₂n − 1.25n", ms.VERMILLION))):
    y_row = 0.965 - 0.062 * i
    ax.text(0.03, y_row, label, transform=ax.transAxes, va="top",
            fontsize=ms.FS_TICK, color=colour)
    ax.text(0.16, y_row, formula, transform=ax.transAxes, va="top",
            fontsize=ms.FS_TICK, color=colour)
ax.text(0.03, 0.965 - 0.062 * 4.3,
        f"Points, mean of {REPS} permutations;\nwhiskers, min to max",
        transform=ax.transAxes, va="top", fontsize=ms.FS_SMALL,
        color=ms.GREY_DARK, linespacing=1.3)

ax.set_xscale("log", base=2)
ax.set_yscale("log")
# Unicode exponents: mathtext superscripts would fall below 5 pt
SUPER = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
ax.xaxis.set_major_formatter(FuncFormatter(
    lambda v, _pos: "2" + str(round(np.log2(v))).translate(SUPER)))
ax.yaxis.set_major_formatter(FuncFormatter(
    lambda v, _pos: "10" + str(round(np.log10(v))).translate(SUPER)))
ax.set_xlim(2 ** 3.6, 2 ** 14.6)
ax.set_ylim(Y_LOW, Y_HIGH)
ax.set_xticks(SIZES[::2])
ax.spines["bottom"].set_bounds(SIZES[0], SIZES[-1])
ax.tick_params(labelbottom=False)
ax.set_ylabel("Comparisons")
ax_gap.set_ylim(*GAP_LIM)
ax_gap.set_yticks(np.arange(0, 6))
ax_gap.spines["left"].set_bounds(0, 5)
ax_gap.text(SIZES[-1] * 1.22, 0, "log₂(n!)", va="center",
            fontsize=ms.FS_TICK)
ax_gap.spines["bottom"].set_bounds(SIZES[0], SIZES[-1])
ax_gap.set_xlabel("Array length n")
ax_gap.set_ylabel("Comparisons above\nlog₂(n!), per key")

ms.assert_aligned([ax, ax_gap], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig200_algorithm_scaling.{ext}")
print("fig200_algorithm_scaling: saved png + pdf")
