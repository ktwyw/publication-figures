"""Fig. 190 - Screening curves and load duration (double column, 183 mm).

The classic two-step generation-planning picture on one shared time
axis. a, screening curves: annualised cost per kilowatt against hours
of operation for three technologies; the least-cost lower envelope is
thick and its crossover hours drop straight down. b, the load-duration
curve of a simulated year; each crossover hour is projected onto the
curve and across to the capacity stack, and the area under the curve is
filled by the technology that serves it. The self-check is that the
layer-cake area under the load-duration curve equals the annual energy
of the hourly series to 1e-9 relative, the energy shares sum to 100%,
each crossover hour equals (F_j − F_i)/(c_i − c_j), and the stacked
capacities sum to the peak load.

Data: 8,760 hourly demands (GW) = seasonal + daily + weekend pattern +
AR(1) noise (s.d. 0.25 GW, lag-1 correlation 0.9); placeholder costs
F + c·h with F = 55, 105, 240 $ per kW-year and c = 95, 48, 18 $ per MWh
for peaker, mid-merit and baseload. All data are simulated.
"""

import numpy as np
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FuncFormatter
from scipy import optimize, signal

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(190)
HOURS = 8760

# ------------------------------------------------------------- DATA ----
# technology: (fixed cost, $ per kW-year; variable cost, $ per MWh; colour)
TECH = {"Peaker": (55.0, 95.0, ms.VERMILLION),
        "Mid-merit": (105.0, 48.0, ms.ORANGE),
        "Baseload": (240.0, 18.0, ms.BLUE)}
MEAN_GW, NOISE_SD, NOISE_RHO = 6.6, 0.25, 0.9

t = np.arange(HOURS)
day, hour = t // 24, t % 24
seasonal = (1.05 * np.cos(2 * np.pi * (day - 15) / 365)        # winter peak
            + 0.35 * np.cos(4 * np.pi * (day - 15) / 365))     # summer bump
daily = (-0.9 * np.cos(2 * np.pi * (hour - 4) / 24)
         + 0.35 * np.cos(4 * np.pi * (hour - 19) / 24))        # evening peak
weekend = np.where(day % 7 >= 5, -0.55, 0.0)
noise = signal.lfilter([np.sqrt(1 - NOISE_RHO ** 2)], [1, -NOISE_RHO],
                       rng.normal(0, 1, HOURS))                # AR(1)
demand = MEAN_GW + seasonal + daily + weekend + NOISE_SD * noise   # GW

# --------------------------------------------------------- ESTIMATOR ---
names = list(TECH)


def annual_cost(name, h):
    """Screening curve: $ per kW-year when run h hours a year."""
    fixed, variable, _ = TECH[name]
    return fixed + variable * h / 1000.0        # $/MWh × h = $/kW-yr × 1000


# crossover hours of neighbours in the merit order, found numerically
crossover = [optimize.brentq(lambda h, i=i, j=j: annual_cost(i, h)
                             - annual_cost(j, h), 0, HOURS, xtol=1e-9)
             for i, j in zip(names[:-1], names[1:])]

# load-duration curve: the load sorted in descending order, one step an hour
ldc = np.sort(demand)[::-1]
peak = ldc[0]
# a load level met for more than h hours belongs to the cheaper-to-run unit
levels = [ldc[int(np.floor(h))] for h in crossover]           # GW at h₁, h₂
edges = np.array([peak, *levels, 0.0])                        # top to bottom
capacity = -np.diff(edges)                                    # GW per unit
energy = np.array([(np.clip(demand, lo, hi) - lo).sum()       # GWh per unit
                   for hi, lo in zip(edges[:-1], edges[1:])])
share = 100 * energy / energy.sum()

# ------------------------------------------------------- SELF-CHECK ---
total_energy = demand.sum()                                    # GWh
# layer cake: each step of height ldc[k] − ldc[k+1] lasts k + 1 hours
layer_cake = (-np.diff(np.append(ldc, 0.0)) * np.arange(1, HOURS + 1)).sum()
assert abs(layer_cake - total_energy) < 1e-9 * total_energy
assert abs(energy.sum() - total_energy) < 1e-9 * total_energy
assert abs(share.sum() - 100) < 1e-9
for h, i, j in zip(crossover, names[:-1], names[1:]):
    analytic = (TECH[j][0] - TECH[i][0]) / (TECH[i][1] - TECH[j][1]) * 1000
    assert abs(h - analytic) < 1e-6 * analytic, (h, analytic)
assert 0 < crossover[0] < crossover[1] < HOURS
assert abs(capacity.sum() - peak) < 1e-12 and np.all(capacity > 0)
bounds = [0.0, *crossover, float(HOURS)]
for k, name in enumerate(names):                # the envelope is as labelled
    mid = (bounds[k] + bounds[k + 1]) / 2
    assert name == min(names, key=lambda n: annual_cost(n, mid))
print(f"fig190: self-check passed (crossovers {crossover[0]:.1f} and "
      f"{crossover[1]:.1f} h; capacities "
      + ", ".join(f"{c:.2f}" for c in capacity)
      + f" GW sum to peak {peak:.2f} GW; energy shares "
      + ", ".join(f"{s:.1f}" for s in share)
      + f"% of {total_energy / 1000:.2f} TWh; layer-cake area error "
      f"{abs(layer_cake / total_energy - 1):.1e})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 118)
X0, WIDTH = 14.0, 128.0                       # mm, the shared time axis
ax_a = ms.axes(fig, X0, 68, WIDTH, 38)
ax_b = ms.axes(fig, X0, 11, WIDTH, 48, sharex=ax_a)
ax_cap = ms.axes(fig, X0 + WIDTH + 3, 11, 5, 48, sharey=ax_b)
GUIDE = dict(color=ms.GREY_DARK, lw=0.5, ls=(0, (3, 2)))
COST_MAX, GW_MAX = 950.0, 11.0

# a, screening curves: thin over the year, thick where least-cost
h_line = np.array([0.0, HOURS])
for k, name in enumerate(names):
    fixed, variable, colour = TECH[name]
    ax_a.plot(h_line, annual_cost(name, h_line), color=colour, lw=0.8)
    best = np.array(bounds[k:k + 2])
    ax_a.plot(best, annual_cost(name, best), color=colour, lw=2.6,
              solid_capstyle="butt")
    ax_a.text(HOURS + 110, annual_cost(name, HOURS),
              f"{name}\nF = {fixed:.0f}, c = {variable:.0f}", color=colour,
              fontsize=ms.FS_TICK, va="center", linespacing=1.2)
ax_a.text(250, COST_MAX - 45, "Annual cost = F + c × hours\n"
          "F, fixed cost (\\$ per kW-year); c, variable cost (\\$ per MWh)\n"
          "Thick line, least-cost envelope", fontsize=ms.FS_TICK, va="top",
          linespacing=1.3)
ax_a.set_ylim(0, COST_MAX)
ax_a.set_yticks(np.arange(0, COST_MAX, 200))
ax_a.set_ylabel("Annualised cost\n(\\$ per kW-year)")
ax_a.tick_params(labelbottom=False)
top = ax_a.secondary_xaxis("top", functions=(lambda h: 100 * h / HOURS,
                                             lambda cf: cf * HOURS / 100))
top.set_xticks(np.arange(0, 101, 20))
top.set_xlabel("Capacity factor (%)")

# b, load-duration curve with the area split by technology
h_edges = np.arange(HOURS + 1)
ldc_steps = np.append(ldc, ldc[-1])
for k, name in enumerate(names):
    hi, lo = edges[k], edges[k + 1]
    n = int((ldc > lo).sum()) + 1            # the band ends where it is empty
    ax_b.fill_between(h_edges[:n], lo, np.clip(ldc_steps[:n], lo, hi),
                      step="post", color=TECH[name][2], alpha=0.35, lw=0)
    ax_cap.fill_between([0, 1], lo, hi, color=TECH[name][2], alpha=0.35,
                        lw=0)
    ax_cap.text(1.25, (lo + hi) / 2, f"{name}\n{capacity[k]:.1f} GW",
                fontsize=ms.FS_TICK, va="center", linespacing=1.2)
ax_b.plot(h_edges, ldc_steps, drawstyle="steps-post", color=ms.INK, lw=0.9)
for k, height in ((1, 0.3), (2, 0.45)):         # bands wide enough to label
    ax_b.text(120, edges[k + 1] + height * capacity[k],
              f"{names[k]}: {share[k]:.1f}% of energy", fontsize=ms.FS_TICK,
              va="center")
ax_b.annotate(f"{names[0]}: {share[0]:.1f}% of energy",
              xy=(0.3 * crossover[0], levels[0] + 0.25 * capacity[0]),
              xytext=(1.5 * crossover[0], peak), fontsize=ms.FS_TICK,
              va="center",
              arrowprops=dict(arrowstyle="-", color=ms.INK, lw=0.5,
                              shrinkA=2, shrinkB=0, relpos=(0.0, 0.5)))
ax_b.text(HOURS - 150, GW_MAX - 0.5,
          f"Peak {peak:.1f} GW; {total_energy / 1000:.1f} TWh a year",
          fontsize=ms.FS_TICK, ha="right", va="top")
ax_b.set_xlim(0, HOURS)
ax_b.set_ylim(0, GW_MAX)
ax_b.set_xticks(np.arange(0, HOURS + 1, 1000))
ax_b.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:,.0f}"))
ax_b.set_yticks(np.arange(0, GW_MAX, 2))
ax_b.set_xlabel("Hours of operation per year, or hours the load is exceeded")
ax_b.set_ylabel("Load (GW)")
ax_cap.set_xlim(0, 1)
ax_cap.axis("off")

# projections: crossover hour straight down onto the curve, then across to
# the capacity stack
for h, level, name in zip(crossover, levels, names):
    cost = annual_cost(name, h)
    ax_a.plot(h, cost, "o", ms=3.6, mfc="white", mec=ms.INK, mew=0.8,
              zorder=5)
    fig.add_artist(ConnectionPatch((h, cost), (h, level), "data", "data",
                                   axesA=ax_a, axesB=ax_b, **GUIDE))
    fig.add_artist(ConnectionPatch((h, level), (0, level), "data", "data",
                                   axesA=ax_b, axesB=ax_cap, **GUIDE))
    ax_b.plot(h, level, "o", ms=3.6, mfc="white", mec=ms.INK, mew=0.8,
              zorder=5)
    ax_a.text(h + 90, 25, f"{h:,.0f} h", fontsize=ms.FS_TICK, va="bottom")

ms.panel_label(ax_a, "a", dx_pt=-34, dy_pt=14)
ms.panel_label(ax_b, "b", dx_pt=-34, dy_pt=2)
ms.assert_aligned([ax_a, ax_b], edges=("left", "right"))
ms.assert_aligned([ax_b, ax_cap])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig190_load_duration_dispatch.{ext}")
print("fig190_load_duration_dispatch: saved png + pdf")
