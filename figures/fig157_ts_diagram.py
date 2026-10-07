"""Fig. 157 - T-S diagram with isopycnals and cabbeling (single column, 89 mm).

The oceanographer's temperature-salinity plot: potential temperature
against salinity over contours of potential density anomaly σ₀ from the
EOS-80 equation of state, a CTD cast coloured by depth, three water
masses, and a straight mixing line between two masses of equal density.
Because the isopycnals curve, every mixture on that line is denser than
both end members (cabbeling). The self-check is that the two UNESCO check values,
ρ(35, 5 °C, 0) = 1027.675 and ρ(0, 5 °C, 0) = 999.967 kg m⁻³, are
reproduced to 0.001, that density rises with salinity at every plotted
temperature, that the cast is stably stratified before noise, and that
all mixtures on the line are denser than its end members.

Model: EOS-80 one-atmosphere equation of state (Millero & Poisson 1981;
UNESCO 1983); σ₀ = ρ − 1000 kg m⁻³. Data: n = 161 samples every 25 m to
4,000 m, mixed from the three water masses with Gaussian noise (s.d.
0.08 °C, 0.01 in salinity); the intermediate water's salinity is solved
so that it shares the deep water's σ₀. All data are simulated.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text
from scipy import optimize

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(157)


# -------------------------------------------------- GOVERNING MODEL ----
def density(s, t):
    """Sea-water density (kg m⁻³) at one standard atmosphere, EOS-80.

    Millero & Poisson, Deep-Sea Res. 28A, 625-629 (1981); UNESCO Tech.
    Pap. Mar. Sci. 44 (1983). s, practical salinity; t, temperature (°C).
    """
    pure = (999.842594 + 6.793952e-2 * t - 9.095290e-3 * t ** 2
            + 1.001685e-4 * t ** 3 - 1.120083e-6 * t ** 4
            + 6.536332e-9 * t ** 5)
    a = (8.24493e-1 - 4.0899e-3 * t + 7.6438e-5 * t ** 2
         - 8.2467e-7 * t ** 3 + 5.3875e-9 * t ** 4)
    b = -5.72466e-3 + 1.0227e-4 * t - 1.6546e-6 * t ** 2
    return pure + a * s + b * s ** 1.5 + 4.8314e-4 * s ** 2


def sigma0(s, t):
    return density(s, t) - 1000.0


CHECK_VALUES = {(35.0, 5.0): 1027.675, (0.0, 5.0): 999.967}  # UNESCO (1983)

# ------------------------------------------------------------- DATA ----
S_LIM, T_LIM = (34.4, 37.0), (0.0, 23.0)
SURFACE, DEEP = (35.7, 20.0), (34.9, 2.5)      # (salinity, θ in °C)
T_INTERMEDIATE = 12.5                          # warm, salty mid-depth water
s_intermediate = optimize.brentq(
    lambda s: sigma0(s, T_INTERMEDIATE) - sigma0(*DEEP), 30.0, 40.0,
    xtol=1e-13)                                # same σ₀ as the deep water
MASSES = {"Surface water": SURFACE,
          "Intermediate water": (s_intermediate, T_INTERMEDIATE),
          "Deep water": DEEP}
depth = np.arange(0.0, 4001.0, 25.0)           # m
NOISE_T, NOISE_S = 0.08, 0.01

# fractions of the three masses with depth: the surface share decays, the
# intermediate water forms a core near 900 m, the rest is deep water
w_mid = 0.35 * np.exp(-((depth - 900.0) / 450.0) ** 2)
w_top = (1 - w_mid) * np.exp(-depth / 700.0)
weights = np.array([w_top, w_mid, 1 - w_top - w_mid])
ends = np.array(list(MASSES.values()))
s_clean, t_clean = weights.T @ ends[:, 0], weights.T @ ends[:, 1]
s_cast = s_clean + rng.normal(0, NOISE_S, depth.size)
t_cast = t_clean + rng.normal(0, NOISE_T, depth.size)

# -------------------------------------------------------- ESTIMATOR ----
s_grid, t_grid = np.meshgrid(np.linspace(*S_LIM, 131),
                             np.linspace(*T_LIM, 116))
sigma_grid = sigma0(s_grid, t_grid)
# mixing conserves heat and salt, so mixtures lie on the straight line
fraction = np.linspace(0, 1, 201)
s_mix = np.interp(fraction, [0, 1], [s_intermediate, DEEP[0]])
t_mix = np.interp(fraction, [0, 1], [T_INTERMEDIATE, DEEP[1]])
sigma_mix = sigma0(s_mix, t_mix)
sigma_ends = sigma0(*DEEP)
excess = sigma_mix.max() - sigma_ends
i_peak = int(np.argmax(sigma_mix))

# ------------------------------------------------------- SELF-CHECK ---
for (s_ref, t_ref), rho_ref in CHECK_VALUES.items():
    assert abs(density(s_ref, t_ref) - rho_ref) < 1e-3, (s_ref, t_ref)
assert np.all(np.diff(sigma_grid, axis=1) > 0)           # denser when saltier
assert np.all(np.diff(sigma0(s_clean, t_clean)) > 0)     # stable cast
assert abs(sigma_mix[0] - sigma_mix[-1]) < 1e-9          # equal end members
assert np.all(sigma_mix[1:-1] > sigma_ends) and excess > 0.05
print(f"fig157: self-check passed (rho(35, 5, 0) = {density(35.0, 5.0):.4f}, "
      f"rho(0, 5, 0) = {density(0.0, 5.0):.4f} kg m⁻³; intermediate water "
      f"S = {s_intermediate:.3f} on sigma0 = {sigma_ends:.3f}; densest "
      f"mixture +{excess:.3f} kg m⁻³ at {100 * fraction[i_peak]:.0f}% deep "
      "water)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 80)
ax = ms.axes(fig, 12, 11, 58, 62)
ax_cbar = ms.axes(fig, 73, 11, 2.5, 62)

# isopycnals as plain lines (the shared one in black, to compare with the
# chord), so that they can be opened wherever type sits on them
levels = np.arange(np.ceil(sigma_grid.min() * 2) / 2, sigma_grid.max(), 0.5)
contours = ax.contour(s_grid, t_grid, sigma_grid,
                      levels=np.sort(np.append(levels, sigma_ends)))
isopycnals = [ax.plot(*segment.T, lw=0.8 if level == sigma_ends else 0.5,
                      color=ms.INK if level == sigma_ends else ms.GREY)[0]
              for level, segments in zip(contours.levels, contours.allsegs)
              for segment in segments]
contours.remove()
LABEL_S = 34.72                            # a clear column left of the cast
for level in levels:
    if sigma0(LABEL_S, T_LIM[1] - 1.0) < level < sigma0(LABEL_S, 6.0):
        t_label = optimize.brentq(lambda t: sigma0(LABEL_S, t) - level, *T_LIM)
        ax.text(LABEL_S, t_label, f"{level:.1f}", fontsize=ms.FS_SMALL,
                color=ms.GREY_DARK, ha="center", va="center")

depth_map = LinearSegmentedColormap.from_list(
    "depth", plt.get_cmap("YlGnBu")(np.linspace(0.3, 1.0, 256)))
points = ax.scatter(s_cast, t_cast, c=depth, cmap=depth_map, s=7,
                    linewidths=0, zorder=3)
ax.plot(s_mix, t_mix, color=ms.VERMILLION, lw=1.1, zorder=4)
ax.plot(s_mix[i_peak], t_mix[i_peak], "o", ms=3.2, mfc="white",
        mec=ms.VERMILLION, mew=0.9, zorder=5)
for (s_end, t_end) in MASSES.values():
    ax.plot(s_end, t_end, "s", ms=4, mfc=ms.INK, mec="white", mew=0.5,
            zorder=6)

ax.annotate("Surface water", SURFACE, xytext=(5, 2),
            textcoords="offset points", fontsize=ms.FS_TICK, va="center")
ax.annotate("Intermediate\nwater", MASSES["Intermediate water"],
            xytext=(-3, 9), textcoords="offset points", fontsize=ms.FS_TICK,
            ha="center", va="bottom", linespacing=1.1)
ax.annotate("Deep water", DEEP, xytext=(3, -4), textcoords="offset points",
            fontsize=ms.FS_TICK, ha="left", va="top")
ax.annotate("Mixing line: every mixture\nis denser than both end\n"
            f"members (up to +{excess:.2f} kg m⁻³)",
            (s_mix[i_peak], t_mix[i_peak]), xytext=(36.95, 3.4),
            fontsize=ms.FS_SMALL, color=ms.VERMILLION, ha="right",
            va="center", linespacing=1.2,
            arrowprops=dict(arrowstyle="-", color=ms.VERMILLION, lw=0.5,
                            shrinkA=2, shrinkB=2.5, relpos=(0.3, 1.0)))
ax.text(0.0, 1.03, "Isopycnals: grey, σ₀ (kg m⁻³); black, the "
        f"σ₀ = {sigma_ends:.2f} of both end members", transform=ax.transAxes,
        fontsize=ms.FS_SMALL, color=ms.GREY_DARK, va="bottom")

ax.set_xlim(*S_LIM)
ax.set_ylim(*T_LIM)
ax.set_xticks(np.arange(34.5, 37.01, 0.5))
ax.set_xlabel("Practical salinity")
ax.set_ylabel("Potential temperature θ (°C)")

cbar = fig.colorbar(points, cax=ax_cbar, ticks=np.arange(0, 4001, 1000))
cbar.ax.invert_yaxis()                         # depth increases downward
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(length=2, width=0.5)
cbar.ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:,.0f}")
cbar.set_label("Depth (m)")

# open the isopycnals where type sits on them, so no stroke crosses a label
fig.canvas.draw()
pad = 1.5 * fig.dpi / 72                 # Text's extent: type without leader
boxes = [Text.get_window_extent(t).padded(pad) for t in ax.texts]
for line in isopycnals:
    xy = line.get_xydata()
    px, py = ax.transData.transform(xy).T
    hidden = np.any([(px > box.x0) & (px < box.x1) & (py > box.y0)
                     & (py < box.y1) for box in boxes], axis=0)
    line.set_ydata(np.where(hidden, np.nan, xy[:, 1]))

ms.assert_aligned([ax, ax_cbar])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig157_ts_diagram.{ext}")
print("fig157_ts_diagram: saved png + pdf")
