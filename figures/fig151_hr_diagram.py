"""Fig. 151 - Hertzsprung-Russell diagram, simulated (single column, 89 mm).

The observer's map of stellar structure, built from three scaling laws:
a main sequence from mass-luminosity and mass-radius power laws, a red
giant branch with its clump, and a white-dwarf cooling sequence at
nearly constant radius. Temperature runs backwards on a log axis, as
convention demands; dashed lines of constant radius show why giants are
bright although cool and white dwarfs faint although hot. The self-check
is that each constant-radius line has slope 4 in log L against log T,
that the Sun lies on the 1 R☉ line, and that every star satisfies
L = 4πR²σT⁴ in SI units with its own R and T.

Model: n = 1,500 stars (1,150 main sequence, 150 giant branch, 80 clump,
120 white dwarfs). Main sequence: masses log-uniform in 0.1-20 M☉ (not
an initial mass function, so that the upper sequence is populated),
L ∝ M^2.3, M^4, M^3.5 in three mass ranges and R ∝ M^0.8, M^0.57, each
with log-normal scatter. Colour repeats the temperature axis. All data
are simulated.
"""

import numpy as np

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(151)

# ------------------------------------------------------------- DATA ----
# IAU 2015 Resolution B3 nominal solar values; CODATA 2018 Stefan-Boltzmann
T_SUN, L_SUN_W, R_SUN_M = 5772.0, 3.828e26, 6.957e8     # K, W, m
SIGMA_SB = 5.670374419e-8                               # W m⁻² K⁻⁴
N_MS, N_RGB, N_CLUMP, N_WD = 1150, 150, 80, 120
RADII = [0.01, 0.1, 1.0, 10.0, 100.0]                   # iso-radius lines
T_LIM, L_LIM = (50000.0, 2000.0), (1e-6, 1e6)           # axis limits


def temperature(lum, radius):
    """Effective temperature (K) from L and R in solar units."""
    return T_SUN * (lum / radius ** 2) ** 0.25


def luminosity(radius, temp):
    """Stefan-Boltzmann in solar units: L = R² (T / T☉)⁴."""
    return radius ** 2 * (temp / T_SUN) ** 4


# main sequence: piecewise mass-luminosity law (Duric 2004, Advanced
# Astrophysics) and the usual two-slope mass-radius law for dwarfs
mass = 10 ** rng.uniform(-1, np.log10(20), N_MS)
lum_ms = np.select([mass < 0.43, mass < 2.0],
                   [0.23 * mass ** 2.3, mass ** 4.0], 1.4 * mass ** 3.5)
rad_ms = np.where(mass < 1.0, mass ** 0.8, mass ** 0.57)
lum_ms *= 10 ** rng.normal(0, 0.10, N_MS)
rad_ms *= 10 ** rng.normal(0, 0.04, N_MS)
temp_ms = temperature(lum_ms, rad_ms)

# giants: the branch cools as it brightens; the clump is a tight knot
lum_rgb = 10 ** rng.uniform(0.9, 3.3, N_RGB)
temp_rgb = (5050 - 520 * (np.log10(lum_rgb) - 0.9)
            + rng.normal(0, 90, N_RGB))
lum_clump = 10 ** rng.normal(1.75, 0.07, N_CLUMP)
temp_clump = rng.normal(4780, 70, N_CLUMP)
lum_giant = np.concatenate([lum_rgb, lum_clump])
temp_giant = np.concatenate([temp_rgb, temp_clump])
rad_giant = np.sqrt(lum_giant) * (T_SUN / temp_giant) ** 2

# white dwarfs: Earth-sized (about 0.0127 R☉ at 0.6 M☉, R ∝ M^(-1/3))
# and cooling at fixed radius
mass_wd = rng.normal(0.60, 0.07, N_WD).clip(0.4, 1.0)
rad_wd = 0.0127 * (mass_wd / 0.6) ** (-1 / 3)
temp_wd = 10 ** rng.uniform(np.log10(5500), np.log10(40000), N_WD)
lum_wd = luminosity(rad_wd, temp_wd)

temp = np.concatenate([temp_ms, temp_giant, temp_wd])
lum = np.concatenate([lum_ms, lum_giant, lum_wd])
rad = np.concatenate([rad_ms, rad_giant, rad_wd])

# constant-radius lines, trimmed to the frame so that nothing is clipped
iso = {}
for radius in RADII:
    t_line = np.geomspace(T_LIM[1], T_LIM[0], 300)
    l_line = luminosity(radius, t_line)
    inside = (l_line >= L_LIM[0]) & (l_line <= L_LIM[1])
    iso[radius] = (t_line[inside], l_line[inside])

# ------------------------------------------------------- SELF-CHECK ---
slopes = [np.polyfit(np.log10(t), np.log10(l), 1)[0] for t, l in iso.values()]
assert np.allclose(slopes, 4.0, atol=1e-9), slopes
assert abs(luminosity(1.0, T_SUN) - 1.0) < 1e-12        # Sun on R = 1 R☉
# every star, in SI units: the nominal solar constants are consistent
# with Stefan-Boltzmann to about 1 part in 10⁴, hence the tolerance
lum_si = 4 * np.pi * (rad * R_SUN_M) ** 2 * SIGMA_SB * temp ** 4
worst = np.abs(lum_si / (lum * L_SUN_W) - 1).max()
assert worst < 1e-3, worst
assert temp.size == 1500
print(f"fig151: self-check passed (iso-radius slopes {min(slopes):.6f} to "
      f"{max(slopes):.6f}; Sun on the 1 R☉ line; L = 4πR²σT⁴ for all "
      f"{temp.size} stars within {worst:.1e} in SI units)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 84)
AX_W, AX_H = 68.0, 68.0                                  # plot area, mm
ax = ms.axes(fig, 16, 11, AX_W, AX_H)
ax.set_xscale("log")
ax.set_yscale("log")

# page angle of a slope-4 line on these axes (temperature reversed)
decades_t = np.log10(T_LIM[0] / T_LIM[1])
decades_l = np.log10(L_LIM[1] / L_LIM[0])
angle = -np.degrees(np.arctan(4 * (AX_H / decades_l) / (AX_W / decades_t)))

# iso-radius lines are broken where their label sits, contour-map style
LABEL_T = {0.01: 3400.0, 0.1: 3400.0, 1.0: 21000.0, 10.0: 21000.0,
           100.0: 7300.0}
GAP = 1.26                                    # half-width of the gap in T
for radius, (t_line, l_line) in iso.items():
    t_lab = LABEL_T[radius]
    for part in (t_line < t_lab / GAP, t_line > t_lab * GAP):
        ax.plot(t_line[part], l_line[part], color=ms.GREY, lw=0.5,
                ls=(0, (4, 2.5)), zorder=1)
    ax.text(t_lab, luminosity(radius, t_lab), f"{radius:g} R☉",
            rotation=angle, rotation_mode="anchor", ha="center",
            va="center", fontsize=ms.FS_SMALL, color=ms.GREY_DARK)

# hot = blue, cool = red; a thin edge keeps the pale mid-tones visible
ax.scatter(temp, lum, c=np.log10(temp), cmap="RdYlBu", vmin=np.log10(2600),
           vmax=np.log10(30000), s=3.2, linewidths=0.15,
           edgecolors=ms.GREY_DARK, rasterized=True, zorder=2)

# the Sun, drawn as its symbol: a ring with a central dot
ax.plot(T_SUN, 1.0, "o", ms=5.2, mfc="none", mec="black", mew=0.8, zorder=4)
ax.plot(T_SUN, 1.0, "o", ms=1.4, color="black", mew=0, zorder=4)
ax.annotate("Sun", xy=(T_SUN, 1.0), xytext=(4200, 1.5e-2),
            fontsize=ms.FS_TICK, ha="center", va="top",
            arrowprops=dict(arrowstyle="-", color=ms.INK, lw=0.5,
                            shrinkA=1, shrinkB=4))

# branch names sit midway between two iso-radius lines (radius given)
MID = 10 ** 0.5
for t_lab, radius, name in ((12500, 10 * MID, "Main\nsequence"),
                            (3350, 500.0, "Red giants"),
                            (2900, 10 * MID, "Red clump"),
                            (25000, 0.001 * MID, "White dwarfs")):
    ax.text(t_lab, luminosity(radius, t_lab), name, fontsize=ms.FS_TICK,
            fontweight="bold", ha="center", va="center", linespacing=1.15)
ax.plot([3950, 4600], [luminosity(10 * MID, 2900), 10 ** 1.75],
        color=ms.INK, lw=0.5)                 # leader from name to clump

ax.text(0.03, 0.03, f"n = {temp.size:,} simulated stars", fontsize=ms.FS_SMALL,
        transform=ax.transAxes, color=ms.GREY_DARK)

ax.set_xlim(*T_LIM)                           # hot on the left, by convention
ax.set_ylim(*L_LIM)
ax.set_xticks([40000, 20000, 10000, 5000, 2500])
ax.set_xticklabels(["40,000", "20,000", "10,000", "5,000", "2,500"])
ax.set_xticks([30000, 15000, 7500, 3500], minor=True)
ax.xaxis.set_minor_formatter(lambda _v, _pos: "")
ax.set_yticks(10.0 ** np.arange(-6, 7, 2))
ax.set_yticks(10.0 ** np.arange(-5, 7, 2), minor=True)
# Unicode exponents: mathtext superscripts would fall below 5 pt
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
ax.yaxis.set_major_formatter(
    lambda v, _pos: "1" if v == 1 else
    "10" + str(int(round(np.log10(v)))).translate(SUPERSCRIPT))
ax.yaxis.set_minor_formatter(lambda _v, _pos: "")
ax.set_xlabel("Effective temperature (K)")
ax.set_ylabel("Luminosity (L☉)")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig151_hr_diagram.{ext}")
print("fig151_hr_diagram: saved png + pdf")
