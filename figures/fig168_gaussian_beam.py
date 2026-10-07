"""Fig. 168 - Gaussian beam focus, phase and curvature (double column, 183 mm).

Everything about a focused TEM₀₀ laser beam follows from two numbers,
the wavelength and the waist. Panel a maps the intensity I(r, z) through
the focus with the 1/e² radius ±w(z), the far-field asymptotes, the
waist and Rayleigh range dimensioned, and wavefronts from R(z); panel b
plots w(z)/w₀, the Gouy phase and the wavefront curvature 1/R on the
same z axis. The self-check is that w(z_R) = √2·w₀, the Gouy phase at
z_R is π/4, the curvature is extremal at z_R with |R| = 2z_R, the
far-field half-angle equals λ/(πw₀), and the power through every
transverse plane (numerical ∫ I·2πr dr) is the same to 1e-4.

Model: paraxial TEM₀₀ beam, λ = 1,064 nm, w₀ = 20 µm, so z_R = πw₀²/λ;
w(z) = w₀√(1 + z²/z_R²), R(z) = z(1 + z_R²/z²), ψ(z) = arctan(z/z_R),
I/I₀ = (w₀/w)²·exp(−2r²/w²). Wavefronts are z = z_c − r²/(2R(z_c)) with
the sag magnified 500 times (at true scale it is under 2 µm).
No data: every curve is computed from the equations.
"""

import numpy as np
from scipy import integrate, optimize

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
WAVELENGTH = 1.064e-3                 # mm (Nd:YAG fundamental, 1,064 nm)
W0 = 20e-3                            # waist radius (mm) = 20 µm
Z_R = np.pi * W0 ** 2 / WAVELENGTH    # Rayleigh range (mm)
THETA = WAVELENGTH / (np.pi * W0)     # far-field half-angle (rad)
Z_MAX, R_MAX = 4 * Z_R, 5 * W0        # plotted half-ranges (mm)
SAG_GAIN = 500.0                      # wavefront sag magnification


def radius(z):
    return W0 * np.sqrt(1 + (z / Z_R) ** 2)          # 1/e² radius w(z)


def curvature(z):
    return z / (z ** 2 + Z_R ** 2)                   # 1/R(z)


def gouy(z):
    return np.arctan(z / Z_R)


def intensity(r, z):
    """I/I₀ for unit on-axis intensity at the waist."""
    w = radius(z)
    return (W0 / w) ** 2 * np.exp(-2 * r ** 2 / w ** 2)


# ------------------------------------------------------------ SOLVER ---
z = np.linspace(-Z_MAX, Z_MAX, 641)
r = np.linspace(-R_MAX, R_MAX, 321)
image = intensity(r[:, None], z[None, :])

# power through each plane: radial integral far beyond the widest beam
r_int = np.linspace(0, 8 * radius(Z_MAX), 6001)
power = integrate.trapezoid(
    intensity(r_int[:, None], z[None, :]) * 2 * np.pi * r_int[:, None],
    r_int, axis=0)
z_extremum = optimize.minimize_scalar(
    lambda v: -curvature(v), bounds=(0.1 * Z_R, 5 * Z_R), method="bounded",
    options=dict(xatol=1e-12)).x
z_far = 1e5 * Z_R
slope_far = (radius(2 * z_far) - radius(z_far)) / z_far

# ------------------------------------------------------- SELF-CHECK ---
assert abs(radius(Z_R) / W0 - np.sqrt(2)) < 1e-12
assert abs(gouy(Z_R) - np.pi / 4) < 1e-12
assert abs(z_extremum / Z_R - 1) < 1e-5, z_extremum
assert abs(1 / curvature(Z_R) - 2 * Z_R) < 1e-12 * Z_R
assert abs(slope_far / THETA - 1) < 1e-8, slope_far
assert np.ptp(power) < 1e-4 * power.mean()
assert abs(power.mean() / (np.pi * W0 ** 2 / 2) - 1) < 1e-4   # P = πw₀²I₀/2
print(f"fig168: self-check passed (z_R = {Z_R:.4f} mm; w(z_R)/w0 = "
      f"{radius(Z_R) / W0:.6f}; Gouy(z_R) = {gouy(Z_R):.6f} rad; 1/R "
      f"extremal at {z_extremum / Z_R:.6f} z_R, R = {1 / curvature(Z_R):.4f}"
      f" mm; half-angle {1e3 * THETA:.3f} mrad; power spread "
      f"{np.ptp(power) / power.mean():.1e} over {z.size} planes)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 78)
ax_a = ms.axes(fig, 14, 11, 90, 60)
ax_cbar = ms.axes(fig, 106, 11, 2.4, 60)
B_X, B_W, B_H = 131.0, 47.0, 17.0
ax_w = ms.axes(fig, B_X, 54, B_W, B_H)
ax_g = ms.axes(fig, B_X, 32.5, B_W, B_H, sharex=ax_w)
ax_c = ms.axes(fig, B_X, 11, B_W, B_H, sharex=ax_w)
UM = 1e3                                             # mm to µm
ZR_TEX = r"$z_\mathrm{R}$"            # mathtext subscript: set at FS_MATH

# a, intensity map (every pixel drawn, rasterized) and the beam geometry
mesh = ax_a.imshow(image, origin="lower", aspect="auto", cmap="magma",
                   vmin=0, vmax=1, interpolation="bilinear", rasterized=True,
                   extent=[-Z_MAX, Z_MAX, -R_MAX * UM, R_MAX * UM])
r_front = np.linspace(-1, 1, 81)
for z_c in (np.arange(-7, 8) + 0.5) * 0.5 * Z_R:
    r_c = r_front * radius(z_c)                      # inside the 1/e² edge
    ax_a.plot(z_c - SAG_GAIN * r_c ** 2 * curvature(z_c) / 2, r_c * UM,
              color="white", lw=0.4, alpha=0.6)
for sign in (-1, 1):
    ax_a.plot(z, sign * radius(z) * UM, color="white", lw=0.9)
    ax_a.plot(z, sign * THETA * z * UM, color="white", lw=0.6,
              ls=(0, (4, 3)))

dimension = dict(arrowstyle="<->", color=ms.SKY, lw=0.9, shrinkA=0,
                 shrinkB=0, mutation_scale=5)
ax_a.annotate("", xy=(0, W0 * UM), xytext=(0, 0), arrowprops=dimension)
ax_a.text(0, 1.55 * W0 * UM, f"w₀ = {W0 * UM:.0f} µm", color="white",
          ha="center", va="bottom", fontsize=ms.FS_BODY)
y_dim = -3.4 * W0 * UM
ax_a.annotate("", xy=(Z_R, y_dim), xytext=(0, y_dim),
              arrowprops={**dimension, "color": "white"})
for z_mark in (0, Z_R):                              # extension lines
    ax_a.plot([z_mark, z_mark], [y_dim - 5, -radius(z_mark) * UM - 4],
              color="white", lw=0.4)
ax_a.text(Z_R / 2, y_dim - 8, f"{ZR_TEX} = {Z_R:.2f} mm", color="white",
          ha="center", va="top", fontsize=ms.FS_MATH)
ax_a.text(-0.96 * Z_MAX, 0.93 * R_MAX * UM, "Solid, 1/e² radius ±w(z)",
          color="white", fontsize=ms.FS_TICK, va="top")
ax_a.text(-0.96 * Z_MAX, -0.93 * R_MAX * UM,
          f"λ = {WAVELENGTH * 1e6:,.0f} nm", color="white",
          fontsize=ms.FS_TICK, va="bottom")
ax_a.text(0.96 * Z_MAX, 0.93 * R_MAX * UM,
          f"Thin, wavefronts (sag ×{SAG_GAIN:.0f})", color="white",
          fontsize=ms.FS_TICK, ha="right", va="top")
ax_a.text(0.96 * Z_MAX, -0.93 * R_MAX * UM, "Dashed, far-field asymptote "
          f"θ = λ/(πw₀) = {1e3 * THETA:.1f} mrad", color="white",
          fontsize=ms.FS_TICK, ha="right", va="bottom")
ax_a.set_xlabel("Axial position z (mm)")
ax_a.set_ylabel("Radial position r (µm)")
cbar = fig.colorbar(mesh, cax=ax_cbar, ticks=np.linspace(0, 1, 6))
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(length=2, width=0.5)
cbar.set_label("Intensity I/I₀")

# b, the three functions of z that define the beam, with ±z_R marked
z_fine = np.linspace(-Z_MAX, Z_MAX, 801)
below, lower = ((5, -1), "left", "top"), ((10, -12), "left", "top")
rows = ((ax_w, radius(z_fine) / W0, ms.INK, np.sqrt(2), "√2", below),
        (ax_g, gouy(z_fine), ms.BLUE, np.pi / 4, "π/4", below),
        (ax_c, curvature(z_fine), ms.VERMILLION, 1 / (2 * Z_R),
         f"1/(2{ZR_TEX})", lower))
for ax, values, colour, at_zr, name, (offset, ha, va) in rows:
    for z_mark in (-Z_R, Z_R):
        ax.axvline(z_mark, color=ms.GREY_LIGHT, lw=0.6, zorder=0)
    ax.plot(z_fine, values, color=colour, lw=1.2)
    ax.plot(Z_R, at_zr, "o", ms=3.2, mfc="white", mec=colour, mew=0.8,
            zorder=4)
    ax.annotate(name, xy=(Z_R, at_zr), xytext=offset,
                textcoords="offset points", fontsize=ms.FS_MATH,
                color=colour, ha=ha, va=va)
    ax.set_xlim(-Z_MAX, Z_MAX)
ax_w.set_ylim(0.6, 4.4)
ax_w.set_yticks([1, 2, 3, 4])
ax_w.set_ylabel("w(z)/w₀")
ax_g.set_ylim(-1.75, 1.75)
ax_g.set_yticks([-np.pi / 2, 0, np.pi / 2], ["−π/2", "0", "π/2"])
ax_g.set_ylabel("Gouy phase ψ")
ax_c.set_ylim(-0.52, 0.52)
ax_c.set_yticks([-0.4, 0, 0.4])
ax_c.set_ylabel("1/R (mm⁻¹)")
ax_c.set_xlabel("Axial position z (mm)")
for ax in (ax_w, ax_g):
    ax.tick_params(labelbottom=False)
for z_mark, name in ((-Z_R, f"−{ZR_TEX}"), (Z_R, f"+{ZR_TEX}")):
    ax_w.annotate(name, xy=(z_mark, 1), xycoords=ax_w.get_xaxis_transform(),
                  xytext=(0, 2), textcoords="offset points", ha="center",
                  va="bottom", fontsize=ms.FS_MATH, color=ms.GREY_DARK)
fig.align_ylabels([ax_w, ax_g, ax_c])

ms.panel_label(ax_a, "a", dx_pt=-30, dy_pt=6)
ms.panel_label(ax_w, "b", dx_pt=-30, dy_pt=6)
ms.assert_aligned([ax_a, ax_c], edges=("bottom",))
ms.assert_aligned([ax_a, ax_w], edges=("top",))
ms.assert_aligned([ax_w, ax_g, ax_c], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig168_gaussian_beam.{ext}")
print("fig168_gaussian_beam: saved png + pdf")
