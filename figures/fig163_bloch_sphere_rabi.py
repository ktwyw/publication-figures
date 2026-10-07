"""Fig. 163 - Rabi oscillations on the Bloch sphere (double column, 183 mm).

The two standard views of a driven two-level system side by side: the
Bloch vector precessing about the effective field (a great circle
through both poles on resonance, a tilted cone that never reaches |1⟩
when detuned) and the excited-state population against time, numerical
integration on top of the Rabi formula. The self-check is that the
Bloch vector keeps unit length to 1e-8, that the numerical populations
match P = (Ω²/Ω_eff²)·sin²(Ω_eff·t/2) to 1e-6, that the maximum
transfer, reached at t = π/Ω_eff, equals Ω²/(Ω² + Δ²), and that the
Bloch vector stays on its cone about Ω_eff (constant projection).

Model: rotating-frame Hamiltonian H = (ħ/2)(Ω·σx + Δ·σz), start in |0⟩,
detunings Δ/Ω = 0, 1, 2; Schrödinger equation by solve_ivp (DOP853,
rtol 1e-11); Bloch vector r = ⟨σ⟩ obeys dr/dt = Ω_eff × r with
Ω_eff = (Ω, 0, Δ). Orthographic view, far-side arcs drawn pale. No data:
every curve is computed from the equations.
"""

import numpy as np
from matplotlib.lines import Line2D
from scipy import integrate

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
OMEGA = 1.0                                   # Rabi frequency: 1/time unit
DETUNINGS = [0.0, 1.0, 2.0]                   # Δ/Ω; first two on the sphere
T_END = 4 * np.pi / OMEGA                     # two resonant Rabi periods
N_POINTS = 41                                 # numerical samples in panel b
ELEV, AZIM = 18.0, 45.0                       # camera angles (degrees)
SIGMA = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


def rabi_formula(t, detuning):
    """Excited-state population for a start in |0⟩ (Rabi 1937)."""
    w_eff = np.hypot(OMEGA, detuning)
    return (OMEGA / w_eff) ** 2 * np.sin(w_eff * t / 2) ** 2


# ------------------------------------------------------------ SOLVER ---
def evolve(detuning):
    """Integrate i dψ/dt = Hψ from |0⟩; dense output for any time."""
    hamiltonian = 0.5 * (OMEGA * SIGMA[0] + detuning * SIGMA[2])
    return integrate.solve_ivp(lambda _t, psi: -1j * hamiltonian @ psi,
                               (0, T_END), np.array([1, 0], dtype=complex),
                               method="DOP853", rtol=1e-11, atol=1e-13,
                               dense_output=True).sol


def bloch(psi):
    """r = ⟨ψ|σ|ψ⟩ for states stacked along the last axis."""
    return np.einsum("it,kij,jt->kt", psi.conj(), SIGMA, psi).real


t_fine = np.linspace(0, T_END, 1201)
t_dots = np.linspace(0, T_END, N_POINTS)
runs = {}
for detuning in DETUNINGS:
    solution = evolve(detuning)
    w_eff = np.hypot(OMEGA, detuning)
    runs[detuning] = dict(
        r=bloch(solution(t_fine)), p_fine=np.abs(solution(t_fine)[1]) ** 2,
        p_dots=np.abs(solution(t_dots)[1]) ** 2,
        p_peak=np.abs(solution(np.pi / w_eff)[1]) ** 2,
        axis=np.array([OMEGA, 0, detuning]) / w_eff)

# ------------------------------------------------------- SELF-CHECK ---
norm_error = formula_error = cone_error = 0.0
for detuning, run in runs.items():
    norm_error = max(norm_error,
                     np.abs(np.linalg.norm(run["r"], axis=0) - 1).max())
    formula_error = max(formula_error, np.abs(
        run["p_fine"] - rabi_formula(t_fine, detuning)).max())
    ceiling = OMEGA ** 2 / (OMEGA ** 2 + detuning ** 2)
    assert abs(run["p_peak"] - ceiling) < 1e-6, (detuning, run["p_peak"])
    assert run["p_fine"].max() <= ceiling + 1e-6
    # precession about Ω_eff: the projection on the axis never changes
    cone_error = max(cone_error, np.ptp(run["axis"] @ run["r"]))
assert norm_error < 1e-8, norm_error
assert formula_error < 1e-6, formula_error
assert cone_error < 1e-8, cone_error
assert np.abs(runs[0.0]["r"][0]).max() < 1e-8        # resonant: x stays 0
print(f"fig163: self-check passed (| |r| − 1 | ≤ {norm_error:.1e}; "
      f"population vs Rabi formula ≤ {formula_error:.1e}; maximum transfer "
      + ", ".join(f"{run['p_peak']:.4f}" for run in runs.values())
      + f" for Δ/Ω = 0, 1, 2; cone drift ≤ {cone_error:.1e})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 80)
ax_a = ms.axes(fig, 6, 1, 78, 78, projection="3d")
ax_b = ms.axes(fig, 104, 11, 56, 61)
COLOURS = {0.0: ms.BLUE, 1.0: ms.VERMILLION, 2.0: ms.GREEN}
MARKERS = {0.0: "o", 1.0: "s", 2.0: "^"}
NAMES = {0.0: "Δ = 0", 1.0: "Δ = Ω", 2.0: "Δ = 2Ω"}

elev, azim = np.radians([ELEV, AZIM])
camera = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim),
                   np.sin(elev)])             # unit vector towards the viewer


def draw(points, colour, lw, zorder, back_alpha=0.3, **kwargs):
    """Line on the sphere: near side solid, far side pale."""
    near = points.T @ camera >= 0
    for side, alpha in ((near, 1.0), (~near, back_alpha)):
        # keep the first point past each crossing so the two parts join
        keep = side | np.roll(side, 1) | np.roll(side, -1)
        ax_a.plot(*np.where(keep, points, np.nan), color=colour, lw=lw,
                  alpha=alpha, zorder=zorder, **kwargs)


# a, wireframe: meridians and parallels every 30°, equator and limb darker
angle = np.linspace(0, 2 * np.pi, 241)
ring = np.array([np.cos(angle), np.sin(angle), np.zeros_like(angle)])
for phi in np.radians(np.arange(0, 180, 30)):
    meridian = np.array([np.cos(phi) * ring[0], np.sin(phi) * ring[0],
                         ring[1]])
    draw(meridian, ms.GREY_LIGHT, 0.4, 1, back_alpha=0.45)
for latitude in np.radians([-60, -30, 30, 60]):
    parallel = np.cos(latitude) * ring + [[0], [0], [np.sin(latitude)]]
    draw(parallel, ms.GREY_LIGHT, 0.4, 1, back_alpha=0.45)
draw(ring, ms.GREY, 0.6, 2, back_alpha=0.45)
right = np.cross([0, 0, 1], camera) / np.linalg.norm(np.cross([0, 0, 1],
                                                              camera))
up = np.cross(camera, right)
limb = np.outer(right, ring[0]) + np.outer(up, ring[1])
ax_a.plot(*limb, color=ms.GREY, lw=0.6, zorder=2)

AXIS_END, LABEL_AT = 1.45, 1.66            # x and y are foreshortened
for direction, name, scale in (([1, 0, 0], "x", 1.0), ([0, 1, 0], "y", 1.0),
                               ([0, 0, 1], "|0⟩", 0.86)):
    direction = scale * np.array(direction, dtype=float)
    ax_a.plot(*np.outer(direction, [0, AXIS_END]), color=ms.GREY_DARK,
              lw=0.6, zorder=3)
    ax_a.text(*(LABEL_AT * direction), name, ha="center", va="center",
              fontsize=ms.FS_BODY)
ax_a.text(0, 0, -1.22, "|1⟩", ha="center", va="center", fontsize=ms.FS_BODY)

# trajectories of the first two detunings and the axis each precesses about
for detuning in DETUNINGS[:2]:
    run, colour = runs[detuning], COLOURS[detuning]
    period = t_fine <= 2 * np.pi / np.hypot(OMEGA, detuning)   # one turn
    draw(run["r"][:, period], colour, 1.4, 5)
tilted = runs[1.0]["axis"]
ax_a.plot(*np.outer(tilted, [0, AXIS_END]), color=COLOURS[1.0], lw=0.8,
          ls=(0, (3, 2)), zorder=4)
ax_a.text(*(LABEL_AT * tilted), "Ω$_{\\mathrm{eff}}$", color=COLOURS[1.0],
          ha="center", va="center", fontsize=ms.FS_MATH)
ax_a.plot([0], [0], [1], "o", ms=4, color=ms.INK, mew=0, zorder=7)
ax_a.plot([0], [0], [-1], "o", ms=3.4, mfc="white", mec=ms.INK, mew=0.7,
          zorder=7)

ax_a.computed_zorder = False              # layers as given, not by depth
ax_a.set_proj_type("ortho")
ax_a.view_init(elev=ELEV, azim=AZIM)
ax_a.set_box_aspect((1, 1, 1), zoom=1.32)
for set_lim in (ax_a.set_xlim, ax_a.set_ylim, ax_a.set_zlim):
    set_lim(-1.3, 1.3)
ax_a.set_axis_off()
fig.text(54 / 183, 73.5 / 80, "Start in |0⟩ (filled dot)",
         fontsize=ms.FS_TICK, va="center")
fig.text(6 / 183, 9 / 80, "Resonant, Δ = 0:\ngreat circle through |1⟩",
         color=COLOURS[0.0], fontsize=ms.FS_TICK, va="center",
         linespacing=1.2)
fig.text(88 / 183, 9 / 80, "Detuned, Δ = Ω: cone about the\ndashed axis, "
         "turning back at z = 0", color=COLOURS[1.0], fontsize=ms.FS_TICK,
         ha="right", va="center", linespacing=1.2)

# b, populations: formula as lines, integration as points
x_fine, x_dots = OMEGA * t_fine / np.pi, OMEGA * t_dots / np.pi
for detuning, run in runs.items():
    colour = COLOURS[detuning]
    ceiling = OMEGA ** 2 / (OMEGA ** 2 + detuning ** 2)
    ax_b.plot(x_fine, rabi_formula(t_fine, detuning), color=colour, lw=1.0)
    ax_b.plot(x_dots, run["p_dots"], MARKERS[detuning], color=colour, ms=2.6,
              mew=0)
    ax_b.text(4.12, ceiling, f"{NAMES[detuning]}\nmax {ceiling:g}",
              color=colour, fontsize=ms.FS_TICK, va="center",
              linespacing=1.15)
ax_b.plot([1, 1], [0, 1], color=ms.GREY, lw=0.6, ls=(0, (4, 3)), zorder=0)
ax_b.text(1, 1.035, "π pulse, t = π/Ω", ha="center", va="bottom",
          fontsize=ms.FS_TICK, color=ms.GREY_DARK)
ax_b.legend(handles=[Line2D([], [], color=ms.GREY_DARK, lw=1.0,
                            label="Rabi formula"),
                     Line2D([], [], color=ms.GREY_DARK, marker="o", ms=2.6,
                            mew=0, ls="none", label="Numerical (solve_ivp)")],
            loc="upper right", ncol=2, frameon=False,
            bbox_to_anchor=(1.0, 1.02))
ax_b.set_xlim(0, 4)
ax_b.set_xticks(np.arange(5))
ax_b.set_ylim(0, 1.24)
ax_b.set_yticks(np.arange(0, 1.01, 0.2))
ax_b.spines["left"].set_bounds(0, 1)
ax_b.set_xlabel("Time Ωt/π")
ax_b.set_ylabel("Excited-state population P")
ax_b.yaxis.set_label_coords(-0.115, 0.5 / 1.24)

ms.panel_label(ax_b, "b", dx_pt=-28)
# a 3-D axes has no plot-area corner: its letter is set level with b
fig.text(4 / 183, 73.4 / 80, "a", fontsize=ms.FS_PANEL, fontweight="bold",
         color="black", ha="left", va="bottom")
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig163_bloch_sphere_rabi.{ext}")
print("fig163_bloch_sphere_rabi: saved png + pdf")
