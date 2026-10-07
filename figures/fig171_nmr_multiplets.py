"""Fig. 171 - First-order ¹H NMR multiplets and integrals (1.5 column, 120 mm).

How to draw a simulated ¹H spectrum the way a spectroscopist reads it:
an ethanol-like A₃M₂X spin system on a reversed ppm axis, every line
placed by the first-order rule (each coupled spin-½ neighbour splits
every line in two, J apart), the running integral as a step trace with
the proton count beside each step, and the triplet and quartet expanded
on a hertz scale so the splitting can be read off. The self-check is
that the numerical integrals over the three multiplet windows are in
the ratio 3 : 2 : 1 within 1%, that line intensities follow Pascal's
triangle, and that adjacent lines are J/ν₀ apart in ppm.

Model: CH₃ δ 1.22 (3 H, coupled to 2 H), CH₂ δ 3.69 (2 H, coupled to
3 H), OH δ 2.61 (1 H, exchange-decoupled singlet); ³J = 7.0 Hz;
ν₀ = 90 MHz; Lorentzian lines, 0.6 Hz full width at half maximum.
No data: every curve is computed from the equations.
"""

import numpy as np
from scipy import integrate, signal, special

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
NU0_MHZ = 90.0                    # spectrometer frequency: 1 ppm = 90 Hz
J_HZ = 7.0                        # vicinal ³J(H,H)
FWHM_HZ = 0.6                     # Lorentzian line width
# group: (shift in ppm, protons, coupled neighbours); illustrative values
# typical of ethanol in CDCl₃
GROUPS = {"CH₂": (3.69, 2, 3), "OH": (2.61, 1, 0), "CH₃": (1.22, 3, 2)}
PPM_LIM = (4.5, 0.5)              # reversed: high shift on the left
WINDOW_PPM = 0.5                  # half-width of each integration window


def multiplet(shift_ppm, protons, neighbours):
    """Line positions (ppm) and intensities by successive splitting."""
    offsets, weights = np.zeros(1), np.full(1, float(protons))
    for _ in range(neighbours):   # one more spin-½ neighbour: ±J/2, halved
        offsets = np.concatenate([offsets - J_HZ / 2, offsets + J_HZ / 2])
        weights = np.concatenate([weights, weights]) / 2
        offsets, merge = np.unique(np.round(offsets, 9), return_inverse=True)
        weights = np.bincount(merge, weights)
    return shift_ppm + offsets / NU0_MHZ, weights


def lorentzian(ppm, centre_ppm, area):
    """Absorption line of unit-normalised area (per Hz) at centre_ppm."""
    half = FWHM_HZ / 2
    d_hz = (ppm - centre_ppm) * NU0_MHZ
    return area * half / np.pi / (d_hz ** 2 + half ** 2)


# ------------------------------------------------------------ SOLVER ---
lines = {name: multiplet(*spec) for name, spec in GROUPS.items()}
ppm = np.linspace(*PPM_LIM, 40001)                 # 0.009 Hz per point
spectrum = sum(lorentzian(ppm, c, a)
               for centres, areas in lines.values()
               for c, a in zip(centres, areas))
hz = (PPM_LIM[0] - ppm) * NU0_MHZ                  # increasing left to right
running = integrate.cumulative_trapezoid(spectrum, hz, initial=0.0)
window_area = {}
for name, (shift, _n, _k) in GROUPS.items():
    inside = np.abs(ppm - shift) <= WINDOW_PPM
    window_area[name] = integrate.trapezoid(spectrum[inside], hz[inside])

# ------------------------------------------------------- SELF-CHECK ---
unit = window_area["OH"]
ratios = {name: area / unit for name, area in window_area.items()}
for name, (_shift, protons, _k) in GROUPS.items():
    assert abs(ratios[name] / protons - 1) < 0.01, ratios
    centres, areas = lines[name]
    n = centres.size - 1
    pascal = special.comb(n, np.arange(n + 1))
    assert np.allclose(areas, protons * pascal / 2 ** n, rtol=1e-12)
    assert np.allclose(np.diff(centres), J_HZ / NU0_MHZ, rtol=0, atol=1e-10)
# the same two rules read back from the synthesised line shape
grid_ppm = abs(ppm[1] - ppm[0])
for name in ("CH₂", "CH₃"):
    shift, protons, _k = GROUPS[name]
    near = np.abs(ppm - shift) <= 0.2
    tops = signal.find_peaks(spectrum[near], prominence=0.01)[0]
    spacing = np.abs(np.diff(ppm[near][tops]))
    assert np.all(np.abs(spacing - J_HZ / NU0_MHZ) <= 2 * grid_ppm), spacing
    heights = spectrum[near][tops]
    assert np.allclose(heights / heights.sum(),
                       lines[name][1] / protons, atol=0.005)
print("fig171: self-check passed (integrals CH₃ : CH₂ : OH = "
      f"{ratios['CH₃']:.3f} : {ratios['CH₂']:.3f} : 1; Pascal intensities "
      f"1:2:1 and 1:3:3:1; line spacing {J_HZ / NU0_MHZ:.5f} ppm = "
      f"{J_HZ:g} Hz at {NU0_MHZ:g} MHz)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 92)
ax_a = ms.axes(fig, 10, 53, 104, 32)
ax_b = ms.axes(fig, 10, 10, 46, 24)
ax_c = ms.axes(fig, 68, 10, 46, 24)
LINE = ms.INK
TRACE = ms.BLUE                    # the integral, the one coloured layer
y = spectrum / spectrum.max()

# a, whole spectrum with the running integral drawn above it
STEP_BASE, STEP_GAIN = 1.25, 0.13                  # trace = base + gain·∫
total = sum(protons for _s, protons, _k in GROUPS.values())
trace = STEP_BASE + STEP_GAIN * running / running[-1] * total
ax_a.plot(ppm, y, color=LINE, lw=0.7)
ax_a.plot(ppm, trace, color=TRACE, lw=0.9)
for name, (shift, protons, neighbours) in GROUPS.items():
    kind = {0: "singlet", 2: "triplet", 3: "quartet"}[neighbours]
    top = y[np.abs(ppm - shift) <= 0.2].max()
    ax_a.text(shift, top + 0.05, f"{name}, {kind}", ha="center",
              va="bottom", fontsize=ms.FS_TICK)
    # the step's value sits right of its riser, under the upper plateau
    right = lines[name][0].min() - 0.07
    ax_a.text(right, np.interp(right, ppm[::-1], trace[::-1]) - 0.05,
              f"{ratios[name]:.2f} ({protons} H)", color=TRACE, ha="left",
              va="top", fontsize=ms.FS_TICK)
ax_a.text(0.0, 1.0, f"CH₃–CH₂–OH, ν₀ = {NU0_MHZ:g} MHz, "
          f"³J = {J_HZ:g} Hz, line width {FWHM_HZ:g} Hz",
          transform=ax_a.transAxes, ha="left", va="top",
          fontsize=ms.FS_TICK)
ax_a.text(PPM_LIM[0] - 0.03, STEP_BASE + 0.05, "Running integral",
          color=TRACE, ha="left", va="bottom", fontsize=ms.FS_TICK)
ax_a.set_xlim(*PPM_LIM)
ax_a.set_ylim(-0.04, 2.4)
ax_a.set_xticks(np.arange(4.5, 0.4, -0.5))
ax_a.set_xlabel("Chemical shift δ (ppm)")
ax_a.set_yticks([])
ax_a.spines["left"].set_visible(False)             # intensity is relative


# b and c, each multiplet expanded; the top scale is hertz from its centre
def expand(ax, name):
    shift, protons, _k = GROUPS[name]
    centres, areas = lines[name]
    half_span = 0.14
    near = np.abs(ppm - shift) <= half_span
    scale = spectrum[near].max()
    ax.plot(ppm[near], spectrum[near] / scale, color=LINE, lw=0.8)
    n = centres.size - 1
    for centre, weight in zip(centres, special.comb(n, np.arange(n + 1))):
        apex = lorentzian(centre, centres, areas).sum() / scale
        ax.text(centre, apex + 0.04, f"{weight:.0f}", ha="center",
                va="bottom", fontsize=ms.FS_TICK, color=ms.GREY_DARK)
    # dimension J, centre to centre, between the two highest-shift lines
    left, right = centres[-1], centres[-2]
    level = 0.08
    ax.annotate("", xy=(right, level), xytext=(left, level),
                arrowprops=dict(arrowstyle="<->", color=TRACE, lw=0.6,
                                shrinkA=0, shrinkB=0, mutation_scale=5))
    ax.text((left + right) / 2, level + 0.05, f"J = {J_HZ:.1f} Hz",
            color=TRACE, ha="center", va="bottom", fontsize=ms.FS_SMALL)
    ax.set_xlim(shift + half_span, shift - half_span)
    ax.set_ylim(-0.03, 1.25)
    ax.set_xticks(np.round(shift + np.array([0.1, 0.0, -0.1]), 2))
    ax.set_xlabel("δ (ppm)")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    top = ax.secondary_xaxis(
        "top", functions=(lambda d: (d - shift) * NU0_MHZ,
                          lambda v: shift + v / NU0_MHZ))
    top.set_xticks(np.round((centres - shift) * NU0_MHZ, 6))
    top.set_xlabel(f"Offset from δ {shift:.2f} (Hz)")


expand(ax_b, "CH₂")
expand(ax_c, "CH₃")

ms.panel_label(ax_a, "a", dx_pt=-12, dy_pt=4)
ms.panel_label(ax_b, "b", dx_pt=-12, dy_pt=16)
ms.panel_label(ax_c, "c", dx_pt=-12, dy_pt=16)
ms.assert_aligned([ax_b, ax_c])
ms.assert_aligned([ax_a, ax_b], edges=("left",))
ms.assert_aligned([ax_a, ax_c], edges=("right",))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig171_nmr_multiplets.{ext}")
print("fig171_nmr_multiplets: saved png + pdf")
