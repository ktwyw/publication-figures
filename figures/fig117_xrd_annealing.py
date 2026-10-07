"""Fig. 117 - Diffraction patterns across an annealing series (89 mm).

Extends fig051, which compares two samples: here four patterns form an
ordered series in which reflections grow out of an amorphous hump and
sharpen (pseudo-Voigt peaks, one width per condition). Traces are
labelled directly in the right margin in a light-to-dark sequence,
Miller indices sit on the local maximum of the top trace, and reference
sticks run below. The self-check is that scipy.signal.find_peaks on the
600 °C trace recovers all six reference reflections within 0.1° (the
(105)/(211) doublet as two), finds none in the as-deposited trace, and
that the measured width of the strongest reflection falls from 450 °C
to 600 °C.

Statistics: one pattern per condition, 2,501 points, normalised to the
strongest reflection of the 600 °C pattern and offset vertically;
sticks, reference positions and relative intensities. All patterns are
simulated.
"""

import numpy as np
from matplotlib.colors import to_hex, to_rgb
from matplotlib.ticker import MultipleLocator
from scipy import signal

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(1717)
TWO_THETA = np.linspace(20, 70, 2501)             # degrees, 0.02° steps
STEP = TWO_THETA[1] - TWO_THETA[0]
# (2θ position in degrees, relative intensity, Miller index)
REFLECTIONS = [(25.3, 1.00, "(101)"), (37.8, 0.22, "(004)"),
               (48.0, 0.30, "(200)"), (53.9, 0.19, "(105)"),
               (55.1, 0.18, "(211)"), (62.7, 0.14, "(204)")]
# condition: (crystalline fraction, peak FWHM in degrees)
SERIES = {"As deposited": (0.00, 1.6), "300 °C": (0.06, 1.3),
          "450 °C": (0.62, 0.62), "600 °C": (1.00, 0.34)}
MID_BLUE = to_hex((np.array(to_rgb(ms.SKY)) + to_rgb(ms.BLUE)) / 2)
COLOURS = [ms.GREY, ms.SKY, MID_BLUE, ms.BLUE]    # ordered: light to dark
OFFSET = 0.62                                     # vertical step per trace
NOISE = 0.008


# ------------------------------------------------------------- DATA ----
def pseudo_voigt(x, centre, fwhm, eta=0.4):
    """Unit-height mix of a Lorentzian (weight eta) and a Gaussian."""
    gauss = np.exp(-4 * np.log(2) * ((x - centre) / fwhm) ** 2)
    lorentz = 1 / (1 + 4 * ((x - centre) / fwhm) ** 2)
    return eta * lorentz + (1 - eta) * gauss


raw = {}
for name, (fraction, fwhm) in SERIES.items():
    crystalline = sum(height * pseudo_voigt(TWO_THETA, centre, fwhm)
                      for centre, height, _hkl in REFLECTIONS)
    amorphous = (0.10 * (1 - fraction)
                 * np.exp(-0.5 * ((TWO_THETA - 27) / 5.5) ** 2))
    raw[name] = (fraction * crystalline * (0.34 / fwhm) ** 0.35 + amorphous
                 + rng.normal(0, NOISE, TWO_THETA.size))
scale = raw["600 °C"].max()
patterns = {name: trace / scale for name, trace in raw.items()}

# ------------------------------------------------------- SELF-CHECK ---
# a reflection: well above the noise ripple and under 2° wide, which
# excludes the amorphous hump (about 13° wide)
SHARP = dict(prominence=8 * NOISE, width=(None, 2.0 / STEP))
found, _ = signal.find_peaks(patterns["600 °C"], **SHARP)
found_deg = TWO_THETA[found]
tabulated = np.array([centre for centre, _height, _hkl in REFLECTIONS])
assert found_deg.size == tabulated.size, found_deg
error = np.abs(found_deg - tabulated)
assert error.max() <= 0.1, error
none_yet, _ = signal.find_peaks(patterns["As deposited"], **SHARP)
assert none_yet.size == 0, TWO_THETA[none_yet]

measured = {}
for name in ("450 °C", "600 °C"):         # FWHM of (101) at half prominence
    apex = np.argmax(patterns[name])
    width = signal.peak_widths(patterns[name], [apex], rel_height=0.5)[0][0]
    measured[name] = width * STEP
    assert abs(measured[name] - SERIES[name][1]) < 0.06, measured
assert measured["600 °C"] < measured["450 °C"]
print(f"fig117: self-check passed (find_peaks recovers "
      f"{found_deg.size}/{tabulated.size} reflections at 600 °C, worst "
      f"offset {error.max():.2f}°; none as deposited; (101) FWHM "
      f"{measured['450 °C']:.2f}° at 450 °C > "
      f"{measured['600 °C']:.2f}° at 600 °C)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 76)
ax = fig.add_subplot(ms.grid(fig, 1, 1, left=9, right=21, top=4,
                             bottom=11)[0])
margin = ax.get_yaxis_transform()

for level, (name, colour) in enumerate(zip(SERIES, COLOURS)):
    ax.plot(TWO_THETA, patterns[name] + level * OFFSET, color=colour, lw=0.8)
    ax.text(1.02, level * OFFSET + 0.06, name, transform=margin,
            color=colour if level else ms.GREY_DARK, va="center", ha="left",
            fontsize=ms.FS_TICK)

# Miller indices on the local maximum of the top trace; the doublet
# shares one stacked label so it stays clear of (200) and (204)
top_trace = patterns["600 °C"] + 3 * OFFSET
labels = [(25.3, "(101)"), (37.8, "(004)"), (48.0, "(200)"),
          (54.5, "(105),\n(211)"), (62.7, "(204)")]
for position, text in labels:
    window = np.abs(TWO_THETA - position) < 1.6
    ax.annotate(text, xy=(position, top_trace[window].max()), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=ms.FS_SMALL, linespacing=1.1)

# reference sticks below the traces, height by relative intensity
for position, height, _hkl in REFLECTIONS:
    ax.plot([position, position],
            [-0.36, -0.36 + 0.2 * (0.35 + 0.65 * height)],
            color=ms.VERMILLION, lw=0.9, solid_capstyle="butt")
ax.text(1.02, -0.29, "Reference", transform=margin, color=ms.VERMILLION,
        va="center", ha="left", fontsize=ms.FS_TICK)

ax.set_xlim(20, 70)
ax.set_ylim(-0.42, 3 * OFFSET + 1.22)
ax.set_xticks(np.arange(20, 71, 10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.set_yticks([])
ax.set_xlabel("2θ (degrees)")
ax.set_ylabel("Normalised intensity (offset)")

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig117_xrd_annealing.{ext}")
print("fig117_xrd_annealing: saved png + pdf")
