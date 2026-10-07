"""Fig. 183 - Orientation tuning, population, decoding (double column, 183 mm).

Three steps from one neuron to a population code: an example neuron's
mean response at 12 orientations with a fitted 180°-periodic von Mises
curve, preferred orientation and half-width; the normalised tuning
curves of 60 neurons sorted by fitted preference, which gives the
diagonal band; and a population-vector read-out of held-out trials
against the presented orientation. The self-check is that the example
neuron's fitted preference is within 5° of the truth, that the rows of
the map are in order of preference, and that the decoder's circular
mean absolute error, computed with 180° periodicity (179° counts as
1°), is below 8°.

Statistics: r(θ) = r₀ + A exp(κ(cos 2(θ − θp) − 1)); Poisson spike counts
in 0.25 s; 60 neurons, 20 training trials per orientation (a, b: mean ±
s.e.m., n = 20; dashed line, mean of 20 blank trials) and 10 held-out
trials per orientation (c, n = 120, drawn unwrapped to within ±90° of the
stimulus); least-squares fits. All data are simulated.
"""

import numpy as np
from scipy import optimize

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(183)

# ------------------------------------------------------------- DATA ----
ORI = np.arange(0.0, 180.0, 15.0)             # presented orientations (°)
N_NEURON, N_TRAIN, N_TEST = 60, 20, 10
WINDOW = 0.25                                 # spike-count window (s)
EXAMPLE = dict(base=4.0, amp=26.0, pref=63.0, kappa=2.6)   # spikes/s, °
MAE_BOUND = 8.0                               # decoder bound (°)


def von_mises(theta, base, amp, pref, kappa):
    """180°-periodic tuning curve (spikes/s); theta and pref in degrees."""
    return base + amp * np.exp(kappa * (np.cos(np.radians(2 * (theta - pref)))
                                        - 1))


def circ_error(a, b):
    """Absolute orientation difference with 180° periodicity."""
    return np.abs((np.asarray(a) - b + 90.0) % 180.0 - 90.0)


# preferences are stratified over 0-180° so that the code has no gaps
true = np.column_stack([
    rng.uniform(1.0, 6.0, N_NEURON), rng.uniform(8.0, 30.0, N_NEURON),
    rng.permutation((np.arange(N_NEURON) + rng.uniform(size=N_NEURON))
                    * 180.0 / N_NEURON),
    rng.uniform(1.5, 4.0, N_NEURON)])
true[0] = list(EXAMPLE.values())              # neuron 0 is the example
rate = np.array([von_mises(ORI, *row) for row in true])     # neuron x ori
expected = rate[:, :, None] * WINDOW           # spikes per trial
train = rng.poisson(expected, (N_NEURON, ORI.size, N_TRAIN)) / WINDOW
test = rng.poisson(expected, (N_NEURON, ORI.size, N_TEST)) / WINDOW
blank = rng.poisson(EXAMPLE["base"] * WINDOW, N_TRAIN) / WINDOW  # no stimulus

# --------------------------------------------------------- ESTIMATOR ---
mean, sem = train.mean(axis=2), train.std(axis=2, ddof=1) / np.sqrt(N_TRAIN)


def fit_tuning(response):
    """Least-squares von Mises; start at the vector-average preference."""
    vector = np.sum(response * np.exp(2j * np.radians(ORI)))
    start = [response.min(), np.ptp(response),
             np.degrees(np.angle(vector)) / 2, 2.0]
    popt, _ = optimize.curve_fit(
        von_mises, ORI, response, p0=start,
        bounds=([0, 0, -np.inf, 0.4], [np.inf, np.inf, np.inf, 20.0]))
    popt[2] %= 180.0
    return popt


fits = np.array([fit_tuning(row) for row in mean])
base_fit, amp_fit, pref_fit, kappa_fit = fits.T
order = np.argsort(pref_fit)
normalised = ((mean - mean.min(axis=1, keepdims=True))
              / np.ptp(mean, axis=1, keepdims=True))[order]
# half-width at half-maximum: exp(κ(cos 2Δ − 1)) = 1/2
half_width = np.degrees(np.arccos(1 + np.log(0.5) / kappa_fit[0])) / 2

# population vector on held-out trials: each neuron votes for its fitted
# preference (on the doubled angle) with its baseline-subtracted, scaled rate
weights = (test - base_fit[:, None, None]) / amp_fit[:, None, None]
votes = np.exp(2j * np.radians(pref_fit))[:, None, None]
decoded = (np.degrees(np.angle((weights * votes).sum(axis=0))) / 2) % 180.0
presented = np.repeat(ORI[:, None], N_TEST, axis=1)
errors = circ_error(decoded, presented)
mae = errors.mean()

# ------------------------------------------------------- SELF-CHECK ---
assert circ_error(179.0, 0.0) == 1.0 and circ_error(1.0, 178.0) == 3.0
assert circ_error(pref_fit[0], EXAMPLE["pref"]) < 5.0, pref_fit[0]
assert np.all(np.diff(pref_fit[order]) >= 0)
assert np.all(circ_error(ORI[np.argmax(normalised, axis=1)],
                         pref_fit[order]) <= 30.0)       # band on diagonal
assert mae < MAE_BOUND, mae
print(f"fig183: self-check passed (example neuron: preferred "
      f"{pref_fit[0]:.1f}° vs true {EXAMPLE['pref']:.0f}°, half-width "
      f"{half_width:.1f}°; {N_NEURON} rows sorted; population-vector "
      f"circular MAE {mae:.2f}° on {errors.size} held-out trials, bound "
      f"{MAE_BOUND:.0f}°; error(179°, 0°) = {circ_error(179.0, 0.0):.0f}°)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 62)
ax_a = ms.axes(fig, 13, 11, 43, 43)
ax_b = ms.axes(fig, 73, 11, 38, 43)
ax_cbar = ms.axes(fig, 113, 11, 2.2, 43)
ax_c = ms.axes(fig, 135, 11, 43, 43)
EX_COLOUR = ms.VERMILLION
TICKS = np.arange(0, 181, 45)

# a, the example neuron: data, fit, and what is read from the fit
theta = np.linspace(0, 180, 361)
curve = von_mises(theta, *fits[0])
ax_a.axhline(blank.mean(), color=ms.GREY, lw=0.6, ls=(0, (4, 3)))
ax_a.plot(theta, curve, color=EX_COLOUR, lw=1.3)
ax_a.errorbar(ORI, mean[0], yerr=sem[0], fmt="o", ms=3.2, mfc="white",
              mec=ms.INK, mew=0.8, ecolor=ms.INK, elinewidth=0.6, capsize=0,
              zorder=3)
peak, half = base_fit[0] + amp_fit[0], base_fit[0] + amp_fit[0] / 2
ax_a.plot([pref_fit[0] - half_width, pref_fit[0] + half_width], [half, half],
          color=ms.INK, lw=0.6, marker="|", ms=4, mew=0.6)
ax_a.text(pref_fit[0] + half_width + 6, half,
          f"Half-width\n{half_width:.0f}° (HWHM)", va="center",
          fontsize=ms.FS_TICK)
ax_a.annotate(f"Preferred {pref_fit[0]:.0f}°", xy=(pref_fit[0], peak),
              xytext=(pref_fit[0] + 22, peak + 1.5), va="center",
              fontsize=ms.FS_TICK,
              arrowprops=dict(arrowstyle="-", color=ms.INK, lw=0.5,
                              shrinkA=1, shrinkB=3, relpos=(0.0, 0.5)))
# the label goes under the lowest error bar, so no data can sit on it
y_clear = min(blank.mean(), (mean[0] - sem[0]).min()) - 0.6
ax_a.text(180, y_clear, "Spontaneous", color=ms.GREY_DARK, ha="right",
          va="top", fontsize=ms.FS_TICK)
ax_a.text(180, 35.5, f"mean ± s.e.m.,\nn = {N_TRAIN} trials", ha="right",
          va="top", fontsize=ms.FS_SMALL, color=ms.GREY_DARK)
ax_a.set_xlim(-6, 186)
ax_a.set_ylim(-3.5, 36)
ax_a.set_xticks(TICKS)
ax_a.set_yticks(np.arange(0, 36, 10))
ax_a.spines["bottom"].set_bounds(0, 180)
ax_a.spines["left"].set_bounds(0, 35)
ax_a.set_xlabel("Orientation (°)")
ax_a.set_ylabel("Response (spikes s⁻¹)")

# b, population map: one row per neuron, sorted by fitted preference
step = ORI[1] - ORI[0]
image = ax_b.imshow(normalised, origin="lower", aspect="auto",
                    cmap="viridis", vmin=0, vmax=1, rasterized=True,
                    extent=[-step / 2, ORI[-1] + step / 2,
                            0.5, N_NEURON + 0.5])
row_example = int(np.flatnonzero(order == 0)[0]) + 1
ax_b.plot(1.035, row_example, "<", ms=3.2, mfc=EX_COLOUR, mew=0,
          transform=ax_b.get_yaxis_transform(), clip_on=False)
ax_b.annotate("Triangle, neuron in a", xy=(1, 1), xytext=(0, 3),
              xycoords="axes fraction", textcoords="offset points",
              ha="right", va="bottom", color=EX_COLOUR, fontsize=ms.FS_SMALL)
ax_b.set_xticks(np.arange(0, 166, 45))
ax_b.set_yticks([1, 20, 40, 60])
ax_b.set_xlabel("Orientation (°)")
ax_b.set_ylabel("Neuron (sorted by preference)")
cbar = fig.colorbar(image, cax=ax_cbar, ticks=[0, 0.5, 1])
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(length=2, width=0.5)
cbar.set_label("Normalised response")

# c, decoding: values are unwrapped to within ±90° of the stimulus, so a
# 178° read-out of a 0° grating is drawn at −2°, next to the identity line
unwrapped = presented + (decoded - presented + 90.0) % 180.0 - 90.0
LIM = (-20.0, 190.0)
ax_c.plot([LIM[0], 180], [LIM[0], 180], color=ms.GREY, lw=0.6, zorder=1)
ax_c.plot(presented.ravel(), unwrapped.ravel(), "o", ms=2.6, mfc=ms.BLUE,
          mew=0, alpha=0.35, zorder=2)
ax_c.text(0.04, 0.97, f"Population vector, {N_NEURON} neurons\n"
          f"Circular mean |error| = {mae:.1f}°\n"
          f"n = {errors.size} held-out trials", transform=ax_c.transAxes,
          va="top", fontsize=ms.FS_TICK, linespacing=1.35)
ax_c.text(0.97, 0.04, "Identity", color=ms.GREY_DARK, ha="right",
          transform=ax_c.transAxes, va="bottom", fontsize=ms.FS_TICK)
ax_c.set_xlim(*LIM)
ax_c.set_ylim(*LIM)
ax_c.set_xticks(TICKS)
ax_c.set_yticks(TICKS)
ax_c.spines["bottom"].set_bounds(0, 180)
ax_c.spines["left"].set_bounds(0, 180)
ax_c.set_xlabel("Presented orientation (°)")
ax_c.set_ylabel("Decoded orientation (°)")

for ax, letter, dx in ((ax_a, "a", -26), (ax_b, "b", -24), (ax_c, "c", -28)):
    ms.panel_label(ax, letter, dx_pt=dx, dy_pt=8)
ms.assert_aligned([ax_a, ax_b, ax_cbar, ax_c])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig183_tuning_curves.{ext}")
print("fig183_tuning_curves: saved png + pdf")
