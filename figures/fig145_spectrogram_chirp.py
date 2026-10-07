"""Fig. 145 - Waveform and STFT spectrogram of a chirp (1.5 column, 120 mm).

A time series hides what a time-frequency map shows at once: the same
test signal (a quadratic chirp, a short constant tone burst and white
noise) is drawn as a waveform and, on the same time axis, as a
short-time Fourier spectrogram in decibels with the chirp's analytic
instantaneous frequency laid over it. The self-check is that the
spectrogram ridge (the loudest bin of each time column inside the chirp
band) follows the analytic f(t) within one frequency bin plus the
frequency the chirp sweeps in half a window.

Model: x(t) = sin φ(t) + tone burst + noise; f(t) = f₀ + (f₁ − f₀)(t/T)²,
f₀ = 40 Hz, f₁ = 320 Hz, T = 4 s; 420 Hz burst, 0.5 s, Hann
envelope; Gaussian noise, s.d. 0.1; sampled at 1 kHz. STFT: Hann window
of 128 samples (128 ms, 7.8 Hz bins), 112 samples (87.5%) overlap; power
spectral density in dB relative to its maximum. All data are simulated.
"""

import numpy as np
from scipy import signal

import manuscript as ms

ms.apply()
HERE = ms.HERE

rng = np.random.default_rng(145)

# ------------------------------------------------------------- DATA ----
FS, T_END = 1000.0, 4.0                   # sampling rate (Hz), record (s)
F0, F1 = 40.0, 320.0                      # chirp start and end (Hz)
TONE_HZ, TONE_T0, TONE_T1, TONE_AMP = 420.0, 2.0, 2.5, 0.8
NOISE_SD = 0.1
NPERSEG, NOVERLAP = 128, 112              # Hann window, 87.5% overlap
DB_FLOOR = -50.0                          # display range: 0 to -50 dB


def chirp_frequency(t):
    """Analytic instantaneous frequency f(t) = (1/2π) dφ/dt."""
    return F0 + (F1 - F0) * (t / T_END) ** 2


t = np.arange(0, T_END, 1 / FS)
phase = 2 * np.pi * (F0 * t + (F1 - F0) * t ** 3 / (3 * T_END ** 2))
in_burst = (t >= TONE_T0) & (t < TONE_T1)
burst = np.zeros_like(t)
burst[in_burst] = (TONE_AMP * signal.windows.hann(in_burst.sum())
                   * np.sin(2 * np.pi * TONE_HZ * t[in_burst]))
x = np.sin(phase) + burst + rng.normal(0, NOISE_SD, t.size)

# --------------------------------------------------------- ESTIMATOR ---
freq, seg_time, psd = signal.spectrogram(
    x, fs=FS, window="hann", nperseg=NPERSEG, noverlap=NOVERLAP,
    detrend=False, scaling="density")     # seg_time = window centres
level_db = 10 * np.log10(psd / psd.max())
bin_hz = freq[1] - freq[0]

# ridge: loudest bin per column, searched below the tone-burst frequency
chirp_band = freq < TONE_HZ - 5 * bin_hz
ridge = freq[chirp_band][np.argmax(psd[chirp_band], axis=0)]

# ------------------------------------------------------- SELF-CHECK ---
interior = (seg_time > 0.25) & (seg_time < T_END - 0.25)
sweep_rate = 2 * (F1 - F0) * seg_time / T_END ** 2          # df/dt (Hz/s)
tolerance = bin_hz + sweep_rate * NPERSEG / FS / 2
deviation = np.abs(ridge - chirp_frequency(seg_time))
assert np.all(deviation[interior] <= tolerance[interior]), deviation.max()
mid_burst = np.argmin(np.abs(seg_time - (TONE_T0 + TONE_T1) / 2))
tone_peak = freq[~chirp_band][np.argmax(psd[~chirp_band, mid_burst])]
assert abs(tone_peak - TONE_HZ) <= bin_hz, tone_peak
print(f"fig145: self-check passed (ridge follows analytic f(t): max "
      f"deviation {deviation[interior].max():.1f} Hz over "
      f"{interior.sum()} columns; bin width {bin_hz:.2f} Hz; window "
      f"{1e3 * NPERSEG / FS:.0f} ms)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 84)
X0, WIDTH = 14.0, 84.0                               # mm, shared by a and b
ax_a = ms.axes(fig, X0, 59, WIDTH, 19)
ax_b = ms.axes(fig, X0, 9.5, WIDTH, 43, sharex=ax_a)
ax_cbar = ms.axes(fig, X0 + WIDTH + 2.5, 9.5, 2.6, 43)

# a, the waveform: the burst is a bump in the envelope, the chirp is not
# readable at all, which is the point of panel b
ax_a.plot(t, x, color=ms.GREY_DARK, lw=0.3)
ax_a.plot([TONE_T0, TONE_T1], [2.4, 2.4], color=ms.VERMILLION, lw=1.2,
          solid_capstyle="butt")
ax_a.text(TONE_T1 + 0.05, 2.4, f"{TONE_HZ:.0f} Hz tone burst",
          color=ms.VERMILLION, fontsize=ms.FS_TICK, va="center")
ax_a.set_xlim(0, T_END)
ax_a.set_ylim(-2.2, 2.9)
ax_a.set_yticks([-2, 0, 2])
ax_a.set_ylabel("Amplitude (a.u.)")
ax_a.tick_params(labelbottom=False)

# b, the spectrogram: every STFT column is drawn; the layer is rasterized
half_step = (seg_time[1] - seg_time[0]) / 2
image = ax_b.imshow(
    level_db, origin="lower", aspect="auto", cmap="magma", vmin=DB_FLOOR,
    vmax=0, interpolation="nearest", rasterized=True,
    extent=[seg_time[0] - half_step, seg_time[-1] + half_step,
            freq[0] - bin_hz / 2, freq[-1] + bin_hz / 2])
ax_b.set_facecolor("black")               # half a window is empty each end
t_fine = np.linspace(0, T_END, 400)
# white dashes vanish on the bright ridge they follow, so they ride on a
# thin black line that is narrower than the ridge
ax_b.plot(t_fine, chirp_frequency(t_fine), color="black", lw=1.1)
ax_b.plot(t_fine, chirp_frequency(t_fine), color="white", lw=0.7,
          ls=(0, (4, 3)))
ax_b.annotate("Analytic f(t) = f₀ + (f₁ − f₀)(t/T)²",
              xy=(1.42, chirp_frequency(1.42) + 6), xytext=(0.3, 215),
              color="white", fontsize=ms.FS_TICK, va="center",
              arrowprops=dict(arrowstyle="-", color="white", lw=0.5,
                              shrinkA=1.5, shrinkB=1.5,
                              relpos=(0.62, 0.0)))
ax_b.text(TONE_T1 + 0.08, TONE_HZ, "Tone burst", color="white",
          fontsize=ms.FS_TICK, va="center")
ax_b.text(0.985, 0.04, f"Hann window {1e3 * NPERSEG / FS:.0f} ms, "
          f"{100 * NOVERLAP / NPERSEG:.1f}% overlap",
          transform=ax_b.transAxes, color="white", fontsize=ms.FS_SMALL,
          ha="right", va="bottom")
ax_b.set_ylim(0, FS / 2)
ax_b.set_yticks(np.arange(0, 501, 100))
ax_b.set_xticks(np.arange(0, T_END + 0.1, 0.5))
ax_b.set_xlabel("Time (s)")
ax_b.set_ylabel("Frequency (Hz)")

cbar = fig.colorbar(image, cax=ax_cbar, ticks=np.arange(DB_FLOOR, 1, 10))
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(length=2, width=0.5)
cbar.set_label("Power spectral density (dB re max)")

ms.panel_label(ax_a, "a", dx_pt=-32)
ms.panel_label(ax_b, "b", dx_pt=-32)
ms.assert_aligned([ax_a, ax_b], edges=("left", "right"))
ms.assert_aligned([ax_b, ax_cbar])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig145_spectrogram_chirp.{ext}")
print("fig145_spectrogram_chirp: saved png + pdf")
