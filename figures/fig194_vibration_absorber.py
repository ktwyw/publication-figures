"""Fig. 194 - Tuned vibration absorber frequency response (1.5 column, 120 mm).

Why a tuned mass damper needs damping: an undamped absorber cancels
the primary motion at one frequency but trades the resonance for two
new ones, whereas every absorber of the same mass and tuning passes
through two invariant points P and Q whatever its damping. Den Hartog's
tuning levels P and Q and his damping puts the peaks on them: the
lowest response over the whole band. Amplitude (a) and phase (b) of
the primary mass come from the 2-DOF complex dynamic-stiffness system
solved at each frequency. The self-check is that the undamped-absorber
response vanishes at the absorber frequency (< 1e-9), that its two
peaks sit on the eigenfrequencies of M⁻¹K, that five absorber dampings
give the same amplitude at P and at Q (to 1e-6), and that P and Q are
level within 1% under Den Hartog tuning.

Model: undamped primary (m₁, k₁) forced harmonically; absorber mass
ratio µ = 0.10, tuning f = ωₐ/ω₁ = 1/(1 + µ), damping ratio
ζ = c₂/(2m₂ω₁) = √(3µ/(8(1 + µ)³)) (Den Hartog, Mechanical Vibrations,
1956). No data: every curve is computed from the equations.
"""

import numpy as np
from scipy import signal

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
M1, K1 = 1.0, 1.0               # primary mass and stiffness (ω₁ = 1)
C1 = 0.0                        # the invariant points need an undamped primary
MU = 0.10                       # absorber mass ratio m₂/m₁
TUNING = 1 / (1 + MU)           # Den Hartog: f = ωₐ/ω₁
ZETA_OPT = np.sqrt(3 * MU / (8 * (1 + MU) ** 3))     # c₂/(2 m₂ ω₁)
ZETA_TEST = [0.03, 0.10, ZETA_OPT, 0.30, 1.00]       # for the invariant points
R_LIM, AMP_LIM = (0.6, 1.4), (0.2, 40.0)
W1 = np.sqrt(K1 / M1)
M2 = MU * M1
K2 = M2 * (TUNING * W1) ** 2


def response(r, zeta=None):
    """Complex X₁/(F/k₁) at frequency ratios r; zeta=None: no absorber."""
    w = np.atleast_1d(r) * W1
    if zeta is None:
        return (K1 / (K1 - w ** 2 * M1 + 1j * w * C1))
    c2 = 2 * zeta * M2 * W1
    mass = np.diag([M1, M2])
    stiff = np.array([[K1 + K2, -K2], [-K2, K2]])
    damp = np.array([[C1 + c2, -c2], [-c2, c2]])
    dynamic = (stiff - w[:, None, None] ** 2 * mass
               + 1j * w[:, None, None] * damp)        # one 2 × 2 per frequency
    force = np.broadcast_to([K1, 0.0], (w.size, 2))[..., None]
    return np.linalg.solve(dynamic, force)[:, 0, 0]


# ------------------------------------------------------------ SOLVER ---
r = np.linspace(*R_LIM, 8000)       # even count: r = 1 is not a grid point
cases = {"Primary alone": response(r),
         "Undamped absorber": response(r, 0.0),
         f"Den Hartog absorber (ζ = {ZETA_OPT:.3f})": response(r, ZETA_OPT)}

# invariant points: amplitude at ζ = 0 equals that at ζ → ∞ (masses locked)
r_p, r_q = np.sqrt(np.sort(np.roots(
    [2 + MU, -2 * (1 + TUNING ** 2 * (1 + MU)), 2 * TUNING ** 2]).real))
at_pq = np.array([np.abs(response([r_p, r_q], z)) for z in ZETA_TEST])
amp_p, amp_q = np.abs(response([r_p, r_q], ZETA_OPT))

eig = np.sqrt(np.sort(np.linalg.eigvals(
    np.linalg.inv(np.diag([M1, M2])) @ np.array([[K1 + K2, -K2], [-K2, K2]])
).real)) / W1
undamped = np.abs(cases["Undamped absorber"])
peaks = r[signal.find_peaks(np.log(undamped), prominence=2.0)[0]]

# ------------------------------------------------------- SELF-CHECK ---
assert np.abs(response(TUNING, 0.0))[0] < 1e-9             # antiresonance
assert peaks.size == 2 and np.abs(peaks - eig).max() <= r[1] - r[0]
assert np.ptp(at_pq, axis=0).max() < 1e-6, np.ptp(at_pq, axis=0)
assert abs(amp_p - amp_q) < 0.01 * amp_p
assert abs(amp_p - np.sqrt(1 + 2 / MU)) < 1e-9             # Den Hartog height
peak_opt = np.abs(response(r, ZETA_OPT)).max()
print(f"fig194: self-check passed (µ = {MU}, f = {TUNING:.4f}, ζ = "
      f"{ZETA_OPT:.4f}; eigenfrequencies {eig[0]:.4f}, {eig[1]:.4f} = "
      f"undamped peaks; P at r = {r_p:.4f}, Q at r = {r_q:.4f}, height "
      f"{amp_p:.4f} = {amp_q:.4f}, spread over ζ "
      f"{np.ptp(at_pq, axis=0).max():.1e}; damped peak {peak_opt:.3f})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(120, 100)
gs = ms.grid(fig, 2, 1, left=15, right=5, top=9, bottom=11, hspace=7,
             height_ratios=[1.9, 1])
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1], sharex=ax_a)
STYLE = [dict(color=ms.GREY, lw=1.1), dict(color=ms.INK, lw=0.7),
         dict(color=ms.VERMILLION, lw=1.7)]

for (label, x1), style in zip(cases.items(), STYLE):
    amp = np.abs(x1)
    inside = (amp >= AMP_LIM[0]) & (amp <= AMP_LIM[1])   # end at the frame
    ax_a.plot(r, np.where(inside, amp, np.nan), label=label, **style)
    lag = np.degrees(np.mod(-np.angle(x1), 2 * np.pi))   # 0-180° behind force
    ax_b.plot(r, lag, **style)
for name, r_inv, amp_inv, dx in (("P", r_p, amp_p, 5), ("Q", r_q, amp_q, -5)):
    ax_a.plot(r_inv, amp_inv, "o", ms=3.6, mfc="white", mec=ms.INK, mew=0.8,
              zorder=5)
    ax_a.annotate(name, xy=(r_inv, amp_inv), xytext=(dx, 5.5),
                  textcoords="offset points", ha="center", va="bottom",
                  fontsize=ms.FS_BODY, fontweight="bold")
ax_a.text(TUNING + 0.045, 0.3, "Antiresonance at the absorber\nfrequency, "
          f"r = f = {TUNING:.3f}", va="center", fontsize=ms.FS_TICK,
          linespacing=1.25)
ax_a.text(0.61, 0.3, f"Mass ratio µ = {MU:.2f}; P and Q\nlevel at "
          f"√(1 + 2/µ) = {amp_p:.2f}", fontsize=ms.FS_TICK, va="center",
          linespacing=1.25)
ax_a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
            handlelength=2.0, columnspacing=1.6)
ax_a.set_yscale("log")
ax_a.set_ylim(*AMP_LIM)
ax_a.set_yticks([1, 10])
ms.plain_log_ticks(ax_a.yaxis)
ax_a.set_ylabel("Amplitude ratio |X₁| / (F/k₁)")
ax_a.tick_params(labelbottom=False)

ax_b.set_xlim(*R_LIM)
ax_b.set_ylim(-8, 188)
ax_b.set_yticks([0, 90, 180])
ax_b.spines["left"].set_bounds(0, 180)
ax_b.set_xlabel("Frequency ratio r = ω/ω₁")
ax_b.set_ylabel("Phase lag of X₁ (°)")

ms.panel_label(ax_a, "a", dx_pt=-32)
ms.panel_label(ax_b, "b", dx_pt=-32)
ms.assert_aligned([ax_a, ax_b], edges=("left", "right"))
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig194_vibration_absorber.{ext}")
print("fig194_vibration_absorber: saved png + pdf")
