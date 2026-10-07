"""Fig. 188 - Butterworth vs elliptic low-pass filter (double column, 183 mm).

One specification, two digital IIR filters of the minimum order that
meets it: the maximally flat Butterworth needs more than twice the order
of the equiripple elliptic design, and the three views show what that
buys and costs. a, magnitude response against the specification mask
(forbidden regions shaded) with the passband enlarged; b, group delay
across the passband; c, poles and zeros in the z-plane. The self-check
is that every pole lies strictly inside the unit circle, both responses
stay within the passband and stopband limits (at the band edges and
everywhere between), the elliptic order does not exceed the Butterworth
order, and the Butterworth magnitude is monotone in the passband.

Model: sample rate 1 kHz; passband to 100 Hz with at most 0.5 dB loss;
stopband from 180 Hz with at least 60 dB attenuation. Orders from
scipy.signal.buttord / ellipord, designs from iirdesign, evaluated as
second-order sections. No data: every curve is computed from the
equations.
"""

import numpy as np
from scipy import signal

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
FS = 1000.0                       # sample rate (Hz)
F_PASS, F_STOP = 100.0, 180.0     # passband and stopband edges (Hz)
G_PASS, G_STOP = 0.5, 60.0        # max passband loss, min attenuation (dB)
DESIGNS = {"butter": ("Butterworth", ms.BLUE, signal.buttord),
           "ellip": ("Elliptic", ms.VERMILLION, signal.ellipord)}
DB_MIN, DB_MAX = -100.0, 8.0      # displayed range of panel a
EDGE_TOL_DB = 1e-6                # numerical slack at the band edges

# ------------------------------------------------------------ SOLVER ---
freq = np.linspace(0, FS / 2, 5001)
f_pass = np.linspace(0, F_PASS, 801)
spec = (F_PASS, F_STOP, G_PASS, G_STOP)


def level_db(sos, f):
    _, h = signal.sosfreqz(sos, worN=f, fs=FS)
    return 20 * np.log10(np.maximum(np.abs(h), 1e-300))


def group_delay_ms(sos, f):
    """Group delays of the cascaded sections add; samples to ms."""
    total = sum(signal.group_delay((sec[:3], sec[3:]), w=f, fs=FS)[1]
                for sec in sos)
    return 1e3 * total / FS


def trim_to_floor(x, y, floor):
    """Blank what lies below the floor and add the exact crossings."""
    below = y < floor
    k = np.flatnonzero(below[1:] != below[:-1])
    x_cross = x[k] + (floor - y[k]) * (x[k + 1] - x[k]) / (y[k + 1] - y[k])
    x_all = np.concatenate([x, x_cross])
    y_all = np.concatenate([np.where(below, np.nan, y),
                            np.full(x_cross.size, floor)])
    order = np.argsort(x_all, kind="stable")
    return x_all[order], y_all[order]


filters = {}
for ftype, (name, colour, order_of) in DESIGNS.items():
    order, _ = order_of(*spec, fs=FS)
    zeros, poles, gain = signal.iirdesign(*spec, ftype=ftype, output="zpk",
                                          fs=FS)
    sos = signal.zpk2sos(zeros, poles, gain)
    filters[ftype] = dict(name=name, colour=colour, order=order, zeros=zeros,
                          poles=poles, sos=sos, db=level_db(sos, freq),
                          db_pass=level_db(sos, f_pass),
                          delay=group_delay_ms(sos, f_pass))

# ------------------------------------------------------- SELF-CHECK ---
in_stop = freq >= F_STOP
for ftype, f in filters.items():
    assert f["poles"].size == f["order"]
    assert np.abs(f["poles"]).max() < 1 - 1e-6, ftype           # stable
    edge_pass, edge_stop = level_db(f["sos"], [F_PASS, F_STOP])
    assert edge_pass >= -G_PASS - EDGE_TOL_DB, (ftype, edge_pass)
    assert edge_stop <= -G_STOP + EDGE_TOL_DB, (ftype, edge_stop)
    assert f["db_pass"].min() >= -G_PASS - EDGE_TOL_DB
    assert f["db_pass"].max() <= EDGE_TOL_DB                    # no gain
    assert f["db"][in_stop].max() <= -G_STOP + EDGE_TOL_DB
butter, ellip = filters["butter"], filters["ellip"]
assert ellip["order"] <= butter["order"]
assert np.all(np.diff(butter["db_pass"]) <= 1e-9)               # monotone
assert np.ptp(ellip["db_pass"]) > 0.9 * G_PASS                  # equiripple
print(f"fig188: self-check passed (orders: Butterworth {butter['order']}, "
      f"elliptic {ellip['order']}; max |pole| "
      f"{np.abs(butter['poles']).max():.4f} and "
      f"{np.abs(ellip['poles']).max():.4f}; passband minimum "
      f"{butter['db_pass'].min():.3f} and {ellip['db_pass'].min():.3f} dB; "
      f"stopband maximum {butter['db'][in_stop].max():.1f} and "
      f"{ellip['db'][in_stop].max():.1f} dB)")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(183, 66)
Y0, H = 11.0, 48.0
ax_a = ms.axes(fig, 13, Y0, 58, H)
ax_b = ms.axes(fig, 86, Y0, 32, H)
ax_c = ms.axes(fig, 131, Y0, H, H)
ax_in = ms.axes(fig, 13 + 29, Y0 + 24.5, 25, 11)    # passband detail, in a
MASK = dict(color=ms.GREY_LIGHT, alpha=0.55, lw=0)


def draw_mask(ax, floor, ceiling):
    """Forbidden regions: too much passband loss, too little attenuation."""
    ax.fill_between([0, F_PASS], floor, -G_PASS, zorder=0, **MASK)
    ax.fill_between([F_STOP, FS / 2], -G_STOP, ceiling, zorder=0, **MASK)


# a, magnitude: responses stop at the frame instead of running below it
draw_mask(ax_a, DB_MIN, DB_MAX)
for f in filters.values():
    ax_a.plot(*trim_to_floor(freq, f["db"], DB_MIN), color=f["colour"],
              lw=1.1)
ax_a.text(F_STOP + 12, 2.5, f"{butter['name']}, order {butter['order']}",
          color=butter["colour"], fontsize=ms.FS_TICK, va="center")
ax_a.text(F_STOP + 12, -6.0, f"{ellip['name']}, order {ellip['order']}",
          color=ellip["colour"], fontsize=ms.FS_TICK, va="center")
ax_a.text(FS / 2 - 8, -G_STOP + 2.5,
          f"Forbidden: attenuation < {G_STOP:.0f} dB", fontsize=ms.FS_SMALL,
          color=ms.GREY_DARK, ha="right", va="bottom")
ax_a.text(F_PASS / 2, -G_STOP, f"Forbidden: loss > {G_PASS} dB",
          fontsize=ms.FS_SMALL, color=ms.GREY_DARK, ha="center", va="center",
          rotation=90)
ax_a.set_xlim(0, FS / 2)
ax_a.set_ylim(DB_MIN, DB_MAX)
ax_a.set_yticks(np.arange(-100, 1, 20))
ax_a.set_xlabel("Frequency (Hz)")
ax_a.set_ylabel("Magnitude (dB)")

draw_mask(ax_in, -0.8, 0.2)
for f in filters.values():
    ax_in.plot(f_pass, f["db_pass"], color=f["colour"], lw=0.9)
ax_in.set_xlim(0, F_PASS)
ax_in.set_ylim(-0.8, 0.2)
ax_in.set_xticks([0, F_PASS / 2, F_PASS])
ax_in.set_yticks([-G_PASS, 0])
ax_in.tick_params(labelsize=ms.FS_SMALL, length=2, pad=1.5)
ax_in.set_title(f"Passband, 0–{F_PASS:.0f} Hz (dB)", fontsize=ms.FS_SMALL,
                pad=2.5)
for side in ("top", "right"):
    ax_in.spines[side].set_visible(True)

# b, group delay: what the sharper elliptic edge costs in phase linearity
for f in filters.values():
    ax_b.plot(f_pass, f["delay"], color=f["colour"], lw=1.1)
for f in filters.values():
    ax_b.text(4, f["delay"][0] + 1.0, f["name"], color=f["colour"],
              fontsize=ms.FS_TICK, va="bottom")
ax_b.set_xlim(0, F_PASS)
ax_b.set_ylim(0, 24)
ax_b.set_yticks(np.arange(0, 25, 4))
ax_b.set_xlabel("Frequency (Hz)")
ax_b.set_ylabel("Group delay (ms)")

# c, z-plane: poles ×, zeros ○; the Butterworth zeros coincide at z = −1
circle = np.exp(1j * np.linspace(0, 2 * np.pi, 361))
ax_c.plot(circle.real, circle.imag, color=ms.GREY, lw=0.6)
ax_c.plot([-1.3, 1.3], [0, 0], color=ms.GREY_LIGHT, lw=0.4, zorder=0)
ax_c.plot([0, 0], [-1.3, 1.3], color=ms.GREY_LIGHT, lw=0.4, zorder=0)
for f in filters.values():
    ax_c.plot(f["poles"].real, f["poles"].imag, "x", color=f["colour"],
              ms=3.6, mew=0.9)
    ax_c.plot(f["zeros"].real, f["zeros"].imag, "o", mfc="none",
              mec=f["colour"], ms=3.6, mew=0.8)
# the zeros that coincide at z = −1 are counted instead of drawn apart
for row, f in enumerate(filters.values()):
    count = int(np.isclose(f["zeros"], -1).sum())
    ax_c.text(-0.88, -0.12 - 0.15 * row,
              f"{count} zero{'s' if count > 1 else ''}", color=f["colour"],
              fontsize=ms.FS_SMALL, ha="left", va="top")
ax_c.text(-0.72, 0.28, "× poles\n○ zeros", fontsize=ms.FS_SMALL,
          color=ms.GREY_DARK, ha="left", va="bottom", linespacing=1.3)
ax_c.set_xlim(-1.3, 1.3)
ax_c.set_ylim(-1.3, 1.3)
ax_c.set_aspect("equal")
ax_c.set_xticks([-1, 0, 1])
ax_c.set_yticks([-1, 0, 1])
ax_c.set_xlabel("Re z")
ax_c.set_ylabel("Im z")

ms.panel_label(ax_a, "a", dx_pt=-30)
ms.panel_label(ax_b, "b", dx_pt=-26)
ms.panel_label(ax_c, "c", dx_pt=-24)
ms.assert_aligned([ax_a, ax_b, ax_c])
ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig188_filter_design.{ext}")
print("fig188_filter_design: saved png + pdf")
