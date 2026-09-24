"""Fig. 89 - Arps decline curves and what b does to reserves (2 panels).

Petroleum production forecasting's workhorse family,

    q(t) = q_i / (1 + b D_i t)^{1/b}   (b > 0),
    q(t) = q_i e^{-D_i t}              (b = 0),

shown two ways. (a) Rate vs time on semilog axes, where the b = 0
exponential is a straight line - the classic diagnostic. (b) Rate vs
cumulative production from the analytic N_p(q) relations: at the same
economic-limit rate the EUR nearly quadruples from exponential to
harmonic, which is why the b exponent is worth arguing about.
Illustrative q_i and D_i.
"""

import numpy as np
import matplotlib.pyplot as plt

import journal_style as js

js.apply()
HERE = js.HERE

QI, DI = 1000.0, 0.5                               # bbl/d, 1/yr
Q_EC = 20.0                                        # economic limit, bbl/d
DAYS = 365.0 / 1000.0                              # bbl/d*yr -> 10^3 bbl

CASES = [(0.0, "C0", "exponential"), (0.5, "C1", "hyperbolic"),
         (1.0, "C2", "harmonic")]


def rate(t, b):
    if b == 0:
        return QI * np.exp(-DI * t)
    return QI / (1 + b * DI * t) ** (1 / b)


def cum(q, b):                                     # N_p(q), 10^3 bbl
    if b == 0:
        return (QI - q) / DI * DAYS
    if b == 1:
        return QI / DI * np.log(QI / q) * DAYS
    return QI / (DI * (1 - b)) * (1 - (q / QI) ** (1 - b)) * DAYS


fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.9))

# ------------------------------------------------- (a) q vs t ----
t = np.linspace(0, 10, 300)
for b, color, name in CASES:
    ax_a.semilogy(t, rate(t, b), color=color, lw=1.2)
    ax_a.text(10.15, rate(10.0, b), f"$b$ = {b:g}\n({name})", fontsize=6.5,
              color=color, va="center")
ax_a.axhline(Q_EC, ls="--", lw=0.8, color="0.45")
ax_a.text(0.15, Q_EC * 1.18, "economic limit, $q_{ec}$", fontsize=6,
          color="0.45")
ax_a.annotate("exponential is straight\non semilog axes", xy=(4.4, 118),
              xytext=(0.4, 32), fontsize=6.5, color="C0",
              arrowprops=dict(arrowstyle="-|>", lw=0.7, color="C0"))

ax_a.set_xlim(0, 10)
ax_a.set_ylim(6, 1500)
ax_a.set_xlabel("time, $t$ (yr)")
ax_a.set_ylabel(r"rate, $q$ (bbl d$^{-1}$)")

# ------------------------------------------------- (b) q vs Np ----
q = np.logspace(np.log10(Q_EC), np.log10(QI), 300)
for b, color, name in CASES:
    ax_b.semilogy(cum(q, b), q, color=color, lw=1.2)
    eur = cum(Q_EC, b)
    ax_b.plot(eur, Q_EC, "o", ms=4, color=color, zorder=5)
    ax_b.plot([eur, eur], [6, Q_EC], ls=":", lw=0.7, color=color)
    ax_b.text(eur, 7.6, f"{eur:.0f}", fontsize=6, color=color,
              ha="center", va="bottom")
ax_b.axhline(Q_EC, ls="--", lw=0.8, color="0.45")
ax_b.text(2900, 700, "EUR at $q_{ec}$\n(10$^3$ bbl)", fontsize=6.5,
          color="0.35", ha="right")
for b, color, x_lab in [(0.0, "C0", 480), (0.5, "C1", 1080),
                        (1.0, "C2", 2150)]:
    ax_b.text(x_lab, 95, f"$b$ = {b:g}", fontsize=6.5, color=color,
              ha="center")

ax_b.set_xlim(0, 3000)
ax_b.set_ylim(6, 1500)
ax_b.set_xlabel(r"cumulative production, $N_p$ (10$^3$ bbl)")
ax_b.set_ylabel(r"rate, $q$ (bbl d$^{-1}$)")

js.panel_label(ax_a, "a", x=-0.17)
js.panel_label(ax_b, "b", x=-0.17)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig089_arps_decline.{ext}")
print("saved fig089_arps_decline.png / .pdf")
