"""Fig. 77 - Strength of materials, two statistical classics.

(a) Hall-Petch: yield strength vs d^(-1/2) with a weighted linear fit
giving the friction stress sigma_0 (intercept) and coefficient k_y
(slope), plus a nonlinear secondary axis on top reading directly in
grain size - the exact-transform-pair trick from the Arrhenius figure.

(b) Weibull analysis of brittle strength: simulated fracture strengths,
median-rank plotting positions F = (i - 0.3)/(N + 0.4), the double-log
linearisation ln ln[1/(1-F)] vs ln(sigma), the fitted Weibull modulus m,
and a secondary right axis reading directly in failure probability.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator
from scipy import stats

import journal_style as js

js.apply()
HERE = js.HERE

rng = np.random.default_rng(77)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.9))

# ------------------------------------------- (a) Hall-Petch ----
SIG0, KY = 120.0, 610.0                            # MPa, MPa um^0.5
d_um = np.array([2, 4, 8, 15, 30, 60, 120, 250], float)
x = 1.0 / np.sqrt(d_um)
sig = SIG0 + KY * x + rng.normal(0, 9, x.size)
sig_err = np.full(x.size, 12.0)

res = stats.linregress(x, sig)
xf = np.linspace(0, 0.78, 50)
ax_a.plot(xf, res.intercept + res.slope * xf, color="C1", lw=1.1)
ax_a.errorbar(x, sig, yerr=sig_err, fmt="o", ms=4, mfc="white", color="C0",
              capsize=2)
ax_a.plot(0, res.intercept, "s", ms=4.5, mfc="white", mec="C1", mew=1.2,
          clip_on=False, zorder=5)
ax_a.text(0.04, 0.96,
          rf"$\sigma_y = \sigma_0 + k_y\, d^{{-1/2}}$"
          "\n"
          rf"$\sigma_0$ = {res.intercept:.0f} MPa"
          "\n"
          rf"$k_y$ = {res.slope:.0f} MPa µm$^{{1/2}}$",
          transform=ax_a.transAxes, va="top", fontsize=6.5)

secax = ax_a.secondary_xaxis(
    "top", functions=(lambda s: 1.0 / np.maximum(s, 1e-3) ** 2,
                      lambda d: 1.0 / np.sqrt(np.maximum(d, 1e-6))))
secax.set_xticks([400, 100, 25, 10, 4, 2])
secax.set_xlabel("grain size, $d$ (µm)", fontsize=9)
secax.tick_params(labelsize=7)
secax.xaxis.set_minor_locator(NullLocator())

ax_a.set_xlim(0, 0.78)
ax_a.set_ylim(80, 640)
ax_a.set_xlabel(r"$d^{-1/2}$ (µm$^{-1/2}$)")
ax_a.set_ylabel(r"yield strength, $\sigma_y$ (MPa)")

# ------------------------------------------- (b) Weibull ----
M_TRUE, S0_TRUE, N = 8.0, 350.0, 30
sig_f = np.sort(S0_TRUE * (-np.log(1 - rng.random(N))) ** (1 / M_TRUE))
F = (np.arange(1, N + 1) - 0.3) / (N + 0.4)        # median ranks
Y = np.log(-np.log(1 - F))

fit = stats.linregress(np.log(sig_f), Y)
m_hat = fit.slope
s0_hat = np.exp(-fit.intercept / fit.slope)

sf = np.linspace(sig_f[0] * 0.93, sig_f[-1] * 1.05, 50)
ax_b.plot(sf, fit.intercept + fit.slope * np.log(sf), color="C1", lw=1.1)
ax_b.plot(sig_f, Y, "o", ms=3.6, mfc="white", color="C0")
ax_b.axhline(0, ls=":", lw=0.7, color="0.6")
ax_b.plot(s0_hat, 0, "s", ms=4.5, mfc="white", mec="C1", mew=1.2, zorder=5)
ax_b.annotate(rf"$\hat\sigma_0$ = {s0_hat:.0f} MPa  ($F$ = 63.2%)",
              (s0_hat, 0), xytext=(-6, 6), textcoords="offset points",
              fontsize=6.5, color="C1", ha="right")
ax_b.text(0.04, 0.95,
          rf"$\hat m$ = {m_hat:.1f} $\pm$ {fit.stderr:.1f}"
          f"\n$N$ = {N} specimens",
          transform=ax_b.transAxes, va="top", fontsize=6.5)

ax_b.set_xscale("log")
ax_b.set_xticks([250, 300, 350, 400],
                ["250", "300", "350", "400"])
ax_b.set_xlim(sig_f[0] * 0.90, sig_f[-1] * 1.08)
ax_b.xaxis.set_minor_formatter(NullFormatter())
ax_b.set_ylim(-4.2, 1.8)
ax_b.set_xlabel(r"fracture strength, $\sigma$ (MPa)")
ax_b.set_ylabel(r"$\ln\,\ln[1/(1-F)]$")

secy = ax_b.secondary_yaxis(
    "right", functions=(lambda y: 1 - np.exp(-np.exp(y)),
                        lambda F_: np.log(-np.log(1 - np.clip(F_, 1e-6,
                                                              1 - 1e-6)))))
secy.set_yticks([0.02, 0.10, 0.50, 0.90, 0.99])
secy.set_ylabel("failure probability, $F$", fontsize=9)
secy.tick_params(labelsize=7)
secy.yaxis.set_minor_locator(NullLocator())

js.panel_label(ax_a, "a", x=-0.18)
js.panel_label(ax_b, "b", x=-0.16)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig077_hallpetch_weibull.{ext}")
print("saved fig077_hallpetch_weibull.png / .pdf")
