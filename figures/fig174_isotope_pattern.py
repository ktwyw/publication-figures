"""Fig. 174 - Isotope envelope of a dichloro compound (single column, 89 mm).

How to draw a calculated mass-spectral isotope pattern: the isotopologue
distribution of a placeholder formula with two chlorines, obtained by
convolving the elemental isotope abundances once per atom, collapsed to
nominal mass and shown as sticks normalised to the base peak, with the
profile an instrument of modest resolving power would record behind
them and the M, M+2, M+4 triplet compared with chlorine alone and with
the 9 : 6 : 1 rule of thumb. The self-check is that the distribution
sums to 1 to 1e-12, that the monoisotopic mass equals the sum of the
lightest isotopes' masses, and that Cl₂ alone gives exactly
(0.7576², 2 · 0.7576 · 0.2424, 0.2424²), about 100 : 64 : 10.

Model: C₁₅H₁₃Cl₂N₃O₂ as the singly charged molecular ion M⁺•; the
probability and the probability-weighted mass of each nominal-mass
channel are convolved together; profile, Gaussians of full width
m/R at half maximum, resolving power R = 1,000. No data: every curve is
computed from the equations.
"""

import numpy as np

import manuscript as ms

ms.apply()
HERE = ms.HERE

# -------------------------------------------------- GOVERNING MODEL ----
FORMULA = {"C": 15, "H": 13, "Cl": 2, "N": 3, "O": 2}      # placeholder
RESOLVING_POWER = 1000.0                  # m / FWHM
# element: (mass number, atomic mass in u, natural abundance) per stable
# isotope. Source: NIST Atomic Weights and Isotopic Compositions
# database (masses from AME2016; abundances, IUPAC representative
# isotopic compositions, Meija et al., Pure Appl. Chem. 88, 293, 2016).
ISOTOPES = {
    "H": [(1, 1.00782503223, 0.999885), (2, 2.01410177812, 0.000115)],
    "C": [(12, 12.0, 0.9893), (13, 13.00335483507, 0.0107)],
    "N": [(14, 14.00307400443, 0.99636), (15, 15.00010889888, 0.00364)],
    "O": [(16, 15.99491461957, 0.99757), (17, 16.99913175650, 0.00038),
          (18, 17.99915961286, 0.00205)],
    "Cl": [(35, 34.968852682, 0.7576), (37, 36.965902602, 0.2424)],
}
ELECTRON_MASS = 0.000548579909            # u, CODATA 2018


def element_arrays(symbol):
    """Abundance and abundance × mass on a grid of extra nucleons."""
    lightest = ISOTOPES[symbol][0][0]
    size = ISOTOPES[symbol][-1][0] - lightest + 1
    prob, moment = np.zeros(size), np.zeros(size)
    for mass_number, mass, abundance in ISOTOPES[symbol]:
        prob[mass_number - lightest] = abundance
        moment[mass_number - lightest] = abundance * mass
    return prob, moment


def envelope(formula):
    """Nominal-mass distribution and mean exact mass of each channel.

    One polynomial multiplication per atom; the first moment follows the
    product rule, so each channel keeps its abundance-weighted mass.
    """
    prob, moment = np.ones(1), np.zeros(1)
    for symbol, count in formula.items():
        p_el, m_el = element_arrays(symbol)
        for _ in range(count):
            moment = np.convolve(moment, p_el) + np.convolve(prob, m_el)
            prob = np.convolve(prob, p_el)
    nominal = sum(ISOTOPES[el][0][0] * n for el, n in formula.items())
    mean_mass = np.divide(moment, prob, out=np.zeros_like(prob),
                          where=prob > 0)             # empty channels: 0
    return nominal + np.arange(prob.size), prob, mean_mass


# ------------------------------------------------------------ SOLVER ---
nominal, prob, mass = envelope(FORMULA)
mz = mass - ELECTRON_MASS                              # z = 1
rel = 100 * prob / prob.max()                          # base peak = 100%
_, cl2, _ = envelope({"Cl": 2})
m_grid = np.linspace(nominal[0] - 1.0, nominal[0] + 7.5, 3400)
sigma = mz / RESOLVING_POWER / (2 * np.sqrt(2 * np.log(2)))
profile = (rel * np.exp(-0.5 * ((m_grid[:, None] - mz) / sigma) ** 2)).sum(1)

# ------------------------------------------------------- SELF-CHECK ---
assert abs(prob.sum() - 1) < 1e-12, prob.sum()
mono = sum(ISOTOPES[el][0][1] * n for el, n in FORMULA.items())
assert abs(mass[0] - mono) < 1e-9, (mass[0], mono)
average = sum(sum(m * a for _, m, a in ISOTOPES[el]) * n
              for el, n in FORMULA.items())
assert abs((prob * mass).sum() - average) < 1e-9       # mean = molar mass
for el, isotopes in ISOTOPES.items():
    assert abs(sum(a for *_, a in isotopes) - 1) < 1e-12, el
p35, p37 = 0.7576, 0.2424
assert np.allclose(cl2[[0, 2, 4]], [p35 ** 2, 2 * p35 * p37, p37 ** 2],
                   rtol=0, atol=1e-12) and cl2[1] == cl2[3] == 0
cl2_rel = 100 * cl2[[0, 2, 4]] / cl2[0]
assert np.allclose(cl2_rel, [100, 64, 10], atol=0.5)
print(f"fig174: self-check passed (sum = 1 within {abs(prob.sum() - 1):.0e};"
      f" monoisotopic mass {mass[0]:.5f} u, average {average:.3f} u; Cl₂ "
      f"alone {cl2_rel[0]:.0f} : {cl2_rel[1]:.1f} : {cl2_rel[2]:.1f}; "
      f"formula M : M+2 : M+4 = 100 : {rel[2]:.1f} : {rel[4]:.1f})")

# ------------------------------------------------------------ FIGURE --
fig = ms.figure(89, 68)
ax = ms.axes(fig, 13, 11, 71, 51)
STICK, PROFILE = ms.INK, ms.SKY

ax.fill_between(m_grid, profile, color=PROFILE, alpha=0.35, lw=0, zorder=1)
shown = rel >= 0.05                       # sticks too short to see are omitted
ax.vlines(mz[shown], 0, rel[shown], color=STICK, lw=1.3, zorder=3)
names = {0: "M", 2: "M+2", 4: "M+4"}
for k in np.flatnonzero(rel >= 1.0):
    label = f"{mz[k]:.2f}"
    if k in names:
        ax.text(mz[k], rel[k] + 8.5, names[k], ha="center", va="bottom",
                fontsize=ms.FS_TICK, fontweight="bold", color=ms.VERMILLION)
    ax.text(mz[k], rel[k] + 2.0, label, ha="center", va="bottom",
            fontsize=ms.FS_SMALL)

# the chlorine signature against the full formula, as a small table
COLS = (nominal[0] + 4.55, nominal[0] + 5.6, nominal[0] + 6.5,
        nominal[0] + 7.4)
rule = 100 * np.array([9, 6, 1]) / 9
table = [("", "M", "M+2", "M+4"),
         ("This formula", *(f"{rel[k]:.1f}" for k in (0, 2, 4))),
         ("Cl₂ alone", *(f"{v:.1f}" for v in cl2_rel)),
         ("9 : 6 : 1 rule", *(f"{v:.1f}" for v in rule))]
for row, cells in enumerate(table):
    y_row = 111 - 7.0 * row
    for col, (x_col, cell) in enumerate(zip(COLS, cells)):
        ax.text(x_col, y_row, cell, ha="right", va="center",
                fontsize=ms.FS_SMALL, fontweight="bold" if row == 0 else None,
                color=ms.VERMILLION if row == 0 else ms.INK)
ax.plot([COLS[0] - 2.6, COLS[-1]], [107.5] * 2, color=ms.INK, lw=0.5)
ax.text(COLS[-1], 111 - 7.0 * len(table), "Relative abundance (%);\n"
        "¹³C, ¹⁸O and ¹⁵N raise M+2 and M+4\nabove the chlorine-only values",
        ha="right", va="top", fontsize=ms.FS_SMALL, color=ms.GREY_DARK,
        linespacing=1.3)
ax.text(COLS[-1], 126, "C₁₅H₁₃Cl₂N₃O₂, M⁺•; shaded, profile at "
        f"R = {RESOLVING_POWER:,.0f}", ha="right", va="top",
        fontsize=ms.FS_TICK)

ax.set_xlim(nominal[0] - 1.0, nominal[0] + 7.5)
ax.set_ylim(0, 126)
ax.set_xticks(nominal[:8])
ax.set_yticks(np.arange(0, 101, 25))
ax.spines["left"].set_bounds(0, 100)
ax.set_xlabel("m/z")
ax.set_ylabel("Relative abundance (%)", y=50 / 126)

ms.assert_min_font(fig)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig174_isotope_pattern.{ext}")
print("fig174_isotope_pattern: saved png + pdf")
