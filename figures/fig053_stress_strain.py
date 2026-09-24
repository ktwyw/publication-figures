"""Fig. 53 - Engineering stress-strain curves (single column).

Shows the full mechanical-test vocabulary: the 0.2% offset construction
for yield strength (dashed line parallel to the elastic slope), the UTS
marker, fracture crosses, and a zoom inset resolving the elastic region
where the offset construction actually happens.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(53)


def curve(E_gpa, sy, uts, e_uts, e_frac, n=0.22, npts=900):
    """Engineering stress-strain (strain in %, stress in MPa)."""
    E = E_gpa * 10.0                                # MPa per % strain
    e = np.linspace(0, e_frac, npts)
    ey = sy / E
    s = np.where(e <= ey, E * e,
                 sy + (uts - sy) * np.clip((e - ey) / (e_uts - ey), 0, 1) ** n)
    post = e > e_uts
    s[post] = uts - (uts * 0.18) * (e[post] - e_uts) / (e_frac - e_uts)
    s *= 1 + rng.normal(0, 0.004, npts)
    return e, s


mats = {"Alloy A (annealed)": dict(E_gpa=200, sy=350, uts=480, e_uts=11.0,
                                   e_frac=15.5),
        "Alloy B (cold-worked)": dict(E_gpa=205, sy=560, uts=640, e_uts=5.0,
                                      e_frac=7.5)}

# ------------------------------------------------------------- PLOT ----
fig, ax = plt.subplots(figsize=(3.5, 2.9))

for i, (name, p) in enumerate(mats.items()):
    e, s = curve(**p)
    ax.plot(e, s, color=f"C{i}", label=name)
    ax.plot(e[-1], s[-1], "x", ms=6, mew=1.4, color=f"C{i}")     # fracture
    i_uts = np.argmax(s)
    ax.plot(e[i_uts], s[i_uts], marker="v", ms=4, color=f"C{i}")

# 0.2% offset construction for Alloy A
E_A = mats["Alloy A (annealed)"]["E_gpa"] * 10
sy_A = mats["Alloy A (annealed)"]["sy"]
e_off = np.array([0.2, 0.2 + (sy_A + 60) / E_A])
ax.plot(e_off, (e_off - 0.2) * E_A, ls="--", lw=0.8, color="0.35")
ax.plot(0.2 + sy_A / E_A, sy_A, "o", ms=4, mfc="white", mec="C0", mew=1.1)
ax.annotate(r"$\sigma_y$ (0.2% offset)", (0.2 + sy_A / E_A, sy_A),
            xytext=(14, -2), textcoords="offset points", fontsize=6)
ax.annotate("UTS", (11.0, 480), xytext=(4, 6), textcoords="offset points",
            fontsize=6)

ax.set_xlim(0, 16.5)
ax.set_ylim(0, 700)
ax.set_xlabel("Engineering strain (%)")
ax.set_ylabel("Engineering stress (MPa)")
ax.legend(loc="lower right", fontsize=6)

# inset: elastic region with the offset line
axins = ax.inset_axes([0.20, 0.06, 0.34, 0.40])
for i, (name, p) in enumerate(mats.items()):
    e, s = curve(**p)
    axins.plot(e, s, color=f"C{i}", lw=0.9)
axins.plot(e_off, (e_off - 0.2) * E_A, ls="--", lw=0.8, color="0.35")
axins.set_xlim(0, 0.75)
axins.set_ylim(0, 640)
axins.tick_params(labelsize=5.5)
ax.indicate_inset_zoom(axins, edgecolor="0.4", lw=0.7)

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig053_stress_strain.{ext}")
print("saved fig053_stress_strain.png / .pdf")
