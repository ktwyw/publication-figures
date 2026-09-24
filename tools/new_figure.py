"""Scaffold a new library figure: tools/new_figure.py N slug "desc"."""
import sys
from pathlib import Path

n = int(sys.argv[1]); slug = sys.argv[2]; desc = sys.argv[3]

body = f'''"""Fig. {n} - {desc}."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng({n})

# ------------------------------------------------- GOVERNING MODEL ----
# Derive every curve from the governing equations; do not sketch.
x = np.linspace(0.0, 1.0, 400)
y = x**2                                   # placeholder model

# ------------------------------------------------------- SELF-CHECK ---
landmark = float(y[-1])
assert abs(landmark - 1.0) < 1e-12, landmark
print("fig{n:03d}: self-check passed (landmark = " + f"{{landmark:.6g}}" + ")")

# ------------------------------------------------------------ FIGURE --
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.plot(x, y)
ax.set_xlabel("x (unit)")
ax.set_ylabel("y (unit)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / ("fig{n:03d}_{slug}." + ext))
print("fig{n:03d}_{slug}: saved png + pdf")
'''

dst = Path(__file__).resolve().parents[1] / "figures" / \
    f"fig{n:03d}_{slug}.py"
assert not dst.exists(), dst
dst.write_text(body)
print(f"wrote {dst}")
print("paste into figures/README.md:")
print(f"- `fig{n:03d}_{slug}.py` \u2014 {desc}")
