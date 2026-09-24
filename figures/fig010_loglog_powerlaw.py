"""Fig. 10 - Log-log power-law fit with residual panel (single column).

The residual panel (via height_ratios + sharex) is the honest way to
show fit quality - reviewers increasingly ask for it. The fit is done
by linear regression in log-log space.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "publication.mplstyle"))

rng = np.random.default_rng(8)

# ------------------------------------------------------------- DATA ----
f = np.logspace(0, 3, 40)                       # 1 Hz - 1 kHz
psd = 3e-3 * f**-1.5 * rng.lognormal(0, 0.12, f.size)

# ---------------------------------------------- FIT IN LOG-LOG SPACE ----
res = stats.linregress(np.log10(f), np.log10(psd))
fit = 10**res.intercept * f**res.slope
resid = np.log10(psd) - np.log10(fit)

# ------------------------------------------------------------- PLOT ----
fig, (ax, axr) = plt.subplots(2, 1, figsize=(3.5, 3.1), sharex=True,
                              height_ratios=[3, 1.1])

ax.loglog(f, psd, "o", ms=3.5, color="C0", mfc="white", label="Measured")
ax.loglog(f, fit, color="C1", label="Power-law fit")
ax.set_ylabel("PSD (V$^2$ Hz$^{-1}$)")
ax.legend(loc="upper right")
ax.text(0.05, 0.08, rf"$\alpha$ = {res.slope:.2f} ± {res.stderr:.2f}",
        transform=ax.transAxes, va="bottom")

axr.semilogx(f, resid, "o", ms=3, color="C0", lw=0)
axr.axhline(0, ls="--", lw=0.7, color="0.4")
axr.set_ylim(-0.45, 0.45)
axr.set_ylabel(r"$\Delta\log_{10}$")
axr.set_xlabel("Frequency (Hz)")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig010_loglog_powerlaw.{ext}")
print("saved fig010_loglog_powerlaw.png / .pdf")
