"""Shared helpers for the journal-style theory figures (fig61 onward).

Usage in a figure script:

    import journal_style as js
    js.apply()                       # style sheet + physics-journal overrides
    js.slope_triangle(ax, ...)       # log-log scaling indicator
    js.panel_label(ax, "a")          # italic (a) outside the frame
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def apply():
    """House style sheet plus the boxed, serif, inward-tick journal look."""
    plt.style.use(str(HERE / "publication.mplstyle"))
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "axes.linewidth": 0.9,
    })


def slope_triangle(ax, x0, y0, dx, slope, color, rise, run):
    """Right-triangle slope indicator for log-log axes (slope may be < 0)."""
    x1, y1 = x0 * 10**dx, y0 * 10 ** (dx * slope)
    ax.plot([x0, x1, x1, x0], [y0, y0, y1, y0], color=color, lw=1.1,
            zorder=5, clip_on=False)
    ax.text(np.sqrt(x0 * x1), y0 * 0.82, run, ha="center", va="top",
            fontsize=9, color=color)
    ax.text(x1 * 1.3, np.sqrt(y0 * y1), rise, ha="left", va="center",
            fontsize=9, color=color)


def panel_label(ax, letter, x=-0.14):
    ax.text(x, 1.01, f"$({letter})$", transform=ax.transAxes,
            fontsize=12, va="bottom")
