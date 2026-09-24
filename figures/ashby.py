"""ashby.py - shared machinery for Ashby-style material selection charts.

Used by fig75 onward. Provides:

    setup(ax, xlim, ylim, xlabel, ylabel, ...)   log-log decade axes
    bubble(ax, cx, cy, thick, length, tilt, ...) family ellipse in log space
    guideline(ax, n, anchor, ...)                merit-index line of slope n,
                                                 label rotated to match the
                                                 axes' true decade aspect

The ellipse is drawn parametrically in log10 coordinates (so it stays an
ellipse on log axes) with its MAJOR axis along the tilt direction. The
guideline label angle is computed from the rendered axes geometry, not
guessed - the recurring failure mode this module retires.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter


def setup(ax, xlim, ylim, xlabel, ylabel, xticks=None, yticks=None):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for axis, ticks in ((ax.xaxis, xticks), (ax.yaxis, yticks)):
        if ticks is not None:
            axis.set_ticks(ticks)
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())


def _decades_per_inch(ax):
    fig = ax.figure
    fig.canvas.draw()                                # realise the layout
    bb = ax.get_position()
    w_in = bb.width * fig.get_size_inches()[0]
    h_in = bb.height * fig.get_size_inches()[1]
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return np.log10(x1 / x0) / w_in, np.log10(y1 / y0) / h_in


def slope_angle(ax, n):
    """Visual angle (deg) of a log-log line with slope n dec/dec."""
    xdpi, ydpi = _decades_per_inch(ax)
    return np.degrees(np.arctan(n * xdpi / ydpi))


def bubble(ax, cx, cy, thick, length, tilt, color, label, dx=0.0, dy=0.0,
           fontsize=7, alpha=0.30):
    """Material-family ellipse; (cx, cy) in log10, sizes in decades."""
    t = np.linspace(0, 2 * np.pi, 200)
    a, b = length / 2, thick / 2                     # major axis along tilt
    ca, sa = np.cos(np.radians(tilt)), np.sin(np.radians(tilt))
    lx = cx + a * np.cos(t) * ca - b * np.sin(t) * sa
    ly = cy + a * np.cos(t) * sa + b * np.sin(t) * ca
    ax.fill(10**lx, 10**ly, facecolor=color, alpha=alpha, lw=0, zorder=2)
    ax.plot(10**lx, 10**ly, color=color, lw=0.9, zorder=3)
    if label:
        ax.text(10 ** (cx + dx), 10 ** (cy + dy), label, ha="center",
                va="center", fontsize=fontsize, color=color, zorder=4,
                fontweight="bold")


def guideline(ax, n, x_anchor, y_anchor, label=None, lab_x=None,
              lab_shift=1.25, color="0.45", ls=":", lw=0.7, fontsize=6):
    """Line y = y_anchor (x/x_anchor)^n across the axes, optional label."""
    x0, x1 = ax.get_xlim()
    xx = np.logspace(np.log10(x0), np.log10(x1), 60)
    ax.plot(xx, y_anchor * (xx / x_anchor) ** n, ls=ls, lw=lw, color=color,
            zorder=1)
    if label is not None:
        y_lab = y_anchor * (lab_x / x_anchor) ** n * lab_shift
        ax.text(lab_x, y_lab, label, rotation=slope_angle(ax, n),
                rotation_mode="anchor", ha="center", fontsize=fontsize,
                color=color, zorder=1)
