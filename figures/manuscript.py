"""Shared helpers for the manuscript-panel figures (fig101 onward).

The house style sheet lets constrained_layout and a tight bounding box
decide how big a figure ends up. A journal decides that instead: 89 mm
or 183 mm wide, type never below 5 pt, panel edges that line up. These
helpers keep the style sheet's fonts and colours but fix the canvas:

    import manuscript as ms
    ms.apply()                           # style sheet + exact-size overrides
    fig = ms.figure(183, 98)             # final size in millimetres
    gs = ms.grid(fig, 2, 2, left=14, right=5, top=7, bottom=11,
                 wspace=19, hspace=17)   # margins and gutters in millimetres
    ms.panel_label(ax, "a")              # bold letter at a fixed point offset
    ms.bracket(ax, 0, 1, y, ms.format_p(p), tick, pad)
    ms.assert_aligned([ax_a, ax_b])      # plot-area edges within 1.5 pt
    ms.assert_min_font(fig)              # no text below 5 pt

Sizes are physical because the saved page *is* the figure: there is no
tight crop, so 89 mm in the script is 89 mm in the PDF.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.transforms import ScaledTranslation

HERE = Path(__file__).resolve().parent

MM = 1.0 / 25.4                          # inches per millimetre
SINGLE, MID, DOUBLE = 89.0, 120.0, 183.0  # common column widths (mm)
FS_BODY, FS_TICK, FS_SMALL, FS_PANEL = 7.0, 6.0, 5.5, 8.0   # points
FS_MATH = 7.2    # for text with a mathtext sub/superscript: 0.7 x 7.2 > 5 pt

# Okabe-Ito colours of the style-sheet cycle, by name
BLUE, VERMILLION, GREEN, PINK = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
ORANGE, SKY, YELLOW = "#E69F00", "#56B4E9", "#F0E442"
INK, GREY_DARK, GREY, GREY_LIGHT = "#272727", "#4D4D4D", "#767676", "#CFCECE"


def apply():
    """House style sheet plus the exact-size, dense-panel overrides."""
    plt.style.use(str(HERE / "publication.mplstyle"))
    plt.rcParams.update({
        # the canvas is set in millimetres, so nothing may resize it
        "figure.constrained_layout.use": False,
        "savefig.bbox": "standard",
        # 7 pt body / 6 pt ticks: readable at final size, dense enough
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "legend.title_fontsize": 6,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "lines.markersize": 3.0, "patch.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
        "axes.labelpad": 2.5, "axes.titlepad": 4.0,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "legend.handlelength": 1.4, "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3, "legend.labelspacing": 0.3,
        "legend.columnspacing": 1.0,
        "pdf.fonttype": 42, "svg.fonttype": "none",   # text stays editable
        "image.interpolation": "nearest",
    })


def figure(width_mm=SINGLE, height_mm=60.0):
    """Figure at its exact final size."""
    return plt.figure(figsize=(width_mm * MM, height_mm * MM))


def _size_mm(fig):
    return tuple(v / MM for v in fig.get_size_inches())


def grid(fig, nrows, ncols, *, left, right, top, bottom, wspace=0.0,
         hspace=0.0, width_ratios=None, height_ratios=None):
    """GridSpec whose margins and gutters are given in millimetres."""
    fw, fh = _size_mm(fig)
    aw = fw - left - right - wspace * (ncols - 1)
    ah = fh - top - bottom - hspace * (nrows - 1)
    assert aw > 0 and ah > 0, "margins and gutters exceed the figure"
    return GridSpec(nrows, ncols, figure=fig,
                    left=left / fw, right=1 - right / fw,
                    top=1 - top / fh, bottom=bottom / fh,
                    wspace=wspace / (aw / ncols),
                    hspace=hspace / (ah / nrows),
                    width_ratios=width_ratios, height_ratios=height_ratios)


def axes(fig, x, y, width, height, **kwargs):
    """Axes from its lower-left corner and size in millimetres."""
    fw, fh = _size_mm(fig)
    return fig.add_axes([x / fw, y / fh, width / fw, height / fh], **kwargs)


def panel_label(ax, letter, dx_pt=-24.0, dy_pt=4.0):
    """Bold lowercase letter at a fixed physical offset from the corner.

    A shared axes-fraction offset (x=-0.16) lands at a different distance
    for every panel size; points do not.
    """
    shift = ScaledTranslation(dx_pt / 72, dy_pt / 72,
                              ax.figure.dpi_scale_trans)
    ax.text(0, 1, letter, transform=ax.transAxes + shift,
            fontsize=FS_PANEL, fontweight="bold", color="black",
            ha="left", va="bottom")


def format_p(p):
    """Exact P value; the threshold form only below 1e-4."""
    if p < 1e-4:
        return "P < 0.0001"
    return f"P = {p:.4f}" if p < 1e-3 else f"P = {p:.3f}"


def bracket(ax, x1, x2, y, text, tick, text_pad, fontsize=FS_SMALL):
    """Comparison bracket at data height y, its label clear of the line."""
    ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick], color=INK,
            lw=0.6, clip_on=False, solid_capstyle="butt")
    ax.text((x1 + x2) / 2, y + text_pad, text, ha="center", va="bottom",
            fontsize=fontsize)


def plain_log_ticks(axis):
    """Decimal tick labels on a log axis (0.01, 0.1, 1, 10)."""
    axis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def plot_area_pt(ax):
    """Plot-area rectangle (left, bottom, right, top) in points."""
    ax.figure.canvas.draw()
    w, h = (v * 72 for v in ax.figure.get_size_inches())
    box = ax.get_position()
    return box.x0 * w, box.y0 * h, box.x1 * w, box.y1 * h


def assert_aligned(axs, edges=("bottom", "top"), tol_pt=1.5):
    """Self-check: the named plot-area edges agree within tol_pt.

    Returns the worst deviation in points so the script can print it.
    """
    index = {"left": 0, "bottom": 1, "right": 2, "top": 3}
    boxes = [plot_area_pt(ax) for ax in axs]
    worst = 0.0
    for edge in edges:
        values = [box[index[edge]] for box in boxes]
        worst = max(worst, max(values) - min(values))
    assert worst <= tol_pt, f"panel edges differ by {worst:.2f} pt"
    return worst


def assert_min_font(fig, min_pt=5.0):
    """Self-check: no visible text is set below min_pt (nominal size).

    Mathtext sub/superscripts render at about 0.7 of the size checked
    here; set such text at FS_MATH so the small glyph still reaches 5 pt.
    """
    import matplotlib.text as mtext
    sizes = [t.get_fontsize() for t in fig.findobj(mtext.Text)
             if t.get_visible() and t.get_text().strip()]
    smallest = min(sizes)
    assert smallest >= min_pt, (
        f"text at {smallest:g} pt is below {min_pt:g} pt")
    return smallest
