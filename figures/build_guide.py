"""build_guide.py - generate 'One Hundred Publication-Quality Figures'.

Parses README.md (the single source of truth for all 100 entries), makes
downsampled thumbnails, sanitises the descriptions into LaTeX, and
compiles figure_guide.pdf: (I) principles, (II) the illustrated catalog,
(III) the failure-mode catalog, (IV) checklists.
"""

import re
import subprocess
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
THUMBS = HERE / "thumbs"
THUMBS.mkdir(exist_ok=True)

# ------------------------------------------------------------------ parse
entries = []                                       # (num, stem, description)
pat = re.compile(r"^- `(fig(\d+)_[a-z0-9_]+)\.py` \u2014 (.+)$")
for line in (HERE / "README.md").read_text().splitlines():
    m = pat.match(line)
    if m:
        entries.append((int(m.group(2)), m.group(1), m.group(3)))
entries.sort()
assert len(entries) >= 100, f"parsed {len(entries)} entries"

# ------------------------------------------------------------- thumbnails
for _, stem, _ in entries:
    src = HERE / f"{stem}.png"
    dst = THUMBS / f"{stem}.png"
    if not dst.exists():
        im = Image.open(src)
        im.thumbnail((520, 520), Image.LANCZOS)
        im.save(dst, optimize=True)

# ------------------------------------------------------------- sanitiser
GREEK = {"\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma",
         "\u03b4": "delta", "\u03b5": "varepsilon", "\u03b6": "zeta",
         "\u03b7": "eta", "\u03b8": "theta", "\u03ba": "kappa",
         "\u03bb": "lambda", "\u00b5": "mu", "\u03bc": "mu",
         "\u03bd": "nu", "\u03be": "xi", "\u03c0": "pi", "\u03c1": "rho",
         "\u03c3": "sigma", "\u03c4": "tau", "\u03c6": "varphi",
         "\u03c7": "chi", "\u03c8": "psi", "\u03c9": "omega",
         "\u0393": "Gamma", "\u0394": "Delta", "\u03a9": "Omega",
         "\u03a3": "Sigma"}
SYMBOL = {"\u00d7": "times", "\u00b7": "cdot", "\u00b1": "pm",
          "\u2248": "approx", "\u2265": "geq", "\u2264": "leq",
          "\u2192": "to", "\u221d": "propto", "\u221e": "infty"}


def sanitise(t):
    # code spans first
    codes = []

    def stash(m):
        codes.append(m.group(1).replace("_", r"\_"))
        return f"@@C{len(codes) - 1}@@"

    t = re.sub(r"`([^`]+)`", stash, t)

    def protect(literal):                          # shield finished LaTeX
        codes.append(literal)
        return f"@@C{len(codes) - 1}@@"

    t = t.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    t = t.replace("\u2014", "---").replace("\u2013", "--")
    t = t.replace("\u201c", "``").replace("\u201d", "''")
    t = t.replace("\u2018", "`").replace("\u2019", "'")
    t = t.replace("\u221a(4T_v/\u03c0)", protect(r"$\sqrt{4T_v/\pi}$"))
    t = t.replace("\u210f\u03c9", protect(r"$\hbar\omega$"))
    t = t.replace("\u210f", protect(r"$\hbar$"))
    t = t.replace("\u2113", protect(r"$\ell$"))
    t = t.replace("\u00bd", protect(r"$\frac{1}{2}$"))
    t = t.replace("\u2212", "-").replace("\u2032", "'")
    for u, name in {**GREEK, **SYMBOL}.items():
        t = t.replace(u, f"\\{name}{{}}")
    # variable_subscript -> math (base may be a letter or \macro{})
    sub = re.compile(r"(\\[A-Za-z]+\{\}|[A-Za-z])_(\{[^{}]{1,14}\}"
                     r"|[A-Za-z0-9]{1,4})")

    def mathsub(m):
        base = m.group(1).replace("{}", "")
        s = m.group(2).strip("{}")
        return f"${base}_{{{s}}}$"

    t = sub.sub(mathsub, t)
    t = t.replace("_", r"\_")
    t = re.sub(r"\^\{([^{}]{1,14})\}", r"\\textsuperscript{\1}", t)
    t = re.sub(r"\^(\d)", r"\\textsuperscript{\1}", t)
    for u, s in {"\u00b2": "2", "\u00b3": "3", "\u00b9": "1"}.items():
        t = t.replace(u, rf"\textsuperscript{{{s}}}")
    t = t.replace("\u207b\u00b9", r"\textsuperscript{-1}")
    for u, s in {"\u2080": "0", "\u2081": "1", "\u2082": "2",
                 "\u2083": "3"}.items():
        t = t.replace(u, rf"\textsubscript{{{s}}}")
    t = t.replace("\u00b0", r"\textdegree{}")
    t = t.replace("\u221a", r"$\surd$")
    t = t.replace("\u2026", r"\ldots{}")
    # bare macros left in text mode -> wrap in math
    t = re.sub(r"(?<!\$)\\([A-Za-z]+)\{\}", r"$\\\1$", t)
    for i, c in enumerate(codes):
        t = t.replace(f"@@C{i}@@", rf"\texttt{{{c}}}")
    return t


# --------------------------------------------------------- titles & tags
def title_of(stem):
    name = stem.split("_", 1)[1]
    OVER = {"curve_fit": "Data + model fit", "qq_plot": "Normal Q--Q plot",
            "roc_curves": "ROC curves", "pca_biplot": "PCA biplot",
            "ecdf_comparison": "ECDF comparison", "upset": "UpSet plot",
            "raster_psth": "Spike raster + PSTH",
            "kaplan_meier": "Kaplan--Meier survival",
            "bland_altman": "Bland--Altman comparison",
            "manhattan": "Manhattan plot (GWAS)",
            "xrd": "X-ray diffraction", "tga_dsc": "TGA--DSC thermogram",
            "stress_strain": "Stress--strain curves",
            "cyclic_voltammetry": "Cyclic voltammetry",
            "nyquist_eis": "Nyquist EIS", "isotherm_psd": "Isotherm + PSD",
            "arrhenius": "Arrhenius plot", "phase_diagram": "Phase diagram",
            "ftir_stack": "FTIR stack", "tauc_bandgap": "Tauc band gap",
            "carreau_flowcurve": "Carreau flow curves (case study)",
            "moody": "Moody chart", "falkner_skan": "Falkner--Skan profiles",
            "thiele": "Thiele modulus", "rtd_tanks": "RTD tanks-in-series",
            "vdw_maxwell": "Van der Waals + Maxwell",
            "transient_conduction": "Transient conduction",
            "mccabe_thiele": "McCabe--Thiele construction",
            "tafel": "Tafel plot", "breakthrough": "Breakthrough curves",
            "ellingham": "Ellingham diagram", "ttt": "TTT diagram",
            "common_tangent": "Common-tangent construction",
            "pourbaix": "Pourbaix diagram", "ashby": "Ashby chart: E--rho",
            "residue_curves": "Residue curve map",
            "hallpetch_weibull": "Hall--Petch + Weibull",
            "evans": "Evans diagram",
            "cstr_multiplicity": "CSTR multiplicity",
            "ashby_strength_density": "Ashby chart: strength--density",
            "ashby_toughness_modulus": "Ashby chart: toughness--modulus",
            "smith_chart": "Smith chart", "ber_waterfall": "BER waterfalls",
            "buckley_leverett": "Buckley--Leverett waterflood",
            "mohr_coulomb": "Mohr--Coulomb failure",
            "streeter_phelps": "Streeter--Phelps DO sag",
            "antenna_patterns": "Antenna patterns",
            "roofline_amdahl": "Roofline + Amdahl",
            "arps_decline": "Arps decline curves",
            "influence_lines": "Influence lines",
            "response_spectrum": "Response spectrum",
            "activated_sludge": "Chemostat washout",
            "planck_blackbody": "Planck blackbody",
            "qho_wavefunctions": "Quantum harmonic oscillator",
            "titration_curves": "Titration curves",
            "eye_diagram": "Eye diagrams",
            "pz_material_balance": "p/Z material balance",
            "terzaghi_consolidation": "Terzaghi consolidation",
            "double_slit": "Double-slit interference",
            "carnot_cycle": "Carnot cycle, both planes"}
    if name in OVER:
        return OVER[name]
    return name.replace("_", " ").capitalize()


DOMTAG = {61: "Rheology", 82: "RF", 83: "Communications", 84: "Petroleum",
          85: "Geotechnical", 86: "Environmental", 87: "Antennas",
          88: "Computer arch.", 89: "Petroleum", 90: "Structural",
          91: "Earthquake", 92: "Bioprocess", 93: "Physics",
          94: "Quantum", 95: "Chemistry", 96: "Communications",
          97: "Petroleum", 98: "Geotechnical", 99: "Optics",
          100: "Thermodynamics"}

TAGMAP = [("inset", "inset axes"), ("GridSpec", "GridSpec"),
          ("gridspec", "GridSpec"), ("rasteriz", "rasterized points"),
          ("polar", "polar axes"), ("secondary", "secondary axis"),
          ("brentq", "root finding"), ("fsolve", "root finding"),
          ("solve\\_ivp", "ODE integration"), ("shooting", "ODE integration"),
          ("series", "series solution"), ("colorbar", "colorbar"),
          ("log-log", "log axes"), ("semilog", "log axes"),
          ("log axes", "log axes"), ("legend", "legend craft"),
          ("annotat", "annotation"), ("label", "direct labels"),
          ("assert", "built-in self-check"), ("check", "built-in self-check"),
          ("seed", "seeded RNG"), ("fill", "shading"),
          ("shad", "shading"), ("envelope", "envelopes"),
          ("tangent", "tangency construction"),
          ("contour", "contours"), ("twin", "twin axes"),
          ("aspect", "aspect-aware rotation"),
          ("rotation", "aspect-aware rotation"),
          ("module", "shared module")]


def tags_of(desc):
    got, low = [], desc.lower()
    for k, v in TAGMAP:
        if k.lower() in low and v not in got:
            got.append(v)
        if len(got) == 4:
            break
    return ", ".join(got)


SECTIONS = {1: "A \\;\\textbullet\\; Statistical and general data graphics "
               "(figs.\\ 1--50)",
            51: "B \\;\\textbullet\\; Chemistry and materials "
                "characterization (figs.\\ 51--60)",
            61: "C \\;\\textbullet\\; Case study: reproducing a journal "
                "theory figure (fig.\\ 61)",
            62: "D \\;\\textbullet\\; Transport phenomena and reactors "
                "(figs.\\ 62--67)",
            68: "E \\;\\textbullet\\; Chemical and materials engineering "
                "classics (figs.\\ 68--73)",
            74: "F \\;\\textbullet\\; Electrochemistry, stability and "
                "selection (figs.\\ 74--79)",
            80: "G \\;\\textbullet\\; Ashby charts on the shared module "
                "(figs.\\ 80--81)",
            82: "H \\;\\textbullet\\; Across science and engineering "
                "(figs.\\ 82--100)"}

SECNOTE = {61: "This single figure carries the reproduction workflow of "
               "\\S I.2 and introduced \\texttt{journal\\_style.py} "
               "(boxed inward ticks, slope triangles, panel letters), "
               "reused by every figure after it.",
           80: "Both charts are thin scripts over \\texttt{ashby.py}: "
               "log-space bubbles with the major axis along the tilt, "
               "decade axes, and guide-line labels rotated from the "
               "\\emph{rendered} axes aspect --- the module that retired "
               "failure pattern 2."}

# ------------------------------------------------------------- catalog TeX
cat = []
for num, stem, desc in entries:
    if num in SECTIONS:
        cat.append(f"\\subsection*{{{SECTIONS[num]}}}")
        if num in SECNOTE:
            cat.append(f"\\noindent\\small {SECNOTE[num]}\\par"
                       "\\vspace{4pt}\\normalsize")
    d = sanitise(desc)
    if d and d[0].islower():
        d = d[0].upper() + d[1:]
    tags = tags_of(d)
    tag = DOMTAG.get(num, "")
    taghdr = (f"\\hfill{{\\scriptsize\\textsc{{{tag}}}}}" if tag else "")
    tagline = (f"\\\\[0pt]{{\\scriptsize\\textit{{Techniques:}} {tags}}}"
               if tags else "")
    cat.append(
        "\\Needspace*{7\\baselineskip}\n"
        "\\noindent\\begin{minipage}[t]{0.150\\textwidth}"
        "\\vspace{1pt}"
        f"\\includegraphics[width=\\linewidth]{{thumbs/{stem}.png}}"
        "\\end{minipage}\\hspace{0.014\\textwidth}"
        "\\begin{minipage}[t]{0.828\\textwidth}\\vspace{0pt}"
        f"\\textbf{{{num}.\\ {title_of(stem)}}}{taghdr}\\\\[1pt]"
        f"{{\\small {d}}}{tagline}"
        "\\end{minipage}\\par\\vspace{6pt}\n")
CATALOG = "\n".join(cat)

# ------------------------------------------------------------- document
TEX = r"""
\documentclass[10pt,a4paper]{article}
\usepackage[margin=2.1cm]{geometry}
\usepackage{amsmath,graphicx,textcomp,xcolor,needspace,enumitem,parskip}
\usepackage[colorlinks=true,linkcolor=blue!50!black]{hyperref}
\setlist{nosep,leftmargin=1.4em}
\newcommand{\pat}[2]{\needspace{4\baselineskip}\paragraph{#1.}#2}
\title{\vspace{-1.2em}One Hundred Publication-Quality Figures\\[2pt]
\large A practical guide, with the complete worked-example library\vspace{-0.4em}}
\author{Yanwei Wang \\ \small companion to the
\texttt{publication-figures} repository (scripts \texttt{fig001}--\texttt{fig100}+,
\texttt{publication.mplstyle}, \texttt{journal\_style.py}, \texttt{ashby.py})}
\date{}
\begin{document}
\maketitle
\vspace{-2.2em}
\tableofcontents

\section{The craft: how these figures are made}

Every figure in this library was produced the same way, and the method is
the point. The one hundred scripts are worked examples of eight habits.

\subsection{One style sheet for the whole paper}
All scripts load a single \texttt{publication.mplstyle}: single-column
figures about 3.5--4.0\,in wide (double column 7.0--7.2\,in), 7--8\,pt
text so labels match journal captions after scaling, the Okabe--Ito
colour-blind-safe cycle, despined axes with outward ticks (the
journal-style module switches to boxed inward ticks when imitating older
journals), \texttt{constrained\_layout} for spacing, and twin outputs:
a 600-dpi PNG for previewing and a vector PDF for submission. Consistency
is not cosmetic --- reviewers read a paper whose figures share one visual
grammar far faster.

\subsection{Derive, don't draw}
No curve in this library is sketched. Each is computed from the
governing equations --- Nernst lines, Butler--Volmer kinetics, the
Buckley--Leverett fractional flow, Planck's law --- with generic textbook
parameters. When the task is to \emph{reproduce} a figure from a paper
(fig.~61), the workflow is: read every label and axis; derive the
underlying solution; verify landmark values the original states;
parameterise the computation; add asymptotic expansions the original
implies; imitate the styling (``style archaeology''); iterate against the
target. A reproduced figure you derived is one you understand --- and one
you may legally redraw, because the physics is nobody's property while
the original's pixels are.

\subsection{Geometry and typography}
Work in inches and points, never pixels: set the final printed size and
let DPI follow. Label every axis with symbol \emph{and} unit. Prefer
mathtext for symbols and remember it is a dialect of \LaTeX{}, not
\LaTeX{} itself (pattern~14). Type \textmu, \textdegree{} and the minus
sign as literal characters --- escape sequences inside raw strings print
themselves (pattern~1).

\subsection{Colour and encoding}
Default to a colour-blind-safe cycle and never let colour carry meaning
alone: pair it with line style or marker shape, and check the figure in
grayscale. Reserve semantic colour (red for failure envelopes, warm for
hot isotherms) for when it teaches.

\subsection{Annotation beats legends}
With few curves, label them directly, in their own colour, where the
curves are \emph{separated} --- converging families (sag curves, titration
tails) must be labelled in their fan, not at their confluence
(pattern~11). Rotated labels along sloped lines must use the rendered
axes aspect, not the data slope (pattern~2); \texttt{ashby.slope\_angle}
automates this. Keep annotations off the data; use thin leaders when the
target region is crowded; aim leaders at the named object's ink, not the
nearest convenient coordinate (pattern~15). Bottom-edge labels collide
with tick numbers --- place them inside the axes (pattern~10).

\subsection{Multi-panel discipline}
One message per panel; bold sequential panel letters; shared axes where
comparison is the point; \texttt{height\_ratios} for sketch strips and
residual panels; insets never over data --- relocate or promote to a
panel (pattern~3).

\subsection{Verification is part of the figure}
Run, \emph{look at the pixels}, fix, rerun --- plausible code routinely
produces wrong figures, and the render is the only witness. Build
self-checks into the physics wherever possible: the Wien locus must
thread every Planck peak (fig.~93); Mohr circles must touch the envelope
because centre-to-line distance equals radius (fig.~85); the Carnot loop
integral must equal the $T$--$s$ rectangle, and an \texttt{assert} says
so (fig.~100). Seed every random generator; print confirmations and key
derived numbers on save.

\subsection{Reproducibility and packaging}
One standalone script per figure with a marked DATA block; shared code
promoted to modules (\texttt{journal\_style.py}, \texttt{ashby.py});
a README indexing every script in one line; the whole library shipped as
a versioned zip. A figure others can regenerate is a figure others can
trust.

\section{The catalog: one hundred worked examples}

Each entry shows the figure, what it demonstrates, and (where the
description names one) the built-in check it carries. Thumbnails are
downsampled; run the script for the full-resolution PNG and vector PDF.

@@CATALOG@@

\section{The failure-mode catalog}

Sixteen named patterns, every one harvested from a real bug in building
this library. Each surfaced as a wrong \emph{picture} (or a crash) from
plausible code; the render-and-inspect loop caught them all.

\pat{Escape sequences in raw strings}{\textbackslash u00b5 inside
\texttt{r"..."} prints itself. Type the literal character. (Recurred
seven times before it became reflex.)}
\pat{Aspect-blind slope labels}{A label rotated by
$\arctan(\text{slope})$ ignores the axes' unit or decade aspect and
slashes across its own line. Compute the visual angle from the rendered
axes geometry (figs.~61, 69, 78; automated in \texttt{ashby.py}).}
\pat{Insets over data}{An inset that covers curves must move or become a
panel (figs.~53, 63).}
\pat{Legend--curve collisions}{Legends are placed blind; check the
swatch column against every curve, or label directly (fig.~70).}
\pat{Nonlinear secondary axes and minor ticks}{A $1/s^2$ transform asks
the minor locator to enumerate ticks to $10^{18}$; disable minor
locators on nonlinear secondary axes (fig.~77).}
\pat{Stiff ODEs need terminal events}{Boundary-layer shooting blows up
past the far boundary; add terminal events instead of praying
(fig.~63).}
\pat{One sign, many symptoms}{A single sign slip in one crossing formula
truncated a boundary, spawned two stray segments and a bow-tie polygon
--- four symptoms, one cause (fig.~74). Multiple anomalies usually share
a root.}
\pat{Positional spill into a keyword}{\texttt{f(ax, *t[1:],
label=t[0])} silently feeds a trailing positional into \texttt{label}.
Unpack tuples explicitly. (Recurred verbatim: figs.~75, 80, 81.)}
\pat{Fixed-point crawl breaks index arrows}{Trajectories crawl near
equilibria, so index-based arrow placement collapses to zero length and
renders with arbitrary orientation; anchor arrows by \emph{value} and
give them a minimum spatial length (fig.~76).}
\pat{Bottom labels vs.\ the tick row}{Symbolic labels under the axis
land on the tick numbers; put them inside the axes (figs.~84, 85).}
\pat{Converging curves pile their labels}{Curves that meet at an edge
stack right-edge labels into mush; label each curve where the family is
spread (figs.~86, 95).}
\pat{Guide-line crossover corridors}{Two guide lines through different
anchors swap vertical order at their crossing; a label corridor that
works on one side fails on the other (fig.~80).}
\pat{Helper contracts are nonlocal}{Calling a log--log helper before the
axes are logarithmic fed $\log_{10}$ a linear autoscale range, produced
NaN, and crashed \emph{inside the font renderer} --- errors surface far
from their cause (fig.~88).}
\pat{The renderer speaks a dialect}{\textbackslash tfrac is fluent
ams\LaTeX{} and a fatal error in mathtext (fig.~94). Habits are not
portable across renderers.}
\pat{Arrows point at coordinates, not meanings}{The Rayleigh--Jeans
annotation landed on the Planck peak instead of the classical curve it
named; aim at the named object's ink (fig.~93).}
\pat{Algebraic label collisions}{Sometimes the collision is a theorem:
a label offset exactly equal to the Henderson--Hasselbalch term put
every titration label on its own curve (fig.~95); a productivity curve
passed through the biomass label because $DX=X$ at $D=1$ there
(fig.~92). When a label lands on a curve at one $x$, suspect the
algebra, then move along the curve.}

\section{Checklists}

\subsection*{Before writing}
\begin{itemize}
\item Identify the governing equations and the landmark values the
figure must honour; plan the built-in self-check.
\item Choose column width, panel count and message-per-panel; pick which
curves get direct labels and where the family is separated.
\end{itemize}

\subsection*{Before rendering}
\begin{itemize}
\item Style sheet loaded; sizes in inches; symbols as literal characters;
axis labels carry units; RNG seeded.
\item Rotated labels use rendered-aspect angles; secondary nonlinear axes
have minor locators disabled; tuples unpacked explicitly.
\end{itemize}

\subsection*{Inspecting the render}
\begin{itemize}
\item Every label: on its object, off the data, inside the frame, clear
of tick rows and of every other label?
\item Every arrow: finite length, pointing at the named ink, correct
direction along the curve?
\item Landmarks and self-checks: do computed values match the physics
(peaks on loci, tangencies exact, areas equal, limits approached)?
\item Anything clipped at a spine? Anything only visible because you
zoomed? Fix, rerun, look again.
\end{itemize}

\subsection*{Shipping}
\begin{itemize}
\item PNG + vector PDF written; confirmation printed with key numbers.
\item README line added; library zip rebuilt; script runs standalone
from a clean directory.
\end{itemize}

\vspace{6pt}\noindent\emph{Closing note.} The library's real product is
not the hundred pictures but the loop that made them: derive, render,
inspect, fix, and let the physics check itself. Everything else in this
guide is that loop, written down.

\end{document}
"""

(HERE / "figure_guide.tex").write_text(TEX.replace("@@CATALOG@@", CATALOG))

for _ in range(2):
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                        "figure_guide.tex"], cwd=HERE, capture_output=True)
out = r.stdout.decode("utf-8", errors="replace")
print("\n".join(out.splitlines()[-6:]))
