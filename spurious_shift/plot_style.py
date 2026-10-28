"""
plot_style.py — Shared figure style for the FAIR-GEM / Fair-DER++ paper.

The same style is used by the Spurious Shift and Bias-in-Bios figures. The
aliases near the bottom (apply_style, ACC_CMAP, FAIR_CMAP, CAT_COLORS, MARKERS,
OURS_EDGE, FULL_W, COL_W) provide alternative names used by some plotting
scripts. Fonts are sized for legibility in a single-column layout.

Style summary:
  * Typography: a single *serif* (Times) stack to match the
    paper body, with Times-compatible math via STIX so symbols always render.
  * Two-channel sequential colour semantic:
        - ACCURACY  white -> green  (greener = more accurate)
        - FAIRNESS  white -> blue   (bluer  = more fair, i.e. smaller gap)
    Value-encoded marks (bars, heat-maps) use these ramps directly.
  * A coordinated categorical palette for line/scatter plots, where method
    identity is the visual channel. Our two methods stay in the green family
    (echoing the accuracy channel = "good"); baselines take warm/violet hues so
    a "blue line" is never confused with the fairness channel.

Import and call `set_style()` (or `apply_style()`) once at the top of a script.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


# ── Symbols (math-text → always render, font-independent) ──────────────────────
STAR   = r"$\star$"      # our methods
DAGGER = r"$\dagger$"    # oracle / upper bound
ELL    = r"$\ell$"       # legitimate-signal strength
# The headline fairness metric is the accuracy-parity gap: max-min per-group
# balanced accuracy. It is not the equalized-odds gap; the equalized-odds gap
# is recorded separately as `eo_true` and carries D_EO.
D_AP   = r"$\Delta_{\mathrm{AP}}$"
D_EO   = r"$\Delta_{\mathrm{EO}}$"
D_DP   = r"$\Delta_{\mathrm{DP}}$"


# ── Global rcParams ────────────────────────────────────────────────────────────

def set_style():
    """Apply the shared rcParams. Call once per script.

    Fonts are sized for legibility in a single-column layout."""
    plt.rcParams.update({
        # Typography — one serif (Times) stack everywhere to match the paper
        # body; math uses STIX, which is metrically Times-compatible.
        "font.family":          "serif",
        "font.serif":           ["Times New Roman", "Times", "STIX Two Text",
                                 "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset":     "stix",
        "mathtext.rm":          "Times New Roman",
        "mathtext.it":          "Times New Roman:italic",
        "mathtext.bf":          "Times New Roman:bold",
        # ── Font sizes ──
        # Figures are generated at the width they are PLACED at (COL_W for
        # single-column body figures, FULL_W for full-width / appendix figures),
        # so these point sizes render ~1:1 in the PDF. Sized to sit at/just above
        # the body text (~10pt) for legibility.
        "font.size":            8.0,
        "axes.labelsize":       8.0,
        "axes.titlesize":       9.0,
        "xtick.labelsize":      7.0,
        "ytick.labelsize":      7.0,
        "legend.fontsize":      7.0,
        "figure.titlesize":     9.0,
        "axes.titleweight":     "regular",
        "axes.titlepad":        5.0,
        "axes.labelpad":        3.0,
        # Frame / ticks — light open frame, only the two axes we read off.
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       0.8,
        "axes.edgecolor":       "black",
        "axes.labelcolor":      "black",
        "text.color":           "black",
        "xtick.color":          "black",
        "ytick.color":          "black",
        "xtick.labelcolor":     "black",
        "ytick.labelcolor":     "black",
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.major.size":     3.0,
        "ytick.major.size":     3.0,
        "xtick.major.width":    0.8,
        "ytick.major.width":    0.8,
        # Lines / patches
        "lines.linewidth":      1.5,
        "lines.markeredgewidth": 0.0,
        "lines.solid_capstyle": "round",
        "patch.linewidth":      0.0,
        # Legend
        "legend.frameon":       True,
        "legend.edgecolor":     "0.80",
        "legend.framealpha":    1.0,
        "legend.fancybox":      False,
        "legend.borderpad":     0.5,
        "legend.labelspacing":  0.35,
        "legend.handlelength":  1.8,
        "legend.columnspacing": 1.4,
        # Output / canvas
        "figure.facecolor":     "white",
        "savefig.facecolor":    "white",
        "figure.dpi":           110,
        "savefig.dpi":          400,
        # No tight-bbox trimming: figures are saved at EXACT figsize so they fill
        # their placement width (\columnwidth / \linewidth) predictably and
        # consistently. Every figure uses constrained_layout + inside ("outside
        # lower center") legends so nothing falls outside the canvas.
        "savefig.bbox":         None,
        "savefig.pad_inches":   0.0,
        "pdf.fonttype":         42,   # embed TrueType (editable / searchable text)
        "ps.fonttype":          42,
    })


GRID_KW = dict(color="0.90", linestyle="-", linewidth=0.6, zorder=0)

# The paper uses a single 5.5-inch text block. Every figure is generated at its
# placement width so Matplotlib point sizes are preserved in the PDF. COL_W is
# kept as a separate alias for the column width.
NEURIPS_TEXTWIDTH = 5.5
NEURIPS_COLWIDTH  = 5.5


# ── Sequential colour ramps ────────────────────────────────────────────────────
CMAP_ACC  = plt.get_cmap("Greens")   # white → green : accuracy
CMAP_FAIR = plt.get_cmap("Blues")    # white → blue  : fairness

# Keep value-encoded bars off pure white so they stay visible on a white page,
# while still reading as "lighter ≈ worse".
_RAMP_LO, _RAMP_HI = 0.20, 0.92


def value_colors(values, channel="acc", higher_better=True, vmin=None, vmax=None):
    """Map a sequence of metric values to colours on the accuracy/fairness ramp.

    channel       : "acc"  → green ramp,  "fair" → blue ramp.
                    Convenience alias "gap" == ("fair", higher_better=False) so a
                    smaller fairness gap reads as the bluer (more fair) colour.
    higher_better : direction of "good". For accuracy/BWT use True; for an EO/DP
                    *gap* use False (or pass channel="gap").
    Returns a list of RGBA tuples; NaNs map to light grey.
    """
    if channel == "gap":                 # convenience alias for fairness gaps
        channel, higher_better = "fair", False
    cmap = CMAP_ACC if channel == "acc" else CMAP_FAIR
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return [(0.85, 0.85, 0.85, 1.0)] * len(v)
    lo = np.min(finite) if vmin is None else vmin
    hi = np.max(finite) if vmax is None else vmax
    span = (hi - lo) if (hi - lo) > 1e-9 else 1.0

    out = []
    for x in v:
        if not np.isfinite(x):
            out.append((0.85, 0.85, 0.85, 1.0))
            continue
        t = (x - lo) / span               # 0 (worst) … 1 (best of the set)
        if not higher_better:
            t = 1.0 - t
        out.append(cmap(_RAMP_LO + (_RAMP_HI - _RAMP_LO) * t))
    return out


def channel_swatch(channel, level=0.62):
    """A single representative colour for a channel (for legend swatches)."""
    cmap = CMAP_ACC if channel == "acc" else CMAP_FAIR
    return cmap(level)


def heatmap_norm(values):
    """Robust Normalize for accuracy heat-maps (5th–95th percentile)."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return Normalize(vmin=np.percentile(finite, 5),
                     vmax=np.percentile(finite, 95))


# ── Categorical palette for line plots (method = visual channel) ───────────────
# Distinct, print-safe, broadly colour-blind-friendly. OUR methods sit in the
# green family to echo the accuracy channel (= "good"); the oracle is neutral
# grey; fine-tune (the cautionary baseline) is red. Every other baseline takes a
# warm or violet hue so the green/blue *hues* stay reserved for the semantic
# ramps. "derpp" is accepted alongside "der_pp" as a method-key spelling.
METHOD_COLORS = {
    "joint_retrain":  "#7F7F7F",   # grey    — oracle reference (neutral, dashed)
    "finetune":       "#D1495B",   # red     — cautionary baseline
    "der_pp":         "#E8833A",   # orange
    "derpp":          "#E8833A",   # orange  (alternate method-key spelling)
    "agem":           "#6A4C93",   # deep violet
    "group_dro":      "#C44E9D",   # magenta
    "fairl":          "#8C6D4F",   # brown
    "fsw":            "#0F8B8D",   # teal    — FSW (Park et al. 2026)
    "fair_gem":       "#2E8B57",   # sea green   (ours)
    "fair_derpp":     "#16513A",   # dark green  (ours)
}

OURS   = {"fair_gem", "fair_derpp"}
ORACLE = "joint_retrain"

# Bare, human-readable method names (no symbols — add those with display_name).
_NAMES = {
    "joint_retrain":  "Joint Retrain",
    "finetune":       "Fine-tune",
    "agem":           "A-GEM",
    "der_pp":         "DER++",
    "derpp":          "DER++",
    "group_dro":      "GroupDRO",
    "fairl":          "FaIRL-insp.",
    "fsw":            "FSW",
    "fair_gem":       "FAIR-GEM",
    "fair_derpp":     "Fair-DER++",
}


def display_name(method, mark=True):
    """Human-readable name, optionally tagged with the ours-star / oracle-dagger."""
    name = _NAMES.get(method, method)
    if not mark:
        return name
    if method in OURS:
        return f"{name} {STAR}"
    if method == ORACLE:
        return f"{name} {DAGGER}"
    return name


def line_style(method):
    """Per-method line styling for trajectory / motivation plots. Colour, marker
    AND dash pattern all vary, so the lines stay separable in greyscale and for
    colour-blind readers (no reliance on hue alone)."""
    # Thinner strokes than default so line plots stay legible when placed in a
    # single column (small). Our methods are only modestly heavier than baselines.
    base = dict(color=METHOD_COLORS.get(method, "0.4"),
                marker="o", ms=2.6, lw=1.0, ls="-", zorder=2)
    if method == ORACLE:
        base.update(marker="s", ms=2.8, lw=1.1, ls="--")
    if method == "finetune":
        base.update(marker="o", ls="-")
    if method == "agem":
        base.update(ls=(0, (5, 1)), marker="P")
    if method in ("der_pp", "derpp"):
        base.update(ls=(0, (3, 1, 1, 1)), marker="h")
    if method == "group_dro":
        base.update(ls="-.", marker="D", ms=2.4)
    if method == "fairl":
        base.update(ls=":", marker="X")
    if method == "fsw":
        base.update(ls=(0, (1, 1)), marker="v")
    if method == "fair_gem":
        base.update(lw=1.6, ms=3.4, zorder=3)
    if method == "fair_derpp":
        base.update(lw=1.6, ms=3.4, ls="--", marker="^", zorder=3)
    return base


# ── State / task-label parsing ─────────────────────────────────────────────────

def parse_state(s):
    """'task_2__legit=1.5__shortcut=3.0' → (idx_str, legit_or_None, shortcut)."""
    idx = s.split("__")[0].replace("task_", "")
    shortcut, legit = 0.0, None
    for part in s.split("__"):
        if part.startswith("shortcut="):
            shortcut = float(part.split("=")[1])
        elif part.startswith("legit="):
            legit = float(part.split("=")[1])
    return idx, legit, shortcut


def task_label(s, short=False):
    """Single-token task tick label, e.g. 'T2'."""
    return f"T{parse_state(s)[0]}"


def task_tick(s):
    """Single-token axis tick for heat-maps, e.g. 'T2'."""
    return f"T{parse_state(s)[0]}"


def schedule_strings(states):
    """Return (shortcut_str, legit_str) for figure captions."""
    parsed = [parse_state(s) for s in states]
    sc = ", ".join(f"{p[2]:g}" for p in parsed)
    ls = ([p[1] for p in parsed])
    ls_str = (", ".join(f"{v:g}" for v in ls if v is not None)
              if any(v is not None for v in ls) else "constant")
    return sc, ls_str


def shortcut_dominant_idxs(states):
    """Indices where the shortcut strength exceeds the legitimate signal."""
    parsed = [parse_state(s) for s in states]
    dom = [i for i, (_, legit, sc) in enumerate(parsed)
           if legit is not None and sc > legit]
    if dom:
        return dom, "shortcut\ndominates"
    shortcuts = [p[2] for p in parsed]
    mx = max(shortcuts)
    return [i for i, v in enumerate(shortcuts) if v >= mx], f"peak shortcut\n(s={mx:g})"


def annotate_shortcut_band(ax, states):
    """Shade the region of the x-axis where the shortcut dominates the
    legitimate signal. Neutral grey so it never competes with the method
    colours or the accuracy/fairness semantic hues."""
    idxs, label = shortcut_dominant_idxs(states)
    if not idxs:
        return
    lo, hi = min(idxs) - 0.45, max(idxs) + 0.45
    ax.axvspan(lo, hi, color="0.55", alpha=0.10, zorder=0, linewidth=0)
    ylo, yhi = ax.get_ylim()
    ax.text((lo + hi) / 2, yhi - (yhi - ylo) * 0.03, label,
            ha="center", va="top", fontsize=6.5, color="0.40")


def tight_ylim(values, margin=0.22, anchor_zero=False, err=None):
    """Axis limits for `values`. Pass `err` so error bars are not clipped —
    the limits are then taken over value±err rather than over the values."""
    v = [x for x in np.ravel(values) if np.isfinite(x)]
    if err is not None:
        vals = np.asarray(np.ravel(values), dtype=float)
        errs = np.asarray(np.ravel(err), dtype=float)
        errs = np.where(np.isfinite(errs), errs, 0.0)
        both = np.concatenate([vals - errs, vals + errs])
        v = [x for x in both if np.isfinite(x)]
    if not v:
        return 0.0, 1.0
    lo, hi = min(v), max(v)
    if anchor_zero:
        lo = min(lo, 0.0)
    span = (hi - lo) if (hi - lo) > 1e-9 else abs(hi) * 0.2 + 0.01
    return lo - margin * span, hi + margin * span


# ── Seed spread / error bars (shared by every figure) ──────────────────────────
# An error bar is +/-1 sample standard deviation (ddof=1) across seeds of the
# statistic plotted, matching the `_mean +/- _std` columns of the summary CSVs.
#
# The statistic is evaluated inside each seed and the spread taken across
# seeds. Pooling the per-cell stds stored in the aggregate is wrong for anything
# combining several cells (a row mean, BWT, an accuracy-weighted gap): the cells
# come from one training run and are correlated.
ERRORBAR_KW = dict(ecolor="0.25", elinewidth=0.8, capsize=2.0, capthick=0.8)
BAND_ALPHA  = 0.13               # shaded +/-1 SD band on trajectory plots
SEED_NOTE   = "error bars: $\\pm$1 SD across seeds"


def seed_sd(values):
    """Sample SD (ddof=1) across seeds; NaN for fewer than two finite values."""
    v = np.asarray([x for x in np.ravel(values) if np.isfinite(x)], dtype=float)
    return float(np.std(v, ddof=1)) if v.size > 1 else float("nan")


def per_seed(data, key, method):
    """The (n_seeds, T, T) stack for one metric/method, or None.

    Returns None for single-seed files and other files without per-seed
    stacks, so callers draw no error bars.
    """
    name = f"{key}__{method}__per_seed"
    return data[name] if name in getattr(data, "files", []) else None


def per_seed_stat(data, key, method, fn):
    """Apply `fn` to each seed's matrix; returns a list or None."""
    stack = per_seed(data, key, method)
    return None if stack is None else [fn(m) for m in stack]


def seed_sd_of(data, key, method, fn):
    """Spread across seeds of `fn` evaluated within each seed. NaN if unknown."""
    vals = per_seed_stat(data, key, method, fn)
    return float("nan") if vals is None else seed_sd(vals)


def n_seeds(data):
    seeds = data["seeds"] if "seeds" in getattr(data, "files", []) else None
    return 0 if seeds is None else int(np.size(seeds))


def spread_note(data):
    """Bare interval description, e.g. '+/-1 SD across 10 seeds'."""
    n = n_seeds(data)
    return (f"$\\pm$1 SD across {n} seeds" if n > 1
            else "$\\pm$1 SD across seeds")


def seed_note(data):
    """Legend note for whisker-style error bars."""
    return f"error bars: {spread_note(data)}"


# ── Aliases ────────────────────────────────────────────────────────────────────
# Alternative names for the constants above, used by some plotting scripts.
apply_style = set_style
ACC_CMAP    = CMAP_ACC
FAIR_CMAP   = CMAP_FAIR
CAT_COLORS  = METHOD_COLORS
FULL_W      = NEURIPS_TEXTWIDTH
COL_W       = NEURIPS_COLWIDTH
OURS_EDGE   = "#0b3d1a"          # dark-green edge to emphasise our methods

# ── Consistent bar styling (shared by every bar chart) ─────────────────────────
# Centralised so bars look the same in every figure: same widths, same edge,
# same ours-hatch.
BAR_WIDTH        = 0.72      # single-series bar charts (one bar per method)
BAR_PAIR_WIDTH   = 0.40      # grouped charts: width of EACH bar in a pair
BAR_EDGE         = "0.35"    # default bar edge colour
BAR_EDGE_W       = 0.6       # default bar edge width
BAR_PAIR_EDGE_W  = 0.5       # edge width for grouped/secondary bars
HATCH_OURS       = "////"    # hatch marking our methods
VALUE_LABEL_SIZE = 7.0       # atop-bar value annotations


def mark_our_bars(bars, methods, ours=None, edge=None, linewidth=1.1):
    """Apply the consistent 'our methods' treatment (hatch + dark-green edge)
    to the bars whose method key is in `ours`."""
    ours = OURS if ours is None else ours
    edge = OURS_EDGE if edge is None else edge
    for b, m in zip(bars, methods):
        if m in ours:
            b.set_hatch(HATCH_OURS)
            b.set_edgecolor(edge)
            b.set_linewidth(linewidth)


# Per-method marker map (matches line_style markers) for scatter plots.
MARKERS = {
    "joint_retrain":  "s",
    "finetune":       "o",
    "agem":           "P",
    "derpp":          "h",
    "der_pp":         "h",
    "group_dro":      "D",
    "fairl":          "X",
    "fair_gem":       "o",
    "fair_derpp":     "^",
}


# ── Shared Pareto rendering ─────────────────────────────────────────────────────
# All Pareto figures call these so grid, markers, frontier, axes and legend are
# identical. Differences (metric on the x-axis, column vs
# full width) are passed in; everything else is fixed here.
PARETO_HEIGHT_COL  = 2.28     # single-column Pareto figure height (in)
PARETO_HEIGHT_FULL = 3.20     # full-width Pareto figure height (in)
_PARETO_S      = {"ours": 58, "oracle": 46, "base": 30}   # marker areas
_PARETO_LW     = {"ours": 1.1, "oracle": 0.9, "base": 0.5}


def pareto_frontier(points):
    """Indices of the (lower-x, higher-y)-optimal points (smaller gap = better,
    higher accuracy = better)."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    front, best_y = [], -np.inf
    for i in order:
        if points[i][1] >= best_y:
            front.append(i)
            best_y = points[i][1]
    return front


def draw_pareto_panel(ax, fair_vals, acc_vals, methods, xlabel, show_ylabel=True,
                      fair_err=None, acc_err=None):
    """Draw ONE standardized Pareto panel: NO gridlines, dashed frontier
    (oracle excluded), 'ideal' marker, standardized point markers/sizes.

    fair_err / acc_err are optional per-point +/-1 SD across seeds. They are
    drawn under the markers so the frontier and the points stay readable; the
    frontier itself is computed from the means.
    """
    ax.set_axisbelow(True)
    ax.grid(False)   # Pareto panels are gridless.
    # Frontier from non-oracle points only (oracle is shown but not connected).
    cand = [(fair_vals[i], acc_vals[i]) for i, m in enumerate(methods)
            if m != ORACLE]
    fpts = sorted([cand[j] for j in pareto_frontier(cand)], key=lambda p: p[0])
    ax.plot([p[0] for p in fpts], [p[1] for p in fpts],
            color="0.55", ls=(0, (4, 3)), lw=1.1, zorder=1)
    ax.annotate("ideal", xy=(0.03, 0.97), xycoords="axes fraction",
                ha="left", va="top", fontsize=7, style="italic", color="0.45")
    if fair_err is not None or acc_err is not None:
        _clean = lambda e: (None if e is None else
                            np.where(np.isfinite(np.asarray(e, dtype=float)),
                                     np.asarray(e, dtype=float), 0.0))
        ax.errorbar(fair_vals, acc_vals,
                    xerr=_clean(fair_err), yerr=_clean(acc_err),
                    fmt="none", zorder=2, **ERRORBAR_KW)
    for i, m in enumerate(methods):
        io, ic = (m in OURS), (m == ORACLE)
        kind = "ours" if io else ("oracle" if ic else "base")
        ax.scatter(fair_vals[i], acc_vals[i],
                   color=METHOD_COLORS.get(m, "0.4"), marker=MARKERS.get(m, "o"),
                   s=_PARETO_S[kind],
                   edgecolors=OURS_EDGE if io else ("0.25" if ic else "0.45"),
                   linewidths=_PARETO_LW[kind], zorder=4 if io else 3)
    ax.set_ylim(bottom=0.5)
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(r"Balanced accuracy $\uparrow$")


def pareto_legend(fig, methods, note=None):
    """Standardized 2-row (ncol=5) Pareto legend, inside the canvas.

    `note` names the error-bar interval above the entries when the panel drew
    error bars.
    """
    def _mk(m):
        return Line2D([0], [0], marker=MARKERS.get(m, "o"), ls="",
                      markerfacecolor=METHOD_COLORS.get(m, "0.4"),
                      markeredgecolor=OURS_EDGE if m in OURS else "0.3",
                      markeredgewidth=1.0, markersize=6, label=display_name(m))
    handles = ([Line2D([0], [0], color="0.55", ls=(0, (4, 3)), lw=1.1,
                       label="Frontier")] + [_mk(m) for m in methods])
    fig.legend(handles=handles, loc="outside lower center", ncol=5,
               frameon=True, edgecolor="0.80",
               columnspacing=0.8, handletextpad=0.2, fontsize=7.0,
               title=note, title_fontsize=6.5)


def resolve_results(aggregate: str, single: str, label: str) -> str:
    """
    Prefer the cross-seed aggregate; fall back to a single-seed file with a
    warning.

    The fallback only applies when an unseeded single run exists and no
    aggregate has been written.
    """
    import os
    import sys

    if os.path.exists(aggregate):
        return aggregate
    if os.path.exists(single):
        print(f"WARNING: {aggregate} not found; falling back to {single}.\n"
              f"         {label} will show a single-seed run, not the "
              f"multi-seed result.\n"
              f"         Run: bash spurious_shift/run.sh   (or ../aggregate.py over runs/)",
              file=sys.stderr)
        return single
    raise SystemExit(
        f"Neither {aggregate} nor {single} found. Run bash spurious_shift/run.sh first.")
