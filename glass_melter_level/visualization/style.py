"""
Publication-quality figure styling (MATLAB-compatible).

Provides MATLAB default color palette, Helvetica fonts, and
rcParams optimised for two-column IEEE/Elsevier journal figures.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# MATLAB default line color palette
COLORS = {
    "blue": "#0072BD",
    "orange": "#D95319",
    "yellow": "#EDB120",
    "purple": "#7E2F8E",
    "green": "#77AC30",
    "cyan": "#4DBEEE",
    "red": "#A2142F",
    "black": "#000000",
}


def get_pub_colors():
    """
    Return semantic color mapping for publication figures.

    Returns:
        dict with keys: true, open_loop, corrected, secondary, highlight
    """
    return {
        "true": COLORS["blue"],
        "open_loop": COLORS["orange"],
        "corrected": COLORS["green"],
        "secondary": COLORS["cyan"],
        "highlight": COLORS["purple"],
    }


def setup_publication_style():
    """
    Configure matplotlib for publication-quality figures.

    Optimised for two-column IEEE/journal format at 0.48\\textwidth (~3.5").
    Uses MATLAB-like fonts and styling.
    """
    plt.style.use("default")

    pub_params = {
        # Figure
        "figure.figsize": (7, 5),
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": "white",
        # Fonts — Helvetica (MATLAB default)
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica",
            "Arial",
            "DejaVu Sans",
            "Liberation Sans",
        ],
        "font.size": 20,
        # Axes
        "axes.labelsize": 22,
        "axes.titlesize": 24,
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        # Grid
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        # Ticks — inward, all 4 sides (MATLAB default)
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        # Legend — thin black border (MATLAB-like)
        "legend.fontsize": 16,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "legend.handlelength": 2.0,
        "legend.frameon": True,
        # Lines
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        # Math text
        "mathtext.fontset": "dejavusans",
    }
    mpl.rcParams.update(pub_params)


def add_subplot_label(
    ax,
    label: str,
    loc: str = "upper left",
    fontsize: int = 24,
    fontweight: str = "bold",
):
    """
    Add alphabetical label (a), (b), (c) to a subplot corner.

    Args:
        ax: matplotlib Axes
        label: Label text, e.g. '(a)'
        loc: 'upper left', 'upper right', 'lower left', 'lower right'
        fontsize: Font size
        fontweight: Font weight
    """
    positions = {
        "upper left": (0.02, 0.95),
        "upper right": (0.98, 0.95),
        "lower left": (0.02, 0.05),
        "lower right": (0.98, 0.05),
    }
    ha_map = {
        "upper left": "left",
        "upper right": "right",
        "lower left": "left",
        "lower right": "right",
    }

    x, y = positions.get(loc, (0.02, 0.95))
    ha = ha_map.get(loc, "left")

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        va="top",
        ha=ha,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
        ),
    )
