"""Shared plotting style (validated light palette: blue / orange / aqua in fixed order)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3, never reordered
REF = MUTED


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 1.6, "figure.dpi": 110, "savefig.dpi": 160,
        "font.family": "sans-serif",
    })


def label_end(ax, x, y, text, color, dx=0.0, dy=0.0, ha="left"):
    """Direct label in ink colour (never the series colour) with a coloured marker beside it."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5,
                color=INK2, ha=ha, va="center",
                bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none", alpha=0.85))
    ax.plot([x], [y], "o", ms=4, color=color, mec=SURFACE, mew=1.2, zorder=5)


def save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
