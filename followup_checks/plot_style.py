"""Shared figure styling for the follow-up checks: one light surface, thin marks, recessive axes."""

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# 分类色按固定顺序取前三个；灰色只用于作为参照的方法。
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
CONTEXT = "#898781"

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "text.color": INK_SECONDARY,
}

# 标记带一圈底色描边，重叠或压在线上时仍能分辨。
MARKER = {"markersize": 8, "markeredgecolor": SURFACE, "markeredgewidth": 1.6}


def style_axes(axis, grid):
    """Hide the top/right frame and draw hairline gridlines along ``grid`` ('x', 'y' or None)."""
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.set_axisbelow(True)
    axis.tick_params(length=3, width=0.8)
    if grid:
        axis.grid(True, axis=grid, which="major")


def plain_log_axis(axis, which, ticks):
    """Log scale with plain-number tick labels at the given positions."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    target = axis.xaxis if which == "x" else axis.yaxis
    (axis.set_xscale if which == "x" else axis.set_yscale)("log")
    target.set_major_locator(FixedLocator(ticks))
    target.set_minor_locator(NullLocator())
    target.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
