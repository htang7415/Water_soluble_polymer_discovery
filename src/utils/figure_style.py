"""Shared publication-style figure configuration."""

from __future__ import annotations

from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
import matplotlib.pyplot as plt
import seaborn as sns


_ORIGINAL_AXES_SET_TITLE = Axes.set_title
_ORIGINAL_FIGURE_SUPTITLE = Figure.suptitle
_ORIGINAL_LEGEND_SET_TITLE = Legend.set_title
_ORIGINAL_FIGURE_SAVEFIG = Figure.savefig
_TITLE_PATCHED = False
_SAVEFIG_PATCHED = False
PUBLICATION_DPI = 600
PUBLICATION_FONT_SIZE = 16
_CURRENT_PUBLICATION_DPI = PUBLICATION_DPI
_NATURE_FONT_FAMILIES = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
_NATURE_COLOR_CYCLE = [
    "#0C5DA5",
    "#00B945",
    "#FF9500",
    "#FF2C00",
    "#845B97",
    "#474747",
]


def _patch_title_calls() -> None:
    """Force all figure/axis/legend titles to render as empty text."""
    global _TITLE_PATCHED
    if _TITLE_PATCHED:
        return

    def _axes_set_title_no_text(self, *args, **kwargs):
        trailing_args = args[1:] if len(args) > 0 else ()
        kwargs.setdefault("pad", 0.0)
        return _ORIGINAL_AXES_SET_TITLE(self, "", *trailing_args, **kwargs)

    def _figure_suptitle_no_text(self, *args, **kwargs):
        trailing_args = args[1:] if len(args) > 0 else ()
        return _ORIGINAL_FIGURE_SUPTITLE(self, "", *trailing_args, **kwargs)

    def _legend_set_title_no_text(self, *args, **kwargs):
        trailing_args = args[1:] if len(args) > 0 else ()
        return _ORIGINAL_LEGEND_SET_TITLE(self, "", *trailing_args, **kwargs)

    Axes.set_title = _axes_set_title_no_text
    Figure.suptitle = _figure_suptitle_no_text
    Legend.set_title = _legend_set_title_no_text
    _TITLE_PATCHED = True


def _patch_savefig_calls() -> None:
    """Force figure saves to PNG at publication DPI."""
    global _SAVEFIG_PATCHED
    if _SAVEFIG_PATCHED:
        return

    def _figure_savefig_png_600(self, fname, *args, **kwargs):
        out_name = fname
        if isinstance(fname, (str, Path)):
            out_name = Path(fname).with_suffix(".png")
        requested_dpi = kwargs.get("dpi", _CURRENT_PUBLICATION_DPI)
        try:
            requested_dpi = int(requested_dpi)
        except Exception:
            requested_dpi = _CURRENT_PUBLICATION_DPI
        kwargs["dpi"] = max(requested_dpi, _CURRENT_PUBLICATION_DPI)
        kwargs["format"] = "png"
        return _ORIGINAL_FIGURE_SAVEFIG(self, out_name, *args, **kwargs)

    Figure.savefig = _figure_savefig_png_600
    _SAVEFIG_PATCHED = True


def apply_publication_figure_style(
    font_size: int = PUBLICATION_FONT_SIZE,
    dpi: int = PUBLICATION_DPI,
    remove_titles: bool = True,
) -> None:
    """Apply a publication-ready plotting style across matplotlib/seaborn."""
    global _CURRENT_PUBLICATION_DPI
    _CURRENT_PUBLICATION_DPI = max(int(dpi), 300)

    _patch_savefig_calls()
    if remove_titles:
        _patch_title_calls()

    base_size = max(int(font_size), PUBLICATION_FONT_SIZE)
    tick_size = base_size
    legend_size = base_size

    sns.set_theme(
        context="paper",
        style="ticks",
        palette=_NATURE_COLOR_CYCLE,
    )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": _NATURE_FONT_FAMILIES,
            "font.size": base_size,
            "axes.labelsize": base_size,
            "axes.titlesize": base_size,
            "axes.labelweight": "regular",
            "axes.titleweight": "regular",
            "axes.labelpad": 4.0,
            "axes.axisbelow": True,
            "mathtext.default": "regular",
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "legend.title_fontsize": legend_size,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.2,
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.prop_cycle": plt.cycler(color=_NATURE_COLOR_CYCLE),
            "figure.facecolor": "white",
            "figure.dpi": _CURRENT_PUBLICATION_DPI,
            "savefig.dpi": _CURRENT_PUBLICATION_DPI,
            "savefig.format": "png",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.transparent": False,
            "legend.frameon": False,
            "lines.linewidth": 2.2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 6.0,
            "patch.edgecolor": "#222222",
            "patch.linewidth": 0.8,
            "errorbar.capsize": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# Enforce global publication defaults for all figure scripts importing this module.
apply_publication_figure_style()
