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


def _patch_title_calls() -> None:
    """Force all figure/axis/legend titles to render as empty text."""
    global _TITLE_PATCHED
    if _TITLE_PATCHED:
        return

    def _axes_set_title_no_text(self, _label, *args, **kwargs):
        return _ORIGINAL_AXES_SET_TITLE(self, "", *args, **kwargs)

    def _figure_suptitle_no_text(self, _title, *args, **kwargs):
        return _ORIGINAL_FIGURE_SUPTITLE(self, "", *args, **kwargs)

    def _legend_set_title_no_text(self, _title, *args, **kwargs):
        return _ORIGINAL_LEGEND_SET_TITLE(self, "", *args, **kwargs)

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
        kwargs["dpi"] = PUBLICATION_DPI
        kwargs["format"] = "png"
        return _ORIGINAL_FIGURE_SAVEFIG(self, out_name, *args, **kwargs)

    Figure.savefig = _figure_savefig_png_600
    _SAVEFIG_PATCHED = True


def apply_publication_figure_style(font_size: int = 11, dpi: int = 600, remove_titles: bool = True) -> None:
    """Apply a publication-ready plotting style across matplotlib/seaborn."""
    _patch_savefig_calls()
    if remove_titles:
        _patch_title_calls()

    base_size = max(int(font_size), 8)
    tick_size = max(base_size - 1, 7)
    legend_size = max(base_size - 1, 7)

    sns.set_theme(
        context="paper",
        style="ticks",
        palette="colorblind",
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": base_size,
            "axes.labelsize": base_size,
            "axes.titlesize": base_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "axes.grid": False,
            "figure.facecolor": "white",
            "figure.dpi": PUBLICATION_DPI,
            "savefig.dpi": PUBLICATION_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 5.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
