#!/usr/bin/env python
"""Step 8: build manuscript/SI paper package from Steps 0-7 outputs.

This script curates key results into:
- manuscript figures (6 multi-panel PNGs)
- supporting information figures (PNG)
- manuscript/SI tables (CSV)
- source-data and metadata manifests

Constraint requested by user:
- Paper package is fixed to split_mode=polymer.
- Only PNG figure outputs are generated.
"""

from __future__ import annotations

import os
import argparse
import ast
import json
import math
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Avoid OpenMP shared-memory failures in restricted environments.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chi.constants import COEFF_NAMES
from src.chi.inverse_design_common import infer_coefficients_for_novel_candidates
from src.chi.model import predict_chi_from_coefficients
from src.utils.config import load_config, save_config
from src.utils.chemistry import canonicalize_smiles
from src.utils.figure_style import apply_publication_figure_style
from src.utils.model_scales import get_results_dir
from src.utils.reproducibility import save_run_metadata, seed_everything
from src.utils.reporting import save_artifact_manifest, save_step_summary, write_initial_log

PAPER_FONT_SIZE = 16
PAPER_DPI = 600
PANEL_CANVAS_WIDTH_PX = 3600
PANEL_CANVAS_HEIGHT_PX = 3000
PANEL_ASPECT = PANEL_CANVAS_WIDTH_PX / PANEL_CANVAS_HEIGHT_PX
COMPOSED_PANEL_WIDTH_IN = 4.15
COMPOSED_PANEL_HEIGHT_IN = COMPOSED_PANEL_WIDTH_IN / PANEL_ASPECT
COMPOSED_GAP_WIDTH_IN = 0.26
COMPOSED_GAP_HEIGHT_IN = 0.28
COMPOSED_MARGIN_LEFT_IN = 0.18
COMPOSED_MARGIN_RIGHT_IN = 0.12
COMPOSED_MARGIN_BOTTOM_IN = 0.12
COMPOSED_MARGIN_TOP_IN = 0.18
COMPOSED_WARNING_BANNER_IN = 0.34
PANEL_IMAGE_MARGIN_FRAC = 0.035
NATURE_BLUE = "#0C5DA5"
NATURE_GREEN = "#00B945"
NATURE_ORANGE = "#FF9500"
NATURE_RED = "#FF2C00"
NATURE_PURPLE = "#845B97"
NATURE_GRAY = "#474747"
NATURE_LIGHT_BLUE = "#8FC2E7"
NATURE_PALETTE = [NATURE_BLUE, NATURE_GREEN, NATURE_ORANGE, NATURE_RED, NATURE_PURPLE, NATURE_GRAY]
WATER_CLASS_PALETTE = {0: NATURE_RED, 1: NATURE_BLUE}


@dataclass
class PanelSpec:
    """One panel in a composed figure."""

    caption: str
    candidates: List[Path]


@dataclass
class FigureSpec:
    """Composed figure definition."""

    figure_id: str
    title: str
    panels: List[PanelSpec]
    ncols: int
    destination: str  # manuscript | si


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p is not None and p.exists():
            return p
    return None


def _safe_read_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_first_row(path: Optional[Path]) -> Dict[str, object]:
    df = _safe_read_csv(path)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def _pick(row: Dict[str, object], keys: List[str], default=np.nan):
    for k in keys:
        if k in row:
            v = row.get(k)
            if isinstance(v, str) and v.strip() == "":
                continue
            if pd.isna(v):
                continue
            return v
    return default


def _is_missing_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _merge_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    merged: Dict[str, object] = {}
    for row in rows:
        if not row:
            continue
        for key, value in row.items():
            if key not in merged or _is_missing_value(merged.get(key)):
                merged[key] = value
    return merged


def _parse_literal_value(raw: object) -> object:
    if raw is None:
        return None
    if isinstance(raw, (list, dict, tuple)):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            return ast.literal_eval(text)
        except Exception:
            return text
    return raw


def _load_step0_summary_fallback(step0_dir: Optional[Path]) -> Dict[str, object]:
    if step0_dir is None:
        return {}

    metrics_dir = step0_dir / "metrics"
    roundtrip_df = _safe_read_csv(metrics_dir / "tokenizer_roundtrip.csv")
    stats_df = _safe_read_csv(metrics_dir / "unlabeled_data_stats.csv")
    if roundtrip_df.empty and stats_df.empty:
        return {}

    summary: Dict[str, object] = {"step": "step0_data_prep"}
    if not roundtrip_df.empty:
        roundtrip_df = roundtrip_df.copy()
        roundtrip_df["split"] = roundtrip_df["split"].astype(str).str.lower()
        train_row = roundtrip_df.loc[roundtrip_df["split"] == "train"]
        val_row = roundtrip_df.loc[roundtrip_df["split"] == "val"]
        if not train_row.empty:
            summary["train_roundtrip_pct"] = _pick(train_row.iloc[0].to_dict(), ["pct"], default=np.nan)
        if not val_row.empty:
            summary["val_roundtrip_pct"] = _pick(val_row.iloc[0].to_dict(), ["pct"], default=np.nan)

    if not stats_df.empty:
        stats_df = stats_df.copy()
        stats_df["split"] = stats_df["split"].astype(str).str.lower()
        train_row = stats_df.loc[stats_df["split"] == "train"]
        val_row = stats_df.loc[stats_df["split"] == "val"]
        if not train_row.empty:
            train_data = train_row.iloc[0].to_dict()
            summary["train_samples"] = _pick(train_data, ["count"], default=np.nan)
            summary["train_mean_length"] = _pick(train_data, ["length_mean"], default=np.nan)
            summary["train_mean_sa"] = _pick(train_data, ["sa_mean"], default=np.nan)
        if not val_row.empty:
            val_data = val_row.iloc[0].to_dict()
            summary["val_samples"] = _pick(val_data, ["count"], default=np.nan)
            summary["val_mean_length"] = _pick(val_data, ["length_mean"], default=np.nan)
            summary["val_mean_sa"] = _pick(val_data, ["sa_mean"], default=np.nan)
    return summary


def _resolve_truetype_font_path() -> Optional[str]:
    """Resolve a readable sans-serif TrueType font path on common systems."""
    candidates: List[str] = []
    try:
        found = font_manager.findfont("DejaVu Sans", fallback_to_default=False)
        if isinstance(found, str) and found.strip():
            candidates.append(found)
    except Exception:
        pass

    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/Library/Fonts/Arial.ttf",
            "DejaVuSans.ttf",
            "Arial.ttf",
        ]
    )

    seen = set()
    for p in candidates:
        norm = str(p).strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        try:
            if Path(norm).is_file():
                return norm
        except Exception:
            continue
    return None


def _make_candidates(base_dirs: List[Optional[Path]], names: List[str]) -> List[Path]:
    out: List[Path] = []
    for d in base_dirs:
        if d is None:
            continue
        for name in names:
            out.append(d / name)
    return out


def _crop_panel_whitespace(img: np.ndarray, threshold: float = 0.985, pad_px: int = 16) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim < 2:
        return arr

    arr_float = arr.astype(np.float32, copy=False)
    if arr_float.size == 0:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        denom = float(np.iinfo(arr.dtype).max)
        if denom > 0:
            arr_float = arr_float / denom
    elif float(np.nanmax(arr_float)) > 1.5:
        arr_float = arr_float / 255.0

    if arr_float.ndim == 2:
        background = arr_float >= threshold
    else:
        rgb = arr_float[..., :3]
        background = np.all(rgb >= threshold, axis=2)
        if arr_float.shape[2] >= 4:
            background = np.logical_or(background, arr_float[..., 3] <= 0.02)

    foreground = ~background
    if not np.any(foreground):
        return arr

    rows = np.where(np.any(foreground, axis=1))[0]
    cols = np.where(np.any(foreground, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return arr

    y0 = max(int(rows[0]) - pad_px, 0)
    y1 = min(int(rows[-1]) + pad_px + 1, arr.shape[0])
    x0 = max(int(cols[0]) - pad_px, 0)
    x1 = min(int(cols[-1]) + pad_px + 1, arr.shape[1])
    if arr.ndim == 2:
        return arr[y0:y1, x0:x1]
    return arr[y0:y1, x0:x1, ...]


def _load_panel_image(src: Path) -> np.ndarray:
    img = mpimg.imread(src)
    return _pad_image_to_panel_aspect(_add_panel_outer_margin(_crop_panel_whitespace(img)))


def _add_panel_outer_margin(img: np.ndarray, margin_frac: float = PANEL_IMAGE_MARGIN_FRAC) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim < 2:
        return arr

    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h <= 0 or w <= 0 or margin_frac <= 0.0:
        return arr

    pad_y = max(1, int(round(h * float(margin_frac))))
    pad_x = max(1, int(round(w * float(margin_frac))))
    fill_val = np.iinfo(arr.dtype).max if np.issubdtype(arr.dtype, np.integer) else 1.0

    if arr.ndim == 2:
        canvas = np.full((h + 2 * pad_y, w + 2 * pad_x), fill_val, dtype=arr.dtype)
        canvas[pad_y : pad_y + h, pad_x : pad_x + w] = arr
        return canvas

    c = int(arr.shape[2])
    canvas = np.full((h + 2 * pad_y, w + 2 * pad_x, c), fill_val, dtype=arr.dtype)
    if c == 4:
        canvas[..., 3] = fill_val
    canvas[pad_y : pad_y + h, pad_x : pad_x + w, :] = arr
    return canvas


def _pad_image_to_panel_aspect(img: np.ndarray, target_aspect: float = PANEL_ASPECT) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim < 2:
        return arr

    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h <= 0 or w <= 0:
        return arr

    current_aspect = float(w) / float(h)
    if np.isclose(current_aspect, float(target_aspect), atol=1.0e-3):
        return arr

    fill_val = np.iinfo(arr.dtype).max if np.issubdtype(arr.dtype, np.integer) else 1.0
    if current_aspect < float(target_aspect):
        target_w = int(math.ceil(h * float(target_aspect)))
        target_h = h
    else:
        target_w = w
        target_h = int(math.ceil(w / float(target_aspect)))

    if arr.ndim == 2:
        canvas = np.full((target_h, target_w), fill_val, dtype=arr.dtype)
        y0 = (target_h - h) // 2
        x0 = (target_w - w) // 2
        canvas[y0 : y0 + h, x0 : x0 + w] = arr
        return canvas

    c = int(arr.shape[2])
    canvas = np.full((target_h, target_w, c), fill_val, dtype=arr.dtype)
    if c == 4:
        canvas[..., 3] = fill_val
    y0 = (target_h - h) // 2
    x0 = (target_w - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w, :] = arr
    return canvas


def _panel_label(ax, label: str, font_size: int) -> None:
    ax.text(
        0.02,
        0.98,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=font_size,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5},
    )


def _draw_missing_panel(ax, caption: str, font_size: int) -> None:
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#f3f4f6", transform=ax.transAxes, zorder=0))
    ax.text(
        0.5,
        0.55,
        "Missing source figure",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=font_size,
        color="#4b5563",
    )
    if isinstance(caption, str) and caption.strip():
        ax.text(
            0.5,
            0.40,
            caption.strip(),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=max(10, font_size - 3),
            color="#6b7280",
            wrap=True,
        )


def _compose_figure(
    spec: FigureSpec,
    out_png: Path,
) -> pd.DataFrame:
    # Hard-enforce paper style for all Step 8 figure annotations.
    dpi = PAPER_DPI
    font_size = PAPER_FONT_SIZE

    panel_rows: List[Dict[str, object]] = []
    n_panels = len(spec.panels)
    ncols = max(1, int(spec.ncols))
    nrows = int(math.ceil(n_panels / ncols))

    fig_w = (
        COMPOSED_MARGIN_LEFT_IN
        + COMPOSED_MARGIN_RIGHT_IN
        + ncols * COMPOSED_PANEL_WIDTH_IN
        + max(0, ncols - 1) * COMPOSED_GAP_WIDTH_IN
    )
    fig_h = (
        COMPOSED_MARGIN_BOTTOM_IN
        + COMPOSED_MARGIN_TOP_IN
        + nrows * COMPOSED_PANEL_HEIGHT_IN
        + max(0, nrows - 1) * COMPOSED_GAP_HEIGHT_IN
    )
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    axes = np.array(axes).reshape(-1)

    letters = "abcdefghijklmnopqrstuvwxyz"
    missing_count = 0
    for i, panel in enumerate(spec.panels):
        ax = axes[i]
        label = letters[i] if i < len(letters) else str(i + 1)
        src = _first_existing(panel.candidates)
        status = "missing"
        src_str = ""

        if src is not None and src.exists():
            try:
                img = _load_panel_image(src)
                ax.imshow(img)
                ax.set_axis_off()
                status = "ok"
                src_str = str(src)
            except Exception:
                _draw_missing_panel(ax, panel.caption, font_size=font_size)
        else:
            _draw_missing_panel(ax, panel.caption, font_size=font_size)

        if status != "ok":
            missing_count += 1

        _panel_label(ax, label=label, font_size=font_size)
        panel_rows.append(
            {
                "figure_id": spec.figure_id,
                "panel_index": i + 1,
                "panel_label": label,
                "caption": panel.caption,
                "source_path": src_str,
                "status": status,
            }
        )

    for j in range(n_panels, len(axes)):
        axes[j].set_axis_off()

    top_margin_in = COMPOSED_MARGIN_TOP_IN
    if missing_count > 0:
        top_margin_in += COMPOSED_WARNING_BANNER_IN
        fig.text(
            0.5,
            0.992,
            f"Incomplete figure: {missing_count}/{n_panels} source panels missing",
            ha="center",
            va="top",
            fontsize=max(10, font_size - 2),
            color="#b91c1c",
        )

    # No global title or panel captions rendered on composed figures.
    fig.subplots_adjust(
        left=COMPOSED_MARGIN_LEFT_IN / fig_w,
        right=1.0 - (COMPOSED_MARGIN_RIGHT_IN / fig_w),
        bottom=COMPOSED_MARGIN_BOTTOM_IN / fig_h,
        top=1.0 - (top_margin_in / fig_h),
        wspace=COMPOSED_GAP_WIDTH_IN / COMPOSED_PANEL_WIDTH_IN,
        hspace=COMPOSED_GAP_HEIGHT_IN / COMPOSED_PANEL_HEIGHT_IN,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    return pd.DataFrame(panel_rows)


def _copy_if_exists(src: Optional[Path], dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _slugify(text: str) -> str:
    # Keep figure filenames portable across OS/filesystems and preserve key symbols semantically.
    norm_text = str(text)
    symbol_map = {
        "χ": "chi",
        "φ": "phi",
        "Φ": "phi",
        "Δ": "delta",
        "δ": "delta",
        "β": "beta",
        "α": "alpha",
    }
    for src, dst in symbol_map.items():
        norm_text = norm_text.replace(src, dst)
    norm_text = (
        unicodedata.normalize("NFKD", norm_text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    chars: List[str] = []
    for ch in norm_text.strip().lower():
        chars.append(ch if ch.isalnum() else "_")
    slug = re.sub(r"_+", "_", "".join(chars)).strip("_")
    return slug if slug else "figure"


def _figure_output_name(spec: FigureSpec) -> str:
    return f"{_slugify(spec.title)}.png"


def _clear_png_files(root_dir: Path) -> None:
    if root_dir is None or not root_dir.exists():
        return
    for png in root_dir.rglob("*.png"):
        try:
            png.unlink()
        except Exception:
            continue


def _pad_png_canvas(
    png_path: Path,
    target_w: int = PANEL_CANVAS_WIDTH_PX,
    target_h: int = PANEL_CANVAS_HEIGHT_PX,
) -> None:
    """Center-pad a generated PNG to a fixed canvas for consistent panel sizing."""
    if png_path is None or not png_path.exists():
        return
    try:
        img = mpimg.imread(png_path)
    except Exception:
        return
    if img is None or not hasattr(img, "shape") or len(img.shape) < 2:
        return

    h, w = int(img.shape[0]), int(img.shape[1])
    if h == target_h and w == target_w:
        return
    if h > target_h or w > target_w:
        return

    fill_val = np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else 1.0
    if len(img.shape) == 2:
        canvas = np.full((target_h, target_w), fill_val, dtype=img.dtype)
        y0 = (target_h - h) // 2
        x0 = (target_w - w) // 2
        canvas[y0 : y0 + h, x0 : x0 + w] = img
    else:
        c = int(img.shape[2])
        canvas = np.full((target_h, target_w, c), fill_val, dtype=img.dtype)
        if c == 4:
            canvas[..., 3] = fill_val
        y0 = (target_h - h) // 2
        x0 = (target_w - w) // 2
        canvas[y0 : y0 + h, x0 : x0 + w, :] = img
    plt.imsave(png_path, canvas, dpi=PAPER_DPI)


def _boxed_legend(
    ax: plt.Axes,
    *,
    loc: str = "upper right",
    title: Optional[str] = None,
    ncol: int = 1,
    fontsize: Optional[int] = None,
):
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) == 0:
        return None
    legend = ax.legend(
        handles=handles,
        labels=labels,
        title=title,
        loc=loc,
        ncol=ncol,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#666666",
        fontsize=fontsize if fontsize is not None else max(10, PAPER_FONT_SIZE - 3),
    )
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_fontsize(fontsize if fontsize is not None else max(10, PAPER_FONT_SIZE - 3))
    return legend


def _regression_summary_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_pred - y_true))) if len(y_true) else np.nan
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2))) if len(y_true) else np.nan
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _build_step2_generative_metrics_panel(
    paths: Dict[str, Optional[Path]],
    metadata_dir: Path,
) -> Optional[Path]:
    """Build Figure 1(f): compact Step 2 generative-quality score summary."""
    step2_dir = paths.get("step2_dir")

    metrics_csv = step2_dir / "metrics" / "sampling_generative_metrics.csv" if step2_dir is not None else None
    metrics_row = _safe_first_row(metrics_csv)
    if not metrics_row:
        metrics_row = _safe_first_row(paths.get("step2_summary"))
    if not metrics_row:
        return None

    metric_specs = [
        ("Validity", ["validity"]),
        ("Validity (star=2)", ["validity_two_stars", "frac_star_eq_2"]),
        ("Uniqueness", ["uniqueness"]),
        ("Novelty", ["novelty"]),
        ("Diversity", ["avg_diversity"]),
    ]
    labels: List[str] = []
    values: List[float] = []
    for label, keys in metric_specs:
        raw_val = _pick(metrics_row, keys, default=np.nan)
        try:
            val = float(raw_val)
        except Exception:
            continue
        if not np.isfinite(val):
            continue
        labels.append(label)
        values.append(float(np.clip(val, 0.0, 1.0)))

    if len(values) < 3:
        return None

    out_png = metadata_dir / "derived_step2_generative_metrics_summary.png"
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ypos = np.arange(len(labels))
    ax.barh(ypos, values, color=NATURE_PALETTE[: len(labels)])

    for i, val in enumerate(values):
        near_right = val > 0.94
        ax.text(
            val - 0.02 if near_right else val + 0.015,
            i,
            f"{val:.3f}",
            va="center",
            ha="right" if near_right else "left",
            fontsize=max(11, PAPER_FONT_SIZE - 2),
            color="#111827",
        )

    ax.set_yticks(ypos, labels=labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Score", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=PAPER_FONT_SIZE)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.8, alpha=0.35)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _build_step2_sampling_information_panel(
    paths: Dict[str, Optional[Path]],
    metadata_dir: Path,
) -> Optional[Path]:
    summary_row = _safe_first_row(paths.get("step2_summary"))
    if not summary_row:
        return None

    goal = _pick(summary_row, ["generation_goal", "target_polymer_count_requested"], default=np.nan)
    generated = _pick(summary_row, ["generated_count", "valid_only_raw_generated"], default=np.nan)
    accepted = _pick(summary_row, ["accepted_count_for_evaluation"], default=np.nan)
    rounds = _pick(summary_row, ["valid_only_rounds"], default=np.nan)
    time_sec = _pick(summary_row, ["sampling_time_sec"], default=np.nan)
    throughput = _pick(summary_row, ["samples_per_sec", "valid_per_sec"], default=np.nan)
    acceptance_rate = _pick(summary_row, ["valid_only_acceptance_rate"], default=np.nan)
    shortfall = _pick(summary_row, ["valid_only_shortfall_count"], default=np.nan)

    try:
        goal_int = int(float(goal))
        generated_int = int(float(generated))
        accepted_int = int(float(accepted))
        rounds_int = int(float(rounds))
        time_sec_float = float(time_sec)
        throughput_float = float(throughput)
    except Exception:
        return None
    if goal_int <= 0 or generated_int <= 0 or accepted_int <= 0:
        return None

    goal_coverage = 100.0 * accepted_int / max(goal_int, 1)
    footer_bits: List[str] = []
    try:
        footer_bits.append(f"Acceptance rate: {100.0 * float(acceptance_rate):.1f}%")
    except Exception:
        pass
    try:
        footer_bits.append(f"Shortfall: {int(float(shortfall)):,}")
    except Exception:
        pass
    footer_bits.append(f"Valid-only rounds: {rounds_int}")

    cards = [
        ("Generation goal", f"{goal_int:,}", NATURE_BLUE),
        ("Raw generated", f"{generated_int:,}", NATURE_GREEN),
        ("Accepted for evaluation", f"{accepted_int:,}", NATURE_ORANGE),
        ("Goal achieved", f"{goal_coverage:.1f}%", NATURE_RED),
        ("Elapsed time", f"{time_sec_float / 60.0:.1f} min", NATURE_PURPLE),
        ("Throughput", f"{throughput_float:.2f}/s", NATURE_GRAY),
    ]

    out_png = metadata_dir / "derived_step2_sampling_information_summary.png"
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.set_axis_off()
    card_w = 0.41
    card_h = 0.20
    x_positions = [0.07, 0.52]
    y_positions = [0.73, 0.47, 0.21]

    for idx, (label, value, color) in enumerate(cards):
        row = idx // 2
        col = idx % 2
        x0 = x_positions[col]
        y0 = y_positions[row]
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                card_w,
                card_h,
                boxstyle="round,pad=0.012,rounding_size=0.03",
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
                alpha=0.96,
            )
        )
        ax.text(
            x0 + 0.035,
            y0 + card_h - 0.050,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=max(10, PAPER_FONT_SIZE - 4),
            color="white",
            fontweight="semibold",
        )
        ax.text(
            x0 + card_w / 2.0,
            y0 + 0.085,
            value,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=max(15, PAPER_FONT_SIZE + 2),
            color="white",
            fontweight="bold",
        )

    ax.text(
        0.5,
        0.055,
        " | ".join(footer_bits),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=max(10, PAPER_FONT_SIZE - 4),
        color="#111827",
    )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _build_step2_star_count_panel(
    paths: Dict[str, Optional[Path]],
    metadata_dir: Path,
) -> Optional[Path]:
    summary_row = _safe_first_row(paths.get("step2_summary"))
    if not summary_row:
        return None

    total = _pick(summary_row, ["accepted_count_for_evaluation", "target_polymer_count_selected"], default=np.nan)
    target_stars = _pick(summary_row, ["target_stars"], default=np.nan)
    frac_target = _pick(summary_row, ["frac_star_eq_2", "validity_two_stars"], default=np.nan)
    try:
        total_int = int(float(total))
        target_star_int = int(float(target_stars))
        frac_target = float(frac_target)
    except Exception:
        return None
    if total_int <= 0 or not np.isfinite(frac_target):
        return None

    target_count = int(round(total_int * np.clip(frac_target, 0.0, 1.0)))
    other_count = max(total_int - target_count, 0)
    labels = [f"Star={target_star_int}", "Other"]
    values = [target_count, other_count]

    out_png = metadata_dir / "derived_step2_star_count_summary.png"
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    bars = ax.bar(labels, values, color=[NATURE_BLUE, "#94a3b8"])
    y_max = max(1, max(values))
    for bar, value in zip(bars, values):
        pct = 100.0 * value / total_int if total_int > 0 else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.03 * y_max,
            f"{value:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=max(10, PAPER_FONT_SIZE - 2),
        )

    ax.text(
        0.98,
        0.96,
        f"Total: {total_int:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=max(10, PAPER_FONT_SIZE - 2),
        color="#111827",
    )
    ax.set_xlabel("Star count", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("Count", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=PAPER_FONT_SIZE)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.35)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _build_screening_funnel_panel(
    summary_csv: Optional[Path],
    out_png: Path,
) -> Optional[Path]:
    row = _safe_first_row(summary_csv)
    if not row:
        return None

    stage_rows: List[tuple[str, int]] = []
    stage_specs = [
        ("Candidates screened", ["step5_candidates_screened_total", "step6_candidates_screened_total", "total_candidates_screened"]),
        ("Qualified after screening", ["step5_qualified_candidate_count", "step6_qualified_candidate_count", "filter_pass_count"]),
        ("After deduplication", ["filter_pass_unique"]),
        ("Final selection", ["step5_target_count_selected", "step6_target_count_selected", "target_count_selected"]),
    ]
    for label, keys in stage_specs:
        value = _pick(row, keys, default=np.nan)
        if _is_missing_value(value):
            continue
        try:
            int_value = int(float(value))
        except Exception:
            continue
        if stage_rows and label == "After deduplication" and int_value == stage_rows[-1][1]:
            continue
        stage_rows.append((label, int_value))
    if len(stage_rows) < 2:
        return None
    values = [value for _, value in stage_rows]
    if max(values) <= 0:
        return None

    colors = [NATURE_BLUE, NATURE_GREEN, NATURE_ORANGE, NATURE_RED]
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ys = np.arange(len(stage_rows))
    bars = ax.barh(ys, values, color=colors[: len(stage_rows)], edgecolor="none", height=0.6)
    x_max = max(values)
    for bar, val in zip(bars, values):
        pct = 100.0 * val / x_max if x_max > 0 else 0.0
        ax.text(
            val + 0.02 * x_max,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:,} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=max(10, PAPER_FONT_SIZE - 2),
        )

    ax.set_yticks(ys, labels=[label for label, _ in stage_rows])
    ax.invert_yaxis()
    ax.set_xlabel("Count", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=PAPER_FONT_SIZE)
    ax.set_xlim(0, x_max * 1.45)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.8, alpha=0.35)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _build_sampling_attempt_progress_panel(
    attempts_csv: Optional[Path],
    out_png: Path,
    target_count: int = 100,
) -> Optional[Path]:
    df = _safe_read_csv(attempts_csv)
    return _build_sampling_attempt_progress_panel_from_df(
        df=df,
        out_png=out_png,
        target_count=target_count,
    )


def _build_sampling_attempt_progress_panel_from_df(
    df: pd.DataFrame,
    out_png: Path,
    target_count: int = 100,
) -> Optional[Path]:
    if df.empty or not {"sampling_attempt", "target_count_selected"}.issubset(df.columns):
        return None

    plot_df = df.copy()
    plot_df["sampling_attempt"] = pd.to_numeric(plot_df["sampling_attempt"], errors="coerce")
    plot_df["target_count_selected"] = pd.to_numeric(plot_df["target_count_selected"], errors="coerce")
    plot_df = plot_df.dropna(subset=["sampling_attempt", "target_count_selected"])
    if plot_df.empty:
        return None

    good_col = "target_count_selected"
    for col in ["n_polymers_pass_all_targets", "qualified_candidate_count", "filter_pass_count", "target_count_selected"]:
        if col in plot_df.columns:
            numeric_vals = pd.to_numeric(plot_df[col], errors="coerce")
            if numeric_vals.notna().any():
                plot_df[col] = numeric_vals
                good_col = col
                break

    plot_df = plot_df.dropna(subset=[good_col]).sort_values("sampling_attempt")
    if plot_df.empty:
        return None

    x = plot_df["sampling_attempt"].to_numpy(dtype=float)
    y = plot_df[good_col].to_numpy(dtype=float)
    y_max = max(float(np.nanmax(y)), float(target_count), 1.0)
    y_span = max(y_max - float(np.nanmin(y)), 1.0)
    y_pad = max(8.0, 0.10 * y_span)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot(
        x,
        y,
        marker="o",
        markersize=7.0,
        markerfacecolor="white",
        markeredgewidth=1.8,
        linewidth=2.6,
        color=NATURE_BLUE,
        zorder=3,
    )
    ax.axhline(float(target_count), color=NATURE_RED, linestyle="--", linewidth=1.8, zorder=1)
    for x_i, y_i in zip(x, y):
        ax.text(
            x_i,
            y_i + 0.04 * y_span + 0.15,
            f"{int(round(y_i))}",
            ha="center",
            va="bottom",
            fontsize=PAPER_FONT_SIZE,
            color="#111827",
        )

    ax.text(
        0.03,
        0.96,
        f"Target = {int(round(float(target_count))):,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PAPER_FONT_SIZE,
        color=NATURE_RED,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.92},
    )
    ax.set_xlabel("Sampling attempt", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("Good polymers", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=PAPER_FONT_SIZE)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)
    ax.set_xlim(float(np.min(x)) - 0.15, float(np.max(x)) + 0.15)
    ax.set_ylim(bottom=max(0.0, float(np.nanmin(y)) - y_pad), top=y_max + y_pad)
    xticks = sorted({int(v) for v in x.tolist()})
    if xticks:
        ax.set_xticks(xticks)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _resampling_attempt_dirs(step_dir: Optional[Path]) -> List[Path]:
    if step_dir is None or not step_dir.exists():
        return []

    def _attempt_key(path: Path) -> int:
        match = re.search(r"(\d+)$", path.name)
        return int(match.group(1)) if match else -1

    return sorted(
        [p for p in step_dir.glob("step2_resampling_attempt_*") if p.is_dir()],
        key=_attempt_key,
    )


def _latest_resampling_attempt_csv(
    step_dir: Optional[Path],
    file_name: str,
) -> Optional[Path]:
    for attempt_dir in reversed(_resampling_attempt_dirs(step_dir)):
        candidate = attempt_dir / "metrics" / file_name
        if candidate.exists():
            return candidate
    return None


def _build_resampling_attempt_progress_panel(
    step_dir: Optional[Path],
    out_png: Path,
) -> Optional[Path]:
    rows: List[Dict[str, object]] = []
    target_goal = 0
    for attempt_dir in _resampling_attempt_dirs(step_dir):
        match = re.search(r"(\d+)$", attempt_dir.name)
        if match is None:
            continue
        attempt_idx = int(match.group(1))
        summary_row = _safe_first_row(attempt_dir / "metrics" / "target_polymer_selection_summary.csv")
        step_row = _safe_first_row(attempt_dir / "metrics" / "step_summary.csv")
        merged = _merge_rows([summary_row, step_row])
        if not merged:
            continue

        selected = _pick(merged, ["target_count_selected", "target_polymer_count_selected"], default=np.nan)
        qualified = _pick(merged, ["filter_pass_count", "filter_pass_unique", "total_evaluated_for_filters"], default=np.nan)
        requested = _pick(merged, ["target_count_requested", "target_polymer_count_requested"], default=np.nan)
        try:
            selected_int = int(float(selected))
        except Exception:
            continue
        row = {
            "sampling_attempt": attempt_idx,
            "target_count_selected": selected_int,
        }
        try:
            row["qualified_candidate_count"] = int(float(qualified))
        except Exception:
            pass
        rows.append(row)
        try:
            target_goal = max(target_goal, int(float(requested)))
        except Exception:
            target_goal = max(target_goal, selected_int)

    if not rows:
        return None

    plot_df = pd.DataFrame(rows).sort_values("sampling_attempt").reset_index(drop=True)
    return _build_sampling_attempt_progress_panel_from_df(
        df=plot_df,
        out_png=out_png,
        target_count=max(target_goal, int(plot_df["target_count_selected"].max())),
    )


def _completed_resampling_attempt_count(step_dir: Optional[Path]) -> int:
    count = 0
    for attempt_dir in _resampling_attempt_dirs(step_dir):
        if (attempt_dir / "metrics" / "target_polymers.csv").exists() or (
            attempt_dir / "metrics" / "target_polymer_selection_summary.csv"
        ).exists():
            count += 1
    return count


def _build_requirement_snapshot_panel(
    target_csv: Optional[Path],
    out_png: Path,
) -> Optional[Path]:
    df = _safe_read_csv(target_csv)
    if df.empty:
        return None

    requirement_specs = [
        ("Soluble hit", ["soluble_hit"], "binary"),
        ("Property hit", ["property_hit"], "binary"),
        ("Valid", ["is_valid"], "binary"),
        ("Two-star", ["star_count"], "star_count"),
        ("Novel vs train", ["is_novel_vs_train", "is_novel"], "binary"),
        ("SA within limit", ["sa_ok"], "binary"),
        ("All target conditions", ["passes_all_target_conditions"], "binary"),
        ("All filters", ["passes_all_filters"], "binary"),
        ("Target class hit", ["polymer_class_hit"], "binary"),
    ]
    rows: List[tuple[str, float]] = []
    for label, col_candidates, mode in requirement_specs:
        col = next((candidate for candidate in col_candidates if candidate in df.columns), None)
        if col is None:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if mode == "star_count":
            vals = (vals >= 2).astype(float)
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue
        rows.append((label, float((vals >= 0.5).mean())))
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    labels = [label for label, _ in rows]
    values = [value for _, value in rows]
    ys = np.arange(len(labels))
    bars = ax.barh(ys, values, color=NATURE_BLUE, edgecolor="none", height=0.65)
    for bar, val in zip(bars, values):
        ax.text(
            min(val + 0.02, 1.02),
            bar.get_y() + bar.get_height() / 2.0,
            f"{100.0 * val:.0f}%",
            va="center",
            ha="left",
            fontsize=max(10, PAPER_FONT_SIZE - 2),
        )
    ax.set_yticks(ys, labels=labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Pass rate across selected targets", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=max(10, PAPER_FONT_SIZE - 2))
    ax.grid(True, axis="x", linestyle="--", linewidth=0.8, alpha=0.35)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _selection_summary_csv_from_target_csv(target_csv: Optional[Path]) -> Optional[Path]:
    if target_csv is None:
        return None
    candidate = target_csv.parent / "target_polymer_selection_summary.csv"
    return candidate if candidate.exists() else None


def _resolve_step6_target_plot_csv(
    *,
    target_csv: Optional[Path],
    inverse_targets_csv: Optional[Path],
    step4_reg_dir: Optional[Path],
    step4_cls_dir: Optional[Path],
    config: Dict,
    model_size: str,
    split_mode: str,
    cache_csv: Path,
) -> Optional[Path]:
    if target_csv is None or not target_csv.exists():
        return None

    source_df = _safe_read_csv(target_csv)
    if source_df.empty:
        return None
    if {"chi_pred_target", "class_prob", "target_chi"}.issubset(source_df.columns):
        return target_csv

    if step4_reg_dir is None or step4_cls_dir is None:
        return target_csv

    reg_checkpoint = step4_reg_dir / "checkpoints" / "chi_regression_best.pt"
    cls_checkpoint = step4_cls_dir / "checkpoints" / "chi_classifier_best.pt"
    if not reg_checkpoint.exists() or not cls_checkpoint.exists():
        return target_csv

    novel_df = source_df.copy()
    smiles_col = "SMILES" if "SMILES" in novel_df.columns else ("smiles" if "smiles" in novel_df.columns else None)
    if smiles_col is None:
        return target_csv

    novel_df["SMILES"] = novel_df[smiles_col].astype(str)
    if "canonical_smiles" in novel_df.columns:
        canonical_series = novel_df["canonical_smiles"].astype(str)
    else:
        canonical_series = pd.Series([""] * len(novel_df), index=novel_df.index, dtype=object)
    canonical_series = canonical_series.where(canonical_series.str.strip() != "", novel_df["SMILES"])
    novel_df["canonical_smiles"] = canonical_series.apply(
        lambda s: canonicalize_smiles(s) or str(s).strip()
    )
    novel_df["Polymer"] = novel_df.get("Polymer", novel_df["canonical_smiles"]).astype(str)
    novel_df["polymer_id"] = np.arange(1, len(novel_df) + 1, dtype=int)

    if cache_csv.exists():
        cache_df = _safe_read_csv(cache_csv)
        required_cols = {"polymer_id", "canonical_smiles", "chi_pred_target", "class_prob", "target_chi", *COEFF_NAMES}
        if required_cols.issubset(cache_df.columns) and len(cache_df) == len(novel_df):
            cached_canonical = cache_df["canonical_smiles"].astype(str).fillna("").tolist()
            source_canonical = novel_df["canonical_smiles"].astype(str).fillna("").tolist()
            if cached_canonical == source_canonical:
                return cache_csv

    inverse_df = _safe_read_csv(inverse_targets_csv)
    inverse_row = inverse_df.iloc[0].to_dict() if not inverse_df.empty else {}
    temperature = _pick(inverse_row, ["temperature"], default=np.nan)
    phi = _pick(inverse_row, ["phi"], default=np.nan)
    target_chi = _pick(inverse_row, ["target_chi"], default=np.nan)
    property_rule = _pick(inverse_row, ["property_rule"], default="")
    target_class = _pick(inverse_row, ["target_polymer_class"], default="")
    if any(not np.isfinite(float(v)) for v in [temperature, phi, target_chi]):
        return target_csv

    chi_cfg = config.get("chi_training", {})
    shared_cfg = chi_cfg.get("shared", {}) if isinstance(chi_cfg.get("shared", {}), dict) else {}
    embedding_cfg = shared_cfg.get("embedding", {}) if isinstance(shared_cfg.get("embedding", {}), dict) else {}
    batch_size = int(embedding_cfg.get("batch_size", 128))
    timestep = int(embedding_cfg.get("timestep", config.get("training_property", {}).get("default_timestep", 1)))
    pooling = str(embedding_cfg.get("pooling", "mean")).strip().lower() or "mean"

    try:
        inferred_df = infer_coefficients_for_novel_candidates(
            novel_df=novel_df[["polymer_id", "Polymer", "SMILES", "canonical_smiles"]].copy(),
            config=config,
            model_size=model_size,
            split_mode=split_mode,
            chi_checkpoint_path=reg_checkpoint,
            class_checkpoint_path=cls_checkpoint,
            backbone_checkpoint_path=None,
            device="cpu",
            timestep=timestep,
            pooling=pooling,
            batch_size=batch_size,
            uncertainty_enabled=False,
        )
    except Exception as exc:
        print(f"[WARN] Step 6 target rescoring failed; Figure 5 chi/probability panels may be unavailable: {exc}")
        return target_csv

    if inferred_df.empty or not {"polymer_id", "class_prob", *COEFF_NAMES}.issubset(inferred_df.columns):
        return target_csv

    merged_df = novel_df.merge(
        inferred_df[["polymer_id", "class_prob", "class_prob_std", *COEFF_NAMES]],
        on="polymer_id",
        how="left",
    )
    merged_df["temperature"] = pd.to_numeric(merged_df.get("temperature", temperature), errors="coerce")
    merged_df["phi"] = pd.to_numeric(merged_df.get("phi", phi), errors="coerce")
    merged_df["target_chi"] = pd.to_numeric(merged_df.get("target_chi", target_chi), errors="coerce")
    merged_df["temperature"] = merged_df["temperature"].fillna(float(temperature))
    merged_df["phi"] = merged_df["phi"].fillna(float(phi))
    merged_df["target_chi"] = merged_df["target_chi"].fillna(float(target_chi))
    if "property_rule" not in merged_df.columns or merged_df["property_rule"].isna().all():
        merged_df["property_rule"] = property_rule
    if "target_polymer_class" not in merged_df.columns or merged_df["target_polymer_class"].isna().all():
        merged_df["target_polymer_class"] = target_class

    coeff_matrix = merged_df[COEFF_NAMES].to_numpy(dtype=float)
    merged_df["chi_pred_target"] = predict_chi_from_coefficients(
        coefficients=coeff_matrix,
        temperature=merged_df["temperature"].to_numpy(dtype=float),
        phi=merged_df["phi"].to_numpy(dtype=float),
    )
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(cache_csv, index=False)
    return cache_csv


def _build_target_chi_parity_panel(
    target_csv: Optional[Path],
    out_png: Path,
) -> Optional[Path]:
    df = _safe_read_csv(target_csv)
    if df.empty or not {"target_chi", "chi_pred_target"}.issubset(df.columns):
        return None

    plot_df = df[["target_chi", "chi_pred_target"]].copy()
    plot_df["target_chi"] = pd.to_numeric(plot_df["target_chi"], errors="coerce")
    plot_df["chi_pred_target"] = pd.to_numeric(plot_df["chi_pred_target"], errors="coerce")
    plot_df = plot_df.dropna(subset=["target_chi", "chi_pred_target"])
    if plot_df.empty:
        return None

    x = plot_df["target_chi"].to_numpy(dtype=float)
    y = plot_df["chi_pred_target"].to_numpy(dtype=float)
    lo = float(np.nanmin(np.concatenate([x, y])))
    hi = float(np.nanmax(np.concatenate([x, y])))
    pad = max(0.02, 0.05 * (hi - lo if hi > lo else 1.0))
    mae = float(np.mean(np.abs(y - x)))

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(x, y, color=NATURE_BLUE, alpha=0.80, s=34)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="#6B7280", linewidth=1.4)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Target χ", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("Predicted χ at target", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=max(10, PAPER_FONT_SIZE - 2))
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.text(
        0.98,
        0.04,
        f"MAE = {mae:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=max(10, PAPER_FONT_SIZE - 2),
        color="#111827",
    )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _build_target_chi_by_rank_panel(
    target_csv: Optional[Path],
    out_png: Path,
) -> Optional[Path]:
    df = _safe_read_csv(target_csv)
    if df.empty or not {"chi_pred_target", "target_chi"}.issubset(df.columns):
        return None

    plot_df = df.copy()
    if "target_rank" not in plot_df.columns:
        plot_df["target_rank"] = np.arange(1, len(plot_df) + 1, dtype=int)
    plot_df["target_rank"] = pd.to_numeric(plot_df["target_rank"], errors="coerce")
    plot_df["chi_pred_target"] = pd.to_numeric(plot_df["chi_pred_target"], errors="coerce")
    plot_df["target_chi"] = pd.to_numeric(plot_df["target_chi"], errors="coerce")
    plot_df = plot_df.dropna(subset=["target_rank", "chi_pred_target", "target_chi"]).sort_values("target_rank")
    if plot_df.empty:
        return None

    y = plot_df["chi_pred_target"].to_numpy(dtype=float)
    target_y = plot_df["target_chi"].to_numpy(dtype=float)
    y_min = float(np.nanmin(np.concatenate([y, target_y])))
    y_max = float(np.nanmax(np.concatenate([y, target_y])))
    pad = max(0.025, 0.10 * max(y_max - y_min, 0.08))
    bins = min(20, max(8, int(round(np.sqrt(len(y))))))
    x_left = y_min - pad
    x_right = y_max + pad
    mean_y = float(np.mean(y))

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.hist(
        y,
        color=NATURE_BLUE,
        bins=np.linspace(x_left, x_right, bins + 1),
        edgecolor="white",
        linewidth=1.0,
        alpha=0.88,
        zorder=2,
        label="Predicted χ",
    )
    ax.axvline(mean_y, color=NATURE_BLUE, linewidth=2.0, alpha=0.95, zorder=3)
    if np.allclose(target_y, target_y[0]):
        ax.axvline(float(target_y[0]), color=NATURE_RED, linestyle="--", linewidth=2.0, label="Target χ", zorder=3)
    else:
        for idx, value in enumerate(np.unique(np.round(target_y, 6))):
            ax.axvline(
                float(value),
                color=NATURE_RED,
                linestyle="--",
                linewidth=1.8,
                label="Target χ" if idx == 0 else None,
                zorder=3,
            )

    ax.set_xlabel("χ at target condition", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("Count", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=PAPER_FONT_SIZE)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)
    ax.set_xlim(x_left, x_right)
    ax.legend(loc="upper right", fontsize=PAPER_FONT_SIZE, frameon=True)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _build_target_probability_by_rank_panel(
    target_csv: Optional[Path],
    out_png: Path,
) -> Optional[Path]:
    df = _safe_read_csv(target_csv)
    if df.empty or "class_prob" not in df.columns:
        return None

    plot_df = df.copy()
    if "target_rank" not in plot_df.columns:
        plot_df["target_rank"] = np.arange(1, len(plot_df) + 1, dtype=int)
    plot_df["target_rank"] = pd.to_numeric(plot_df["target_rank"], errors="coerce")
    plot_df["class_prob"] = pd.to_numeric(plot_df["class_prob"], errors="coerce")
    plot_df = plot_df.dropna(subset=["target_rank", "class_prob"]).sort_values("target_rank")
    if plot_df.empty:
        return None

    y = plot_df["class_prob"].to_numpy(dtype=float)
    prob_min = float(np.nanmin(y))
    prob_max = float(np.nanmax(y))
    span = max(prob_max - prob_min, 0.02)
    x_left = max(0.0, prob_min - 0.12 * span)
    x_right = min(1.02, prob_max + 0.12 * span)
    bins = min(20, max(8, int(round(np.sqrt(len(y))))))
    mean_prob = float(np.mean(y))

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.hist(
        y,
        color=NATURE_GREEN,
        bins=np.linspace(x_left, x_right, bins + 1),
        edgecolor="white",
        linewidth=1.0,
        alpha=0.88,
        zorder=2,
    )
    ax.axvline(mean_prob, color=NATURE_GREEN, linewidth=2.0, alpha=0.95, zorder=3)
    if x_left <= 0.5 <= x_right:
        ax.axvline(0.5, color=NATURE_GRAY, linestyle="--", linewidth=1.8, zorder=3)
    else:
        ax.text(
            0.03,
            0.06,
            "Threshold = 0.50",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=PAPER_FONT_SIZE,
            color="#111827",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.92},
        )
    ax.set_xlabel("Water-miscible probability", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("Count", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=PAPER_FONT_SIZE)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)
    ax.set_xlim(x_left, x_right)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _build_target_confidence_by_rank_panel(
    target_csv: Optional[Path],
    out_png: Path,
) -> Optional[Path]:
    df = _safe_read_csv(target_csv)
    if df.empty:
        return None

    plot_df = df.copy()
    if "target_rank" not in plot_df.columns:
        plot_df["target_rank"] = np.arange(1, len(plot_df) + 1, dtype=int)
    plot_df["target_rank"] = pd.to_numeric(plot_df["target_rank"], errors="coerce")
    prob_cols = [col for col in ["class_prob", "class_prob_lcb"] if col in plot_df.columns]
    if not prob_cols:
        return None
    for col in prob_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=["target_rank"] + prob_cols).sort_values("target_rank")
    if plot_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    color_map = {"class_prob": NATURE_BLUE, "class_prob_lcb": NATURE_GREEN}
    label_map = {"class_prob": "Class probability", "class_prob_lcb": "Conservative class probability"}
    for col in prob_cols:
        ax.plot(
            plot_df["target_rank"],
            plot_df[col],
            linewidth=2.0,
            color=color_map.get(col, NATURE_BLUE),
            label=label_map.get(col, col),
        )
    ax.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1.4)
    ax.set_xlabel("Selected target rank", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("Solubility confidence", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=max(10, PAPER_FONT_SIZE - 2))
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.legend(loc="lower right", fontsize=max(10, PAPER_FONT_SIZE - 4), frameon=True)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _build_target_sa_by_rank_panel(
    target_csv: Optional[Path],
    out_png: Path,
) -> Optional[Path]:
    df = _safe_read_csv(target_csv)
    if df.empty or not {"sa_score"}.issubset(df.columns):
        return None

    plot_df = df.copy()
    if "target_rank" not in plot_df.columns:
        plot_df["target_rank"] = np.arange(1, len(plot_df) + 1, dtype=int)
    plot_df["target_rank"] = pd.to_numeric(plot_df["target_rank"], errors="coerce")
    plot_df["sa_score"] = pd.to_numeric(plot_df["sa_score"], errors="coerce")
    plot_df = plot_df.dropna(subset=["target_rank", "sa_score"]).sort_values("target_rank")
    if plot_df.empty:
        return None

    mean_sa = float(plot_df["sa_score"].mean())
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.plot(
        plot_df["target_rank"],
        plot_df["sa_score"],
        color=NATURE_PURPLE,
        linewidth=1.8,
        marker="o",
        markersize=4.8,
    )
    ax.axhline(mean_sa, color=NATURE_GRAY, linestyle="--", linewidth=1.2)
    ax.text(
        0.98,
        0.96,
        f"Mean SA = {mean_sa:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=max(10, PAPER_FONT_SIZE - 3),
        color="#111827",
    )
    ax.set_xlabel("Selected target rank", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("SA score", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=max(10, PAPER_FONT_SIZE - 2))
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _format_numeric_range(values: pd.Series, decimals: int = 2, suffix: str = "") -> str:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return "n/a"
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if np.isclose(lo, hi):
        return f"{lo:.{decimals}f}{suffix}"
    return f"{lo:.{decimals}f}-{hi:.{decimals}f}{suffix}"


def _build_inverse_target_summary_panel(
    inverse_targets_csv: Optional[Path],
    out_png: Path,
) -> Optional[Path]:
    df = _safe_read_csv(inverse_targets_csv)
    if df.empty:
        return None

    target_classes = sorted({str(v).strip() for v in df.get("target_polymer_class", pd.Series(dtype=object)).tolist() if str(v).strip()})
    property_rules = sorted({str(v).strip() for v in df.get("property_rule", pd.Series(dtype=object)).tolist() if str(v).strip()})
    class_text = target_classes[0] if len(target_classes) == 1 else f"{len(target_classes)} classes"
    rule_text = property_rules[0] if len(property_rules) == 1 else f"{len(property_rules)} rules"
    cards = [
        ("Inverse targets", f"{len(df):,}", NATURE_BLUE),
        ("Target class", class_text, NATURE_GREEN),
        ("Temperature", _format_numeric_range(df.get("temperature", pd.Series(dtype=float)), decimals=2, suffix=" K"), NATURE_ORANGE),
        ("Volume fraction φ", _format_numeric_range(df.get("phi", pd.Series(dtype=float)), decimals=2), NATURE_RED),
        ("Target χ", _format_numeric_range(df.get("target_chi", pd.Series(dtype=float)), decimals=3), NATURE_PURPLE),
        ("Property rule", rule_text, NATURE_GRAY),
    ]

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.set_axis_off()
    card_w = 0.41
    card_h = 0.20
    x_positions = [0.07, 0.52]
    y_positions = [0.73, 0.47, 0.21]

    for idx, (label, value, color) in enumerate(cards):
        row = idx // 2
        col = idx % 2
        x0 = x_positions[col]
        y0 = y_positions[row]
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                card_w,
                card_h,
                boxstyle="round,pad=0.012,rounding_size=0.03",
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
                alpha=0.96,
            )
        )
        ax.text(
            x0 + 0.035,
            y0 + card_h - 0.050,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=max(10, PAPER_FONT_SIZE - 4),
            color="white",
            fontweight="semibold",
        )
        ax.text(
            x0 + card_w / 2.0,
            y0 + 0.085,
            value,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=max(13, PAPER_FONT_SIZE + 1),
            color="white",
            fontweight="bold",
            wrap=True,
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=PAPER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _build_selected_target_summary_panel(
    target_csv: Optional[Path],
    out_png: Path,
    *,
    inverse_targets_csv: Optional[Path] = None,
    selection_summary_csv: Optional[Path] = None,
    attempt_count_override: Optional[int] = None,
) -> Optional[Path]:
    df = _safe_read_csv(target_csv)
    if df.empty:
        return None

    inverse_df = _safe_read_csv(inverse_targets_csv)
    summary_row = _safe_first_row(selection_summary_csv or _selection_summary_csv_from_target_csv(target_csv))
    selected_count = int(len(df))

    attempt_text = "n/a"
    if "sampling_attempt" in df.columns:
        attempts = pd.to_numeric(df["sampling_attempt"], errors="coerce").dropna().astype(int)
        if not attempts.empty:
            lo = int(attempts.min())
            hi = int(attempts.max())
            attempt_text = str(lo) if lo == hi else f"{lo}-{hi}"
    if attempt_text == "n/a" and attempt_count_override is not None and int(attempt_count_override) > 0:
        attempt_text = f"1-{int(attempt_count_override)}" if int(attempt_count_override) > 1 else "1"

    novel_col = "is_novel_vs_train" if "is_novel_vs_train" in df.columns else ("is_novel" if "is_novel" in df.columns else None)
    novel_rate = np.nan
    if novel_col is not None:
        novel_vals = pd.to_numeric(df[novel_col], errors="coerce").dropna()
        if not novel_vals.empty:
            novel_rate = float((novel_vals >= 0.5).mean())
    if not np.isfinite(novel_rate):
        novel_rate = float(_pick(summary_row, ["final_novelty"], default=np.nan))

    mean_prob = np.nan
    if "class_prob" in df.columns:
        prob_vals = pd.to_numeric(df["class_prob"], errors="coerce").dropna()
        if not prob_vals.empty:
            mean_prob = float(prob_vals.mean())

    diversity = float(_pick(summary_row, ["final_diversity"], default=np.nan))
    if not np.isfinite(diversity):
        family_col = "polymer_family" if "polymer_family" in df.columns else None
        if family_col is not None:
            families = {str(v).strip() for v in df[family_col].tolist() if str(v).strip()}
            if families:
                diversity = float(len(families)) / max(selected_count, 1)

    selection_success = float(
        _pick(summary_row, ["selection_success_rate"], default=np.nan)
    )
    if not np.isfinite(selection_success):
        selection_success = float(
            _pick(summary_row, ["target_count_selected"], default=np.nan)
        ) / max(float(_pick(summary_row, ["total_generated"], default=np.nan)), 1.0)
        if not np.isfinite(selection_success):
            selection_success = np.nan

    mean_sa = np.nan
    if "sa_score" in df.columns:
        sa_vals = pd.to_numeric(df["sa_score"], errors="coerce").dropna()
        if not sa_vals.empty:
            mean_sa = float(sa_vals.mean())
    if not np.isfinite(mean_sa):
        mean_sa = float(_pick(summary_row, ["final_mean_sa"], default=np.nan))

    target_classes = sorted(
        {
            str(v).strip()
            for v in inverse_df.get("target_polymer_class", pd.Series(dtype=object)).tolist()
            if str(v).strip()
        }
    )
    target_class_text = ""
    if target_classes:
        target_class_text = target_classes[0] if len(target_classes) == 1 else f"{len(target_classes)} classes"

    metric_specs = [
        ("Diversity", diversity, NATURE_GREEN, "float"),
        ("Novelty", novel_rate, NATURE_ORANGE, "pct"),
        ("Mean probability", mean_prob, NATURE_PURPLE, "float"),
        ("Selection success", selection_success, NATURE_BLUE, "pct"),
    ]
    metric_specs = [spec for spec in metric_specs if np.isfinite(spec[1])]
    if not metric_specs:
        return None

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ys = np.arange(len(metric_specs))[::-1]
    for y, (label, value, color, value_mode) in zip(ys, metric_specs):
        ax.hlines(y, 0.0, value, color="#D1D5DB", linewidth=4.0, zorder=1)
        ax.scatter([value], [y], s=120, color=color, edgecolor="white", linewidth=1.1, zorder=3)
        value_text = f"{100.0 * value:.1f}%" if value_mode == "pct" else f"{value:.2f}"
        ax.text(
            min(value + 0.03, 1.03),
            y,
            value_text,
            ha="left",
            va="center",
            fontsize=PAPER_FONT_SIZE,
            color="#111827",
        )

    ax.set_yticks(ys, labels=[label for label, _, _, _ in metric_specs])
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Summary metric value", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=PAPER_FONT_SIZE)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.8, alpha=0.35)
    info_lines = [
        f"Selected polymers: {selected_count:,}",
        f"Sampling attempts: {attempt_text}",
    ]
    if np.isfinite(mean_sa):
        info_lines.append(f"Mean SA: {mean_sa:.2f}")
    if target_class_text:
        info_lines.append(f"Target class: {target_class_text}")
    ax.text(
        0.03,
        0.97,
        "\n".join(info_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PAPER_FONT_SIZE,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.96},
    )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _build_step3_metric_heatmap(
    paths: Dict[str, Optional[Path]],
    metadata_dir: Path,
    value_col: str,
    out_name: str,
    colorbar_label: str,
    cmap_name: str,
    annotate_fmt: str = "{:.2f}",
) -> Optional[Path]:
    step3_dir = paths.get("step3_dir")
    if step3_dir is None:
        return None

    best_csv = step3_dir / "metrics" / "chi_target_best_by_condition.csv"
    df = _safe_read_csv(best_csv)
    if df.empty or not {"temperature", "phi", value_col}.issubset(df.columns):
        return None

    plot_df = (
        df[["temperature", "phi", value_col]]
        .dropna(subset=["temperature", "phi", value_col])
        .copy()
    )
    if plot_df.empty:
        return None

    plot_df["temperature"] = plot_df["temperature"].astype(float)
    plot_df["phi"] = plot_df["phi"].astype(float)
    plot_df[value_col] = plot_df[value_col].astype(float)
    pivot = plot_df.pivot(index="temperature", columns="phi", values=value_col).sort_index().sort_index(axis=1)
    if pivot.empty:
        return None

    out_png = metadata_dir / out_name
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    cmap = plt.colormaps[cmap_name] if hasattr(plt, "colormaps") else plt.get_cmap(cmap_name)
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap=cmap, origin="upper")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{float(v):.1f}" for v in pivot.columns], fontsize=PAPER_FONT_SIZE)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{float(v):.2f}" for v in pivot.index], fontsize=PAPER_FONT_SIZE)
    ax.set_xlabel("φ", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("Temperature (K)", fontsize=PAPER_FONT_SIZE)

    values = pivot.to_numpy(dtype=float)
    finite_vals = values[np.isfinite(values)]
    text_threshold = float(np.nanmedian(finite_vals)) if finite_vals.size else np.nan
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if not np.isfinite(val):
                continue
            text_color = "white" if np.isfinite(text_threshold) and val >= text_threshold else "#111827"
            ax.text(j, i, annotate_fmt.format(float(val)), ha="center", va="center", fontsize=12, color=text_color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(colorbar_label, fontsize=PAPER_FONT_SIZE)
    cbar.ax.tick_params(labelsize=max(10, PAPER_FONT_SIZE - 2))

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _build_step3_global_threshold_curve(
    paths: Dict[str, Optional[Path]],
    metadata_dir: Path,
) -> Optional[Path]:
    """Build balanced-accuracy vs χ-threshold curve (non-heatmap) for Figure 3(b)."""
    step3_dir = paths.get("step3_dir")
    if step3_dir is None:
        return None

    scan_csv = step3_dir / "metrics" / "chi_target_global_scan.csv"
    best_csv = step3_dir / "metrics" / "chi_target_global_best.csv"
    scan_df = _safe_read_csv(scan_csv)
    if scan_df.empty or not {"threshold", "balanced_accuracy"}.issubset(scan_df.columns):
        return None

    scan_df = (
        scan_df[["threshold", "balanced_accuracy"]]
        .dropna(subset=["threshold", "balanced_accuracy"])
        .sort_values("threshold")
    )
    if scan_df.empty:
        return None

    best_row = _safe_first_row(best_csv)
    chi_star = _pick(best_row, ["chi_target", "threshold"], default=np.nan)
    bal_acc = _pick(best_row, ["balanced_accuracy"], default=np.nan)
    ci_low = _pick(best_row, ["chi_target_boot_q025"], default=np.nan)
    ci_high = _pick(best_row, ["chi_target_boot_q975"], default=np.nan)

    out_png = metadata_dir / "derived_step3_global_threshold_curve.png"
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot(
        scan_df["threshold"].to_numpy(dtype=float),
        scan_df["balanced_accuracy"].to_numpy(dtype=float),
        color=NATURE_BLUE,
        linewidth=2.2,
    )
    if np.isfinite(ci_low) and np.isfinite(ci_high) and float(ci_high) >= float(ci_low):
        ax.axvspan(float(ci_low), float(ci_high), color=NATURE_LIGHT_BLUE, alpha=0.30, linewidth=0)
    if np.isfinite(chi_star):
        ax.axvline(float(chi_star), color=NATURE_RED, linestyle="--", linewidth=2.0)
    if np.isfinite(chi_star) and np.isfinite(bal_acc):
        ax.scatter([float(chi_star)], [float(bal_acc)], color=NATURE_RED, s=45, zorder=5)
        x_min = float(scan_df["threshold"].min())
        x_max = float(scan_df["threshold"].max())
        y_min = float(scan_df["balanced_accuracy"].min())
        y_max = float(scan_df["balanced_accuracy"].max())
        x_span = max(1e-12, x_max - x_min)
        y_span = max(1e-12, y_max - y_min)
        near_right = float(chi_star) > (x_min + 0.82 * x_span)
        near_top = float(bal_acc) > (y_min + 0.90 * y_span)
        text_x = float(chi_star) - 0.015 * x_span if near_right else float(chi_star) + 0.015 * x_span
        text_y = float(bal_acc) - 0.015 * y_span if near_top else float(bal_acc) + 0.015 * y_span
        ax.text(
            text_x,
            text_y,
            f"  χ*={float(chi_star):.3f}\n  BA={float(bal_acc):.3f}",
            fontsize=PAPER_FONT_SIZE,
            ha="right" if near_right else "left",
            va="top" if near_top else "bottom",
            color="#111827",
        )

    ax.set_xlabel("Global χ threshold", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("Balanced accuracy", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=PAPER_FONT_SIZE)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _build_step3_threshold_regions_panel(
    paths: Dict[str, Optional[Path]],
    metadata_dir: Path,
) -> Optional[Path]:
    """Build Figure 1(d) by replacing legend text only, keeping the original legend style."""
    step3_dir = paths.get("step3_dir")
    if step3_dir is None:
        return None

    src_png = step3_dir / "figures" / "chi_distribution_global_threshold.png"
    best_csv = step3_dir / "metrics" / "chi_target_global_best.csv"
    if not src_png.exists():
        return None

    best_row = _safe_first_row(best_csv)
    chi_star = _pick(best_row, ["chi_target", "threshold"], default=np.nan)

    out_png = metadata_dir / "derived_step3_threshold_regions.png"
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        _copy_if_exists(src_png, out_png)
        return out_png if out_png.exists() else None

    with Image.open(src_png).convert("RGBA") as img:
        w, h = img.size
        draw = ImageDraw.Draw(img)

        # Use normalized coordinates so behavior is stable across DPI changes.
        legend_cover = (0.63, 0.08, 0.965, 0.26)
        x0, y0 = int(legend_cover[0] * w), int(legend_cover[1] * h)
        x1, y1 = int(legend_cover[2] * w), int(legend_cover[3] * h)

        # If upstream layout moved, avoid adding a misaligned white box.
        arr = np.array(img)
        roi = arr[y0:y1, x0:x1, :3]
        non_white_pixels = int(np.count_nonzero(np.any(roi < 245, axis=2))) if roi.size else 0
        roi_pixels = int(roi.shape[0] * roi.shape[1]) if roi.size else 0
        min_non_white_pixels = max(40, int(round(roi_pixels * 0.00002)))
        if non_white_pixels < min_non_white_pixels:
            print(
                "[WARN] Step3 legend rewrite skipped: expected legend region appears blank "
                f"(non_white={non_white_pixels}, required>={min_non_white_pixels})."
            )
            out_png.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_png, dpi=(PAPER_DPI, PAPER_DPI))
            return out_png

        # Cover only text area inside original legend box; keep legend frame and line handles.
        draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255, 255))

        # Match paper style exactly: 16 pt at 600 DPI.
        font_px = max(12, int(round(PAPER_FONT_SIZE * PAPER_DPI / 72.0)))
        font = None
        font_path = _resolve_truetype_font_path()
        if font_path is not None:
            try:
                font = ImageFont.truetype(font_path, font_px)
            except Exception:
                font = None
        if font is None:
            print(
                "[WARN] Step3 legend rewrite could not load a TrueType font; "
                "using original figure without relabeling."
            )
            out_png.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_png, dpi=(PAPER_DPI, PAPER_DPI))
            return out_png

        text_color = (31, 41, 55, 255)
        tx = int(0.645 * w)
        y1_txt = int(0.105 * h)
        y2_txt = int(0.163 * h)
        y3_txt = int(0.221 * h)

        draw.text((tx, y1_txt), "Water-miscible", fill=text_color, font=font)
        draw.text((tx, y2_txt), "Water-immiscible", fill=text_color, font=font)
        thr_txt = f"Global χ*={float(chi_star):.3f}" if np.isfinite(chi_star) else "Global χ*"
        draw.text((tx, y3_txt), thr_txt, fill=text_color, font=font)

        out_png.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_png, dpi=(PAPER_DPI, PAPER_DPI))
    return out_png


def _build_step3_condition_profiles_panel(
    paths: Dict[str, Optional[Path]],
    metadata_dir: Path,
) -> Optional[Path]:
    """Build Figure 3(d): condition-wise χ_target profiles with bootstrap CI (non-heatmap)."""
    step3_dir = paths.get("step3_dir")
    if step3_dir is None:
        return None

    best_csv = step3_dir / "metrics" / "chi_target_best_by_condition.csv"
    df = _safe_read_csv(best_csv)
    if df.empty or not {"temperature", "phi", "chi_target"}.issubset(df.columns):
        return None

    q025_col = "chi_target_boot_q025" if "chi_target_boot_q025" in df.columns else None
    q975_col = "chi_target_boot_q975" if "chi_target_boot_q975" in df.columns else None

    out_png = metadata_dir / "derived_step3_condition_profiles.png"
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    plot_df = df.copy()
    plot_df["temperature"] = plot_df["temperature"].astype(float)
    plot_df["phi"] = plot_df["phi"].astype(float)
    plot_df["chi_target"] = plot_df["chi_target"].astype(float)

    temps = sorted(plot_df["temperature"].dropna().unique().tolist())
    cmap = plt.colormaps["viridis"] if hasattr(plt, "colormaps") else plt.get_cmap("viridis")
    for idx, t in enumerate(temps):
        sub = plot_df[plot_df["temperature"] == t].sort_values("phi")
        if sub.empty:
            continue
        x = sub["phi"].to_numpy(dtype=float)
        y = sub["chi_target"].to_numpy(dtype=float)
        color = cmap(idx / max(1, len(temps) - 1))
        ax.plot(x, y, marker="o", markersize=5.0, linewidth=2.0, color=color, label=f"{t:.2f} K")
        if q025_col is not None and q975_col is not None:
            lo = sub[q025_col].astype(float).to_numpy(dtype=float)
            hi = sub[q975_col].astype(float).to_numpy(dtype=float)
            yerr = np.vstack([np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)])
            ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.2, capsize=3, alpha=0.55)

    ax.set_xlabel("φ", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("χ_target", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=PAPER_FONT_SIZE)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.legend(title="Temperature", fontsize=PAPER_FONT_SIZE, title_fontsize=PAPER_FONT_SIZE, loc="best", frameon=True)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _build_step3_temperature_trends_panel(
    paths: Dict[str, Optional[Path]],
    metadata_dir: Path,
) -> Optional[Path]:
    step3_dir = paths.get("step3_dir")
    if step3_dir is None:
        return None

    best_csv = step3_dir / "metrics" / "chi_target_best_by_condition.csv"
    df = _safe_read_csv(best_csv)
    if df.empty or not {"temperature", "phi", "chi_target"}.issubset(df.columns):
        return None

    plot_df = df.copy()
    plot_df["temperature"] = plot_df["temperature"].astype(float)
    plot_df["phi"] = plot_df["phi"].astype(float)
    plot_df["chi_target"] = plot_df["chi_target"].astype(float)

    q025_col = "chi_target_boot_q025" if "chi_target_boot_q025" in plot_df.columns else None
    q975_col = "chi_target_boot_q975" if "chi_target_boot_q975" in plot_df.columns else None
    phis = sorted(plot_df["phi"].dropna().unique().tolist())
    if not phis:
        return None

    out_png = metadata_dir / "derived_step3_temperature_trends.png"
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    cmap = plt.colormaps["plasma"] if hasattr(plt, "colormaps") else plt.get_cmap("plasma")
    for idx, phi in enumerate(phis):
        sub = plot_df[plot_df["phi"] == phi].sort_values("temperature")
        if sub.empty:
            continue
        x = sub["temperature"].to_numpy(dtype=float)
        y = sub["chi_target"].to_numpy(dtype=float)
        color = cmap(idx / max(1, len(phis) - 1))
        ax.plot(x, y, marker="o", markersize=5.0, linewidth=2.0, color=color, label=f"φ={phi:.1f}")
        if q025_col is not None and q975_col is not None:
            lo = sub[q025_col].astype(float).to_numpy(dtype=float)
            hi = sub[q975_col].astype(float).to_numpy(dtype=float)
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)

    ax.set_xlabel("Temperature (K)", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("χ_target", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=PAPER_FONT_SIZE)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.legend(fontsize=max(10, PAPER_FONT_SIZE - 2), loc="best", frameon=True)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _build_step4_regression_parity_panel(
    paths: Dict[str, Optional[Path]],
    metadata_dir: Path,
) -> Optional[Path]:
    step4_reg_dir = paths.get("step4_reg_dir")
    pred_csv = step4_reg_dir / "metrics" / "chi_predictions_test.csv" if step4_reg_dir is not None else None
    df = _safe_read_csv(pred_csv)
    required = {"chi", "chi_pred", "water_miscible"}
    if df.empty or not required.issubset(df.columns):
        return None

    plot_df = df[list(required)].copy()
    for col in ["chi", "chi_pred", "water_miscible"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=["chi", "chi_pred", "water_miscible"])
    if plot_df.empty:
        return None

    palette = WATER_CLASS_PALETTE
    out_png = metadata_dir / "derived_step4_regression_parity_test.png"
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for class_value, label in [(0, "water-immiscible"), (1, "water-miscible")]:
        sub = plot_df[plot_df["water_miscible"].astype(int) == class_value]
        if sub.empty:
            continue
        ax.scatter(
            sub["chi"].to_numpy(dtype=float),
            sub["chi_pred"].to_numpy(dtype=float),
            color=palette[class_value],
            alpha=0.75,
            s=28,
            linewidths=0.4,
            edgecolors="white",
            label=label,
        )

    x = plot_df["chi"].to_numpy(dtype=float)
    y = plot_df["chi_pred"].to_numpy(dtype=float)
    lo = float(np.nanmin(np.concatenate([x, y])))
    hi = float(np.nanmax(np.concatenate([x, y])))
    pad = max(0.02, 0.04 * max(hi - lo, 1.0))
    lo_plot = lo - pad
    hi_plot = hi + pad
    ax.plot([lo_plot, hi_plot], [lo_plot, hi_plot], linestyle="--", color="black", linewidth=1.2)
    ax.set_xlim(lo_plot, hi_plot)
    ax.set_ylim(lo_plot, hi_plot)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True χ", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("Predicted χ", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=max(10, PAPER_FONT_SIZE - 2))
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)

    reg = _regression_summary_stats(x, y)
    ax.text(
        0.03,
        0.97,
        f"MAE={reg['mae']:.3f}\nRMSE={reg['rmse']:.3f}\nR2={reg['r2']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=max(10, PAPER_FONT_SIZE - 2),
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.92},
    )
    _boxed_legend(ax, loc="upper right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _build_step4_regression_residual_panel(
    paths: Dict[str, Optional[Path]],
    metadata_dir: Path,
) -> Optional[Path]:
    step4_reg_dir = paths.get("step4_reg_dir")
    pred_csv = step4_reg_dir / "metrics" / "chi_predictions_all.csv" if step4_reg_dir is not None else None
    df = _safe_read_csv(pred_csv)
    if df.empty or not {"chi_error", "split"}.issubset(df.columns):
        return None

    plot_df = df[["chi_error", "split"]].copy()
    plot_df["chi_error"] = pd.to_numeric(plot_df["chi_error"], errors="coerce")
    plot_df["split"] = plot_df["split"].astype(str).str.lower()
    plot_df = plot_df.dropna(subset=["chi_error"])
    if plot_df.empty:
        return None

    out_png = metadata_dir / "derived_step4_regression_residual_distribution.png"
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    plotted_any = False
    for split_name, color in [("train", NATURE_BLUE), ("val", NATURE_ORANGE), ("test", NATURE_GREEN)]:
        sub = plot_df.loc[plot_df["split"] == split_name, "chi_error"].to_numpy(dtype=float)
        sub = sub[np.isfinite(sub)]
        if sub.size == 0:
            continue
        if sub.size >= 2 and not np.isclose(np.std(sub), 0.0):
            sns.kdeplot(x=sub, ax=ax, color=color, linewidth=2.0, fill=False, label=split_name)
        else:
            ax.hist(sub, bins=min(12, max(4, sub.size)), density=True, histtype="step", linewidth=2.0, color=color, label=split_name)
        plotted_any = True

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("χ prediction error", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("Density", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=max(10, PAPER_FONT_SIZE - 2))
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    if plotted_any:
        _boxed_legend(ax, loc="upper right")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _build_step4_class_confusion_panel(
    paths: Dict[str, Optional[Path]],
    metadata_dir: Path,
) -> Optional[Path]:
    step4_cls_dir = paths.get("step4_cls_dir")
    pred_csv = step4_cls_dir / "metrics" / "class_predictions_test.csv" if step4_cls_dir is not None else None
    df = _safe_read_csv(pred_csv)
    required = {"water_miscible", "class_pred"}
    if df.empty or not required.issubset(df.columns):
        return None

    plot_df = df[list(required)].copy()
    plot_df["water_miscible"] = pd.to_numeric(plot_df["water_miscible"], errors="coerce")
    plot_df["class_pred"] = pd.to_numeric(plot_df["class_pred"], errors="coerce")
    plot_df = plot_df.dropna(subset=["water_miscible", "class_pred"])
    if plot_df.empty:
        return None

    y_true = plot_df["water_miscible"].astype(int).to_numpy()
    y_pred = plot_df["class_pred"].astype(int).to_numpy()
    cm = np.zeros((2, 2), dtype=int)
    for true_value, pred_value in zip(y_true, y_pred):
        if true_value in (0, 1) and pred_value in (0, 1):
            cm[int(true_value), int(pred_value)] += 1

    out_png = metadata_dir / "derived_step4_class_confusion_test.png"
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    sns.heatmap(
        cm,
        annot=False,
        cbar=False,
        cmap="Blues",
        square=True,
        xticklabels=["water-\nimmiscible", "water-\nmiscible"],
        yticklabels=["water-\nimmiscible", "water-\nmiscible"],
        ax=ax,
    )
    vmax = float(np.max(cm)) if np.size(cm) else 0.0
    threshold = 0.55 * vmax if vmax > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            text_color = "white" if float(value) >= threshold else "#1f1f1f"
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{value:d}",
                ha="center",
                va="center",
                fontsize=PAPER_FONT_SIZE + 2,
                color=text_color,
            )
    ax.set_xlabel("Predicted class", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("True class", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=max(10, PAPER_FONT_SIZE - 2))
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _build_step4_class_prob_distribution_panel(
    paths: Dict[str, Optional[Path]],
    metadata_dir: Path,
) -> Optional[Path]:
    step4_cls_dir = paths.get("step4_cls_dir")
    pred_csv = step4_cls_dir / "metrics" / "class_predictions_test.csv" if step4_cls_dir is not None else None
    df = _safe_read_csv(pred_csv)
    required = {"water_miscible", "class_prob"}
    if df.empty or not required.issubset(df.columns):
        return None

    plot_df = df[list(required)].copy()
    plot_df["water_miscible"] = pd.to_numeric(plot_df["water_miscible"], errors="coerce")
    plot_df["class_prob"] = pd.to_numeric(plot_df["class_prob"], errors="coerce")
    plot_df = plot_df.dropna(subset=["water_miscible", "class_prob"])
    if plot_df.empty:
        return None

    palette = WATER_CLASS_PALETTE
    out_png = metadata_dir / "derived_step4_class_prob_distribution_test.png"
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for class_value, label in [(0, "water-immiscible"), (1, "water-miscible")]:
        sub = plot_df.loc[plot_df["water_miscible"].astype(int) == class_value, "class_prob"].to_numpy(dtype=float)
        sub = sub[np.isfinite(sub)]
        if sub.size == 0:
            continue
        if sub.size >= 2 and not np.isclose(np.std(sub), 0.0):
            sns.kdeplot(
                x=sub,
                ax=ax,
                color=palette[class_value],
                linewidth=2.0,
                fill=True,
                alpha=0.22,
                label=label,
            )
        else:
            ax.hist(
                sub,
                bins=min(12, max(4, sub.size)),
                density=True,
                histtype="stepfilled",
                alpha=0.22,
                linewidth=1.5,
                color=palette[class_value],
                label=label,
            )

    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Predicted water-miscible probability", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("Density", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=max(10, PAPER_FONT_SIZE - 2))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    _boxed_legend(ax, loc="upper center", ncol=2)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _build_step6_target_class_coverage_panel(
    paths: Dict[str, Optional[Path]],
    metadata_dir: Path,
) -> Optional[Path]:
    summary_row = _safe_first_row(paths.get("step6_summary"))
    if not summary_row:
        return None

    raw_classes = _parse_literal_value(_pick(summary_row, ["target_polymer_classes"], default=[]))
    raw_hits = _parse_literal_value(_pick(summary_row, ["candidate_polymer_class_hits"], default={}))
    selected_count = _pick(summary_row, ["target_polymer_count_selected"], default=np.nan)

    classes = [str(x) for x in raw_classes] if isinstance(raw_classes, (list, tuple)) else []
    hits = raw_hits if isinstance(raw_hits, dict) else {}
    if not classes and hits:
        classes = [str(k) for k in hits.keys()]
    if not classes:
        return None

    values = [float(hits.get(cls, 0) or 0) for cls in classes]
    out_png = metadata_dir / "derived_step6_target_polymer_class_coverage.png"
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    bars = ax.bar(classes, values, color=NATURE_BLUE)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.03 * max(1.0, max(values) if values else 1.0),
            f"{int(round(val))}",
            ha="center",
            va="bottom",
            fontsize=max(10, PAPER_FONT_SIZE - 2),
        )

    if not _is_missing_value(selected_count):
        ax.text(
            0.98,
            0.96,
            f"Selected targets: {int(float(selected_count))}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=max(10, PAPER_FONT_SIZE - 2),
            color="#111827",
        )

    ax.set_xlabel("Target polymer class", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("Discovered candidates", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="x", labelrotation=35, labelsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=PAPER_FONT_SIZE)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.35)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    _pad_png_canvas(out_png)
    return out_png


def _resolve_input_artifacts(
    config: Dict,
    model_size: str,
    split_mode: str,
) -> Dict[str, Optional[Path]]:
    base_results = Path(config["paths"]["results_dir"])
    results_dir = Path(get_results_dir(model_size, config["paths"]["results_dir"], split_mode=split_mode))
    project_root = Path(__file__).resolve().parents[1]

    step0_dir = _first_existing([base_results / "step0_data_prep", results_dir / "step0_data_prep"])
    step1_dir = _first_existing([results_dir / "step1_backbone", base_results / "step1_backbone"])
    step2_dir = _first_existing([results_dir / "step2_sampling", base_results / "step2_sampling"])
    step3_dir = _first_existing(
        [
            results_dir / "step3_chi_target_learning" / split_mode,
            base_results / "step3_chi_target_learning" / split_mode,
        ]
    )
    step4_reg_dir = _first_existing(
        [
            results_dir / "step4_1_regression" / split_mode,
            base_results / "step4_1_regression" / split_mode,
            results_dir / "step4_chi_training" / split_mode / "step4_1_regression",
            base_results / "step4_chi_training" / split_mode / "step4_1_regression",
        ]
    )
    step4_cls_dir = _first_existing(
        [
            results_dir / "step4_2_classification" / split_mode,
            base_results / "step4_2_classification" / split_mode,
            results_dir / "step4_2_classification",
            base_results / "step4_2_classification",
            results_dir / "step4_chi_training" / split_mode / "step4_2_classification",
            base_results / "step4_chi_training" / split_mode / "step4_2_classification",
        ]
    )
    step4_pipe_metrics = _first_existing(
        [
            results_dir / f"step4_split_pipeline_{split_mode}" / "pipeline_metrics",
            base_results / f"step4_split_pipeline_{split_mode}" / "pipeline_metrics",
            results_dir / "step4_chi_training" / split_mode / "metrics",
            base_results / "step4_chi_training" / split_mode / "metrics",
        ]
    )
    step5_dir = _first_existing(
        [
            results_dir / "step5_water_soluble_inverse_design" / split_mode,
            base_results / "step5_water_soluble_inverse_design" / split_mode,
        ]
    )
    step6_dir = _first_existing(
        [
            results_dir / "step6_polymer_class_water_soluble_inverse_design" / split_mode,
            base_results / "step6_polymer_class_water_soluble_inverse_design" / split_mode,
        ]
    )
    step7_dir = _first_existing(
        [
            results_dir / "step7_chem_physics_analysis" / split_mode,
            base_results / "step7_chem_physics_analysis" / split_mode,
        ]
    )
    step4_compare_dir = _first_existing(
        [
            project_root / "traditional_step4" / f"results_4_4_comparation_{split_mode}",
            project_root / "traditional_step4" / "results_4_4_comparation_polymer",
            base_results.parent / "traditional_step4" / f"results_4_4_comparation_{split_mode}",
            base_results.parent / "traditional_step4" / "results_4_4_comparation_polymer",
        ]
    )

    out: Dict[str, Optional[Path]] = {
        "results_dir": results_dir,
        "base_results_dir": base_results,
        "step0_dir": step0_dir,
        "step1_dir": step1_dir,
        "step2_dir": step2_dir,
        "step3_dir": step3_dir,
        "step4_reg_dir": step4_reg_dir,
        "step4_cls_dir": step4_cls_dir,
        "step4_pipeline_metrics_dir": step4_pipe_metrics,
        "step5_dir": step5_dir,
        "step6_dir": step6_dir,
        "step7_dir": step7_dir,
        "step4_compare_dir": step4_compare_dir,
    }
    if out["step7_dir"] is not None:
        for blk in "abcdefghi":
            out[f"step7_block_{blk}_fig_dir"] = out["step7_dir"] / "figures" / f"block_{blk}"

    out.update(
        {
            "step0_summary": _first_existing(
                [
                    base_results / "step0_data_prep" / "metrics" / "step_summary.csv",
                    results_dir / "step0_data_prep" / "metrics" / "step_summary.csv",
                ]
            ),
            "step1_summary": _first_existing(
                [
                    results_dir / "step1_backbone" / "metrics" / "step_summary.csv",
                    base_results / "step1_backbone" / "metrics" / "step_summary.csv",
                ]
            ),
            "step2_summary": _first_existing(
                [
                    results_dir / "step2_sampling" / "metrics" / "step_summary.csv",
                    base_results / "step2_sampling" / "metrics" / "step_summary.csv",
                ]
            ),
            "step3_summary": _first_existing(
                [
                    results_dir / "step3_chi_target_learning" / split_mode / "metrics" / "step_summary.csv",
                    base_results / "step3_chi_target_learning" / split_mode / "metrics" / "step_summary.csv",
                ]
            ),
            "step4_summary": _first_existing(
                [
                    results_dir / f"step4_split_pipeline_{split_mode}" / "pipeline_metrics" / "step_summary.csv",
                    base_results / f"step4_split_pipeline_{split_mode}" / "pipeline_metrics" / "step_summary.csv",
                    step4_reg_dir / "pipeline_metrics" / "step_summary.csv" if step4_reg_dir is not None else None,
                    step4_cls_dir / "pipeline_metrics" / "step_summary.csv" if step4_cls_dir is not None else None,
                    results_dir / "step4_chi_training" / split_mode / "metrics" / "step_summary.csv",
                    base_results / "step4_chi_training" / split_mode / "metrics" / "step_summary.csv",
                ]
            ),
            "step4_reg_pipeline_summary": _first_existing(
                [
                    step4_reg_dir / "pipeline_metrics" / "step_summary.csv" if step4_reg_dir is not None else None,
                ]
            ),
            "step4_cls_pipeline_summary": _first_existing(
                [
                    step4_cls_dir / "pipeline_metrics" / "step_summary.csv" if step4_cls_dir is not None else None,
                ]
            ),
            "step5_summary": _first_existing(
                [
                    results_dir / "step5_water_soluble_inverse_design" / split_mode / "metrics" / "step_summary.csv",
                    base_results / "step5_water_soluble_inverse_design" / split_mode / "metrics" / "step_summary.csv",
                ]
            ),
            "step5_selected_target_candidate_ranked": _first_existing(
                [
                    results_dir / "step5_water_soluble_inverse_design" / split_mode / "metrics" / "selected_target_candidate_ranked.csv",
                    base_results / "step5_water_soluble_inverse_design" / split_mode / "metrics" / "selected_target_candidate_ranked.csv",
                ]
            ),
            "step5_sampling_process_summary": _first_existing(
                [
                    results_dir / "step5_water_soluble_inverse_design" / split_mode / "metrics" / "sampling_process_summary.csv",
                    base_results / "step5_water_soluble_inverse_design" / split_mode / "metrics" / "sampling_process_summary.csv",
                ]
            ),
            "step5_sampling_attempts": _first_existing(
                [
                    results_dir / "step5_water_soluble_inverse_design" / split_mode / "metrics" / "sampling_attempts.csv",
                    base_results / "step5_water_soluble_inverse_design" / split_mode / "metrics" / "sampling_attempts.csv",
                ]
            ),
            "step6_summary": _first_existing(
                [
                    results_dir / "step6_polymer_class_water_soluble_inverse_design" / split_mode / "metrics" / "step_summary.csv",
                    base_results / "step6_polymer_class_water_soluble_inverse_design" / split_mode / "metrics" / "step_summary.csv",
                ]
            ),
            "step6_selected_target_candidate_ranked": _first_existing(
                [
                    results_dir / "step6_polymer_class_water_soluble_inverse_design" / split_mode / "metrics" / "selected_target_candidate_ranked.csv",
                    base_results / "step6_polymer_class_water_soluble_inverse_design" / split_mode / "metrics" / "selected_target_candidate_ranked.csv",
                ]
            ),
            "step6_sampling_process_summary": _first_existing(
                [
                    results_dir / "step6_polymer_class_water_soluble_inverse_design" / split_mode / "metrics" / "sampling_process_summary.csv",
                    base_results / "step6_polymer_class_water_soluble_inverse_design" / split_mode / "metrics" / "sampling_process_summary.csv",
                ]
            ),
            "step6_sampling_attempts": _first_existing(
                [
                    results_dir / "step6_polymer_class_water_soluble_inverse_design" / split_mode / "metrics" / "sampling_attempts.csv",
                    base_results / "step6_polymer_class_water_soluble_inverse_design" / split_mode / "metrics" / "sampling_attempts.csv",
                ]
            ),
            "step7_summary": _first_existing(
                [
                    results_dir / "step7_chem_physics_analysis" / split_mode / "metrics" / "step_summary.csv",
                    base_results / "step7_chem_physics_analysis" / split_mode / "metrics" / "step_summary.csv",
                ]
            ),
            "step7_inverse_design_sampling_rollup": _first_existing(
                [
                    results_dir / "step7_chem_physics_analysis" / split_mode / "metrics" / "inverse_design_sampling_process_rollup.csv",
                    base_results / "step7_chem_physics_analysis" / split_mode / "metrics" / "inverse_design_sampling_process_rollup.csv",
                ]
            ),
            "step4_compare_summary": _first_existing(
                [
                    step4_compare_dir / "metrics" / "step_summary.csv"
                    if step4_compare_dir is not None
                    else None,
                ]
            ),
        }
    )
    return out


def _build_figure_specs(paths: Dict[str, Optional[Path]]) -> List[FigureSpec]:
    step0_fig_dirs = [paths.get("step0_dir") / "figures"] if paths.get("step0_dir") is not None else [None]
    step1_fig_dirs = [paths.get("step1_dir") / "figures"] if paths.get("step1_dir") is not None else [None]
    step2_fig_dirs = [paths.get("step2_dir") / "figures"] if paths.get("step2_dir") is not None else [None]
    step3_fig_dirs = [paths.get("step3_dir") / "figures"] if paths.get("step3_dir") is not None else [None]
    step4_reg_fig_dirs = [paths.get("step4_reg_dir") / "figures"] if paths.get("step4_reg_dir") is not None else [None]
    step4_cls_fig_dirs = [paths.get("step4_cls_dir") / "figures"] if paths.get("step4_cls_dir") is not None else [None]
    step4_reg_tuning_dirs = [paths.get("step4_reg_dir") / "tuning"] if paths.get("step4_reg_dir") is not None else [None]
    step4_cls_tuning_dirs = [paths.get("step4_cls_dir") / "tuning"] if paths.get("step4_cls_dir") is not None else [None]
    step5_fig_dirs = [paths.get("step5_dir") / "figures"] if paths.get("step5_dir") is not None else [None]
    step6_fig_dirs = [paths.get("step6_dir") / "figures"] if paths.get("step6_dir") is not None else [None]
    step7_fig_dirs = [paths.get("step7_dir") / "figures"] if paths.get("step7_dir") is not None else [None]
    if paths.get("step4_compare_dir") is not None:
        step4_compare_fig_dirs = [
            paths.get("step4_compare_dir") / "figures" / "regression",
            paths.get("step4_compare_dir") / "figures" / "classification",
            paths.get("step4_compare_dir") / "figures" / "shared",
            paths.get("step4_compare_dir") / "figures",
        ]
    else:
        step4_compare_fig_dirs = [None]
    def _blk(paths_obj: Dict[str, Optional[Path]], blk: str) -> List[Optional[Path]]:
        d = paths_obj.get(f"step7_block_{blk}_fig_dir")
        return [d] if d is not None else [None]

    block_a_fig_dirs = _blk(paths, "a")
    block_b_fig_dirs = _blk(paths, "b")
    block_c_fig_dirs = _blk(paths, "c")
    block_d_fig_dirs = _blk(paths, "d")
    block_e_fig_dirs = _blk(paths, "e")
    block_f_fig_dirs = _blk(paths, "f")
    block_g_fig_dirs = _blk(paths, "g")
    block_h_fig_dirs = _blk(paths, "h")
    block_i_fig_dirs = _blk(paths, "i")

    step3_global_curve_derived = paths.get("step3_global_threshold_curve_derived")
    step3_threshold_regions_derived = paths.get("step3_threshold_regions_derived")
    step3_condition_profiles_derived = paths.get("step3_condition_profiles_derived")
    step3_threshold_quality_heatmap_derived = paths.get("step3_threshold_quality_heatmap_derived")
    step3_chi_target_heatmap_derived = paths.get("step3_chi_target_heatmap_derived")
    step3_temperature_trends_derived = paths.get("step3_temperature_trends_derived")
    step4_regression_parity_test_derived = paths.get("step4_regression_parity_test_derived")
    step4_regression_residual_distribution_derived = paths.get("step4_regression_residual_distribution_derived")
    step4_class_confusion_test_derived = paths.get("step4_class_confusion_test_derived")
    step4_class_prob_distribution_test_derived = paths.get("step4_class_prob_distribution_test_derived")
    step2_sampling_information_derived = paths.get("step2_sampling_information_derived")
    step2_generative_metrics_derived = paths.get("step2_generative_metrics_derived")
    step2_star_count_derived = paths.get("step2_star_count_derived")
    step5_screening_funnel_derived = paths.get("step5_screening_funnel_derived")
    step6_screening_funnel_derived = paths.get("step6_screening_funnel_derived")
    step6_target_class_coverage_derived = paths.get("step6_target_class_coverage_derived")
    step5_sampling_attempt_progress_derived = paths.get("step5_sampling_attempt_progress_derived")
    step6_sampling_attempt_progress_derived = paths.get("step6_sampling_attempt_progress_derived")
    step5_requirement_snapshot_derived = paths.get("step5_requirement_snapshot_derived")
    step6_requirement_snapshot_derived = paths.get("step6_requirement_snapshot_derived")
    step5_target_chi_parity_derived = paths.get("step5_target_chi_parity_derived")
    step6_target_chi_parity_derived = paths.get("step6_target_chi_parity_derived")
    step5_target_chi_by_rank_derived = paths.get("step5_target_chi_by_rank_derived")
    step6_target_chi_by_rank_derived = paths.get("step6_target_chi_by_rank_derived")
    step5_target_confidence_by_rank_derived = paths.get("step5_target_confidence_by_rank_derived")
    step6_target_confidence_by_rank_derived = paths.get("step6_target_confidence_by_rank_derived")
    step5_target_probability_by_rank_derived = paths.get("step5_target_probability_by_rank_derived")
    step6_target_probability_by_rank_derived = paths.get("step6_target_probability_by_rank_derived")
    step6_target_sa_by_rank_derived = paths.get("step6_target_sa_by_rank_derived")
    step6_inverse_target_summary_derived = paths.get("step6_inverse_target_summary_derived")
    step5_selected_target_summary_derived = paths.get("step5_selected_target_summary_derived")
    step6_selected_target_summary_derived = paths.get("step6_selected_target_summary_derived")

    return [
        FigureSpec(
            figure_id="FigureS1",
            title="Figure S1. Polymer design foundation: training corpus quality and thermodynamic target landscape",
            destination="si",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="Token-length histogram of the training corpus",
                    candidates=_make_candidates(step0_fig_dirs, ["length_hist_train_val.png"]),
                ),
                PanelSpec(
                    caption="Synthetic accessibility score distribution",
                    candidates=_make_candidates(step0_fig_dirs, ["sa_hist_train_val.png"]),
                ),
                PanelSpec(
                    caption="Condition-wise threshold-quality heatmap over (T, φ)",
                    candidates=(
                        [step3_threshold_quality_heatmap_derived]
                        if isinstance(step3_threshold_quality_heatmap_derived, Path)
                        else []
                    )
                    + _make_candidates(
                        step3_fig_dirs,
                        [
                            "chi_target_balanced_accuracy_heatmap.png",
                            "chi_target_accuracy_heatmap.png",
                            "chi_target_f1_heatmap.png",
                            "chi_target_youden_j_heatmap.png",
                        ],
                    ),
                ),
                PanelSpec(
                    caption="Global χ distribution with class-separation threshold",
                    candidates=(
                        [step3_threshold_regions_derived]
                        if isinstance(step3_threshold_regions_derived, Path)
                        else []
                    )
                    + (
                        [step3_global_curve_derived]
                        if isinstance(step3_global_curve_derived, Path)
                        else []
                    ),
                ),
                PanelSpec(
                    caption="Condition-wise χ_target map over (T, φ)",
                    candidates=(
                        [step3_chi_target_heatmap_derived]
                        if isinstance(step3_chi_target_heatmap_derived, Path)
                        else []
                    )
                    + _make_candidates(step3_fig_dirs, ["chi_target_heatmap.png"]),
                ),
                PanelSpec(
                    caption="χ_target trend vs temperature with bootstrap CI",
                    candidates=(
                        [step3_temperature_trends_derived]
                        if isinstance(step3_temperature_trends_derived, Path)
                        else []
                    )
                    + _make_candidates(
                        step3_fig_dirs,
                        ["chi_target_vs_temperature_with_ci.png", "chi_target_vs_temperature.png"],
                    ),
                ),
            ],
        ),
        FigureSpec(
            figure_id="Figure1",
            title="Figure 1. Diffusion backbone training and novel polymer generation quality",
            destination="manuscript",
            ncols=3,
            panels=[
                PanelSpec(
                    caption="Backbone training loss convergence",
                    candidates=_make_candidates(step1_fig_dirs, ["backbone_loss_curve.png"]),
                ),
                PanelSpec(
                    caption="Bits-per-byte convergence",
                    candidates=_make_candidates(step1_fig_dirs, ["backbone_bpb_curve.png"]),
                ),
                PanelSpec(
                    caption="SA-score: generated vs training",
                    candidates=_make_candidates(step2_fig_dirs, ["sa_hist_train_vs_uncond.png"]),
                ),
                PanelSpec(
                    caption="Length distribution: generated vs training",
                    candidates=_make_candidates(step2_fig_dirs, ["length_hist_train_vs_uncond.png"]),
                ),
                PanelSpec(
                    caption="Sampling-information summary for valid-only generation",
                    candidates=(
                        [step2_sampling_information_derived]
                        if isinstance(step2_sampling_information_derived, Path)
                        else []
                    )
                    + (
                        [step2_star_count_derived]
                        if isinstance(step2_star_count_derived, Path)
                        else []
                    )
                    + _make_candidates(step2_fig_dirs, ["star_count_hist_uncond.png"]),
                ),
                PanelSpec(
                    caption="Core generative quality metrics summary",
                    candidates=(
                        [step2_generative_metrics_derived]
                        if isinstance(step2_generative_metrics_derived, Path)
                        else []
                    ),
                ),
            ],
        ),
        FigureSpec(
            figure_id="Figure2",
            title="Figure 2. Data-driven condition-aware χ_target learning with bootstrap-validated thermodynamic stability",
            destination="manuscript",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="χ_target condition heatmap over (T, φ)",
                    candidates=(
                        [step3_chi_target_heatmap_derived]
                        if isinstance(step3_chi_target_heatmap_derived, Path)
                        else []
                    )
                    + _make_candidates(step3_fig_dirs, ["chi_target_heatmap.png"]),
                ),
                PanelSpec(
                    caption="Global balanced-accuracy scan with selected χ threshold",
                    candidates=(
                        [step3_global_curve_derived]
                        if isinstance(step3_global_curve_derived, Path)
                        else []
                    )
                    + _make_candidates(step3_fig_dirs, ["chi_distribution_global_threshold.png"]),
                ),
                PanelSpec(
                    caption="χ_target trend vs temperature with bootstrap CI",
                    candidates=(
                        [step3_temperature_trends_derived]
                        if isinstance(step3_temperature_trends_derived, Path)
                        else []
                    )
                    + _make_candidates(
                        step3_fig_dirs,
                        ["chi_target_vs_temperature_with_ci.png"],
                    ),
                ),
                PanelSpec(
                    caption="Condition-wise χ_target profiles with bootstrap confidence intervals",
                    candidates=(
                        [step3_condition_profiles_derived]
                        if isinstance(step3_condition_profiles_derived, Path)
                        else []
                    )
                    + _make_candidates(block_a_fig_dirs, ["chi_vs_temperature_by_phi_and_class.png"])
                    + _make_candidates(step3_fig_dirs, ["chi_target_vs_temperature.png"]),
                ),
            ],
        ),
        FigureSpec(
            figure_id="Figure3",
            title="Figure 3. Physics-informed neural network χ regression and binary water-miscibility classification",
            destination="manuscript",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="χ parity on holdout test set",
                    candidates=(
                        [step4_regression_parity_test_derived]
                        if isinstance(step4_regression_parity_test_derived, Path)
                        else []
                    )
                    + _make_candidates(step4_reg_fig_dirs, ["chi_parity_test.png"]),
                ),
                PanelSpec(
                    caption="χ residual distribution",
                    candidates=(
                        [step4_regression_residual_distribution_derived]
                        if isinstance(step4_regression_residual_distribution_derived, Path)
                        else []
                    )
                    + _make_candidates(step4_reg_fig_dirs, ["chi_residual_distribution.png"]),
                ),
                PanelSpec(
                    caption="Confusion matrix on test set",
                    candidates=(
                        [step4_class_confusion_test_derived]
                        if isinstance(step4_class_confusion_test_derived, Path)
                        else []
                    )
                    + _make_candidates(
                        step4_cls_fig_dirs,
                        ["class_confusion_matrix_test.png", "chi_classifier_confusion_matrix_test.png"],
                    ),
                ),
                PanelSpec(
                    caption="Class probability distribution",
                    candidates=(
                        [step4_class_prob_distribution_test_derived]
                        if isinstance(step4_class_prob_distribution_test_derived, Path)
                        else []
                    )
                    + _make_candidates(
                        step4_cls_fig_dirs,
                        ["class_prob_distribution_test.png", "chi_class_prob_distribution_test.png"],
                    ),
                ),
            ],
        ),
        FigureSpec(
            figure_id="Figure4",
            title="Figure 4. Unconstrained inverse design: sampling success, target-set summary, predicted χ values, and water-miscible probabilities",
            destination="manuscript",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="Number of good polymers discovered in each sampling attempt",
                    candidates=(
                        [step5_sampling_attempt_progress_derived]
                        if isinstance(step5_sampling_attempt_progress_derived, Path)
                        else []
                    )
                    + _make_candidates(step5_fig_dirs, ["sampling_attempt_progress.png"]),
                ),
                PanelSpec(
                    caption="Target-set summary across the final selected polymers",
                    candidates=(
                        [step5_selected_target_summary_derived]
                        if isinstance(step5_selected_target_summary_derived, Path)
                        else []
                    )
                ),
                PanelSpec(
                    caption="Predicted χ values for the selected target polymers",
                    candidates=(
                        [step5_target_chi_by_rank_derived]
                        if isinstance(step5_target_chi_by_rank_derived, Path)
                        else []
                    ),
                ),
                PanelSpec(
                    caption="Water-miscible probabilities for the selected target polymers",
                    candidates=(
                        [step5_target_probability_by_rank_derived]
                        if isinstance(step5_target_probability_by_rank_derived, Path)
                        else []
                    ),
                ),
            ],
        ),
        FigureSpec(
            figure_id="Figure5",
            title="Figure 5. Polymer-class-conditioned inverse design: sampling success, target-set summary, predicted χ values, and water-miscible probabilities",
            destination="manuscript",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="Number of good polymers discovered in each sampling attempt",
                    candidates=(
                        [step6_sampling_attempt_progress_derived]
                        if isinstance(step6_sampling_attempt_progress_derived, Path)
                        else []
                    )
                    + _make_candidates(step6_fig_dirs, ["sampling_attempt_progress.png"]),
                ),
                PanelSpec(
                    caption="Target-set summary across the final selected polymers",
                    candidates=(
                        [step6_selected_target_summary_derived]
                        if isinstance(step6_selected_target_summary_derived, Path)
                        else []
                    )
                ),
                PanelSpec(
                    caption="Predicted χ values for the selected target polymers",
                    candidates=(
                        [step6_target_chi_by_rank_derived]
                        if isinstance(step6_target_chi_by_rank_derived, Path)
                        else []
                    ),
                ),
                PanelSpec(
                    caption="Water-miscible probabilities for the selected target polymers",
                    candidates=(
                        [step6_target_probability_by_rank_derived]
                        if isinstance(step6_target_probability_by_rank_derived, Path)
                        else []
                    )
                ),
            ],
        ),
        FigureSpec(
            figure_id="Figure6",
            title="Figure 6. Cross-step analysis of the selected target polymers: sampling, overlap, requirements, and chemical novelty",
            destination="manuscript",
            ncols=3,
            panels=[
                PanelSpec(
                    caption="Pipeline selection success rates across generation and inverse design steps",
                    candidates=_make_candidates(block_a_fig_dirs, ["pipeline_selection_success_rates.png"])
                    + _make_candidates(step7_fig_dirs, ["pipeline_selection_success_rates.png"]),
                ),
                PanelSpec(
                    caption="Cross-step target sampling funnel for Step 5 vs Step 6",
                    candidates=_make_candidates(block_a_fig_dirs, ["target_sampling_funnel_by_step.png"])
                    + _make_candidates(step7_fig_dirs, ["target_sampling_funnel_by_step.png"]),
                ),
                PanelSpec(
                    caption="Unique-target overlap between unconstrained and class-conditioned design",
                    candidates=_make_candidates(block_c_fig_dirs, ["selected_target_source_overlap.png"])
                    + _make_candidates(step7_fig_dirs, ["selected_target_source_overlap.png"]),
                ),
                PanelSpec(
                    caption="Combined requirement snapshot across Step 5 and Step 6 selected targets",
                    candidates=_make_candidates(block_c_fig_dirs, ["selected_target_requirement_snapshot.png"])
                    + _make_candidates(step7_fig_dirs, ["selected_target_requirement_snapshot.png"]),
                ),
                PanelSpec(
                    caption="Descriptor shift of discovered targets relative to training polymers",
                    candidates=_make_candidates(block_c_fig_dirs, ["descriptor_shift_vs_training.png"])
                    + _make_candidates(step7_fig_dirs, ["descriptor_shift_vs_training.png"]),
                ),
                PanelSpec(
                    caption="Chemical space PCA of known vs discovered polymers",
                    candidates=_make_candidates(block_g_fig_dirs, ["chemical_space_pca_known_vs_discovered.png"])
                    + _make_candidates(step7_fig_dirs, ["chemical_space_pca_known_vs_discovered.png"]),
                ),
            ],
        ),
        FigureSpec(
            figure_id="FigureS2",
            title="Figure S2. Hyperparameter Tuning and Training Diagnostics",
            destination="si",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="Regression objective trace",
                    candidates=_make_candidates(
                        step4_reg_tuning_dirs,
                        ["optuna_optimization_objective.png", "optuna_optimization_chi_r2.png"],
                    ),
                ),
                PanelSpec(
                    caption="Classification objective trace",
                    candidates=_make_candidates(step4_cls_tuning_dirs, ["optuna_optimization_objective.png"]),
                ),
                PanelSpec(
                    caption="Regression training curve",
                    candidates=_make_candidates(step4_reg_fig_dirs, ["chi_loss_curve.png"]),
                ),
                PanelSpec(
                    caption="Classification training curve",
                    candidates=_make_candidates(step4_cls_fig_dirs, ["class_loss_curve.png"]),
                ),
            ],
        ),
        FigureSpec(
            figure_id="FigureS3",
            title="Figure S3. Detailed χ-prediction and solubility-confidence diagnostics for the final selected targets",
            destination="si",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="Target χ parity for Step 5 selected polymers",
                    candidates=(
                        [step5_target_chi_parity_derived]
                        if isinstance(step5_target_chi_parity_derived, Path)
                        else []
                    )
                    + _make_candidates(
                        step5_fig_dirs,
                        [
                            "selected_target_chi_parity.png",
                            "target_vs_top1_chi_parity.png",
                            "selected_polymer_chi_parity_all_conditions.png",
                        ],
                    ),
                ),
                PanelSpec(
                    caption="Solubility confidence by rank for Step 5 selected polymers",
                    candidates=(
                        [step5_target_confidence_by_rank_derived]
                        if isinstance(step5_target_confidence_by_rank_derived, Path)
                        else []
                    )
                    + _make_candidates(
                        step5_fig_dirs,
                        ["selected_target_solubility_confidence_by_rank.png", "top1_confidence_vs_error.png"],
                    ),
                ),
                PanelSpec(
                    caption="Target χ parity for Step 6 selected polymers",
                    candidates=(
                        [step6_target_chi_parity_derived]
                        if isinstance(step6_target_chi_parity_derived, Path)
                        else []
                    )
                    + _make_candidates(
                        step6_fig_dirs,
                        [
                            "selected_target_chi_parity.png",
                            "target_vs_top1_chi_parity.png",
                            "selected_polymer_chi_parity_all_conditions.png",
                        ],
                    ),
                ),
                PanelSpec(
                    caption="Solubility confidence by rank for Step 6 selected polymers",
                    candidates=(
                        [step6_target_confidence_by_rank_derived]
                        if isinstance(step6_target_confidence_by_rank_derived, Path)
                        else []
                    )
                    + _make_candidates(
                        step6_fig_dirs,
                        ["selected_target_solubility_confidence_by_rank.png", "top1_confidence_vs_error.png"],
                    ),
                ),
            ],
        ),
        FigureSpec(
            figure_id="FigureS4",
            title="Figure S4. Extended sampling-process and screening diagnostics for the two inverse-design workflows",
            destination="si",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="Step 5 sampling-attempt progress",
                    candidates=(
                        [step5_sampling_attempt_progress_derived]
                        if isinstance(step5_sampling_attempt_progress_derived, Path)
                        else []
                    )
                    + _make_candidates(step5_fig_dirs, ["sampling_attempt_progress.png"]),
                ),
                PanelSpec(
                    caption="Step 6 sampling-attempt progress",
                    candidates=(
                        [step6_sampling_attempt_progress_derived]
                        if isinstance(step6_sampling_attempt_progress_derived, Path)
                        else []
                    )
                    + _make_candidates(step6_fig_dirs, ["sampling_attempt_progress.png"]),
                ),
                PanelSpec(
                    caption="Step 5 screening funnel from screened candidates to final selection",
                    candidates=(
                        [step5_screening_funnel_derived]
                        if isinstance(step5_screening_funnel_derived, Path)
                        else []
                    )
                    + _make_candidates(step5_fig_dirs, ["candidate_screening_funnel.png"]),
                ),
                PanelSpec(
                    caption="Step 6 screening funnel from screened candidates to final selection",
                    candidates=(
                        [step6_screening_funnel_derived]
                        if isinstance(step6_screening_funnel_derived, Path)
                        else []
                    )
                    + _make_candidates(step6_fig_dirs, ["candidate_screening_funnel.png"]),
                ),
            ],
        ),
        FigureSpec(
            figure_id="FigureS5",
            title="Figure S5. Cross-step candidate novelty and scoring diagnostics for the selected target polymers",
            destination="si",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="Selection trade-off between χ prediction and soluble confidence",
                    candidates=_make_candidates(block_c_fig_dirs, ["selection_tradeoff_chi_vs_solubility_confidence.png"])
                    + _make_candidates(step7_fig_dirs, ["selection_tradeoff_chi_vs_solubility_confidence.png"]),
                ),
                PanelSpec(
                    caption="Selected-target χ and confidence comparison by source step",
                    candidates=_make_candidates(block_c_fig_dirs, ["selected_target_chi_confidence_by_source.png"])
                    + _make_candidates(step7_fig_dirs, ["selected_target_chi_confidence_by_source.png"]),
                ),
                PanelSpec(
                    caption="Novelty similarity distribution relative to training polymers",
                    candidates=_make_candidates(block_c_fig_dirs, ["novelty_similarity_histogram.png"])
                    + _make_candidates(step7_fig_dirs, ["novelty_similarity_histogram.png"]),
                ),
                PanelSpec(
                    caption="Class-coverage of Step 6 selected targets within the cross-step analysis",
                    candidates=(
                        [step6_target_class_coverage_derived]
                        if isinstance(step6_target_class_coverage_derived, Path)
                        else []
                    )
                    + _make_candidates(block_c_fig_dirs, ["step6_target_polymer_class_coverage.png"])
                    + _make_candidates(step7_fig_dirs, ["step6_target_polymer_class_coverage.png"]),
                ),
            ],
        ),
        *(
            [
                FigureSpec(
                    figure_id="FigureS10",
                    title="Figure S10. DiT vs traditional ML across all baseline models, including train and test performance",
                    destination="si",
                    ncols=2,
                    panels=[
                        PanelSpec(
                            caption="Regression overview across DiT and all traditional models",
                            candidates=_make_candidates(
                                step4_compare_fig_dirs,
                                [
                                    "regression_overview_panel.png",
                                    "regression_delta_heatmap.png",
                                    "comparison_no_valid_data_notice.png",
                                ],
                            ),
                        ),
                        PanelSpec(
                            caption="Classification overview across DiT and all traditional models",
                            candidates=_make_candidates(
                                step4_compare_fig_dirs,
                                [
                                    "classification_overview_panel.png",
                                    "classification_delta_heatmap.png",
                                    "comparison_no_valid_data_notice.png",
                                ],
                            ),
                        ),
                        PanelSpec(
                            caption="Per-metric winner counts across DiT and traditional baselines",
                            candidates=_make_candidates(
                                step4_compare_fig_dirs,
                                [
                                    "winner_counts_by_metric.png",
                                    "comparison_no_valid_data_notice.png",
                                ],
                            ),
                        ),
                        PanelSpec(
                            caption="Missing/invalid comparison inputs by stage and model size",
                            candidates=_make_candidates(
                                step4_compare_fig_dirs,
                                [
                                    "missing_inputs_by_stage.png",
                                    "artifact_counts_by_category.png",
                                    "comparison_no_valid_data_notice.png",
                                ],
                            ),
                        ),
                    ],
                )
            ]
            if paths.get("step4_compare_dir") is not None
            else []
        ),
    ]


def _write_storyline(
    manuscript_text_dir: Path,
    si_text_dir: Path,
    model_size: str,
    split_mode: str,
) -> None:
    manuscript_text_dir.mkdir(parents=True, exist_ok=True)
    si_text_dir.mkdir(parents=True, exist_ok=True)

    manuscript_outline = [
        "# Manuscript Storyline",
        "",
        "## Abstract Summary",
        "We present a physics-guided diffusion model framework for discovering water-miscible polymers.",
        "A transformer-based diffusion backbone trained on >100k polymer SMILES generates structurally",
        "diverse candidates. Condition-aware Flory-Huggins χ_target thresholds, learned from labeled",
        "thermodynamic data, define design criteria across (T, φ) space. A PINN-augmented regression",
        "model predicts χ(T, φ) coefficients and a binary classifier screens for water-miscibility.",
        "Two inverse-design workflows - unconstrained and polymer-family-conditioned - identify",
        "novel water-miscible candidates whose novelty and thermodynamic profiles are validated",
        "by mechanistic chemistry-physics analysis.",
        "",
        "## Key Innovations",
        "1. Condition-aware χ_target learning: thresholds vary by (T, φ) and are bootstrap-validated",
        "2. PINN regression: physically constrained χ(T, φ) = (a0 + a1/T + a2lnT + a3T)(1 + b1(1-φ) + b2(1-φ)^2)",
        "3. Two-stage inverse design with class conditioning and multi-constraint scoring",
        "",
        "## Figure Narrative",
        "- Manuscript contains 6 main figures (Figures 1-6).",
        "- Corpus/target-landscape context figure is moved to Supporting Information.",
        "- Figure 1: Validates backbone training convergence and generation quality",
        "- Figure 2: Shows how condition-specific χ_target thresholds are learned with statistical confidence",
        "- Figure 3: Demonstrates PINN χ regression accuracy and binary miscibility classification",
        "- Figure 4: Summarizes Step 5 sampling progress, screening, and constraint satisfaction for the final 100 targets",
        "- Figure 5: Summarizes Step 6 class-conditioned sampling, screening selectivity, and polymer-class compliance",
        "- Figure 6: Compares Step 5 and Step 6 selected targets across overlap, requirements, novelty, and chemical space",
    ]
    (manuscript_text_dir / "manuscript_outline.md").write_text(
        "\n".join(manuscript_outline) + "\n",
        encoding="utf-8",
    )

    si_outline = [
        "# Supporting Information Outline (Step 8)",
        "",
        "## SI Figure Blocks",
        "Current Step 8 build emits six SI composite figures: Figures S1-S5 and Figure S10.",
        "1. Figure S1: Polymer design foundation: training-corpus quality and thermodynamic target landscape context.",
        "2. Figure S2: Hyperparameter tuning trajectories and learning diagnostics.",
        "3. Figure S3: Detailed χ-prediction and solubility-confidence diagnostics for selected targets.",
        "4. Figure S4: Extended sampling-process and screening diagnostics for Step 5 and Step 6.",
        "5. Figure S5: Cross-step candidate novelty and scoring diagnostics for selected targets.",
        "6. Figure S10: DiT vs traditional baseline comparison across model sizes.",
        "",
        "## SI Tables",
        "Include artifact/input status, top selected polymers from Step 5/6, and PINN coefficient tables for reproducible scientific interpretation.",
    ]
    (si_text_dir / "si_outline.md").write_text("\n".join(si_outline) + "\n", encoding="utf-8")


def _write_figure_name_lists(
    specs: List[FigureSpec],
    manuscript_figures_dir: Path,
    si_figures_dir: Path,
    metadata_dir: Path,
) -> Dict[str, Path]:
    letters = "abcdefghijklmnopqrstuvwxyz"

    def _panel_text(spec: FigureSpec) -> str:
        parts: List[str] = []
        for i, panel in enumerate(spec.panels):
            label = letters[i] if i < len(letters) else str(i + 1)
            parts.append(f"({label}) {panel.caption}")
        return " ".join(parts)

    manuscript_specs = [spec for spec in specs if spec.destination == "manuscript"]
    si_specs = [spec for spec in specs if spec.destination == "si"]
    manuscript_lines = [_figure_output_name(spec) for spec in manuscript_specs]
    si_lines = [_figure_output_name(spec) for spec in si_specs]

    manuscript_txt = manuscript_figures_dir / "figure_names.txt"
    si_txt = si_figures_dir / "figure_names.txt"
    combined_txt = metadata_dir / "figure_names_all.txt"

    manuscript_blocks: List[str] = ["# Manuscript Figures"]
    for spec in manuscript_specs:
        manuscript_blocks.extend(
            [
                f"Filename: {_figure_output_name(spec)}",
                spec.title,
                _panel_text(spec),
                "",
            ]
        )
    manuscript_txt.write_text("\n".join(manuscript_blocks).rstrip() + "\n", encoding="utf-8")

    si_blocks: List[str] = ["# Supporting Information Figures"]
    for spec in si_specs:
        si_blocks.extend(
            [
                f"Filename: {_figure_output_name(spec)}",
                spec.title,
                _panel_text(spec),
                "",
            ]
        )
    si_txt.write_text("\n".join(si_blocks).rstrip() + "\n", encoding="utf-8")

    combined_lines: List[str] = [
        "# Manuscript Figures",
    ]
    for spec in manuscript_specs:
        combined_lines.extend(
            [
                f"Filename: {_figure_output_name(spec)}",
                spec.title,
                _panel_text(spec),
                "",
            ]
        )
    combined_lines.extend(
        [
            "# Supporting Information Figures",
        ]
    )
    for spec in si_specs:
        combined_lines.extend(
            [
                f"Filename: {_figure_output_name(spec)}",
                spec.title,
                _panel_text(spec),
                "",
            ]
        )
    combined_lines.extend(
        [
            "# Quick Copy Filenames",
            *manuscript_lines,
            *si_lines,
            "",
            "# Figure Titles",
        ]
    )
    for spec in specs:
        combined_lines.append(f"{_figure_output_name(spec)}: {spec.title}")
    combined_txt.write_text("\n".join(combined_lines).rstrip() + "\n", encoding="utf-8")

    return {
        "manuscript_figure_names_txt": manuscript_txt,
        "si_figure_names_txt": si_txt,
        "all_figure_names_txt": combined_txt,
    }


def _summarize_figure_availability(panel_manifest: pd.DataFrame) -> List[str]:
    if panel_manifest.empty:
        return []
    lines: List[str] = []
    for figure_id, sub in panel_manifest.groupby("figure_id", sort=False):
        found = int((sub["status"] == "ok").sum())
        total = int(len(sub))
        if found < total:
            lines.append(f"{figure_id}: found {found}/{total}")
    return lines


def _find_placeholder_tables(table_paths: Dict[str, Path]) -> List[str]:
    placeholders: List[str] = []
    for label, path in sorted(table_paths.items()):
        df = _safe_read_csv(path)
        if df.empty and path.exists():
            placeholders.append(path.name)
            continue
        if list(df.columns) == ["note"] and len(df) == 1:
            placeholders.append(path.name)
    return placeholders


def _write_verification_summary(
    manuscript_figures_dir: Path,
    si_figures_dir: Path,
    si_tables_dir: Path,
    metadata_dir: Path,
    panel_manifest: pd.DataFrame,
    specs: List[FigureSpec],
    placeholder_tables: List[str],
) -> Path:
    manuscript_specs = [s for s in specs if s.destination == "manuscript"]
    si_specs = [s for s in specs if s.destination == "si"]

    manuscript_pngs = sorted(manuscript_figures_dir.glob("*.png"))
    si_pngs = sorted(si_figures_dir.glob("*.png"))
    final_manuscript_figure_name = _figure_output_name(manuscript_specs[-1]) if manuscript_specs else ""

    lines = [
        "Step 8 Verification Summary",
        "",
        f"manuscript_png_count: {len(manuscript_pngs)} (expected {len(manuscript_specs)})",
        f"supporting_information_png_count: {len(si_pngs)} (expected {len(si_specs)})",
        f"tableS4_exists: {int((si_tables_dir / 'tableS4_pinn_coefficients.csv').exists())}",
        f"final_manuscript_figure_expected_filename: {final_manuscript_figure_name}",
        "",
        "filename_style: full title slug (*.png), not Figure1/FigureS1 short names",
        "figure_name_lists:",
        f"- {manuscript_figures_dir / 'figure_names.txt'}",
        f"- {si_figures_dir / 'figure_names.txt'}",
        f"- {metadata_dir / 'figure_names_all.txt'}",
    ]

    if not panel_manifest.empty:
        n_found = int((panel_manifest["status"] == "ok").sum())
        n_missing = int((panel_manifest["status"] != "ok").sum())
        lines.append(f"n_panels_found: {n_found}")
        lines.append(f"n_panels_missing: {n_missing}")
        lines.append(f"package_complete: {int(n_missing == 0 and len(placeholder_tables) == 0)}")
        lines.extend(_summarize_figure_availability(panel_manifest))

    if placeholder_tables:
        lines.append("placeholder_tables:")
        lines.extend([f"- {name}" for name in placeholder_tables])

    out_path = metadata_dir / "verification_summary.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _write_status_report(
    step_dir: Path,
    manuscript_figures_dir: Path,
    si_figures_dir: Path,
    manuscript_tables_dir: Path,
    si_tables_dir: Path,
    metadata_dir: Path,
    specs: List[FigureSpec],
    panel_manifest: pd.DataFrame,
    dpi: int,
    font_size: int,
    placeholder_tables: List[str],
) -> Path:
    manuscript_specs = [s for s in specs if s.destination == "manuscript"]
    si_specs = [s for s in specs if s.destination == "si"]
    manuscript_pngs = sorted(manuscript_figures_dir.glob("*.png"))
    si_pngs = sorted(si_figures_dir.glob("*.png"))
    manuscript_tables = sorted(manuscript_tables_dir.glob("*.csv"))
    si_tables = sorted(si_tables_dir.glob("*.csv"))
    found_panels = int((panel_manifest["status"] == "ok").sum()) if not panel_manifest.empty else 0
    missing_panels = int((panel_manifest["status"] != "ok").sum()) if not panel_manifest.empty else 0
    package_complete = missing_panels == 0 and len(placeholder_tables) == 0
    package_status = "complete" if package_complete else "incomplete_missing_upstream_artifacts"

    lines = [
        "# Step 8 Status Report",
        "",
        "## Summary",
        f"- Status: {package_status}",
        f"- Manuscript figures: {len(manuscript_pngs)} (expected {len(manuscript_specs)})",
        f"- Supporting information figures: {len(si_pngs)} (expected {len(si_specs)})",
        f"- Figure style: PNG only, dpi={dpi}, font_size={font_size}, no global figure title",
        f"- Panel availability: found={found_panels}, missing={missing_panels}",
        "",
        "## Output Structure",
        f"- Step8 root: `{step_dir}`",
        f"- Manuscript figures dir: `{manuscript_figures_dir}`",
        f"- Supporting information figures dir: `{si_figures_dir}`",
        f"- Manuscript tables dir: `{manuscript_tables_dir}`",
        f"- Supporting information tables dir: `{si_tables_dir}`",
        "",
        "## Filenames",
        "- Figure filenames use full title slugs (e.g., `figure_1_...png`), not short `Figure1.png` names.",
        "- Copy-ready figure filename lists:",
        f"  - `{manuscript_figures_dir / 'figure_names.txt'}`",
        f"  - `{si_figures_dir / 'figure_names.txt'}`",
        f"  - `{metadata_dir / 'figure_names_all.txt'}`",
        "",
        "## Manuscript Tables",
    ]
    lines.extend([f"- `{p.name}`" for p in manuscript_tables])
    lines.extend(
        [
            "",
            "## Supporting Information Tables",
        ]
    )
    lines.extend([f"- `{p.name}`" for p in si_tables])
    lines.extend(
        [
            "",
            "## Key Checks",
            f"- `tableS4_pinn_coefficients.csv` exists: {int((si_tables_dir / 'tableS4_pinn_coefficients.csv').exists())}",
            f"- Verification summary: `{metadata_dir / 'verification_summary.txt'}`",
            "",
            "## Incomplete Figures",
        ]
    )
    incomplete_figures = _summarize_figure_availability(panel_manifest)
    if incomplete_figures:
        lines.extend([f"- `{entry}`" for entry in incomplete_figures])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Placeholder Tables",
        ]
    )
    if placeholder_tables:
        lines.extend([f"- `{name}`" for name in placeholder_tables])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Notes",
            "- Verification phrasing in external docs should use SI=8 and full-title PNG filenames.",
        ]
    )

    out_path = metadata_dir / "status_report.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _build_manuscript_tables(
    paths: Dict[str, Optional[Path]],
    manuscript_tables_dir: Path,
    si_tables_dir: Path,
) -> Dict[str, Path]:
    manuscript_tables_dir.mkdir(parents=True, exist_ok=True)
    si_tables_dir.mkdir(parents=True, exist_ok=True)

    s0 = _safe_first_row(paths.get("step0_summary"))
    if not s0:
        s0 = _load_step0_summary_fallback(paths.get("step0_dir"))
    s1 = _safe_first_row(paths.get("step1_summary"))
    s2 = _safe_first_row(paths.get("step2_summary"))
    s3 = _safe_first_row(paths.get("step3_summary"))
    s4 = _merge_rows(
        [
            _safe_first_row(paths.get("step4_summary")),
            _safe_first_row(paths.get("step4_reg_pipeline_summary")),
            _safe_first_row(paths.get("step4_cls_pipeline_summary")),
        ]
    )
    s5 = _safe_first_row(paths.get("step5_summary"))
    s6 = _safe_first_row(paths.get("step6_summary"))
    s7 = _safe_first_row(paths.get("step7_summary"))

    table1_rows = [
        {
            "step": "Step 0",
            "focus": "Data + tokenizer quality",
            "key_metric": "train_roundtrip_pct",
            "value": _pick(s0, ["train_roundtrip_pct"]),
            "secondary_metric": "vocab_size",
            "secondary_value": _pick(s0, ["vocab_size"]),
        },
        {
            "step": "Step 1",
            "focus": "Backbone training quality",
            "key_metric": "best_val_bpb",
            "value": _pick(s1, ["best_val_bpb"]),
            "secondary_metric": "num_params",
            "secondary_value": _pick(s1, ["num_params"]),
        },
        {
            "step": "Step 2",
            "focus": "Generative quality + target selection",
            "key_metric": "target_polymer_selection_success_rate",
            "value": _pick(s2, ["target_polymer_selection_success_rate"]),
            "secondary_metric": "novelty",
            "secondary_value": _pick(s2, ["novelty"]),
        },
        {
            "step": "Step 3",
            "focus": "chi_target learning",
            "key_metric": "global_chi_target",
            "value": _pick(s3, ["global_chi_target"]),
            "secondary_metric": "global_test_balanced_accuracy",
            "secondary_value": _pick(s3, ["global_test_balanced_accuracy", "global_balanced_accuracy"]),
        },
        {
            "step": "Step 4",
            "focus": "Predictor performance",
            "key_metric": "step4_1_test_r2",
            "value": _pick(s4, ["step4_1_test_r2", "step4_test_r2"]),
            "secondary_metric": "step4_2_test_balanced_accuracy",
            "secondary_value": _pick(
                s4, ["step4_2_test_balanced_accuracy", "step4_test_balanced_accuracy"]
            ),
        },
        {
            "step": "Step 5",
            "focus": "Water-soluble inverse design",
            "key_metric": "target_success_rate",
            "value": _pick(s5, ["target_success_rate", "target_polymer_selection_success_rate"]),
            "secondary_metric": "mean_top1_abs_error",
            "secondary_value": _pick(s5, ["mean_top1_abs_error"]),
        },
        {
            "step": "Step 6",
            "focus": "Class-conditioned inverse design",
            "key_metric": "target_success_rate",
            "value": _pick(s6, ["target_success_rate", "target_polymer_selection_success_rate"]),
            "secondary_metric": "mean_top1_abs_error",
            "secondary_value": _pick(s6, ["mean_top1_abs_error"]),
        },
        {
            "step": "Step 7",
            "focus": "Chemistry + physics interpretation",
            "key_metric": "analysis_blocks_completed",
            "value": _pick(s7, ["analysis_blocks_completed"]),
            "secondary_metric": "n_numbered_figures",
            "secondary_value": _pick(s7, ["n_numbered_figures"]),
        },
    ]
    table1 = pd.DataFrame(table1_rows)
    table1_path = manuscript_tables_dir / "table1_pipeline_key_metrics.csv"
    table1.to_csv(table1_path, index=False)

    table2_rows = [
        {
            "workflow": "Step5_water_soluble",
            "target_polymer_selection_success_rate": _pick(s5, ["target_polymer_selection_success_rate"]),
            "target_success_rate": _pick(s5, ["target_success_rate"]),
            "target_polymer_screening_yield": _pick(s5, ["target_polymer_screening_yield"]),
            "mean_top1_abs_error": _pick(s5, ["mean_top1_abs_error"]),
            "target_polymer_diversity": _pick(s5, ["target_polymer_diversity"]),
            "target_polymer_mean_sa": _pick(s5, ["target_polymer_mean_sa"]),
            "qualified_candidate_count": _pick(s5, ["qualified_candidate_count"]),
            "qualified_candidate_fraction_of_screened": _pick(
                s5, ["qualified_candidate_fraction_of_screened"]
            ),
            "selected_fraction_of_qualified": _pick(s5, ["selected_fraction_of_qualified"]),
            "sampling_attempts_used": _pick(s5, ["sampling_attempts_used"]),
            "resampling_target_polymer_count": _pick(s5, ["resampling_target_polymer_count"]),
        },
        {
            "workflow": "Step6_class_conditioned",
            "target_polymer_selection_success_rate": _pick(s6, ["target_polymer_selection_success_rate"]),
            "target_success_rate": _pick(s6, ["target_success_rate"]),
            "target_polymer_screening_yield": _pick(s6, ["target_polymer_screening_yield"]),
            "mean_top1_abs_error": _pick(s6, ["mean_top1_abs_error"]),
            "target_polymer_diversity": _pick(s6, ["target_polymer_diversity"]),
            "target_polymer_mean_sa": _pick(s6, ["target_polymer_mean_sa"]),
            "qualified_candidate_count": _pick(s6, ["qualified_candidate_count"]),
            "qualified_candidate_fraction_of_screened": _pick(
                s6, ["qualified_candidate_fraction_of_screened"]
            ),
            "selected_fraction_of_qualified": _pick(s6, ["selected_fraction_of_qualified"]),
            "sampling_attempts_used": _pick(s6, ["sampling_attempts_used"]),
            "resampling_target_polymer_count": _pick(s6, ["resampling_target_polymer_count"]),
        },
    ]
    table2 = pd.DataFrame(table2_rows)
    table2_path = manuscript_tables_dir / "table2_inverse_design_comparison.csv"
    table2.to_csv(table2_path, index=False)

    step7_artifact = None
    if paths.get("step7_dir") is not None:
        step7_artifact = paths["step7_dir"] / "metrics" / "input_artifact_status.csv"
    artifact_df = _safe_read_csv(step7_artifact)
    if artifact_df.empty:
        artifact_rows = []
        for name, p in sorted(paths.items()):
            if name in {"results_dir", "base_results_dir"} or name.endswith("_derived"):
                continue
            if isinstance(p, Path):
                artifact_rows.append(
                    {"artifact": name, "path": str(p), "exists": int(p.exists())}
                )
            else:
                artifact_rows.append({"artifact": name, "path": "", "exists": 0})
        artifact_df = pd.DataFrame(artifact_rows)
    si_table1_path = si_tables_dir / "tableS1_input_artifact_status.csv"
    artifact_df.to_csv(si_table1_path, index=False)

    step5_targets = _first_existing(
        [
            paths.get("step5_selected_target_candidate_ranked"),
            paths["step5_dir"] / "metrics" / "target_polymers.csv" if paths.get("step5_dir") is not None else None,
        ]
    )
    step6_targets = _first_existing(
        [
            paths.get("step6_selected_target_candidate_ranked"),
            paths["step6_dir"] / "metrics" / "target_polymers.csv" if paths.get("step6_dir") is not None else None,
        ]
    )
    step5_df = _safe_read_csv(step5_targets)
    step6_df = _safe_read_csv(step6_targets)
    table_s2_path = si_tables_dir / "tableS2_step5_target_polymers_top50.csv"
    table_s3_path = si_tables_dir / "tableS3_step6_target_polymers_top50.csv"
    if step5_df.empty:
        pd.DataFrame([{"note": "Step5 target_polymers.csv not found for this run."}]).to_csv(table_s2_path, index=False)
    else:
        step5_df.head(50).to_csv(table_s2_path, index=False)
    if step6_df.empty:
        pd.DataFrame([{"note": "Step6 target_polymers.csv not found for this run."}]).to_csv(table_s3_path, index=False)
    else:
        step6_df.head(50).to_csv(table_s3_path, index=False)

    coeff_path = (
        _first_existing(
            [
                paths["step4_reg_dir"] / "metrics" / "chi_coefficients.csv"
                if paths.get("step4_reg_dir") is not None
                else None,
                paths["step4_reg_dir"] / "metrics" / "polymer_coefficients_regression_only.csv"
                if paths.get("step4_reg_dir") is not None
                else None,
            ]
        )
    )
    table_s4_path = si_tables_dir / "tableS4_pinn_coefficients.csv"
    coeff_df = _safe_read_csv(coeff_path)
    if coeff_df.empty:
        pd.DataFrame(
            [{"note": "Step4 chi_coefficients.csv not found for this run."}]
        ).to_csv(table_s4_path, index=False)
    else:
        coeff_df.to_csv(table_s4_path, index=False)

    step4_compare_metrics_dir = (
        paths["step4_compare_dir"] / "metrics" if paths.get("step4_compare_dir") is not None else None
    )
    reg_cmp_df = _safe_read_csv(
        step4_compare_metrics_dir / "regression_model_size_comparison.csv"
        if step4_compare_metrics_dir is not None
        else None
    )
    cls_cmp_df = _safe_read_csv(
        step4_compare_metrics_dir / "classification_model_size_comparison.csv"
        if step4_compare_metrics_dir is not None
        else None
    )
    table_s5_path = si_tables_dir / "tableS5_step4_dit_vs_traditional_comparison.csv"
    if reg_cmp_df.empty and cls_cmp_df.empty:
        pd.DataFrame(
            [{"note": "Step4_4 comparison metrics not found for this run."}]
        ).to_csv(table_s5_path, index=False)
    else:
        reg_cols = [
            "traditional_model_name",
            "traditional_rank",
            "model_size",
            "dit_train_r2",
            "traditional_train_r2",
            "dit_r2",
            "traditional_r2",
            "delta_r2_traditional_minus_dit",
            "dit_train_rmse",
            "traditional_train_rmse",
            "dit_rmse",
            "traditional_rmse",
            "delta_rmse_traditional_minus_dit",
            "dit_train_mae",
            "traditional_train_mae",
            "dit_mae",
            "traditional_mae",
            "delta_mae_traditional_minus_dit",
        ]
        cls_cols = [
            "traditional_model_name",
            "traditional_rank",
            "model_size",
            "dit_train_balanced_accuracy",
            "traditional_train_balanced_accuracy",
            "dit_balanced_accuracy",
            "traditional_balanced_accuracy",
            "delta_balanced_accuracy_traditional_minus_dit",
            "dit_train_auroc",
            "traditional_train_auroc",
            "dit_auroc",
            "traditional_auroc",
            "delta_auroc_traditional_minus_dit",
            "dit_train_f1",
            "traditional_train_f1",
            "dit_f1",
            "traditional_f1",
            "delta_f1_traditional_minus_dit",
        ]
        reg_sub = reg_cmp_df[[c for c in reg_cols if c in reg_cmp_df.columns]].copy()
        cls_sub = cls_cmp_df[[c for c in cls_cols if c in cls_cmp_df.columns]].copy()
        frames = []
        if not reg_sub.empty:
            reg_sub.insert(0, "task", "regression")
            frames.append(reg_sub)
        if not cls_sub.empty:
            cls_sub.insert(0, "task", "classification")
            frames.append(cls_sub)
        combined = pd.concat(frames, axis=0, ignore_index=True, sort=False) if frames else pd.DataFrame()
        if not combined.empty:
            order_map = {"regression": 0, "classification": 1}
            size_order_map = {"small": 0, "medium": 1, "large": 2, "xl": 3}
            combined["_task_order"] = combined["task"].astype(str).str.lower().map(order_map).fillna(999)
            if "model_size" in combined.columns:
                combined["_size_order"] = combined["model_size"].astype(str).str.lower().map(size_order_map).fillna(999)
            else:
                combined["_size_order"] = 999
            if "traditional_rank" in combined.columns:
                combined["_rank_order"] = pd.to_numeric(combined["traditional_rank"], errors="coerce").fillna(999)
            else:
                combined["_rank_order"] = 999
            if "traditional_model_name" in combined.columns:
                combined["_model_order"] = combined["traditional_model_name"].astype(str).str.lower()
            else:
                combined["_model_order"] = ""
            combined = combined.sort_values(["_task_order", "_size_order", "_rank_order", "_model_order"]).drop(
                columns=["_task_order", "_size_order", "_rank_order", "_model_order"]
            )
        combined.to_csv(table_s5_path, index=False)

    return {
        "table1_pipeline_key_metrics": table1_path,
        "table2_inverse_design_comparison": table2_path,
        "tableS1_input_artifact_status": si_table1_path,
        "tableS2_step5_target_polymers_top50": table_s2_path,
        "tableS3_step6_target_polymers_top50": table_s3_path,
        "tableS4_pinn_coefficients": table_s4_path,
        "tableS5_step4_dit_vs_traditional_comparison": table_s5_path,
    }


def _copy_source_data(
    paths: Dict[str, Optional[Path]],
    source_data_dir: Path,
) -> pd.DataFrame:
    source_data_dir.mkdir(parents=True, exist_ok=True)

    copy_manifest: List[Dict[str, object]] = []
    to_copy: Dict[str, Optional[Path]] = {
        "step0_summary": paths.get("step0_summary"),
        "step1_summary": paths.get("step1_summary"),
        "step2_summary": paths.get("step2_summary"),
        "step3_summary": paths.get("step3_summary"),
        "step4_summary": paths.get("step4_summary"),
        "step4_reg_pipeline_summary": paths.get("step4_reg_pipeline_summary"),
        "step4_cls_pipeline_summary": paths.get("step4_cls_pipeline_summary"),
        "step5_summary": paths.get("step5_summary"),
        "step6_summary": paths.get("step6_summary"),
        "step7_summary": paths.get("step7_summary"),
    }

    if paths.get("step3_dir") is not None:
        to_copy["step3_chi_target_for_inverse_design"] = (
            paths["step3_dir"] / "metrics" / "chi_target_for_inverse_design.csv"
        )
    if paths.get("step4_reg_dir") is not None:
        to_copy["step4_reg_chi_metrics_overall"] = (
            paths["step4_reg_dir"] / "metrics" / "chi_metrics_overall.csv"
        )
    if paths.get("step4_cls_dir") is not None:
        to_copy["step4_cls_class_metrics_overall"] = (
            paths["step4_cls_dir"] / "metrics" / "class_metrics_overall.csv"
        )
    if paths.get("step5_dir") is not None:
        to_copy["step5_inverse_aggregate_metrics"] = (
            paths["step5_dir"] / "metrics" / "inverse_aggregate_metrics.csv"
        )
        to_copy["step5_target_polymers"] = paths["step5_dir"] / "metrics" / "target_polymers.csv"
    to_copy["step5_selected_target_candidate_ranked"] = paths.get("step5_selected_target_candidate_ranked")
    to_copy["step5_sampling_process_summary"] = paths.get("step5_sampling_process_summary")
    to_copy["step5_sampling_attempts"] = paths.get("step5_sampling_attempts")
    if paths.get("step6_dir") is not None:
        to_copy["step6_inverse_aggregate_metrics"] = (
            paths["step6_dir"] / "metrics" / "inverse_aggregate_metrics.csv"
        )
        to_copy["step6_target_polymers"] = paths["step6_dir"] / "metrics" / "target_polymers.csv"
    to_copy["step6_selected_target_candidate_ranked"] = paths.get("step6_selected_target_candidate_ranked")
    to_copy["step6_sampling_process_summary"] = paths.get("step6_sampling_process_summary")
    to_copy["step6_sampling_attempts"] = paths.get("step6_sampling_attempts")
    if paths.get("step7_dir") is not None:
        to_copy["step7_rollup"] = (
            paths["step7_dir"] / "metrics" / "step1_to_step6_summary_rollup.csv"
        )
        to_copy["step7_inverse_design_sampling_rollup"] = paths.get("step7_inverse_design_sampling_rollup")
        to_copy["step7_success_rates"] = paths["step7_dir"] / "metrics" / "step2_step5_step6_success_rates.csv"
        to_copy["step7_target_sampling_funnel"] = (
            paths["step7_dir"] / "metrics" / "inverse_design_target_sampling_funnel.csv"
        )
        to_copy["step7_sampling_attempt_progress"] = (
            paths["step7_dir"] / "metrics" / "inverse_design_sampling_attempt_progress.csv"
        )
        to_copy["step7_selected_target_source_overlap"] = (
            paths["step7_dir"] / "metrics" / "selected_target_source_overlap.csv"
        )
        to_copy["step7_selected_target_requirement_snapshot"] = (
            paths["step7_dir"] / "metrics" / "selected_target_requirement_snapshot.csv"
        )
        to_copy["step7_selected_target_source_comparison"] = (
            paths["step7_dir"] / "metrics" / "selected_target_source_comparison.csv"
        )
        to_copy["step7_discovered_descriptor_stats"] = (
            paths["step7_dir"] / "metrics" / "discovered_descriptor_stats.csv"
        )
    if paths.get("step4_compare_dir") is not None:
        to_copy["step4_compare_summary"] = (
            paths["step4_compare_dir"] / "metrics" / "step_summary.csv"
        )
        to_copy["step4_compare_regression_model_size_comparison"] = (
            paths["step4_compare_dir"] / "metrics" / "regression_model_size_comparison.csv"
        )
        to_copy["step4_compare_classification_model_size_comparison"] = (
            paths["step4_compare_dir"] / "metrics" / "classification_model_size_comparison.csv"
        )
        to_copy["step4_compare_missing_or_invalid_inputs"] = (
            paths["step4_compare_dir"] / "metrics" / "missing_or_invalid_inputs.csv"
        )

    for tag, src in to_copy.items():
        if src is None:
            copy_manifest.append({"tag": tag, "source_path": "", "copied_path": "", "copied": 0})
            continue
        dst = source_data_dir / f"{tag}.csv"
        copied = _copy_if_exists(src, dst)
        copy_manifest.append(
            {
                "tag": tag,
                "source_path": str(src),
                "copied_path": str(dst) if copied else "",
                "copied": int(copied),
            }
        )
    return pd.DataFrame(copy_manifest)


def main(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    split_mode = str(args.split_mode or "polymer").strip().lower()
    if split_mode != "polymer":
        raise ValueError(
            "Step 8 paper package only supports split_mode='polymer' per project requirement."
        )

    model_size = str(args.model_size or "small").strip().lower()
    if model_size not in {"small", "medium", "large", "xl"}:
        raise ValueError("model_size must be one of {'small','medium','large','xl'}")

    results_dir = Path(get_results_dir(model_size, config["paths"]["results_dir"], split_mode=split_mode))
    step_dir = results_dir / "step8_paper_package" / split_mode
    metrics_dir = step_dir / "metrics"
    manuscript_dir = step_dir / "manuscript"
    manuscript_figures_dir = manuscript_dir / "figures"
    manuscript_tables_dir = manuscript_dir / "tables"
    manuscript_text_dir = manuscript_dir / "text"
    si_dir = step_dir / "supporting_information"
    si_figures_dir = si_dir / "figures"
    si_tables_dir = si_dir / "tables"
    si_text_dir = si_dir / "text"
    source_data_dir = step_dir / "source_data"
    metadata_dir = step_dir / "metadata"

    for d in [
        metrics_dir,
        manuscript_figures_dir,
        manuscript_tables_dir,
        manuscript_text_dir,
        si_figures_dir,
        si_tables_dir,
        si_text_dir,
        source_data_dir,
        metadata_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # Fixed paper style per user request.
    dpi = PAPER_DPI
    font_size = PAPER_FONT_SIZE
    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)

    seed_value = int(config["data"]["random_seed"])
    seed_info = seed_everything(seed_value)
    save_config(config, step_dir / "config_used.yaml")
    save_run_metadata(step_dir, args.config, seed_info)
    write_initial_log(
        step_dir=step_dir,
        step_name="step8_paper_package",
        context={
            "config_path": args.config,
            "model_size": model_size,
            "split_mode": split_mode,
            "results_dir": str(results_dir),
            "only_png_outputs": True,
            "random_seed": seed_value,
        },
    )

    print("=" * 70)
    print("Step 8: paper package builder")
    print(f"model_size={model_size}")
    print(f"split_mode={split_mode} (fixed)")
    print(f"output_dir={step_dir}")
    print("=" * 70)

    # Remove legacy duplicated figure directory from older Step8 runs.
    legacy_figure_root = step_dir / "figures"
    if legacy_figure_root.exists():
        shutil.rmtree(legacy_figure_root, ignore_errors=True)

    # Make reruns deterministic: remove previous Step8 PNGs before rebuilding.
    _clear_png_files(manuscript_figures_dir)
    _clear_png_files(si_figures_dir)

    paths = _resolve_input_artifacts(config=config, model_size=model_size, split_mode=split_mode)
    artifact_rows = []
    for name, p in sorted(paths.items()):
        if name in {"results_dir", "base_results_dir"}:
            continue
        if isinstance(p, Path):
            artifact_rows.append({"artifact": name, "path": str(p), "exists": int(p.exists())})
        else:
            artifact_rows.append({"artifact": name, "path": "", "exists": 0})
    artifact_df = pd.DataFrame(artifact_rows)
    artifact_df.to_csv(metadata_dir / "input_artifact_status.csv", index=False)

    # Build derived panels used by manuscript Figures 1 and 2.
    paths["step2_sampling_information_derived"] = _build_step2_sampling_information_panel(
        paths=paths,
        metadata_dir=metadata_dir,
    )
    paths["step2_generative_metrics_derived"] = _build_step2_generative_metrics_panel(
        paths=paths,
        metadata_dir=metadata_dir,
    )
    paths["step2_star_count_derived"] = _build_step2_star_count_panel(
        paths=paths,
        metadata_dir=metadata_dir,
    )
    # Build non-heatmap Step 3 panels for manuscript Figure 2 and SI Figure S1.
    paths["step3_threshold_quality_heatmap_derived"] = _build_step3_metric_heatmap(
        paths=paths,
        metadata_dir=metadata_dir,
        value_col="balanced_accuracy",
        out_name="derived_step3_threshold_quality_heatmap.png",
        colorbar_label="Balanced accuracy",
        cmap_name="YlGnBu",
    )
    paths["step3_chi_target_heatmap_derived"] = _build_step3_metric_heatmap(
        paths=paths,
        metadata_dir=metadata_dir,
        value_col="chi_target",
        out_name="derived_step3_chi_target_heatmap.png",
        colorbar_label="χ_target",
        cmap_name="YlOrRd",
    )
    paths["step3_global_threshold_curve_derived"] = _build_step3_global_threshold_curve(
        paths=paths,
        metadata_dir=metadata_dir,
    )
    paths["step3_threshold_regions_derived"] = _build_step3_threshold_regions_panel(
        paths=paths,
        metadata_dir=metadata_dir,
    )
    paths["step3_condition_profiles_derived"] = _build_step3_condition_profiles_panel(
        paths=paths,
        metadata_dir=metadata_dir,
    )
    paths["step3_temperature_trends_derived"] = _build_step3_temperature_trends_panel(
        paths=paths,
        metadata_dir=metadata_dir,
    )
    paths["step4_regression_parity_test_derived"] = _build_step4_regression_parity_panel(
        paths=paths,
        metadata_dir=metadata_dir,
    )
    paths["step4_regression_residual_distribution_derived"] = _build_step4_regression_residual_panel(
        paths=paths,
        metadata_dir=metadata_dir,
    )
    paths["step4_class_confusion_test_derived"] = _build_step4_class_confusion_panel(
        paths=paths,
        metadata_dir=metadata_dir,
    )
    paths["step4_class_prob_distribution_test_derived"] = _build_step4_class_prob_distribution_panel(
        paths=paths,
        metadata_dir=metadata_dir,
    )
    paths["step6_target_class_coverage_derived"] = _build_step6_target_class_coverage_panel(
        paths=paths,
        metadata_dir=metadata_dir,
    )
    step5_target_csv = (
        paths["step5_dir"] / "metrics" / "target_polymers.csv"
        if paths.get("step5_dir") is not None
        else None
    )
    step5_selection_summary_csv = (
        paths["step5_dir"] / "metrics" / "target_polymer_selection_summary.csv"
        if paths.get("step5_dir") is not None
        else None
    )
    step6_target_csv = _first_existing(
        [
            paths["step6_dir"] / "metrics" / "target_polymers.csv"
            if paths.get("step6_dir") is not None
            else None,
            _latest_resampling_attempt_csv(paths.get("step6_dir"), "target_polymers.csv"),
        ]
    )
    step6_selection_summary_csv = (
        step6_target_csv.parent / "target_polymer_selection_summary.csv"
        if isinstance(step6_target_csv, Path)
        else None
    )
    step6_inverse_targets_csv = (
        paths.get("step6_dir") / "metrics" / "inverse_targets.csv"
        if paths.get("step6_dir") is not None
        else None
    )
    step6_plot_target_csv = _resolve_step6_target_plot_csv(
        target_csv=step6_target_csv,
        inverse_targets_csv=step6_inverse_targets_csv,
        step4_reg_dir=paths.get("step4_reg_dir"),
        step4_cls_dir=paths.get("step4_cls_dir"),
        config=config,
        model_size=model_size,
        split_mode=split_mode,
        cache_csv=metadata_dir / "derived_step6_target_polymers_scored.csv",
    )
    if not isinstance(step6_plot_target_csv, Path) or not step6_plot_target_csv.exists():
        step6_plot_target_csv = step6_target_csv
    step6_attempt_count = _completed_resampling_attempt_count(paths.get("step6_dir"))
    paths["step5_sampling_attempt_progress_derived"] = _build_sampling_attempt_progress_panel(
        attempts_csv=paths.get("step5_sampling_attempts"),
        out_png=metadata_dir / "derived_step5_sampling_attempt_progress.png",
    )
    paths["step6_sampling_attempt_progress_derived"] = _build_sampling_attempt_progress_panel(
        attempts_csv=paths.get("step6_sampling_attempts"),
        out_png=metadata_dir / "derived_step6_sampling_attempt_progress.png",
    )
    if not isinstance(paths.get("step6_sampling_attempt_progress_derived"), Path) or not paths["step6_sampling_attempt_progress_derived"].exists():
        paths["step6_sampling_attempt_progress_derived"] = _build_resampling_attempt_progress_panel(
            step_dir=paths.get("step6_dir"),
            out_png=metadata_dir / "derived_step6_sampling_attempt_progress.png",
        )
    paths["step5_requirement_snapshot_derived"] = _build_requirement_snapshot_panel(
        target_csv=step5_target_csv,
        out_png=metadata_dir / "derived_step5_requirement_snapshot.png",
    )
    paths["step6_requirement_snapshot_derived"] = _build_requirement_snapshot_panel(
        target_csv=step6_plot_target_csv,
        out_png=metadata_dir / "derived_step6_requirement_snapshot.png",
    )
    paths["step5_target_chi_parity_derived"] = _build_target_chi_parity_panel(
        target_csv=step5_target_csv,
        out_png=metadata_dir / "derived_step5_target_chi_parity.png",
    )
    paths["step6_target_chi_parity_derived"] = _build_target_chi_parity_panel(
        target_csv=step6_plot_target_csv,
        out_png=metadata_dir / "derived_step6_target_chi_parity.png",
    )
    paths["step5_target_chi_by_rank_derived"] = _build_target_chi_by_rank_panel(
        target_csv=step5_target_csv,
        out_png=metadata_dir / "derived_step5_target_chi_by_rank.png",
    )
    paths["step6_target_chi_by_rank_derived"] = _build_target_chi_by_rank_panel(
        target_csv=step6_plot_target_csv,
        out_png=metadata_dir / "derived_step6_target_chi_by_rank.png",
    )
    paths["step5_target_confidence_by_rank_derived"] = _build_target_confidence_by_rank_panel(
        target_csv=step5_target_csv,
        out_png=metadata_dir / "derived_step5_target_confidence_by_rank.png",
    )
    paths["step6_target_confidence_by_rank_derived"] = _build_target_confidence_by_rank_panel(
        target_csv=step6_plot_target_csv,
        out_png=metadata_dir / "derived_step6_target_confidence_by_rank.png",
    )
    paths["step5_target_probability_by_rank_derived"] = _build_target_probability_by_rank_panel(
        target_csv=step5_target_csv,
        out_png=metadata_dir / "derived_step5_target_probability_by_rank.png",
    )
    paths["step6_target_probability_by_rank_derived"] = _build_target_probability_by_rank_panel(
        target_csv=step6_plot_target_csv,
        out_png=metadata_dir / "derived_step6_target_probability_by_rank.png",
    )
    paths["step6_target_sa_by_rank_derived"] = _build_target_sa_by_rank_panel(
        target_csv=step6_plot_target_csv,
        out_png=metadata_dir / "derived_step6_target_sa_by_rank.png",
    )
    paths["step5_selected_target_summary_derived"] = _build_selected_target_summary_panel(
        target_csv=step5_target_csv,
        out_png=metadata_dir / "derived_step5_selected_target_summary.png",
        selection_summary_csv=step5_selection_summary_csv,
    )
    paths["step6_selected_target_summary_derived"] = _build_selected_target_summary_panel(
        target_csv=step6_plot_target_csv,
        out_png=metadata_dir / "derived_step6_selected_target_summary.png",
        inverse_targets_csv=step6_inverse_targets_csv,
        selection_summary_csv=step6_selection_summary_csv,
        attempt_count_override=step6_attempt_count,
    )
    paths["step6_inverse_target_summary_derived"] = _build_inverse_target_summary_panel(
        inverse_targets_csv=step6_inverse_targets_csv,
        out_png=metadata_dir / "derived_step6_inverse_target_summary.png",
    )
    paths["step5_screening_funnel_derived"] = _build_screening_funnel_panel(
        summary_csv=_first_existing(
            [
                paths.get("step5_sampling_process_summary"),
                paths["step5_dir"] / "metrics" / "target_polymer_selection_summary.csv"
                if paths.get("step5_dir") is not None
                else None,
            ]
        ),
        out_png=metadata_dir / "derived_step5_screening_funnel.png",
    )
    paths["step6_screening_funnel_derived"] = _build_screening_funnel_panel(
        summary_csv=_first_existing(
            [
                paths.get("step6_sampling_process_summary"),
                paths["step6_dir"] / "metrics" / "target_polymer_selection_summary.csv"
                if paths.get("step6_dir") is not None
                else None,
            ]
        ),
        out_png=metadata_dir / "derived_step6_screening_funnel.png",
    )
    for key in [
        "step2_sampling_information_derived",
        "step2_generative_metrics_derived",
        "step2_star_count_derived",
        "step3_threshold_quality_heatmap_derived",
        "step3_chi_target_heatmap_derived",
        "step3_global_threshold_curve_derived",
        "step3_threshold_regions_derived",
        "step3_condition_profiles_derived",
        "step3_temperature_trends_derived",
        "step4_regression_parity_test_derived",
        "step4_regression_residual_distribution_derived",
        "step4_class_confusion_test_derived",
        "step4_class_prob_distribution_test_derived",
        "step5_sampling_attempt_progress_derived",
        "step6_sampling_attempt_progress_derived",
        "step5_requirement_snapshot_derived",
        "step6_requirement_snapshot_derived",
        "step5_target_chi_parity_derived",
        "step6_target_chi_parity_derived",
        "step5_target_chi_by_rank_derived",
        "step6_target_chi_by_rank_derived",
        "step5_target_confidence_by_rank_derived",
        "step6_target_confidence_by_rank_derived",
        "step5_target_probability_by_rank_derived",
        "step6_target_probability_by_rank_derived",
        "step6_target_sa_by_rank_derived",
        "step6_inverse_target_summary_derived",
        "step5_selected_target_summary_derived",
        "step6_selected_target_summary_derived",
        "step5_screening_funnel_derived",
        "step6_screening_funnel_derived",
        "step6_target_class_coverage_derived",
    ]:
        p = paths.get(key)
        if not isinstance(p, Path) or not p.exists():
            print(
                f"[WARN] Derived panel not generated: {key}. "
                "Composer will use fallback candidates or a missing placeholder."
            )

    specs = _build_figure_specs(paths)
    panel_manifest_frames: List[pd.DataFrame] = []
    generated_main: List[Path] = []
    generated_si: List[Path] = []

    for spec in specs:
        out_dir = manuscript_figures_dir if spec.destination == "manuscript" else si_figures_dir
        out_png = out_dir / _figure_output_name(spec)
        panel_df = _compose_figure(spec=spec, out_png=out_png)
        panel_manifest_frames.append(panel_df)
        if spec.destination == "manuscript":
            generated_main.append(out_png)
        else:
            generated_si.append(out_png)

    panel_manifest = (
        pd.concat(panel_manifest_frames, ignore_index=True)
        if panel_manifest_frames
        else pd.DataFrame(columns=["figure_id", "panel_index", "panel_label", "caption", "source_path", "status"])
    )
    panel_manifest.to_csv(metadata_dir / "figure_panel_manifest.csv", index=False)

    table_paths = _build_manuscript_tables(
        paths=paths,
        manuscript_tables_dir=manuscript_tables_dir,
        si_tables_dir=si_tables_dir,
    )
    placeholder_tables = _find_placeholder_tables(table_paths)
    source_manifest = _copy_source_data(paths=paths, source_data_dir=source_data_dir)
    source_manifest.to_csv(metadata_dir / "source_data_copy_manifest.csv", index=False)

    _write_storyline(
        manuscript_text_dir=manuscript_text_dir,
        si_text_dir=si_text_dir,
        model_size=model_size,
        split_mode=split_mode,
    )
    _write_figure_name_lists(
        specs=specs,
        manuscript_figures_dir=manuscript_figures_dir,
        si_figures_dir=si_figures_dir,
        metadata_dir=metadata_dir,
    )
    _write_verification_summary(
        manuscript_figures_dir=manuscript_figures_dir,
        si_figures_dir=si_figures_dir,
        si_tables_dir=si_tables_dir,
        metadata_dir=metadata_dir,
        panel_manifest=panel_manifest,
        specs=specs,
        placeholder_tables=placeholder_tables,
    )
    _write_status_report(
        step_dir=step_dir,
        manuscript_figures_dir=manuscript_figures_dir,
        si_figures_dir=si_figures_dir,
        manuscript_tables_dir=manuscript_tables_dir,
        si_tables_dir=si_tables_dir,
        metadata_dir=metadata_dir,
        specs=specs,
        panel_manifest=panel_manifest,
        dpi=dpi,
        font_size=font_size,
        placeholder_tables=placeholder_tables,
    )

    missing_panels = int((panel_manifest["status"] != "ok").sum()) if not panel_manifest.empty else 0
    found_panels = int((panel_manifest["status"] == "ok").sum()) if not panel_manifest.empty else 0
    copied_source_files = int(source_manifest["copied"].sum()) if not source_manifest.empty else 0
    package_complete = missing_panels == 0 and len(placeholder_tables) == 0

    summary = {
        "step": "step8_paper_package",
        "model_size": model_size,
        "split_mode": split_mode,
        "only_png_outputs": True,
        "n_manuscript_figures": int(len(generated_main)),
        "n_supporting_information_figures": int(len(generated_si)),
        "n_total_figures": int(len(generated_main) + len(generated_si)),
        "n_total_panels": int(len(panel_manifest)),
        "n_panels_found": found_panels,
        "n_panels_missing": missing_panels,
        "package_complete": bool(package_complete),
        "n_placeholder_tables": int(len(placeholder_tables)),
        "n_key_tables": int(len(table_paths)),
        "n_source_data_files_copied": copied_source_files,
        "step8_output_dir": str(step_dir),
        "manuscript_dir": str(manuscript_dir),
        "supporting_information_dir": str(si_dir),
    }

    with open(metrics_dir / "step8_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    save_step_summary(summary, metrics_dir)
    save_artifact_manifest(step_dir=step_dir, metrics_dir=metrics_dir, figures_dir=None, dpi=dpi)

    print("Step 8 complete.")
    if not package_complete:
        print(
            f"[WARN] Step 8 package is incomplete: missing_panels={missing_panels}, "
            f"placeholder_tables={len(placeholder_tables)}"
        )
    print(f"Manuscript package: {manuscript_dir}")
    print(f"Supporting information package: {si_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 8: build paper package from Steps 0-7 outputs (PNG only, polymer split only)"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "medium", "large", "xl"])
    parser.add_argument(
        "--split_mode",
        type=str,
        default="polymer",
        help="Paper package split mode. Must be polymer.",
    )
    main(parser.parse_args())
