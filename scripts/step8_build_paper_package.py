#!/usr/bin/env python
"""Step 8: build manuscript/SI paper package from Steps 0-7 outputs.

This script curates key results into:
- manuscript figures (7 multi-panel PNGs)
- supporting information figures (PNG)
- manuscript/SI tables (CSV)
- source-data and metadata manifests

Constraint requested by user:
- Paper package is fixed to split_mode=polymer.
- Only PNG figure outputs are generated.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, save_config
from src.utils.figure_style import apply_publication_figure_style
from src.utils.model_scales import get_results_dir
from src.utils.reproducibility import save_run_metadata, seed_everything
from src.utils.reporting import save_artifact_manifest, save_step_summary, write_initial_log

PAPER_FONT_SIZE = 16
PAPER_DPI = 600


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


def _make_candidates(base_dirs: List[Optional[Path]], names: List[str]) -> List[Path]:
    out: List[Path] = []
    for d in base_dirs:
        if d is None:
            continue
        for name in names:
            out.append(d / name)
    return out


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

    fig_w = 6.5 * ncols
    fig_h = 5.0 * nrows + 0.7
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    letters = "abcdefghijklmnopqrstuvwxyz"
    for i, panel in enumerate(spec.panels):
        ax = axes[i]
        label = letters[i] if i < len(letters) else str(i + 1)
        src = _first_existing(panel.candidates)
        status = "missing"
        src_str = ""

        if src is not None and src.exists():
            try:
                img = mpimg.imread(src)
                ax.imshow(img)
                ax.set_axis_off()
                status = "ok"
                src_str = str(src)
            except Exception:
                _draw_missing_panel(ax, panel.caption, font_size=font_size)
        else:
            _draw_missing_panel(ax, panel.caption, font_size=font_size)

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

    # No global title or panel captions rendered on composed figures.
    fig.subplots_adjust(top=0.98, wspace=0.12, hspace=0.18)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(dpi))
    plt.close(fig)
    return pd.DataFrame(panel_rows)


def _copy_if_exists(src: Optional[Path], dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _slugify(text: str) -> str:
    chars: List[str] = []
    for ch in str(text).strip().lower():
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


def _build_step3_objective_heatmap_from_metrics(
    paths: Dict[str, Optional[Path]],
    metadata_dir: Path,
) -> Optional[Path]:
    """Build a condition-wise balanced-accuracy heatmap for Figure 3(b)."""
    step3_dir = paths.get("step3_dir")
    if step3_dir is None:
        return None

    best_csv = step3_dir / "metrics" / "chi_target_best_by_condition.csv"
    scan_csv = step3_dir / "metrics" / "chi_target_scan_by_condition.csv"

    df = _safe_read_csv(best_csv)
    if df.empty:
        scan_df = _safe_read_csv(scan_csv)
        if scan_df.empty or not {"temperature", "phi", "balanced_accuracy"}.issubset(scan_df.columns):
            return None
        df = (
            scan_df.groupby(["temperature", "phi"], as_index=False)["balanced_accuracy"]
            .max()
            .sort_values(["temperature", "phi"])
        )
    elif not {"temperature", "phi", "balanced_accuracy"}.issubset(df.columns):
        return None

    pivot = (
        df.pivot(index="temperature", columns="phi", values="balanced_accuracy")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    if pivot.empty:
        return None

    arr = pivot.to_numpy(dtype=float)
    out_png = metadata_dir / "derived_step3_balanced_accuracy_heatmap.png"

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    vmin = float(np.nanmin(arr)) if np.isfinite(arr).any() else 0.0
    vmax = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1e-6
    im = ax.imshow(arr, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)

    xlabels = [f"{x:.2f}" for x in pivot.columns.to_list()]
    ylabels = [f"{y:.2f}" for y in pivot.index.to_list()]
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=PAPER_FONT_SIZE)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=PAPER_FONT_SIZE)
    ax.set_xlabel("phi", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("temperature (K)", fontsize=PAPER_FONT_SIZE)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            txt = "nan" if pd.isna(v) else f"{v:.2f}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=PAPER_FONT_SIZE,
                color="white" if np.isfinite(v) and (v > (vmin + vmax) / 2.0) else "black",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=PAPER_FONT_SIZE)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI)
    plt.close(fig)
    return out_png


def _build_step3_bootstrap_uncertainty_heatmap(
    paths: Dict[str, Optional[Path]],
    metadata_dir: Path,
) -> Optional[Path]:
    """Build χ_target bootstrap-uncertainty heatmap for Figure 3(b)."""
    step3_dir = paths.get("step3_dir")
    if step3_dir is None:
        return None

    best_csv = step3_dir / "metrics" / "chi_target_best_by_condition.csv"
    df = _safe_read_csv(best_csv)
    if df.empty or not {"temperature", "phi"}.issubset(df.columns):
        return None

    if "chi_target_boot_std" in df.columns:
        val_col = "chi_target_boot_std"
    elif {"chi_target_boot_q975", "chi_target_boot_q025"}.issubset(df.columns):
        val_col = "chi_target_boot_iqr95"
        df[val_col] = df["chi_target_boot_q975"] - df["chi_target_boot_q025"]
    else:
        return None

    pivot = (
        df.pivot(index="temperature", columns="phi", values=val_col)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    if pivot.empty:
        return None

    arr = pivot.to_numpy(dtype=float)
    out_png = metadata_dir / "derived_step3_chi_bootstrap_uncertainty_heatmap.png"

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    vmin = float(np.nanmin(arr)) if np.isfinite(arr).any() else 0.0
    vmax = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1e-6
    im = ax.imshow(arr, cmap="magma", aspect="auto", vmin=vmin, vmax=vmax)

    xlabels = [f"{x:.2f}" for x in pivot.columns.to_list()]
    ylabels = [f"{y:.2f}" for y in pivot.index.to_list()]
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=PAPER_FONT_SIZE)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=PAPER_FONT_SIZE)
    ax.set_xlabel("phi", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("temperature (K)", fontsize=PAPER_FONT_SIZE)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            txt = "nan" if pd.isna(v) else f"{v:.3f}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=PAPER_FONT_SIZE,
                color="white" if np.isfinite(v) and (v > (vmin + vmax) / 2.0) else "black",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=PAPER_FONT_SIZE)
    cbar.set_label("χ_target bootstrap uncertainty", fontsize=PAPER_FONT_SIZE)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PAPER_DPI)
    plt.close(fig)
    return out_png


def _resolve_input_artifacts(
    config: Dict,
    model_size: str,
    split_mode: str,
) -> Dict[str, Optional[Path]]:
    base_results = Path(config["paths"]["results_dir"])
    results_dir = Path(get_results_dir(model_size, config["paths"]["results_dir"], split_mode=None))

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
                    results_dir / "step4_chi_training" / split_mode / "metrics" / "step_summary.csv",
                    base_results / "step4_chi_training" / split_mode / "metrics" / "step_summary.csv",
                ]
            ),
            "step5_summary": _first_existing(
                [
                    results_dir / "step5_water_soluble_inverse_design" / split_mode / "metrics" / "step_summary.csv",
                    base_results / "step5_water_soluble_inverse_design" / split_mode / "metrics" / "step_summary.csv",
                ]
            ),
            "step6_summary": _first_existing(
                [
                    results_dir / "step6_polymer_class_water_soluble_inverse_design" / split_mode / "metrics" / "step_summary.csv",
                    base_results / "step6_polymer_class_water_soluble_inverse_design" / split_mode / "metrics" / "step_summary.csv",
                ]
            ),
            "step7_summary": _first_existing(
                [
                    results_dir / "step7_chem_physics_analysis" / split_mode / "metrics" / "step_summary.csv",
                    base_results / "step7_chem_physics_analysis" / split_mode / "metrics" / "step_summary.csv",
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

    step3_boot_uncertainty_derived = paths.get("step3_bootstrap_uncertainty_heatmap_derived")
    step3_bal_acc_derived = paths.get("step3_balanced_accuracy_heatmap_derived")
    step3_bal_acc_candidates: List[Path] = []
    # Prefer richer Step 7 condition-wise class-contrast maps for Figure 3(b).
    step3_bal_acc_candidates += _make_candidates(
        block_a_fig_dirs,
        ["chi_class_significance_heatmap.png", "chi_class_delta_heatmap.png"],
    )
    step3_bal_acc_candidates += _make_candidates(
        step3_fig_dirs,
        [
            "chi_target_balanced_accuracy_heatmap.png",
            "chi_target_classification_objective_heatmap.png",
            "chi_target_objective_heatmap.png",
        ],
    )
    if isinstance(step3_boot_uncertainty_derived, Path):
        step3_bal_acc_candidates.append(step3_boot_uncertainty_derived)
    if isinstance(step3_bal_acc_derived, Path):
        step3_bal_acc_candidates.append(step3_bal_acc_derived)

    return [
        FigureSpec(
            figure_id="Figure1",
            title="Figure 1. Polymer design foundation: training corpus quality and thermodynamic target landscape",
            destination="manuscript",
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
                    caption="Condition-wise χ_target map over (T, φ)",
                    candidates=_make_candidates(step3_fig_dirs, ["chi_target_heatmap.png"]),
                ),
                PanelSpec(
                    caption="Global χ distribution with class-separation threshold",
                    candidates=_make_candidates(step3_fig_dirs, ["chi_distribution_global_threshold.png"]),
                ),
            ],
        ),
        FigureSpec(
            figure_id="Figure2",
            title="Figure 2. Diffusion backbone training and novel polymer generation quality",
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
                    caption="Star-token structural quality",
                    candidates=_make_candidates(step2_fig_dirs, ["star_count_hist_uncond.png"]),
                ),
            ],
        ),
        FigureSpec(
            figure_id="Figure3",
            title="Figure 3. Data-driven condition-aware χ_target learning with bootstrap-validated thermodynamic stability",
            destination="manuscript",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="χ_target condition heatmap over (T, φ)",
                    candidates=_make_candidates(step3_fig_dirs, ["chi_target_heatmap.png"]),
                ),
                PanelSpec(
                    caption="Condition-wise classification-objective map",
                    candidates=step3_bal_acc_candidates,
                ),
                PanelSpec(
                    caption="χ_target trend vs temperature with bootstrap CI",
                    candidates=_make_candidates(
                        step3_fig_dirs,
                        ["chi_target_vs_temperature_with_ci.png", "chi_target_vs_temperature.png"],
                    ),
                ),
                PanelSpec(
                    caption="Global χ distribution and class separation",
                    candidates=_make_candidates(step3_fig_dirs, ["chi_distribution_global_threshold.png"]),
                ),
            ],
        ),
        FigureSpec(
            figure_id="Figure4",
            title="Figure 4. Physics-informed neural network χ regression and binary water-miscibility classification",
            destination="manuscript",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="χ parity on holdout test set",
                    candidates=_make_candidates(step4_reg_fig_dirs, ["chi_parity_test.png"]),
                ),
                PanelSpec(
                    caption="χ residual distribution",
                    candidates=_make_candidates(step4_reg_fig_dirs, ["chi_residual_distribution.png"]),
                ),
                PanelSpec(
                    caption="ROC curve on test set",
                    candidates=_make_candidates(
                        step4_cls_fig_dirs,
                        ["classifier_roc_test.png", "chi_classifier_roc_test.png"],
                    ),
                ),
                PanelSpec(
                    caption="Class probability distribution",
                    candidates=_make_candidates(
                        step4_cls_fig_dirs,
                        ["class_prob_distribution_test.png", "chi_class_prob_distribution_test.png"],
                    ),
                ),
            ],
        ),
        FigureSpec(
            figure_id="Figure5",
            title="Figure 5. Unconstrained inverse design: water-soluble candidate selection and design-space coverage",
            destination="manuscript",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="Top-k target coverage curve",
                    candidates=_make_candidates(step5_fig_dirs, ["topk_target_success_curve.png"]),
                ),
                PanelSpec(
                    caption="Candidate screening funnel",
                    candidates=_make_candidates(step5_fig_dirs, ["candidate_screening_funnel.png"]),
                ),
                PanelSpec(
                    caption="Selected polymer χ parity across all (T, φ) conditions",
                    candidates=_make_candidates(step5_fig_dirs, ["selected_polymer_chi_parity_all_conditions.png"]),
                ),
                PanelSpec(
                    caption="Top-1 joint hit heatmap by condition",
                    candidates=_make_candidates(step5_fig_dirs, ["top1_joint_hit_heatmap.png"]),
                ),
            ],
        ),
        FigureSpec(
            figure_id="Figure6",
            title="Figure 6. Polymer-family-conditioned inverse design and class-specific thermodynamic coverage",
            destination="manuscript",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="Class-conditioned top-k success curve",
                    candidates=_make_candidates(step6_fig_dirs, ["topk_target_success_curve.png"]),
                ),
                PanelSpec(
                    caption="Target success by polymer class",
                    candidates=_make_candidates(step6_fig_dirs, ["target_success_by_polymer_class.png"]),
                ),
                PanelSpec(
                    caption="Joint hit heatmap: polymer class × condition",
                    candidates=_make_candidates(step6_fig_dirs, ["top1_joint_hit_class_condition_heatmap.png"]),
                ),
                PanelSpec(
                    caption="Polymer class coverage of discovered targets",
                    candidates=_make_candidates(block_c_fig_dirs, ["step6_target_polymer_class_coverage.png"])
                    + _make_candidates(step7_fig_dirs, ["step6_target_polymer_class_coverage.png"]),
                ),
            ],
        ),
        FigureSpec(
            figure_id="Figure7",
            title="Figure 7. Mechanistic interpretation: chemical novelty, thermodynamic profiles, and structure-property coupling",
            destination="manuscript",
            ncols=3,
            panels=[
                PanelSpec(
                    caption="Chemical space PCA: known vs discovered polymers",
                    candidates=_make_candidates(block_g_fig_dirs, ["chemical_space_pca_known_vs_discovered.png"])
                    + _make_candidates(step7_fig_dirs, ["chemical_space_pca_known_vs_discovered.png"]),
                ),
                PanelSpec(
                    caption="Spinodal phase diagram from Flory-Huggins theory",
                    candidates=_make_candidates(block_e_fig_dirs, ["spinodal_phase_diagram.png"])
                    + _make_candidates(step7_fig_dirs, ["spinodal_phase_diagram.png"]),
                ),
                PanelSpec(
                    caption="Mean χ(T, φ) surface by class",
                    candidates=_make_candidates(block_e_fig_dirs, ["chi_surface_mean_by_class.png"])
                    + _make_candidates(step7_fig_dirs, ["chi_surface_mean_by_class.png"]),
                ),
                PanelSpec(
                    caption="PINN coefficient distributions by class",
                    candidates=_make_candidates(block_d_fig_dirs, ["coefficient_violin_by_class.png"])
                    + _make_candidates(step7_fig_dirs, ["coefficient_violin_by_class.png"]),
                ),
                PanelSpec(
                    caption="Embedding-PINN correlation heatmap",
                    candidates=_make_candidates(block_i_fig_dirs, ["embedding_pinn_correlation_heatmap.png"])
                    + _make_candidates(step7_fig_dirs, ["embedding_pinn_correlation_heatmap.png"]),
                ),
                PanelSpec(
                    caption="Molecular descriptor shift: discovered vs training",
                    candidates=_make_candidates(block_c_fig_dirs, ["descriptor_shift_vs_training.png"])
                    + _make_candidates(block_f_fig_dirs, ["descriptor_boxplot_by_class.png"])
                    + _make_candidates(step7_fig_dirs, ["descriptor_shift_vs_training.png", "descriptor_boxplot_by_class.png"]),
                ),
            ],
        ),
        FigureSpec(
            figure_id="FigureS1",
            title="Figure S1. Hyperparameter Tuning and Training Diagnostics",
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
            figure_id="FigureS2",
            title="Figure S2. Inverse-Design Screening and Confidence Diagnostics",
            destination="si",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="Confidence-error relation (unconstrained design)",
                    candidates=_make_candidates(step5_fig_dirs, ["top1_confidence_vs_error.png"]),
                ),
                PanelSpec(
                    caption="Confidence-error relation (class-conditioned design)",
                    candidates=_make_candidates(step6_fig_dirs, ["top1_confidence_vs_error.png"]),
                ),
                PanelSpec(
                    caption="Top-5 polymer selection frequency",
                    candidates=_make_candidates(step5_fig_dirs, ["top5_polymer_selection_frequency.png"]),
                ),
                PanelSpec(
                    caption="Candidate screening funnel (class-conditioned)",
                    candidates=_make_candidates(step6_fig_dirs, ["candidate_screening_funnel.png"]),
                ),
            ],
        ),
        FigureSpec(
            figure_id="FigureS3",
            title="Figure S3. Extended Chemical and Functional-Group Analysis",
            destination="si",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="Descriptor boxplot by class",
                    candidates=_make_candidates(step7_fig_dirs, ["descriptor_boxplot_by_class.png"]),
                ),
                PanelSpec(
                    caption="Functional-group frequencies by class",
                    candidates=_make_candidates(step7_fig_dirs, ["functional_group_frequency_by_class.png"]),
                ),
                PanelSpec(
                    caption="Classification-vs-χ descriptor shift",
                    candidates=_make_candidates(block_h_fig_dirs, ["classification_vs_chi_descriptor_shift.png"])
                    + _make_candidates(step7_fig_dirs, ["classification_vs_chi_descriptor_shift.png"]),
                ),
                PanelSpec(
                    caption="Classification-vs-χ functional-group shift",
                    candidates=_make_candidates(block_h_fig_dirs, ["classification_vs_chi_fg_frequency.png"])
                    + _make_candidates(step7_fig_dirs, ["classification_vs_chi_fg_frequency.png"]),
                ),
            ],
        ),
        FigureSpec(
            figure_id="FigureS4",
            title="Figure S4. PINN Polynomial Thermodynamics and Flory-Huggins Phase Analysis",
            destination="si",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="dχ/dT distribution by class",
                    candidates=_make_candidates(block_d_fig_dirs, ["dchi_dT_distribution_by_class.png"])
                    + _make_candidates(step7_fig_dirs, ["dchi_dT_distribution_by_class.png"]),
                ),
                PanelSpec(
                    caption="PINN coefficient scatter (a1 vs a3)",
                    candidates=_make_candidates(block_d_fig_dirs, ["coefficient_a1_vs_a3_by_class.png"])
                    + _make_candidates(step7_fig_dirs, ["coefficient_a1_vs_a3_by_class.png"]),
                ),
                PanelSpec(
                    caption="Free energy of mixing by class",
                    candidates=_make_candidates(block_e_fig_dirs, ["free_energy_mixing_by_class.png"])
                    + _make_candidates(step7_fig_dirs, ["free_energy_mixing_by_class.png"]),
                ),
                PanelSpec(
                    caption="Miscible-fraction below spinodal heatmap",
                    candidates=_make_candidates(block_e_fig_dirs, ["miscible_fraction_below_spinodal_by_class.png"])
                    + _make_candidates(step7_fig_dirs, ["miscible_fraction_below_spinodal_by_class.png"]),
                ),
            ],
        ),
        FigureSpec(
            figure_id="FigureS5",
            title="Figure S5. Embedding Space, PINN Sensitivity, and χ Dataset Analysis",
            destination="si",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="Embedding PCA by source",
                    candidates=_make_candidates(block_i_fig_dirs, ["step1_embedding_pca_by_source.png"])
                    + _make_candidates(step7_fig_dirs, ["step1_embedding_pca_by_source.png"]),
                ),
                PanelSpec(
                    caption="PINN coefficient sensitivity by class",
                    candidates=_make_candidates(block_i_fig_dirs, ["pinn_coefficient_sensitivity_by_class.png"])
                    + _make_candidates(step7_fig_dirs, ["pinn_coefficient_sensitivity_by_class.png"]),
                ),
                PanelSpec(
                    caption="χ significance heatmap",
                    candidates=_make_candidates(block_a_fig_dirs, ["chi_class_significance_heatmap.png"])
                    + _make_candidates(step7_fig_dirs, ["chi_class_significance_heatmap.png"]),
                ),
                PanelSpec(
                    caption="Descriptor-χ correlation heatmap",
                    candidates=_make_candidates(block_f_fig_dirs, ["descriptor_chi_correlation_heatmap.png"])
                    + _make_candidates(step7_fig_dirs, ["descriptor_chi_correlation_heatmap.png"]),
                ),
            ],
        ),
        FigureSpec(
            figure_id="FigureS6",
            title="Figure S6. Thermodynamic Class Contrast and Target Context",
            destination="si",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="Class-contrast delta heatmap across (T, φ)",
                    candidates=_make_candidates(block_a_fig_dirs, ["chi_class_delta_heatmap.png"])
                    + _make_candidates(step7_fig_dirs, ["chi_class_delta_heatmap.png"]),
                ),
                PanelSpec(
                    caption="Class-contrast significance heatmap",
                    candidates=_make_candidates(block_a_fig_dirs, ["chi_class_significance_heatmap.png"])
                    + _make_candidates(step7_fig_dirs, ["chi_class_significance_heatmap.png"]),
                ),
                PanelSpec(
                    caption="χ vs temperature trajectories by class and φ",
                    candidates=_make_candidates(block_a_fig_dirs, ["chi_vs_temperature_by_phi_and_class.png"])
                    + _make_candidates(step7_fig_dirs, ["chi_vs_temperature_by_phi_and_class.png"]),
                ),
                PanelSpec(
                    caption="Target χ relative to class mean trends",
                    candidates=_make_candidates(block_a_fig_dirs, ["step3_target_vs_class_means.png"])
                    + _make_candidates(step7_fig_dirs, ["step3_target_vs_class_means.png"]),
                ),
            ],
        ),
        FigureSpec(
            figure_id="FigureS7",
            title="Figure S7. Predictor Error and Thermodynamic-Gradient Consistency",
            destination="si",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="Condition-level MAE heatmap",
                    candidates=_make_candidates(block_b_fig_dirs, ["step4_test_mae_heatmap.png"])
                    + _make_candidates(step7_fig_dirs, ["step4_test_mae_heatmap.png"]),
                ),
                PanelSpec(
                    caption="dχ/dT gradient consistency",
                    candidates=_make_candidates(block_b_fig_dirs, ["step4_gradient_consistency_dchi_dT.png"])
                    + _make_candidates(step7_fig_dirs, ["step4_gradient_consistency_dchi_dT.png"]),
                ),
                PanelSpec(
                    caption="dχ/dφ gradient consistency",
                    candidates=_make_candidates(block_b_fig_dirs, ["step4_gradient_consistency_dchi_dphi.png"])
                    + _make_candidates(step7_fig_dirs, ["step4_gradient_consistency_dchi_dphi.png"]),
                ),
                PanelSpec(
                    caption="Selection trade-off: confidence vs χ error",
                    candidates=_make_candidates(block_c_fig_dirs, ["selection_tradeoff_chi_vs_solubility_confidence.png"])
                    + _make_candidates(step7_fig_dirs, ["selection_tradeoff_chi_vs_solubility_confidence.png"]),
                ),
            ],
        ),
        FigureSpec(
            figure_id="FigureS8",
            title="Figure S8. Candidate Novelty and Chemical-Space Diagnostics",
            destination="si",
            ncols=2,
            panels=[
                PanelSpec(
                    caption="Descriptor shift relative to training set",
                    candidates=_make_candidates(block_c_fig_dirs, ["descriptor_shift_vs_training.png"])
                    + _make_candidates(step7_fig_dirs, ["descriptor_shift_vs_training.png"]),
                ),
                PanelSpec(
                    caption="Novelty similarity distribution",
                    candidates=_make_candidates(block_c_fig_dirs, ["novelty_similarity_histogram.png"])
                    + _make_candidates(step7_fig_dirs, ["novelty_similarity_histogram.png"]),
                ),
                PanelSpec(
                    caption="Scoring landscape: χ vs class confidence",
                    candidates=_make_candidates(block_g_fig_dirs, ["chi_vs_class_prob_scoring_landscape.png"])
                    + _make_candidates(step7_fig_dirs, ["chi_vs_class_prob_scoring_landscape.png"]),
                ),
                PanelSpec(
                    caption="Discovered-candidate descriptor distribution",
                    candidates=_make_candidates(block_g_fig_dirs, ["discovered_descriptor_boxplot.png"])
                    + _make_candidates(step7_fig_dirs, ["discovered_descriptor_boxplot.png"]),
                ),
            ],
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
        "- Figure 1: Establishes the training corpus scope and thermodynamic design targets",
        "- Figure 2: Validates backbone training convergence and generation quality",
        "- Figure 3: Shows how condition-specific χ_target thresholds are learned with statistical confidence",
        "- Figure 4: Demonstrates PINN χ regression accuracy and binary miscibility classification",
        "- Figure 5: Quantifies unconstrained inverse design coverage across thermodynamic conditions",
        "- Figure 6: Extends design to polymer-family constraints, maintaining coverage",
        "- Figure 7: Confirms novelty of discovered polymers and reveals thermodynamic mechanisms",
    ]
    (manuscript_text_dir / "manuscript_outline.md").write_text(
        "\n".join(manuscript_outline) + "\n",
        encoding="utf-8",
    )

    si_outline = [
        "# Supporting Information Outline (Step 8)",
        "",
        "## SI Figure Blocks",
        "1. Figure S1: Hyperparameter tuning trajectories and learning diagnostics.",
        "2. Figure S2: Inverse-design screening behavior, confidence-error relations, and funnel behavior.",
        "3. Figure S3: Extended chemistry/functional-group representativeness analysis.",
        "4. Figure S4: PINN polynomial and Flory-Huggins thermodynamic interpretation diagnostics.",
        "5. Figure S5: Embedding geometry, PINN sensitivity, and χ-dataset interpretation.",
        "6. Figure S6: Thermodynamic class contrast and target context analyses.",
        "7. Figure S7: Predictor error and thermodynamic-gradient consistency diagnostics.",
        "8. Figure S8: Candidate novelty and chemical-space diagnostics.",
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


def _write_verification_summary(
    manuscript_figures_dir: Path,
    si_figures_dir: Path,
    si_tables_dir: Path,
    metadata_dir: Path,
    panel_manifest: pd.DataFrame,
    specs: List[FigureSpec],
) -> Path:
    manuscript_specs = [s for s in specs if s.destination == "manuscript"]
    si_specs = [s for s in specs if s.destination == "si"]

    manuscript_pngs = sorted(manuscript_figures_dir.glob("*.png"))
    si_pngs = sorted(si_figures_dir.glob("*.png"))
    figure7_name = ""
    for s in manuscript_specs:
        if s.figure_id == "Figure7":
            figure7_name = _figure_output_name(s)
            break

    lines = [
        "Step 8 Verification Summary",
        "",
        f"manuscript_png_count: {len(manuscript_pngs)} (expected {len(manuscript_specs)})",
        f"supporting_information_png_count: {len(si_pngs)} (expected {len(si_specs)})",
        f"tableS4_exists: {int((si_tables_dir / 'tableS4_pinn_coefficients.csv').exists())}",
        f"figure7_expected_filename: {figure7_name}",
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
) -> Path:
    manuscript_specs = [s for s in specs if s.destination == "manuscript"]
    si_specs = [s for s in specs if s.destination == "si"]
    manuscript_pngs = sorted(manuscript_figures_dir.glob("*.png"))
    si_pngs = sorted(si_figures_dir.glob("*.png"))
    manuscript_tables = sorted(manuscript_tables_dir.glob("*.csv"))
    si_tables = sorted(si_tables_dir.glob("*.csv"))
    found_panels = int((panel_manifest["status"] == "ok").sum()) if not panel_manifest.empty else 0
    missing_panels = int((panel_manifest["status"] != "ok").sum()) if not panel_manifest.empty else 0

    lines = [
        "# Step 8 Status Report",
        "",
        "## Summary",
        "- Status: implemented and verified from generated artifacts",
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
    s1 = _safe_first_row(paths.get("step1_summary"))
    s2 = _safe_first_row(paths.get("step2_summary"))
    s3 = _safe_first_row(paths.get("step3_summary"))
    s4 = _safe_first_row(paths.get("step4_summary"))
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
            "secondary_value": _pick(s3, ["global_test_balanced_accuracy"]),
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
            "mean_top1_abs_error": _pick(s5, ["mean_top1_abs_error"]),
            "target_polymer_diversity": _pick(s5, ["target_polymer_diversity"]),
            "target_polymer_mean_sa": _pick(s5, ["target_polymer_mean_sa"]),
        },
        {
            "workflow": "Step6_class_conditioned",
            "target_polymer_selection_success_rate": _pick(s6, ["target_polymer_selection_success_rate"]),
            "target_success_rate": _pick(s6, ["target_success_rate"]),
            "mean_top1_abs_error": _pick(s6, ["mean_top1_abs_error"]),
            "target_polymer_diversity": _pick(s6, ["target_polymer_diversity"]),
            "target_polymer_mean_sa": _pick(s6, ["target_polymer_mean_sa"]),
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
            if name in {"results_dir", "base_results_dir"}:
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

    step5_targets = (
        paths["step5_dir"] / "metrics" / "target_polymers.csv" if paths.get("step5_dir") is not None else None
    )
    step6_targets = (
        paths["step6_dir"] / "metrics" / "target_polymers.csv" if paths.get("step6_dir") is not None else None
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
        paths["step4_reg_dir"] / "metrics" / "chi_coefficients.csv"
        if paths.get("step4_reg_dir") is not None
        else None
    )
    table_s4_path = si_tables_dir / "tableS4_pinn_coefficients.csv"
    coeff_df = _safe_read_csv(coeff_path)
    if coeff_df.empty:
        pd.DataFrame(
            [{"note": "Step4 chi_coefficients.csv not found for this run."}]
        ).to_csv(table_s4_path, index=False)
    else:
        coeff_df.to_csv(table_s4_path, index=False)

    return {
        "table1_pipeline_key_metrics": table1_path,
        "table2_inverse_design_comparison": table2_path,
        "tableS1_input_artifact_status": si_table1_path,
        "tableS2_step5_target_polymers_top50": table_s2_path,
        "tableS3_step6_target_polymers_top50": table_s3_path,
        "tableS4_pinn_coefficients": table_s4_path,
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
    if paths.get("step6_dir") is not None:
        to_copy["step6_inverse_aggregate_metrics"] = (
            paths["step6_dir"] / "metrics" / "inverse_aggregate_metrics.csv"
        )
        to_copy["step6_target_polymers"] = paths["step6_dir"] / "metrics" / "target_polymers.csv"
    if paths.get("step7_dir") is not None:
        to_copy["step7_rollup"] = (
            paths["step7_dir"] / "metrics" / "step1_to_step6_summary_rollup.csv"
        )
        to_copy["step7_discovered_descriptor_stats"] = (
            paths["step7_dir"] / "metrics" / "discovered_descriptor_stats.csv"
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

    results_dir = Path(get_results_dir(model_size, config["paths"]["results_dir"], split_mode=None))
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

    # Build a robust condition-wise objective panel for Figure 3(b) from Step 3 metrics.
    paths["step3_balanced_accuracy_heatmap_derived"] = _build_step3_objective_heatmap_from_metrics(
        paths=paths,
        metadata_dir=metadata_dir,
    )
    paths["step3_bootstrap_uncertainty_heatmap_derived"] = _build_step3_bootstrap_uncertainty_heatmap(
        paths=paths,
        metadata_dir=metadata_dir,
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
    )

    missing_panels = int((panel_manifest["status"] != "ok").sum()) if not panel_manifest.empty else 0
    found_panels = int((panel_manifest["status"] == "ok").sum()) if not panel_manifest.empty else 0
    copied_source_files = int(source_manifest["copied"].sum()) if not source_manifest.empty else 0

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
