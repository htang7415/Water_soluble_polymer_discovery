"""Utility helpers to write standardized per-step reports."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd


def _to_scalar(value):
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def save_step_summary(summary: Dict, metrics_dir: Path, filename: str = "step_summary.csv") -> Path:
    """Save one-row step summary table."""
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    row = {k: _to_scalar(v) for k, v in summary.items()}
    out_csv = metrics_dir / filename
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    return out_csv


def write_initial_log(
    step_dir: Path,
    step_name: str,
    context: Dict | None = None,
    filename: str = "log.txt",
) -> Path:
    """Write an initial run log file at step startup."""
    step_dir = Path(step_dir)
    step_dir.mkdir(parents=True, exist_ok=True)
    out_path = step_dir / filename

    lines = [
        f"step: {step_name}",
        f"start_time_utc: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
    ]

    if context:
        for k in sorted(context.keys()):
            lines.append(f"{k}: {_to_scalar(context[k])}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def save_artifact_manifest(step_dir: Path, metrics_dir: Path, figures_dir: Path | None = None) -> Dict[str, Path]:
    """Save artifact manifest and simple artifact-count overview files.

    Outputs:
    - artifact_manifest.csv
    - artifact_counts_by_type.csv
    - artifact_counts_by_category.csv
    - artifact_counts_by_category.png (if figures_dir provided)
    """
    step_dir = Path(step_dir)
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(step_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(step_dir)
        parts = rel.parts
        if parts and parts[0] == "metrics":
            category = "metrics"
        elif parts and parts[0] == "figures":
            category = "figures"
        elif parts and parts[0] == "checkpoints":
            category = "checkpoints"
        else:
            category = "other"

        rows.append(
            {
                "relative_path": str(rel),
                "file_name": p.name,
                "category": category,
                "suffix": p.suffix.lower(),
                "size_bytes": int(p.stat().st_size),
            }
        )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        manifest = pd.DataFrame(columns=["relative_path", "file_name", "category", "suffix", "size_bytes"])

    manifest_csv = metrics_dir / "artifact_manifest.csv"
    manifest.to_csv(manifest_csv, index=False)

    by_type = (
        manifest.groupby(["category", "suffix"], as_index=False)
        .size()
        .rename(columns={"size": "n_files"})
        .sort_values(["category", "n_files"], ascending=[True, False])
    )
    by_type_csv = metrics_dir / "artifact_counts_by_type.csv"
    by_type.to_csv(by_type_csv, index=False)

    by_cat = (
        manifest.groupby("category", as_index=False)
        .size()
        .rename(columns={"size": "n_files"})
        .sort_values("n_files", ascending=False)
    )
    by_cat_csv = metrics_dir / "artifact_counts_by_category.csv"
    by_cat.to_csv(by_cat_csv, index=False)

    outputs = {
        "artifact_manifest": manifest_csv,
        "artifact_counts_by_type": by_type_csv,
        "artifact_counts_by_category": by_cat_csv,
    }

    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig_path = figures_dir / "artifact_counts_by_category.png"
        try:
            fig, ax = plt.subplots(figsize=(5, 4))
            if not by_cat.empty:
                ax.bar(by_cat["category"], by_cat["n_files"], color="#4c78a8")
                for i, v in enumerate(by_cat["n_files"].tolist()):
                    ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
            ax.set_xlabel("Category")
            ax.set_ylabel("File count")
            ax.set_title("Artifact counts by category")
            fig.tight_layout()
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            outputs["artifact_counts_figure"] = fig_path
        except Exception as exc:
            # Keep pipeline robust even if plotting backend is unavailable.
            (metrics_dir / "artifact_counts_figure_error.txt").write_text(str(exc))

    return outputs
