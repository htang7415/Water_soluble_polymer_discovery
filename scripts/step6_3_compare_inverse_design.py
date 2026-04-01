#!/usr/bin/env python
"""Compare Step 6_2 inverse-design runs for one target class."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.step6_2.config import load_step6_2_config
from src.step6_2.study_families import STUDY_BASE_RUNS
from src.step6_2.plotting import (
    plot_overall_success_all_runs,
    plot_overall_success_by_family,
    plot_per_target_difficulty_ranked,
    plot_per_target_success_compare,
    plot_success_gate_funnel_compare,
    plot_success_vs_oracle_budget,
)
from src.utils.reporting import save_artifact_manifest, write_initial_log


REQUIRED_RUN_FILES = [
    "metrics/method_metrics.json",
    "metrics/round_metrics.csv",
    "metrics/target_row_summary.csv",
    "metrics/evaluation_results.csv",
]


def _as_yamlable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _as_yamlable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_yamlable(v) for v in value]
    return value


def _resolve_selected_runs(resolved, runs_arg: str | None) -> List[str]:
    existing_run_dirs = {
        path.name
        for path in resolved.benchmark_root.iterdir()
        if path.is_dir() and path.name != "_shared"
    } if resolved.benchmark_root.exists() else set()
    if runs_arg:
        requested = [item.strip() for item in str(runs_arg).split(",") if item.strip()]
        allowed = set(resolved.enabled_runs) | existing_run_dirs
        unknown = [run for run in requested if run not in allowed]
        if unknown:
            raise ValueError(
                "Requested Step 6_3 runs are neither enabled in config6_2 nor present under the "
                f"Step 6_2 benchmark root: {unknown}"
            )
        return requested
    if bool(resolved.step6_3.get("compare_all_enabled_runs", True)):
        if bool(resolved.step6_2_hpo.get("enabled", False)):
            preferred_runs: List[str] = []
            if "S0_raw_unconditional" in resolved.enabled_runs and "S0_raw_unconditional" in existing_run_dirs:
                preferred_runs.append("S0_raw_unconditional")
            for base_run_name in STUDY_BASE_RUNS.values():
                tuned_name = f"{base_run_name}_optuna"
                if tuned_name in existing_run_dirs:
                    preferred_runs.append(tuned_name)
            if preferred_runs:
                return preferred_runs
        return [run for run in resolved.enabled_runs if run in existing_run_dirs]
    return [run for run in resolved.enabled_runs if (resolved.benchmark_root / run).exists()]


def _load_run_outputs(
    resolved,
    *,
    run_name: str,
) -> Dict[str, object]:
    run_dir = resolved.benchmark_root / run_name
    missing = [path for path in REQUIRED_RUN_FILES if not (run_dir / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Run {run_name} is missing required Step 6_2 outputs: {missing}")

    with open(run_dir / "metrics" / "method_metrics.json", "r", encoding="utf-8") as handle:
        method_metrics = json.load(handle)
    round_metrics_df = pd.read_csv(run_dir / "metrics" / "round_metrics.csv")
    target_row_summary_df = pd.read_csv(run_dir / "metrics" / "target_row_summary.csv")
    evaluation_results_df = pd.read_csv(run_dir / "metrics" / "evaluation_results.csv")
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "method_metrics": method_metrics,
        "round_metrics_df": round_metrics_df,
        "target_row_summary_df": target_row_summary_df,
        "evaluation_results_df": evaluation_results_df,
    }


def _build_run_comparison_row(run_payload: Dict[str, object]) -> Dict[str, object]:
    method_metrics = dict(run_payload["method_metrics"])
    round_metrics_df = run_payload["round_metrics_df"]
    evaluation_results_df = run_payload["evaluation_results_df"]

    row = {
        "run_name": str(method_metrics["run_name"]),
        "canonical_family": str(method_metrics["canonical_family"]),
        "mean_success_hit_rate": float(method_metrics["mean_success_hit_rate"]),
        "std_success_hit_rate": float(method_metrics["std_success_hit_rate"]),
        "macro_average_row_mean_success_hit_rate": float(method_metrics["macro_average_row_mean_success_hit_rate"]),
        "mean_benchmark_soluble_oracle_calls": float(method_metrics.get("mean_benchmark_soluble_oracle_calls", 0.0)),
        "mean_benchmark_chi_oracle_calls": float(method_metrics.get("mean_benchmark_chi_oracle_calls", 0.0)),
        "mean_training_soluble_oracle_calls": float(method_metrics.get("mean_training_soluble_oracle_calls", 0.0)),
        "mean_training_chi_oracle_calls": float(method_metrics.get("mean_training_chi_oracle_calls", 0.0)),
        "mean_class_guidance_suppressed_steps": float(method_metrics.get("mean_class_guidance_suppressed_steps", 0.0)),
    }
    for gate in ["valid_ok", "novel_ok", "star_ok", "sa_ok", "soluble_ok", "class_ok", "chi_ok"]:
        col = f"mean_{gate}_rate"
        row[col] = float(round_metrics_df[col].mean()) if col in round_metrics_df.columns and not round_metrics_df.empty else float("nan")

    row["num_rounds"] = int(round_metrics_df["round_id"].nunique()) if not round_metrics_df.empty else 0
    row["num_generated_samples"] = int(len(evaluation_results_df))
    row["mean_total_oracle_calls"] = (
        row["mean_benchmark_soluble_oracle_calls"]
        + row["mean_benchmark_chi_oracle_calls"]
        + row["mean_training_soluble_oracle_calls"]
        + row["mean_training_chi_oracle_calls"]
    )
    return row


def _build_per_target_run_comparison(run_payloads: List[Dict[str, object]]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for payload in run_payloads:
        df = payload["target_row_summary_df"].copy()
        keep = [
            "run_name",
            "canonical_family",
            "target_row_id",
            "target_row_key",
            "c_target",
            "temperature",
            "phi",
            "chi_target",
            "mean_valid_ok_rate",
            "mean_novel_ok_rate",
            "mean_star_ok_rate",
            "mean_sa_ok_rate",
            "mean_soluble_ok_rate",
            "mean_class_ok_rate",
            "mean_chi_ok_rate",
            "mean_success_hit_rate",
            "std_success_hit_rate",
            "num_rounds",
        ]
        rows.append(df[keep].copy())
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["target_row_id", "run_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_difficulty_summary(per_target_run_df: pd.DataFrame) -> pd.DataFrame:
    if per_target_run_df.empty:
        return pd.DataFrame()
    group_cols = ["target_row_id", "target_row_key", "c_target", "temperature", "phi", "chi_target"]
    rows: List[Dict[str, object]] = []
    for keys, sub in per_target_run_df.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, keys)}
        row["num_runs"] = int(sub["run_name"].nunique())
        row["mean_success_hit_rate_across_runs"] = float(sub["mean_success_hit_rate"].mean())
        row["std_success_hit_rate_across_runs"] = float(sub["mean_success_hit_rate"].std(ddof=0))
        row["mean_class_ok_rate_across_runs"] = float(sub["mean_class_ok_rate"].mean())
        row["mean_soluble_ok_rate_across_runs"] = float(sub["mean_soluble_ok_rate"].mean())
        row["mean_chi_ok_rate_across_runs"] = float(sub["mean_chi_ok_rate"].mean())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(
        ["mean_success_hit_rate_across_runs", "target_row_id"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    out["difficulty_rank"] = range(1, len(out) + 1)
    return out


def _build_canonical_family_comparison(run_comparison_df: pd.DataFrame) -> pd.DataFrame:
    if run_comparison_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for canonical_family, sub in run_comparison_df.groupby("canonical_family", sort=True):
        ordered = sub.sort_values(
            ["mean_success_hit_rate", "run_name"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        best = ordered.iloc[0]
        rows.append(
            {
                "canonical_family": str(canonical_family),
                "num_runs_compared": int(len(sub)),
                "best_run_name": str(best["run_name"]),
                "best_mean_success_hit_rate": float(best["mean_success_hit_rate"]),
                "best_std_success_hit_rate": float(best["std_success_hit_rate"]),
                "family_mean_success_hit_rate": float(sub["mean_success_hit_rate"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["best_mean_success_hit_rate", "canonical_family"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _write_compare_outputs(
    *,
    resolved,
    compare_root: Path,
    selected_runs: List[str],
    run_comparison_df: pd.DataFrame,
    per_target_run_df: pd.DataFrame,
    difficulty_df: pd.DataFrame,
    canonical_family_df: pd.DataFrame,
    partial_compare: bool,
    skipped_runs: List[str],
    config_path: str,
) -> None:
    metrics_dir = compare_root / "metrics"
    figures_dir = compare_root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_comparison_df.to_csv(metrics_dir / "run_comparison.csv", index=False)
    per_target_run_df.to_csv(metrics_dir / "per_target_run_comparison.csv", index=False)
    difficulty_df.to_csv(metrics_dir / "per_target_difficulty_summary.csv", index=False)
    canonical_family_df.to_csv(metrics_dir / "canonical_family_comparison.csv", index=False)

    best_run = run_comparison_df.iloc[0].to_dict() if not run_comparison_df.empty else {}
    payload = {
        "config_path": config_path,
        "compare_root": str(compare_root),
        "c_target": resolved.c_target,
        "split_mode": resolved.split_mode,
        "model_size": resolved.model_size,
        "partial_compare": bool(partial_compare),
        "selected_runs": selected_runs,
        "skipped_runs": skipped_runs,
        "best_run_overall": best_run,
        "best_run_by_family": canonical_family_df.to_dict(orient="records"),
    }
    with open(metrics_dir / "run_comparison.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    snapshot = {
        "model_size": resolved.model_size,
        "split_mode": resolved.split_mode,
        "classification_split_mode": resolved.classification_split_mode,
        "c_target": resolved.c_target,
        "config_path": config_path,
        "compare_all_enabled_runs": bool(resolved.step6_3.get("compare_all_enabled_runs", True)),
        "summarize_by_canonical_family": bool(resolved.step6_3.get("summarize_by_canonical_family", True)),
        "selected_runs": selected_runs,
        "skipped_runs": skipped_runs,
        "partial_compare": bool(partial_compare),
        "benchmark_root": str(resolved.benchmark_root),
        "compare_root": str(compare_root),
    }
    with open(compare_root / "config_snapshot.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(_as_yamlable(snapshot), handle, sort_keys=False)

    font_size = int(resolved.step6_2["figure_font_size"])
    if not run_comparison_df.empty:
        plot_overall_success_all_runs(
            run_comparison_df,
            figures_dir / "overall_success_hit_rate_all_runs.png",
            font_size=font_size,
        )
        plot_success_gate_funnel_compare(
            run_comparison_df,
            figures_dir / "success_gate_funnel_compare_all_runs.png",
            font_size=font_size,
        )
        plot_success_vs_oracle_budget(
            run_comparison_df,
            figures_dir / "success_hit_vs_oracle_budget_all_runs.png",
            font_size=font_size,
        )
    if not canonical_family_df.empty:
        plot_overall_success_by_family(
            canonical_family_df,
            figures_dir / "overall_success_hit_rate_by_family.png",
            font_size=font_size,
        )
    if not per_target_run_df.empty:
        plot_per_target_success_compare(
            per_target_run_df,
            figures_dir / "per_target_success_hit_rate_compare_all_runs.png",
            font_size=font_size,
        )
    if not difficulty_df.empty:
        plot_per_target_difficulty_ranked(
            difficulty_df,
            figures_dir / "per_target_difficulty_ranked.png",
            font_size=font_size,
        )

    write_initial_log(
        compare_root,
        "step6_3_compare_inverse_design",
        context={
            "config_path": config_path,
            "model_size": resolved.model_size,
            "split_mode": resolved.split_mode,
            "c_target": resolved.c_target,
            "partial_compare": bool(partial_compare),
            "selected_runs": ",".join(selected_runs),
            "skipped_runs": ",".join(skipped_runs),
        },
    )
    save_artifact_manifest(compare_root, metrics_dir, figures_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Step 6_2 inverse-design runs.")
    parser.add_argument("--config", default="configs/config6_2.yaml")
    parser.add_argument("--base_config", default="configs/config.yaml")
    parser.add_argument("--model_size", default=None)
    parser.add_argument("--runs", default=None, help="Comma-separated subset of enabled runs to compare.")
    parser.add_argument("--allow_partial", action="store_true", help="Skip missing/incomplete runs for development.")
    args = parser.parse_args()

    resolved = load_step6_2_config(
        config_path=args.config,
        base_config_path=args.base_config,
        model_size=args.model_size,
    )
    selected_runs = _resolve_selected_runs(resolved, args.runs)

    run_payloads: List[Dict[str, object]] = []
    skipped_runs: List[str] = []
    for run_name in selected_runs:
        try:
            run_payloads.append(_load_run_outputs(resolved, run_name=run_name))
        except FileNotFoundError:
            if not args.allow_partial:
                raise
            skipped_runs.append(run_name)

    if not run_payloads:
        raise ValueError("No completed Step 6_2 runs are available for Step 6_3 comparison.")

    run_rows = [_build_run_comparison_row(payload) for payload in run_payloads]
    run_comparison_df = pd.DataFrame(run_rows).sort_values(
        ["mean_success_hit_rate", "run_name"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    run_comparison_df["rank"] = range(1, len(run_comparison_df) + 1)

    per_target_run_df = _build_per_target_run_comparison(run_payloads)
    difficulty_df = _build_difficulty_summary(per_target_run_df)
    canonical_family_df = _build_canonical_family_comparison(run_comparison_df)

    compare_root = resolved.compare_root
    _write_compare_outputs(
        resolved=resolved,
        compare_root=compare_root,
        selected_runs=[payload["run_name"] for payload in run_payloads],
        run_comparison_df=run_comparison_df,
        per_target_run_df=per_target_run_df,
        difficulty_df=difficulty_df,
        canonical_family_df=canonical_family_df,
        partial_compare=bool(args.allow_partial and skipped_runs),
        skipped_runs=skipped_runs,
        config_path=args.config,
    )

    print(f"Step 6_3 comparison written to: {compare_root}")
    print(f"Compared runs ({len(run_payloads)}): {[payload['run_name'] for payload in run_payloads]}")
    if skipped_runs:
        print(f"Skipped runs ({len(skipped_runs)}): {skipped_runs}")


if __name__ == "__main__":
    main()
