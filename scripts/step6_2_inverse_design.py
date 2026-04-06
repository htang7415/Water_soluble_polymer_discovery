#!/usr/bin/env python
"""Step 6_2 inverse-design driver."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.step6_2.config import build_run_config, load_step6_2_config
from src.step6_2.run_core import execute_step62_run


SUPPORTED_RUNS = {
    "S0_raw_unconditional",
    "S1_guided_frozen",
    "S2_conditional",
    "S2_cfg_0p0",
    "S2_cfg_2p0",
    "S2_mt",
    "S3_conditional_guided",
    "S3_cfg_2p0",
    "S4_dpo",
    "S4_rl_finetuned",
    "S4_ppo",
    "S4_grpo",
}


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _as_yamlable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _as_yamlable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_yamlable(v) for v in value]
    return value


def _prepare_shared_artifacts(resolved, *, config_path: str) -> None:
    shared_dir = resolved.benchmark_root / "_shared"
    metrics_dir = shared_dir / "metrics"
    shared_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(shared_dir / "config_snapshot.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(_as_yamlable(resolved.config_snapshot), handle, sort_keys=False)

    _write_frame(resolved.target_base_df, metrics_dir / "d_target_base.csv")
    _write_frame(resolved.target_family_df, metrics_dir / "d_target_family.csv")
    _write_frame(resolved.rl_proxy_df, metrics_dir / "rl_proxy_targets.csv")
    if not resolved.hpo_target_df.empty:
        _write_frame(resolved.hpo_target_df, metrics_dir / "d_hpo_family.csv")
    _write_frame(resolved.chi_split_df, metrics_dir / "d_chi_with_split.csv")

    run_rows = [build_run_config(resolved, run_name) for run_name in resolved.enabled_runs]
    run_manifest = pd.DataFrame(
        [
            {
                "run_name": row["run_name"],
                "canonical_family": row["canonical_family"],
                "cfg_scale": row.get("s2", {}).get(
                    "cfg_scale",
                    row.get("s3", {}).get("cfg_scale", row.get("s4", {}).get("cfg_scale")),
                ),
                "alignment_mode": row.get("s4", {}).get("alignment_mode", ""),
            }
            for row in run_rows
        ]
    )
    _write_frame(run_manifest, metrics_dir / "enabled_runs.csv")
    _write_json(
        {
            "config_path": config_path,
            "c_target": resolved.c_target,
            "split_mode": resolved.split_mode,
            "classification_split_mode": resolved.classification_split_mode,
            "model_size": resolved.model_size,
            "num_target_rows": int(len(resolved.target_family_df)),
            "num_rl_proxy_rows": int(len(resolved.rl_proxy_df)),
            "num_hpo_rows": int(len(resolved.hpo_target_df)),
            "enabled_runs": resolved.enabled_runs,
            "benchmark_root": str(resolved.benchmark_root),
        },
        metrics_dir / "prepare_summary.json",
    )


def _resolve_requested_runs(resolved, runs_arg: str | None, allow_partial: bool) -> List[str]:
    if runs_arg:
        requested = [run.strip() for run in runs_arg.split(",") if run.strip()]
        unknown = [run for run in requested if run not in resolved.enabled_runs]
        if unknown:
            raise ValueError(f"Requested runs are not enabled in config6_2: {unknown}")
        selected = requested
    else:
        selected = list(resolved.enabled_runs)

    unsupported = [run for run in selected if run not in SUPPORTED_RUNS]
    if unsupported and not allow_partial:
        raise NotImplementedError(
            "This implementation increment supports only the currently wired Step 6_2 runs. "
            f"Unsupported selected runs: {unsupported}. "
            "Use --runs with a supported subset or pass --allow_partial."
        )
    if unsupported and allow_partial:
        print(f"Skipping unsupported runs in this increment: {unsupported}")
        selected = [run for run in selected if run in SUPPORTED_RUNS]
    if not selected:
        raise ValueError("No supported runs selected.")
    return selected


def _execute_run(
    *,
    resolved,
    run_name: str,
    device: str,
    config_path: str,
) -> None:
    execute_step62_run(
        resolved=resolved,
        run_name=run_name,
        device=device,
        config_path=config_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 6_2 inverse design.")
    parser.add_argument("--config", default="configs/config6_2.yaml")
    parser.add_argument("--base_config", default="configs/config.yaml")
    parser.add_argument("--model_size", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--runs", default=None, help="Comma-separated subset of enabled runs.")
    parser.add_argument("--allow_partial", action="store_true", help="Skip unsupported runs for development.")
    args = parser.parse_args()

    resolved = load_step6_2_config(
        config_path=args.config,
        base_config_path=args.base_config,
        model_size=args.model_size,
    )
    _prepare_shared_artifacts(resolved, config_path=args.config)

    print(f"Prepared Step 6_2 shared artifacts under: {resolved.benchmark_root / '_shared'}")
    print(f"c_target={resolved.c_target} | num_target_rows={len(resolved.target_family_df)}")
    print("Enabled runs:")
    for run_name in resolved.enabled_runs:
        print(f"  - {run_name}")

    if args.prepare_only:
        return

    selected_runs = _resolve_requested_runs(resolved, args.runs, args.allow_partial)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    for run_name in selected_runs:
        _execute_run(
            resolved=resolved,
            run_name=run_name,
            device=device,
            config_path=args.config,
        )


if __name__ == "__main__":
    main()
