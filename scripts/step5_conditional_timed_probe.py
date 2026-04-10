#!/usr/bin/env python
"""Timed Step 5 conditional sampler probe for per-family DiT checks."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.polymer_class import PolymerClassifier
from src.step5.conditional_sampling import create_conditional_sampler, sample_conditional_with_class_prior
from src.step5.config import load_step5_config
from src.step5.dataset import build_inference_condition_bundle
from src.step5.frozen_sampling import (
    ClassConstrainedSamplingQuotaError,
    _sample_raw_smiles_with_prior,
    resolve_class_sampling_prior,
)
from src.step5.run_core import (
    _load_s2_training_artifacts_from_existing_checkpoint,
    create_run_dirs,
    save_run_config_snapshot,
)
from src.utils.chemistry import canonicalize_smiles, check_validity, count_stars, has_terminal_connection_stars
from src.utils.reproducibility import seed_everything


def _parse_int_csv(raw: Optional[str], *, default: List[int]) -> List[int]:
    if raw is None or str(raw).strip() == "":
        return [int(value) for value in default]
    return [int(token.strip()) for token in str(raw).split(",") if token.strip()]


def _parse_optional_bool(raw: Optional[str]) -> Optional[bool]:
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not parse optional boolean value: {raw!r}")


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_temp_yaml(payload: Dict[str, Any]) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        return handle.name


def _build_temp_step5_config(
    *,
    config_path: str,
    c_target: str,
    enabled_runs: List[str],
) -> str:
    payload = deepcopy(_load_yaml(config_path))
    step5_cfg = payload.setdefault("step5", {})
    step5_cfg["c_target"] = str(c_target)
    step5_cfg["enabled_runs"] = [str(name) for name in enabled_runs]
    return _write_temp_yaml(payload)


def _build_temp_base_config(
    *,
    base_config_path: str,
    c_target: str,
    override_length_prior_min_tokens: Optional[int] = None,
    override_length_prior_max_tokens: Optional[int] = None,
    override_class_match_min_request_size: Optional[int] = None,
    override_terminal_star_anchor: Optional[bool] = None,
    override_backbone_template_max_templates: Optional[int] = None,
    override_cycle_backbone_template_cores_across_targets: Optional[bool] = None,
) -> str:
    payload = deepcopy(_load_yaml(base_config_path))
    chi_training_cfg = payload.setdefault("chi_training", {})
    step5_cfg = chi_training_cfg.setdefault("step5_inverse_design", {})
    target_key = str(c_target).strip().lower()
    if override_length_prior_min_tokens is not None:
        overrides = dict(step5_cfg.get("decode_constraint_length_prior_min_tokens_overrides", {}) or {})
        overrides[target_key] = int(override_length_prior_min_tokens)
        step5_cfg["decode_constraint_length_prior_min_tokens_overrides"] = overrides
    if override_length_prior_max_tokens is not None:
        overrides = dict(step5_cfg.get("decode_constraint_length_prior_max_tokens_overrides", {}) or {})
        overrides[target_key] = int(override_length_prior_max_tokens)
        step5_cfg["decode_constraint_length_prior_max_tokens_overrides"] = overrides
    if override_class_match_min_request_size is not None:
        overrides = dict(step5_cfg.get("decode_constraint_class_match_min_request_size_overrides", {}) or {})
        overrides[target_key] = int(override_class_match_min_request_size)
        step5_cfg["decode_constraint_class_match_min_request_size_overrides"] = overrides
    if override_terminal_star_anchor is not None:
        overrides = dict(step5_cfg.get("decode_constraint_backbone_template_terminal_star_anchor_overrides", {}) or {})
        overrides[target_key] = bool(override_terminal_star_anchor)
        step5_cfg["decode_constraint_backbone_template_terminal_star_anchor_overrides"] = overrides
    if override_backbone_template_max_templates is not None:
        overrides = dict(step5_cfg.get("decode_constraint_backbone_template_max_templates_overrides", {}) or {})
        overrides[target_key] = int(override_backbone_template_max_templates)
        step5_cfg["decode_constraint_backbone_template_max_templates_overrides"] = overrides
    if override_cycle_backbone_template_cores_across_targets is not None:
        overrides = dict(
            step5_cfg.get("decode_constraint_cycle_backbone_template_cores_across_targets_overrides", {}) or {}
        )
        overrides[target_key] = bool(override_cycle_backbone_template_cores_across_targets)
        step5_cfg["decode_constraint_cycle_backbone_template_cores_across_targets_overrides"] = overrides
    return _write_temp_yaml(payload)


def _find_latest_conditional_checkpoint(results_dir: Path, family: str) -> Optional[Path]:
    family_root = results_dir / "step5_inverse_design" / "polymer" / str(family).strip().lower()
    if not family_root.exists():
        return None
    candidates = [path for path in family_root.rglob("conditional_diffusion_best.pt") if "_warm_start" not in path.parts]
    if not candidates:
        return None
    candidates.sort(key=lambda path: (path.stat().st_mtime, -len(path.parts), str(path)), reverse=True)
    return candidates[0]


def _build_condition_bundle(artifacts, target_row: pd.Series, *, device: str) -> torch.Tensor:
    chi_goal = float(target_row["chi_target"] if "chi_target" in target_row.index else target_row["chi"])
    return torch.tensor(
        build_inference_condition_bundle(
            temperature=float(target_row["temperature"]),
            phi=float(target_row["phi"]),
            chi_goal=chi_goal,
            scaler=artifacts.scaler,
            soluble=1,
        ),
        dtype=torch.float32,
        device=device,
    )


def _quality_stats(smiles_list: List[str], *, classifier: PolymerClassifier, family: str) -> Dict[str, Any]:
    valid_flags = [bool(check_validity(smiles)) for smiles in smiles_list]
    class_flags = [
        bool(valid and classifier.classify(smiles).get(str(family).strip().lower(), False))
        for smiles, valid in zip(smiles_list, valid_flags)
    ]
    star_flags = [
        bool(valid and count_stars(smiles) == 2 and has_terminal_connection_stars(smiles, expected_stars=2))
        for smiles, valid in zip(smiles_list, valid_flags)
    ]
    canonicals = [canonicalize_smiles(smiles) if valid else None for smiles, valid in zip(smiles_list, valid_flags)]
    return {
        "n": int(len(smiles_list)),
        "valid_rate": float(sum(valid_flags) / len(valid_flags)) if valid_flags else 0.0,
        "class_rate": float(sum(class_flags) / len(class_flags)) if class_flags else 0.0,
        "star_rate": float(sum(star_flags) / len(star_flags)) if star_flags else 0.0,
        "unique_valid_canonical": int(len({canonical for canonical in canonicals if canonical})),
        "first_smiles": str(smiles_list[0]) if smiles_list else "",
    }


def _probe_acceptance(
    *,
    sampler,
    tokenizer,
    prior,
    resolved,
    generation_budget: int,
    sampling_state: Dict[str, object] | None = None,
) -> Dict[str, Any]:
    try:
        smiles, meta = sample_conditional_with_class_prior(
            sampler=sampler,
            tokenizer=tokenizer,
            prior=prior,
            resolved=resolved,
            num_samples=int(generation_budget),
            show_progress=False,
            sampling_state=sampling_state,
        )
        quota_ok = 1
    except ClassConstrainedSamplingQuotaError as exc:
        smiles = list(exc.accepted_smiles)
        meta = dict(exc.metadata)
        quota_ok = 0
    return {
        "quota_ok": int(quota_ok),
        "returned_count": int(len(smiles)),
        "raw_draws": int(meta.get("total_raw_samples_drawn", 0)),
        "attempts": int(meta.get("class_match_sampling_attempts", 0)),
        "acceptance_rate": float(meta.get("class_match_acceptance_rate", 0.0)),
        "total_wall_time_seconds": meta.get("class_match_total_wall_time_seconds"),
        "raw_sampling_wall_time_seconds": meta.get("class_match_total_raw_sampling_wall_time_seconds"),
        "filter_wall_time_seconds": meta.get("class_match_total_filter_wall_time_seconds"),
        "wall_time_seconds_per_raw_draw": meta.get("class_match_wall_time_seconds_per_raw_draw"),
        "attempt_log": meta.get("attempt_log", []),
        "smiles": smiles,
    }


def _probe_raw(
    *,
    sampler,
    tokenizer,
    prior,
    resolved,
    num_samples: int,
    sampling_state: Dict[str, object] | None = None,
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    smiles, raw_meta = _sample_raw_smiles_with_prior(
        sampler=sampler,
        tokenizer=tokenizer,
        prior=prior,
        resolved=resolved,
        num_samples=int(num_samples),
        show_progress=False,
        sampling_state=sampling_state,
    )
    wall_time_seconds = time.perf_counter() - start_time
    return {
        "quota_ok": 1,
        "returned_count": int(len(smiles)),
        "raw_draws": int(len(smiles)),
        "attempts": 1,
        "acceptance_rate": None,
        "total_wall_time_seconds": float(wall_time_seconds),
        "raw_sampling_wall_time_seconds": float(wall_time_seconds),
        "filter_wall_time_seconds": 0.0,
        "wall_time_seconds_per_raw_draw": (
            float(wall_time_seconds) / float(len(smiles)) if smiles else None
        ),
        "attempt_log": [],
        "sampled_lengths": [int(length) for length in raw_meta.get("sampled_lengths", [])],
        "smiles": smiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Timed Step 5 conditional sampler probe.")
    parser.add_argument("--config", default="configs/config5_smoke.yaml")
    parser.add_argument("--base_config", default="configs/config.yaml")
    parser.add_argument("--model_size", default="small")
    parser.add_argument("--family", required=True)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--probe_mode", choices=["acceptance", "raw"], default="acceptance")
    parser.add_argument("--target_row_index", type=int, default=0)
    parser.add_argument("--target_row_indices", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--generation_budget", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=0.5)
    parser.add_argument("--candidate_early_frac", type=float, default=0.75)
    parser.add_argument("--candidate_ramp_start_frac", type=float, default=0.6)
    parser.add_argument("--candidate_retry_multiplier", type=float, default=1.25)
    parser.add_argument("--override_length_prior_min_tokens", type=int, default=None)
    parser.add_argument("--override_length_prior_max_tokens", type=int, default=None)
    parser.add_argument("--override_class_match_min_request_size", type=int, default=None)
    parser.add_argument("--override_terminal_star_anchor", default=None)
    parser.add_argument("--override_backbone_template_max_templates", type=int, default=None)
    parser.add_argument("--override_cycle_backbone_template_cores_across_targets", default=None)
    parser.add_argument("--output_tag", default=None)
    parser.add_argument(
        "--output_dir",
        default="results_small/step5_inverse_design/polymer/_dit_timed_probes",
    )
    args = parser.parse_args()

    device = str(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    temp_config_path = _build_temp_step5_config(
        config_path=args.config,
        c_target=str(args.family),
        enabled_runs=["S2_conditional"],
    )
    temp_base_config_path = _build_temp_base_config(
        base_config_path=args.base_config,
        c_target=str(args.family),
        override_length_prior_min_tokens=args.override_length_prior_min_tokens,
        override_length_prior_max_tokens=args.override_length_prior_max_tokens,
        override_class_match_min_request_size=args.override_class_match_min_request_size,
        override_terminal_star_anchor=_parse_optional_bool(args.override_terminal_star_anchor),
        override_backbone_template_max_templates=args.override_backbone_template_max_templates,
        override_cycle_backbone_template_cores_across_targets=_parse_optional_bool(
            args.override_cycle_backbone_template_cores_across_targets
        ),
    )
    resolved = load_step5_config(
        config_path=temp_config_path,
        base_config_path=temp_base_config_path,
        model_size=args.model_size,
    )
    run_cfg = deepcopy(resolved.step5)
    run_cfg["run_name"] = f"S2_conditional_{str(args.family).strip().lower()}_dit_timed_probe"
    run_cfg["canonical_family"] = "S2"
    run_cfg["s2"]["cfg_scale"] = float(args.cfg_scale)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / str(args.family).strip().lower()
    run_dirs = create_run_dirs(run_dir)
    save_run_config_snapshot(run_dirs, run_cfg=run_cfg, resolved=resolved, config_path=temp_config_path)
    target_row_indices = _parse_int_csv(args.target_row_indices, default=[int(args.target_row_index)])
    seeds = _parse_int_csv(args.seeds, default=[int(args.seed)])

    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else _find_latest_conditional_checkpoint(resolved.results_dir, str(args.family))
    )
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint for family={args.family!r}")

    artifacts = _load_s2_training_artifacts_from_existing_checkpoint(
        resolved=resolved,
        run_cfg=run_cfg,
        checkpoint_path=checkpoint_path,
        metrics_dir=run_dirs["metrics_dir"],
        device=device,
    )
    prior = resolve_class_sampling_prior(resolved, run_cfg, artifacts.tokenizer, metrics_dir=run_dirs["metrics_dir"])
    classifier = PolymerClassifier(patterns=resolved.polymer_patterns)

    settings = {
        "baseline_cfg_const": {
            "conditional_cfg_scale_early_frac": 1.0,
            "conditional_cfg_scale_ramp_start_frac": 0.0,
            "conditional_cfg_scale_retry_multiplier": 1.0,
        },
        "dit_cfg_ramp_retry": {
            "conditional_cfg_scale_early_frac": float(args.candidate_early_frac),
            "conditional_cfg_scale_ramp_start_frac": float(args.candidate_ramp_start_frac),
            "conditional_cfg_scale_retry_multiplier": float(args.candidate_retry_multiplier),
        },
    }

    results: List[Dict[str, Any]] = []
    for seed in seeds:
        for target_row_index in target_row_indices:
            target_row = resolved.target_family_df.iloc[int(target_row_index)]
            condition_bundle = _build_condition_bundle(artifacts, target_row, device=device)
            for setting_name, overrides in settings.items():
                seed_everything(int(seed), deterministic=True)
                resolved_variant = deepcopy(resolved)
                for key, value in overrides.items():
                    resolved_variant.step5[key] = value
                sampling_state = {} if bool(prior.cycle_backbone_template_cores_across_targets) else None
                sampler = create_conditional_sampler(
                    diffusion_model=artifacts.diffusion_model,
                    tokenizer=artifacts.tokenizer,
                    resolved=resolved_variant,
                    prior=prior,
                    condition_bundle=condition_bundle,
                    cfg_scale=float(run_cfg["s2"]["cfg_scale"]),
                    device=device,
                    num_steps=int(args.num_steps),
                )
                if args.probe_mode == "acceptance":
                    probe_row = _probe_acceptance(
                        sampler=sampler,
                        tokenizer=artifacts.tokenizer,
                        prior=prior,
                        resolved=resolved_variant,
                        generation_budget=int(args.generation_budget),
                        sampling_state=sampling_state,
                    )
                else:
                    probe_row = _probe_raw(
                        sampler=sampler,
                        tokenizer=artifacts.tokenizer,
                        prior=prior,
                        resolved=resolved_variant,
                        num_samples=int(args.generation_budget),
                        sampling_state=sampling_state,
                    )
                quality = _quality_stats(probe_row["smiles"], classifier=classifier, family=str(args.family))
                results.append(
                    {
                        "family": str(args.family),
                        "probe_mode": str(args.probe_mode),
                        "seed": int(seed),
                        "target_row_index": int(target_row_index),
                        "target_row_id": int(target_row["target_row_id"]),
                        "num_steps": int(args.num_steps),
                        "cfg_scale": float(run_cfg["s2"]["cfg_scale"]),
                        "setting": str(setting_name),
                        **probe_row,
                        **quality,
                    }
                )

    summary_df = pd.DataFrame(results)
    row_label = "-".join(str(int(value)) for value in target_row_indices)
    seed_label = "-".join(str(int(value)) for value in seeds)
    probe_suffix = (
        f"{str(args.probe_mode).strip().lower()}"
        f"_rows{row_label}"
        f"_seeds{seed_label}"
        f"_g{int(args.generation_budget)}"
        f"_steps{int(args.num_steps)}"
    )
    if args.override_length_prior_min_tokens is not None:
        probe_suffix += f"_lmin{int(args.override_length_prior_min_tokens)}"
    if args.override_length_prior_max_tokens is not None:
        probe_suffix += f"_lmax{int(args.override_length_prior_max_tokens)}"
    if args.override_class_match_min_request_size is not None:
        probe_suffix += f"_mreq{int(args.override_class_match_min_request_size)}"
    if args.override_terminal_star_anchor is not None:
        probe_suffix += f"_anchor{str(_parse_optional_bool(args.override_terminal_star_anchor)).lower()}"
    if args.override_backbone_template_max_templates is not None:
        probe_suffix += f"_maxtpl{int(args.override_backbone_template_max_templates)}"
    if args.override_cycle_backbone_template_cores_across_targets is not None:
        probe_suffix += (
            f"_corecycle{str(_parse_optional_bool(args.override_cycle_backbone_template_cores_across_targets)).lower()}"
        )
    if args.output_tag:
        probe_suffix += f"_{str(args.output_tag).strip()}"
    summary_path = run_dirs["metrics_dir"] / f"conditional_timed_probe_{probe_suffix}_summary.csv"
    details_path = run_dirs["metrics_dir"] / f"conditional_timed_probe_{probe_suffix}_details.json"
    summary_df.to_csv(summary_path, index=False)
    with open(details_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "family": str(args.family),
                "probe_mode": str(args.probe_mode),
                "device": device,
                "seeds": [int(value) for value in seeds],
                "target_row_indices": [int(value) for value in target_row_indices],
                "override_length_prior_min_tokens": args.override_length_prior_min_tokens,
                "override_length_prior_max_tokens": args.override_length_prior_max_tokens,
                "override_class_match_min_request_size": args.override_class_match_min_request_size,
                "override_terminal_star_anchor": _parse_optional_bool(args.override_terminal_star_anchor),
                "override_backbone_template_max_templates": args.override_backbone_template_max_templates,
                "override_cycle_backbone_template_cores_across_targets": _parse_optional_bool(
                    args.override_cycle_backbone_template_cores_across_targets
                ),
                "temp_config_path": str(temp_config_path),
                "temp_base_config_path": str(temp_base_config_path),
                "checkpoint_path": str(checkpoint_path),
                "results": results,
            },
            handle,
            indent=2,
        )

    print(f"summary={summary_path}")
    print(f"details={details_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
