"""Reusable Step 6_2 run execution helpers."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
import yaml

from .config import build_run_config
from .conditional_sampling import create_conditional_sampler, sample_conditional_with_class_prior
from .dataset import build_inference_condition_bundle
from .dpo import train_s4_dpo_alignment
from .evaluation import (
    aggregate_round_metrics,
    aggregate_target_row_metrics,
    build_generated_samples_frame,
    build_method_metrics,
    evaluate_generated_samples,
    load_step62_evaluator,
    summarize_target_rows,
)
from .frozen_sampling import (
    create_constrained_sampler,
    load_step1_diffusion,
    resolve_class_sampling_prior,
    sample_unconditional_with_class_prior,
)
from .guided_sampler import GuidedConditionalSampler, GuidedSampler
from .plotting import plot_generated_chi_vs_target, plot_per_target_success, plot_success_gate_funnel
from .rl_trainer import train_s4_rl_alignment
from .train_s2 import train_s2_supervised_run
from src.utils.reproducibility import save_run_metadata, seed_everything
from src.utils.reporting import save_artifact_manifest, write_initial_log


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_json(payload: Dict[str, Any], path: Path) -> None:
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


def create_run_dirs(run_dir: Path) -> Dict[str, Path]:
    metrics_dir = run_dir / "metrics"
    figures_dir = run_dir / "figures"
    checkpoints_dir = run_dir / "checkpoints"
    for path in [run_dir, metrics_dir, figures_dir, checkpoints_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "metrics_dir": metrics_dir,
        "figures_dir": figures_dir,
        "checkpoints_dir": checkpoints_dir,
    }


def save_run_config_snapshot(
    run_dirs: Dict[str, Path],
    *,
    run_cfg: Dict[str, object],
    resolved,
    config_path: str,
    extra_context: Optional[Dict[str, Any]] = None,
) -> None:
    snapshot = {
        "model_size": resolved.model_size,
        "split_mode": resolved.split_mode,
        "classification_split_mode": resolved.classification_split_mode,
        "c_target": resolved.c_target,
        "run_config": run_cfg,
        "paths": {
            "benchmark_root": str(resolved.benchmark_root),
            "step4_reg_metrics_dir": str(resolved.step4_reg_metrics_dir),
            "step4_cls_metrics_dir": str(resolved.step4_cls_metrics_dir),
        },
    }
    if extra_context:
        snapshot["extra_context"] = extra_context
    with open(run_dirs["run_dir"] / "config_snapshot.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(_as_yamlable(snapshot), handle, sort_keys=False)

    seed_info = seed_everything(int(resolved.step6_2["random_seed"]), deterministic=True)
    save_run_metadata(run_dirs["run_dir"], config_path, seed_info)
    write_initial_log(
        step_dir=run_dirs["run_dir"],
        step_name="step6_2_inverse_design",
        context={
            "config_path": config_path,
            "model_size": resolved.model_size,
            "run_name": run_cfg["run_name"],
            "canonical_family": run_cfg["canonical_family"],
            "split_mode": resolved.split_mode,
            "c_target": resolved.c_target,
            **(extra_context or {}),
        },
    )


def build_s4_warm_start_run_cfg(resolved, run_cfg: Dict[str, object]) -> Dict[str, object]:
    source_name = str(run_cfg["s4"].get("supervised_train_config_source", "s2")).strip().lower()
    if source_name != "s2":
        raise NotImplementedError(
            f"Unsupported Step 6_2 S4 warm-start source: {source_name}. Only 's2' is implemented."
        )
    warm_cfg = build_run_config(resolved, "S2_conditional")
    if isinstance(run_cfg.get("s2"), dict):
        warm_cfg["s2"] = deepcopy(run_cfg["s2"])
    supervised_override = run_cfg.get("s4", {}).get("supervised_train_override")
    if isinstance(supervised_override, dict):
        if isinstance(supervised_override.get("s2"), dict):
            warm_cfg["s2"].update(deepcopy(supervised_override["s2"]))
        else:
            warm_cfg["s2"].update(deepcopy(supervised_override))
    warm_cfg["run_name"] = f"{run_cfg['run_name']}__warm_start"
    return warm_cfg


def run_single_target_sampling(
    *,
    run_cfg: Dict[str, object],
    resolved,
    target_row: pd.Series,
    tokenizer,
    diffusion_model,
    prior,
    device: str,
    generation_budget: int,
    evaluator=None,
    s2_scaler=None,
) -> Tuple[List[str], Dict[str, int], Dict[str, object]]:
    if run_cfg["run_name"] == "S0_raw_unconditional":
        sampler = create_constrained_sampler(
            diffusion_model=diffusion_model,
            tokenizer=tokenizer,
            resolved=resolved,
            prior=prior,
            device=device,
        )
        smiles, sample_meta = sample_unconditional_with_class_prior(
            sampler=sampler,
            tokenizer=tokenizer,
            prior=prior,
            resolved=resolved,
            num_samples=int(generation_budget),
            show_progress=False,
        )
        return smiles, {"training_soluble_oracle_calls": 0, "training_chi_oracle_calls": 0}, sample_meta

    if run_cfg["run_name"] == "S1_guided_frozen":
        if evaluator is None:
            raise ValueError("S1 guided frozen sampling requires a loaded Step 6_2 evaluator.")
        s1_cfg = run_cfg["s1"]
        sampler = GuidedSampler(
            diffusion_model=diffusion_model,
            tokenizer=tokenizer,
            num_steps=resolved.base_config["diffusion"]["num_steps"],
            temperature=float(resolved.step6_2["sampling_temperature"]),
            top_k=resolved.base_config.get("sampling", {}).get("top_k"),
            top_p=resolved.base_config.get("sampling", {}).get("top_p"),
            target_stars=int(resolved.base_config.get("sampling", {}).get("target_stars", 2)),
            use_constraints=bool(resolved.base_config.get("sampling", {}).get("use_constraints", True)),
            device=device,
            evaluator=evaluator,
            target_row=target_row.to_dict(),
            best_of_k=int(s1_cfg["best_of_k"]),
            guidance_start_frac=float(s1_cfg["guidance_start_frac"]),
            class_guidance_start_frac=float(s1_cfg["class_guidance_start_frac"]),
            class_guidance_min_valid_frac=float(s1_cfg["class_guidance_min_valid_frac"]),
            class_surrogate_mode=str(s1_cfg["class_surrogate_mode"]),
            sol_log_prob_floor=float(s1_cfg["sol_log_prob_floor"]),
            w_sol=float(s1_cfg["w_sol"]),
            w_chi=float(s1_cfg["w_chi"]),
            w_class=float(s1_cfg["w_class"]),
        )
        sampler.set_class_token_bias_start_frac(float(resolved.step6_2.get("class_token_bias_start_frac", 0.0)))
        if prior.class_token_logit_bias is not None:
            sampler.set_class_token_logit_bias(prior.class_token_logit_bias)
        smiles, sample_meta = sample_unconditional_with_class_prior(
            sampler=sampler,
            tokenizer=tokenizer,
            prior=prior,
            resolved=resolved,
            num_samples=int(generation_budget),
            show_progress=False,
        )
        return smiles, sampler.get_guidance_stats(), sample_meta

    if str(run_cfg["canonical_family"]) == "S2":
        if s2_scaler is None:
            raise ValueError("S2 conditional sampling requires the fitted ConditionScaler.")
        condition_bundle = torch.tensor(
            build_inference_condition_bundle(
                temperature=float(target_row["temperature"]),
                phi=float(target_row["phi"]),
                chi_goal=float(target_row["chi_target"]),
                scaler=s2_scaler,
                soluble=1,
            ),
            dtype=torch.float32,
            device=device,
        )
        sampler = create_conditional_sampler(
            diffusion_model=diffusion_model,
            tokenizer=tokenizer,
            resolved=resolved,
            prior=prior,
            condition_bundle=condition_bundle,
            cfg_scale=float(run_cfg["s2"]["cfg_scale"]),
            device=device,
        )
        smiles, sample_meta = sample_conditional_with_class_prior(
            sampler=sampler,
            tokenizer=tokenizer,
            prior=prior,
            resolved=resolved,
            num_samples=int(generation_budget),
            show_progress=False,
        )
        return smiles, {"training_soluble_oracle_calls": 0, "training_chi_oracle_calls": 0}, sample_meta

    if str(run_cfg["canonical_family"]) == "S3":
        if evaluator is None:
            raise ValueError("S3 conditional guided sampling requires a loaded Step 6_2 evaluator.")
        if s2_scaler is None:
            raise ValueError("S3 conditional guided sampling requires the fitted ConditionScaler.")
        s3_cfg = run_cfg["s3"]
        condition_bundle = torch.tensor(
            build_inference_condition_bundle(
                temperature=float(target_row["temperature"]),
                phi=float(target_row["phi"]),
                chi_goal=float(target_row["chi_target"]),
                scaler=s2_scaler,
                soluble=1,
            ),
            dtype=torch.float32,
            device=device,
        )
        sampler = GuidedConditionalSampler(
            diffusion_model=diffusion_model,
            tokenizer=tokenizer,
            num_steps=resolved.base_config["diffusion"]["num_steps"],
            temperature=float(resolved.step6_2["sampling_temperature"]),
            top_k=resolved.base_config.get("sampling", {}).get("top_k"),
            top_p=resolved.base_config.get("sampling", {}).get("top_p"),
            target_stars=int(resolved.base_config.get("sampling", {}).get("target_stars", 2)),
            use_constraints=bool(resolved.base_config.get("sampling", {}).get("use_constraints", True)),
            device=device,
            condition_bundle=condition_bundle,
            cfg_scale=float(s3_cfg["cfg_scale"]),
            evaluator=evaluator,
            target_row=target_row.to_dict(),
            best_of_k=int(s3_cfg["best_of_k"]),
            guidance_start_frac=float(s3_cfg["guidance_start_frac"]),
            class_guidance_start_frac=float(s3_cfg["class_guidance_start_frac"]),
            class_guidance_min_valid_frac=float(s3_cfg["class_guidance_min_valid_frac"]),
            class_surrogate_mode=str(s3_cfg["class_surrogate_mode"]),
            sol_log_prob_floor=float(s3_cfg["sol_log_prob_floor"]),
            w_sol=float(s3_cfg["w_sol"]),
            w_chi=float(s3_cfg["w_chi"]),
            w_class=float(s3_cfg["w_class"]),
        )
        sampler.set_class_token_bias_start_frac(float(resolved.step6_2.get("class_token_bias_start_frac", 0.0)))
        if prior.class_token_logit_bias is not None:
            sampler.set_class_token_logit_bias(prior.class_token_logit_bias)
        smiles, sample_meta = sample_conditional_with_class_prior(
            sampler=sampler,
            tokenizer=tokenizer,
            prior=prior,
            resolved=resolved,
            num_samples=int(generation_budget),
            show_progress=False,
        )
        return smiles, sampler.get_guidance_stats(), sample_meta

    if str(run_cfg["canonical_family"]) == "S4":
        if s2_scaler is None:
            raise ValueError("S4 sampling requires the fitted ConditionScaler from the warm start.")
        condition_bundle = torch.tensor(
            build_inference_condition_bundle(
                temperature=float(target_row["temperature"]),
                phi=float(target_row["phi"]),
                chi_goal=float(target_row["chi_target"]),
                scaler=s2_scaler,
                soluble=1,
            ),
            dtype=torch.float32,
            device=device,
        )
        sampler = create_conditional_sampler(
            diffusion_model=diffusion_model,
            tokenizer=tokenizer,
            resolved=resolved,
            prior=prior,
            condition_bundle=condition_bundle,
            cfg_scale=float(run_cfg["s4"]["cfg_scale"]),
            device=device,
        )
        smiles, sample_meta = sample_conditional_with_class_prior(
            sampler=sampler,
            tokenizer=tokenizer,
            prior=prior,
            resolved=resolved,
            num_samples=int(generation_budget),
            show_progress=False,
        )
        return smiles, {"training_soluble_oracle_calls": 0, "training_chi_oracle_calls": 0}, sample_meta

    raise NotImplementedError(f"Unsupported Step 6_2 run: {run_cfg['run_name']}")


def execute_step62_run(
    *,
    resolved,
    run_name: str,
    device: str,
    config_path: str,
    run_cfg: Optional[Dict[str, object]] = None,
    run_dir: Optional[Path] = None,
    shared_evaluator=None,
    target_rows_df: Optional[pd.DataFrame] = None,
    generation_budget: Optional[int] = None,
    sampling_seeds: Optional[List[int]] = None,
    num_rounds: Optional[int] = None,
    save_figures: bool = True,
    extra_context: Optional[Dict[str, Any]] = None,
    pruning_callback: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    run_cfg = run_cfg or build_run_config(resolved, run_name)
    run_dir = run_dir or (resolved.benchmark_root / str(run_cfg["run_name"]))
    target_rows_df = target_rows_df.copy() if target_rows_df is not None else resolved.target_family_df.copy()
    generation_budget = int(generation_budget if generation_budget is not None else resolved.step6_2["generation_budget"])
    sampling_seeds = list(sampling_seeds if sampling_seeds is not None else resolved.step6_2["sampling_seeds"])
    num_rounds = int(num_rounds if num_rounds is not None else resolved.step6_2["num_sampling_rounds"])
    if num_rounds > len(sampling_seeds):
        raise ValueError("num_rounds exceeds the number of provided sampling seeds.")

    run_dirs = create_run_dirs(run_dir)
    save_run_config_snapshot(
        run_dirs,
        run_cfg=run_cfg,
        resolved=resolved,
        config_path=config_path,
        extra_context=extra_context,
    )
    evaluator = shared_evaluator
    s2_scaler = None
    canonical_family = str(run_cfg["canonical_family"])

    if canonical_family in {"S2", "S3"}:
        training_artifacts = train_s2_supervised_run(
            resolved=resolved,
            run_cfg=run_cfg,
            run_dirs=run_dirs,
            device=device,
            pruning_callback=pruning_callback,
            pruning_stage="s2",
        )
        tokenizer = training_artifacts.tokenizer
        diffusion_model = training_artifacts.diffusion_model
        prior = resolve_class_sampling_prior(resolved, run_cfg, tokenizer, metrics_dir=run_dirs["metrics_dir"])
        s2_scaler = training_artifacts.scaler
    elif canonical_family == "S4" and str(run_cfg["s4"]["alignment_mode"]).strip().lower() == "dpo":
        warm_run_cfg = build_s4_warm_start_run_cfg(resolved, run_cfg)
        warm_dirs = create_run_dirs(run_dirs["run_dir"] / "_warm_start")
        save_run_config_snapshot(
            warm_dirs,
            run_cfg=warm_run_cfg,
            resolved=resolved,
            config_path=config_path,
            extra_context={"parent_run_name": run_cfg["run_name"], **(extra_context or {})},
        )
        warm_start_artifacts = train_s2_supervised_run(
            resolved=resolved,
            run_cfg=warm_run_cfg,
            run_dirs=warm_dirs,
            device=device,
            pruning_callback=pruning_callback,
            pruning_stage="warm_start",
        )
        prior = resolve_class_sampling_prior(
            resolved,
            run_cfg,
            warm_start_artifacts.tokenizer,
            metrics_dir=run_dirs["metrics_dir"],
        )
        pair_source = str(run_cfg["s4"]["dpo"]["pair_source"]).strip().lower()
        if pair_source == "target_row_synthetic" and evaluator is None:
            evaluator = load_step62_evaluator(resolved, device=device)
        dpo_artifacts = train_s4_dpo_alignment(
            resolved=resolved,
            run_cfg=run_cfg,
            run_dirs=run_dirs,
            warm_start=warm_start_artifacts,
            prior=prior,
            evaluator=evaluator,
            device=device,
            pruning_callback=pruning_callback,
        )
        tokenizer = dpo_artifacts.tokenizer
        diffusion_model = dpo_artifacts.policy_model
        s2_scaler = dpo_artifacts.scaler
        _write_json(
            {
                "warm_start_run_name": warm_run_cfg["run_name"],
                "warm_start_dir": str(warm_dirs["run_dir"]),
                "warm_start_best_checkpoint": str(warm_start_artifacts.checkpoint_path),
                "aligned_best_checkpoint": str(dpo_artifacts.checkpoint_path),
            },
            run_dirs["metrics_dir"] / "s4_alignment_summary.json",
        )
    elif canonical_family == "S4" and str(run_cfg["s4"]["alignment_mode"]).strip().lower() == "rl":
        warm_run_cfg = build_s4_warm_start_run_cfg(resolved, run_cfg)
        warm_dirs = create_run_dirs(run_dirs["run_dir"] / "_warm_start")
        save_run_config_snapshot(
            warm_dirs,
            run_cfg=warm_run_cfg,
            resolved=resolved,
            config_path=config_path,
            extra_context={"parent_run_name": run_cfg["run_name"], **(extra_context or {})},
        )
        warm_start_artifacts = train_s2_supervised_run(
            resolved=resolved,
            run_cfg=warm_run_cfg,
            run_dirs=warm_dirs,
            device=device,
            pruning_callback=pruning_callback,
            pruning_stage="warm_start",
        )
        prior = resolve_class_sampling_prior(
            resolved,
            run_cfg,
            warm_start_artifacts.tokenizer,
            metrics_dir=run_dirs["metrics_dir"],
        )
        if evaluator is None:
            evaluator = load_step62_evaluator(resolved, device=device)
        rl_artifacts = train_s4_rl_alignment(
            resolved=resolved,
            run_cfg=run_cfg,
            run_dirs=run_dirs,
            warm_start=warm_start_artifacts,
            prior=prior,
            evaluator=evaluator,
            device=device,
            pruning_callback=pruning_callback,
        )
        tokenizer = rl_artifacts.tokenizer
        diffusion_model = rl_artifacts.policy_model
        s2_scaler = rl_artifacts.scaler
        _write_json(
            {
                "warm_start_run_name": warm_run_cfg["run_name"],
                "warm_start_dir": str(warm_dirs["run_dir"]),
                "warm_start_best_checkpoint": str(warm_start_artifacts.checkpoint_path),
                "aligned_best_checkpoint": str(rl_artifacts.checkpoint_path),
            },
            run_dirs["metrics_dir"] / "s4_alignment_summary.json",
        )
    else:
        tokenizer, diffusion_model, checkpoint_path = load_step1_diffusion(resolved, device=device)
        prior = resolve_class_sampling_prior(resolved, run_cfg, tokenizer, metrics_dir=run_dirs["metrics_dir"])
        _write_json({"checkpoint_path": str(checkpoint_path)}, run_dirs["metrics_dir"] / "step1_checkpoint.json")

    generated_frames: List[pd.DataFrame] = []
    evaluation_frames: List[pd.DataFrame] = []
    round_oracle_rows: List[Dict[str, int]] = []
    sample_id_start = 1

    if canonical_family in {"S1", "S3"} and evaluator is None:
        evaluator = load_step62_evaluator(resolved, device=device)

    for round_id, sampling_seed in enumerate(sampling_seeds[:num_rounds], start=1):
        seed_everything(int(sampling_seed), deterministic=True)
        round_soluble_calls = 0
        round_chi_calls = 0
        round_class_guidance_suppressed_steps = 0

        for _, target_row in target_rows_df.iterrows():
            smiles, guidance_stats, _sample_meta = run_single_target_sampling(
                run_cfg=run_cfg,
                resolved=resolved,
                target_row=target_row,
                tokenizer=tokenizer,
                diffusion_model=diffusion_model,
                prior=prior,
                evaluator=evaluator,
                device=device,
                s2_scaler=s2_scaler,
                generation_budget=generation_budget,
            )
            generated_df = build_generated_samples_frame(
                smiles,
                target_row=target_row,
                round_id=round_id,
                sampling_seed=int(sampling_seed),
                run_name=str(run_cfg["run_name"]),
                canonical_family=str(run_cfg["canonical_family"]),
                sample_id_start=sample_id_start,
            )
            sample_id_start += int(len(generated_df))
            if evaluator is None:
                evaluator = load_step62_evaluator(resolved, device=device)
            evaluation_df = evaluate_generated_samples(generated_df, evaluator)
            generated_frames.append(generated_df)
            evaluation_frames.append(evaluation_df)
            round_soluble_calls += int(guidance_stats.get("training_soluble_oracle_calls", 0))
            round_chi_calls += int(guidance_stats.get("training_chi_oracle_calls", 0))
            round_class_guidance_suppressed_steps += int(guidance_stats.get("class_guidance_suppressed_steps", 0))

        round_oracle_rows.append(
            {
                "run_name": str(run_cfg["run_name"]),
                "canonical_family": str(run_cfg["canonical_family"]),
                "round_id": round_id,
                "sampling_seed": int(sampling_seed),
                "training_soluble_oracle_calls": int(round_soluble_calls),
                "training_chi_oracle_calls": int(round_chi_calls),
                "class_guidance_suppressed_steps": int(round_class_guidance_suppressed_steps),
            }
        )

    generated_samples_df = pd.concat(generated_frames, ignore_index=True) if generated_frames else pd.DataFrame()
    evaluation_results_df = pd.concat(evaluation_frames, ignore_index=True) if evaluation_frames else pd.DataFrame()
    target_row_metrics_df = aggregate_target_row_metrics(evaluation_results_df)
    target_row_summary_df = summarize_target_rows(target_row_metrics_df)
    round_metrics_df = aggregate_round_metrics(evaluation_results_df, target_row_metrics_df)
    round_oracle_df = pd.DataFrame(round_oracle_rows)
    if not round_oracle_df.empty:
        round_metrics_df = round_metrics_df.drop(
            columns=["training_soluble_oracle_calls", "training_chi_oracle_calls"],
            errors="ignore",
        ).merge(
            round_oracle_df,
            on=["run_name", "canonical_family", "round_id", "sampling_seed"],
            how="left",
        )

    method_metrics = build_method_metrics(round_metrics_df, target_row_summary_df)
    method_metrics.update(
        {
            "run_name": str(run_cfg["run_name"]),
            "canonical_family": str(run_cfg["canonical_family"]),
            "mean_training_soluble_oracle_calls": (
                float(round_metrics_df["training_soluble_oracle_calls"].mean())
                if "training_soluble_oracle_calls" in round_metrics_df.columns and not round_metrics_df.empty
                else 0.0
            ),
            "mean_training_chi_oracle_calls": (
                float(round_metrics_df["training_chi_oracle_calls"].mean())
                if "training_chi_oracle_calls" in round_metrics_df.columns and not round_metrics_df.empty
                else 0.0
            ),
            "mean_class_guidance_suppressed_steps": (
                float(round_metrics_df["class_guidance_suppressed_steps"].mean())
                if "class_guidance_suppressed_steps" in round_metrics_df.columns and not round_metrics_df.empty
                else 0.0
            ),
        }
    )

    _write_frame(generated_samples_df, run_dirs["metrics_dir"] / "generated_samples.csv")
    _write_frame(evaluation_results_df, run_dirs["metrics_dir"] / "evaluation_results.csv")
    _write_frame(round_metrics_df, run_dirs["metrics_dir"] / "round_metrics.csv")
    _write_frame(target_row_metrics_df, run_dirs["metrics_dir"] / "target_row_metrics.csv")
    _write_frame(target_row_summary_df, run_dirs["metrics_dir"] / "target_row_summary.csv")
    _write_json(method_metrics, run_dirs["metrics_dir"] / "method_metrics.json")

    if save_figures and not evaluation_results_df.empty:
        plot_success_gate_funnel(
            evaluation_results_df,
            run_dirs["figures_dir"] / "candidate_screening_funnel.png",
            font_size=int(resolved.step6_2["figure_font_size"]),
        )
        plot_generated_chi_vs_target(
            evaluation_results_df,
            run_dirs["figures_dir"] / "generated_chi_vs_target.png",
            font_size=int(resolved.step6_2["figure_font_size"]),
        )
    if save_figures and not target_row_summary_df.empty:
        plot_per_target_success(
            target_row_summary_df,
            run_dirs["figures_dir"] / "per_target_success_hit_rate.png",
            font_size=int(resolved.step6_2["figure_font_size"]),
        )

    save_artifact_manifest(
        step_dir=run_dirs["run_dir"],
        metrics_dir=run_dirs["metrics_dir"],
        figures_dir=run_dirs["figures_dir"],
    )
    return {
        "run_cfg": run_cfg,
        "run_dirs": run_dirs,
        "method_metrics": method_metrics,
        "evaluation_results_df": evaluation_results_df,
        "round_metrics_df": round_metrics_df,
        "target_row_summary_df": target_row_summary_df,
    }
