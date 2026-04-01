"""On-policy RL alignment for Step 6_2 S4_rl."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch

from .conditional_sampling import create_conditional_sampler, sample_conditional_with_class_prior
from .dataset import (
    ConditionScaler,
    build_inference_condition_bundle,
    build_step62_supervised_frames,
)
from .evaluation import evaluate_generated_samples
from .frozen_sampling import ResolvedClassSamplingPrior
from .rewards import compute_success_shaped_rewards
from .supervised import build_optimizer_and_scheduler, load_step62_checkpoint_into_modules
from .train_s2 import S2TrainingArtifacts
from .trajectory import TrajectoryConditionalSampler, sample_trajectories_with_class_prior


@dataclass
class RlTrainingArtifacts:
    """Artifacts returned by Step 6_2 S4_rl alignment."""

    tokenizer: object
    policy_model: torch.nn.Module
    reference_model: torch.nn.Module
    checkpoint_path: Path
    last_checkpoint_path: Path
    scaler: ConditionScaler
    history_df: pd.DataFrame
    proxy_history_df: pd.DataFrame


def _build_rollout_frame(
    *,
    smiles: List[str],
    prompt_df: pd.DataFrame,
    sample_id_start: int,
    run_name: str,
    canonical_family: str,
    step_idx: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for offset, (smiles_value, prompt_row) in enumerate(zip(smiles, prompt_df.to_dict(orient="records"))):
        rows.append(
            {
                "sample_id": int(sample_id_start + offset),
                "target_row_id": int(prompt_row["target_row_id"]),
                "target_row_key": str(prompt_row["target_row_key"]),
                "round_id": int(step_idx),
                "sampling_seed": int(prompt_row.get("sampling_seed", 0)),
                "run_name": str(run_name),
                "canonical_family": str(canonical_family),
                "c_target": str(prompt_row["c_target"]),
                "temperature": float(prompt_row["temperature"]),
                "phi": float(prompt_row["phi"]),
                "chi_target": float(prompt_row["chi_target"]),
                "property_rule": str(prompt_row.get("property_rule", "upper_bound")),
                "smiles": str(smiles_value),
            }
        )
    return pd.DataFrame(rows)


def _sample_benchmark_prompt_rows(
    resolved,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    source = resolved.target_family_df.reset_index(drop=True)
    indices = rng.integers(0, len(source), size=int(batch_size))
    prompt_df = source.iloc[indices].copy().reset_index(drop=True)
    return prompt_df


def _sample_generic_prompt_rows(
    resolved,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    frames = build_step62_supervised_frames(resolved)
    train_d_chi = frames["train_d_chi"].copy()
    train_d_chi = train_d_chi.loc[train_d_chi["water_miscible"].astype(int) == 1].copy()
    if train_d_chi.empty:
        raise ValueError("No water-miscible D_chi training rows available for generic RL prompt sampling.")
    train_d_chi = train_d_chi.sort_values(["temperature", "phi", "row_id"]).reset_index(drop=True)
    indices = rng.integers(0, len(train_d_chi), size=int(batch_size))
    selected = train_d_chi.iloc[indices].copy().reset_index(drop=True)
    selected["target_row_id"] = selected["row_id"].astype(int) + 1_000_000
    selected["target_row_key"] = selected["row_id"].map(lambda row_id: f"generic_train_row_{int(row_id)}")
    selected["c_target"] = resolved.c_target
    selected["chi_target"] = selected["chi"].astype(float)
    selected["property_rule"] = "upper_bound"
    return selected[
        ["target_row_id", "target_row_key", "c_target", "temperature", "phi", "chi_target", "property_rule"]
    ].copy()


def _sample_prompt_rows(
    resolved,
    run_cfg: Dict[str, object],
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    prompt_source = str(run_cfg["s4"]["rl_prompt_source"]).strip().lower()
    if prompt_source == "benchmark_target_rows":
        return _sample_benchmark_prompt_rows(resolved, batch_size=batch_size, rng=rng)
    if prompt_source == "generic_condition_distribution":
        return _sample_generic_prompt_rows(resolved, batch_size=batch_size, rng=rng)
    raise NotImplementedError(f"Unsupported Step 6_2 rl_prompt_source: {prompt_source}")


def _prompt_df_to_condition_tensor(prompt_df: pd.DataFrame, *, scaler: ConditionScaler, device: str) -> torch.Tensor:
    bundles = [
        build_inference_condition_bundle(
            temperature=float(row["temperature"]),
            phi=float(row["phi"]),
            chi_goal=float(row["chi_target"]),
            scaler=scaler,
            soluble=1,
        )
        for row in prompt_df.to_dict(orient="records")
    ]
    return torch.tensor(np.asarray(bundles, dtype=np.float32), dtype=torch.float32, device=device)


def _create_trajectory_sampler(
    *,
    diffusion_model,
    tokenizer,
    resolved,
    prior: ResolvedClassSamplingPrior,
    condition_bundle: torch.Tensor,
    cfg_scale: float,
    device: str,
) -> TrajectoryConditionalSampler:
    sampling_cfg = resolved.base_config.get("sampling", {})
    sampler = TrajectoryConditionalSampler(
        diffusion_model=diffusion_model,
        tokenizer=tokenizer,
        num_steps=resolved.base_config["diffusion"]["num_steps"],
        temperature=float(resolved.step6_2["sampling_temperature"]),
        top_k=sampling_cfg.get("top_k"),
        top_p=sampling_cfg.get("top_p"),
        target_stars=int(sampling_cfg.get("target_stars", 2)),
        use_constraints=bool(sampling_cfg.get("use_constraints", True)),
        device=device,
        condition_bundle=condition_bundle,
        cfg_scale=float(cfg_scale),
    )
    sampler.set_class_token_bias_start_frac(float(resolved.step6_2.get("class_token_bias_start_frac", 0.0)))
    if prior.class_token_logit_bias is not None:
        sampler.set_class_token_logit_bias(prior.class_token_logit_bias)
    return sampler


def _save_rl_checkpoint(
    *,
    checkpoint_path: Path,
    policy_model,
    reference_checkpoint_path: Path,
    optimizer,
    scheduler,
    step_idx: int,
    best_proxy_success_hit_rate: float,
    run_cfg: Dict[str, object],
    warm_start: S2TrainingArtifacts,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": policy_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step_idx": int(step_idx),
        "best_proxy_success_hit_rate": float(best_proxy_success_hit_rate),
        "run_name": str(run_cfg["run_name"]),
        "alignment_mode": "rl",
        "reference_checkpoint_path": str(reference_checkpoint_path),
        "warm_start_checkpoint_path": str(warm_start.checkpoint_path),
        "condition_scaler": {
            "temperature_min": float(warm_start.scaler.temperature_min),
            "temperature_max": float(warm_start.scaler.temperature_max),
            "phi_min": float(warm_start.scaler.phi_min),
            "phi_max": float(warm_start.scaler.phi_max),
            "chi_goal_min": float(warm_start.scaler.chi_goal_min),
            "chi_goal_max": float(warm_start.scaler.chi_goal_max),
        },
    }
    torch.save(payload, checkpoint_path)


def _evaluate_proxy_success_hit_rate(
    *,
    resolved,
    run_cfg: Dict[str, object],
    policy_model,
    tokenizer,
    scaler: ConditionScaler,
    prior: ResolvedClassSamplingPrior,
    evaluator,
    device: str,
    step_idx: int,
) -> Tuple[float, pd.DataFrame]:
    rows: List[pd.DataFrame] = []
    sample_id_start = 1
    for _, target_row in resolved.rl_proxy_df.iterrows():
        condition_bundle = torch.tensor(
            build_inference_condition_bundle(
                temperature=float(target_row["temperature"]),
                phi=float(target_row["phi"]),
                chi_goal=float(target_row["chi_target"]),
                scaler=scaler,
                soluble=1,
            ),
            dtype=torch.float32,
            device=device,
        )
        sampler = create_conditional_sampler(
            diffusion_model=policy_model,
            tokenizer=tokenizer,
            resolved=resolved,
            prior=prior,
            condition_bundle=condition_bundle,
            cfg_scale=float(run_cfg["s4"]["cfg_scale"]),
            device=device,
        )
        smiles, _metadata = sample_conditional_with_class_prior(
            sampler=sampler,
            tokenizer=tokenizer,
            prior=prior,
            resolved=resolved,
            num_samples=int(run_cfg["s4"]["rl_proxy_generation_budget"]),
            show_progress=False,
        )
        sample_df = pd.DataFrame(
            {
                "sample_id": np.arange(sample_id_start, sample_id_start + len(smiles), dtype=int),
                "target_row_id": int(target_row["target_row_id"]),
                "target_row_key": str(target_row["target_row_key"]),
                "round_id": int(step_idx),
                "sampling_seed": int(step_idx),
                "run_name": str(run_cfg["run_name"]),
                "canonical_family": str(run_cfg["canonical_family"]),
                "c_target": str(target_row["c_target"]),
                "temperature": float(target_row["temperature"]),
                "phi": float(target_row["phi"]),
                "chi_target": float(target_row["chi_target"]),
                "property_rule": str(target_row.get("property_rule", "upper_bound")),
                "smiles": smiles,
            }
        )
        sample_id_start += len(smiles)
        rows.append(evaluate_generated_samples(sample_df, evaluator))
    if not rows:
        return float("nan"), pd.DataFrame()
    eval_df = pd.concat(rows, ignore_index=True)
    return float(eval_df["success_hit"].astype(float).mean()), eval_df


def train_s4_rl_alignment(
    *,
    resolved,
    run_cfg: Dict[str, object],
    run_dirs: Dict[str, Path],
    warm_start: S2TrainingArtifacts,
    prior: ResolvedClassSamplingPrior,
    evaluator,
    device: str,
    pruning_callback: Optional[Callable[..., None]] = None,
) -> RlTrainingArtifacts:
    """Train the Step 6_2 on-policy RL branch from a supervised warm start."""

    s4_cfg = dict(run_cfg["s4"])
    if str(s4_cfg.get("rl_checkpoint_selection_mode", "proxy_success_hit_rate")).strip().lower() not in {
        "proxy_success_hit_rate",
        "final_checkpoint",
    }:
        raise NotImplementedError(
            "Step 6_2 RL currently supports only rl_checkpoint_selection_mode in "
            "{'proxy_success_hit_rate', 'final_checkpoint'}."
        )

    policy_model = deepcopy(warm_start.diffusion_model).to(device)
    reference_model = deepcopy(warm_start.diffusion_model).to(device)
    if hasattr(policy_model.backbone, "set_gradient_checkpointing"):
        policy_model.backbone.set_gradient_checkpointing(bool(s4_cfg.get("gradient_checkpointing", True)))
    if hasattr(reference_model.backbone, "set_gradient_checkpointing"):
        reference_model.backbone.set_gradient_checkpointing(False)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad_(False)

    optimizer, scheduler = build_optimizer_and_scheduler(
        modules={"policy_model": policy_model},
        learning_rate=float(s4_cfg["learning_rate"]),
        weight_decay=float(s4_cfg["weight_decay"]),
        warmup_steps=int(s4_cfg["warmup_steps"]),
        max_steps=int(s4_cfg["rl_num_steps"]),
        warmup_schedule=str(s4_cfg["warmup_schedule"]),
        lr_schedule=str(s4_cfg["lr_schedule"]),
    )

    rng = np.random.default_rng(int(resolved.step6_2["random_seed"]))
    reward_weights = dict(s4_cfg["reward_weights"])
    best_checkpoint_path = run_dirs["checkpoints_dir"] / "aligned_rl_best.pt"
    last_checkpoint_path = run_dirs["checkpoints_dir"] / "aligned_rl_last.pt"
    history_rows: List[Dict[str, Any]] = []
    proxy_rows: List[Dict[str, Any]] = []
    best_proxy_success = float("-inf")
    sample_id_start = 1

    for step_idx in range(1, int(s4_cfg["rl_num_steps"]) + 1):
        prompt_df = _sample_prompt_rows(
            resolved,
            run_cfg,
            batch_size=int(s4_cfg["trajectories_per_batch"]),
            rng=rng,
        ).reset_index(drop=True)
        prompt_df["sampling_seed"] = int(step_idx)
        condition_bundle = _prompt_df_to_condition_tensor(prompt_df, scaler=warm_start.scaler, device=device)

        trajectory_sampler = _create_trajectory_sampler(
            diffusion_model=policy_model,
            tokenizer=warm_start.tokenizer,
            resolved=resolved,
            prior=prior,
            condition_bundle=condition_bundle,
            cfg_scale=float(s4_cfg["cfg_scale"]),
            device=device,
        )
        smiles, trajectories, rollout_meta = sample_trajectories_with_class_prior(
            sampler=trajectory_sampler,
            tokenizer=warm_start.tokenizer,
            prior=prior,
            resolved=resolved,
            num_samples=int(s4_cfg["trajectories_per_batch"]),
            show_progress=False,
        )

        rollout_df = _build_rollout_frame(
            smiles=smiles,
            prompt_df=prompt_df,
            sample_id_start=sample_id_start,
            run_name=str(run_cfg["run_name"]),
            canonical_family=str(run_cfg["canonical_family"]),
            step_idx=step_idx,
        )
        sample_id_start += len(rollout_df)
        evaluation_df = evaluate_generated_samples(rollout_df, evaluator).sort_values("sample_id").reset_index(drop=True)
        rewards, reward_metrics = compute_success_shaped_rewards(
            evaluation_df,
            reward_weights=reward_weights,
            sol_log_prob_floor=float(s4_cfg["sol_log_prob_floor"]),
        )
        advantages = rewards - rewards.mean()

        optimizer.zero_grad(set_to_none=True)
        total_policy_term = torch.tensor(0.0, dtype=torch.float32, device=device)
        total_kl_term = torch.tensor(0.0, dtype=torch.float32, device=device)
        offset = 0
        for trajectory in trajectories:
            batch_size = int(trajectory.final_ids.shape[0])
            advantage_chunk = advantages[offset : offset + batch_size].to(device=device, dtype=torch.float32)
            replay = trajectory_sampler.replay_trajectory_logprob(trajectory, grad_enabled=True)
            kl_stats = trajectory_sampler.compute_trajectory_kl(
                trajectory,
                reference_diffusion_model=reference_model,
            )
            total_policy_term = total_policy_term - (advantage_chunk.detach() * replay["trajectory_logprob"]).sum()
            total_kl_term = total_kl_term + kl_stats["trajectory_kl"].sum()
            offset += batch_size

        denom = max(1, len(evaluation_df))
        loss = (total_policy_term + float(s4_cfg["kl_weight"]) * total_kl_term) / float(denom)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history_row = {
            "step_idx": int(step_idx),
            "rollout_batch_size": int(len(evaluation_df)),
            "loss": float(loss.item()),
            "baseline_reward": float(rewards.mean().item()) if len(rewards) else float("nan"),
            "trajectory_logprob_mean": float((-total_policy_term / max(1, len(evaluation_df))).item()),
            "trajectory_kl_mean": float((total_kl_term / max(1, len(evaluation_df))).item()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "w_success": float(reward_weights.get("w_success", 0.0)),
            "kl_weight": float(s4_cfg["kl_weight"]),
            **reward_metrics,
            **{
                "motif_count": int(rollout_meta.get("motif_count", 0)),
                "class_token_bias_enabled": int(bool(rollout_meta.get("class_token_bias_enabled", False))),
            },
        }

        checkpoint_mode = str(s4_cfg.get("rl_checkpoint_selection_mode", "proxy_success_hit_rate")).strip().lower()
        should_eval_proxy = (
            checkpoint_mode == "proxy_success_hit_rate"
            and (
                step_idx % int(s4_cfg["rl_proxy_eval_interval_steps"]) == 0
                or step_idx == int(s4_cfg["rl_num_steps"])
            )
        )
        if should_eval_proxy:
            proxy_success, proxy_eval_df = _evaluate_proxy_success_hit_rate(
                resolved=resolved,
                run_cfg=run_cfg,
                policy_model=policy_model,
                tokenizer=warm_start.tokenizer,
                scaler=warm_start.scaler,
                prior=prior,
                evaluator=evaluator,
                device=device,
                step_idx=step_idx,
            )
            proxy_rows.append(
                {
                    "step_idx": int(step_idx),
                    "proxy_success_hit_rate": float(proxy_success),
                    "proxy_num_samples": int(len(proxy_eval_df)),
                }
            )
            history_row["proxy_success_hit_rate"] = float(proxy_success)
            if np.isfinite(proxy_success) and proxy_success > best_proxy_success:
                best_proxy_success = float(proxy_success)
                _save_rl_checkpoint(
                    checkpoint_path=best_checkpoint_path,
                    policy_model=policy_model,
                    reference_checkpoint_path=warm_start.checkpoint_path,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step_idx=step_idx,
                    best_proxy_success_hit_rate=best_proxy_success,
                    run_cfg=run_cfg,
                    warm_start=warm_start,
                )
            if pruning_callback is not None and np.isfinite(proxy_success):
                pruning_callback(
                    stage="rl",
                    step=int(step_idx),
                    value=float(proxy_success),
                    metrics={**history_row, "proxy_success_hit_rate": float(proxy_success)},
                )

        history_rows.append(history_row)

    _save_rl_checkpoint(
        checkpoint_path=last_checkpoint_path,
        policy_model=policy_model,
        reference_checkpoint_path=warm_start.checkpoint_path,
        optimizer=optimizer,
        scheduler=scheduler,
        step_idx=int(s4_cfg["rl_num_steps"]),
        best_proxy_success_hit_rate=best_proxy_success,
        run_cfg=run_cfg,
        warm_start=warm_start,
    )

    checkpoint_mode = str(s4_cfg.get("rl_checkpoint_selection_mode", "proxy_success_hit_rate")).strip().lower()
    if checkpoint_mode == "proxy_success_hit_rate" and best_checkpoint_path.exists():
        load_step62_checkpoint_into_modules(
            checkpoint_path=best_checkpoint_path,
            diffusion_model=policy_model,
            aux_heads=None,
            device=device,
        )
    elif checkpoint_mode == "final_checkpoint":
        load_step62_checkpoint_into_modules(
            checkpoint_path=last_checkpoint_path,
            diffusion_model=policy_model,
            aux_heads=None,
            device=device,
        )

    history_df = pd.DataFrame(history_rows)
    proxy_history_df = pd.DataFrame(proxy_rows)
    history_df.to_csv(run_dirs["metrics_dir"] / "rl_training_history.csv", index=False)
    if not proxy_history_df.empty:
        proxy_history_df.to_csv(run_dirs["metrics_dir"] / "rl_proxy_history.csv", index=False)
    with open(run_dirs["metrics_dir"] / "rl_training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_name": str(run_cfg["run_name"]),
                "rl_prompt_source": str(s4_cfg["rl_prompt_source"]),
                "best_proxy_success_hit_rate": float(best_proxy_success),
                "checkpoint_selection_mode": checkpoint_mode,
                "best_checkpoint_path": str(best_checkpoint_path),
                "last_checkpoint_path": str(last_checkpoint_path),
                "num_history_rows": int(len(history_df)),
                "num_proxy_evals": int(len(proxy_history_df)),
            },
            handle,
            indent=2,
        )

    return RlTrainingArtifacts(
        tokenizer=warm_start.tokenizer,
        policy_model=policy_model,
        reference_model=reference_model,
        checkpoint_path=best_checkpoint_path if best_checkpoint_path.exists() else last_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        scaler=warm_start.scaler,
        history_df=history_df,
        proxy_history_df=proxy_history_df,
    )
