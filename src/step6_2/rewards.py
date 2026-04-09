"""Reward helpers for Step 6_2 guidance and alignment."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src.step6_2.evaluation import (
    Step62Evaluator,
    compute_chi_penalty_from_bounds,
    resolve_effective_chi_bounds,
)
from src.utils.chemistry import check_validity


def predict_step4_scores_from_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    temperature: float,
    phi: float,
    evaluator: Step62Evaluator,
) -> Dict[str, torch.Tensor]:
    """Run Step 4 oracles directly on token ids."""

    cache = evaluator.inference_cache
    device = input_ids.device
    batch_size = int(input_ids.shape[0])

    reg_model = cache["reg_model"]
    cls_model = cache["cls_model"]
    step1_backbone = cache.get("step1_backbone")
    reg_needs_step1_embeddings = bool(cache["reg_needs_step1_embeddings"])
    cls_needs_step1_embeddings = bool(cache["cls_needs_step1_embeddings"])
    reg_finetune_last_layers = int(cache["reg_finetune_last_layers"])
    cls_finetune_last_layers = int(cache["cls_finetune_last_layers"])
    reg_timestep = int(cache["reg_timestep"])
    cls_timestep = int(cache["cls_timestep"])

    with torch.no_grad():
        emb_reg = None
        emb_cls = None
        if reg_needs_step1_embeddings:
            timesteps = torch.full((batch_size,), reg_timestep, device=device, dtype=torch.long)
            emb_reg = step1_backbone.get_pooled_output(
                input_ids=input_ids,
                timesteps=timesteps,
                attention_mask=attention_mask,
                pooling="mean",
            )
        if cls_needs_step1_embeddings:
            if reg_needs_step1_embeddings and cls_timestep == reg_timestep:
                emb_cls = emb_reg
            else:
                timesteps = torch.full((batch_size,), cls_timestep, device=device, dtype=torch.long)
                emb_cls = step1_backbone.get_pooled_output(
                    input_ids=input_ids,
                    timesteps=timesteps,
                    attention_mask=attention_mask,
                    pooling="mean",
                )

        temp_tensor = torch.full((batch_size,), float(temperature), device=device, dtype=torch.float32)
        phi_tensor = torch.full((batch_size,), float(phi), device=device, dtype=torch.float32)

        if reg_finetune_last_layers > 0:
            reg_out = reg_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=temp_tensor,
                phi=phi_tensor,
            )
        else:
            reg_out = reg_model(
                embedding=emb_reg,
                temperature=temp_tensor,
                phi=phi_tensor,
            )

        if cls_finetune_last_layers > 0:
            cls_out = cls_model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            cls_out = cls_model(embedding=emb_cls)

        class_logit = cls_out["class_logit"]
        class_prob = torch.sigmoid(class_logit)
        chi_pred = reg_out["chi_pred"]
        return {
            "class_logit": class_logit.detach(),
            "class_prob": class_prob.detach(),
            "chi_pred": chi_pred.detach(),
        }


def score_guidance_batch(
    provisional_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    target_row: Dict[str, object],
    evaluator: Step62Evaluator,
    tokenizer,
    sol_log_prob_floor: float,
    w_sol: float,
    w_chi: float,
    invalid_reward_penalty: float,
) -> Dict[str, object]:
    """Score provisional complete sequences for S1 guidance."""

    scores = predict_step4_scores_from_ids(
        provisional_ids,
        attention_mask,
        temperature=float(target_row["temperature"]),
        phi=float(target_row["phi"]),
        evaluator=evaluator,
    )
    class_prob = scores["class_prob"].cpu()
    chi_pred = scores["chi_pred"].cpu()
    smiles = tokenizer.batch_decode(provisional_ids.detach().cpu().tolist(), skip_special_tokens=True)

    valid_mask: List[int] = []
    for smi in smiles:
        is_valid = bool(check_validity(smi))
        valid_mask.append(int(is_valid))

    valid_tensor = torch.tensor(valid_mask, dtype=torch.float32)
    valid_frac = float(valid_tensor.mean().item()) if len(valid_tensor) else 0.0

    sol_term = torch.log(class_prob.clamp(min=np.exp(float(sol_log_prob_floor))))
    sol_term = torch.maximum(sol_term, torch.full_like(sol_term, float(sol_log_prob_floor)))

    property_rule = str(target_row.get("property_rule", "upper_bound")).strip().lower()
    chi_target = float(target_row["chi_target"])
    bounds = resolve_effective_chi_bounds(
        row=target_row,
        chi_target=chi_target,
        property_rule=property_rule,
        epsilon=float(evaluator.resolved.step6_2.get("chi_band_epsilon", 0.25)),
        step6_cfg=evaluator.resolved.step6_2,
    )
    chi_term = torch.tensor(
        [
            compute_chi_penalty_from_bounds(
                float(pred),
                lower_bound=float(bounds["chi_target_effective_lower"]),
                upper_bound=float(bounds["chi_target_effective_upper"]),
            )
            for pred in chi_pred.cpu().numpy().tolist()
        ],
        dtype=torch.float32,
    )

    valid_reward = float(w_sol) * sol_term + float(w_chi) * chi_term
    invalid_penalty = torch.full_like(valid_reward, float(invalid_reward_penalty))
    reward = torch.where(valid_tensor > 0.0, valid_reward, invalid_penalty)
    return {
        "reward": reward.cpu(),
        "smiles": smiles,
        "valid_mask": valid_tensor.cpu(),
        "valid_frac": valid_frac,
        "oracle_calls_soluble": int(provisional_ids.shape[0]),
        "oracle_calls_chi": int(provisional_ids.shape[0]),
    }


def compute_success_shaped_rewards(
    evaluation_df: pd.DataFrame,
    *,
    reward_weights: Dict[str, float],
    sol_log_prob_floor: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the Step 6_2 success-shaped reward on completed samples."""

    class_prob = pd.to_numeric(evaluation_df["class_prob"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    class_prob_tensor = torch.tensor(class_prob.to_numpy(dtype=np.float32), dtype=torch.float32)
    sol_term = torch.log(class_prob_tensor.clamp(min=float(np.exp(float(sol_log_prob_floor)))))
    sol_term = torch.maximum(sol_term, torch.full_like(sol_term, float(sol_log_prob_floor)))

    chi_pred = pd.to_numeric(evaluation_df["chi_pred_target"], errors="coerce").to_numpy(dtype=np.float32)
    chi_target = pd.to_numeric(evaluation_df["chi_target"], errors="coerce").to_numpy(dtype=np.float32)
    property_rules = evaluation_df["property_rule"].astype(str).tolist()
    lower_series = (
        evaluation_df["chi_target_effective_lower"]
        if "chi_target_effective_lower" in evaluation_df.columns
        else pd.Series(np.nan, index=evaluation_df.index)
    )
    upper_series = (
        evaluation_df["chi_target_effective_upper"]
        if "chi_target_effective_upper" in evaluation_df.columns
        else pd.Series(np.nan, index=evaluation_df.index)
    )
    lower_bounds = pd.to_numeric(lower_series, errors="coerce").to_numpy(dtype=np.float32)
    upper_bounds = pd.to_numeric(upper_series, errors="coerce").to_numpy(dtype=np.float32)
    chi_term_values: List[float] = []
    for chi_pred_value, chi_target_value, property_rule, lower_bound, upper_bound in zip(
        chi_pred,
        chi_target,
        property_rules,
        lower_bounds,
        upper_bounds,
    ):
        if not np.isfinite(chi_pred_value):
            chi_term_values.append(-abs(float(chi_target_value)) if np.isfinite(chi_target_value) else -1.0)
            continue
        rule = str(property_rule).strip().lower()
        lower = float(lower_bound)
        upper = float(upper_bound)
        if not np.isfinite(lower) and rule in {"lower_bound", "band"}:
            lower = float(chi_target_value) if rule == "lower_bound" else float(chi_target_value)
        if not np.isfinite(upper) and rule in {"upper_bound", "band"}:
            upper = float(chi_target_value) if rule == "upper_bound" else float(chi_target_value)
        chi_term_values.append(
            compute_chi_penalty_from_bounds(
                float(chi_pred_value),
                lower_bound=lower,
                upper_bound=upper,
            )
        )
    chi_term = torch.tensor(np.asarray(chi_term_values, dtype=np.float32), dtype=torch.float32)

    success_col = (
        "property_success_hit_discovery"
        if "property_success_hit_discovery" in evaluation_df.columns
        else "property_success_hit"
    )
    sa_col = "sa_ok_discovery" if "sa_ok_discovery" in evaluation_df.columns else "sa_ok"
    if "target_sa_max_discovery" in evaluation_df.columns:
        sa_threshold = pd.to_numeric(evaluation_df["target_sa_max_discovery"], errors="coerce").to_numpy(dtype=np.float32)
    elif "target_sa_max_reporting" in evaluation_df.columns:
        sa_threshold = pd.to_numeric(evaluation_df["target_sa_max_reporting"], errors="coerce").to_numpy(dtype=np.float32)
    else:
        sa_threshold = np.full(len(evaluation_df), np.nan, dtype=np.float32)
    sa_score = pd.to_numeric(evaluation_df.get("sa_score"), errors="coerce").to_numpy(dtype=np.float32)
    sa_continuous_values: List[float] = []
    for score_value, threshold_value in zip(sa_score, sa_threshold):
        if not np.isfinite(score_value) or not np.isfinite(threshold_value) or float(threshold_value) <= 0.0:
            sa_continuous_values.append(-1.0)
            continue
        raw_value = (float(threshold_value) - float(score_value)) / float(threshold_value)
        sa_continuous_values.append(max(-1.0, min(1.0, raw_value)))
    sa_continuous = torch.tensor(np.asarray(sa_continuous_values, dtype=np.float32), dtype=torch.float32)

    reward = (
        float(reward_weights.get("w_success", 0.0)) * torch.tensor(evaluation_df[success_col].to_numpy(dtype=np.float32))
        + float(reward_weights.get("w_valid", 0.0)) * torch.tensor(evaluation_df["valid_ok"].to_numpy(dtype=np.float32))
        + float(reward_weights.get("w_novel", 0.0)) * torch.tensor(evaluation_df["novel_ok"].to_numpy(dtype=np.float32))
        + float(reward_weights.get("w_star", 0.0)) * torch.tensor(evaluation_df["star_ok"].to_numpy(dtype=np.float32))
        + float(reward_weights.get("w_sa", 0.0)) * torch.tensor(evaluation_df[sa_col].to_numpy(dtype=np.float32))
        + float(reward_weights.get("w_sa_continuous", 0.0)) * sa_continuous
        + float(reward_weights.get("w_sol", 0.0)) * sol_term
        + float(reward_weights.get("w_chi", 0.0)) * chi_term
    )

    valid_count = int(evaluation_df["valid_ok"].astype(int).sum())
    metrics = {
        "reward_mean": float(reward.mean().item()) if len(reward) else float("nan"),
        "reward_std": float(reward.std(unbiased=False).item()) if len(reward) else float("nan"),
        "success_rate": float(evaluation_df[success_col].astype(float).mean()) if len(evaluation_df) else float("nan"),
        "reward_success_metric": str(success_col),
        "reward_sa_metric": str(sa_col),
        "sa_continuous_mean": float(sa_continuous.mean().item()) if len(sa_continuous) else float("nan"),
        "training_soluble_oracle_calls": valid_count,
        "training_chi_oracle_calls": valid_count,
    }
    return reward, metrics
