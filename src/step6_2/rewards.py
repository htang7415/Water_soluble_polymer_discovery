"""Reward helpers for Step 6_2 guidance and alignment."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src.step6_2.evaluation import Step62Evaluator
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
    class_surrogate_mode: str,
    class_term_enabled: bool,
    sol_log_prob_floor: float,
    w_sol: float,
    w_chi: float,
    w_class: float,
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
    class_surrogate_vals: List[float] = []
    target_class = str(target_row["c_target"])
    mode = str(class_surrogate_mode).strip().lower()
    if mode != "binary_exact":
        raise NotImplementedError(
            f"class_surrogate_mode={class_surrogate_mode!r} is not implemented in the current S1 increment."
        )
    for smi in smiles:
        is_valid = bool(check_validity(smi))
        valid_mask.append(int(is_valid))
        if not is_valid:
            class_surrogate_vals.append(0.0)
            continue
        matches = evaluator.polymer_classifier.classify(smi)
        class_surrogate_vals.append(1.0 if matches.get(target_class, False) else 0.0)

    valid_tensor = torch.tensor(valid_mask, dtype=torch.float32)
    class_surrogate = torch.tensor(class_surrogate_vals, dtype=torch.float32)
    valid_frac = float(valid_tensor.mean().item()) if len(valid_tensor) else 0.0

    sol_term = torch.log(class_prob.clamp(min=np.exp(float(sol_log_prob_floor))))
    sol_term = torch.maximum(sol_term, torch.full_like(sol_term, float(sol_log_prob_floor)))

    property_rule = str(target_row.get("property_rule", "upper_bound")).strip().lower()
    chi_target = float(target_row["chi_target"])
    if property_rule == "lower_bound":
        chi_term = -torch.relu(torch.tensor(float(chi_target)) - chi_pred)
    elif property_rule == "band":
        chi_term = -torch.abs(chi_pred - float(chi_target))
    else:
        chi_term = -torch.relu(chi_pred - float(chi_target))

    reward = float(w_sol) * sol_term + float(w_chi) * chi_term
    if class_term_enabled:
        reward = reward + float(w_class) * class_surrogate

    return {
        "reward": reward.cpu(),
        "smiles": smiles,
        "valid_frac": valid_frac,
        "class_surrogate": class_surrogate,
        "class_term_enabled": bool(class_term_enabled),
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
    chi_term_values: List[float] = []
    for chi_pred_value, chi_target_value, property_rule in zip(chi_pred, chi_target, property_rules):
        if not np.isfinite(chi_pred_value):
            chi_term_values.append(-abs(float(chi_target_value)) if np.isfinite(chi_target_value) else -1.0)
            continue
        rule = str(property_rule).strip().lower()
        if rule == "lower_bound":
            chi_term_values.append(-max(0.0, float(chi_target_value) - float(chi_pred_value)))
        elif rule == "band":
            chi_term_values.append(-abs(float(chi_pred_value) - float(chi_target_value)))
        else:
            chi_term_values.append(-max(0.0, float(chi_pred_value) - float(chi_target_value)))
    chi_term = torch.tensor(np.asarray(chi_term_values, dtype=np.float32), dtype=torch.float32)

    reward = (
        float(reward_weights.get("w_success", 0.0)) * torch.tensor(evaluation_df["success_hit"].to_numpy(dtype=np.float32))
        + float(reward_weights.get("w_valid", 0.0)) * torch.tensor(evaluation_df["valid_ok"].to_numpy(dtype=np.float32))
        + float(reward_weights.get("w_novel", 0.0)) * torch.tensor(evaluation_df["novel_ok"].to_numpy(dtype=np.float32))
        + float(reward_weights.get("w_star", 0.0)) * torch.tensor(evaluation_df["star_ok"].to_numpy(dtype=np.float32))
        + float(reward_weights.get("w_sa", 0.0)) * torch.tensor(evaluation_df["sa_ok"].to_numpy(dtype=np.float32))
        + float(reward_weights.get("w_class", 0.0)) * torch.tensor(evaluation_df["class_ok"].to_numpy(dtype=np.float32))
        + float(reward_weights.get("w_sol", 0.0)) * sol_term
        + float(reward_weights.get("w_chi", 0.0)) * chi_term
    )

    valid_count = int(evaluation_df["valid_ok"].astype(int).sum())
    metrics = {
        "reward_mean": float(reward.mean().item()) if len(reward) else float("nan"),
        "reward_std": float(reward.std(unbiased=False).item()) if len(reward) else float("nan"),
        "success_rate": float(evaluation_df["success_hit"].astype(float).mean()) if len(evaluation_df) else float("nan"),
        "training_soluble_oracle_calls": valid_count,
        "training_chi_oracle_calls": valid_count,
    }
    return reward, metrics
