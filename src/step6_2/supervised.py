"""Supervised Step 6_2 model-construction and optimization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR, LinearLR, SequentialLR

from src.chi.embeddings import load_backbone_from_step1
from src.data.tokenizer import PSmilesTokenizer

from .condition_encoder import ConditionEncoder
from .conditional_diffusion import ConditionalDiscreteMaskingDiffusion
from .conditional_dit import ConditionalDiffusionBackbone
from .config import ResolvedStep62Config
from .dataset import get_step62_condition_dim


class Step62AuxHeads(nn.Module):
    """Optional S2-mt auxiliary heads on pooled conditional backbone output."""

    soluble_head: nn.Module
    chi_head: nn.Module

    def __init__(self, hidden_size: int):
        super().__init__()
        self.soluble_head = nn.Linear(hidden_size, 1)
        self.chi_head = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.soluble_head.weight)
        nn.init.zeros_(self.soluble_head.bias)
        nn.init.xavier_uniform_(self.chi_head.weight)
        nn.init.zeros_(self.chi_head.bias)

    def forward(self, pooled_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "soluble_logit": self.soluble_head(pooled_output).squeeze(-1),
            "chi_pred": self.chi_head(pooled_output).squeeze(-1),
        }


def build_s2_components_from_step1(
    resolved: ResolvedStep62Config,
    *,
    device: str,
    run_cfg: Optional[Dict[str, object]] = None,
) -> Tuple[
    PSmilesTokenizer,
    ConditionalDiscreteMaskingDiffusion,
    Optional[Step62AuxHeads],
    Path,
    Dict[str, object],
]:
    """Load Step 1 backbone and wrap it into the Step 6_2 conditional model."""

    try:
        tokenizer, pretrained_backbone, checkpoint_path = load_backbone_from_step1(
            config=resolved.base_config,
            model_size=resolved.model_size,
            split_mode=resolved.split_mode,
            checkpoint_path=None,
            device=device,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Step 6_2 supervised training requires an existing Step 1 backbone checkpoint and tokenizer. "
            f"model_size={resolved.model_size!r}, split_mode={resolved.split_mode!r}. "
            f"Original error: {exc}"
        ) from exc
    hidden_size = int(pretrained_backbone.hidden_size)
    condition_dim = get_step62_condition_dim(resolved.available_target_classes)
    condition_encoder = ConditionEncoder(
        input_dim=condition_dim,
        hidden_dim=hidden_size,
        output_dim=hidden_size,
    ).to(device)
    conditional_backbone = ConditionalDiffusionBackbone(
        pretrained_backbone=pretrained_backbone,
        condition_encoder=condition_encoder,
        modulate_final_norm=False,
    ).to(device)
    s2_cfg = (run_cfg or resolved.step6_2).get("s2", {})
    finetune_info = conditional_backbone.configure_backbone_finetune(s2_cfg.get("finetune_last_layers"))
    diffusion = ConditionalDiscreteMaskingDiffusion(
        backbone=conditional_backbone,
        condition_dropout_rate=float(s2_cfg.get("condition_dropout_rate", 0.0)),
        num_steps=resolved.base_config["diffusion"]["num_steps"],
        beta_min=resolved.base_config["diffusion"]["beta_min"],
        beta_max=resolved.base_config["diffusion"]["beta_max"],
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ).to(device)
    variant = str(s2_cfg.get("variant", "pure")).strip().lower()
    aux_heads = Step62AuxHeads(hidden_size).to(device) if variant == "mt" else None
    return tokenizer, diffusion, aux_heads, checkpoint_path, finetune_info


def load_step62_checkpoint_into_modules(
    *,
    checkpoint_path: Path,
    diffusion_model: ConditionalDiscreteMaskingDiffusion,
    aux_heads: Optional[Step62AuxHeads],
    device: str,
) -> Dict[str, object]:
    """Restore Step 6_2 supervised or aligned weights into live modules."""

    payload = torch.load(checkpoint_path, map_location=device)
    diffusion_model.load_state_dict(payload["model_state_dict"])
    aux_state = payload.get("aux_state_dict")
    if aux_heads is not None and aux_state is not None:
        aux_heads.load_state_dict(aux_state)
    return payload


def build_optimizer_and_scheduler(
    modules: Dict[str, nn.Module],
    *,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    max_steps: int,
    warmup_schedule: str,
    lr_schedule: str,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Build the benchmark-default AdamW + linear-warmup scheduler stack."""

    params = []
    for module in modules.values():
        params.extend([param for param in module.parameters() if param.requires_grad])
    optimizer = AdamW(params, lr=float(learning_rate), weight_decay=float(weight_decay))

    warmup_schedule = str(warmup_schedule).strip().lower()
    lr_schedule = str(lr_schedule).strip().lower()
    if warmup_schedule != "linear":
        raise ValueError(f"Unsupported Step 6_2 warmup_schedule: {warmup_schedule}")
    if lr_schedule != "constant_after_warmup":
        raise ValueError(f"Unsupported Step 6_2 lr_schedule: {lr_schedule}")

    warmup_steps = max(0, int(warmup_steps))
    max_steps = max(1, int(max_steps))
    if warmup_steps == 0:
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=max_steps)
        return optimizer, scheduler

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    constant_scheduler = ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=max(1, max_steps - warmup_steps),
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, constant_scheduler],
        milestones=[warmup_steps],
    )
    return optimizer, scheduler


def compute_s2_mt_losses(
    *,
    pooled_output: torch.Tensor,
    aux_heads: Step62AuxHeads,
    soluble_target: torch.Tensor,
    chi_target_aux: torch.Tensor,
    soluble_loss_weight: float,
    chi_loss_weight: float,
) -> Dict[str, torch.Tensor]:
    """Compute optional S2-mt auxiliary losses."""

    head_out = aux_heads(pooled_output)
    bce = nn.BCEWithLogitsLoss()
    smooth_l1 = nn.SmoothL1Loss()
    soluble_loss = bce(head_out["soluble_logit"], soluble_target.float())

    chi_mask = torch.isfinite(chi_target_aux)
    if bool(chi_mask.any()):
        chi_loss = smooth_l1(head_out["chi_pred"][chi_mask], chi_target_aux[chi_mask].float())
    else:
        chi_loss = pooled_output.new_tensor(0.0)

    total = float(soluble_loss_weight) * soluble_loss + float(chi_loss_weight) * chi_loss
    return {
        "aux_total_loss": total,
        "aux_soluble_loss": soluble_loss,
        "aux_chi_loss": chi_loss,
        "soluble_logit": head_out["soluble_logit"],
        "chi_pred": head_out["chi_pred"],
    }
