"""Conditional discrete masking diffusion for Step 5."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from src.model.diffusion import DiscreteMaskingDiffusion

from .conditional_dit import ConditionalDiffusionBackbone


class ConditionalDiscreteMaskingDiffusion(DiscreteMaskingDiffusion):
    """Discrete masking diffusion wrapper with Step 5 conditional inputs."""

    backbone: ConditionalDiffusionBackbone

    def __init__(
        self,
        *,
        backbone: ConditionalDiffusionBackbone,
        condition_dropout_rate: float,
        **kwargs,
    ):
        super().__init__(backbone=backbone, **kwargs)
        self.condition_dropout_rate = float(condition_dropout_rate)

    def sample_condition_drop_mask(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        if self.condition_dropout_rate <= 0.0:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        return torch.rand(batch_size, device=device) < self.condition_dropout_rate

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        *,
        condition_bundle: torch.Tensor,
        condition_drop_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = input_ids.shape[0]
        device = input_ids.device
        if timesteps is None:
            timesteps = torch.randint(1, self.num_steps + 1, (batch_size,), device=device)
        if condition_drop_mask is None:
            if self.training:
                condition_drop_mask = self.sample_condition_drop_mask(batch_size, device=device)
            else:
                condition_drop_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        noisy_ids, mask_indicator = self.forward_process(input_ids, timesteps, attention_mask)
        logits = self.backbone(
            noisy_ids,
            timesteps,
            attention_mask,
            condition_bundle=condition_bundle,
            condition_drop_mask=condition_drop_mask,
        )
        loss = self._compute_loss(logits, input_ids, mask_indicator, attention_mask)
        return {
            "loss": loss,
            "logits": logits,
            "noisy_ids": noisy_ids,
            "mask_indicator": mask_indicator,
            "timesteps": timesteps,
            "condition_drop_mask": condition_drop_mask,
        }

    def conditional_logits_impl(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        *,
        condition_bundle: torch.Tensor,
        condition_drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if condition_drop_mask is None:
            condition_drop_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        return self.backbone(
            input_ids,
            timesteps,
            attention_mask,
            condition_bundle=condition_bundle,
            condition_drop_mask=condition_drop_mask,
        )

    def classifier_free_guidance_logits_impl(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        *,
        condition_bundle: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        logits_cond = self.backbone(
            input_ids,
            timesteps,
            attention_mask,
            condition_bundle=condition_bundle,
            condition_drop_mask=torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device),
        )
        if float(cfg_scale) == 0.0:
            return logits_cond
        logits_uncond = self.backbone(
            input_ids,
            timesteps,
            attention_mask,
            condition_bundle=condition_bundle,
            condition_drop_mask=torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device),
        )
        return (1.0 + float(cfg_scale)) * logits_cond - float(cfg_scale) * logits_uncond

    @torch.no_grad()
    def conditional_logits(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        *,
        condition_bundle: torch.Tensor,
        condition_drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.conditional_logits_impl(
            input_ids,
            timesteps,
            attention_mask,
            condition_bundle=condition_bundle,
            condition_drop_mask=condition_drop_mask,
        )

    @torch.no_grad()
    def classifier_free_guidance_logits(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        *,
        condition_bundle: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        return self.classifier_free_guidance_logits_impl(
            input_ids,
            timesteps,
            attention_mask,
            condition_bundle=condition_bundle,
            cfg_scale=cfg_scale,
        )
