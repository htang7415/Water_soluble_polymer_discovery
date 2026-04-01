"""Conditional DiT backbone for Step 6_2."""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.model.backbone import DiffusionBackbone

from .condition_encoder import ConditionEncoder


class ConditionalDiffusionBackbone(nn.Module):
    """Step 1 backbone plus per-block shift-only conditional modulation."""

    def __init__(
        self,
        pretrained_backbone: DiffusionBackbone,
        *,
        condition_encoder: ConditionEncoder,
        modulate_final_norm: bool = False,
    ):
        super().__init__()
        backbone = deepcopy(pretrained_backbone)
        self.vocab_size = backbone.vocab_size
        self.hidden_size = backbone.hidden_size
        self.num_layers = backbone.num_layers
        self.pad_token_id = backbone.pad_token_id
        self.token_embedding = backbone.token_embedding
        self.position_embedding = backbone.position_embedding
        self.timestep_embedding = backbone.timestep_embedding
        self.embedding_dropout = backbone.embedding_dropout
        self.layers = backbone.layers
        self.final_norm = backbone.final_norm
        self.output_proj = backbone.output_proj

        self.condition_encoder = condition_encoder
        self.condition_shift_projections = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size * 2) for _ in range(self.num_layers)]
        )
        self.modulate_final_norm = bool(modulate_final_norm)
        self.final_norm_shift = nn.Linear(self.hidden_size, self.hidden_size) if self.modulate_final_norm else None
        self.gradient_checkpointing = False
        self._init_condition_path()

    def _init_condition_path(self) -> None:
        for projection in self.condition_shift_projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        if self.final_norm_shift is not None:
            nn.init.zeros_(self.final_norm_shift.weight)
            nn.init.zeros_(self.final_norm_shift.bias)

    def _compute_condition_embedding(
        self,
        condition_bundle: torch.Tensor,
        condition_drop_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.condition_encoder(
            condition_bundle=condition_bundle,
            condition_drop_mask=condition_drop_mask,
        )

    def configure_backbone_finetune(
        self,
        finetune_last_layers: Optional[int | str],
    ) -> dict:
        """Apply Step 6_2 backbone freeze policy.

        Semantics:
        - ``None`` or ``"full"``: full backbone fine-tune, matching the legacy Step 6_2 default.
        - integer ``k`` in ``[0, num_layers]``:
          train the last ``k`` transformer blocks plus ``final_norm`` and ``output_proj``;
          keep pretrained token/position/timestep embeddings frozen.
        - ``0`` means the pretrained Step 1 backbone is fully frozen and only the Step 6_2
          condition path remains trainable.
        """

        def _set_module_trainable(module: Optional[nn.Module], enabled: bool) -> None:
            if module is None:
                return
            for param in module.parameters():
                param.requires_grad = bool(enabled)

        # Step 6_2 condition path is always trainable.
        _set_module_trainable(self.condition_encoder, True)
        _set_module_trainable(self.condition_shift_projections, True)
        _set_module_trainable(self.final_norm_shift, True)

        if finetune_last_layers is None or str(finetune_last_layers).strip().lower() == "full":
            for module in [
                self.token_embedding,
                self.position_embedding,
                self.timestep_embedding,
                self.layers,
                self.final_norm,
                self.output_proj,
            ]:
                _set_module_trainable(module, True)
            return {
                "backbone_num_layers": int(self.num_layers),
                "finetune_last_layers": None,
                "backbone_finetune_mode": "full",
                "backbone_finetune_enabled": True,
            }

        n_layers = int(self.num_layers)
        k = int(finetune_last_layers)
        if k < 0 or k > n_layers:
            raise ValueError(
                f"finetune_last_layers must be null/'full' or an integer in [0, {n_layers}], got {finetune_last_layers!r}"
            )

        for module in [
            self.token_embedding,
            self.position_embedding,
            self.timestep_embedding,
            self.layers,
            self.final_norm,
            self.output_proj,
        ]:
            _set_module_trainable(module, False)
        if k > 0:
            for layer in self.layers[-k:]:
                _set_module_trainable(layer, True)
            _set_module_trainable(self.final_norm, True)
            _set_module_trainable(self.output_proj, True)
        return {
            "backbone_num_layers": int(self.num_layers),
            "finetune_last_layers": int(k),
            "backbone_finetune_mode": "last_n_layers",
            "backbone_finetune_enabled": bool(k > 0),
        }

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = bool(enabled)

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        condition_bundle: torch.Tensor,
        condition_drop_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        if timesteps.dim() == 2:
            timesteps = timesteps.squeeze(-1)
        time_emb = self.timestep_embedding(timesteps).unsqueeze(1)

        x = token_emb + pos_emb + time_emb
        x = self.embedding_dropout(x)

        cond = self._compute_condition_embedding(condition_bundle, condition_drop_mask)
        for layer_idx, layer in enumerate(self.layers):
            def _layer_forward(hidden: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
                shift1, shift2 = self.condition_shift_projections[layer_idx](cond_embed).chunk(2, dim=-1)
                h1 = layer.norm1(hidden) + shift1.unsqueeze(1)
                hidden = hidden + layer.attn(h1, attention_mask)
                h2 = layer.norm2(hidden) + shift2.unsqueeze(1)
                hidden = hidden + layer.ffn(h2)
                return hidden

            if self.gradient_checkpointing and torch.is_grad_enabled():
                x = checkpoint(_layer_forward, x, cond, use_reentrant=False)
            else:
                x = _layer_forward(x, cond)

        x = self.final_norm(x)
        if self.final_norm_shift is not None:
            x = x + self.final_norm_shift(cond).unsqueeze(1)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        condition_bundle: torch.Tensor,
        condition_drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = self._forward_hidden(
            input_ids=input_ids,
            timesteps=timesteps,
            attention_mask=attention_mask,
            condition_bundle=condition_bundle,
            condition_drop_mask=condition_drop_mask,
        )
        return self.output_proj(hidden)

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        condition_bundle: torch.Tensor,
        condition_drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._forward_hidden(
            input_ids=input_ids,
            timesteps=timesteps,
            attention_mask=attention_mask,
            condition_bundle=condition_bundle,
            condition_drop_mask=condition_drop_mask,
        )

    def get_pooled_output(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        condition_bundle: torch.Tensor,
        condition_drop_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
    ) -> torch.Tensor:
        hidden = self.get_hidden_states(
            input_ids=input_ids,
            timesteps=timesteps,
            attention_mask=attention_mask,
            condition_bundle=condition_bundle,
            condition_drop_mask=condition_drop_mask,
        )
        if pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                hidden = hidden * mask
                return hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return hidden.mean(dim=1)
        if pooling == "cls":
            return hidden[:, 0]
        if pooling == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                hidden = hidden * mask + (1.0 - mask) * (-1.0e9)
            return hidden.max(dim=1)[0]
        raise ValueError(f"Unknown pooling method: {pooling}")
