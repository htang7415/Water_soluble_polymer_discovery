from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DiTConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    ffn_hidden_size: int
    dropout: float
    max_position_embeddings: int
    num_steps: int


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=device).float() * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.proj(emb)


class DiT(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.time_embed = SinusoidalTimeEmbedding(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_hidden_size,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        time_emb = self.time_embed(timesteps)
        x = x + time_emb.unsqueeze(1)
        x = self.dropout(x)
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = self.forward_hidden(input_ids, timesteps, attention_mask)
        return self.lm_head(hidden)

    def freeze_backbone(self) -> None:
        for p in self.token_embed.parameters():
            p.requires_grad = False
        for p in self.pos_embed.parameters():
            p.requires_grad = False
        for p in self.time_embed.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_last_blocks(self, k: int) -> None:
        layers = list(self.encoder.layers)
        total = len(layers)
        k = max(0, min(k, total))
        for i, layer in enumerate(layers):
            requires = i >= total - k
            for p in layer.parameters():
                p.requires_grad = requires

    def set_freeze_mode(self, k: int) -> None:
        self.freeze_backbone()
        self.unfreeze_last_blocks(k)
        # embeddings remain frozen unless k == max_blocks
        if k >= self.config.num_layers:
            for p in self.token_embed.parameters():
                p.requires_grad = True
            for p in self.pos_embed.parameters():
                p.requires_grad = True
            for p in self.time_embed.parameters():
                p.requires_grad = True
