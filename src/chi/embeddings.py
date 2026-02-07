"""Backbone embedding cache utilities for chi modeling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.data.tokenizer import PSmilesTokenizer
from src.model.backbone import DiffusionBackbone
from src.model.diffusion import DiscreteMaskingDiffusion
from src.utils.model_scales import get_model_config, get_results_dir



def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()



def _normalize_checkpoint_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict



def _make_cache_key(
    polymer_df: pd.DataFrame,
    checkpoint_path: Path,
    model_size: Optional[str],
    timestep: int,
    pooling: str,
) -> str:
    records = polymer_df[["polymer_id", "Polymer", "SMILES", "water_soluble"]].sort_values("polymer_id")
    payload = {
        "checkpoint": str(checkpoint_path.resolve()),
        "model_size": model_size or "base",
        "timestep": int(timestep),
        "pooling": pooling,
        "records": records.to_dict(orient="records"),
    }
    return _sha256(json.dumps(payload, sort_keys=True))



def load_backbone_from_step1(
    config: Dict,
    model_size: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
):
    """Load tokenizer + backbone encoder from step1 checkpoint."""
    base_results_dir = Path(config["paths"]["results_dir"])
    results_dir = Path(get_results_dir(model_size, config["paths"]["results_dir"]))

    tokenizer_path = results_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = base_results_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    tokenizer = PSmilesTokenizer.load(tokenizer_path)

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
    else:
        ckpt_path = results_dir / "step1_backbone" / "checkpoints" / "backbone_best.pt"
        if not ckpt_path.exists():
            ckpt_path = base_results_dir / "step1_backbone" / "checkpoints" / "backbone_best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {ckpt_path}")

    backbone_cfg = get_model_config(model_size, config, model_type="sequence")
    backbone = DiffusionBackbone(
        vocab_size=tokenizer.vocab_size,
        hidden_size=backbone_cfg["hidden_size"],
        num_layers=backbone_cfg["num_layers"],
        num_heads=backbone_cfg["num_heads"],
        ffn_hidden_size=backbone_cfg["ffn_hidden_size"],
        max_position_embeddings=backbone_cfg["max_position_embeddings"],
        num_diffusion_steps=config["diffusion"]["num_steps"],
        dropout=backbone_cfg["dropout"],
        pad_token_id=tokenizer.pad_token_id,
    )

    diffusion = DiscreteMaskingDiffusion(
        backbone=backbone,
        num_steps=config["diffusion"]["num_steps"],
        beta_min=config["diffusion"]["beta_min"],
        beta_max=config["diffusion"]["beta_max"],
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = _normalize_checkpoint_state(checkpoint["model_state_dict"])
    diffusion.load_state_dict(state_dict)

    backbone = diffusion.backbone.to(device)
    backbone.eval()
    return tokenizer, backbone, ckpt_path


@torch.no_grad()
def compute_polymer_embeddings(
    polymer_df: pd.DataFrame,
    tokenizer: PSmilesTokenizer,
    backbone,
    device: str,
    timestep: int = 1,
    pooling: str = "mean",
    batch_size: int = 128,
) -> pd.DataFrame:
    """Compute pooled backbone embeddings for unique polymers."""
    records = polymer_df[["polymer_id", "Polymer", "SMILES", "water_soluble"]].drop_duplicates("polymer_id")
    records = records.sort_values("polymer_id").reset_index(drop=True)

    all_embeddings: List[np.ndarray] = []
    smiles_list = records["SMILES"].astype(str).tolist()

    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i : i + batch_size]
        encoded = tokenizer.batch_encode(batch_smiles)
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=device)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long, device=device)
        t = torch.full((input_ids.shape[0],), int(timestep), device=device, dtype=torch.long)

        pooled = backbone.get_pooled_output(
            input_ids=input_ids,
            timesteps=t,
            attention_mask=attention_mask,
            pooling=pooling,
        )
        all_embeddings.append(pooled.detach().cpu().numpy())

    emb = np.concatenate(all_embeddings, axis=0)
    out = records.copy()
    out["embedding"] = list(emb)
    return out



def save_embedding_cache(cache_npz: Path, embedding_df: pd.DataFrame, metadata: Dict) -> None:
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_npz,
        polymer_id=embedding_df["polymer_id"].to_numpy(dtype=int),
        polymer=embedding_df["Polymer"].astype(str).to_numpy(),
        smiles=embedding_df["SMILES"].astype(str).to_numpy(),
        water_soluble=embedding_df["water_soluble"].to_numpy(dtype=int),
        embedding=np.stack(embedding_df["embedding"].to_list(), axis=0),
    )
    with open(cache_npz.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=2)



def load_embedding_cache(cache_npz: Path) -> Tuple[pd.DataFrame, Dict]:
    arr = np.load(cache_npz, allow_pickle=False)
    df = pd.DataFrame(
        {
            "polymer_id": arr["polymer_id"].astype(int),
            "Polymer": arr["polymer"].astype(str),
            "SMILES": arr["smiles"].astype(str),
            "water_soluble": arr["water_soluble"].astype(int),
            "embedding": list(arr["embedding"]),
        }
    )
    meta_path = cache_npz.with_suffix(".json")
    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    return df, metadata



def build_or_load_embedding_cache(
    polymer_df: pd.DataFrame,
    config: Dict,
    cache_npz: str | Path,
    model_size: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    timestep: int = 1,
    pooling: str = "mean",
    batch_size: int = 128,
) -> pd.DataFrame:
    """Load embedding cache if key matches; otherwise recompute and save."""
    cache_npz = Path(cache_npz)

    _, _, resolved_ckpt = load_backbone_from_step1(
        config=config,
        model_size=model_size,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    cache_key = _make_cache_key(
        polymer_df=polymer_df,
        checkpoint_path=resolved_ckpt,
        model_size=model_size,
        timestep=timestep,
        pooling=pooling,
    )

    if cache_npz.exists() and cache_npz.with_suffix(".json").exists():
        cached_df, cached_meta = load_embedding_cache(cache_npz)
        if cached_meta.get("cache_key") == cache_key:
            return cached_df.sort_values("polymer_id").reset_index(drop=True)

    tokenizer, backbone, _ = load_backbone_from_step1(
        config=config,
        model_size=model_size,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    emb_df = compute_polymer_embeddings(
        polymer_df=polymer_df,
        tokenizer=tokenizer,
        backbone=backbone,
        device=device,
        timestep=timestep,
        pooling=pooling,
        batch_size=batch_size,
    )

    metadata = {
        "cache_key": cache_key,
        "checkpoint": str(resolved_ckpt),
        "model_size": model_size or "base",
        "timestep": int(timestep),
        "pooling": pooling,
        "n_polymers": int(len(emb_df)),
    }
    save_embedding_cache(cache_npz, emb_df, metadata)
    return emb_df.sort_values("polymer_id").reset_index(drop=True)



def embedding_table_from_cache(embedding_df: pd.DataFrame) -> np.ndarray:
    """Convert cached embedding dataframe to dense table indexed by polymer_id."""
    emb_df = embedding_df.sort_values("polymer_id").reset_index(drop=True)
    table = np.stack(emb_df["embedding"].to_list(), axis=0)
    expected = np.arange(len(emb_df))
    got = emb_df["polymer_id"].to_numpy(dtype=int)
    if not np.array_equal(expected, got):
        raise ValueError("polymer_id must be contiguous from 0..N-1 for embedding table")
    return table.astype(np.float32)
