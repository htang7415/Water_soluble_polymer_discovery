from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class SmilesDataset(Dataset):
    def __init__(
        self,
        smiles: List[str],
        tokenizer,
        max_length: Optional[int] = None,
        cache_tokenization: bool = False,
    ):
        self.smiles = smiles
        self.tokenizer = tokenizer
        if max_length:
            self.tokenizer.max_length = max_length
        self.cache_tokenization = cache_tokenization
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

        if cache_tokenization:
            self._pretokenize()

    def _encode(self, smiles: str) -> Dict[str, torch.Tensor]:
        ids = self.tokenizer.encode(smiles, add_special=True, pad=True)
        attn = [1 if t != self.tokenizer.pad_id else 0 for t in ids]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    def _pretokenize(self) -> None:
        for idx, smiles in enumerate(self.smiles):
            self._cache[idx] = self._encode(smiles)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_tokenization and idx in self._cache:
            return self._cache[idx]
        return self._encode(self.smiles[idx])


class SmilesRegressionDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        smiles_col: str = "SMILES",
        target_col: str = "y",
        normalize: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        max_length: Optional[int] = None,
        cache_tokenization: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.smiles = self.df[smiles_col].astype(str).tolist()
        self.targets = self.df[target_col].astype(float).tolist()
        self.tokenizer = tokenizer
        if max_length:
            self.tokenizer.max_length = max_length
        self.normalize = normalize
        self.mean = float(mean) if mean is not None else float(sum(self.targets) / max(len(self.targets), 1))
        if std is not None:
            self.std = float(std)
        else:
            vals = torch.tensor(self.targets, dtype=torch.float)
            self.std = float(vals.std().item()) if len(vals) > 1 else 1.0
        if self.std == 0:
            self.std = 1.0
        self.cache_tokenization = cache_tokenization
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

        if cache_tokenization:
            self._pretokenize()

    def _encode(self, smiles: str) -> Dict[str, torch.Tensor]:
        ids = self.tokenizer.encode(smiles, add_special=True, pad=True)
        attn = [1 if t != self.tokenizer.pad_id else 0 for t in ids]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    def _pretokenize(self) -> None:
        for idx, smiles in enumerate(self.smiles):
            payload = self._encode(smiles)
            label = self.targets[idx]
            if self.normalize:
                label = (label - self.mean) / self.std
            payload["labels"] = torch.tensor(label, dtype=torch.float)
            self._cache[idx] = payload

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_tokenization and idx in self._cache:
            return self._cache[idx]
        payload = self._encode(self.smiles[idx])
        label = self.targets[idx]
        if self.normalize:
            label = (label - self.mean) / self.std
        payload["labels"] = torch.tensor(label, dtype=torch.float)
        return payload

    def get_normalization_params(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    def denormalize(self, value: float) -> float:
        return value * self.std + self.mean


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in batch[0].keys():
        out[key] = torch.stack([item[key] for item in batch])
    return out
