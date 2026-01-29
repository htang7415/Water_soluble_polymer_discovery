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


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in batch[0].keys():
        out[key] = torch.stack([item[key] for item in batch])
    return out
