import csv
import gzip
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, payload: Dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def stream_gzip_csv(path: str) -> Iterator[Dict[str, str]]:
    with gzip.open(path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def get_d5_path(config_path: str, data_dir: str) -> str:
    # Support both COSMOSAC and COSMOC filenames
    primary = Path(data_dir) / "OMG_DFT_COSMOSAC_chi.csv"
    fallback = Path(data_dir) / "OMG_DFT_COSMOC_chi.csv"
    return str(primary if primary.exists() else fallback)


def split_train_val_test_by_group(
    groups: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[set, set, set]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    unique = list(sorted(set(groups)))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_groups = set(unique[:n_train])
    val_groups = set(unique[n_train : n_train + n_val])
    test_groups = set(unique[n_train + n_val :])
    return train_groups, val_groups, test_groups


def leave_one_group_out(groups: Sequence[str]) -> List[Tuple[str, set]]:
    unique = list(sorted(set(groups)))
    folds = []
    for g in unique:
        folds.append((g, set([g])))
    return folds


def quantiles(values: Sequence[float], qs: Sequence[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    return {f"q{int(q*100)}": float(np.quantile(arr, q)) for q in qs}


@dataclass
class SmilesRecord:
    smiles: str
    y: Optional[float] = None
    group: Optional[str] = None


def read_csv_column(path: str, column: str) -> List[str]:
    out = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(row[column])
    return out


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def iter_csv_rows(path: str) -> Iterator[Dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def write_csv(path: str, fieldnames: List[str], rows: List[Dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def batch_iter(iterable: Iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
