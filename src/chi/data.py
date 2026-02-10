"""Data utilities for chi(T, phi) modeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_CHI_COLUMNS = [
    "Polymer",
    "SMILES",
    "temperature",
    "phi",
    "chi",
    "water_soluble",
]

COEFF_NAMES = ["a0", "a1", "a2", "a3", "b1", "b2"]


@dataclass(frozen=True)
class SplitConfig:
    """Split configuration for chi dataset."""

    split_mode: str = "polymer"
    train_ratio: float = 0.70
    val_ratio: float = 0.14
    test_ratio: float = 0.16
    seed: int = 42



def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio}+{val_ratio}+{test_ratio}={total}"
        )



def _safe_train_test_split(*args, stratify, random_state: int, test_size: float):
    """Fallback to unstratified split if stratification is impossible."""
    try:
        return train_test_split(
            *args,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            *args,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )



def _standardize_water_soluble_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known typo variants of water_soluble column."""
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in {"water_solubel", "water_solubility", "water_soluble"}:
            rename_map[col] = "water_soluble"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df



def load_chi_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load chi dataset and add stable polymer ids.

    Returns a dataframe with normalized dtypes and a deterministic polymer id mapping.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"chi dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _standardize_water_soluble_column(df)

    missing = [c for c in REQUIRED_CHI_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required chi columns: {missing}")

    out = df.copy()
    out["temperature"] = out["temperature"].astype(float)
    out["phi"] = out["phi"].astype(float)
    out["chi"] = out["chi"].astype(float)
    out["water_soluble"] = out["water_soluble"].astype(int)

    polymer_order = sorted(out["Polymer"].astype(str).unique())
    polymer_to_id = {p: i for i, p in enumerate(polymer_order)}
    out["polymer_id"] = out["Polymer"].map(polymer_to_id).astype(int)

    out = out.reset_index(drop=True)
    out["row_id"] = out.index.astype(int)
    return out



def make_split_assignments(df: pd.DataFrame, split_cfg: SplitConfig) -> pd.DataFrame:
    """Create split assignment dataframe with columns: row_id, split."""
    _validate_ratios(split_cfg.train_ratio, split_cfg.val_ratio, split_cfg.test_ratio)

    split_mode = split_cfg.split_mode.strip().lower()
    if split_mode not in {"polymer", "random"}:
        raise ValueError("split_mode must be one of {'polymer', 'random'}")

    assignments = pd.DataFrame({"row_id": df["row_id"].astype(int)})
    assignments["split"] = ""

    if split_mode == "polymer":
        polymer_df = (
            df[["polymer_id", "Polymer", "water_soluble"]]
            .drop_duplicates(subset=["polymer_id"])
            .sort_values("polymer_id")
            .reset_index(drop=True)
        )

        train_poly, temp_poly = _safe_train_test_split(
            polymer_df,
            stratify=polymer_df["water_soluble"],
            random_state=split_cfg.seed,
            test_size=(1.0 - split_cfg.train_ratio),
        )

        temp_test_ratio = split_cfg.test_ratio / (split_cfg.val_ratio + split_cfg.test_ratio)
        val_poly, test_poly = _safe_train_test_split(
            temp_poly,
            stratify=temp_poly["water_soluble"],
            random_state=split_cfg.seed,
            test_size=temp_test_ratio,
        )

        split_map: Dict[int, str] = {}
        split_map.update({int(pid): "train" for pid in train_poly["polymer_id"].tolist()})
        split_map.update({int(pid): "val" for pid in val_poly["polymer_id"].tolist()})
        split_map.update({int(pid): "test" for pid in test_poly["polymer_id"].tolist()})

        assignments["split"] = df["polymer_id"].map(split_map)

    else:
        # Row-level random split.
        row_df = df[["row_id", "water_soluble"]].copy()
        train_rows, temp_rows = _safe_train_test_split(
            row_df,
            stratify=row_df["water_soluble"],
            random_state=split_cfg.seed,
            test_size=(1.0 - split_cfg.train_ratio),
        )
        temp_test_ratio = split_cfg.test_ratio / (split_cfg.val_ratio + split_cfg.test_ratio)
        val_rows, test_rows = _safe_train_test_split(
            temp_rows,
            stratify=temp_rows["water_soluble"],
            random_state=split_cfg.seed,
            test_size=temp_test_ratio,
        )

        split_map: Dict[int, str] = {}
        split_map.update({int(rid): "train" for rid in train_rows["row_id"].tolist()})
        split_map.update({int(rid): "val" for rid in val_rows["row_id"].tolist()})
        split_map.update({int(rid): "test" for rid in test_rows["row_id"].tolist()})
        assignments["split"] = assignments["row_id"].map(split_map)

    if assignments["split"].isna().any() or (assignments["split"] == "").any():
        raise RuntimeError("Failed to assign all rows to train/val/test")

    return assignments



def add_split_column(df: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Attach split labels to a dataframe by row_id."""
    merged = df.merge(assignments[["row_id", "split"]], on="row_id", how="left")
    if merged["split"].isna().any():
        raise RuntimeError("Some rows are missing split labels")
    return merged



def physics_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add explicit physics-inspired scalar features for baseline models."""
    out = df.copy()
    out["inv_T"] = 1.0 / out["temperature"]
    out["log_T"] = np.log(out["temperature"])
    out["one_minus_phi"] = 1.0 - out["phi"]
    out["one_minus_phi_sq"] = out["one_minus_phi"] ** 2
    return out
