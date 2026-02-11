"""Shared utilities for traditional Step 4_3 and Step 4_4 workflows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.chi.data import SplitConfig, add_split_column, load_chi_dataset, make_split_assignments
from src.utils.config import load_config
from src.utils.model_scales import get_results_dir


@dataclass(frozen=True)
class FingerprintConfig:
    radius: int = 3
    n_bits: int = 1024
    use_chirality: bool = False
    use_features: bool = False


def load_traditional_config(config_path: str) -> Dict:
    cfg = load_config(config_path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config file (not a mapping): {config_path}")
    return cfg


def get_traditional_results_dir(results_root: str, model_size: str, split_mode: str) -> Path:
    base = str(Path(results_root) / "results_traditional")
    return Path(get_results_dir(model_size=model_size, base_dir=base, split_mode=split_mode))


def normalize_split_mode(split_mode: str) -> str:
    out = str(split_mode).strip().lower()
    if out not in {"polymer", "random"}:
        raise ValueError("split_mode must be one of {'polymer', 'random'}")
    return out


def resolve_split_ratios(holdout_test_ratio: float, tuning_cv_folds: int) -> Dict[str, float]:
    test_ratio = float(holdout_test_ratio)
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("holdout_test_ratio must be in (0, 1)")
    folds = int(max(2, tuning_cv_folds))
    dev_ratio = 1.0 - test_ratio
    val_ratio = dev_ratio / float(folds)
    train_ratio = dev_ratio - val_ratio
    out = {
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
    }
    total = sum(out.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Resolved split ratios must sum to 1, got {out}")
    if min(out.values()) <= 0.0:
        raise ValueError(f"Resolved split ratios must all be > 0, got {out}")
    return out


def load_classification_dataset(
    csv_path: str | Path,
    default_temperature: float = 293.15,
    default_phi: float = 0.2,
    default_chi: float = 0.0,
) -> pd.DataFrame:
    """Load Step4_3_2 classification dataset with compatibility normalization."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"classification dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    rename_map = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"water_soluble", "water_solubel", "water_solubility"}:
            rename_map[col] = "water_soluble"
    if rename_map:
        df = df.rename(columns=rename_map)

    required_base = {"Polymer", "SMILES", "water_soluble"}
    missing_base = sorted(required_base - set(df.columns))
    if missing_base:
        raise ValueError(f"classification dataset missing required columns: {missing_base}")

    out = df.copy()
    if "temperature" not in out.columns:
        out["temperature"] = float(default_temperature)
    if "phi" not in out.columns:
        out["phi"] = float(default_phi)
    if "chi" not in out.columns:
        out["chi"] = float(default_chi)

    out["temperature"] = out["temperature"].fillna(float(default_temperature)).astype(float)
    out["phi"] = out["phi"].fillna(float(default_phi)).astype(float)
    out["chi"] = out["chi"].fillna(float(default_chi)).astype(float)
    out["water_soluble"] = out["water_soluble"].astype(int)

    polymer_order = sorted(out["Polymer"].astype(str).unique())
    polymer_to_id = {p: i for i, p in enumerate(polymer_order)}
    out["polymer_id"] = out["Polymer"].map(polymer_to_id).astype(int)
    out = out.reset_index(drop=True)
    out["row_id"] = out.index.astype(int)
    return out


def load_split_dataset(
    dataset_path: str | Path,
    split_mode: str,
    split_ratios: Dict[str, float],
    seed: int,
    is_classification_dataset: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if is_classification_dataset:
        df = load_classification_dataset(dataset_path)
    else:
        df = load_chi_dataset(dataset_path)
    split_assign = make_split_assignments(
        df,
        SplitConfig(
            split_mode=split_mode,
            train_ratio=float(split_ratios["train_ratio"]),
            val_ratio=float(split_ratios["val_ratio"]),
            test_ratio=float(split_ratios["test_ratio"]),
            seed=int(seed),
        ),
    )
    split_df = add_split_column(df, split_assign)
    return split_df, split_assign


def build_final_fit_split_df(split_df: pd.DataFrame) -> pd.DataFrame:
    out = split_df.copy()
    out.loc[out["split"] == "val", "split"] = "train"
    return out


def build_tuning_cv_folds(split_df: pd.DataFrame, split_mode: str, tuning_cv_folds: int, seed: int) -> Tuple[List[pd.DataFrame], Dict[str, object]]:
    """Build CV folds from non-test rows using Step4-style stratification."""
    dev_df = split_df[split_df["split"].isin(["train", "val"])].copy().reset_index(drop=True)
    if dev_df.empty:
        raise ValueError("No train/val rows available for CV tuning.")

    requested_folds = int(max(2, tuning_cv_folds))
    if split_mode == "polymer":
        unit_df = (
            dev_df[["polymer_id", "water_soluble"]]
            .drop_duplicates(subset=["polymer_id"])
            .sort_values("polymer_id")
            .reset_index(drop=True)
        )
        unit_key = "polymer_id"
        strategy = "polymer_group_stratified"
    else:
        unit_df = (
            dev_df[["row_id", "water_soluble"]]
            .drop_duplicates(subset=["row_id"])
            .sort_values("row_id")
            .reset_index(drop=True)
        )
        unit_key = "row_id"
        strategy = "row_stratified"

    class_counts = unit_df["water_soluble"].value_counts()
    max_folds = int(min(len(unit_df), class_counts.min())) if not class_counts.empty else 0
    if max_folds < 2:
        fallback = dev_df.copy()
        if fallback["split"].nunique() < 2:
            idx = np.arange(len(fallback))
            fallback["split"] = np.where((idx % 5) == 0, "val", "train")
        return [fallback.reset_index(drop=True)], {
            "strategy": f"{strategy}_fallback_original_split",
            "requested_folds": requested_folds,
            "resolved_folds": 1,
            "dev_rows": int(len(dev_df)),
            "dev_units": int(len(unit_df)),
        }

    resolved_folds = int(min(requested_folds, max_folds))
    skf = StratifiedKFold(n_splits=resolved_folds, shuffle=True, random_state=int(seed))
    unit_ids = unit_df[unit_key].to_numpy()
    labels = unit_df["water_soluble"].to_numpy(dtype=int)

    folds: List[pd.DataFrame] = []
    for _, val_idx in skf.split(unit_ids, labels):
        val_ids = set(unit_ids[val_idx].tolist())
        fold_df = dev_df.copy()
        fold_df["split"] = np.where(fold_df[unit_key].isin(val_ids), "val", "train")
        folds.append(fold_df.reset_index(drop=True))

    return folds, {
        "strategy": strategy,
        "requested_folds": requested_folds,
        "resolved_folds": resolved_folds,
        "dev_rows": int(len(dev_df)),
        "dev_units": int(len(unit_df)),
    }


def summarize_cv_folds(cv_folds: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for i, fold_df in enumerate(cv_folds, start=1):
        for split in ["train", "val"]:
            sub = fold_df[fold_df["split"] == split]
            n_rows = int(len(sub))
            n_pos = int(sub["water_soluble"].sum()) if n_rows > 0 else 0
            rows.append(
                {
                    "fold": i,
                    "split": split,
                    "n_rows": n_rows,
                    "n_polymers": int(sub["polymer_id"].nunique()) if n_rows > 0 else 0,
                    "n_positive": n_pos,
                    "n_negative": int(n_rows - n_pos),
                }
            )
    return pd.DataFrame(rows)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fingerprint_cache_key(polymer_df: pd.DataFrame, fp_cfg: FingerprintConfig) -> str:
    records = polymer_df[["polymer_id", "Polymer", "SMILES"]].sort_values("polymer_id")
    payload = {
        "fingerprint": {
            "radius": int(fp_cfg.radius),
            "n_bits": int(fp_cfg.n_bits),
            "use_chirality": bool(fp_cfg.use_chirality),
            "use_features": bool(fp_cfg.use_features),
        },
        "records": records.to_dict(orient="records"),
    }
    return _sha256(json.dumps(payload, sort_keys=True))


def smiles_to_morgan_fp(smiles: str, fp_cfg: FingerprintConfig) -> Tuple[np.ndarray, bool]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    arr = np.zeros((int(fp_cfg.n_bits),), dtype=np.float32)
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return arr, False
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        int(fp_cfg.radius),
        nBits=int(fp_cfg.n_bits),
        useChirality=bool(fp_cfg.use_chirality),
        useFeatures=bool(fp_cfg.use_features),
    )
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32, copy=False), True


def compute_polymer_fingerprint_table(df: pd.DataFrame, fp_cfg: FingerprintConfig) -> Tuple[np.ndarray, Dict[str, object]]:
    unique_df = (
        df[["polymer_id", "Polymer", "SMILES"]]
        .drop_duplicates(subset=["polymer_id"])
        .sort_values("polymer_id")
        .reset_index(drop=True)
    )
    if unique_df.empty:
        raise ValueError("No polymers found for fingerprint table.")

    n_polymers = int(unique_df["polymer_id"].max()) + 1
    table = np.zeros((n_polymers, int(fp_cfg.n_bits)), dtype=np.float32)
    invalid_rows = []
    for row in unique_df.itertuples(index=False):
        fp_vec, valid = smiles_to_morgan_fp(row.SMILES, fp_cfg=fp_cfg)
        table[int(row.polymer_id)] = fp_vec
        if not valid:
            invalid_rows.append({"polymer_id": int(row.polymer_id), "Polymer": str(row.Polymer), "SMILES": str(row.SMILES)})

    meta = {
        "n_polymers": int(len(unique_df)),
        "n_bits": int(fp_cfg.n_bits),
        "radius": int(fp_cfg.radius),
        "use_chirality": bool(fp_cfg.use_chirality),
        "use_features": bool(fp_cfg.use_features),
        "n_invalid_smiles": int(len(invalid_rows)),
        "invalid_smiles_rows": invalid_rows,
    }
    return table, meta


def build_or_load_fingerprint_cache(
    df: pd.DataFrame,
    fp_cfg: FingerprintConfig,
    cache_npz: str | Path,
) -> Tuple[np.ndarray, Dict[str, object]]:
    cache_npz = Path(cache_npz)
    cache_json = cache_npz.with_suffix(".json")
    poly_df = (
        df[["polymer_id", "Polymer", "SMILES"]]
        .drop_duplicates(subset=["polymer_id"])
        .sort_values("polymer_id")
        .reset_index(drop=True)
    )
    cache_key = _fingerprint_cache_key(poly_df, fp_cfg)

    if cache_npz.exists() and cache_json.exists():
        try:
            with open(cache_json, "r") as f:
                meta = json.load(f)
            if meta.get("cache_key") == cache_key:
                with np.load(cache_npz, allow_pickle=False) as arr:
                    table = np.asarray(arr["fingerprint_table"], dtype=np.float32)
                return table, meta
        except Exception:
            pass

    table, meta = compute_polymer_fingerprint_table(df, fp_cfg)
    meta = dict(meta)
    meta["cache_key"] = cache_key
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_npz, fingerprint_table=table.astype(np.float32, copy=False))
    with open(cache_json, "w") as f:
        json.dump(meta, f, indent=2)
    return table, meta


def features_from_table(split_df: pd.DataFrame, fingerprint_table: np.ndarray) -> np.ndarray:
    polymer_ids = split_df["polymer_id"].to_numpy(dtype=np.int64)
    if len(polymer_ids) == 0:
        return np.zeros((0, int(fingerprint_table.shape[1])), dtype=np.float32)
    return fingerprint_table[polymer_ids].astype(np.float32, copy=False)
