"""Shared utilities for traditional Step 4_3 and Step 4_4 workflows."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.chi.data import fill_missing_polymer_names_from_smiles
from src.utils.config import load_config
from src.utils.model_scales import get_results_dir

CLASS_LABEL_INTERNAL = "water_miscible"
CLASS_LABEL_PUBLIC = "water_miscible"


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


def get_traditional_results_dir(results_root: str, split_mode: str | None = None) -> Path:
    """Resolve traditional Step4 base results directory.

    New layout defaults to a split-independent root:
      <results_root>/results_traditional

    When split_mode is provided, keep backward-compatible split-suffixed
    resolution:
      <results_root>/results_traditional_<split_mode>
    """
    base = str(Path(results_root) / "results_traditional")
    resolved_split = normalize_split_mode(split_mode) if split_mode is not None else None
    return Path(get_results_dir(model_size=None, base_dir=base, split_mode=resolved_split))


def normalize_split_mode(split_mode: str) -> str:
    out = str(split_mode).strip().lower()
    if out not in {"polymer", "random"}:
        raise ValueError("split_mode must be one of {'polymer', 'random'}")
    return out


def _safe_train_test_split(
    *args,
    stratify,
    random_state: int,
    test_size: float,
    allow_unstratified_fallback: bool = True,
):
    """Fallback to unstratified split if stratification is impossible."""
    try:
        return train_test_split(
            *args,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError as exc:
        msg = (
            "Stratified split failed"
            + (
                "; unstratified fallback disabled."
                if not bool(allow_unstratified_fallback)
                else "; falling back to unstratified split."
            )
            + f" reason={exc}"
        )
        if not bool(allow_unstratified_fallback):
            raise ValueError(msg) from exc
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        print(f"[warning] {msg}")
        return train_test_split(
            *args,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )


def resolve_split_ratios(holdout_test_ratio: float, tuning_cv_folds: int) -> Dict[str, float]:
    """Resolve holdout split ratios for Step4_3.

    Step4_3 follows the workflow:
    1) one holdout split into train/test,
    2) CV only inside train for hyperparameter tuning,
    3) refit on all train and evaluate on test.

    Therefore, we do not reserve a separate fixed validation split here.
    """
    test_ratio = float(holdout_test_ratio)
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("holdout_test_ratio must be in (0, 1)")
    _ = int(max(2, tuning_cv_folds))  # kept for API compatibility / validation context
    train_ratio = 1.0 - test_ratio
    val_ratio = 0.0
    out = {
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
    }
    total = sum(out.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Resolved split ratios must sum to 1, got {out}")
    if out["train_ratio"] <= 0.0 or out["test_ratio"] <= 0.0:
        raise ValueError(f"Resolved train/test ratios must be > 0, got {out}")
    return out


def _resolve_classification_dataset_paths(csv_path: str | Path | List[str] | Tuple[str, ...]) -> List[Path]:
    specs: List[str] = []
    if isinstance(csv_path, (list, tuple)):
        specs = [str(x).strip() for x in csv_path if str(x).strip()]
    else:
        raw = str(csv_path).strip()
        if len(raw) == 0:
            raise ValueError("classification dataset path is empty")
        if "," in raw:
            specs = [x.strip() for x in raw.split(",") if x.strip()]
        else:
            specs = [raw]

    paths: List[Path] = []
    for spec in specs:
        p = Path(spec)
        if p.is_dir():
            csvs = sorted(q for q in p.glob("*.csv") if q.is_file())
            if len(csvs) == 0:
                raise FileNotFoundError(f"No CSV files found under classification dataset directory: {p}")
            paths.extend(csvs)
        else:
            if not p.exists():
                raise FileNotFoundError(f"classification dataset not found: {p}")
            paths.append(p)
    if len(paths) == 0:
        raise ValueError("No classification dataset CSV paths resolved.")
    return paths


def load_classification_dataset(
    csv_path: str | Path | List[str] | Tuple[str, ...],
    default_temperature: float = 293.15,
    default_phi: float = 0.2,
    default_chi: float = 0.0,
) -> pd.DataFrame:
    """Load Step4_3_2 classification dataset with compatibility normalization."""
    paths = _resolve_classification_dataset_paths(csv_path)
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    label_aliases = {
        "water_soluble",
        "water_solubel",
        "water_solubility",
        "water_miscible",
        "water miscible",
        "watermiscible",
        "water_missible",
    }
    matched = []
    for col in df.columns:
        key = str(col).strip().lower()
        if key in label_aliases:
            matched.append(col)
    if CLASS_LABEL_INTERNAL not in df.columns and len(matched) > 0:
        primary = matched[0]
        if primary != CLASS_LABEL_INTERNAL:
            df = df.rename(columns={primary: CLASS_LABEL_INTERNAL})
        matched = [CLASS_LABEL_INTERNAL] + [c for c in matched if c != primary]
    for col in matched:
        if col == CLASS_LABEL_INTERNAL or col not in df.columns:
            continue
        df[CLASS_LABEL_INTERNAL] = df[CLASS_LABEL_INTERNAL].where(
            df[CLASS_LABEL_INTERNAL].notna(),
            df[col],
        )

    required_base = {"Polymer", "SMILES", CLASS_LABEL_INTERNAL}
    missing_base = sorted(required_base - set(df.columns))
    if missing_base:
        raise ValueError(f"classification dataset missing required columns: {missing_base}")

    out = df.copy()
    out = fill_missing_polymer_names_from_smiles(out, source_name="traditional classification dataset")
    if "temperature" not in out.columns:
        out["temperature"] = float(default_temperature)
    if "phi" not in out.columns:
        out["phi"] = float(default_phi)
    if "chi" not in out.columns:
        out["chi"] = float(default_chi)

    out["temperature"] = out["temperature"].fillna(float(default_temperature)).astype(float)
    out["phi"] = out["phi"].fillna(float(default_phi)).astype(float)
    out["chi"] = out["chi"].fillna(float(default_chi)).astype(float)
    out[CLASS_LABEL_INTERNAL] = pd.to_numeric(out[CLASS_LABEL_INTERNAL], errors="coerce").fillna(0).astype(int)
    out[CLASS_LABEL_PUBLIC] = out[CLASS_LABEL_INTERNAL]

    polymer_order = sorted(out["Polymer"].astype(str).unique())
    polymer_to_id = {p: i for i, p in enumerate(polymer_order)}
    out["polymer_id"] = out["Polymer"].map(polymer_to_id).astype(int)
    out = out.reset_index(drop=True)
    out["row_id"] = out.index.astype(int)
    return out


def load_split_dataset(
    dataset_path: str | Path | List[str] | Tuple[str, ...],
    split_mode: str,
    split_ratios: Dict[str, float],
    seed: int,
    is_classification_dataset: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from src.chi.data import SplitConfig, add_split_column, load_chi_dataset, make_split_assignments

    if is_classification_dataset:
        df = load_classification_dataset(dataset_path)
    else:
        df = load_chi_dataset(dataset_path)

    train_ratio = float(split_ratios["train_ratio"])
    val_ratio = float(split_ratios["val_ratio"])
    test_ratio = float(split_ratios["test_ratio"])

    if val_ratio <= 0.0:
        # Preferred Step4_3 mode: holdout split only (train/test), CV happens inside train.
        assignments = pd.DataFrame({"row_id": df["row_id"].astype(int)})
        assignments["split"] = ""
        mode = normalize_split_mode(split_mode)

        if mode == "polymer":
            polymer_df = (
                df[["polymer_id", "water_miscible"]]
                .drop_duplicates(subset=["polymer_id"])
                .sort_values("polymer_id")
                .reset_index(drop=True)
            )
            train_poly, test_poly = _safe_train_test_split(
                polymer_df,
                stratify=polymer_df["water_miscible"],
                random_state=int(seed),
                test_size=float(test_ratio),
                allow_unstratified_fallback=not bool(is_classification_dataset),
            )
            split_map: Dict[int, str] = {}
            split_map.update({int(pid): "train" for pid in train_poly["polymer_id"].tolist()})
            split_map.update({int(pid): "test" for pid in test_poly["polymer_id"].tolist()})
            assignments["split"] = df["polymer_id"].map(split_map)
        else:
            row_df = df[["row_id", "water_miscible"]].copy()
            train_rows, test_rows = _safe_train_test_split(
                row_df,
                stratify=row_df["water_miscible"],
                random_state=int(seed),
                test_size=float(test_ratio),
                allow_unstratified_fallback=not bool(is_classification_dataset),
            )
            split_map = {}
            split_map.update({int(rid): "train" for rid in train_rows["row_id"].tolist()})
            split_map.update({int(rid): "test" for rid in test_rows["row_id"].tolist()})
            assignments["split"] = assignments["row_id"].map(split_map)

        if assignments["split"].isna().any() or (assignments["split"] == "").any():
            raise RuntimeError("Failed to assign all rows to train/test")
        split_assign = assignments
    else:
        # Backward-compatible path when caller explicitly requests a fixed val split.
        split_assign = make_split_assignments(
            df,
            SplitConfig(
                split_mode=split_mode,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
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
    from sklearn.model_selection import StratifiedKFold

    dev_df = split_df[split_df["split"].isin(["train", "val"])].copy().reset_index(drop=True)
    if dev_df.empty:
        raise ValueError("No train/val rows available for CV tuning.")

    requested_folds = int(max(2, tuning_cv_folds))
    if split_mode == "polymer":
        unit_df = (
            dev_df[["polymer_id", "water_miscible"]]
            .drop_duplicates(subset=["polymer_id"])
            .sort_values("polymer_id")
            .reset_index(drop=True)
        )
        unit_key = "polymer_id"
        strategy = "polymer_group_stratified"
    else:
        unit_df = (
            dev_df[["row_id", "water_miscible"]]
            .drop_duplicates(subset=["row_id"])
            .sort_values("row_id")
            .reset_index(drop=True)
        )
        unit_key = "row_id"
        strategy = "row_stratified"

    class_counts = unit_df["water_miscible"].value_counts()
    max_folds = int(min(len(unit_df), class_counts.min())) if not class_counts.empty else 0
    if max_folds < 2:
        warnings.warn(
            (
                "Insufficient class support for StratifiedKFold "
                f"(max_folds={max_folds}, dev_units={len(unit_df)}); "
                "falling back to a single train/val split."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        fallback = dev_df.copy()
        if len(unit_df) < 2:
            raise ValueError("Cannot build CV fallback split: fewer than 2 dev units.")

        unit_ids = unit_df[unit_key].to_numpy()
        labels = unit_df["water_miscible"].to_numpy(dtype=int)
        n_classes = int(pd.Series(labels).nunique()) if len(labels) > 0 else 0
        val_size = int(max(1, np.ceil(0.2 * len(unit_ids))))
        if n_classes > 0:
            val_size = max(val_size, n_classes)
        val_size = min(val_size, len(unit_ids) - 1)

        val_ids = None
        fallback_suffix = "single_split_unit_balanced"
        can_stratify_single_split = n_classes >= 2 and int(class_counts.min()) >= 2 and val_size >= n_classes
        if can_stratify_single_split:
            try:
                _, val_units = train_test_split(
                    unit_ids,
                    test_size=int(val_size),
                    random_state=int(seed),
                    stratify=labels,
                )
                val_ids = set(np.asarray(val_units).tolist())
                fallback_suffix = "single_split_stratified"
            except ValueError as exc:
                warnings.warn(
                    "Single-split stratified fallback failed; using class-aware unit-balanced fallback. "
                    f"reason={exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if val_ids is None:
            rng = np.random.default_rng(int(seed))
            chosen: List[object] = []
            for _, sub in unit_df.groupby("water_miscible", sort=True):
                cls_ids = sub[unit_key].to_numpy()
                if len(cls_ids) <= 1:
                    continue
                cls_take = int(max(1, np.floor(0.2 * len(cls_ids))))
                cls_take = min(cls_take, len(cls_ids) - 1)
                if cls_take <= 0:
                    continue
                perm = rng.permutation(len(cls_ids))
                chosen.extend(np.asarray(cls_ids)[perm[:cls_take]].tolist())
            if len(chosen) == 0:
                chosen = [unit_ids[int(rng.integers(0, len(unit_ids)))]]
            val_ids = set(chosen)
            if len(val_ids) >= len(unit_ids):
                val_ids = set(sorted(val_ids)[: len(unit_ids) - 1])

        fallback["split"] = np.where(fallback[unit_key].isin(val_ids), "val", "train")
        if fallback["split"].nunique() < 2:
            raise ValueError("Fallback CV split failed to produce both train and val splits.")

        train_cls = int(fallback.loc[fallback["split"] == "train", "water_miscible"].nunique())
        val_cls = int(fallback.loc[fallback["split"] == "val", "water_miscible"].nunique())
        if train_cls < 2 or val_cls < 2:
            warnings.warn(
                "Fallback CV split has single-class train/val; classification tuning metrics may be invalid.",
                RuntimeWarning,
                stacklevel=2,
            )
        return [fallback.reset_index(drop=True)], {
            "strategy": f"{strategy}_fallback_{fallback_suffix}",
            "requested_folds": requested_folds,
            "resolved_folds": 1,
            "dev_rows": int(len(dev_df)),
            "dev_units": int(len(unit_df)),
            "fallback_train_n_classes": train_cls,
            "fallback_val_n_classes": val_cls,
        }

    resolved_folds = int(min(requested_folds, max_folds))
    skf = StratifiedKFold(n_splits=resolved_folds, shuffle=True, random_state=int(seed))
    unit_ids = unit_df[unit_key].to_numpy()
    labels = unit_df["water_miscible"].to_numpy(dtype=int)

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
            n_pos = int(sub["water_miscible"].sum()) if n_rows > 0 else 0
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
