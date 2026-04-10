"""Supervised dataset helpers for Step 6_2 S2-S4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.chi.data import SplitConfig, add_split_column, make_split_assignments
from src.data.tokenizer import PSmilesTokenizer

from .config import ResolvedStep62Config, _resolve_split_ratios


DEFAULT_WATER_TEMPERATURE = 293.15
DEFAULT_WATER_PHI = 0.2
DEFAULT_WATER_CHI = 0.0
BASE_CONDITION_BUNDLE_DIM = 7


def get_step62_condition_dim() -> int:
    return int(BASE_CONDITION_BUNDLE_DIM)


def _build_step62_condition_bundle_from_values(
    *,
    soluble: int,
    temperature: float,
    phi: float,
    chi_goal: float,
    chi_goal_lower: float,
    chi_goal_upper: float,
    scaler: ConditionScaler,
) -> np.ndarray:
    t_present = float(np.isfinite(temperature))
    phi_present = float(np.isfinite(phi))
    chi_present = float(np.isfinite(chi_goal))
    del chi_goal_lower, chi_goal_upper
    base_bundle = np.asarray(
        [
            float(int(soluble)),
            scaler.scale_temperature(float(temperature)) if t_present else 0.0,
            t_present,
            scaler.scale_phi(float(phi)) if phi_present else 0.0,
            phi_present,
            scaler.scale_chi_goal(float(chi_goal)) if chi_present else 0.0,
            chi_present,
        ],
        dtype=np.float32,
    )
    return base_bundle


def _normalize_water_miscible_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lstrip("\ufeff") for c in out.columns]
    aliases = {
        "water_miscible",
        "water miscible",
        "watermiscible",
        "water_missible",
        "water_soluble",
        "water solubel",
    }
    matched = [col for col in out.columns if str(col).strip().lower() in aliases]
    if not matched:
        return out
    primary = matched[0]
    if primary != "water_miscible":
        out = out.rename(columns={primary: "water_miscible"})
    for col in matched[1:]:
        if col in out.columns:
            out["water_miscible"] = out["water_miscible"].where(out["water_miscible"].notna(), out[col])
    out["water_miscible"] = pd.to_numeric(out["water_miscible"], errors="coerce").fillna(0).astype(int)
    return out


def _resolve_d_water_paths(base_config: Dict[str, Any]) -> List[Path]:
    chi_cfg = base_config.get("chi_training", {})
    step42_cfg = chi_cfg.get("step4_2_classification", {}) if isinstance(chi_cfg.get("step4_2_classification", {}), dict) else {}
    dataset_spec = step42_cfg.get(
        "dataset_path",
        chi_cfg.get(
            "step4_2_dataset_path",
            chi_cfg.get(
                "classification_dataset_path",
                [
                    "Data/water_solvent/water_miscible_polymer.csv",
                    "Data/water_solvent/water_immiscible_polymer.csv",
                ],
            ),
        ),
    )
    if isinstance(dataset_spec, (list, tuple)):
        paths = [Path(p) for p in dataset_spec]
    else:
        paths = [Path(dataset_spec)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Step 6_2 D_water CSVs not found: {missing}")
    return paths


def load_step62_water_dataset(
    base_config: Dict[str, Any],
) -> pd.DataFrame:
    """Load the Step 6_2 D_water dataset with explicit missing continuous fields."""

    paths = _resolve_d_water_paths(base_config)
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    df = _normalize_water_miscible_column(df)
    required = {"Polymer", "SMILES", "water_miscible"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Step 6_2 D_water dataset missing required columns: {sorted(missing)}")

    out = df.copy()
    polymer_order = sorted(out["Polymer"].astype(str).unique())
    polymer_to_id = {polymer: idx for idx, polymer in enumerate(polymer_order)}
    out["polymer_id"] = out["Polymer"].map(polymer_to_id).astype(int)
    out = out.reset_index(drop=True)
    out["row_id"] = out.index.astype(int)
    out["temperature"] = np.nan
    out["phi"] = np.nan
    out["chi"] = np.nan
    out["condition_source"] = "d_water"
    return out


def add_d_water_split_column(
    water_df: pd.DataFrame,
    *,
    base_config: Dict[str, Any],
    split_mode: str,
    random_seed: int,
) -> pd.DataFrame:
    """Build a split-consistent D_water split column."""

    split_ratios = _resolve_split_ratios(base_config)
    assignments = make_split_assignments(
        water_df,
        SplitConfig(
            split_mode=split_mode,
            train_ratio=float(split_ratios["train_ratio"]),
            val_ratio=float(split_ratios["val_ratio"]),
            test_ratio=float(split_ratios["test_ratio"]),
            seed=int(random_seed),
        ),
    )
    out = add_split_column(water_df, assignments)
    out["condition_source"] = "d_water"
    return out


@dataclass(frozen=True)
class ConditionScaler:
    """Min-max scaler for Step 6_2 continuous conditions."""

    temperature_min: float
    temperature_max: float
    phi_min: float
    phi_max: float
    chi_goal_min: float
    chi_goal_max: float

    def _scale(self, value: float, lower: float, upper: float) -> float:
        if not np.isfinite(value):
            return 0.0
        if np.isclose(upper, lower):
            return 0.0
        return float((float(value) - float(lower)) / (float(upper) - float(lower)))

    def scale_temperature(self, value: float) -> float:
        return self._scale(value, self.temperature_min, self.temperature_max)

    def scale_phi(self, value: float) -> float:
        return self._scale(value, self.phi_min, self.phi_max)

    def scale_chi_goal(self, value: float) -> float:
        return self._scale(value, self.chi_goal_min, self.chi_goal_max)


def build_condition_scaler(resolved: ResolvedStep62Config) -> ConditionScaler:
    stats = resolved.chi_train_stats
    return ConditionScaler(
        temperature_min=float(stats["temperature_min"]),
        temperature_max=float(stats["temperature_max"]),
        phi_min=float(stats["phi_min"]),
        phi_max=float(stats["phi_max"]),
        chi_goal_min=float(stats["chi_goal_min"]),
        chi_goal_max=float(stats["chi_goal_max"]),
    )


def build_step62_supervised_frames(resolved: ResolvedStep62Config) -> Dict[str, pd.DataFrame]:
    """Build split-aware supervised frames for D_chi and D_water."""

    chi_df = resolved.chi_split_df.copy()
    chi_df["condition_source"] = "d_chi"
    water_df = add_d_water_split_column(
        load_step62_water_dataset(resolved.base_config),
        base_config=resolved.base_config,
        split_mode=resolved.classification_split_mode,
        random_seed=int(resolved.step6_2["random_seed"]),
    )
    unified_cols = [
        "row_id",
        "polymer_id",
        "Polymer",
        "SMILES",
        "water_miscible",
        "temperature",
        "phi",
        "chi",
        "split",
        "condition_source",
    ]
    return {
        "d_chi": chi_df[unified_cols].copy(),
        "d_water": water_df[unified_cols].copy(),
        "train_d_chi": chi_df.loc[chi_df["split"] == "train", unified_cols].copy(),
        "val_d_chi": chi_df.loc[chi_df["split"] == "val", unified_cols].copy(),
        "train_d_water": water_df.loc[water_df["split"] == "train", unified_cols].copy(),
        "val_d_water": water_df.loc[water_df["split"] == "val", unified_cols].copy(),
    }


def build_source_batch_counts(batch_size: int, source_mix: Dict[str, float]) -> Dict[str, int]:
    """Resolve normalized within-minibatch source counts."""

    keys = list(source_mix.keys())
    weights = np.asarray([float(source_mix[key]) for key in keys], dtype=float)
    if (weights < 0).any():
        raise ValueError(f"train_batch_mix must be nonnegative, got {source_mix}")
    if np.isclose(weights.sum(), 0.0):
        raise ValueError(f"train_batch_mix sums to zero: {source_mix}")
    weights = weights / weights.sum()
    raw = weights * int(batch_size)
    counts = np.floor(raw).astype(int)
    remainder = int(batch_size) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    return {key: int(count) for key, count in zip(keys, counts)}


def build_condition_bundle(
    row: Dict[str, Any],
    *,
    scaler: ConditionScaler,
    chi_lookup,
    augment_with_step3_target: bool,
) -> Dict[str, Any]:
    """Build the raw Step 6_2 condition vector and metadata."""

    soluble = float(int(row["water_miscible"]))
    temperature = row.get("temperature", np.nan)
    phi = row.get("phi", np.nan)
    chi_observed = row.get("chi", np.nan)

    chi_goal = float(chi_observed) if np.isfinite(chi_observed) else np.nan
    chi_goal_lower = float(chi_observed) if np.isfinite(chi_observed) else np.nan
    chi_goal_upper = float(chi_observed) if np.isfinite(chi_observed) else np.nan
    step3_target = None
    augmentation_eligible = False
    augmented = False
    if np.isfinite(temperature) and np.isfinite(phi):
        step3_target_row = chi_lookup.lookup_row(float(temperature), float(phi), warn_on_missing=False)
    else:
        step3_target_row = None
    if step3_target_row is not None:
        step3_target = float(step3_target_row["chi_target"])
    if step3_target is not None and np.isfinite(chi_observed):
        augmentation_eligible = bool(
            int(row["water_miscible"]) == 1 and float(chi_observed) < float(step3_target)
        )
    if augment_with_step3_target and augmentation_eligible and step3_target is not None:
        chi_goal = float(step3_target)
        augmented = True

    bundle = _build_step62_condition_bundle_from_values(
        soluble=int(soluble),
        temperature=float(temperature) if np.isfinite(temperature) else np.nan,
        phi=float(phi) if np.isfinite(phi) else np.nan,
        chi_goal=float(chi_goal) if np.isfinite(chi_goal) else np.nan,
        chi_goal_lower=float(chi_goal_lower) if np.isfinite(chi_goal_lower) else np.nan,
        chi_goal_upper=float(chi_goal_upper) if np.isfinite(chi_goal_upper) else np.nan,
        scaler=scaler,
    )
    return {
        "condition_bundle": bundle,
        "chi_goal": float(chi_goal) if np.isfinite(chi_goal) else np.nan,
        "chi_goal_lower": float(chi_goal_lower) if np.isfinite(chi_goal_lower) else np.nan,
        "chi_goal_upper": float(chi_goal_upper) if np.isfinite(chi_goal_upper) else np.nan,
        "step3_chi_target": np.nan if step3_target is None else float(step3_target),
        "step3_chi_target_lower": np.nan,
        "step3_chi_target_upper": np.nan,
        "augmentation_eligible": int(augmentation_eligible),
        "augmented_to_step3_target": int(augmented),
    }


def build_inference_condition_bundle(
    *,
    temperature: float,
    phi: float,
    chi_goal: float,
    scaler: ConditionScaler,
    soluble: int = 1,
) -> np.ndarray:
    """Build the canonical Step 6_2 inference condition bundle."""
    return _build_step62_condition_bundle_from_values(
        soluble=int(soluble),
        temperature=float(temperature),
        phi=float(phi),
        chi_goal=float(chi_goal),
        chi_goal_lower=np.nan,
        chi_goal_upper=np.nan,
        scaler=scaler,
    )


def build_inference_condition_bundle_from_target_row(
    target_row: Mapping[str, Any],
    *,
    scaler: ConditionScaler,
    soluble: int = 1,
) -> np.ndarray:
    """Build the canonical Step 6_2 inference condition bundle from a target-row mapping."""

    return build_inference_condition_bundle(
        temperature=float(target_row["temperature"]),
        phi=float(target_row["phi"]),
        chi_goal=float(target_row["chi_target"]),
        scaler=scaler,
        soluble=int(soluble),
    )


def summarize_chi_augmentation_eligibility(
    df: pd.DataFrame,
    *,
    scaler: ConditionScaler,
    chi_lookup,
) -> pd.DataFrame:
    """Summarize exact-bucket Step 3 augmentation eligibility on D_chi rows."""

    if df.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "temperature",
                "phi",
                "row_count",
                "exact_step3_match_count",
                "eligible_count",
                "eligible_fraction",
                "step3_chi_target",
            ]
        )

    annotated_rows: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        info = build_condition_bundle(
            row,
            scaler=scaler,
            chi_lookup=chi_lookup,
            augment_with_step3_target=False,
        )
        annotated_rows.append(
            {
                "split": str(row.get("split", "")),
                "temperature": float(row["temperature"]),
                "phi": float(row["phi"]),
                "exact_step3_match": int(np.isfinite(info["step3_chi_target"])),
                "augmentation_eligible": int(info["augmentation_eligible"]),
                "step3_chi_target": info["step3_chi_target"],
            }
        )

    ann_df = pd.DataFrame(annotated_rows)
    rows: List[Dict[str, Any]] = []
    for keys, sub in ann_df.groupby(["split", "temperature", "phi"], dropna=False):
        split, temperature, phi = keys
        row = {
            "split": split,
            "temperature": float(temperature),
            "phi": float(phi),
            "row_count": int(len(sub)),
            "exact_step3_match_count": int(sub["exact_step3_match"].sum()),
            "eligible_count": int(sub["augmentation_eligible"].sum()),
            "eligible_fraction": float(sub["augmentation_eligible"].mean()) if len(sub) else np.nan,
            "step3_chi_target": float(sub["step3_chi_target"].dropna().iloc[0]) if sub["step3_chi_target"].notna().any() else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "temperature", "phi"]).reset_index(drop=True)


class Step62ConditionalDataset(Dataset):
    """Tokenized supervised dataset with on-the-fly Step 3 chi-target augmentation."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        tokenizer: PSmilesTokenizer,
        scaler: ConditionScaler,
        chi_lookup,
        split: str,
        chi_target_augmentation_rate: float,
        train: bool,
        random_seed: int,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.scaler = scaler
        self.chi_lookup = chi_lookup
        self.split = str(split)
        self.train = bool(train)
        self.chi_target_augmentation_rate = float(chi_target_augmentation_rate)
        self.rng = np.random.default_rng(int(random_seed))

    def __len__(self) -> int:
        return len(self.df)

    def _should_augment(self, row: Dict[str, Any]) -> bool:
        if not self.train:
            return False
        if str(row["condition_source"]) != "d_chi":
            return False
        if self.chi_target_augmentation_rate <= 0.0:
            return False
        return bool(self.rng.random() < self.chi_target_augmentation_rate)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx].to_dict()
        encoded = self.tokenizer.encode(
            str(row["SMILES"]),
            add_special_tokens=True,
            padding=True,
            return_attention_mask=True,
        )
        condition_info = build_condition_bundle(
            row,
            scaler=self.scaler,
            chi_lookup=self.chi_lookup,
            augment_with_step3_target=self._should_augment(row),
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "condition_bundle": torch.tensor(condition_info["condition_bundle"], dtype=torch.float32),
            "soluble_target": torch.tensor(float(row["water_miscible"]), dtype=torch.float32),
            "chi_goal": torch.tensor(condition_info["chi_goal"], dtype=torch.float32),
            "chi_observed": torch.tensor(
                float(row["chi"]) if np.isfinite(row.get("chi", np.nan)) else np.nan,
                dtype=torch.float32,
            ),
            "step3_chi_target": torch.tensor(condition_info["step3_chi_target"], dtype=torch.float32),
            "augmentation_eligible": torch.tensor(condition_info["augmentation_eligible"], dtype=torch.long),
            "augmented_to_step3_target": torch.tensor(condition_info["augmented_to_step3_target"], dtype=torch.long),
            "source_is_d_chi": torch.tensor(int(str(row["condition_source"]) == "d_chi"), dtype=torch.long),
            "source_is_d_water": torch.tensor(int(str(row["condition_source"]) == "d_water"), dtype=torch.long),
        }


def step62_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate Step 6_2 supervised samples into one batch."""

    return {key: torch.stack([item[key] for item in batch], dim=0) for key in batch[0].keys()}
