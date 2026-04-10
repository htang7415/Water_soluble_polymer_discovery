"""Config and target-table utilities for Step 6_2."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

from src.chi.data import SplitConfig, add_split_column, load_chi_dataset, make_split_assignments
from src.chi.inverse_design_common import load_soluble_targets
from src.evaluation.class_decode_constraints import load_decode_constraint_source_smiles
from src.evaluation.polymer_class import BACKBONE_CLASS_MATCH_CLASSES, PolymerClassifier
from src.utils.config import load_config
from src.utils.model_scales import get_results_dir
from src.utils.chemistry import canonicalize_smiles


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _as_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _as_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [_as_serializable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _format_float(value: float) -> str:
    return np.format_float_positional(float(value), trim="-")


def _condition_key(temperature: float, phi: float) -> Tuple[str, str]:
    return (_format_float(temperature), _format_float(phi))


def _first_existing(paths: Iterable[Path]) -> Path:
    paths = list(paths)
    for path in paths:
        if path.exists():
            return path
    if not paths:
        raise ValueError("No candidate paths provided.")
    return paths[0]


def _build_target_row_key(c_target: str, temperature: float, phi: float, chi_target: float) -> str:
    return (
        f"{c_target}|T={_format_float(temperature)}"
        f"|phi={_format_float(phi)}|chi={_format_float(chi_target)}"
    )


def _resolve_class_numeric_overrides(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    resolved: Dict[str, float] = {}
    for key, value in raw.items():
        class_name = str(key).strip().lower()
        if not class_name:
            continue
        try:
            resolved[class_name] = float(value)
        except (TypeError, ValueError):
            continue
    return resolved


def resolve_step62_generation_budget(step6_cfg: Dict[str, Any], c_target: str) -> int:
    """Resolve the per-class Step 6_2 generation budget."""
    base_budget = int(step6_cfg["generation_budget"])
    overrides = _resolve_class_numeric_overrides(step6_cfg.get("generation_budget_by_class", {}))
    return int(round(overrides.get(str(c_target).strip().lower(), float(base_budget))))


def resolve_step62_hpo_generation_budget(hpo_cfg: Dict[str, Any], c_target: str) -> int:
    """Resolve the per-class Step 6_2 HPO generation budget."""
    base_budget = int(hpo_cfg["hpo_generation_budget"])
    overrides = _resolve_class_numeric_overrides(hpo_cfg.get("hpo_generation_budget_by_class", {}))
    return int(round(overrides.get(str(c_target).strip().lower(), float(base_budget))))


def resolve_step62_sa_thresholds(step6_cfg: Dict[str, Any], c_target: str) -> Dict[str, float]:
    """Resolve reporting and discovery SA thresholds for a target class."""
    target_key = str(c_target).strip().lower()
    reporting_default = float(step6_cfg["target_sa_max"])
    reporting_overrides = _resolve_class_numeric_overrides(step6_cfg.get("reporting_target_sa_max_by_class", {}))
    discovery_overrides = _resolve_class_numeric_overrides(step6_cfg.get("discovery_target_sa_max_by_class", {}))
    reporting = float(reporting_overrides.get(target_key, reporting_default))
    discovery = float(discovery_overrides.get(target_key, reporting))
    return {
        "reporting": reporting,
        "discovery": discovery,
    }


def select_step62_proxy_target_rows(
    source_df: pd.DataFrame,
    *,
    num_targets: int,
) -> pd.DataFrame:
    """Select a small deterministic proxy slice from benchmark target rows.

    This is used by S4 checkpoint selection so the proxy objective stays aligned
    with the actual inverse-design target distribution instead of a disjoint
    validation table.
    """

    work = source_df.reset_index(drop=True).copy()
    if work.empty:
        return work
    limit = max(1, min(int(num_targets), int(len(work))))
    if limit >= len(work):
        return work

    candidate_indices = np.linspace(0, len(work) - 1, num=limit, dtype=int)
    seen: set[int] = set()
    ordered_indices: List[int] = []
    for idx in candidate_indices.tolist():
        idx = int(idx)
        if idx in seen:
            continue
        seen.add(idx)
        ordered_indices.append(idx)
    if len(ordered_indices) < limit:
        for idx in range(len(work)):
            if idx in seen:
                continue
            seen.add(idx)
            ordered_indices.append(int(idx))
            if len(ordered_indices) >= limit:
                break
    return work.iloc[ordered_indices].reset_index(drop=True)


@dataclass(frozen=True)
class ExactChiTargetLookup:
    """Exact-match Step 3 chi-target lookup."""

    mapping: Dict[Tuple[str, str], float]
    source_path: Path
    property_rule_mapping: Dict[Tuple[str, str], str]
    q025_mapping: Dict[Tuple[str, str], float]
    q975_mapping: Dict[Tuple[str, str], float]

    def lookup(
        self,
        temperature: float,
        phi: float,
        *,
        warn_on_missing: bool = False,
    ) -> Optional[float]:
        value = self.mapping.get(_condition_key(temperature, phi))
        if value is None and warn_on_missing:
            warnings.warn(
                (
                    "Missing exact Step 3 chi_target lookup for "
                    f"(T={temperature}, phi={phi}) in {self.source_path}"
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        return value

    def lookup_row(
        self,
        temperature: float,
        phi: float,
        *,
        warn_on_missing: bool = False,
    ) -> Optional[Dict[str, Any]]:
        key = _condition_key(temperature, phi)
        chi_target = self.mapping.get(key)
        if chi_target is None:
            if warn_on_missing:
                warnings.warn(
                    (
                        "Missing exact Step 3 chi_target lookup for "
                        f"(T={temperature}, phi={phi}) in {self.source_path}"
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
            return None
        q025 = self.q025_mapping.get(key, np.nan)
        q975 = self.q975_mapping.get(key, np.nan)
        return {
            "chi_target": float(chi_target),
            "property_rule": str(self.property_rule_mapping.get(key, "upper_bound")).strip().lower(),
            "chi_target_boot_q025": float(q025) if np.isfinite(q025) else np.nan,
            "chi_target_boot_q975": float(q975) if np.isfinite(q975) else np.nan,
        }


@dataclass(frozen=True)
class ResolvedStep62Config:
    """Fully resolved Step 6_2 configuration."""

    base_config: Dict[str, Any]
    step6_2: Dict[str, Any]
    step6_3: Dict[str, Any]
    step6_2_hpo: Dict[str, Any]
    model_size: Optional[str]
    split_mode: str
    classification_split_mode: str
    c_target: str
    enabled_runs: List[str]
    available_target_classes: List[str]
    polymer_patterns: Dict[str, str]
    results_dir: Path
    results_dir_nosplit: Path
    base_results_dir: Path
    benchmark_root: Path
    compare_root: Path
    step4_reg_dir: Path
    step4_cls_dir: Path
    step4_reg_metrics_dir: Path
    step4_cls_metrics_dir: Path
    step3_targets_path: Path
    chi_lookup: ExactChiTargetLookup
    target_base_df: pd.DataFrame
    target_family_df: pd.DataFrame
    rl_proxy_df: pd.DataFrame
    hpo_target_df: pd.DataFrame
    chi_split_df: pd.DataFrame
    chi_train_stats: Dict[str, float]
    class_support_stats: Dict[str, Any]
    config_snapshot: Dict[str, Any]


def _resolve_step4_dirs(
    base_config: Dict[str, Any],
    *,
    model_size: Optional[str],
    split_mode: str,
) -> Tuple[Path, Path, Path, Path, Path, Path]:
    base_results_dir = Path(base_config["paths"]["results_dir"])
    results_dir = Path(get_results_dir(model_size, base_results_dir, split_mode))
    results_dir_nosplit = Path(get_results_dir(model_size, base_results_dir, None))

    reg_candidates = [
        results_dir_nosplit / "step4_1_regression" / split_mode,
        results_dir / "step4_1_regression" / split_mode,
        results_dir_nosplit / "step4_chi_training" / "step4_1_regression" / split_mode,
        results_dir / "step4_chi_training" / split_mode / "step4_1_regression",
    ]
    cls_candidates = [
        results_dir_nosplit / "step4_2_classification",
        results_dir / "step4_2_classification",
        results_dir_nosplit / "step4_chi_training" / "step4_2_classification",
        results_dir / "step4_chi_training" / split_mode / "step4_2_classification",
    ]

    step4_reg_dir = _first_existing(reg_candidates)
    step4_cls_dir = _first_existing(cls_candidates)
    return (
        results_dir,
        results_dir_nosplit,
        base_results_dir,
        step4_reg_dir,
        step4_cls_dir,
        step4_reg_dir / "metrics",
    )


def _resolve_step4_metrics_dirs(
    base_config: Dict[str, Any],
    *,
    model_size: Optional[str],
    split_mode: str,
) -> Tuple[Path, Path, Path, Path, Path, Path, Path]:
    (
        results_dir,
        results_dir_nosplit,
        base_results_dir,
        step4_reg_dir,
        step4_cls_dir,
        step4_reg_metrics_dir,
    ) = _resolve_step4_dirs(base_config, model_size=model_size, split_mode=split_mode)
    step4_cls_metrics_dir = step4_cls_dir / "metrics"
    return (
        results_dir,
        results_dir_nosplit,
        base_results_dir,
        step4_reg_dir,
        step4_cls_dir,
        step4_reg_metrics_dir,
        step4_cls_metrics_dir,
    )


def _resolve_split_ratios(base_config: Dict[str, Any]) -> Dict[str, float]:
    chi_cfg = base_config.get("chi_training", {})
    shared_cfg = chi_cfg.get("shared", {}) if isinstance(chi_cfg.get("shared"), dict) else {}
    split_cfg = shared_cfg.get("split", {}) if isinstance(shared_cfg.get("split"), dict) else {}
    step41_cfg = (
        chi_cfg.get("step4_1_regression", {})
        if isinstance(chi_cfg.get("step4_1_regression"), dict)
        else {}
    )
    tuning_cv_folds = int(step41_cfg.get("tuning_cv_folds", 5))
    holdout_test_ratio = split_cfg.get("holdout_test_ratio")
    if holdout_test_ratio is None:
        holdout_test_ratio = 1.0 / float(max(2, tuning_cv_folds))
    test_ratio = float(holdout_test_ratio)
    dev_ratio = 1.0 - test_ratio
    val_ratio = dev_ratio / float(max(2, tuning_cv_folds))
    train_ratio = dev_ratio - val_ratio
    return {
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
    }


def _load_or_build_chi_split_df(
    *,
    base_config: Dict[str, Any],
    step4_reg_dir: Path,
    step4_reg_metrics_dir: Path,
    split_mode: str,
    random_seed: int,
) -> pd.DataFrame:
    split_candidates = [
        step4_reg_metrics_dir / "chi_dataset_with_split.csv",
        step4_reg_dir / "shared" / f"chi_dataset_with_split_step4_1_{split_mode}.csv",
        step4_reg_dir / "shared" / "chi_dataset_with_split.csv",
    ]
    split_path = next((path for path in split_candidates if path.exists()), None)
    if split_path is not None:
        df = pd.read_csv(split_path)
        if "split" not in df.columns:
            raise ValueError(f"Split-aware chi dataset missing 'split' column: {split_path}")
        return df

    warnings.warn(
        (
            "Step 4 split-aware chi dataset not found; rebuilding deterministic "
            "Step 6_2 D_chi split locally from config."
        ),
        RuntimeWarning,
        stacklevel=2,
    )
    dataset_path = base_config["chi_training"]["shared"]["dataset_path"]
    split_ratios = _resolve_split_ratios(base_config)
    chi_df = load_chi_dataset(dataset_path)
    assignments = make_split_assignments(
        chi_df,
        SplitConfig(
            split_mode=split_mode,
            train_ratio=split_ratios["train_ratio"],
            val_ratio=split_ratios["val_ratio"],
            test_ratio=split_ratios["test_ratio"],
            seed=int(random_seed),
        ),
    )
    return add_split_column(chi_df, assignments)


def _build_lookup_and_target_tables(
    *,
    base_config: Dict[str, Any],
    step6_cfg: Dict[str, Any],
    results_dir: Path,
    base_results_dir: Path,
    split_mode: str,
) -> Tuple[Path, ExactChiTargetLookup, pd.DataFrame, pd.DataFrame]:
    target_base_df, path_used = load_soluble_targets(
        targets_csv=None,
        results_dir=results_dir,
        base_results_dir=base_results_dir,
        split_mode=split_mode,
    )
    target_base_df = target_base_df.rename(columns={"target_chi": "chi_target"}).copy()
    target_base_df["temperature"] = target_base_df["temperature"].astype(float)
    target_base_df["phi"] = target_base_df["phi"].astype(float)
    target_base_df["chi_target"] = target_base_df["chi_target"].astype(float)
    lookup = ExactChiTargetLookup(
        mapping={
            _condition_key(float(row["temperature"]), float(row["phi"])): float(row["chi_target"])
            for _, row in target_base_df.iterrows()
        },
        source_path=Path(path_used),
        property_rule_mapping={
            _condition_key(float(row["temperature"]), float(row["phi"])): str(
                row.get("property_rule", "upper_bound")
            ).strip().lower()
            for _, row in target_base_df.iterrows()
        },
        q025_mapping={
            _condition_key(float(row["temperature"]), float(row["phi"])): float(value)
            for _, row in target_base_df.iterrows()
            for value in [pd.to_numeric(row.get("chi_target_boot_q025", np.nan), errors="coerce")]
            if np.isfinite(value)
        },
        q975_mapping={
            _condition_key(float(row["temperature"]), float(row["phi"])): float(value)
            for _, row in target_base_df.iterrows()
            for value in [pd.to_numeric(row.get("chi_target_boot_q975", np.nan), errors="coerce")]
            if np.isfinite(value)
        },
    )

    c_target = str(step6_cfg["c_target"]).strip().lower()
    rows: List[Dict[str, Any]] = []
    for idx, row in target_base_df.sort_values(["temperature", "phi"]).reset_index(drop=True).iterrows():
        temperature = float(row["temperature"])
        phi = float(row["phi"])
        chi_target = float(row["chi_target"])
        property_rule = str(row.get("property_rule", "upper_bound")).strip().lower()
        target_row_id = int(idx + 1)
        target_row_key = _build_target_row_key(c_target, temperature, phi, chi_target)
        row_payload = {
            "target_row_id": target_row_id,
            "target_id": target_row_id,
            "target_row_key": target_row_key,
            "c_target": c_target,
            "target_polymer_class": c_target,
            "temperature": temperature,
            "phi": phi,
            "chi_target": chi_target,
            "target_chi": chi_target,
            "property_rule": property_rule,
        }
        for optional_col in ("chi_target_boot_q025", "chi_target_boot_q975"):
            optional_value = pd.to_numeric(row.get(optional_col, np.nan), errors="coerce")
            row_payload[optional_col] = float(optional_value) if np.isfinite(optional_value) else np.nan
        rows.append(row_payload)
    target_family_df = pd.DataFrame(rows)
    return Path(path_used), lookup, target_base_df, target_family_df


def _build_validation_target_tables(
    *,
    chi_split_df: pd.DataFrame,
    c_target: str,
    chi_lookup: ExactChiTargetLookup,
    rl_proxy_num_targets: int,
    hpo_enabled: bool,
    hpo_num_targets: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    val_df = chi_split_df.loc[chi_split_df["split"].astype(str) == "val"].copy()
    if val_df.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "coverage_warning": "no_val_rows",
                "proxy_target_gap_warning": False,
                "proxy_target_gap_warning_threshold": 0.3,
            },
            pd.DataFrame(),
        )

    val_df["canonical_smiles"] = val_df["SMILES"].astype(str).map(canonicalize_smiles)
    val_df["canonical_smiles"] = val_df["canonical_smiles"].where(
        val_df["canonical_smiles"].notna(),
        val_df["SMILES"].astype(str),
    )
    val_df["step3_chi_target"] = [
        chi_lookup.lookup(float(t), float(p), warn_on_missing=False)
        for t, p in zip(val_df["temperature"], val_df["phi"])
    ]
    eligible_df = val_df.loc[val_df["step3_chi_target"].notna()].copy()
    if eligible_df.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "coverage_warning": "no_exact_step3_match",
                "proxy_target_gap_warning": False,
                "proxy_target_gap_warning_threshold": 0.3,
            },
            pd.DataFrame(),
        )

    ordered_buckets = (
        eligible_df[["temperature", "phi"]]
        .drop_duplicates()
        .sort_values(["temperature", "phi"])
        .reset_index(drop=True)
    )
    rl_bucket_count = min(int(rl_proxy_num_targets), int(len(ordered_buckets)))
    hpo_bucket_count = (
        min(int(hpo_num_targets), max(0, int(len(ordered_buckets)) - rl_bucket_count))
        if hpo_enabled
        else 0
    )

    rl_buckets = ordered_buckets.iloc[:rl_bucket_count].copy()
    hpo_buckets = ordered_buckets.iloc[rl_bucket_count : rl_bucket_count + hpo_bucket_count].copy()

    def _bucket_rows(bucket_df: pd.DataFrame, *, source_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rows: List[Dict[str, Any]] = []
        drift_rows: List[Dict[str, Any]] = []
        for idx, bucket in bucket_df.reset_index(drop=True).iterrows():
            bucket_rows = eligible_df.loc[
                np.isclose(eligible_df["temperature"].astype(float), float(bucket["temperature"]))
                & np.isclose(eligible_df["phi"].astype(float), float(bucket["phi"]))
            ].copy()
            bucket_rows["abs_step3_gap"] = (
                pd.to_numeric(bucket_rows["chi"], errors="coerce")
                - pd.to_numeric(bucket_rows["step3_chi_target"], errors="coerce")
            ).abs()
            bucket_rows = bucket_rows.sort_values(
                ["abs_step3_gap", "canonical_smiles", "Polymer", "row_id"]
            ).reset_index(drop=True)
            selected = bucket_rows.iloc[0]
            chi_target = float(selected["chi"])
            step3_target = float(selected["step3_chi_target"])
            abs_step3_gap = abs(chi_target - step3_target)
            target_row_id = int(idx + 1)
            rows.append(
                {
                    "target_row_id": target_row_id,
                    "target_id": target_row_id,
                    "target_row_key": _build_target_row_key(
                        c_target,
                        float(selected["temperature"]),
                        float(selected["phi"]),
                        chi_target,
                    ),
                    "c_target": c_target,
                    "target_polymer_class": c_target,
                    "temperature": float(selected["temperature"]),
                    "phi": float(selected["phi"]),
                    "chi_target": chi_target,
                    "target_chi": chi_target,
                    "property_rule": "upper_bound",
                    "proxy_source": source_name,
                    "source_row_id": int(selected["row_id"]),
                    "source_polymer": str(selected["Polymer"]),
                    "source_canonical_smiles": str(selected["canonical_smiles"]),
                    "step3_chi_target": step3_target,
                    "abs_step3_gap": abs_step3_gap,
                }
            )
            drift_rows.append(
                {
                    "proxy_source": source_name,
                    "temperature": float(selected["temperature"]),
                    "phi": float(selected["phi"]),
                    "proxy_chi_target": chi_target,
                    "step3_chi_target": step3_target,
                    "abs_step3_gap": abs_step3_gap,
                }
            )
        return pd.DataFrame(rows), pd.DataFrame(drift_rows)

    rl_proxy_df, rl_drift_df = _bucket_rows(rl_buckets, source_name="rl_proxy")
    hpo_target_df, hpo_drift_df = _bucket_rows(hpo_buckets, source_name="hpo") if hpo_enabled else (pd.DataFrame(), pd.DataFrame())
    drift_df = pd.concat([rl_drift_df, hpo_drift_df], ignore_index=True) if not rl_drift_df.empty or not hpo_drift_df.empty else pd.DataFrame()
    rl_mean_gap = float(rl_drift_df["abs_step3_gap"].mean()) if not rl_drift_df.empty else np.nan
    rl_max_gap = float(rl_drift_df["abs_step3_gap"].max()) if not rl_drift_df.empty else np.nan
    hpo_mean_gap = float(hpo_drift_df["abs_step3_gap"].mean()) if not hpo_drift_df.empty else np.nan
    hpo_max_gap = float(hpo_drift_df["abs_step3_gap"].max()) if not hpo_drift_df.empty else np.nan
    gap_threshold = 0.3
    diagnostics = {
        "eligible_validation_buckets": int(len(ordered_buckets)),
        "rl_proxy_bucket_count": int(rl_bucket_count),
        "hpo_bucket_count": int(hpo_bucket_count),
        "hpo_enabled": bool(hpo_enabled),
        "rl_proxy_mean_abs_step3_gap": rl_mean_gap,
        "rl_proxy_max_abs_step3_gap": rl_max_gap,
        "hpo_mean_abs_step3_gap": hpo_mean_gap,
        "hpo_max_abs_step3_gap": hpo_max_gap,
        "proxy_target_gap_warning_threshold": float(gap_threshold),
        "proxy_target_gap_warning": bool(
            (np.isfinite(rl_mean_gap) and rl_mean_gap > gap_threshold)
            or (np.isfinite(hpo_mean_gap) and hpo_mean_gap > gap_threshold)
        ),
        "coverage_warning": None,
    }
    if hpo_enabled and hpo_bucket_count < int(hpo_num_targets):
        diagnostics["coverage_warning"] = (
            f"Requested hpo_num_targets={int(hpo_num_targets)} but only {int(hpo_bucket_count)} "
            "disjoint validation buckets remained after RL proxy reservation."
        )
    return rl_proxy_df, hpo_target_df, diagnostics, drift_df


def _compute_chi_train_stats(chi_split_df: pd.DataFrame) -> Dict[str, float]:
    train_df = chi_split_df.loc[chi_split_df["split"].astype(str) == "train"].copy()
    if train_df.empty:
        raise ValueError("No train rows found in split-aware D_chi dataframe.")
    return {
        "temperature_min": float(train_df["temperature"].min()),
        "temperature_max": float(train_df["temperature"].max()),
        "phi_min": float(train_df["phi"].min()),
        "phi_max": float(train_df["phi"].max()),
        "chi_goal_min": float(train_df["chi"].min()),
        "chi_goal_max": float(train_df["chi"].max()),
    }


def _compute_class_support_stats(
    *,
    c_target: str,
    polymer_patterns: Dict[str, str],
    source_smiles: Iterable[str],
) -> Dict[str, Any]:
    source_smiles = [str(smi) for smi in source_smiles]
    classifier = PolymerClassifier(patterns=polymer_patterns)
    positive_loose = 0
    positive_strict = 0
    target_is_backbone = bool(str(c_target).strip().lower() in BACKBONE_CLASS_MATCH_CLASSES)
    for smiles in source_smiles:
        try:
            if classifier.classify(smiles).get(c_target, False):
                positive_loose += 1
            if classifier.classify_backbone(smiles).get(c_target, False):
                positive_strict += 1
        except Exception:
            continue
    return {
        "c_target": c_target,
        "target_class_backbone_defined": bool(target_is_backbone),
        "source_corpus_size": int(len(source_smiles)),
        "source_positive_count": int(positive_loose),
        "source_positive_count_loose": int(positive_loose),
        "source_positive_count_strict": int(positive_strict),
        "token_bias_available": bool(positive_loose >= 10),
        "sparse_class_warning": bool(positive_loose < 20),
        "strict_sparse_class_warning": bool(positive_strict < 20),
    }


def _validate_step6_2_config(
    *,
    step6_cfg: Dict[str, Any],
    polymer_patterns: Dict[str, str],
) -> None:
    split_mode = str(step6_cfg["split_mode"]).strip().lower()
    classification_split_mode = str(step6_cfg["classification_split_mode"]).strip().lower()
    if split_mode not in {"polymer", "random"}:
        raise ValueError("step6_2.split_mode must be one of {'polymer', 'random'}")
    if classification_split_mode not in {"polymer", "random"}:
        raise ValueError("step6_2.classification_split_mode must be one of {'polymer', 'random'}")

    c_target = str(step6_cfg["c_target"]).strip().lower()
    available = [str(x).strip().lower() for x in step6_cfg.get("available_target_classes", [])]
    if c_target not in available:
        raise ValueError(
            f"step6_2.c_target={c_target!r} is not in available_target_classes={available}"
        )
    if c_target not in polymer_patterns:
        raise ValueError(
            f"step6_2.c_target={c_target!r} is not defined in config.yaml polymer_classes"
        )
    seeds = list(step6_cfg["sampling_seeds"])
    if len(seeds) != int(step6_cfg["num_sampling_rounds"]):
        raise ValueError("sampling_seeds length must equal num_sampling_rounds")
    enabled_runs = list(step6_cfg["enabled_runs"])
    if not enabled_runs:
        raise ValueError("step6_2.enabled_runs is empty")


def build_run_config(resolved: ResolvedStep62Config, run_name: str) -> Dict[str, Any]:
    if run_name not in resolved.enabled_runs:
        raise ValueError(f"Run {run_name!r} is not enabled in step6_2.enabled_runs")
    run_cfg = deepcopy(resolved.step6_2)
    override = deepcopy(run_cfg.get("run_overrides", {}).get(run_name, {}))
    canonical_family = str(
        override.get("canonical_family", run_name.split("_", 1)[0])
    ).strip()
    run_cfg = _deep_merge(run_cfg, override)
    run_cfg["run_name"] = run_name
    run_cfg["canonical_family"] = canonical_family
    run_cfg["c_target"] = resolved.c_target
    return _apply_step62_class_overrides(run_cfg, c_target=resolved.c_target)


def _apply_step62_class_overrides(value: Any, *, c_target: str) -> Any:
    if isinstance(value, dict):
        resolved = {
            str(key): _apply_step62_class_overrides(item, c_target=c_target)
            for key, item in value.items()
        }
        for key, item in list(resolved.items()):
            if not key.endswith("_by_class_overrides") or not isinstance(item, dict):
                continue
            base_key = key[: -len("_by_class_overrides")]
            if c_target in item:
                override_value = _apply_step62_class_overrides(item[c_target], c_target=c_target)
                if isinstance(resolved.get(base_key), dict) and isinstance(override_value, dict):
                    resolved[base_key] = _deep_merge(resolved[base_key], override_value)
                else:
                    resolved[base_key] = override_value
        return resolved
    if isinstance(value, list):
        return [_apply_step62_class_overrides(item, c_target=c_target) for item in value]
    return value


def _build_snapshot(
    *,
    base_config: Dict[str, Any],
    step6_cfg: Dict[str, Any],
    step63_cfg: Dict[str, Any],
    hpo_cfg: Dict[str, Any],
    model_size: Optional[str],
    paths: Dict[str, Path],
    target_family_df: pd.DataFrame,
    rl_proxy_df: pd.DataFrame,
    hpo_target_df: pd.DataFrame,
    chi_train_stats: Dict[str, float],
    class_support_stats: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "model_size": model_size,
        "paths": _as_serializable(paths),
        "step6_2": _as_serializable(step6_cfg),
        "step6_3": _as_serializable(step63_cfg),
        "step6_2_hpo": _as_serializable(hpo_cfg),
        "derived": {
            "num_target_rows": int(len(target_family_df)),
            "num_rl_proxy_rows": int(len(rl_proxy_df)),
            "num_hpo_rows": int(len(hpo_target_df)),
            "chi_train_stats": _as_serializable(chi_train_stats),
            "class_support_stats": _as_serializable(class_support_stats),
            "validation_bucket_diagnostics": _as_serializable(diagnostics),
            "base_results_dir": str(base_config["paths"]["results_dir"]),
        },
    }


def load_step6_2_config(
    *,
    config_path: str = "configs/config6_2.yaml",
    base_config_path: str = "configs/config.yaml",
    model_size: Optional[str] = None,
    force_hpo_enabled: bool = False,
) -> ResolvedStep62Config:
    base_config = load_config(base_config_path)
    step6_bundle = load_config(config_path)
    step6_cfg = deepcopy(step6_bundle.get("step6_2", {}))
    step63_cfg = deepcopy(step6_bundle.get("step6_3", {}))
    hpo_cfg = deepcopy(step6_bundle.get("step6_2_hpo", {}))
    if force_hpo_enabled:
        hpo_cfg["enabled"] = True
    if not step6_cfg:
        raise ValueError(f"No step6_2 block found in {config_path}")

    polymer_patterns = {
        str(key).strip().lower(): str(value)
        for key, value in base_config.get("polymer_classes", {}).items()
    }
    _validate_step6_2_config(step6_cfg=step6_cfg, polymer_patterns=polymer_patterns)

    split_mode = str(step6_cfg["split_mode"]).strip().lower()
    classification_split_mode = str(step6_cfg["classification_split_mode"]).strip().lower()
    c_target = str(step6_cfg["c_target"]).strip().lower()
    step6_cfg = _apply_step62_class_overrides(step6_cfg, c_target=c_target)
    enabled_runs = [str(name) for name in step6_cfg["enabled_runs"]]
    available_target_classes = [str(x).strip().lower() for x in step6_cfg["available_target_classes"]]

    (
        results_dir,
        results_dir_nosplit,
        base_results_dir,
        step4_reg_dir,
        step4_cls_dir,
        step4_reg_metrics_dir,
        step4_cls_metrics_dir,
    ) = _resolve_step4_metrics_dirs(base_config, model_size=model_size, split_mode=split_mode)
    benchmark_root = (
        results_dir / "step6_2_inverse_design" / split_mode / c_target
    )
    compare_root = (
        results_dir / "step6_3_inverse_design_compare" / split_mode / c_target
    )

    step3_targets_path, chi_lookup, target_base_df, target_family_df = _build_lookup_and_target_tables(
        base_config=base_config,
        step6_cfg=step6_cfg,
        results_dir=results_dir,
        base_results_dir=base_results_dir,
        split_mode=split_mode,
    )
    chi_split_df = _load_or_build_chi_split_df(
        base_config=base_config,
        step4_reg_dir=step4_reg_dir,
        step4_reg_metrics_dir=step4_reg_metrics_dir,
        split_mode=split_mode,
        random_seed=int(step6_cfg["random_seed"]),
    )
    rl_proxy_df, hpo_target_df, diagnostics, drift_df = _build_validation_target_tables(
        chi_split_df=chi_split_df,
        c_target=c_target,
        chi_lookup=chi_lookup,
        rl_proxy_num_targets=int(step6_cfg["s4"]["rl_proxy_num_targets"]),
        hpo_enabled=bool(hpo_cfg.get("enabled", False)),
        hpo_num_targets=int(hpo_cfg.get("hpo_num_targets", 0)),
    )
    chi_train_stats = _compute_chi_train_stats(chi_split_df)
    training_source_smiles = load_decode_constraint_source_smiles(Path(base_config["paths"]["data_dir"]))
    class_support_stats = _compute_class_support_stats(
        c_target=c_target,
        polymer_patterns=polymer_patterns,
        source_smiles=training_source_smiles,
    )
    if class_support_stats["sparse_class_warning"]:
        warnings.warn(
            (
                f"Low-support c_target={c_target!r}: "
                f"source_positive_count_loose={class_support_stats['source_positive_count_loose']} "
                f"source_positive_count_strict={class_support_stats['source_positive_count_strict']}"
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    paths = {
        "results_dir": results_dir,
        "results_dir_nosplit": results_dir_nosplit,
        "base_results_dir": base_results_dir,
        "benchmark_root": benchmark_root,
        "compare_root": compare_root,
        "step3_targets_path": step3_targets_path,
        "step4_reg_dir": step4_reg_dir,
        "step4_cls_dir": step4_cls_dir,
        "step4_reg_metrics_dir": step4_reg_metrics_dir,
        "step4_cls_metrics_dir": step4_cls_metrics_dir,
    }
    snapshot = _build_snapshot(
        base_config=base_config,
        step6_cfg=step6_cfg,
        step63_cfg=step63_cfg,
        hpo_cfg=hpo_cfg,
        model_size=model_size,
        paths=paths,
        target_family_df=target_family_df,
        rl_proxy_df=rl_proxy_df,
        hpo_target_df=hpo_target_df,
        chi_train_stats=chi_train_stats,
        class_support_stats=class_support_stats,
        diagnostics=diagnostics,
    )
    if not drift_df.empty:
        snapshot["derived"]["validation_target_drift"] = _as_serializable(
            {
                "mean_abs_step3_gap": float(drift_df["abs_step3_gap"].mean()),
                "max_abs_step3_gap": float(drift_df["abs_step3_gap"].max()),
            }
        )

    return ResolvedStep62Config(
        base_config=base_config,
        step6_2=step6_cfg,
        step6_3=step63_cfg,
        step6_2_hpo=hpo_cfg,
        model_size=model_size,
        split_mode=split_mode,
        classification_split_mode=classification_split_mode,
        c_target=c_target,
        enabled_runs=enabled_runs,
        available_target_classes=available_target_classes,
        polymer_patterns=polymer_patterns,
        results_dir=results_dir,
        results_dir_nosplit=results_dir_nosplit,
        base_results_dir=base_results_dir,
        benchmark_root=benchmark_root,
        compare_root=compare_root,
        step4_reg_dir=step4_reg_dir,
        step4_cls_dir=step4_cls_dir,
        step4_reg_metrics_dir=step4_reg_metrics_dir,
        step4_cls_metrics_dir=step4_cls_metrics_dir,
        step3_targets_path=step3_targets_path,
        chi_lookup=chi_lookup,
        target_base_df=target_base_df,
        target_family_df=target_family_df,
        rl_proxy_df=rl_proxy_df,
        hpo_target_df=hpo_target_df,
        chi_split_df=chi_split_df,
        chi_train_stats=chi_train_stats,
        class_support_stats=class_support_stats,
        config_snapshot=snapshot,
    )
