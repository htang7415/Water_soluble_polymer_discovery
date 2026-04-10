"""Common utilities for chi inverse-design steps."""

from __future__ import annotations

from pathlib import Path
import gzip
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple
import warnings

import torch
import numpy as np
import pandas as pd

from src.chi.embeddings import (
    build_backbone_architecture,
    load_backbone_only_from_step1,
    resolve_step1_artifacts,
)
from src.data.tokenizer import PSmilesTokenizer
from src.chi.model import (
    BackboneDirectChiModel,
    BackbonePhysicsGuidedChiModel,
    BackboneSolubilityClassifierModel,
    DirectChiRegressor,
    PhysicsGuidedChiModel,
    SolubilityClassifier,
)
from src.utils.chemistry import canonicalize_smiles, check_validity, count_stars, has_terminal_connection_stars
from src.utils.figure_style import apply_publication_figure_style
from src.utils.numerics import stable_sigmoid

CLASS_LABEL_INTERNAL = "water_miscible"
CLASS_LABEL_PUBLIC = "water_miscible"

CLASS_NAME_MAP = {1: "Water-miscible", 0: "Water-immiscible"}
_TRAINING_SMILES_CACHE: Dict[Tuple[str, str], set[str]] = {}


def unique_preserving_order(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _ensure_internal_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if CLASS_LABEL_INTERNAL not in out.columns and CLASS_LABEL_PUBLIC in out.columns:
        out[CLASS_LABEL_INTERNAL] = out[CLASS_LABEL_PUBLIC]
    if CLASS_LABEL_INTERNAL in out.columns and CLASS_LABEL_PUBLIC not in out.columns:
        out[CLASS_LABEL_PUBLIC] = out[CLASS_LABEL_INTERNAL]
    return out


def default_chi_config(config: Dict, step: str | None = None) -> Dict:
    chi_cfg = config.get("chi_training", {})
    shared = chi_cfg.get("shared", {}) if isinstance(chi_cfg.get("shared", {}), dict) else {}
    shared_embedding = shared.get("embedding", {}) if isinstance(shared.get("embedding", {}), dict) else {}
    step5_cfg = (
        chi_cfg.get("step5_class_inverse_design", {})
        if isinstance(chi_cfg.get("step5_class_inverse_design", {}), dict)
        else {}
    )

    defaults = {
        "split_mode": "polymer",
        "epsilon": 0.05,
        "class_weight": None,  # deprecated (kept for backward-compatible parsing)
        "polymer_class_weight": None,  # deprecated (kept for backward-compatible parsing)
        "candidate_source": "novel",
        "property_rule": "upper_bound",
        "coverage_topk": 5,
        "target_temperature": 293.15,
        "target_phi": 0.2,
        "target_polymer_class": "all",
        "target_polymer_count": 100,
        "target_sa_max": 4.0,
        "embedding_batch_size": 128,
        "embedding_timestep": int(config.get("training_property", {}).get("default_timestep", 1)),
        "uncertainty_enabled": False,
        "uncertainty_mc_samples": 20,
        "uncertainty_class_z": 1.0,
        "uncertainty_property_z": 1.0,
        "uncertainty_score_weight": 0.0,
        "uncertainty_seed": int(config.get("data", {}).get("random_seed", 42)),
    }

    out = defaults.copy()
    out["split_mode"] = str(shared.get("split_mode", chi_cfg.get("split_mode", defaults["split_mode"])))
    out["embedding_batch_size"] = int(
        shared_embedding.get("batch_size", chi_cfg.get("embedding_batch_size", defaults["embedding_batch_size"]))
    )
    out["embedding_timestep"] = int(
        shared_embedding.get("timestep", chi_cfg.get("embedding_timestep", defaults["embedding_timestep"]))
    )

    if step == "step5":
        out.update(step5_cfg)
    else:
        # Backward-compatible union if caller does not specify a step.
        out.update(step5_cfg)

    # Legacy flat keys still override defaults when present.
    for key in [
        "epsilon",
        "class_weight",
        "polymer_class_weight",
        "candidate_source",
        "property_rule",
        "coverage_topk",
        "target_temperature",
        "target_phi",
        "target_polymer_class",
        "target_polymer_count",
        "target_sa_max",
        "uncertainty_enabled",
        "uncertainty_mc_samples",
        "uncertainty_class_z",
        "uncertainty_property_z",
        "uncertainty_score_weight",
        "uncertainty_seed",
    ]:
        if key in chi_cfg:
            out[key] = chi_cfg[key]

    return out


def set_plot_style(font_size: int) -> None:
    apply_publication_figure_style(font_size=font_size, remove_titles=True)


def _safe_numeric(value, default=np.nan):
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def load_step2_resampling_step_summary(summary_csv: str | Path | None) -> Dict[str, object]:
    if summary_csv is None:
        return {}
    path = Path(summary_csv)
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty:
        return {}

    row = df.iloc[0].to_dict()
    target_met_raw = row.get("valid_only_target_met", None)
    target_met = None
    if target_met_raw is not None and not pd.isna(target_met_raw):
        if isinstance(target_met_raw, str):
            target_met = int(target_met_raw.strip().lower() in {"1", "true", "yes", "y"})
        else:
            target_met = int(bool(target_met_raw))
    skip_novelty_raw = row.get("valid_only_skip_novelty_filter", None)
    skip_novelty = None
    if skip_novelty_raw is not None and not pd.isna(skip_novelty_raw):
        if isinstance(skip_novelty_raw, str):
            skip_novelty = int(skip_novelty_raw.strip().lower() in {"1", "true", "yes", "y"})
        else:
            skip_novelty = int(bool(skip_novelty_raw))
    skip_sa_raw = row.get("valid_only_skip_sa_filter", None)
    skip_sa = None
    if skip_sa_raw is not None and not pd.isna(skip_sa_raw):
        if isinstance(skip_sa_raw, str):
            skip_sa = int(skip_sa_raw.strip().lower() in {"1", "true", "yes", "y"})
        else:
            skip_sa = int(bool(skip_sa_raw))

    return {
        "step2_generation_goal": int(_safe_numeric(row.get("generation_goal"), default=np.nan))
        if np.isfinite(_safe_numeric(row.get("generation_goal"), default=np.nan))
        else None,
        "step2_generated_count_raw": int(_safe_numeric(row.get("generated_count"), default=np.nan))
        if np.isfinite(_safe_numeric(row.get("generated_count"), default=np.nan))
        else None,
        "step2_accepted_count": int(_safe_numeric(row.get("accepted_count_for_evaluation"), default=np.nan))
        if np.isfinite(_safe_numeric(row.get("accepted_count_for_evaluation"), default=np.nan))
        else None,
        "step2_valid_only_rounds": int(_safe_numeric(row.get("valid_only_rounds"), default=np.nan))
        if np.isfinite(_safe_numeric(row.get("valid_only_rounds"), default=np.nan))
        else None,
        "step2_valid_only_acceptance_rate": _safe_numeric(row.get("valid_only_acceptance_rate"), default=np.nan),
        "step2_valid_only_shortfall_count": int(_safe_numeric(row.get("valid_only_shortfall_count"), default=np.nan))
        if np.isfinite(_safe_numeric(row.get("valid_only_shortfall_count"), default=np.nan))
        else None,
        "step2_valid_only_target_met": target_met,
        "step2_valid_only_skip_novelty_filter": skip_novelty,
        "step2_valid_only_skip_sa_filter": skip_sa,
        "step2_sampling_time_sec": _safe_numeric(row.get("sampling_time_sec"), default=np.nan),
        "step2_samples_per_sec": _safe_numeric(row.get("samples_per_sec"), default=np.nan),
    }


def parse_candidate_source(value: str) -> str:
    v = value.strip().lower()
    if v in {"novel", "generated", "step2"}:
        return "novel"
    raise ValueError(
        "candidate_source must be 'novel' for this workflow "
        "(aliases: generated/step2 -> novel). "
        "Step 5/6 candidate pools are restricted to Step 2-generated polymers."
    )


def _normalize_checkpoint_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(str(k).startswith("_orig_mod.") for k in state_dict.keys()):
        return {str(k).replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _candidate_pool_columns() -> List[str]:
    return unique_preserving_order([
        "polymer_id",
        "Polymer",
        "SMILES",
        "canonical_smiles",
        "target_id",
        "temperature",
        "phi",
        "target_chi",
        "property_rule",
        CLASS_LABEL_INTERNAL,
        CLASS_LABEL_PUBLIC,
        "class_logit",
        "class_logit_std",
        "class_prob",
        "class_prob_std",
        "chi_pred_target",
        "chi_pred_std_target",
        "candidate_source",
        "is_novel_vs_train",
    ])


def _empty_candidate_pool() -> pd.DataFrame:
    return pd.DataFrame(columns=_candidate_pool_columns())


def _condition_key_series(temperature: pd.Series, phi: pd.Series) -> pd.Series:
    temperature = pd.to_numeric(temperature, errors="coerce")
    phi = pd.to_numeric(phi, errors="coerce")
    return (
        temperature.map(lambda v: "nan" if pd.isna(v) else f"{float(v):.12g}")
        + "|"
        + phi.map(lambda v: "nan" if pd.isna(v) else f"{float(v):.12g}")
    )


def _prepare_target_conditions(target_df: pd.DataFrame) -> pd.DataFrame:
    if target_df is None or target_df.empty:
        raise ValueError("target_df is required and must contain at least one target condition.")

    out = target_df.copy().reset_index(drop=True)
    required = {"target_id", "temperature", "phi", "target_chi"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"target_df missing required columns: {missing}")

    out["target_id"] = pd.to_numeric(out["target_id"], errors="coerce").astype(int)
    out["temperature"] = pd.to_numeric(out["temperature"], errors="coerce")
    out["phi"] = pd.to_numeric(out["phi"], errors="coerce")
    out["target_chi"] = pd.to_numeric(out["target_chi"], errors="coerce")
    if out[["temperature", "phi", "target_chi"]].isna().any().any():
        raise ValueError("target_df contains non-numeric temperature, phi, or target_chi values.")
    if "property_rule" not in out.columns:
        out["property_rule"] = "upper_bound"
    out["property_rule"] = out["property_rule"].fillna("upper_bound").astype(str)
    out["_condition_key"] = _condition_key_series(out["temperature"], out["phi"])
    return out


def _count_candidate_polymers(candidate_df: pd.DataFrame) -> int:
    if candidate_df.empty:
        return 0
    if "canonical_smiles" in candidate_df.columns:
        return int(candidate_df["canonical_smiles"].astype(str).nunique())
    if "polymer_id" in candidate_df.columns:
        return int(pd.to_numeric(candidate_df["polymer_id"], errors="coerce").nunique())
    return int(candidate_df["SMILES"].astype(str).nunique()) if "SMILES" in candidate_df.columns else int(len(candidate_df))


def _load_known_candidates_from_step4_metrics(
    step4_reg_metrics_dir: Path,
    step4_cls_metrics_dir: Path,
    training_canonical: set[str],
    target_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    target_info = _prepare_target_conditions(target_df)

    reg_all = step4_reg_metrics_dir / "chi_predictions_all.csv"
    if reg_all.exists():
        reg_df = pd.read_csv(reg_all)
        reg_source = reg_all
    else:
        split_paths = [
            step4_reg_metrics_dir / "chi_predictions_train.csv",
            step4_reg_metrics_dir / "chi_predictions_val.csv",
            step4_reg_metrics_dir / "chi_predictions_test.csv",
        ]
        available = [p for p in split_paths if p.exists()]
        if not available:
            raise FileNotFoundError(
                "Known candidate source requires Step4 regression chi predictions. "
                f"Expected: {reg_all} or split files under {step4_reg_metrics_dir}"
            )
        reg_df = pd.concat([pd.read_csv(p) for p in available], ignore_index=True)
        reg_source = available[0]

    required_reg = {"polymer_id", "Polymer", "SMILES", "temperature", "phi", "chi_pred"}
    missing_reg = sorted(required_reg - set(reg_df.columns))
    if missing_reg:
        raise ValueError(f"Regression known-candidate file missing columns: {missing_reg}")

    reg_df = reg_df.copy()
    reg_df["temperature"] = pd.to_numeric(reg_df["temperature"], errors="coerce")
    reg_df["phi"] = pd.to_numeric(reg_df["phi"], errors="coerce")
    reg_df["chi_pred"] = pd.to_numeric(reg_df["chi_pred"], errors="coerce")
    reg_df = reg_df.dropna(subset=["temperature", "phi", "chi_pred"]).reset_index(drop=True)
    reg_df["canonical_smiles"] = reg_df["SMILES"].astype(str).map(canonicalize_smiles)
    reg_df["canonical_smiles"] = reg_df["canonical_smiles"].where(
        reg_df["canonical_smiles"].notna(),
        reg_df["SMILES"].astype(str),
    )
    reg_df["_condition_key"] = _condition_key_series(reg_df["temperature"], reg_df["phi"])

    reg_group_cols = ["polymer_id", "Polymer", "SMILES", "canonical_smiles", "_condition_key"]
    reg_poly = (
        reg_df.groupby(reg_group_cols, as_index=False)
        .agg(
            temperature=("temperature", "mean"),
            phi=("phi", "mean"),
            chi_pred_target=("chi_pred", "mean"),
        )
    )
    reg_std = (
        reg_df.groupby(reg_group_cols, as_index=False)["chi_pred"]
        .std(ddof=0)
        .rename(columns={"chi_pred": "chi_pred_std_target"})
    )
    reg_std["chi_pred_std_target"] = reg_std["chi_pred_std_target"].fillna(0.0).astype(float)
    reg_poly = reg_poly.merge(reg_std, on=reg_group_cols, how="left")
    if CLASS_LABEL_INTERNAL in reg_df.columns:
        reg_class = (
            reg_df.groupby(reg_group_cols, as_index=False)[CLASS_LABEL_INTERNAL]
            .max()
        )
        reg_poly = reg_poly.merge(reg_class, on=reg_group_cols, how="left")

    cls_all = step4_cls_metrics_dir / "class_predictions_all.csv"
    if cls_all.exists():
        cls_df = pd.read_csv(cls_all)
        cls_source = cls_all
    else:
        split_paths = [
            step4_cls_metrics_dir / "class_predictions_train.csv",
            step4_cls_metrics_dir / "class_predictions_val.csv",
            step4_cls_metrics_dir / "class_predictions_test.csv",
        ]
        available = [p for p in split_paths if p.exists()]
        if not available:
            raise FileNotFoundError(
                "Known candidate source requires Step4 classification predictions. "
                f"Expected: {cls_all} or split files under {step4_cls_metrics_dir}"
            )
        cls_df = pd.concat([pd.read_csv(p) for p in available], ignore_index=True)
        cls_source = available[0]

    cls_df = _ensure_internal_label(cls_df)

    required_cls = {"polymer_id", "class_prob"}
    missing_cls = sorted(required_cls - set(cls_df.columns))
    if missing_cls:
        raise ValueError(f"Classification known-candidate file missing columns: {missing_cls}")

    by_poly = cls_df.groupby("polymer_id", as_index=False)
    cls_poly = by_poly["class_prob"].mean().rename(columns={"class_prob": "class_prob"})
    cls_poly["class_prob_std"] = (
        by_poly["class_prob"].std(ddof=0)["class_prob"].fillna(0.0).astype(float)
    )
    if "class_logit" in cls_df.columns:
        cls_poly["class_logit"] = by_poly["class_logit"].mean()["class_logit"]
        cls_poly["class_logit_std"] = (
            by_poly["class_logit"].std(ddof=0)["class_logit"].fillna(0.0).astype(float)
        )
    if CLASS_LABEL_INTERNAL in cls_df.columns:
        cls_poly[CLASS_LABEL_INTERNAL] = by_poly[CLASS_LABEL_INTERNAL].max()[CLASS_LABEL_INTERNAL]

    target_join_cols = [
        c for c in target_info.columns
        if c not in {"temperature", "phi"}
    ]
    out = reg_poly.merge(target_info[target_join_cols], on="_condition_key", how="inner")
    if out.empty:
        summary = {
            "known_regression_predictions_csv": str(reg_source),
            "known_class_predictions_csv": str(cls_source),
            "known_candidate_count": 0,
        }
        return _empty_candidate_pool(), summary

    out = out.merge(cls_poly, on="polymer_id", how="left", suffixes=("", "_cls"))
    if out["class_prob"].isna().any():
        missing = int(out["class_prob"].isna().sum())
        raise ValueError(
            "Failed to attach class probabilities for known candidates: "
            f"{missing} polymers missing class_prob after merge."
        )

    out = out.copy()
    out["candidate_source"] = "known_step4"
    out["is_novel_vs_train"] = (~out["canonical_smiles"].isin(training_canonical)).astype(int)
    if CLASS_LABEL_INTERNAL not in out.columns and f"{CLASS_LABEL_INTERNAL}_cls" in out.columns:
        out[CLASS_LABEL_INTERNAL] = out[f"{CLASS_LABEL_INTERNAL}_cls"]
    if CLASS_LABEL_INTERNAL not in out.columns:
        out[CLASS_LABEL_INTERNAL] = out.get(CLASS_LABEL_PUBLIC, -1)
    if CLASS_LABEL_INTERNAL not in out.columns:
        out[CLASS_LABEL_INTERNAL] = -1
    out = _ensure_internal_label(out)
    if "class_logit" not in out.columns:
        out["class_logit"] = np.nan
    if "class_prob_std" not in out.columns:
        out["class_prob_std"] = 0.0
    if "class_logit_std" not in out.columns:
        out["class_logit_std"] = 0.0
    out["class_prob_std"] = out["class_prob_std"].fillna(0.0).astype(float)
    out["class_logit_std"] = out["class_logit_std"].fillna(0.0).astype(float)
    if "chi_pred_std_target" not in out.columns:
        out["chi_pred_std_target"] = 0.0
    out["chi_pred_std_target"] = pd.to_numeric(out["chi_pred_std_target"], errors="coerce").fillna(0.0).astype(float)
    out = out.drop(columns=[c for c in ["_condition_key", f"{CLASS_LABEL_INTERNAL}_cls"] if c in out.columns])
    out = out[_candidate_pool_columns() + [c for c in out.columns if c not in _candidate_pool_columns()]]

    summary = {
        "known_regression_predictions_csv": str(reg_source),
        "known_class_predictions_csv": str(cls_source),
        "known_candidate_count": int(_count_candidate_polymers(out)),
    }
    return out, summary


def resolve_training_smiles(results_dir: Path, base_results_dir: Path) -> set[str]:
    train_path = results_dir / "train_unlabeled.csv"
    if not train_path.exists():
        train_path = base_results_dir / "train_unlabeled.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training set not found for novelty reference: {train_path}")

    cache_key = (str(train_path.resolve()), str(train_path.stat().st_mtime_ns))
    cached = _TRAINING_SMILES_CACHE.get(cache_key)
    if cached is not None:
        return set(cached)

    cache_txt = train_path.with_name(f"{train_path.stem}_canonical_smiles_cache.txt.gz")
    cache_meta = cache_txt.with_suffix(".json")
    expected_meta = {
        "source_path": str(train_path.resolve()),
        "source_size": int(train_path.stat().st_size),
        "source_mtime_ns": int(train_path.stat().st_mtime_ns),
    }
    if cache_txt.exists() and cache_meta.exists():
        try:
            with open(cache_meta, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            if (
                str(meta.get("source_path", "")) == expected_meta["source_path"]
                and int(meta.get("source_size", -1)) == expected_meta["source_size"]
                and int(meta.get("source_mtime_ns", -1)) == expected_meta["source_mtime_ns"]
            ):
                with gzip.open(cache_txt, "rt", encoding="utf-8") as handle:
                    canonical_set = {line.strip() for line in handle if line.strip()}
                _TRAINING_SMILES_CACHE[cache_key] = canonical_set
                return set(canonical_set)
        except Exception:
            pass

    header_df = pd.read_csv(train_path, nrows=0)
    if "p_smiles" in header_df.columns:
        smiles_col = "p_smiles"
    elif "smiles" in header_df.columns:
        smiles_col = "smiles"
    else:
        raise ValueError(f"Training CSV missing smiles column: {train_path}")

    verbose_progress = str(os.environ.get("STEP62_EVAL_LOAD_PROGRESS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    chunk_size = int(str(os.environ.get("STEP62_TRAINING_SMILES_CHUNKSIZE", "50000")).strip() or "50000")
    canonical_set: set[str] = set()
    rows_seen = 0
    for chunk_index, chunk_df in enumerate(
        pd.read_csv(train_path, usecols=[smiles_col], chunksize=chunk_size),
        start=1,
    ):
        canonical = chunk_df[smiles_col].astype(str).map(canonicalize_smiles)
        canonical = canonical[canonical.notna()]
        canonical_set.update(canonical.tolist())
        rows_seen += int(len(chunk_df))
        if verbose_progress and (chunk_index == 1 or chunk_index % 10 == 0):
            print(
                "[step5_evaluator_load] "
                f"stage=novelty_cache_build rows_seen={rows_seen} unique_canonical={len(canonical_set)}",
                flush=True,
            )
    try:
        with gzip.open(cache_txt, "wt", encoding="utf-8") as handle:
            for smi in sorted(canonical_set):
                handle.write(f"{smi}\n")
        with open(cache_meta, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    **expected_meta,
                    "n_canonical_smiles": int(len(canonical_set)),
                },
                handle,
                indent=2,
            )
    except Exception:
        pass
    _TRAINING_SMILES_CACHE[cache_key] = canonical_set
    return set(canonical_set)


def prepare_novel_candidates(
    generated_csv: Path,
    smiles_column: str,
    training_canonical: set[str],
    require_two_stars: bool = True,
    max_novel_candidates: int | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if not generated_csv.exists():
        raise FileNotFoundError(f"Generated samples CSV not found: {generated_csv}")

    raw_df = pd.read_csv(generated_csv)
    if smiles_column not in raw_df.columns:
        fallback_cols = ["smiles", "p_smiles", "SMILES"]
        found = next((c for c in fallback_cols if c in raw_df.columns), None)
        if found is None:
            raise ValueError(f"Generated CSV missing smiles column '{smiles_column}'")
        smiles_column = found

    input_smiles = raw_df[smiles_column].astype(str).tolist()

    rows = []
    n_valid = 0
    n_valid_two_star = 0
    n_valid_two_terminal_star = 0
    for smi in input_smiles:
        if not check_validity(smi):
            continue
        n_valid += 1
        if require_two_stars and count_stars(smi) != 2:
            continue
        n_valid_two_star += 1
        if require_two_stars and not has_terminal_connection_stars(smi, expected_stars=2):
            continue
        if require_two_stars:
            n_valid_two_terminal_star += 1
        canon = canonicalize_smiles(smi)
        if canon is None:
            continue
        rows.append({"SMILES": smi, "canonical_smiles": canon})

    cand_df = pd.DataFrame(rows)
    n_valid_unique = int(cand_df["canonical_smiles"].nunique()) if not cand_df.empty else 0

    if cand_df.empty:
        summary = {
            "n_generated_input": int(len(input_smiles)),
            "n_valid": int(n_valid),
            "n_valid_two_stars": int(n_valid_two_star),
            "n_valid_two_terminal_stars": int(n_valid_two_terminal_star),
            "n_valid_unique": int(n_valid_unique),
            "n_novel_unique": 0,
            "novel_fraction_among_valid_unique": 0.0,
        }
        return cand_df, summary

    cand_df = cand_df.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
    novel_mask = ~cand_df["canonical_smiles"].isin(training_canonical)
    cand_df = cand_df[novel_mask].reset_index(drop=True)

    if max_novel_candidates is not None and max_novel_candidates > 0:
        cand_df = cand_df.head(int(max_novel_candidates)).copy()

    cand_df = cand_df.reset_index(drop=True)
    cand_df["polymer_id"] = np.arange(len(cand_df), dtype=int)
    cand_df["Polymer"] = [f"Novel_{i + 1:06d}" for i in range(len(cand_df))]

    n_novel_unique = int(len(cand_df))
    summary = {
        "n_generated_input": int(len(input_smiles)),
        "n_valid": int(n_valid),
        "n_valid_two_stars": int(n_valid_two_star),
        "n_valid_two_terminal_stars": int(n_valid_two_terminal_star),
        "n_valid_unique": int(n_valid_unique),
        "n_novel_unique": int(n_novel_unique),
        "novel_fraction_among_valid_unique": float(n_novel_unique / n_valid_unique) if n_valid_unique > 0 else 0.0,
    }
    return cand_df, summary


def launch_fresh_step2_resampling(
    args,
    split_mode: str,
    resampling_step_dir: Path,
    target_polymer_count: int | None = None,
    random_seed: int | None = None,
) -> Tuple[Path, Dict[str, object]]:
    repo_root = Path(__file__).resolve().parents[2]
    step2_script = repo_root / "scripts" / "step2_sample_and_evaluate.py"
    if not step2_script.exists():
        raise FileNotFoundError(f"Step 2 sampling script not found: {step2_script}")

    cmd = [
        sys.executable,
        str(step2_script),
        "--config",
        str(args.config),
        "--model_size",
        str(args.model_size),
        "--split_mode",
        str(split_mode),
        "--output_step_dir",
        str(resampling_step_dir),
    ]
    if target_polymer_count is not None:
        cmd.extend(["--target_polymer_count", str(int(target_polymer_count))])
    if getattr(args, "backbone_checkpoint", None):
        cmd.extend(["--checkpoint", str(args.backbone_checkpoint)])
    if random_seed is not None:
        cmd.extend(["--random_seed", str(int(random_seed))])
    if getattr(args, "decode_constraint_class", None):
        cmd.extend(["--decode_constraint_class", str(args.decode_constraint_class)])
    if getattr(args, "decode_constraint_motif_bank_json", None):
        cmd.extend(["--decode_constraint_motif_bank_json", str(args.decode_constraint_motif_bank_json)])
    if getattr(args, "decode_constraint_length_prior_json", None):
        cmd.extend(["--decode_constraint_length_prior_json", str(args.decode_constraint_length_prior_json)])
    if getattr(args, "decode_constraint_spans_per_sample", None) is not None:
        cmd.extend(["--decode_constraint_spans_per_sample", str(int(args.decode_constraint_spans_per_sample))])
    if getattr(args, "decode_constraint_center_min_frac", None) is not None:
        cmd.extend(["--decode_constraint_center_min_frac", str(float(args.decode_constraint_center_min_frac))])
    if getattr(args, "decode_constraint_center_max_frac", None) is not None:
        cmd.extend(["--decode_constraint_center_max_frac", str(float(args.decode_constraint_center_max_frac))])
    if bool(getattr(args, "decode_constraint_enforce_class_match", False)):
        cmd.append("--decode_constraint_enforce_class_match")
    if bool(getattr(args, "decode_constraint_enforce_backbone_class_match", False)):
        cmd.append("--decode_constraint_enforce_backbone_class_match")
    if getattr(args, "decode_constraint_class_token_bias_json", None):
        cmd.extend(["--decode_constraint_class_token_bias_json", str(args.decode_constraint_class_token_bias_json)])
    if bool(getattr(args, "resampling_skip_novelty_filter", False)):
        cmd.append("--valid_only_skip_novelty_filter")
    if bool(getattr(args, "resampling_skip_sa_filter", False)):
        cmd.append("--valid_only_skip_sa_filter")

    print(f"Launching fresh Step 2 resampling into: {resampling_step_dir}")
    try:
        subprocess.run(cmd, cwd=str(repo_root), check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Fresh Step 2 resampling failed during Step 5/6 candidate generation. "
            f"Command: {' '.join(cmd)}"
        ) from exc

    metrics_dir = resampling_step_dir / "metrics"
    generated_csv = metrics_dir / "generated_samples.csv"
    if not generated_csv.exists():
        raise FileNotFoundError(
            "Fresh Step 2 resampling completed but generated_samples.csv is missing. "
            f"Expected: {generated_csv}"
        )

    summary = {
        "step2_resampling_step_dir": str(resampling_step_dir),
        "step2_resampling_metrics_dir": str(metrics_dir),
        "step2_resampling_generated_csv": str(generated_csv),
        "step2_resampling_target_csv": str(metrics_dir / "target_polymers.csv"),
        "step2_resampling_summary_csv": str(metrics_dir / "step_summary.csv"),
        "step2_resampling_target_polymer_count": None if target_polymer_count is None else int(target_polymer_count),
        "step2_resampling_random_seed": None if random_seed is None else int(random_seed),
    }
    return generated_csv, summary


def _resolve_novel_inference_checkpoints(
    args,
    results_dir: Path,
    step4_reg_metrics_dir: Path,
    step4_cls_metrics_dir: Path,
) -> Tuple[Path, Path]:
    default_reg_candidates = [
        step4_reg_metrics_dir.parent / "checkpoints" / "chi_regression_best.pt",
        step4_reg_metrics_dir.parent / "checkpoints" / "chi_physics_best.pt",
        results_dir / "checkpoints" / "chi_regression_best.pt",
        results_dir / "checkpoints" / "chi_physics_best.pt",
    ]
    if getattr(args, "step4_checkpoint", None):
        chi_checkpoint = Path(args.step4_checkpoint)
    else:
        chi_checkpoint = next((p for p in default_reg_candidates if p.exists()), default_reg_candidates[0])
    if not chi_checkpoint.exists():
        raise FileNotFoundError(f"Step4 chi checkpoint not found: {chi_checkpoint}")

    if getattr(args, "step4_class_checkpoint", None):
        class_checkpoint = Path(args.step4_class_checkpoint)
    else:
        default_cls_candidates = [
            step4_cls_metrics_dir.parent / "checkpoints" / "chi_classifier_best.pt",
            results_dir / "checkpoints" / "chi_classifier_best.pt",
        ]
        class_checkpoint = next((p for p in default_cls_candidates if p.exists()), default_cls_candidates[0])
    if not class_checkpoint.exists():
        raise FileNotFoundError(
            "Step4 classification checkpoint not found. "
            f"Expected: {class_checkpoint}. "
            "Run Step 4 to produce Step4_2 checkpoint or pass --step4_class_checkpoint."
        )
    return chi_checkpoint, class_checkpoint


def prepare_novel_inference_cache(
    args,
    config: Dict,
    chi_cfg: Dict,
    results_dir: Path,
    step4_reg_metrics_dir: Path,
    step4_cls_metrics_dir: Path,
    device: str,
    split_mode: str,
) -> Dict[str, object]:
    load_start = time.perf_counter()
    verbose_progress = str(os.environ.get("STEP62_EVAL_LOAD_PROGRESS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }

    def _log_progress(stage: str) -> None:
        if verbose_progress:
            elapsed = time.perf_counter() - load_start
            print(f"[step5_evaluator_load] stage={stage} elapsed_s={elapsed:.2f}", flush=True)

    uncertainty_enabled = bool(getattr(args, "uncertainty_enabled", chi_cfg.get("uncertainty_enabled", False)))
    uncertainty_mc_samples = int(
        getattr(args, "uncertainty_mc_samples", chi_cfg.get("uncertainty_mc_samples", 20))
        or chi_cfg.get("uncertainty_mc_samples", 20)
    )
    timestep = int(chi_cfg.get("embedding_timestep", 1))
    pooling = str(args.embedding_pooling)
    mc_enabled = bool(uncertainty_enabled and int(uncertainty_mc_samples) >= 2)
    chi_checkpoint, class_checkpoint = _resolve_novel_inference_checkpoints(
        args=args,
        results_dir=results_dir,
        step4_reg_metrics_dir=step4_reg_metrics_dir,
        step4_cls_metrics_dir=step4_cls_metrics_dir,
    )

    tokenizer_path, _ = resolve_step1_artifacts(
        config=config,
        model_size=args.model_size,
        split_mode=split_mode,
        checkpoint_path=args.backbone_checkpoint,
    )
    tokenizer = PSmilesTokenizer.load(tokenizer_path)
    step1_backbone = None
    _log_progress("tokenizer_loaded")

    reg_ckpt = torch.load(chi_checkpoint, map_location=device, weights_only=True)
    reg_state = _normalize_checkpoint_state_dict(reg_ckpt["model_state_dict"])
    reg_finetune_last_layers = int(reg_ckpt.get("finetune_last_layers", 0) or 0)
    reg_timestep = int(reg_ckpt.get("timestep_for_embedding", timestep))
    regression_mode = str(reg_ckpt.get("regression_mode", "physics_guided")).strip().lower()
    if reg_finetune_last_layers > 0:
        reg_backbone = build_backbone_architecture(
            config=config,
            tokenizer=tokenizer,
            model_size=args.model_size,
        )
        if regression_mode == "direct_chi":
            reg_head = DirectChiRegressor(
                embedding_dim=int(reg_ckpt["embedding_dim"]),
                hidden_sizes=list(reg_ckpt["hidden_sizes"]),
                dropout=float(reg_ckpt["dropout"]),
            )
            reg_model = BackboneDirectChiModel(
                backbone=reg_backbone,
                chi_head=reg_head,
                timestep=reg_timestep,
                pooling=pooling,
            ).to(device)
        else:
            reg_head = PhysicsGuidedChiModel(
                embedding_dim=int(reg_ckpt["embedding_dim"]),
                hidden_sizes=list(reg_ckpt["hidden_sizes"]),
                dropout=float(reg_ckpt["dropout"]),
            )
            reg_model = BackbonePhysicsGuidedChiModel(
                backbone=reg_backbone,
                chi_head=reg_head,
                timestep=reg_timestep,
                pooling=pooling,
            ).to(device)
        reg_model.load_state_dict(reg_state, strict=True)
    else:
        if regression_mode == "direct_chi":
            reg_model = DirectChiRegressor(
                embedding_dim=int(reg_ckpt["embedding_dim"]),
                hidden_sizes=list(reg_ckpt["hidden_sizes"]),
                dropout=float(reg_ckpt["dropout"]),
            ).to(device)
        else:
            reg_model = PhysicsGuidedChiModel(
                embedding_dim=int(reg_ckpt["embedding_dim"]),
                hidden_sizes=list(reg_ckpt["hidden_sizes"]),
                dropout=float(reg_ckpt["dropout"]),
            ).to(device)
        if isinstance(reg_state, dict) and any(str(k).startswith("chi_head.") for k in reg_state.keys()):
            reg_state = {
                str(k)[len("chi_head."):]: v
                for k, v in reg_state.items()
                if str(k).startswith("chi_head.")
            }
        reg_model.load_state_dict(reg_state, strict=True)
    if mc_enabled:
        reg_model.train()
    else:
        reg_model.eval()
    _log_progress("regression_model_loaded")

    cls_model = None
    cls_finetune_last_layers = 0
    cls_timestep = int(timestep)
    cls_ckpt = torch.load(class_checkpoint, map_location=device, weights_only=True)
    cls_state = _normalize_checkpoint_state_dict(cls_ckpt["model_state_dict"])
    cls_finetune_last_layers = int(cls_ckpt.get("finetune_last_layers", 0) or 0)
    cls_timestep = int(cls_ckpt.get("timestep_for_embedding", timestep))
    if cls_finetune_last_layers > 0:
        cls_backbone = build_backbone_architecture(
            config=config,
            tokenizer=tokenizer,
            model_size=args.model_size,
        )
        cls_head = SolubilityClassifier(
            embedding_dim=int(cls_ckpt["embedding_dim"]),
            hidden_sizes=list(cls_ckpt["hidden_sizes"]),
            dropout=float(cls_ckpt["dropout"]),
        )
        cls_model = BackboneSolubilityClassifierModel(
            backbone=cls_backbone,
            classifier_head=cls_head,
            timestep=cls_timestep,
            pooling=pooling,
        ).to(device)
        cls_model.load_state_dict(cls_state, strict=True)
    else:
        cls_model = SolubilityClassifier(
            embedding_dim=int(cls_ckpt["embedding_dim"]),
            hidden_sizes=list(cls_ckpt["hidden_sizes"]),
            dropout=float(cls_ckpt["dropout"]),
        ).to(device)
        if isinstance(cls_state, dict) and any(str(k).startswith("classifier_head.") for k in cls_state.keys()):
            cls_state = {
                str(k)[len("classifier_head."):]: v
                for k, v in cls_state.items()
                if str(k).startswith("classifier_head.")
            }
        cls_model.load_state_dict(cls_state, strict=True)
    if mc_enabled:
        cls_model.train()
    else:
        cls_model.eval()
    _log_progress("classification_model_loaded")

    if reg_finetune_last_layers > 0:
        warnings.warn(
            (
                f"Loaded Step4 regression checkpoint with finetuned backbone "
                f"(finetune_last_layers={reg_finetune_last_layers}) for novel-candidate inference."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    if cls_finetune_last_layers > 0:
        warnings.warn(
            (
                f"Loaded Step4 classification checkpoint with finetuned backbone "
                f"(finetune_last_layers={cls_finetune_last_layers}) for novel-candidate inference."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    reg_needs_step1_embeddings = reg_finetune_last_layers == 0
    cls_needs_step1_embeddings = cls_finetune_last_layers == 0
    if reg_needs_step1_embeddings or cls_needs_step1_embeddings:
        step1_backbone = load_backbone_only_from_step1(
            config=config,
            tokenizer=tokenizer,
            model_size=args.model_size,
            split_mode=split_mode,
            checkpoint_path=args.backbone_checkpoint,
            device=device,
        )
        _log_progress("step1_backbone_loaded")

    _log_progress("prepare_novel_inference_cache_done")

    return {
        "tokenizer": tokenizer,
        "step1_backbone": step1_backbone,
        "reg_model": reg_model,
        "cls_model": cls_model,
        "reg_timestep": int(reg_timestep),
        "cls_timestep": int(cls_timestep),
        "reg_finetune_last_layers": int(reg_finetune_last_layers),
        "cls_finetune_last_layers": int(cls_finetune_last_layers),
        "regression_mode": regression_mode,
        "reg_needs_step1_embeddings": bool(reg_needs_step1_embeddings),
        "cls_needs_step1_embeddings": bool(cls_needs_step1_embeddings),
        "mc_enabled": bool(mc_enabled),
        "device": device,
        "chi_checkpoint_path": str(chi_checkpoint),
        "class_checkpoint_path": str(class_checkpoint),
    }


@torch.no_grad()
def infer_coefficients_for_novel_candidates(
    novel_df: pd.DataFrame,
    target_df: pd.DataFrame,
    config: Dict,
    model_size: str | None,
    split_mode: str | None,
    chi_checkpoint_path: Path,
    class_checkpoint_path: Path | None,
    backbone_checkpoint_path: str | None,
    device: str,
    timestep: int,
    pooling: str,
    batch_size: int,
    uncertainty_enabled: bool = False,
    uncertainty_mc_samples: int = 20,
    uncertainty_seed: int | None = None,
    inference_cache: Dict[str, object] | None = None,
) -> pd.DataFrame:
    if novel_df.empty:
        return _empty_candidate_pool()

    target_info = _prepare_target_conditions(target_df)

    if uncertainty_seed is not None:
        torch.manual_seed(int(uncertainty_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(uncertainty_seed))

    mc_enabled = bool(uncertainty_enabled and int(uncertainty_mc_samples) >= 2)
    if inference_cache is None:
        class _InlineInferenceArgs:
            pass

        inline_args = _InlineInferenceArgs()
        inline_args.model_size = model_size
        inline_args.step4_checkpoint = str(chi_checkpoint_path)
        inline_args.step4_class_checkpoint = (
            str(class_checkpoint_path) if class_checkpoint_path is not None else None
        )
        inline_args.backbone_checkpoint = backbone_checkpoint_path
        inline_args.embedding_pooling = pooling
        inline_args.uncertainty_enabled = bool(uncertainty_enabled)
        inline_args.uncertainty_mc_samples = int(uncertainty_mc_samples)
        inference_cache = prepare_novel_inference_cache(
            args=inline_args,
            config=config,
            chi_cfg={
                "embedding_timestep": int(timestep),
                "uncertainty_enabled": bool(uncertainty_enabled),
                "uncertainty_mc_samples": int(uncertainty_mc_samples),
            },
            results_dir=chi_checkpoint_path.parent.parent.parent,
            step4_reg_metrics_dir=chi_checkpoint_path.parent.parent / "metrics",
            step4_cls_metrics_dir=(
                class_checkpoint_path.parent.parent / "metrics"
                if class_checkpoint_path is not None
                else chi_checkpoint_path.parent.parent / "metrics"
            ),
            device=device,
            split_mode=str(split_mode) if split_mode is not None else "polymer",
        )

    tokenizer = inference_cache["tokenizer"]
    step1_backbone = inference_cache.get("step1_backbone")
    reg_model = inference_cache["reg_model"]
    cls_model = inference_cache.get("cls_model")
    reg_timestep = int(inference_cache["reg_timestep"])
    cls_timestep = int(inference_cache["cls_timestep"])
    reg_finetune_last_layers = int(inference_cache["reg_finetune_last_layers"])
    cls_finetune_last_layers = int(inference_cache["cls_finetune_last_layers"])
    reg_needs_step1_embeddings = bool(inference_cache["reg_needs_step1_embeddings"])
    cls_needs_step1_embeddings = bool(inference_cache["cls_needs_step1_embeddings"])
    mc_enabled = bool(inference_cache.get("mc_enabled", mc_enabled))
    result_frames: List[pd.DataFrame] = []
    smiles_list = novel_df["SMILES"].astype(str).tolist()
    for i in range(0, len(smiles_list), batch_size):
        batch_df = novel_df.iloc[i : i + batch_size].copy().reset_index(drop=True)
        batch_smiles = batch_df["SMILES"].astype(str).tolist()
        encoded = tokenizer.batch_encode(batch_smiles)
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=device)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long, device=device)

        emb_reg = None
        emb_cls = None
        if reg_needs_step1_embeddings:
            t_reg = torch.full((input_ids.shape[0],), reg_timestep, device=device, dtype=torch.long)
            emb_reg = step1_backbone.get_pooled_output(
                input_ids=input_ids,
                timesteps=t_reg,
                attention_mask=attention_mask,
                pooling=pooling,
            )
        if cls_needs_step1_embeddings:
            if reg_needs_step1_embeddings and cls_timestep == reg_timestep:
                emb_cls = emb_reg
            else:
                t_cls = torch.full((input_ids.shape[0],), cls_timestep, device=device, dtype=torch.long)
                emb_cls = step1_backbone.get_pooled_output(
                    input_ids=input_ids,
                    timesteps=t_cls,
                    attention_mask=attention_mask,
                    pooling=pooling,
                )

        def _forward_reg_once(temperature_tensor: torch.Tensor, phi_tensor: torch.Tensor) -> np.ndarray:
            if reg_finetune_last_layers > 0:
                reg_out_local = reg_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    temperature=temperature_tensor,
                    phi=phi_tensor,
                )
            else:
                reg_out_local = reg_model(
                    embedding=emb_reg,
                    temperature=temperature_tensor,
                    phi=phi_tensor,
                )
            return reg_out_local["chi_pred"].detach().cpu().numpy()

        def _forward_cls_once() -> np.ndarray:
            if cls_model is None:
                return np.zeros((input_ids.shape[0],), dtype=float)
            if cls_finetune_last_layers > 0:
                cls_out_local = cls_model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                cls_out_local = cls_model(embedding=emb_cls)
            return cls_out_local["class_logit"].detach().cpu().numpy()

        if mc_enabled:
            logit_samples: List[np.ndarray] = []
            prob_samples: List[np.ndarray] = []
            for _ in range(int(uncertainty_mc_samples)):
                logit_s = _forward_cls_once()
                logit_samples.append(logit_s)
                prob_samples.append(stable_sigmoid(logit_s))
            logit_stack = np.stack(logit_samples, axis=0)
            prob_stack = np.stack(prob_samples, axis=0)
            logit_mean = np.mean(logit_stack, axis=0)
            logit_std = np.std(logit_stack, axis=0, ddof=0)
            prob_mean = np.mean(prob_stack, axis=0)
            prob_std = np.std(prob_stack, axis=0, ddof=0)
        else:
            logit_det = _forward_cls_once()
            logit_mean = logit_det
            logit_std = np.zeros_like(logit_det)
            prob_mean = stable_sigmoid(logit_det)
            prob_std = np.zeros_like(prob_mean)

        for _, target_row in target_info.iterrows():
            temperature_tensor = torch.full(
                (input_ids.shape[0],),
                float(target_row["temperature"]),
                dtype=torch.float32,
                device=device,
            )
            phi_tensor = torch.full(
                (input_ids.shape[0],),
                float(target_row["phi"]),
                dtype=torch.float32,
                device=device,
            )

            if mc_enabled:
                chi_samples: List[np.ndarray] = []
                for _ in range(int(uncertainty_mc_samples)):
                    chi_samples.append(_forward_reg_once(temperature_tensor, phi_tensor))
                chi_stack = np.stack(chi_samples, axis=0)
                chi_mean = np.mean(chi_stack, axis=0)
                chi_std = np.std(chi_stack, axis=0, ddof=0)
            else:
                chi_mean = _forward_reg_once(temperature_tensor, phi_tensor)
                chi_std = np.zeros_like(chi_mean)

            result = batch_df[["polymer_id", "Polymer", "SMILES", "canonical_smiles"]].copy()
            for col in target_info.columns:
                if col == "_condition_key":
                    continue
                result[col] = target_row[col]
            result[CLASS_LABEL_INTERNAL] = -1
            result[CLASS_LABEL_PUBLIC] = result[CLASS_LABEL_INTERNAL]
            result["class_logit"] = logit_mean
            result["class_logit_std"] = logit_std
            result["class_prob"] = prob_mean
            result["class_prob_std"] = prob_std
            result["chi_pred_target"] = chi_mean
            result["chi_pred_std_target"] = chi_std
            result["candidate_source"] = "novel_generated"
            result["is_novel_vs_train"] = 1
            result_frames.append(result)

    if not result_frames:
        return _empty_candidate_pool()

    result_df = pd.concat(result_frames, ignore_index=True)
    ordered_cols = _candidate_pool_columns() + [c for c in result_df.columns if c not in _candidate_pool_columns()]
    return result_df[ordered_cols]


def load_soluble_targets(
    targets_csv: str | None,
    results_dir: Path,
    base_results_dir: Path | None,
    split_mode: str,
    target_temperature: float | None = None,
    target_phi: float | None = None,
) -> Tuple[pd.DataFrame, str | None]:
    if targets_csv:
        target_path = Path(targets_csv)
    else:
        candidate_paths = [
            results_dir / "step3_chi_target_learning" / split_mode / "metrics" / "chi_target_for_inverse_design.csv"
        ]
        if base_results_dir is not None:
            base_path = Path(base_results_dir) / "step3_chi_target_learning" / split_mode / "metrics" / "chi_target_for_inverse_design.csv"
            if str(base_path) not in {str(p) for p in candidate_paths}:
                candidate_paths.append(base_path)
        target_path = next((p for p in candidate_paths if p.exists()), candidate_paths[0])
    if not target_path.exists():
        raise FileNotFoundError(
            "Learned χ_target file not found. "
            "Run Step 3 first or provide --targets_csv. "
            f"Expected: {target_path}"
        )
    target_df = pd.read_csv(target_path)

    if "target_class" in target_df.columns:
        target_df = target_df[target_df["target_class"].astype(int) == 1].copy()
    else:
        target_df["target_class"] = 1

    required_cols = {"temperature", "phi", "target_chi"}
    missing = required_cols - set(target_df.columns)
    if missing:
        raise ValueError(f"targets CSV missing required columns: {sorted(missing)}")

    if "target_class_name" not in target_df.columns:
        target_df["target_class_name"] = CLASS_NAME_MAP[1]
    if "property_rule" not in target_df.columns:
        target_df["property_rule"] = "upper_bound"
    if "target_id" not in target_df.columns:
        target_df.insert(0, "target_id", np.arange(1, len(target_df) + 1))

    target_df = target_df.copy()
    target_df["temperature"] = target_df["temperature"].astype(float)
    target_df["phi"] = target_df["phi"].astype(float)

    if target_temperature is not None:
        t_value = float(target_temperature)
        t_mask = np.isclose(target_df["temperature"].to_numpy(dtype=float), t_value, atol=1.0e-8)
        if not np.any(t_mask):
            available_t = sorted(target_df["temperature"].astype(float).unique().tolist())
            raise ValueError(
                f"No targets found for target_temperature={t_value}. Available temperatures: {available_t}"
            )
        target_df = target_df.loc[t_mask].copy()

    if target_phi is not None:
        phi_value = float(target_phi)
        phi_mask = np.isclose(target_df["phi"].to_numpy(dtype=float), phi_value, atol=1.0e-8)
        if not np.any(phi_mask):
            available_phi = sorted(target_df["phi"].astype(float).unique().tolist())
            raise ValueError(
                f"No targets found for target_phi={phi_value}. Available phi values: {available_phi}"
            )
        target_df = target_df.loc[phi_mask].copy()

    target_df = target_df.sort_values(["temperature", "phi"]).reset_index(drop=True)
    if target_df.empty:
        raise ValueError("No water-miscible targets found in χ_target file (class=1).")
    target_df["target_id"] = np.arange(1, len(target_df) + 1)
    return target_df, str(target_path)


def build_candidate_pool(
    args,
    config: Dict,
    chi_cfg: Dict,
    target_df: pd.DataFrame,
    results_dir: Path,
    base_results_dir: Path,
    step4_reg_metrics_dir: Path,
    step4_cls_metrics_dir: Path,
    device: str,
    split_mode: str,
    resampling_step_dir: Path | None = None,
    resampling_target_polymer_count: int | None = None,
    resampling_random_seed: int | None = None,
    training_canonical: set[str] | None = None,
    novel_inference_cache: Dict[str, object] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, object], set[str]]:
    source = parse_candidate_source(args.candidate_source)
    if training_canonical is None:
        training_canonical = resolve_training_smiles(results_dir, base_results_dir)
    uncertainty_enabled = bool(getattr(args, "uncertainty_enabled", chi_cfg.get("uncertainty_enabled", False)))
    uncertainty_mc_samples = int(
        getattr(args, "uncertainty_mc_samples", chi_cfg.get("uncertainty_mc_samples", 20))
        or chi_cfg.get("uncertainty_mc_samples", 20)
    )
    uncertainty_seed = getattr(args, "uncertainty_seed", chi_cfg.get("uncertainty_seed", None))

    summary: Dict[str, object] = {
        "candidate_source": source,
        "step4_regression_metrics_dir": str(step4_reg_metrics_dir),
        "step4_classification_metrics_dir": str(step4_cls_metrics_dir),
        "uncertainty_enabled": bool(uncertainty_enabled),
        "uncertainty_mc_samples": int(uncertainty_mc_samples),
        "uncertainty_seed": None if uncertainty_seed is None else int(uncertainty_seed),
    }

    pool_frames: List[pd.DataFrame] = []

    if source in {"known", "hybrid"}:
        known_df, known_summary = _load_known_candidates_from_step4_metrics(
            step4_reg_metrics_dir=step4_reg_metrics_dir,
            step4_cls_metrics_dir=step4_cls_metrics_dir,
            training_canonical=training_canonical,
            target_df=target_df,
        )
        summary.update(known_summary)
        pool_frames.append(known_df)

    if source in {"novel", "hybrid"}:
        if getattr(args, "generated_csv", None):
            raise ValueError(
                "--generated_csv is unsupported for Step 5/6. "
                "These steps must launch a fresh Step 2 sampling run from the Step 1 checkpoint."
            )
        if resampling_step_dir is None:
            raise ValueError("resampling_step_dir is required for fresh Step 2 candidate generation.")

        generated_csv, resampling_summary = launch_fresh_step2_resampling(
            args=args,
            split_mode=split_mode,
            resampling_step_dir=resampling_step_dir,
            target_polymer_count=resampling_target_polymer_count,
            random_seed=resampling_random_seed,
        )
        summary.update(resampling_summary)

        novel_df, novel_summary = prepare_novel_candidates(
            generated_csv=generated_csv,
            smiles_column=args.generated_smiles_column,
            training_canonical=training_canonical,
            require_two_stars=(not args.allow_non_two_stars),
            max_novel_candidates=args.max_novel_candidates,
        )
        summary.update(novel_summary)
        summary["generated_csv"] = str(generated_csv)
        chi_checkpoint, class_checkpoint = _resolve_novel_inference_checkpoints(
            args=args,
            results_dir=results_dir,
            step4_reg_metrics_dir=step4_reg_metrics_dir,
            step4_cls_metrics_dir=step4_cls_metrics_dir,
        )

        novel_coeff_df = infer_coefficients_for_novel_candidates(
            novel_df=novel_df,
            target_df=target_df,
            config=config,
            model_size=args.model_size,
            split_mode=getattr(args, "split_mode", None),
            chi_checkpoint_path=chi_checkpoint,
            class_checkpoint_path=class_checkpoint,
            backbone_checkpoint_path=args.backbone_checkpoint,
            device=device,
            timestep=int(chi_cfg.get("embedding_timestep", 1)),
            pooling=args.embedding_pooling,
            batch_size=int(args.embedding_batch_size or chi_cfg.get("embedding_batch_size", 128)),
            uncertainty_enabled=uncertainty_enabled,
            uncertainty_mc_samples=uncertainty_mc_samples,
            uncertainty_seed=uncertainty_seed,
            inference_cache=novel_inference_cache,
        )
        summary["novel_candidate_count"] = int(_count_candidate_polymers(novel_coeff_df))
        summary["novel_candidate_row_count"] = int(len(novel_coeff_df))
        pool_frames.append(novel_coeff_df)

    if not pool_frames:
        empty = _empty_candidate_pool()
        summary["candidate_count_total"] = 0
        summary["candidate_count_after_dedup"] = 0
        return empty, summary, training_canonical

    coeff_df = pd.concat(pool_frames, ignore_index=True)
    if "canonical_smiles" not in coeff_df.columns:
        coeff_df["canonical_smiles"] = coeff_df["SMILES"].astype(str).map(canonicalize_smiles)
    coeff_df["canonical_smiles"] = coeff_df["canonical_smiles"].where(
        coeff_df["canonical_smiles"].notna(),
        coeff_df["SMILES"].astype(str),
    )
    if "candidate_source" not in coeff_df.columns:
        coeff_df["candidate_source"] = "unknown"
    if "is_novel_vs_train" not in coeff_df.columns:
        coeff_df["is_novel_vs_train"] = (~coeff_df["canonical_smiles"].isin(training_canonical)).astype(int)
    if "class_prob_std" not in coeff_df.columns:
        coeff_df["class_prob_std"] = 0.0
    if "class_logit_std" not in coeff_df.columns:
        coeff_df["class_logit_std"] = 0.0
    if "chi_pred_std_target" not in coeff_df.columns:
        coeff_df["chi_pred_std_target"] = 0.0
    coeff_df["class_prob_std"] = coeff_df["class_prob_std"].fillna(0.0).astype(float)
    coeff_df["class_logit_std"] = coeff_df["class_logit_std"].fillna(0.0).astype(float)
    coeff_df["chi_pred_std_target"] = pd.to_numeric(coeff_df["chi_pred_std_target"], errors="coerce").fillna(0.0).astype(float)

    summary["candidate_row_count_total"] = int(len(coeff_df))
    summary["candidate_count_total"] = int(_count_candidate_polymers(coeff_df))
    dedup_subset = ["canonical_smiles", "target_id"] if "target_id" in coeff_df.columns else ["canonical_smiles"]
    coeff_df = coeff_df.drop_duplicates(subset=dedup_subset, keep="first").reset_index(drop=True)
    coeff_df["source_polymer_id"] = coeff_df.get("polymer_id", pd.Series([-1] * len(coeff_df))).astype(int)
    polymer_keys = pd.Index(pd.unique(coeff_df["canonical_smiles"].astype(str)))
    polymer_id_map = {key: idx for idx, key in enumerate(polymer_keys)}
    coeff_df["polymer_id"] = coeff_df["canonical_smiles"].astype(str).map(polymer_id_map).astype(int)
    coeff_df = coeff_df[_candidate_pool_columns() + [c for c in coeff_df.columns if c not in _candidate_pool_columns()]]
    summary["candidate_row_count_after_dedup"] = int(len(coeff_df))
    summary["candidate_count_after_dedup"] = int(_count_candidate_polymers(coeff_df))
    return coeff_df, summary, training_canonical
