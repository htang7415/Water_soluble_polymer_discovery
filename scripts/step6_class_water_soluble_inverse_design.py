#!/usr/bin/env python
"""Step 6: polymer-family class + water-soluble inverse design on chi(T,phi)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chi.inverse_design_common import (
    CLASS_LABEL_INTERNAL,
    CLASS_LABEL_PUBLIC,
    build_candidate_pool,
    default_chi_config,
    load_soluble_targets,
    load_step2_resampling_step_summary,
    parse_candidate_source,
    prepare_novel_inference_cache,
    resolve_training_smiles,
    set_plot_style,
)
from src.chi.model import predict_chi_mean_std_from_coefficients
from src.chi.constants import COEFF_NAMES
from src.data.tokenizer import PSmilesTokenizer
from src.evaluation.class_decode_constraints import (
    load_decode_constraint_source_smiles,
    resolve_class_decode_motifs,
)
from src.evaluation.polymer_class import PolymerClassifier
from src.utils.chemistry import (
    compute_sa_score,
    check_validity,
    count_stars,
    canonicalize_smiles,
    batch_compute_fingerprints,
    compute_pairwise_diversity,
)
from src.utils.config import load_config, save_config
from src.utils.model_scales import get_results_dir
from src.utils.reproducibility import save_run_metadata, seed_everything
from src.utils.reporting import save_step_summary, save_artifact_manifest, write_initial_log


K_LIST = [1, 3, 5, 10]


def _parse_target_polymer_classes(value: str, available: List[str]) -> List[str]:
    v = (value or "all").strip().lower()
    if v in {"all", "*"}:
        return available

    requested = [x.strip().lower() for x in v.split(",") if x.strip()]
    unknown = [x for x in requested if x not in available]
    if unknown:
        raise ValueError(
            "Unknown polymer class target(s): "
            + ", ".join(unknown)
            + ". Available: "
            + ", ".join(available)
        )
    return requested


def _annotate_polymer_family_matches(coeff_df: pd.DataFrame, patterns: Dict[str, str]) -> pd.DataFrame:
    out = coeff_df.copy()
    classifier = PolymerClassifier(patterns=patterns)

    classes = list(patterns.keys())
    for name in classes:
        out[f"polymer_class_{name}"] = 0

    matches = []
    for smi in out["SMILES"].astype(str).tolist():
        matches.append(classifier.classify(smi))

    for name in classes:
        out[f"polymer_class_{name}"] = [int(m.get(name, False)) for m in matches]

    out["polymer_class_any"] = out[[f"polymer_class_{c}" for c in classes]].max(axis=1)
    return out


def _build_targets_for_step6(base_targets: pd.DataFrame, polymer_classes: List[str]) -> pd.DataFrame:
    rows = []
    for _, row in base_targets.iterrows():
        for cls in polymer_classes:
            rows.append(
                {
                    "temperature": float(row["temperature"]),
                    "phi": float(row["phi"]),
                    "target_chi": float(row["target_chi"]),
                    "property_rule": row.get("property_rule", "upper_bound"),
                    "target_polymer_class": cls,
                }
            )
    out = pd.DataFrame(rows)
    out.insert(0, "target_id", np.arange(1, len(out) + 1))
    return out


def _resolve_step2_tokenizer_path(results_dir: Path, base_results_dir: Path) -> Path:
    for candidate in [results_dir / "tokenizer.json", base_results_dir / "tokenizer.json"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Step 2 tokenizer.json not found in either {results_dir} or {base_results_dir}"
    )


def _compute_target_candidates(
    target_row: pd.Series,
    coeff_df: pd.DataFrame,
    epsilon: float,
    default_property_rule: str,
    uncertainty_enabled: bool,
    uncertainty_class_z: float,
    uncertainty_property_z: float,
    uncertainty_score_weight: float,
) -> pd.DataFrame:
    target_chi = float(target_row["target_chi"])
    t = float(target_row["temperature"])
    phi = float(target_row["phi"])
    target_polymer_class = str(target_row["target_polymer_class"]).strip().lower()

    class_col = f"polymer_class_{target_polymer_class}"
    required_cols = ["polymer_id", "Polymer", "SMILES", "class_prob", *COEFF_NAMES, class_col]
    missing = [c for c in required_cols if c not in coeff_df.columns]
    if missing:
        raise ValueError(f"Candidate pool missing required columns: {missing}")

    coeff_std_cols = [f"{name}_std" for name in COEFF_NAMES]
    copy_cols = required_cols + [
        c
        for c in [
            CLASS_LABEL_INTERNAL,
            CLASS_LABEL_PUBLIC,
            "candidate_source",
            "sampling_attempt",
            "canonical_smiles",
            "class_logit",
            "class_logit_std",
            "class_prob_std",
            "is_novel_vs_train",
            *coeff_std_cols,
        ]
        if c in coeff_df.columns
    ]
    out = coeff_df[copy_cols].copy()
    for col in ["class_prob_std", "class_logit_std", *coeff_std_cols]:
        if col not in out.columns:
            out[col] = 0.0

    coeff_mean = out[COEFF_NAMES].to_numpy(dtype=float)
    coeff_std = out[coeff_std_cols].to_numpy(dtype=float)
    pred_mean, pred_std = predict_chi_mean_std_from_coefficients(
        coeff_mean=coeff_mean,
        coeff_std=coeff_std,
        temperature=np.full(len(out), t, dtype=float),
        phi=np.full(len(out), phi, dtype=float),
    )

    out["target_id"] = int(target_row["target_id"])
    out["temperature"] = t
    out["phi"] = phi
    out["target_chi"] = target_chi
    out["target_polymer_class"] = target_polymer_class
    out["chi_pred_target"] = pred_mean
    out["chi_pred_std_target"] = pred_std
    out["chi_error"] = out["chi_pred_target"] - target_chi
    out["abs_error"] = np.abs(out["chi_error"])

    property_rule = str(target_row.get("property_rule", default_property_rule)).strip().lower()
    if property_rule not in {"band", "upper_bound", "lower_bound"}:
        property_rule = default_property_rule
    out["property_rule"] = property_rule

    out["class_prob_std"] = out["class_prob_std"].astype(float).fillna(0.0)
    if uncertainty_enabled:
        out["class_prob_lcb"] = np.clip(
            out["class_prob"].astype(float) - float(uncertainty_class_z) * out["class_prob_std"],
            0.0,
            1.0,
        )
    else:
        out["class_prob_lcb"] = out["class_prob"].astype(float)
    out["soluble_confidence"] = out["class_prob_lcb"]
    out["pred_soluble"] = (out["soluble_confidence"] >= 0.5).astype(int)
    out["soluble_hit"] = out["pred_soluble"].astype(int)

    out["polymer_class_hit"] = out[class_col].astype(int)

    prop_z = float(uncertainty_property_z) if uncertainty_enabled else 0.0
    if property_rule == "upper_bound":
        out["chi_pred_conservative"] = out["chi_pred_target"] + prop_z * out["chi_pred_std_target"]
        out["property_error"] = np.maximum(out["chi_pred_conservative"] - target_chi, 0.0)
        out["property_hit"] = (out["chi_pred_conservative"] <= target_chi).astype(int)
    elif property_rule == "lower_bound":
        out["chi_pred_conservative"] = out["chi_pred_target"] - prop_z * out["chi_pred_std_target"]
        out["property_error"] = np.maximum(target_chi - out["chi_pred_conservative"], 0.0)
        out["property_hit"] = (out["chi_pred_conservative"] >= target_chi).astype(int)
    else:
        out["chi_pred_conservative"] = out["chi_pred_target"]
        out["property_error"] = out["abs_error"] + prop_z * out["chi_pred_std_target"]
        out["property_hit"] = (out["property_error"] <= float(epsilon)).astype(int)

    out["step4_requirement_hit_count"] = out["soluble_hit"] + out["property_hit"]
    out["step4_requirement_miss_count"] = 2 - out["step4_requirement_hit_count"]
    out["total_requirement_hit_count"] = out["step4_requirement_hit_count"] + out["polymer_class_hit"]
    out["total_requirement_miss_count"] = 3 - out["total_requirement_hit_count"]
    out["joint_hit"] = (
        (out["soluble_hit"] == 1)
        & (out["polymer_class_hit"] == 1)
        & (out["property_hit"] == 1)
    ).astype(int)

    out["score"] = (
        out["property_error"]
        + float(uncertainty_score_weight) * out["chi_pred_std_target"]
    )

    out = out.sort_values(
        [
            "total_requirement_miss_count",
            "step4_requirement_miss_count",
            "score",
            "polymer_class_hit",
            "property_error",
            "chi_pred_std_target",
            "soluble_confidence",
        ],
        ascending=[True, True, True, False, True, True, False],
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def _target_metrics(cand: pd.DataFrame) -> Dict[str, float]:
    row = {
        "target_id": int(cand["target_id"].iloc[0]),
        "target_polymer_class": cand["target_polymer_class"].iloc[0],
        "property_rule": cand["property_rule"].iloc[0],
        "temperature": float(cand["temperature"].iloc[0]),
        "phi": float(cand["phi"].iloc[0]),
        "target_chi": float(cand["target_chi"].iloc[0]),
        "n_candidates": int(len(cand)),
        "n_soluble_hit": int(cand["soluble_hit"].sum()),
        "n_polymer_class_hit": int(cand["polymer_class_hit"].sum()),
        "n_property_hit": int(cand["property_hit"].sum()),
        "n_joint_hit": int(cand["joint_hit"].sum()),
        "soluble_hit_rate": float(cand["soluble_hit"].mean()),
        "polymer_class_hit_rate": float(cand["polymer_class_hit"].mean()),
        "property_hit_rate": float(cand["property_hit"].mean()),
        "joint_hit_rate": float(cand["joint_hit"].mean()),
        "top1_polymer": cand["Polymer"].iloc[0],
        "top1_candidate_source": cand["candidate_source"].iloc[0] if "candidate_source" in cand.columns else "unknown",
        "top1_class_prob": float(cand["class_prob"].iloc[0]),
        "top1_class_prob_std": float(cand["class_prob_std"].iloc[0]) if "class_prob_std" in cand.columns else 0.0,
        "top1_class_prob_lcb": float(cand["class_prob_lcb"].iloc[0]) if "class_prob_lcb" in cand.columns else float(cand["class_prob"].iloc[0]),
        "top1_soluble_hit": int(cand["soluble_hit"].iloc[0]),
        "top1_polymer_class_hit": int(cand["polymer_class_hit"].iloc[0]),
        "top1_property_hit": int(cand["property_hit"].iloc[0]),
        "top1_joint_hit": int(cand["joint_hit"].iloc[0]),
        "top1_property_error": float(cand["property_error"].iloc[0]),
        "top1_abs_error": float(cand["abs_error"].iloc[0]),
        "top1_pred_chi": float(cand["chi_pred_target"].iloc[0]),
        "top1_pred_chi_std": float(cand["chi_pred_std_target"].iloc[0]) if "chi_pred_std_target" in cand.columns else 0.0,
        "top1_is_novel_vs_train": int(cand["is_novel_vs_train"].iloc[0]) if "is_novel_vs_train" in cand.columns else np.nan,
    }

    for k in K_LIST:
        topk = cand[cand["rank"] <= k]
        row[f"top{k}_soluble_hit"] = int(topk["soluble_hit"].max())
        row[f"top{k}_polymer_class_hit"] = int(topk["polymer_class_hit"].max())
        row[f"top{k}_property_hit"] = int(topk["property_hit"].max())
        row[f"top{k}_joint_hit"] = int(topk["joint_hit"].max())
        row[f"top{k}_joint_hit_rate"] = float(topk["joint_hit"].mean()) if len(topk) else np.nan
        if "is_novel_vs_train" in cand.columns:
            row[f"top{k}_novel_rate"] = float(topk["is_novel_vs_train"].mean()) if len(topk) else np.nan

    joint = cand[cand["joint_hit"] == 1]
    row["best_joint_abs_error"] = float(joint["abs_error"].min()) if not joint.empty else np.nan
    row["first_joint_rank"] = int(joint["rank"].min()) if not joint.empty else np.nan
    row["mrr_joint"] = float(1.0 / row["first_joint_rank"]) if not np.isnan(row["first_joint_rank"]) else 0.0
    return row


def _aggregate_metrics(target_metrics_df: pd.DataFrame) -> pd.DataFrame:
    def summarize(scope: str, sub: pd.DataFrame) -> Dict[str, float]:
        out = {
            "scope": scope,
            "n_targets": int(len(sub)),
            "target_success_rate": float(np.mean(sub["n_joint_hit"] > 0)),
            "mean_top1_abs_error": float(np.mean(sub["top1_abs_error"])),
            "median_top1_abs_error": float(np.median(sub["top1_abs_error"])),
            "mean_mrr_joint": float(np.mean(sub["mrr_joint"])),
            "mean_joint_hit_rate": float(np.mean(sub["joint_hit_rate"])),
            "mean_soluble_hit_rate": float(np.mean(sub["soluble_hit_rate"])),
            "mean_polymer_class_hit_rate": float(np.mean(sub["polymer_class_hit_rate"])),
            "mean_property_hit_rate": float(np.mean(sub["property_hit_rate"])),
        }
        if "top1_pred_chi_std" in sub.columns:
            out["mean_top1_pred_chi_std"] = float(np.mean(sub["top1_pred_chi_std"]))
        if "top1_class_prob_std" in sub.columns:
            out["mean_top1_class_prob_std"] = float(np.mean(sub["top1_class_prob_std"]))
        for k in K_LIST:
            out[f"target_success_top{k}"] = float(np.mean(sub[f"top{k}_joint_hit"] > 0))
        return out

    rows = [summarize("overall", target_metrics_df)]
    for cls, sub in target_metrics_df.groupby("target_polymer_class"):
        rows.append(summarize(f"polymer_class_{cls}", sub))
    return pd.DataFrame(rows)


def _postprocess_metrics(candidate_df: pd.DataFrame, target_metrics_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    by_condition = (
        target_metrics_df.groupby(["target_polymer_class", "temperature", "phi"], as_index=False)
        .agg(
            top1_property_error_mean=("top1_property_error", "mean"),
            top1_abs_error_mean=("top1_abs_error", "mean"),
            target_success=("n_joint_hit", lambda x: float(np.mean(np.asarray(x) > 0))),
            top1_joint_hit_rate=("top1_joint_hit", "mean"),
            top5_joint_hit_rate=("top5_joint_hit", "mean"),
        )
        .sort_values(["target_polymer_class", "temperature", "phi"])
    )

    top1 = candidate_df[candidate_df["rank"] == 1].copy()
    top1_sa_rows = []
    for _, row in top1.iterrows():
        sa = compute_sa_score(str(row["SMILES"]))
        top1_sa_rows.append(
            {
                "target_id": int(row["target_id"]),
                "target_polymer_class": row["target_polymer_class"],
                "polymer_id": int(row["polymer_id"]),
                "Polymer": row["Polymer"],
                "candidate_source": row.get("candidate_source", "unknown"),
                "sa_score": float(sa) if sa is not None else np.nan,
            }
        )
    top1_sa = pd.DataFrame(top1_sa_rows)

    return {
        "by_condition": by_condition,
        "top1_sa": top1_sa,
    }


def _build_polymer_family_map(coeff_df: pd.DataFrame) -> Dict[int, str]:
    class_cols = [
        c for c in coeff_df.columns
        if c.startswith("polymer_class_") and c not in {"polymer_class_any"}
    ]
    out: Dict[int, str] = {}
    if not class_cols:
        return out

    for _, row in coeff_df.iterrows():
        pid = int(row["polymer_id"])
        matched = [c.replace("polymer_class_", "") for c in class_cols if int(row[c]) == 1]
        out[pid] = "|".join(sorted(matched)) if matched else "unclassified"
    return out


def _compute_class_coverage(coeff_df: pd.DataFrame, selected_classes: List[str]) -> Dict[str, int]:
    return {
        cls: int(coeff_df[f"polymer_class_{cls}"].sum()) if f"polymer_class_{cls}" in coeff_df.columns else 0
        for cls in selected_classes
    }


def _accumulate_candidate_pools(pool_frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not pool_frames:
        return pd.DataFrame()

    coeff_df = pd.concat(pool_frames, ignore_index=True)
    if "canonical_smiles" not in coeff_df.columns:
        coeff_df["canonical_smiles"] = coeff_df["SMILES"].astype(str).map(canonicalize_smiles)
    coeff_df["canonical_smiles"] = coeff_df["canonical_smiles"].where(
        coeff_df["canonical_smiles"].notna(),
        coeff_df["SMILES"].astype(str),
    )
    if "source_polymer_id" not in coeff_df.columns:
        coeff_df["source_polymer_id"] = coeff_df.get("polymer_id", pd.Series([-1] * len(coeff_df))).astype(int)
    else:
        coeff_df["source_polymer_id"] = coeff_df["source_polymer_id"].fillna(-1).astype(int)
    coeff_df = coeff_df.drop_duplicates(subset=["canonical_smiles"], keep="first").reset_index(drop=True)
    coeff_df["polymer_id"] = np.arange(len(coeff_df), dtype=int)
    return coeff_df


def _empty_target_metrics(target_row: pd.Series) -> Dict[str, float]:
    row = {
        "target_id": int(target_row["target_id"]),
        "target_polymer_class": str(target_row["target_polymer_class"]),
        "property_rule": str(target_row.get("property_rule", "upper_bound")),
        "temperature": float(target_row["temperature"]),
        "phi": float(target_row["phi"]),
        "target_chi": float(target_row["target_chi"]),
        "n_candidates": 0,
        "n_soluble_hit": 0,
        "n_polymer_class_hit": 0,
        "n_property_hit": 0,
        "n_joint_hit": 0,
        "soluble_hit_rate": 0.0,
        "polymer_class_hit_rate": 0.0,
        "property_hit_rate": 0.0,
        "joint_hit_rate": 0.0,
        "top1_polymer": "",
        "top1_candidate_source": "unknown",
        "top1_class_prob": np.nan,
        "top1_class_prob_std": 0.0,
        "top1_class_prob_lcb": np.nan,
        "top1_soluble_hit": 0,
        "top1_polymer_class_hit": 0,
        "top1_property_hit": 0,
        "top1_joint_hit": 0,
        "top1_property_error": np.nan,
        "top1_abs_error": np.nan,
        "top1_pred_chi": np.nan,
        "top1_pred_chi_std": 0.0,
        "top1_is_novel_vs_train": np.nan,
        "best_joint_abs_error": np.nan,
        "first_joint_rank": np.nan,
        "mrr_joint": 0.0,
    }
    for k in K_LIST:
        row[f"top{k}_soluble_hit"] = 0
        row[f"top{k}_polymer_class_hit"] = 0
        row[f"top{k}_property_hit"] = 0
        row[f"top{k}_joint_hit"] = 0
        row[f"top{k}_joint_hit_rate"] = 0.0
        row[f"top{k}_novel_rate"] = np.nan
    return row


def _score_candidate_pool(
    coeff_df: pd.DataFrame,
    target_df: pd.DataFrame,
    selected_classes: List[str],
    epsilon: float,
    default_property_rule: str,
    uncertainty_enabled: bool,
    uncertainty_class_z: float,
    uncertainty_property_z: float,
    uncertainty_score_weight: float,
    coverage_topk: int,
    training_canonical: set[str],
    target_polymer_count: int,
    target_stars: int,
    target_sa_max: float,
    n_base_conditions: int,
) -> Dict[str, object]:
    class_coverage = _compute_class_coverage(coeff_df, selected_classes)
    polymer_family_map = _build_polymer_family_map(coeff_df)

    all_candidates = []
    target_metrics = []
    for _, row in target_df.iterrows():
        cand = _compute_target_candidates(
            target_row=row,
            coeff_df=coeff_df,
            epsilon=epsilon,
            default_property_rule=default_property_rule,
            uncertainty_enabled=uncertainty_enabled,
            uncertainty_class_z=uncertainty_class_z,
            uncertainty_property_z=uncertainty_property_z,
            uncertainty_score_weight=uncertainty_score_weight,
        )
        all_candidates.append(cand)
        target_metrics.append(_target_metrics(cand) if not cand.empty else _empty_target_metrics(row))

    candidate_df = pd.concat(all_candidates, ignore_index=True)
    target_metrics_df = pd.DataFrame(target_metrics)
    aggregate_df = _aggregate_metrics(target_metrics_df)
    post = _postprocess_metrics(candidate_df, target_metrics_df)

    topk = candidate_df[candidate_df["rank"] <= coverage_topk].copy()
    if "class_prob_lcb" not in topk.columns:
        topk["class_prob_lcb"] = topk["class_prob"]
    if "chi_pred_std_target" not in topk.columns:
        topk["chi_pred_std_target"] = 0.0
    coverage = (
        topk.groupby(["target_polymer_class", "candidate_source", "polymer_id", "Polymer"], as_index=False)
        .agg(
            selected_count=("rank", "size"),
            mean_class_prob=("class_prob", "mean"),
            mean_class_prob_lcb=("class_prob_lcb", "mean"),
            mean_property_error=("property_error", "mean"),
            mean_abs_error=("abs_error", "mean"),
            mean_pred_chi_std=("chi_pred_std_target", "mean"),
            mean_novel_rate=("is_novel_vs_train", "mean"),
            mean_polymer_class_hit=("polymer_class_hit", "mean"),
        )
        .sort_values(["target_polymer_class", "selected_count"], ascending=[True, False])
    )
    coverage["selected_rate"] = coverage["selected_count"] / float(n_base_conditions)

    target_poly_df, target_poly_summary = _select_final_target_polymers(
        candidate_df=candidate_df,
        training_canonical=training_canonical,
        polymer_family_map=polymer_family_map,
        target_count=target_polymer_count,
        target_stars=target_stars,
        sa_max=target_sa_max,
        total_sampling_points=int(len(coeff_df)),
    )

    return {
        "class_coverage": class_coverage,
        "candidate_df": candidate_df,
        "target_metrics_df": target_metrics_df,
        "aggregate_df": aggregate_df,
        "post": post,
        "coverage": coverage,
        "target_poly_df": target_poly_df,
        "target_poly_summary": target_poly_summary,
    }


def _select_final_target_polymers(
    candidate_df: pd.DataFrame,
    training_canonical: set[str],
    polymer_family_map: Dict[int, str],
    target_count: int,
    target_stars: int,
    sa_max: float,
    total_sampling_points: int | None = None,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    confidence_col = "soluble_confidence" if "soluble_confidence" in candidate_df.columns else "class_prob"
    ranked = candidate_df.sort_values(
        [
            "step4_requirement_miss_count",
            "total_requirement_miss_count",
            "score",
            "rank",
            "polymer_class_hit",
            "property_error",
            "chi_pred_std_target",
            confidence_col,
        ],
        ascending=[True, True, True, True, False, True, True, False],
    ).copy()

    dedup_key = "polymer_id" if "polymer_id" in ranked.columns else "SMILES"
    required_targets = int(candidate_df["target_id"].nunique()) if "target_id" in candidate_df.columns else 1

    by_polymer = candidate_df.copy()
    by_polymer["joint_condition_hit"] = (
        (by_polymer["soluble_hit"] == 1)
        & (by_polymer["property_hit"] == 1)
        & (by_polymer["polymer_class_hit"] == 1)
    ).astype(int)
    if "target_id" in by_polymer.columns:
        per_polymer = by_polymer.groupby(dedup_key, as_index=False).agg(
            n_targets_evaluated=("target_id", "nunique"),
            n_targets_joint_hit=("joint_condition_hit", "sum"),
            mean_property_error_all_targets=("property_error", "mean"),
            max_property_error_all_targets=("property_error", "max"),
        )
    else:
        per_polymer = by_polymer.groupby(dedup_key, as_index=False).agg(
            n_targets_evaluated=("joint_condition_hit", "size"),
            n_targets_joint_hit=("joint_condition_hit", "sum"),
            mean_property_error_all_targets=("property_error", "mean"),
            max_property_error_all_targets=("property_error", "max"),
        )

    required_by_class: Dict[str, int] = {}
    if {"target_id", "target_polymer_class"}.issubset(by_polymer.columns):
        per_polymer_class = by_polymer.groupby(
            [dedup_key, "target_polymer_class"], as_index=False
        ).agg(
            n_targets_evaluated=("target_id", "nunique"),
            n_targets_joint_hit=("joint_condition_hit", "sum"),
        )
        required_by_class = (
            by_polymer.groupby("target_polymer_class")["target_id"]
            .nunique()
            .astype(int)
            .to_dict()
        )
        per_polymer_class["n_targets_required_for_class"] = per_polymer_class["target_polymer_class"].map(required_by_class).astype(int)
        per_polymer_class["passes_target_class"] = (
            (per_polymer_class["n_targets_evaluated"] == per_polymer_class["n_targets_required_for_class"])
            & (per_polymer_class["n_targets_joint_hit"] == per_polymer_class["n_targets_required_for_class"])
        ).astype(int)
        class_level = per_polymer_class.groupby(dedup_key, as_index=False).agg(
            n_target_classes_required=("target_polymer_class", "nunique"),
            n_target_classes_pass=("passes_target_class", "sum"),
        )
        class_level["passes_any_target_class"] = (class_level["n_target_classes_pass"] >= 1).astype(int)
    else:
        class_level = per_polymer[[dedup_key]].copy()
        class_level["n_target_classes_required"] = 1
        class_level["n_target_classes_pass"] = (
            (per_polymer["n_targets_evaluated"] == int(required_targets))
            & (per_polymer["n_targets_joint_hit"] == int(required_targets))
        ).astype(int)
        class_level["passes_any_target_class"] = class_level["n_target_classes_pass"]

    required_targets_for_any_class = int(max(required_by_class.values())) if required_by_class else int(required_targets)
    n_target_classes_required = int(len(required_by_class)) if required_by_class else 1

    per_polymer["n_targets_required"] = int(required_targets)
    per_polymer = per_polymer.merge(class_level, on=dedup_key, how="left")
    per_polymer["passes_all_target_conditions"] = per_polymer["passes_any_target_class"].astype(int)

    ranked = ranked.drop_duplicates(subset=[dedup_key], keep="first").reset_index(drop=True)
    ranked = ranked.merge(per_polymer, on=dedup_key, how="left")

    rows = []
    for _, row in ranked.iterrows():
        smiles = str(row["SMILES"])
        is_valid = check_validity(smiles)
        canonical = row.get("canonical_smiles")
        if not canonical or pd.isna(canonical):
            canonical = canonicalize_smiles(smiles) if is_valid else None
        star_count = count_stars(smiles)
        is_novel = bool(canonical) and canonical not in training_canonical
        if "is_novel_vs_train" in row and not pd.isna(row["is_novel_vs_train"]):
            is_novel = bool(int(row["is_novel_vs_train"]))

        sa_score = compute_sa_score(smiles) if is_valid else None
        sa_ok = sa_score is not None and float(sa_score) < float(sa_max)
        pid = int(row["polymer_id"]) if "polymer_id" in row else -1
        polymer_family = polymer_family_map.get(pid, "unclassified")

        rows.append(
            {
                "target_id": int(row["target_id"]),
                "target_polymer_class": row["target_polymer_class"],
                "Polymer": row["Polymer"],
                "SMILES": smiles,
                "canonical_smiles": canonical,
                "candidate_source": row.get("candidate_source", "unknown"),
                "sampling_attempt": int(row["sampling_attempt"]) if not pd.isna(row.get("sampling_attempt", np.nan)) else np.nan,
                "temperature": float(row["temperature"]),
                "phi": float(row["phi"]),
                "target_chi": float(row["target_chi"]),
                "property_rule": str(row.get("property_rule", "upper_bound")),
                "chi_pred_target": float(row["chi_pred_target"]),
                "property_error": float(row["property_error"]),
                "abs_error": float(row["abs_error"]),
                "class_prob": float(row["class_prob"]),
                "class_prob_std": float(row["class_prob_std"]) if not pd.isna(row.get("class_prob_std", np.nan)) else 0.0,
                "class_prob_lcb": float(row["class_prob_lcb"]) if not pd.isna(row.get("class_prob_lcb", np.nan)) else float(row["class_prob"]),
                "soluble_hit": int(row["soluble_hit"]),
                "score": float(row["score"]),
                "rank_in_target": int(row["rank"]),
                "chi_pred_std_target": float(row["chi_pred_std_target"]) if not pd.isna(row.get("chi_pred_std_target", np.nan)) else 0.0,
                "chi_pred_conservative": float(row["chi_pred_conservative"]) if not pd.isna(row.get("chi_pred_conservative", np.nan)) else float(row["chi_pred_target"]),
                "property_hit": int(row["property_hit"]),
                "polymer_class_hit": int(row["polymer_class_hit"]),
                "step4_requirement_hit_count": int(
                    row.get("step4_requirement_hit_count", row["soluble_hit"] + row["property_hit"])
                ),
                "step4_requirement_miss_count": int(
                    row.get("step4_requirement_miss_count", 2 - (row["soluble_hit"] + row["property_hit"]))
                ),
                "total_requirement_hit_count": int(
                    row.get("total_requirement_hit_count", row["soluble_hit"] + row["property_hit"] + row["polymer_class_hit"])
                ),
                "total_requirement_miss_count": int(
                    row.get("total_requirement_miss_count", 3 - (row["soluble_hit"] + row["property_hit"] + row["polymer_class_hit"]))
                ),
                "is_valid": int(is_valid),
                "star_count": int(star_count),
                "is_novel_vs_train": int(is_novel),
                "sa_score": float(sa_score) if sa_score is not None else np.nan,
                "sa_ok": int(sa_ok),
                "polymer_family": polymer_family,
                "n_targets_required": int(row["n_targets_required"]) if not pd.isna(row.get("n_targets_required", np.nan)) else 0,
                "n_targets_evaluated": int(row["n_targets_evaluated"]) if not pd.isna(row.get("n_targets_evaluated", np.nan)) else 0,
                "n_targets_joint_hit": int(row["n_targets_joint_hit"]) if not pd.isna(row.get("n_targets_joint_hit", np.nan)) else 0,
                "n_target_classes_required": int(row["n_target_classes_required"]) if not pd.isna(row.get("n_target_classes_required", np.nan)) else 0,
                "n_target_classes_pass": int(row["n_target_classes_pass"]) if not pd.isna(row.get("n_target_classes_pass", np.nan)) else 0,
                "passes_any_target_class": int(row["passes_any_target_class"]) if not pd.isna(row.get("passes_any_target_class", np.nan)) else 0,
                "passes_all_target_conditions": int(row["passes_all_target_conditions"]) if not pd.isna(row.get("passes_all_target_conditions", np.nan)) else 0,
                "mean_property_error_all_targets": float(row["mean_property_error_all_targets"]) if not pd.isna(row.get("mean_property_error_all_targets", np.nan)) else np.nan,
                "max_property_error_all_targets": float(row["max_property_error_all_targets"]) if not pd.isna(row.get("max_property_error_all_targets", np.nan)) else np.nan,
            }
        )

    all_df = pd.DataFrame(rows)
    real_total_sampling_points = int(total_sampling_points) if total_sampling_points is not None else int(len(all_df))
    if all_df.empty:
        return pd.DataFrame(), {
            "required_targets_per_polymer": int(required_targets_for_any_class),
            "required_target_classes": int(n_target_classes_required),
            "n_polymers_pass_all_targets": 0,
            "n_polymers_pass_any_target_class": 0,
            "target_count_requested": int(target_count),
            "total_candidates_screened": int(real_total_sampling_points),
            "total_candidates_evaluated_after_target_aggregation": 0,
            "screen_after_valid_count": 0,
            "screen_after_target_stars_count": 0,
            "screen_after_novel_count": 0,
            "screen_after_sa_count": 0,
            "screen_after_target_requirements_count": 0,
            "filter_pass_count": 0,
            "filter_pass_unique": 0,
            "target_count_selected": 0,
            "selection_success_rate": 0.0,
            "final_diversity": 0.0,
            "final_mean_sa": np.nan,
            "final_std_sa": np.nan,
            "final_mean_property_error": np.nan,
        }

    valid_mask = all_df["is_valid"] == 1
    star_mask = valid_mask & (all_df["star_count"] == int(target_stars))
    novel_mask = star_mask & (all_df["is_novel_vs_train"] == 1)
    sa_mask = novel_mask & (all_df["sa_ok"] == 1)
    filter_mask = sa_mask & (all_df["passes_all_target_conditions"] == 1)
    filtered = all_df.loc[filter_mask].copy()
    filtered = filtered.drop_duplicates(subset=["canonical_smiles"], keep="first").reset_index(drop=True)

    selected = filtered.head(int(target_count)).copy()
    selected.insert(0, "target_rank", np.arange(1, len(selected) + 1))
    selected["is_unique"] = 1
    selected["passes_all_filters"] = 1

    diversity = 0.0
    if len(selected) >= 2:
        fps, _ = batch_compute_fingerprints(selected["SMILES"].astype(str).tolist())
        if len(fps) >= 2:
            diversity = float(compute_pairwise_diversity(fps))

    sa_vals = selected["sa_score"].to_numpy(dtype=float) if not selected.empty else np.array([])
    prop_vals = selected["max_property_error_all_targets"].to_numpy(dtype=float) if not selected.empty else np.array([])
    summary = {
        "required_targets_per_polymer": int(required_targets_for_any_class),
        "required_target_classes": int(n_target_classes_required),
        "n_polymers_pass_all_targets": int(all_df["passes_all_target_conditions"].sum()),
        "n_polymers_pass_any_target_class": int(all_df["passes_any_target_class"].sum()),
        "target_count_requested": int(target_count),
        "total_candidates_screened": int(real_total_sampling_points),
        "total_candidates_evaluated_after_target_aggregation": int(len(all_df)),
        "screen_after_valid_count": int(valid_mask.sum()),
        "screen_after_target_stars_count": int(star_mask.sum()),
        "screen_after_novel_count": int(novel_mask.sum()),
        "screen_after_sa_count": int(sa_mask.sum()),
        "screen_after_target_requirements_count": int(filter_mask.sum()),
        "filter_pass_count": int(filter_mask.sum()),
        "filter_pass_unique": int(len(filtered)),
        "target_count_selected": int(len(selected)),
        "selection_success_rate": float(len(selected) / real_total_sampling_points) if real_total_sampling_points > 0 else 0.0,
        "final_diversity": float(diversity),
        "final_mean_sa": float(np.nanmean(sa_vals)) if sa_vals.size else np.nan,
        "final_std_sa": float(np.nanstd(sa_vals)) if sa_vals.size else np.nan,
        "final_mean_property_error": float(np.nanmean(prop_vals)) if prop_vals.size else np.nan,
    }
    return selected, summary


def _append_step_log(step_dir: Path, lines: List[str]) -> None:
    log_path = step_dir / "log.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n")
        for line in lines:
            f.write(f"{line}\n")


def _build_selected_target_candidate_ranked_df(target_poly_df: pd.DataFrame) -> pd.DataFrame:
    if target_poly_df.empty:
        return target_poly_df.copy()

    preferred_cols = [
        "target_rank",
        "Polymer",
        "SMILES",
        "target_polymer_class",
        "polymer_family",
        "candidate_source",
        "sampling_attempt",
        "class_prob",
        "class_prob_lcb",
        "polymer_class_hit",
        "max_property_error_all_targets",
        "mean_property_error_all_targets",
        "n_targets_joint_hit",
        "n_target_classes_pass",
        "passes_any_target_class",
        "passes_all_target_conditions",
        "sa_score",
    ]
    cols = [c for c in preferred_cols if c in target_poly_df.columns]
    out = target_poly_df[cols].copy()
    if "target_rank" in out.columns:
        out = out.sort_values("target_rank").reset_index(drop=True)
    return out


def _build_sampling_process_summary(
    attempt_rows: List[Dict[str, object]],
    target_poly_summary: Dict[str, float],
    resampling_target_polymer_count: int,
    sampling_attempts_max: int,
) -> pd.DataFrame:
    total_raw = int(sum(int(row.get("step2_generated_count_raw", 0) or 0) for row in attempt_rows))
    total_accepted = int(sum(int(row.get("step2_accepted_count", 0) or 0) for row in attempt_rows))
    total_shortfall = int(sum(int(row.get("step2_valid_only_shortfall_count", 0) or 0) for row in attempt_rows))
    qualified = int(target_poly_summary.get("filter_pass_unique", 0))
    selected = int(target_poly_summary.get("target_count_selected", 0))
    screened = int(target_poly_summary.get("total_candidates_screened", 0))
    return pd.DataFrame(
        [
            {
                "sampling_attempts_max": int(sampling_attempts_max),
                "sampling_attempts_used": int(len(attempt_rows)),
                "resampling_target_polymer_count_per_attempt": int(resampling_target_polymer_count),
                "step2_raw_generated_total": total_raw,
                "step2_accepted_total": total_accepted,
                "step2_shortfall_total": total_shortfall,
                "step2_overall_acceptance_rate": float(total_accepted / total_raw) if total_raw > 0 else np.nan,
                "step6_candidates_screened_total": screened,
                "step6_qualified_candidate_count": qualified,
                "step6_qualified_fraction_of_screened": float(qualified / screened) if screened > 0 else np.nan,
                "step6_target_count_selected": selected,
                "step6_selected_fraction_of_qualified": float(selected / qualified) if qualified > 0 else np.nan,
                "stop_reason": (
                    "target_count_reached"
                    if selected >= int(target_poly_summary.get("target_count_requested", 0))
                    else "max_attempts_reached"
                ),
            }
        ]
    )


def _cleanup_previous_figures(out_dir: Path) -> None:
    for png_path in out_dir.glob("*.png"):
        try:
            png_path.unlink()
        except OSError:
            continue


def _compute_property_requirement_margin(df: pd.DataFrame, epsilon: float) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)

    rule = (
        df["property_rule"].astype(str).str.lower()
        if "property_rule" in df.columns
        else pd.Series(["upper_bound"] * len(df), index=df.index, dtype=object)
    )
    margin = pd.Series(np.nan, index=df.index, dtype=float)

    upper_mask = rule == "upper_bound"
    lower_mask = rule == "lower_bound"
    band_mask = ~(upper_mask | lower_mask)

    if {"target_chi", "chi_pred_conservative"}.issubset(df.columns):
        margin.loc[upper_mask] = (
            df.loc[upper_mask, "target_chi"].to_numpy(dtype=float)
            - df.loc[upper_mask, "chi_pred_conservative"].to_numpy(dtype=float)
        )
        margin.loc[lower_mask] = (
            df.loc[lower_mask, "chi_pred_conservative"].to_numpy(dtype=float)
            - df.loc[lower_mask, "target_chi"].to_numpy(dtype=float)
        )

    if "property_error" in df.columns:
        margin.loc[band_mask] = float(epsilon) - df.loc[band_mask, "property_error"].to_numpy(dtype=float)
    return margin


def _save_figures(
    target_metrics_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    target_poly_df: pd.DataFrame,
    target_poly_summary: Dict[str, float],
    sampling_attempts_df: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    font_size: int,
    epsilon: float,
    target_sa_max: float,
    target_stars: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_previous_figures(out_dir)
    set_plot_style(font_size)

    # Sampling progress across attempts
    if not sampling_attempts_df.empty and "sampling_attempt" in sampling_attempts_df.columns:
        progress = sampling_attempts_df.sort_values("sampling_attempt").copy()
        selected_attempt_counts = (
            target_poly_df["sampling_attempt"].dropna().astype(int).value_counts().sort_index()
            if not target_poly_df.empty and "sampling_attempt" in target_poly_df.columns
            else pd.Series(dtype=int)
        )
        n_panels = 3 if not selected_attempt_counts.empty else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(5.3 * n_panels, 4.5))
        axes = np.atleast_1d(axes)

        count_specs = [
            ("step2_accepted_count", "Step 2 accepted", "#4c78a8"),
            ("accumulated_candidate_count", "Accumulated screened", "#54a24b"),
            ("n_polymers_pass_any_target_class", "Class-qualified", "#72b7b2"),
            ("n_polymers_pass_all_targets", "Qualified", "#f58518"),
            ("target_count_selected", "Selected", "#e45756"),
        ]
        for col, label, color in count_specs:
            if col in progress.columns:
                axes[0].plot(
                    progress["sampling_attempt"].to_numpy(dtype=float),
                    progress[col].to_numpy(dtype=float),
                    marker="o",
                    linewidth=2,
                    color=color,
                    label=label,
                )
        axes[0].axhline(
            float(target_poly_summary.get("target_count_requested", 0)),
            color="#474747",
            linewidth=1.0,
            linestyle=":",
            label="Requested target count",
        )
        axes[0].set_xlabel("Sampling attempt")
        axes[0].set_ylabel("Count")
        axes[0].legend(frameon=False, fontsize=max(font_size - 4, 8))

        if "step2_valid_only_acceptance_rate" in progress.columns:
            axes[1].plot(
                progress["sampling_attempt"].to_numpy(dtype=float),
                progress["step2_valid_only_acceptance_rate"].to_numpy(dtype=float),
                marker="o",
                linewidth=2,
                color="#4c78a8",
                label="Step 2 acceptance rate",
            )
        if "accumulated_candidate_count" in progress.columns and "n_polymers_pass_all_targets" in progress.columns:
            denom = progress["accumulated_candidate_count"].replace(0, np.nan).to_numpy(dtype=float)
            qualified_rate = progress["n_polymers_pass_all_targets"].to_numpy(dtype=float) / denom
            axes[1].plot(
                progress["sampling_attempt"].to_numpy(dtype=float),
                qualified_rate,
                marker="s",
                linewidth=2,
                linestyle="--",
                color="#f58518",
                label="Qualified / screened",
            )
        if "accumulated_candidate_count" in progress.columns and "target_count_selected" in progress.columns:
            denom = progress["accumulated_candidate_count"].replace(0, np.nan).to_numpy(dtype=float)
            selected_rate = progress["target_count_selected"].to_numpy(dtype=float) / denom
            axes[1].plot(
                progress["sampling_attempt"].to_numpy(dtype=float),
                selected_rate,
                marker="^",
                linewidth=2,
                linestyle="-.",
                color="#e45756",
                label="Selected / screened",
            )
        axes[1].set_xlabel("Sampling attempt")
        axes[1].set_ylabel("Rate")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(frameon=False, fontsize=max(font_size - 4, 8))

        if n_panels == 3:
            axes[2].bar(
                selected_attempt_counts.index.to_numpy(dtype=float),
                selected_attempt_counts.to_numpy(dtype=float),
                color="#4c78a8",
                width=0.7,
            )
            axes[2].set_xlabel("Sampling attempt")
            axes[2].set_ylabel("Selected polymers")
            axes[2].set_title("Final selected polymers by attempt")
        fig.tight_layout()
        fig.savefig(out_dir / "sampling_attempt_progress.png", dpi=dpi)
        plt.close(fig)

    stage_counts = [
        int(target_poly_summary.get("total_candidates_evaluated_after_target_aggregation", 0)),
        int(target_poly_summary.get("screen_after_valid_count", 0)),
        int(target_poly_summary.get("screen_after_target_stars_count", 0)),
        int(target_poly_summary.get("screen_after_novel_count", 0)),
        int(target_poly_summary.get("screen_after_sa_count", 0)),
        int(target_poly_summary.get("screen_after_target_requirements_count", 0)),
        int(target_poly_summary.get("filter_pass_unique", 0)),
        int(target_poly_summary.get("target_count_selected", 0)),
    ]
    if stage_counts[0] > 0:
        stage_labels = [
            "Screened polymers",
            "Valid SMILES",
            f"{target_stars} stars",
            "Novel vs train",
            f"SA < {target_sa_max:.1f}",
            "Pass class + property + solubility",
            "After deduplication",
            "Final selected",
        ]
        fig, ax = plt.subplots(figsize=(8.8, 4.6))
        ys = np.arange(len(stage_labels))
        bars = ax.barh(
            ys,
            stage_counts,
            color=sns.color_palette("Blues", n_colors=len(stage_labels)),
            edgecolor="none",
            height=0.6,
        )
        x_max = max(stage_counts)
        for bar, count in zip(bars, stage_counts):
            pct = 100.0 * float(count) / float(stage_counts[0]) if stage_counts[0] > 0 else 0.0
            ax.text(
                count + 0.02 * x_max,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({pct:.1f}%)",
                va="center",
                ha="left",
                fontsize=max(font_size - 3, 9),
            )
        ax.set_yticks(ys)
        ax.set_yticklabels(stage_labels)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_title("Step 6 target-polymer screening funnel")
        ax.set_xlim(0, x_max * 1.35)
        fig.tight_layout()
        fig.savefig(out_dir / "candidate_screening_funnel.png", dpi=dpi)
        plt.close(fig)

    if not candidate_df.empty and not target_poly_df.empty:
        joint_selected = (
            (target_poly_df["soluble_hit"] == 1)
            & (target_poly_df["property_hit"] == 1)
            & (target_poly_df["polymer_class_hit"] == 1)
        ).astype(int)
        rate_df = pd.DataFrame(
            [
                {"group": "Screened", "requirement": "Solubility", "pass_rate": float(candidate_df["soluble_hit"].mean())},
                {"group": "Screened", "requirement": "Property", "pass_rate": float(candidate_df["property_hit"].mean())},
                {"group": "Screened", "requirement": "Polymer class", "pass_rate": float(candidate_df["polymer_class_hit"].mean())},
                {"group": "Screened", "requirement": "Joint target", "pass_rate": float(candidate_df["joint_hit"].mean())},
                {"group": "Selected 100", "requirement": "Solubility", "pass_rate": float(target_poly_df["soluble_hit"].mean())},
                {"group": "Selected 100", "requirement": "Property", "pass_rate": float(target_poly_df["property_hit"].mean())},
                {"group": "Selected 100", "requirement": "Polymer class", "pass_rate": float(target_poly_df["polymer_class_hit"].mean())},
                {"group": "Selected 100", "requirement": "Joint target", "pass_rate": float(joint_selected.mean())},
            ]
        )
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        sns.barplot(data=rate_df, x="requirement", y="pass_rate", hue="group", palette=["#9ecae1", "#1f77b4"], ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Pass rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("Requirement pass rates: screened vs selected")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "requirement_pass_rates.png", dpi=dpi)
        plt.close(fig)

    if not target_poly_df.empty and "target_rank" in target_poly_df.columns:
        sel = target_poly_df.sort_values("target_rank").copy()
        prob_col = "class_prob" if "class_prob" in sel.columns else ("class_prob_lcb" if "class_prob_lcb" in sel.columns else None)

        plot_specs = []
        if {"chi_pred_target", "target_chi"}.issubset(sel.columns):
            plot_specs.append(
                {
                    "col": "chi_pred_target",
                    "ylabel": "Predicted χ",
                    "color": "#e45756",
                    "target_col": "target_chi",
                    "target_label": "Target χ",
                }
            )
        if prob_col is not None:
            plot_specs.append(
                {
                    "col": prob_col,
                    "ylabel": "Water-miscible probability",
                    "color": "#4c78a8",
                    "ref_line": 0.5,
                    "ref_label": "Miscible threshold",
                    "ylim": (0.0, 1.05),
                }
            )
        if "sa_score" in sel.columns:
            plot_specs.append(
                {
                    "col": "sa_score",
                    "ylabel": "SA score",
                    "color": "#54a24b",
                    "ref_line": float(target_sa_max),
                    "ref_label": f"SA limit ({target_sa_max:.1f})",
                }
            )

        if plot_specs:
            fig, axes = plt.subplots(len(plot_specs), 1, figsize=(9, 2.4 * len(plot_specs) + 1.2), sharex=True)
            axes = np.atleast_1d(axes)
            for ax, spec in zip(axes, plot_specs):
                x_vals = sel["target_rank"].to_numpy(dtype=float)
                y_vals = sel[spec["col"]].to_numpy(dtype=float)
                ax.plot(x_vals, y_vals, color=spec["color"], linewidth=1.6, alpha=0.8)
                ax.scatter(x_vals, y_vals, color=spec["color"], s=22, alpha=0.9)

                target_col = spec.get("target_col")
                if target_col is not None and target_col in sel.columns:
                    ax.plot(
                        x_vals,
                        sel[target_col].to_numpy(dtype=float),
                        color="#474747",
                        linewidth=1.0,
                        linestyle=":",
                        label=spec.get("target_label", "Target"),
                    )
                    ax.legend(frameon=False, fontsize=max(font_size - 4, 8), loc="best")

                ref_line = spec.get("ref_line")
                if ref_line is not None:
                    ax.axhline(
                        float(ref_line),
                        color="#474747",
                        linewidth=1.0,
                        linestyle=":",
                        label=spec.get("ref_label"),
                    )
                    ax.legend(frameon=False, fontsize=max(font_size - 4, 8), loc="best")

                if spec.get("ylim") is not None:
                    ax.set_ylim(*spec["ylim"])
                ax.set_ylabel(spec["ylabel"])

            axes[-1].set_xlabel("Selected polymer rank")
            fig.suptitle("Step 6 selected polymers: predicted values by rank", y=0.98)
            fig.tight_layout()
            fig.savefig(out_dir / "selected_target_requirements_by_rank.png", dpi=dpi)
            plt.close(fig)

    if not target_poly_df.empty and {"target_chi", "chi_pred_target"}.issubset(target_poly_df.columns):
        sel = target_poly_df.copy()
        if "target_rank" not in sel.columns:
            sel["target_rank"] = np.arange(1, len(sel) + 1, dtype=int)
        sel = sel.sort_values("target_rank").copy()
        fig, ax = plt.subplots(figsize=(5.8, 5.2))
        if "target_polymer_class" in sel.columns and sel["target_polymer_class"].nunique() > 1:
            sns.scatterplot(data=sel, x="target_chi", y="chi_pred_target", hue="target_polymer_class", s=60, ax=ax)
            ax.legend(frameon=False, fontsize=max(font_size - 4, 8))
        else:
            sns.scatterplot(data=sel, x="target_chi", y="chi_pred_target", color="#4c78a8", s=60, ax=ax)
        if "chi_pred_std_target" in sel.columns and np.nanmax(sel["chi_pred_std_target"].to_numpy(dtype=float)) > 0:
            ax.errorbar(
                sel["target_chi"].to_numpy(dtype=float),
                sel["chi_pred_target"].to_numpy(dtype=float),
                yerr=sel["chi_pred_std_target"].to_numpy(dtype=float),
                fmt="none",
                ecolor="#4c78a8",
                elinewidth=1.0,
                capsize=2,
                alpha=0.7,
            )
        lo = float(min(sel["target_chi"].min(), sel["chi_pred_target"].min()))
        hi = float(max(sel["target_chi"].max(), sel["chi_pred_target"].max()))
        pad = 0.05 * (hi - lo) if hi > lo else 0.1
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1)
        ax.set_xlabel("Target χ")
        ax.set_ylabel("Predicted χ")
        ax.set_title("Step 6 selected polymers: target vs predicted χ")
        fig.tight_layout()
        fig.savefig(out_dir / "selected_target_chi_parity.png", dpi=dpi)
        plt.close(fig)

        if "target_polymer_class" in sel.columns and sel["target_polymer_class"].nunique() > 1:
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            sns.boxplot(data=sel, x="target_polymer_class", y="chi_pred_target", color="#e45756", ax=ax)
            sns.stripplot(data=sel, x="target_polymer_class", y="chi_pred_target", color="#7f1d1d", size=3, alpha=0.4, ax=ax)
            ax.tick_params(axis="x", rotation=35)
        else:
            fig, ax = plt.subplots(figsize=(6.2, 4.8))
            sns.histplot(sel["chi_pred_target"].dropna(), bins=min(20, max(8, len(sel) // 6)), color="#e45756", ax=ax)
        target_vals = pd.to_numeric(sel["target_chi"], errors="coerce").dropna().to_numpy(dtype=float)
        if "target_polymer_class" in sel.columns and sel["target_polymer_class"].nunique() > 1:
            for idx, value in enumerate(np.unique(np.round(target_vals, 6))):
                ax.axhline(
                    float(value),
                    color="#474747",
                    linewidth=1.0,
                    linestyle=":",
                    label="Target χ" if idx == 0 else None,
                )
            ax.set_xlabel("Target polymer class")
            ax.set_ylabel("Predicted χ at target condition")
            ax.set_title("Step 6 selected polymers: predicted χ by target polymer class")
            if target_vals.size > 0:
                ax.legend(frameon=False)
        else:
            for idx, value in enumerate(np.unique(np.round(target_vals, 6))):
                ax.axvline(
                    float(value),
                    color="#474747",
                    linewidth=1.0,
                    linestyle=":",
                    label="Target χ" if idx == 0 else None,
                )
            ax.set_xlabel("Predicted χ at target condition")
            ax.set_ylabel("Count")
            ax.set_title("Step 6 selected polymers: predicted χ distribution")
            if target_vals.size > 0:
                ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "selected_target_chi_margin_distribution.png", dpi=dpi)
        plt.close(fig)

    prob_col = "class_prob" if "class_prob" in target_poly_df.columns else ("class_prob_lcb" if "class_prob_lcb" in target_poly_df.columns else None)
    if not target_poly_df.empty and prob_col is not None:
        sel = target_poly_df.copy()
        if "target_rank" not in sel.columns:
            sel["target_rank"] = np.arange(1, len(sel) + 1, dtype=int)
        sel = sel.sort_values("target_rank").copy()
        probability_label = "Predicted water-miscible probability" if prob_col == "class_prob" else "Estimated water-miscible probability"

        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        ax.plot(
            sel["target_rank"].to_numpy(dtype=float),
            sel[prob_col].to_numpy(dtype=float),
            marker="o",
            linewidth=1.8,
            color="#1f77b4",
        )
        ax.axhline(0.5, color="#474747", linewidth=1.0, linestyle=":")
        ax.set_xlabel("Selected polymer rank")
        ax.set_ylabel(probability_label)
        ax.set_ylim(0, 1.05)
        ax.set_title("Step 6 selected polymers: water-miscible probability by rank")
        fig.tight_layout()
        fig.savefig(out_dir / "selected_target_solubility_confidence_by_rank.png", dpi=dpi)
        plt.close(fig)

        if "target_polymer_class" in sel.columns and sel["target_polymer_class"].nunique() > 1:
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            sns.boxplot(data=sel, x="target_polymer_class", y=prob_col, color="#1f77b4", ax=ax)
            sns.stripplot(data=sel, x="target_polymer_class", y=prob_col, color="#0b1f33", size=3, alpha=0.45, ax=ax)
            ax.tick_params(axis="x", rotation=35)
            ax.axhline(0.5, color="#474747", linewidth=1.0, linestyle=":")
            ax.set_xlabel("Target polymer class")
            ax.set_ylabel(probability_label)
        else:
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            sns.histplot(
                sel[prob_col].dropna(),
                bins=min(20, max(8, len(sel) // 6)),
                color="#1f77b4",
                alpha=0.75,
                ax=ax,
            )
            ax.axvline(0.5, color="#474747", linewidth=1.0, linestyle=":")
            ax.set_xlabel(probability_label)
            ax.set_ylabel("Count")
        ax.set_title("Step 6 selected polymers: water-miscible probability distribution")
        fig.tight_layout()
        fig.savefig(out_dir / "selected_target_solubility_confidence_distribution.png", dpi=dpi)
        plt.close(fig)


def main(args):
    config = load_config(args.config)
    chi_cfg = default_chi_config(config, step="step6")

    split_mode = str(args.split_mode if args.split_mode is not None else chi_cfg["split_mode"]).strip().lower()
    if split_mode not in {"polymer", "random"}:
        raise ValueError("split_mode must be one of {'polymer','random'}")

    epsilon = float(args.epsilon if args.epsilon is not None else chi_cfg.get("epsilon", 0.05))
    legacy_class_weight_raw = args.class_weight if args.class_weight is not None else chi_cfg.get("class_weight", None)
    legacy_polymer_class_weight_raw = (
        args.polymer_class_weight if args.polymer_class_weight is not None else chi_cfg.get("polymer_class_weight", None)
    )
    legacy_class_weight = None if legacy_class_weight_raw is None else float(legacy_class_weight_raw)
    legacy_polymer_class_weight = (
        None if legacy_polymer_class_weight_raw is None else float(legacy_polymer_class_weight_raw)
    )
    uncertainty_enabled = bool(args.uncertainty_enabled or chi_cfg.get("uncertainty_enabled", False))
    uncertainty_mc_samples = int(
        args.uncertainty_mc_samples
        if args.uncertainty_mc_samples is not None
        else chi_cfg.get("uncertainty_mc_samples", 20)
    )
    uncertainty_class_z = float(
        args.uncertainty_class_z
        if args.uncertainty_class_z is not None
        else chi_cfg.get("uncertainty_class_z", 1.0)
    )
    uncertainty_property_z = float(
        args.uncertainty_property_z
        if args.uncertainty_property_z is not None
        else chi_cfg.get("uncertainty_property_z", 1.0)
    )
    uncertainty_score_weight = float(
        args.uncertainty_score_weight
        if args.uncertainty_score_weight is not None
        else chi_cfg.get("uncertainty_score_weight", 0.0)
    )
    uncertainty_seed = (
        int(args.uncertainty_seed)
        if args.uncertainty_seed is not None
        else int(chi_cfg.get("uncertainty_seed", config["data"]["random_seed"]))
    )
    candidate_source_value = args.candidate_source if args.candidate_source is not None else str(chi_cfg.get("candidate_source", "novel"))
    candidate_source = parse_candidate_source(candidate_source_value)
    args.candidate_source = candidate_source
    args.uncertainty_enabled = uncertainty_enabled
    args.uncertainty_mc_samples = uncertainty_mc_samples
    args.uncertainty_seed = uncertainty_seed
    default_property_rule = str(args.property_rule if args.property_rule is not None else chi_cfg.get("property_rule", "upper_bound")).strip().lower()
    coverage_topk = int(args.coverage_topk if args.coverage_topk is not None else chi_cfg.get("coverage_topk", 5))
    target_temperature_value = args.target_temperature if args.target_temperature is not None else chi_cfg.get("target_temperature", None)
    target_temperature = None if target_temperature_value is None else float(target_temperature_value)
    target_phi_value = args.target_phi if args.target_phi is not None else chi_cfg.get("target_phi", None)
    target_phi = None if target_phi_value is None else float(target_phi_value)
    sampling_cfg = config.get("sampling", {})
    target_polymer_count = int(chi_cfg.get("target_polymer_count", sampling_cfg.get("target_polymer_count", 100)))
    target_sa_max = float(chi_cfg.get("target_sa_max", sampling_cfg.get("target_sa_max", 4.0)))
    resampling_target_polymer_count = int(
        args.resampling_target_polymer_count
        if args.resampling_target_polymer_count is not None
        else chi_cfg.get("resampling_target_polymer_count", max(target_polymer_count * 10, 1000))
    )
    sampling_attempts_max = int(
        args.sampling_attempts_max
        if args.sampling_attempts_max is not None
        else chi_cfg.get("sampling_attempts_max", 5)
    )
    target_stars = int(sampling_cfg.get("target_stars", 2))
    decode_constraint_enabled = bool(chi_cfg.get("decode_constraint_enabled", False))
    if args.decode_constraint_enabled:
        decode_constraint_enabled = True
    if args.no_decode_constraint:
        decode_constraint_enabled = False
    decode_constraint_motif_bank_json = (
        Path(args.decode_constraint_motif_bank_json)
        if args.decode_constraint_motif_bank_json is not None
        else (
            Path(chi_cfg["decode_constraint_motif_bank_json"])
            if chi_cfg.get("decode_constraint_motif_bank_json")
            else None
        )
    )
    decode_constraint_max_motifs = int(
        args.decode_constraint_max_motifs
        if args.decode_constraint_max_motifs is not None
        else chi_cfg.get("decode_constraint_max_motifs", 6)
    )
    decode_constraint_center_min_frac = float(
        args.decode_constraint_center_min_frac
        if args.decode_constraint_center_min_frac is not None
        else chi_cfg.get("decode_constraint_center_min_frac", 0.25)
    )
    decode_constraint_center_max_frac = float(
        args.decode_constraint_center_max_frac
        if args.decode_constraint_center_max_frac is not None
        else chi_cfg.get("decode_constraint_center_max_frac", 0.75)
    )
    decode_constraint_enforce_class_match = bool(
        chi_cfg.get("decode_constraint_enforce_class_match", True)
    )
    if args.decode_constraint_disable_class_match_filter:
        decode_constraint_enforce_class_match = False
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")
    if legacy_class_weight is not None and legacy_class_weight < 0:
        raise ValueError("class_weight must be >= 0 when provided")
    if legacy_polymer_class_weight is not None and legacy_polymer_class_weight < 0:
        raise ValueError("polymer_class_weight must be >= 0 when provided")
    if uncertainty_mc_samples < 1:
        raise ValueError("uncertainty_mc_samples must be >= 1")
    if uncertainty_enabled and uncertainty_mc_samples < 2:
        raise ValueError("uncertainty_enabled=True requires uncertainty_mc_samples >= 2")
    if uncertainty_class_z < 0:
        raise ValueError("uncertainty_class_z must be >= 0")
    if uncertainty_property_z < 0:
        raise ValueError("uncertainty_property_z must be >= 0")
    if uncertainty_score_weight < 0:
        raise ValueError("uncertainty_score_weight must be >= 0")
    if default_property_rule not in {"band", "upper_bound", "lower_bound"}:
        raise ValueError("property_rule must be one of {'band', 'upper_bound', 'lower_bound'}")
    if coverage_topk < 1:
        raise ValueError("coverage_topk must be >= 1")
    if target_phi is not None and not (0.0 <= target_phi <= 1.0):
        raise ValueError("target_phi must be within [0, 1]")
    if target_polymer_count < 1:
        raise ValueError("target_polymer_count must be >= 1")
    if target_sa_max <= 0:
        raise ValueError("target_sa_max must be > 0")
    if resampling_target_polymer_count < target_polymer_count:
        raise ValueError("resampling_target_polymer_count must be >= target_polymer_count")
    if sampling_attempts_max < 1:
        raise ValueError("sampling_attempts_max must be >= 1")
    if decode_constraint_max_motifs < 1:
        raise ValueError("decode_constraint_max_motifs must be >= 1")
    if not (0.0 <= decode_constraint_center_min_frac <= decode_constraint_center_max_frac <= 1.0):
        raise ValueError(
            "decode_constraint_center_min_frac and decode_constraint_center_max_frac must satisfy 0 <= min <= max <= 1"
        )
    if legacy_class_weight is not None and abs(legacy_class_weight) > 1e-12:
        print(
            "Note: class_weight is deprecated and ignored. "
            "Step 6 now ranks by independent hard filters (soluble_hit, property_hit, polymer_class_hit) first."
        )
    if legacy_polymer_class_weight is not None and abs(legacy_polymer_class_weight) > 1e-12:
        print(
            "Note: polymer_class_weight is deprecated and ignored. "
            "Step 6 now ranks by independent hard filters (soluble_hit, property_hit, polymer_class_hit) first."
        )

    polymer_patterns = config.get("polymer_classes", {})
    if not polymer_patterns:
        raise ValueError("config.yaml polymer_classes is empty; Step 6 needs polymer family SMARTS patterns")
    available_classes = sorted([k.lower() for k in polymer_patterns.keys()])
    target_polymer_class_value = args.target_polymer_class if args.target_polymer_class is not None else str(chi_cfg.get("target_polymer_class", "all"))
    selected_classes = _parse_target_polymer_classes(target_polymer_class_value, available_classes)

    results_dir = Path(get_results_dir(args.model_size, config["paths"]["results_dir"], split_mode))
    results_dir_nosplit = Path(get_results_dir(args.model_size, config["paths"]["results_dir"], split_mode=None))
    base_results_dir = Path(config["paths"]["results_dir"])

    def _first_existing(paths: List[Path]) -> Path:
        for p in paths:
            if p.exists():
                return p
        return paths[0]

    if args.step4_reg_dir is not None:
        step4_reg_dir = Path(args.step4_reg_dir)
    else:
        if args.step4_dir is not None:
            step4_root = Path(args.step4_dir)
            reg_candidates = [
                step4_root / "step4_1_regression" / split_mode,
                step4_root / split_mode / "step4_1_regression",
                step4_root / "step4_1_regression",
            ]
        else:
            reg_candidates = [
                results_dir_nosplit / "step4_1_regression" / split_mode,
                results_dir / "step4_1_regression" / split_mode,
                results_dir_nosplit / "step4_chi_training" / "step4_1_regression" / split_mode,
                results_dir / "step4_chi_training" / split_mode / "step4_1_regression",
            ]
        step4_reg_dir = _first_existing(reg_candidates)

    if args.step4_cls_dir is not None:
        step4_cls_dir = Path(args.step4_cls_dir)
    else:
        if args.step4_dir is not None:
            step4_root = Path(args.step4_dir)
            cls_candidates = [
                step4_root / "step4_2_classification",
                step4_root / split_mode / "step4_2_classification",
            ]
        else:
            cls_candidates = [
                results_dir_nosplit / "step4_2_classification",
                results_dir / "step4_2_classification",
                results_dir_nosplit / "step4_chi_training" / "step4_2_classification",
                results_dir / "step4_chi_training" / split_mode / "step4_2_classification",
            ]
        step4_cls_dir = _first_existing(cls_candidates)

    step4_reg_metrics_dir = step4_reg_dir / "metrics"
    step4_cls_metrics_dir = step4_cls_dir / "metrics"

    step_dir = results_dir / "step6_polymer_class_water_soluble_inverse_design" / split_mode
    metrics_dir = step_dir / "metrics"
    figures_dir = step_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    resolved_decode_motifs: List[str] = []
    resolved_decode_source = None
    resolved_decode_motif_bank_json = None
    args.decode_constraint_class = None
    args.decode_constraint_motif_bank_json = None
    args.decode_constraint_center_min_frac = None
    args.decode_constraint_center_max_frac = None
    args.decode_constraint_enforce_class_match = False
    if decode_constraint_enabled:
        if len(selected_classes) != 1:
            raise ValueError(
                "Step 6 decode-time class constraints currently support exactly one target_polymer_class. "
                f"Requested: {selected_classes}"
            )
        tokenizer = PSmilesTokenizer.load(
            _resolve_step2_tokenizer_path(results_dir=results_dir, base_results_dir=base_results_dir)
        )
        configured_bank_path = (
            decode_constraint_motif_bank_json.resolve()
            if decode_constraint_motif_bank_json is not None
            else None
        )
        motif_source_smiles = load_decode_constraint_source_smiles(Path(config["paths"]["data_dir"]))
        resolved_decode_motifs, resolved_decode_source = resolve_class_decode_motifs(
            target_class=selected_classes[0],
            tokenizer=tokenizer,
            source_smiles=motif_source_smiles,
            patterns={str(k).strip().lower(): v for k, v in polymer_patterns.items()},
            configured_bank_path=configured_bank_path,
            max_motifs=decode_constraint_max_motifs,
        )
        resolved_decode_motif_bank_json = metrics_dir / "decode_constraint_motif_bank_resolved.json"
        with open(resolved_decode_motif_bank_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "target_class": selected_classes[0],
                    "motifs": resolved_decode_motifs,
                    "source": resolved_decode_source,
                },
                f,
                indent=2,
            )
        args.decode_constraint_class = selected_classes[0]
        args.decode_constraint_motif_bank_json = str(resolved_decode_motif_bank_json)
        args.decode_constraint_center_min_frac = float(decode_constraint_center_min_frac)
        args.decode_constraint_center_max_frac = float(decode_constraint_center_max_frac)
        args.decode_constraint_enforce_class_match = bool(decode_constraint_enforce_class_match)

    seed_info = seed_everything(int(config["data"]["random_seed"]))
    save_config(config, step_dir / "config_used.yaml")
    save_run_metadata(step_dir, args.config, seed_info)
    write_initial_log(
        step_dir=step_dir,
        step_name="step6_polymer_class_water_soluble_inverse_design",
        context={
            "config_path": args.config,
            "model_size": args.model_size,
            "results_dir": str(results_dir),
            "split_mode": split_mode,
            "candidate_source": candidate_source,
            "step4_regression_dir": str(step4_reg_dir),
            "step4_classification_dir": str(step4_cls_dir),
            "target_polymer_classes": ",".join(selected_classes),
            "epsilon": epsilon,
            "legacy_class_weight_ignored": legacy_class_weight,
            "legacy_polymer_class_weight_ignored": legacy_polymer_class_weight,
            "uncertainty_enabled": uncertainty_enabled,
            "uncertainty_mc_samples": uncertainty_mc_samples,
            "uncertainty_class_z": uncertainty_class_z,
            "uncertainty_property_z": uncertainty_property_z,
            "uncertainty_score_weight": uncertainty_score_weight,
            "uncertainty_seed": uncertainty_seed,
            "property_rule_default": default_property_rule,
            "coverage_topk": coverage_topk,
            "target_temperature": target_temperature,
            "target_phi": target_phi,
            "target_polymer_count": target_polymer_count,
            "target_sa_max": target_sa_max,
            "resampling_target_polymer_count": resampling_target_polymer_count,
            "target_stars": target_stars,
            "sampling_attempts_max": sampling_attempts_max,
            "decode_constraint_enabled": bool(decode_constraint_enabled),
            "decode_constraint_class": args.decode_constraint_class,
            "decode_constraint_motif_bank_json": args.decode_constraint_motif_bank_json,
            "decode_constraint_motif_count": int(len(resolved_decode_motifs)),
            "decode_constraint_source": resolved_decode_source,
            "decode_constraint_center_min_frac": decode_constraint_center_min_frac,
            "decode_constraint_center_max_frac": decode_constraint_center_max_frac,
            "decode_constraint_enforce_class_match": bool(decode_constraint_enforce_class_match),
            "step2_resampling_root": str(step_dir),
            "random_seed": config["data"]["random_seed"],
        },
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Step 6: polymer-class + water-soluble inverse design")
    print(f"split_mode={split_mode}")
    print(f"candidate_source={candidate_source}")
    print(f"target_polymer_class={','.join(selected_classes)}")
    print(f"epsilon={epsilon}")
    print(f"resampling_target_polymer_count={resampling_target_polymer_count}")
    print(f"sampling_attempts_max={sampling_attempts_max}")
    print(f"decode_constraint_enabled={decode_constraint_enabled}")
    if decode_constraint_enabled:
        print(
            "decode_constraint: "
            f"class={args.decode_constraint_class}, motifs={len(resolved_decode_motifs)}, "
            f"source={resolved_decode_source}, center={decode_constraint_center_min_frac:.2f}-{decode_constraint_center_max_frac:.2f}, "
            f"enforce_class_match={decode_constraint_enforce_class_match}"
        )
    if legacy_class_weight is not None:
        print(f"legacy_class_weight_ignored={legacy_class_weight}")
    if legacy_polymer_class_weight is not None:
        print(f"legacy_polymer_class_weight_ignored={legacy_polymer_class_weight}")
    print(f"uncertainty_enabled={uncertainty_enabled}")
    if uncertainty_enabled:
        print(
            "uncertainty: "
            f"mc_samples={uncertainty_mc_samples}, "
            f"class_z={uncertainty_class_z}, "
            f"property_z={uncertainty_property_z}, "
            f"score_weight={uncertainty_score_weight}"
        )
    print(f"target_temperature={target_temperature}")
    print(f"target_phi={target_phi}")
    print(f"device={device}")
    print("=" * 70)

    print("Preparing cached novelty reference and Step 4 inference models...")
    training_canonical = resolve_training_smiles(results_dir, base_results_dir)
    novel_inference_cache = prepare_novel_inference_cache(
        args=args,
        config=config,
        chi_cfg=chi_cfg,
        results_dir=results_dir,
        step4_reg_metrics_dir=step4_reg_metrics_dir,
        step4_cls_metrics_dir=step4_cls_metrics_dir,
        device=device,
        split_mode=split_mode,
    )

    base_target_df, target_path_used = load_soluble_targets(
        targets_csv=args.targets_csv,
        results_dir=results_dir,
        base_results_dir=base_results_dir,
        split_mode=split_mode,
        target_temperature=target_temperature,
        target_phi=target_phi,
    )
    target_df = _build_targets_for_step6(base_target_df, selected_classes)
    target_df.to_csv(metrics_dir / "inverse_targets.csv", index=False)

    accumulated_pools: List[pd.DataFrame] = []
    attempt_rows: List[Dict[str, object]] = []
    attempt_manifests: List[Dict[str, object]] = []
    score_outputs: Dict[str, object] | None = None
    coeff_df = pd.DataFrame()

    for attempt_idx in range(1, sampling_attempts_max + 1):
        attempt_resampling_dir = step_dir / f"step2_resampling_attempt_{attempt_idx:02d}"
        attempt_random_seed = int(config["data"]["random_seed"]) + attempt_idx - 1
        print(f"Sampling attempt {attempt_idx}/{sampling_attempts_max}...")
        attempt_coeff_df, attempt_pool_summary, attempt_training_canonical = build_candidate_pool(
            args=args,
            config=config,
            chi_cfg=chi_cfg,
            results_dir=results_dir,
            base_results_dir=base_results_dir,
            step4_reg_metrics_dir=step4_reg_metrics_dir,
            step4_cls_metrics_dir=step4_cls_metrics_dir,
            device=device,
            split_mode=split_mode,
            resampling_step_dir=attempt_resampling_dir,
            resampling_target_polymer_count=resampling_target_polymer_count,
            resampling_random_seed=attempt_random_seed,
            training_canonical=training_canonical,
            novel_inference_cache=novel_inference_cache,
        )
        training_canonical = attempt_training_canonical
        attempt_pool_summary = dict(attempt_pool_summary)
        attempt_pool_summary["sampling_attempt"] = attempt_idx
        attempt_pool_summary["sampling_random_seed"] = attempt_random_seed
        attempt_pool_summary["selected_polymer_classes"] = selected_classes
        attempt_pool_summary.update(
            load_step2_resampling_step_summary(attempt_pool_summary.get("step2_resampling_summary_csv"))
        )

        if not attempt_coeff_df.empty:
            attempt_coeff_df["sampling_attempt"] = attempt_idx
            attempt_coeff_df = _annotate_polymer_family_matches(
                attempt_coeff_df,
                patterns={k.lower(): v for k, v in polymer_patterns.items()},
            )
            accumulated_pools.append(attempt_coeff_df)
            coeff_df = _accumulate_candidate_pools(accumulated_pools)
            score_outputs = _score_candidate_pool(
                coeff_df=coeff_df,
                target_df=target_df,
                selected_classes=selected_classes,
                epsilon=epsilon,
                default_property_rule=default_property_rule,
                uncertainty_enabled=uncertainty_enabled,
                uncertainty_class_z=uncertainty_class_z,
                uncertainty_property_z=uncertainty_property_z,
                uncertainty_score_weight=uncertainty_score_weight,
                coverage_topk=coverage_topk,
                training_canonical=training_canonical,
                target_polymer_count=target_polymer_count,
                target_stars=target_stars,
                target_sa_max=target_sa_max,
                n_base_conditions=int(len(base_target_df)),
            )
            class_coverage = score_outputs["class_coverage"]
            target_poly_summary = score_outputs["target_poly_summary"]
        elif score_outputs is not None:
            class_coverage = score_outputs["class_coverage"]
            target_poly_summary = score_outputs["target_poly_summary"]
        else:
            class_coverage = _compute_class_coverage(coeff_df, selected_classes)
            target_poly_summary = {
                "n_polymers_pass_all_targets": 0,
                "n_polymers_pass_any_target_class": 0,
                "target_count_selected": 0,
                "target_count_requested": int(target_polymer_count),
            }

        attempt_pool_summary["candidate_polymer_class_hits"] = class_coverage
        attempt_pool_summary["accumulated_candidate_count"] = int(len(coeff_df))
        attempt_pool_summary["target_count_selected"] = int(target_poly_summary["target_count_selected"])
        attempt_manifests.append(attempt_pool_summary)

        row = {
            "sampling_attempt": int(attempt_idx),
            "attempt_candidate_count_after_dedup": int(len(attempt_coeff_df)),
            "accumulated_candidate_count": int(len(coeff_df)),
            "target_count_selected": int(target_poly_summary["target_count_selected"]),
            "n_polymers_pass_all_targets": int(target_poly_summary.get("n_polymers_pass_all_targets", 0)),
            "n_polymers_pass_any_target_class": int(target_poly_summary.get("n_polymers_pass_any_target_class", 0)),
            "sampling_random_seed": int(attempt_random_seed),
            "step2_generation_goal": attempt_pool_summary.get("step2_generation_goal"),
            "step2_generated_count_raw": attempt_pool_summary.get("step2_generated_count_raw"),
            "step2_accepted_count": attempt_pool_summary.get("step2_accepted_count"),
            "step2_valid_only_rounds": attempt_pool_summary.get("step2_valid_only_rounds"),
            "step2_valid_only_acceptance_rate": attempt_pool_summary.get("step2_valid_only_acceptance_rate"),
            "step2_valid_only_shortfall_count": attempt_pool_summary.get("step2_valid_only_shortfall_count"),
            "step2_valid_only_target_met": attempt_pool_summary.get("step2_valid_only_target_met"),
            "step2_sampling_time_sec": attempt_pool_summary.get("step2_sampling_time_sec"),
            "step2_samples_per_sec": attempt_pool_summary.get("step2_samples_per_sec"),
        }
        for cls in selected_classes:
            row[f"class_hits_{cls}"] = int(class_coverage.get(cls, 0))
        attempt_rows.append(row)

        class_msg = ", ".join(f"{cls}={class_coverage.get(cls, 0)}" for cls in selected_classes)
        print(
            "  accumulated candidates="
            f"{len(coeff_df)}, selected={target_poly_summary['target_count_selected']}/{target_polymer_count}, "
            f"class_hits[{class_msg}]"
        )
        if int(target_poly_summary["target_count_selected"]) >= int(target_polymer_count):
            print("  Sampling requirement met; stopping resampling loop.")
            break

    if training_canonical is None or score_outputs is None:
        raise RuntimeError("Step 6 failed to build any candidate pool from fresh Step 2 resampling.")

    class_coverage = score_outputs["class_coverage"]
    candidate_df = score_outputs["candidate_df"]
    target_metrics_df = score_outputs["target_metrics_df"]
    aggregate_df = score_outputs["aggregate_df"]
    post = score_outputs["post"]
    coverage = score_outputs["coverage"]
    target_poly_df = score_outputs["target_poly_df"]
    target_poly_summary = score_outputs["target_poly_summary"]

    if int(target_poly_summary["target_count_selected"]) == 0:
        coverage_msg = ", ".join(f"{cls}={class_coverage.get(cls, 0)}" for cls in selected_classes)
        raise RuntimeError(
            "Step 6 could not find any target polymers that satisfy the class + solubility + property requirements "
            f"after {len(attempt_rows)} fresh sampling attempts. class_hits[{coverage_msg}]"
        )

    pool_summary = {
        "candidate_source": candidate_source,
        "step4_regression_metrics_dir": str(step4_reg_metrics_dir),
        "step4_classification_metrics_dir": str(step4_cls_metrics_dir),
        "uncertainty_enabled": bool(uncertainty_enabled),
        "uncertainty_mc_samples": int(uncertainty_mc_samples),
        "uncertainty_seed": int(uncertainty_seed),
        "resampling_target_polymer_count": int(resampling_target_polymer_count),
        "sampling_attempts_max": int(sampling_attempts_max),
        "sampling_attempts_used": int(len(attempt_rows)),
        "n_generated_input_total": int(sum(int(m.get("n_generated_input", 0)) for m in attempt_manifests)),
        "n_valid_total": int(sum(int(m.get("n_valid", 0)) for m in attempt_manifests)),
        "n_valid_two_stars_total": int(sum(int(m.get("n_valid_two_stars", 0)) for m in attempt_manifests)),
        "novel_candidate_count_total_pre_dedup": int(sum(int(m.get("novel_candidate_count", 0)) for m in attempt_manifests)),
        "candidate_count_total_before_cross_attempt_dedup": int(
            sum(int(m.get("candidate_count_after_dedup", 0)) for m in attempt_manifests)
        ),
        "candidate_count_after_dedup": int(len(coeff_df)),
        "selected_polymer_classes": selected_classes,
        "candidate_polymer_class_hits": class_coverage,
        "decode_constraint_enabled": bool(decode_constraint_enabled),
        "decode_constraint_class": args.decode_constraint_class,
        "decode_constraint_motif_count": int(len(resolved_decode_motifs)),
        "decode_constraint_source": resolved_decode_source,
        "decode_constraint_motif_bank_json": None if resolved_decode_motif_bank_json is None else str(resolved_decode_motif_bank_json),
        "step2_resampling_step_dirs": [m.get("step2_resampling_step_dir", "") for m in attempt_manifests],
        "step2_resampling_generated_csvs": [m.get("step2_resampling_generated_csv", "") for m in attempt_manifests],
        "sampling_attempt_log_csv": str(metrics_dir / "sampling_attempts.csv"),
    }

    coeff_df.to_csv(metrics_dir / "inverse_candidate_pool.csv", index=False)
    with open(metrics_dir / "inverse_candidate_pool_summary.json", "w") as f:
        json.dump(pool_summary, f, indent=2)
    with open(metrics_dir / "sampling_attempt_manifest.json", "w") as f:
        json.dump(attempt_manifests, f, indent=2)
    sampling_attempts_df = pd.DataFrame(attempt_rows)
    sampling_attempts_df.to_csv(metrics_dir / "sampling_attempts.csv", index=False)

    candidate_df.to_csv(metrics_dir / "inverse_candidates_all.csv", index=False)
    target_metrics_df.to_csv(metrics_dir / "inverse_target_metrics.csv", index=False)
    aggregate_df.to_csv(metrics_dir / "inverse_aggregate_metrics.csv", index=False)
    post["by_condition"].to_csv(metrics_dir / "inverse_metrics_by_class_condition.csv", index=False)
    post["top1_sa"].to_csv(metrics_dir / "inverse_top1_sa_scores.csv", index=False)
    coverage.to_csv(metrics_dir / "inverse_polymer_coverage.csv", index=False)

    target_poly_df.to_csv(metrics_dir / "target_polymers.csv", index=False)
    _build_selected_target_candidate_ranked_df(target_poly_df).to_csv(
        metrics_dir / "selected_target_candidate_ranked.csv", index=False
    )
    _build_sampling_process_summary(
        attempt_rows=attempt_rows,
        target_poly_summary=target_poly_summary,
        resampling_target_polymer_count=resampling_target_polymer_count,
        sampling_attempts_max=sampling_attempts_max,
    ).to_csv(metrics_dir / "sampling_process_summary.csv", index=False)
    pd.DataFrame([target_poly_summary]).to_csv(metrics_dir / "target_polymer_selection_summary.csv", index=False)
    if target_poly_summary["target_count_selected"] < target_poly_summary["target_count_requested"]:
        print(
            "Warning: selected target polymers are fewer than requested. "
            f"selected={target_poly_summary['target_count_selected']}, "
            f"requested={target_poly_summary['target_count_requested']}"
        )

    overall_row = aggregate_df[aggregate_df["scope"] == "overall"].iloc[0].to_dict()
    target_success_rate = float(overall_row.get("target_success_rate", np.nan))
    screening_yield = float(target_poly_summary["selection_success_rate"])

    summary = {
        "n_targets": int(len(target_df)),
        "n_base_conditions": int(len(base_target_df)),
        "epsilon": epsilon,
        "legacy_class_weight_ignored": legacy_class_weight,
        "legacy_polymer_class_weight_ignored": legacy_polymer_class_weight,
        "uncertainty_enabled": bool(uncertainty_enabled),
        "uncertainty_mc_samples": int(uncertainty_mc_samples),
        "uncertainty_class_z": float(uncertainty_class_z),
        "uncertainty_property_z": float(uncertainty_property_z),
        "uncertainty_score_weight": float(uncertainty_score_weight),
        "uncertainty_seed": int(uncertainty_seed),
        "split_mode": split_mode,
        "candidate_source": candidate_source,
        "target_polymer_classes": selected_classes,
        "device": device,
        "targets_csv_used": target_path_used,
        "target_temperature": target_temperature,
        "target_phi": target_phi,
        "property_rule_default": default_property_rule,
        "coverage_topk": coverage_topk,
        "decode_constraint_enabled": bool(decode_constraint_enabled),
        "decode_constraint_class": args.decode_constraint_class,
        "decode_constraint_motif_count": int(len(resolved_decode_motifs)),
        "decode_constraint_source": resolved_decode_source,
        "decode_constraint_motif_bank_json": None if resolved_decode_motif_bank_json is None else str(resolved_decode_motif_bank_json),
        "decode_constraint_center_min_frac": float(decode_constraint_center_min_frac),
        "decode_constraint_center_max_frac": float(decode_constraint_center_max_frac),
        "decode_constraint_enforce_class_match": bool(decode_constraint_enforce_class_match),
        "target_polymer_count_requested": int(target_poly_summary["target_count_requested"]),
        "target_polymer_count_selected": int(target_poly_summary["target_count_selected"]),
        "target_polymer_selection_success_rate": target_success_rate,
        "target_polymer_screening_yield": screening_yield,
        "target_polymer_diversity": float(target_poly_summary["final_diversity"]),
        "target_polymer_mean_sa": float(target_poly_summary["final_mean_sa"]),
        "target_polymer_std_sa": float(target_poly_summary["final_std_sa"]),
        "target_polymer_mean_property_error": float(target_poly_summary["final_mean_property_error"]),
        "qualified_candidate_count": int(target_poly_summary.get("filter_pass_unique", 0)),
        "qualified_candidate_fraction_of_screened": (
            float(target_poly_summary.get("filter_pass_unique", 0) / target_poly_summary["total_candidates_screened"])
            if float(target_poly_summary.get("total_candidates_screened", 0)) > 0
            else np.nan
        ),
        "selected_fraction_of_qualified": (
            float(target_poly_summary["target_count_selected"] / target_poly_summary.get("filter_pass_unique", 0))
            if float(target_poly_summary.get("filter_pass_unique", 0)) > 0
            else np.nan
        ),
        **pool_summary,
    }
    if "chi_pred_std_target" in candidate_df.columns:
        summary["mean_candidate_pred_chi_std"] = float(np.nanmean(candidate_df["chi_pred_std_target"]))
    if "class_prob_std" in coeff_df.columns:
        summary["mean_candidate_class_prob_std"] = float(np.nanmean(coeff_df["class_prob_std"]))
    for key, value in overall_row.items():
        if isinstance(value, (np.floating, np.integer)):
            summary[key] = float(value)
        else:
            summary[key] = value

    if not post["top1_sa"].empty:
        summary["top1_sa_mean"] = float(np.nanmean(post["top1_sa"]["sa_score"]))
        summary["top1_sa_std"] = float(np.nanstd(post["top1_sa"]["sa_score"]))

    with open(metrics_dir / "step6_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    save_step_summary(summary, metrics_dir)
    _append_step_log(
        step_dir=step_dir,
        lines=[
            "final_target_polymer_selection:",
            f"total_sampling_examples: {target_poly_summary['total_candidates_screened']}",
            f"target_count_requested: {target_poly_summary['target_count_requested']}",
            f"target_count_selected: {target_poly_summary['target_count_selected']}",
            f"target_success_rate: {target_success_rate:.6f}",
            f"screening_yield: {screening_yield:.6f}",
            f"sampling_attempts_used: {len(attempt_rows)}",
            f"diversity: {target_poly_summary['final_diversity']:.6f}",
            f"mean_sa: {target_poly_summary['final_mean_sa']:.6f}",
            f"std_sa: {target_poly_summary['final_std_sa']:.6f}",
            f"mean_property_error: {target_poly_summary['final_mean_property_error']:.6f}",
            "target_csv: metrics/target_polymers.csv",
        ],
    )

    dpi = int(config.get("plotting", {}).get("dpi", 600))
    font_size = int(config.get("plotting", {}).get("font_size", 16))
    _save_figures(
        target_metrics_df=target_metrics_df,
        candidate_df=candidate_df,
        aggregate_df=aggregate_df,
        target_poly_df=target_poly_df,
        target_poly_summary=target_poly_summary,
        sampling_attempts_df=sampling_attempts_df,
        out_dir=figures_dir,
        dpi=dpi,
        font_size=font_size,
        epsilon=epsilon,
        target_sa_max=target_sa_max,
        target_stars=target_stars,
    )
    save_artifact_manifest(step_dir=step_dir, metrics_dir=metrics_dir, figures_dir=figures_dir)

    print("Step 6 complete.")
    print(f"Outputs: {step_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 6: polymer-class + water-soluble inverse design")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config path")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "medium", "large", "xl"], help="Step1 model size tag")
    parser.add_argument(
        "--split_mode",
        type=str,
        default=None,
        choices=["polymer", "random"],
        help="Optional split mode override (otherwise uses config chi_training.shared.split_mode).",
    )
    parser.add_argument("--step4_dir", type=str, default=None, help="Optional base path containing Step4_1 and Step4_2 directories")
    parser.add_argument("--step4_reg_dir", type=str, default=None, help="Optional explicit Step4_1 regression directory")
    parser.add_argument("--step4_cls_dir", type=str, default=None, help="Optional explicit Step4_2 classification directory")
    parser.add_argument("--step4_checkpoint", type=str, default=None, help="Optional explicit path to Step4 chi checkpoint")
    parser.add_argument("--step4_class_checkpoint", type=str, default=None, help="Optional explicit path to Step4 class checkpoint")
    parser.add_argument("--backbone_checkpoint", type=str, default=None, help="Optional explicit path to Step1 backbone checkpoint")

    parser.add_argument("--targets_csv", type=str, default=None, help="Custom χ_target CSV. If omitted, auto-uses Step 3 output.")
    parser.add_argument("--target_polymer_class", type=str, default=None, help="One class, comma list, or all")
    parser.add_argument("--property_rule", type=str, default=None, choices=["band", "upper_bound", "lower_bound"], help="Default property rule when targets file has none")
    parser.add_argument("--target_temperature", type=float, default=None, help="Optional target temperature filter (e.g., 293.15 for room temperature)")
    parser.add_argument("--target_phi", type=float, default=None, help="Optional target fraction filter (e.g., 0.2)")

    parser.add_argument(
        "--candidate_source",
        type=str,
        default=None,
        help="Candidate pool source. Only fresh Step 2 resampling is supported: novel (aliases: generated/step2).",
    )
    parser.add_argument(
        "--generated_csv",
        type=str,
        default=None,
        help="Unsupported override. Step 6 always launches a fresh Step 2 sampling run.",
    )
    parser.add_argument("--generated_smiles_column", type=str, default="smiles", help="SMILES column name in the fresh Step 2 samples CSV")
    parser.add_argument("--allow_non_two_stars", action="store_true", help="Allow generated candidates without exactly two '*' tokens")
    parser.add_argument("--max_novel_candidates", type=int, default=50000, help="Max number of novel generated candidates to keep")

    parser.add_argument("--epsilon", type=float, default=None, help="Property tolerance for 'band' rule")
    parser.add_argument(
        "--class_weight",
        type=float,
        default=None,
        help="Deprecated; ignored (ranking uses independent hard filters).",
    )
    parser.add_argument(
        "--polymer_class_weight",
        type=float,
        default=None,
        help="Deprecated; ignored (ranking uses independent hard filters).",
    )
    parser.add_argument("--uncertainty_enabled", action="store_true", help="Enable MC-dropout uncertainty-aware ranking")
    parser.add_argument("--uncertainty_mc_samples", type=int, default=None, help="MC-dropout forward passes when uncertainty is enabled")
    parser.add_argument("--uncertainty_class_z", type=float, default=None, help="z-value for conservative soluble confidence (class_prob - z*std)")
    parser.add_argument("--uncertainty_property_z", type=float, default=None, help="z-value for conservative property prediction bounds")
    parser.add_argument("--uncertainty_score_weight", type=float, default=None, help="Additional score penalty weight for predictive chi std")
    parser.add_argument("--uncertainty_seed", type=int, default=None, help="Random seed used for MC-dropout inference")
    parser.add_argument("--coverage_topk", type=int, default=None, help="Top-k used for coverage summary")
    parser.add_argument(
        "--resampling_target_polymer_count",
        type=int,
        default=None,
        help="Accepted Step 2 sample target per fresh resampling attempt (default: chi_training.step6_class_inverse_design.resampling_target_polymer_count or max(10x target_count, 1000))",
    )
    parser.add_argument(
        "--sampling_attempts_max",
        type=int,
        default=None,
        help="Maximum fresh Step 2 sampling attempts to accumulate before giving up (default: chi_training.step6_class_inverse_design.sampling_attempts_max or 5)",
    )
    parser.add_argument(
        "--decode_constraint_enabled",
        action="store_true",
        help="Enable Step 6-only decode-time class-constrained sampling during fresh Step 2 resampling",
    )
    parser.add_argument(
        "--no_decode_constraint",
        action="store_true",
        help="Disable decode-time class-constrained sampling even if config enables it",
    )
    parser.add_argument(
        "--decode_constraint_motif_bank_json",
        type=str,
        default=None,
        help="Optional JSON file containing class -> motif fragments for decode-time constraints",
    )
    parser.add_argument(
        "--decode_constraint_max_motifs",
        type=int,
        default=None,
        help="Maximum number of decode-time motif fragments to keep for the target class",
    )
    parser.add_argument(
        "--decode_constraint_center_min_frac",
        type=float,
        default=None,
        help="Lower bound for motif center placement as a fraction of sequence length",
    )
    parser.add_argument(
        "--decode_constraint_center_max_frac",
        type=float,
        default=None,
        help="Upper bound for motif center placement as a fraction of sequence length",
    )
    parser.add_argument(
        "--decode_constraint_disable_class_match_filter",
        action="store_true",
        help="Disable exact class-match filtering during constrained Step 2 resampling",
    )

    parser.add_argument("--embedding_pooling", type=str, default="mean", choices=["mean", "cls", "max"], help="Pooling for embedding extraction on novel candidates")
    parser.add_argument("--embedding_batch_size", type=int, default=None, help="Batch size for novel embedding extraction")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    args = parser.parse_args()
    main(args)
