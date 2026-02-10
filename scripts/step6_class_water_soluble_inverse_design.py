#!/usr/bin/env python
"""Step 6: polymer-family class + water-soluble inverse design on chi(T,phi)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chi.inverse_design_common import (
    build_candidate_pool,
    default_chi_config,
    load_soluble_targets,
    parse_candidate_source,
    set_plot_style,
)
from src.chi.model import predict_chi_from_coefficients
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


def _compute_target_candidates(
    target_row: pd.Series,
    coeff_df: pd.DataFrame,
    epsilon: float,
    class_weight: float,
    polymer_class_weight: float,
    default_property_rule: str,
) -> pd.DataFrame:
    target_chi = float(target_row["target_chi"])
    t = float(target_row["temperature"])
    phi = float(target_row["phi"])
    target_polymer_class = str(target_row["target_polymer_class"]).strip().lower()

    class_col = f"polymer_class_{target_polymer_class}"
    required_cols = ["polymer_id", "Polymer", "SMILES", "class_prob", "a0", "a1", "a2", "a3", "b1", "b2", class_col]
    missing = [c for c in required_cols if c not in coeff_df.columns]
    if missing:
        raise ValueError(f"Candidate pool missing required columns: {missing}")

    copy_cols = required_cols + [
        c for c in ["water_soluble", "candidate_source", "canonical_smiles", "class_logit", "is_novel_vs_train"] if c in coeff_df.columns
    ]
    out = coeff_df[copy_cols].copy()

    coeff = out[["a0", "a1", "a2", "a3", "b1", "b2"]].to_numpy(dtype=float)
    pred = predict_chi_from_coefficients(coeff, np.full(len(out), t), np.full(len(out), phi))

    out["target_id"] = int(target_row["target_id"])
    out["temperature"] = t
    out["phi"] = phi
    out["target_chi"] = target_chi
    out["target_polymer_class"] = target_polymer_class
    out["chi_pred_target"] = pred
    out["chi_error"] = out["chi_pred_target"] - target_chi
    out["abs_error"] = np.abs(out["chi_error"])

    property_rule = str(target_row.get("property_rule", default_property_rule)).strip().lower()
    if property_rule not in {"band", "upper_bound", "lower_bound"}:
        property_rule = default_property_rule
    out["property_rule"] = property_rule

    out["soluble_confidence"] = out["class_prob"]
    out["pred_soluble"] = (out["class_prob"] >= 0.5).astype(int)
    out["soluble_hit"] = out["pred_soluble"].astype(int)

    out["polymer_class_hit"] = out[class_col].astype(int)

    if property_rule == "upper_bound":
        out["property_error"] = np.maximum(out["chi_pred_target"] - target_chi, 0.0)
        out["property_hit"] = (out["chi_pred_target"] <= target_chi).astype(int)
    elif property_rule == "lower_bound":
        out["property_error"] = np.maximum(target_chi - out["chi_pred_target"], 0.0)
        out["property_hit"] = (out["chi_pred_target"] >= target_chi).astype(int)
    else:
        out["property_error"] = out["abs_error"]
        out["property_hit"] = (out["abs_error"] <= float(epsilon)).astype(int)

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
        + float(class_weight) * (1.0 - out["soluble_confidence"])
        + float(polymer_class_weight) * (1.0 - out["polymer_class_hit"])
    )

    out = out.sort_values(
        [
            "step4_requirement_miss_count",
            "total_requirement_miss_count",
            "score",
            "polymer_class_hit",
            "property_error",
            "soluble_confidence",
        ],
        ascending=[True, True, True, False, True, False],
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
        "top1_soluble_hit": int(cand["soluble_hit"].iloc[0]),
        "top1_polymer_class_hit": int(cand["polymer_class_hit"].iloc[0]),
        "top1_property_hit": int(cand["property_hit"].iloc[0]),
        "top1_joint_hit": int(cand["joint_hit"].iloc[0]),
        "top1_property_error": float(cand["property_error"].iloc[0]),
        "top1_abs_error": float(cand["abs_error"].iloc[0]),
        "top1_pred_chi": float(cand["chi_pred_target"].iloc[0]),
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


def _select_final_target_polymers(
    candidate_df: pd.DataFrame,
    training_canonical: set[str],
    polymer_family_map: Dict[int, str],
    target_count: int,
    target_stars: int,
    sa_max: float,
    total_sampling_points: int | None = None,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    ranked = candidate_df.sort_values(
        [
            "step4_requirement_miss_count",
            "total_requirement_miss_count",
            "score",
            "rank",
            "polymer_class_hit",
            "property_error",
            "class_prob",
        ],
        ascending=[True, True, True, True, False, True, False],
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
    per_polymer["n_targets_required"] = int(required_targets)
    per_polymer["passes_all_target_conditions"] = (
        (per_polymer["n_targets_evaluated"] == int(required_targets))
        & (per_polymer["n_targets_joint_hit"] == int(required_targets))
    ).astype(int)

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
                "temperature": float(row["temperature"]),
                "phi": float(row["phi"]),
                "target_chi": float(row["target_chi"]),
                "chi_pred_target": float(row["chi_pred_target"]),
                "property_error": float(row["property_error"]),
                "abs_error": float(row["abs_error"]),
                "class_prob": float(row["class_prob"]),
                "soluble_hit": int(row["soluble_hit"]),
                "score": float(row["score"]),
                "rank_in_target": int(row["rank"]),
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
                "passes_all_target_conditions": int(row["passes_all_target_conditions"]) if not pd.isna(row.get("passes_all_target_conditions", np.nan)) else 0,
                "mean_property_error_all_targets": float(row["mean_property_error_all_targets"]) if not pd.isna(row.get("mean_property_error_all_targets", np.nan)) else np.nan,
                "max_property_error_all_targets": float(row["max_property_error_all_targets"]) if not pd.isna(row.get("max_property_error_all_targets", np.nan)) else np.nan,
            }
        )

    all_df = pd.DataFrame(rows)
    real_total_sampling_points = int(total_sampling_points) if total_sampling_points is not None else int(len(all_df))
    if all_df.empty:
        return pd.DataFrame(), {
            "required_targets_per_polymer": int(required_targets),
            "n_polymers_pass_all_targets": 0,
            "target_count_requested": int(target_count),
            "total_candidates_screened": int(real_total_sampling_points),
            "total_candidates_evaluated_after_target_aggregation": 0,
            "filter_pass_count": 0,
            "filter_pass_unique": 0,
            "target_count_selected": 0,
            "selection_success_rate": 0.0,
            "final_diversity": 0.0,
            "final_mean_sa": np.nan,
            "final_std_sa": np.nan,
            "final_mean_property_error": np.nan,
        }

    filter_mask = (
        (all_df["is_valid"] == 1)
        & (all_df["star_count"] == int(target_stars))
        & (all_df["is_novel_vs_train"] == 1)
        & (all_df["sa_ok"] == 1)
        & (all_df["passes_all_target_conditions"] == 1)
    )
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
        "required_targets_per_polymer": int(required_targets),
        "n_polymers_pass_all_targets": int(all_df["passes_all_target_conditions"].sum()),
        "target_count_requested": int(target_count),
        "total_candidates_screened": int(real_total_sampling_points),
        "total_candidates_evaluated_after_target_aggregation": int(len(all_df)),
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


def _save_figures(
    target_metrics_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    font_size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style(font_size)

    # Top-k success curve
    curve_rows = []
    overall = aggregate_df[aggregate_df["scope"] == "overall"]
    if not overall.empty:
        row = overall.iloc[0]
        for k in K_LIST:
            curve_rows.append({"k": k, "success_rate": row[f"target_success_top{k}"]})
    curve_df = pd.DataFrame(curve_rows)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.lineplot(data=curve_df, x="k", y="success_rate", marker="o", linewidth=2, ax=ax)
    ax.set_xlabel("Top-k")
    ax.set_ylabel("Target success rate")
    ax.set_title("Step 6 top-k target success")
    fig.tight_layout()
    fig.savefig(out_dir / "topk_target_success_curve.png", dpi=dpi)
    plt.close(fig)

    # Success by polymer class
    class_df = aggregate_df[aggregate_df["scope"].str.startswith("polymer_class_")].copy()
    class_df["target_polymer_class"] = class_df["scope"].str.replace("polymer_class_", "", regex=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=class_df, x="target_polymer_class", y="target_success_rate", color="#4c78a8", ax=ax)
    ax.set_xlabel("Target polymer class")
    ax.set_ylabel("Target success rate")
    ax.set_title("Target success by polymer class")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_dir / "target_success_by_polymer_class.png", dpi=dpi)
    plt.close(fig)

    # Top1 error distribution by class
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=target_metrics_df, x="target_polymer_class", y="top1_property_error", ax=ax)
    ax.set_xlabel("Target polymer class")
    ax.set_ylabel("Top-1 property error")
    ax.set_title("Top-1 error by polymer class")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_dir / "top1_error_by_polymer_class.png", dpi=dpi)
    plt.close(fig)

    # Heatmap (class x condition)
    target_metrics_df = target_metrics_df.copy()
    target_metrics_df["condition"] = target_metrics_df.apply(lambda r: f"{r['temperature']:.2f}K|ϕ={r['phi']:.1f}", axis=1)
    heat = target_metrics_df.pivot_table(index="target_polymer_class", columns="condition", values="top1_joint_hit", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heat, cmap="YlGnBu", vmin=0, vmax=1, annot=True, fmt=".2f", cbar_kws={"label": "Top-1 joint hit"}, ax=ax)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Target polymer class")
    ax.set_title("Top-1 joint hit map")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(out_dir / "top1_joint_hit_class_condition_heatmap.png", dpi=dpi)
    plt.close(fig)

    # Top1 confidence vs error
    top1 = candidate_df[candidate_df["rank"] == 1].copy()
    if not top1.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=top1,
            x="property_error",
            y="class_prob",
            hue="target_polymer_class",
            style="polymer_class_hit",
            s=70,
            ax=ax,
        )
        ax.set_xlabel("Top-1 property error")
        ax.set_ylabel("Top-1 soluble confidence")
        ax.set_title("Top-1 confidence vs error")
        fig.tight_layout()
        fig.savefig(out_dir / "top1_confidence_vs_error.png", dpi=dpi)
        plt.close(fig)


def main(args):
    config = load_config(args.config)
    chi_cfg = default_chi_config(config, step="step6")

    split_mode = str(chi_cfg["split_mode"]).strip().lower()
    if split_mode not in {"polymer", "random"}:
        raise ValueError("split_mode must be one of {'polymer','random'}")

    epsilon = float(args.epsilon if args.epsilon is not None else chi_cfg.get("epsilon", 0.05))
    class_weight = float(args.class_weight if args.class_weight is not None else chi_cfg.get("class_weight", 0.25))
    polymer_class_weight = float(
        args.polymer_class_weight if args.polymer_class_weight is not None else chi_cfg.get("polymer_class_weight", 0.50)
    )
    candidate_source_value = args.candidate_source if args.candidate_source is not None else str(chi_cfg.get("candidate_source", "novel"))
    candidate_source = parse_candidate_source(candidate_source_value)
    args.candidate_source = candidate_source
    default_property_rule = str(args.property_rule if args.property_rule is not None else chi_cfg.get("property_rule", "upper_bound")).strip().lower()
    coverage_topk = int(args.coverage_topk if args.coverage_topk is not None else chi_cfg.get("coverage_topk", 5))
    sampling_cfg = config.get("sampling", {})
    target_polymer_count = int(chi_cfg.get("target_polymer_count", sampling_cfg.get("target_polymer_count", 100)))
    target_sa_max = float(chi_cfg.get("target_sa_max", sampling_cfg.get("target_sa_max", 4.0)))
    target_stars = int(sampling_cfg.get("target_stars", 2))
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")
    if class_weight < 0:
        raise ValueError("class_weight must be >= 0")
    if polymer_class_weight < 0:
        raise ValueError("polymer_class_weight must be >= 0")
    if default_property_rule not in {"band", "upper_bound", "lower_bound"}:
        raise ValueError("property_rule must be one of {'band', 'upper_bound', 'lower_bound'}")
    if coverage_topk < 1:
        raise ValueError("coverage_topk must be >= 1")
    if target_polymer_count < 1:
        raise ValueError("target_polymer_count must be >= 1")
    if target_sa_max <= 0:
        raise ValueError("target_sa_max must be > 0")

    polymer_patterns = config.get("polymer_classes", {})
    if not polymer_patterns:
        raise ValueError("config.yaml polymer_classes is empty; Step 6 needs polymer family SMARTS patterns")
    available_classes = sorted([k.lower() for k in polymer_patterns.keys()])
    target_polymer_class_value = args.target_polymer_class if args.target_polymer_class is not None else str(chi_cfg.get("target_polymer_class", "all"))
    selected_classes = _parse_target_polymer_classes(target_polymer_class_value, available_classes)

    results_dir = Path(get_results_dir(args.model_size, config["paths"]["results_dir"]))
    base_results_dir = Path(config["paths"]["results_dir"])
    step4_base_dir = Path(args.step4_dir) if args.step4_dir else results_dir / "step4_chi_training" / split_mode
    step4_reg_dir = Path(args.step4_reg_dir) if args.step4_reg_dir else step4_base_dir / "step4_1_regression"
    step4_cls_dir = Path(args.step4_cls_dir) if args.step4_cls_dir else step4_base_dir / "step4_2_classification"
    step4_reg_metrics_dir = step4_reg_dir / "metrics"
    step4_cls_metrics_dir = step4_cls_dir / "metrics"

    step_dir = results_dir / "step6_polymer_class_water_soluble_inverse_design" / split_mode
    metrics_dir = step_dir / "metrics"
    figures_dir = step_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

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
            "class_weight": class_weight,
            "polymer_class_weight": polymer_class_weight,
            "property_rule_default": default_property_rule,
            "coverage_topk": coverage_topk,
            "target_polymer_count": target_polymer_count,
            "target_sa_max": target_sa_max,
            "target_stars": target_stars,
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
    print(f"class_weight={class_weight}")
    print(f"polymer_class_weight={polymer_class_weight}")
    print(f"device={device}")
    print("=" * 70)

    base_target_df, target_path_used = load_soluble_targets(
        targets_csv=args.targets_csv,
        results_dir=results_dir,
        base_results_dir=base_results_dir,
        split_mode=split_mode,
    )
    target_df = _build_targets_for_step6(base_target_df, selected_classes)
    target_df.to_csv(metrics_dir / "inverse_targets.csv", index=False)

    coeff_df, pool_summary, training_canonical = build_candidate_pool(
        args=args,
        config=config,
        chi_cfg=chi_cfg,
        results_dir=results_dir,
        base_results_dir=base_results_dir,
        step4_reg_metrics_dir=step4_reg_metrics_dir,
        step4_cls_metrics_dir=step4_cls_metrics_dir,
        device=device,
    )
    if coeff_df.empty:
        raise RuntimeError("Candidate pool is empty after filtering. Relax filters or generate more samples.")

    coeff_df = _annotate_polymer_family_matches(coeff_df, patterns={k.lower(): v for k, v in polymer_patterns.items()})
    coeff_df.to_csv(metrics_dir / "inverse_candidate_pool.csv", index=False)
    polymer_family_map = _build_polymer_family_map(coeff_df)

    class_coverage = {
        cls: int(coeff_df[f"polymer_class_{cls}"].sum())
        for cls in selected_classes
    }
    pool_summary["selected_polymer_classes"] = selected_classes
    pool_summary["candidate_polymer_class_hits"] = class_coverage

    with open(metrics_dir / "inverse_candidate_pool_summary.json", "w") as f:
        json.dump(pool_summary, f, indent=2)

    # Per-target candidate ranking
    all_candidates = []
    target_metrics = []
    for _, row in target_df.iterrows():
        cand = _compute_target_candidates(
            target_row=row,
            coeff_df=coeff_df,
            epsilon=epsilon,
            class_weight=class_weight,
            polymer_class_weight=polymer_class_weight,
            default_property_rule=default_property_rule,
        )
        all_candidates.append(cand)
        target_metrics.append(_target_metrics(cand))

    candidate_df = pd.concat(all_candidates, ignore_index=True)
    target_metrics_df = pd.DataFrame(target_metrics)
    aggregate_df = _aggregate_metrics(target_metrics_df)

    candidate_df.to_csv(metrics_dir / "inverse_candidates_all.csv", index=False)
    target_metrics_df.to_csv(metrics_dir / "inverse_target_metrics.csv", index=False)
    aggregate_df.to_csv(metrics_dir / "inverse_aggregate_metrics.csv", index=False)

    post = _postprocess_metrics(candidate_df, target_metrics_df)
    post["by_condition"].to_csv(metrics_dir / "inverse_metrics_by_class_condition.csv", index=False)
    post["top1_sa"].to_csv(metrics_dir / "inverse_top1_sa_scores.csv", index=False)

    topk = candidate_df[candidate_df["rank"] <= coverage_topk].copy()
    coverage = (
        topk.groupby(["target_polymer_class", "candidate_source", "polymer_id", "Polymer"], as_index=False)
        .agg(
            selected_count=("rank", "size"),
            mean_class_prob=("class_prob", "mean"),
            mean_property_error=("property_error", "mean"),
            mean_abs_error=("abs_error", "mean"),
            mean_novel_rate=("is_novel_vs_train", "mean"),
            mean_polymer_class_hit=("polymer_class_hit", "mean"),
        )
        .sort_values(["target_polymer_class", "selected_count"], ascending=[True, False])
    )
    coverage["selected_rate"] = coverage["selected_count"] / float(len(base_target_df))
    coverage.to_csv(metrics_dir / "inverse_polymer_coverage.csv", index=False)

    target_poly_df, target_poly_summary = _select_final_target_polymers(
        candidate_df=candidate_df,
        training_canonical=training_canonical,
        polymer_family_map=polymer_family_map,
        target_count=target_polymer_count,
        target_stars=target_stars,
        sa_max=target_sa_max,
        total_sampling_points=int(len(coeff_df)),
    )
    target_poly_df.to_csv(metrics_dir / "target_polymers.csv", index=False)
    pd.DataFrame([target_poly_summary]).to_csv(metrics_dir / "target_polymer_selection_summary.csv", index=False)
    if target_poly_summary["target_count_selected"] < target_poly_summary["target_count_requested"]:
        print(
            "Warning: selected target polymers are fewer than requested. "
            f"selected={target_poly_summary['target_count_selected']}, "
            f"requested={target_poly_summary['target_count_requested']}"
        )

    summary = {
        "n_targets": int(len(target_df)),
        "n_base_conditions": int(len(base_target_df)),
        "epsilon": epsilon,
        "class_weight": class_weight,
        "polymer_class_weight": polymer_class_weight,
        "split_mode": split_mode,
        "candidate_source": candidate_source,
        "target_polymer_classes": selected_classes,
        "device": device,
        "targets_csv_used": target_path_used,
        "property_rule_default": default_property_rule,
        "coverage_topk": coverage_topk,
        "target_polymer_count_requested": int(target_poly_summary["target_count_requested"]),
        "target_polymer_count_selected": int(target_poly_summary["target_count_selected"]),
        "target_polymer_selection_success_rate": float(target_poly_summary["selection_success_rate"]),
        "target_polymer_diversity": float(target_poly_summary["final_diversity"]),
        "target_polymer_mean_sa": float(target_poly_summary["final_mean_sa"]),
        "target_polymer_std_sa": float(target_poly_summary["final_std_sa"]),
        "target_polymer_mean_property_error": float(target_poly_summary["final_mean_property_error"]),
        **pool_summary,
    }
    overall_row = aggregate_df[aggregate_df["scope"] == "overall"].iloc[0].to_dict()
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
            f"success_rate: {target_poly_summary['selection_success_rate']:.6f}",
            f"diversity: {target_poly_summary['final_diversity']:.6f}",
            f"mean_sa: {target_poly_summary['final_mean_sa']:.6f}",
            f"std_sa: {target_poly_summary['final_std_sa']:.6f}",
            f"mean_property_error: {target_poly_summary['final_mean_property_error']:.6f}",
            "target_csv: metrics/target_polymers.csv",
        ],
    )

    dpi = int(config.get("plotting", {}).get("dpi", 600))
    font_size = int(config.get("plotting", {}).get("font_size", 12))
    _save_figures(
        target_metrics_df=target_metrics_df,
        candidate_df=candidate_df,
        aggregate_df=aggregate_df,
        out_dir=figures_dir,
        dpi=dpi,
        font_size=font_size,
    )
    save_artifact_manifest(step_dir=step_dir, metrics_dir=metrics_dir, figures_dir=figures_dir)

    print("Step 6 complete.")
    print(f"Outputs: {step_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 6: polymer-class + water-soluble inverse design")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config path")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "medium", "large", "xl"], help="Step1 model size tag")
    parser.add_argument("--step4_dir", type=str, default=None, help="Optional base path containing Step4_1 and Step4_2 directories")
    parser.add_argument("--step4_reg_dir", type=str, default=None, help="Optional explicit Step4_1 regression directory")
    parser.add_argument("--step4_cls_dir", type=str, default=None, help="Optional explicit Step4_2 classification directory")
    parser.add_argument("--step4_checkpoint", type=str, default=None, help="Optional explicit path to Step4 chi checkpoint")
    parser.add_argument("--step4_class_checkpoint", type=str, default=None, help="Optional explicit path to Step4 class checkpoint")
    parser.add_argument("--backbone_checkpoint", type=str, default=None, help="Optional explicit path to Step1 backbone checkpoint")

    parser.add_argument("--targets_csv", type=str, default=None, help="Custom χ_target CSV. If omitted, auto-uses Step 3 output.")
    parser.add_argument("--target_polymer_class", type=str, default=None, help="One class, comma list, or all")
    parser.add_argument("--property_rule", type=str, default=None, choices=["band", "upper_bound", "lower_bound"], help="Default property rule when targets file has none")

    parser.add_argument(
        "--candidate_source",
        type=str,
        default=None,
        help="Only novel/generated/step2 are supported for Step 6.",
    )
    parser.add_argument("--generated_csv", type=str, default=None, help="Generated samples CSV from Step 2")
    parser.add_argument("--generated_smiles_column", type=str, default="smiles", help="SMILES column name in generated_csv")
    parser.add_argument("--allow_non_two_stars", action="store_true", help="Allow generated candidates without exactly two '*' tokens")
    parser.add_argument("--max_novel_candidates", type=int, default=50000, help="Max number of novel generated candidates to keep")

    parser.add_argument("--epsilon", type=float, default=None, help="Property tolerance for 'band' rule")
    parser.add_argument("--class_weight", type=float, default=None, help="Weight for soluble-confidence penalty in score")
    parser.add_argument("--polymer_class_weight", type=float, default=None, help="Weight for polymer-class mismatch penalty")
    parser.add_argument("--coverage_topk", type=int, default=None, help="Top-k used for coverage summary")

    parser.add_argument("--embedding_pooling", type=str, default="mean", choices=["mean", "cls", "max"], help="Pooling for embedding extraction on novel candidates")
    parser.add_argument("--embedding_batch_size", type=int, default=None, help="Batch size for novel embedding extraction")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    args = parser.parse_args()
    main(args)
