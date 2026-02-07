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
    check_required_step4_files,
    default_chi_config,
    load_soluble_targets,
    parse_candidate_source,
    set_plot_style,
)
from src.chi.model import predict_chi_from_coefficients
from src.evaluation.polymer_class import PolymerClassifier
from src.utils.chemistry import compute_sa_score
from src.utils.config import load_config, save_config
from src.utils.model_scales import get_results_dir
from src.utils.reproducibility import save_run_metadata, seed_everything


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
        ["score", "polymer_class_hit", "property_error", "soluble_confidence"],
        ascending=[True, False, True, False],
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
    chi_cfg = default_chi_config(config)

    split_mode = (args.split_mode or chi_cfg["split_mode"]).strip().lower()
    if split_mode not in {"polymer", "random"}:
        raise ValueError("split_mode must be one of {'polymer','random'}")

    epsilon = float(args.epsilon if args.epsilon is not None else chi_cfg.get("epsilon", 0.05))
    class_weight = float(args.class_weight if args.class_weight is not None else chi_cfg.get("class_weight", 0.25))
    polymer_class_weight = float(
        args.polymer_class_weight if args.polymer_class_weight is not None else chi_cfg.get("polymer_class_weight", 0.50)
    )
    candidate_source = parse_candidate_source(args.candidate_source)

    polymer_patterns = config.get("polymer_classes", {})
    if not polymer_patterns:
        raise ValueError("config.yaml polymer_classes is empty; Step 6 needs polymer family SMARTS patterns")
    available_classes = sorted([k.lower() for k in polymer_patterns.keys()])
    selected_classes = _parse_target_polymer_classes(args.target_polymer_class, available_classes)

    results_dir = Path(get_results_dir(args.model_size, config["paths"]["results_dir"]))
    base_results_dir = Path(config["paths"]["results_dir"])
    step4_dir = Path(args.step4_dir) if args.step4_dir else results_dir / "step4_chi_training" / split_mode
    step4_metrics_dir = step4_dir / "metrics"
    check_required_step4_files(step4_metrics_dir)

    step_dir = results_dir / "step6_polymer_class_water_soluble_inverse_design" / split_mode
    metrics_dir = step_dir / "metrics"
    figures_dir = step_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    seed_info = seed_everything(int(config["data"]["random_seed"]))
    save_config(config, step_dir / "config_used.yaml")
    save_run_metadata(step_dir, args.config, seed_info)

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
        split_mode=split_mode,
    )
    target_df = _build_targets_for_step6(base_target_df, selected_classes)
    target_df.to_csv(metrics_dir / "inverse_targets.csv", index=False)

    coeff_df, pool_summary, _ = build_candidate_pool(
        args=args,
        config=config,
        chi_cfg=chi_cfg,
        results_dir=results_dir,
        base_results_dir=base_results_dir,
        step4_metrics_dir=step4_metrics_dir,
        device=device,
    )
    if coeff_df.empty:
        raise RuntimeError("Candidate pool is empty after filtering. Relax filters or generate more samples.")

    coeff_df = _annotate_polymer_family_matches(coeff_df, patterns={k.lower(): v for k, v in polymer_patterns.items()})
    coeff_df.to_csv(metrics_dir / "inverse_candidate_pool.csv", index=False)

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
            default_property_rule=args.property_rule,
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

    topk = candidate_df[candidate_df["rank"] <= args.coverage_topk].copy()
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
        "property_rule_default": args.property_rule,
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

    print("Step 6 complete.")
    print(f"Outputs: {step_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 6: polymer-class + water-soluble inverse design")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config path")
    parser.add_argument("--model_size", type=str, default=None, choices=["small", "medium", "large", "xl"], help="Step1 model size tag")
    parser.add_argument("--split_mode", type=str, default=None, choices=["polymer", "random"], help="Split mode (must match Step 4 output)")
    parser.add_argument("--step4_dir", type=str, default=None, help="Optional direct path to Step 4 directory")
    parser.add_argument("--step4_checkpoint", type=str, default=None, help="Optional explicit path to Step4 chi checkpoint")
    parser.add_argument("--backbone_checkpoint", type=str, default=None, help="Optional explicit path to Step1 backbone checkpoint")

    parser.add_argument("--targets_csv", type=str, default=None, help="Custom χ_target CSV. If omitted, auto-uses Step 3 output.")
    parser.add_argument("--target_polymer_class", type=str, default="all", help="One class, comma list, or all")
    parser.add_argument("--property_rule", type=str, default="upper_bound", choices=["band", "upper_bound", "lower_bound"], help="Default property rule when targets file has none")

    parser.add_argument("--candidate_source", type=str, default="novel", help="known | novel | hybrid")
    parser.add_argument("--generated_csv", type=str, default=None, help="Step2 generated samples CSV")
    parser.add_argument("--generated_smiles_column", type=str, default="smiles", help="SMILES column name in generated_csv")
    parser.add_argument("--allow_non_two_stars", action="store_true", help="Allow generated candidates without exactly two '*' tokens")
    parser.add_argument("--max_novel_candidates", type=int, default=50000, help="Max number of novel generated candidates to keep")

    parser.add_argument("--epsilon", type=float, default=None, help="Property tolerance for 'band' rule")
    parser.add_argument("--class_weight", type=float, default=None, help="Weight for soluble-confidence penalty in score")
    parser.add_argument("--polymer_class_weight", type=float, default=None, help="Weight for polymer-class mismatch penalty")
    parser.add_argument("--coverage_topk", type=int, default=5, help="Top-k used for coverage summary")

    parser.add_argument("--embedding_pooling", type=str, default="mean", choices=["mean", "cls", "max"], help="Pooling for embedding extraction on novel candidates")
    parser.add_argument("--embedding_batch_size", type=int, default=None, help="Batch size for novel embedding extraction")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    args = parser.parse_args()
    main(args)
