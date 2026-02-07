#!/usr/bin/env python
"""Step 5: inverse design for water-soluble polymers (no polymer-family class target)."""

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
from src.utils.chemistry import compute_sa_score
from src.utils.config import load_config, save_config
from src.utils.model_scales import get_results_dir
from src.utils.reproducibility import save_run_metadata, seed_everything
from src.utils.reporting import save_step_summary, save_artifact_manifest, write_initial_log


K_LIST = [1, 3, 5, 10]


def _compute_target_candidates(
    target_row: pd.Series,
    coeff_df: pd.DataFrame,
    epsilon: float,
    class_weight: float,
    default_property_rule: str,
) -> pd.DataFrame:
    target_chi = float(target_row["target_chi"])
    t = float(target_row["temperature"])
    phi = float(target_row["phi"])

    required_cols = ["polymer_id", "Polymer", "SMILES", "class_prob", "a0", "a1", "a2", "a3", "b1", "b2"]
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

    if property_rule == "upper_bound":
        out["property_error"] = np.maximum(out["chi_pred_target"] - target_chi, 0.0)
        out["property_hit"] = (out["chi_pred_target"] <= target_chi).astype(int)
    elif property_rule == "lower_bound":
        out["property_error"] = np.maximum(target_chi - out["chi_pred_target"], 0.0)
        out["property_hit"] = (out["chi_pred_target"] >= target_chi).astype(int)
    else:
        out["property_error"] = out["abs_error"]
        out["property_hit"] = (out["abs_error"] <= float(epsilon)).astype(int)

    out["joint_hit"] = ((out["soluble_hit"] == 1) & (out["property_hit"] == 1)).astype(int)
    out["score"] = out["property_error"] + float(class_weight) * (1.0 - out["soluble_confidence"])

    out = out.sort_values(["score", "property_error", "soluble_confidence"], ascending=[True, True, False]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def _target_metrics(cand: pd.DataFrame) -> Dict[str, float]:
    row = {
        "target_id": int(cand["target_id"].iloc[0]),
        "property_rule": cand["property_rule"].iloc[0] if "property_rule" in cand.columns else "upper_bound",
        "temperature": float(cand["temperature"].iloc[0]),
        "phi": float(cand["phi"].iloc[0]),
        "target_chi": float(cand["target_chi"].iloc[0]),
        "n_candidates": int(len(cand)),
        "n_soluble_hit": int(cand["soluble_hit"].sum()),
        "n_property_hit": int(cand["property_hit"].sum()),
        "n_joint_hit": int(cand["joint_hit"].sum()),
        "soluble_hit_rate": float(cand["soluble_hit"].mean()),
        "property_hit_rate": float(cand["property_hit"].mean()),
        "joint_hit_rate": float(cand["joint_hit"].mean()),
        "top1_polymer": cand["Polymer"].iloc[0],
        "top1_candidate_source": cand["candidate_source"].iloc[0] if "candidate_source" in cand.columns else "unknown",
        "top1_class_prob": float(cand["class_prob"].iloc[0]),
        "top1_soluble_hit": int(cand["soluble_hit"].iloc[0]),
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
            "mean_property_hit_rate": float(np.mean(sub["property_hit_rate"])),
        }
        for k in K_LIST:
            out[f"target_success_top{k}"] = float(np.mean(sub[f"top{k}_joint_hit"] > 0))
            if f"top{k}_novel_rate" in sub.columns:
                out[f"mean_top{k}_novel_rate"] = float(np.nanmean(sub[f"top{k}_novel_rate"]))
        return out

    rows = [summarize("overall", target_metrics_df)]

    if "top1_candidate_source" in target_metrics_df.columns:
        for source, sub in target_metrics_df.groupby("top1_candidate_source"):
            rows.append(summarize(f"top1_source_{source}", sub))

    return pd.DataFrame(rows)


def _postprocess_metrics(candidate_df: pd.DataFrame, target_metrics_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    by_condition = (
        target_metrics_df.groupby(["temperature", "phi"], as_index=False)
        .agg(
            top1_property_error_mean=("top1_property_error", "mean"),
            top1_abs_error_mean=("top1_abs_error", "mean"),
            target_success=("n_joint_hit", lambda x: float(np.mean(np.asarray(x) > 0))),
            top1_joint_hit_rate=("top1_joint_hit", "mean"),
            top5_joint_hit_rate=("top5_joint_hit", "mean"),
        )
        .sort_values(["temperature", "phi"])  # stable order for downstream plots
    )

    topk_rows = []
    for k in K_LIST:
        topk = candidate_df[candidate_df["rank"] <= k].copy()
        per_target = topk.groupby("target_id", as_index=False).agg(
            novel_rate=("is_novel_vs_train", "mean"),
            mean_class_prob=("class_prob", "mean"),
            mean_property_error=("property_error", "mean"),
        )
        topk_rows.append(
            {
                "k": int(k),
                "mean_novel_rate": float(per_target["novel_rate"].mean()) if len(per_target) else np.nan,
                "mean_class_prob": float(per_target["mean_class_prob"].mean()) if len(per_target) else np.nan,
                "mean_property_error": float(per_target["mean_property_error"].mean()) if len(per_target) else np.nan,
            }
        )
    topk_novelty = pd.DataFrame(topk_rows)

    top1 = candidate_df[candidate_df["rank"] == 1].copy()
    top1_sa_rows = []
    for _, row in top1.iterrows():
        sa = compute_sa_score(str(row["SMILES"]))
        top1_sa_rows.append(
            {
                "target_id": int(row["target_id"]),
                "polymer_id": int(row["polymer_id"]),
                "Polymer": row["Polymer"],
                "candidate_source": row.get("candidate_source", "unknown"),
                "sa_score": float(sa) if sa is not None else np.nan,
            }
        )
    top1_sa = pd.DataFrame(top1_sa_rows)

    return {
        "by_condition": by_condition,
        "topk_novelty": topk_novelty,
        "top1_sa": top1_sa,
    }


def _save_figures(
    target_metrics_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    topk_novelty: pd.DataFrame,
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
    ax.set_title("Step 5 top-k target success")
    fig.tight_layout()
    fig.savefig(out_dir / "topk_target_success_curve.png", dpi=dpi)
    plt.close(fig)

    # Top1 error distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.histplot(data=target_metrics_df, x="top1_property_error", bins=12, kde=True, color="#4c78a8", ax=ax)
    ax.set_xlabel("Top-1 property error")
    ax.set_ylabel("Count")
    ax.set_title("Top-1 property error distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "top1_property_error_distribution.png", dpi=dpi)
    plt.close(fig)

    # Target vs top1 parity
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(data=target_metrics_df, x="target_chi", y="top1_pred_chi", s=70, color="#1f77b4", ax=ax)
    lo = float(min(target_metrics_df["target_chi"].min(), target_metrics_df["top1_pred_chi"].min()))
    hi = float(max(target_metrics_df["target_chi"].max(), target_metrics_df["top1_pred_chi"].max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel("Target χ")
    ax.set_ylabel("Top-1 predicted χ")
    ax.set_title("Target χ vs top-1 χ")
    fig.tight_layout()
    fig.savefig(out_dir / "target_vs_top1_chi_parity.png", dpi=dpi)
    plt.close(fig)

    # Condition heatmap (top1 joint hit)
    pivot = target_metrics_df.pivot(index="temperature", columns="phi", values="top1_joint_hit")
    fig, ax = plt.subplots(figsize=(5.8, 5))
    sns.heatmap(pivot, cmap="YlGnBu", vmin=0, vmax=1, annot=True, fmt=".0f", cbar_kws={"label": "Top-1 joint hit"}, ax=ax)
    ax.set_xlabel("ϕ")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Top-1 joint hit by condition")
    fig.tight_layout()
    fig.savefig(out_dir / "top1_joint_hit_heatmap.png", dpi=dpi)
    plt.close(fig)

    # Top5 selection frequency
    top5 = candidate_df[candidate_df["rank"] <= 5].copy()
    if not top5.empty:
        freq = (
            top5.groupby(["polymer_id", "Polymer"], as_index=False)
            .size()
            .rename(columns={"size": "selection_count"})
            .sort_values("selection_count", ascending=False)
            .head(12)
        )
        fig, ax = plt.subplots(figsize=(8.5, 5))
        sns.barplot(data=freq, y="Polymer", x="selection_count", color="#4c78a8", ax=ax)
        ax.set_xlabel("Selection count in top-5")
        ax.set_ylabel("Polymer")
        ax.set_title("Most frequently selected polymers")
        fig.tight_layout()
        fig.savefig(out_dir / "top5_polymer_selection_frequency.png", dpi=dpi)
        plt.close(fig)

    # Top1 confidence vs error
    top1 = candidate_df[candidate_df["rank"] == 1].copy()
    if not top1.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=top1, x="property_error", y="class_prob", hue="candidate_source", palette="Set2", s=70, ax=ax)
        ax.set_xlabel("Top-1 property error")
        ax.set_ylabel("Top-1 soluble confidence")
        ax.set_title("Top-1 confidence vs error")
        fig.tight_layout()
        fig.savefig(out_dir / "top1_confidence_vs_error.png", dpi=dpi)
        plt.close(fig)

    # Novelty and quality vs top-k
    if not topk_novelty.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        sns.lineplot(data=topk_novelty, x="k", y="mean_novel_rate", marker="o", linewidth=2, ax=axes[0], color="#54a24b")
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("Mean novelty rate")
        axes[0].set_title("Novelty among top-k")

        sns.lineplot(data=topk_novelty, x="k", y="mean_property_error", marker="o", linewidth=2, ax=axes[1], color="#e45756")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("Mean property error")
        axes[1].set_title("Property error among top-k")

        fig.tight_layout()
        fig.savefig(out_dir / "topk_novelty_and_error_curve.png", dpi=dpi)
        plt.close(fig)


def main(args):
    config = load_config(args.config)
    chi_cfg = default_chi_config(config)

    split_mode = str(chi_cfg["split_mode"]).strip().lower()
    if split_mode not in {"polymer", "random"}:
        raise ValueError("split_mode must be one of {'polymer','random'}")

    epsilon = float(args.epsilon if args.epsilon is not None else chi_cfg.get("epsilon", 0.05))
    class_weight = float(args.class_weight if args.class_weight is not None else chi_cfg.get("class_weight", 0.25))
    candidate_source_value = args.candidate_source if args.candidate_source is not None else str(chi_cfg.get("candidate_source", "novel"))
    candidate_source = parse_candidate_source(candidate_source_value)
    args.candidate_source = candidate_source
    default_property_rule = str(args.property_rule if args.property_rule is not None else chi_cfg.get("property_rule", "upper_bound")).strip().lower()
    coverage_topk = int(args.coverage_topk if args.coverage_topk is not None else chi_cfg.get("coverage_topk", 5))

    results_dir = Path(get_results_dir(args.model_size, config["paths"]["results_dir"]))
    base_results_dir = Path(config["paths"]["results_dir"])
    step4_dir = Path(args.step4_dir) if args.step4_dir else results_dir / "step4_chi_training" / split_mode
    step4_metrics_dir = step4_dir / "metrics"
    check_required_step4_files(step4_metrics_dir)

    step_dir = results_dir / "step5_water_soluble_inverse_design" / split_mode
    metrics_dir = step_dir / "metrics"
    figures_dir = step_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    seed_info = seed_everything(int(config["data"]["random_seed"]))
    save_config(config, step_dir / "config_used.yaml")
    save_run_metadata(step_dir, args.config, seed_info)
    write_initial_log(
        step_dir=step_dir,
        step_name="step5_water_soluble_inverse_design",
        context={
            "config_path": args.config,
            "model_size": args.model_size,
            "results_dir": str(results_dir),
            "split_mode": split_mode,
            "candidate_source": candidate_source,
            "epsilon": epsilon,
            "class_weight": class_weight,
            "property_rule_default": default_property_rule,
            "coverage_topk": coverage_topk,
            "random_seed": config["data"]["random_seed"],
        },
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Step 5: water-soluble inverse design on χ(T,ϕ)")
    print(f"split_mode={split_mode}")
    print(f"candidate_source={candidate_source}")
    print(f"epsilon={epsilon}")
    print(f"class_weight={class_weight}")
    print(f"device={device}")
    print("=" * 70)

    target_df, target_path_used = load_soluble_targets(
        targets_csv=args.targets_csv,
        results_dir=results_dir,
        base_results_dir=base_results_dir,
        split_mode=split_mode,
    )
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

    coeff_df.to_csv(metrics_dir / "inverse_candidate_pool.csv", index=False)
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
    post["by_condition"].to_csv(metrics_dir / "inverse_metrics_by_condition.csv", index=False)
    post["topk_novelty"].to_csv(metrics_dir / "inverse_topk_novelty_metrics.csv", index=False)
    post["top1_sa"].to_csv(metrics_dir / "inverse_top1_sa_scores.csv", index=False)

    topk = candidate_df[candidate_df["rank"] <= coverage_topk].copy()
    coverage = (
        topk.groupby(["candidate_source", "polymer_id", "Polymer"], as_index=False)
        .agg(
            selected_count=("rank", "size"),
            mean_class_prob=("class_prob", "mean"),
            mean_property_error=("property_error", "mean"),
            mean_abs_error=("abs_error", "mean"),
            mean_novel_rate=("is_novel_vs_train", "mean"),
        )
        .sort_values("selected_count", ascending=False)
    )
    coverage["selected_rate"] = coverage["selected_count"] / float(len(target_df))
    coverage.to_csv(metrics_dir / "inverse_polymer_coverage.csv", index=False)

    summary = {
        "n_targets": int(len(target_df)),
        "epsilon": epsilon,
        "class_weight": class_weight,
        "split_mode": split_mode,
        "candidate_source": candidate_source,
        "device": device,
        "targets_csv_used": target_path_used,
        "property_rule_default": default_property_rule,
        "coverage_topk": coverage_topk,
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

    with open(metrics_dir / "step5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    save_step_summary(summary, metrics_dir)

    dpi = int(config.get("plotting", {}).get("dpi", 600))
    font_size = int(config.get("plotting", {}).get("font_size", 12))
    _save_figures(
        target_metrics_df=target_metrics_df,
        candidate_df=candidate_df,
        aggregate_df=aggregate_df,
        topk_novelty=post["topk_novelty"],
        out_dir=figures_dir,
        dpi=dpi,
        font_size=font_size,
    )
    save_artifact_manifest(step_dir=step_dir, metrics_dir=metrics_dir, figures_dir=figures_dir)

    print("Step 5 complete.")
    print(f"Outputs: {step_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: water-soluble inverse design")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config path")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "medium", "large", "xl"], help="Step1 model size tag")
    parser.add_argument("--step4_dir", type=str, default=None, help="Optional direct path to Step 4 directory")
    parser.add_argument("--step4_checkpoint", type=str, default=None, help="Optional explicit path to Step4 chi checkpoint")
    parser.add_argument("--backbone_checkpoint", type=str, default=None, help="Optional explicit path to Step1 backbone checkpoint")

    parser.add_argument("--targets_csv", type=str, default=None, help="Custom χ_target CSV. If omitted, auto-uses Step 3 output.")
    parser.add_argument("--property_rule", type=str, default=None, choices=["band", "upper_bound", "lower_bound"], help="Default property rule when targets file has none")

    parser.add_argument("--candidate_source", type=str, default=None, help="known | novel | hybrid")
    parser.add_argument("--generated_csv", type=str, default=None, help="Step2 generated samples CSV")
    parser.add_argument("--generated_smiles_column", type=str, default="smiles", help="SMILES column name in generated_csv")
    parser.add_argument("--allow_non_two_stars", action="store_true", help="Allow generated candidates without exactly two '*' tokens")
    parser.add_argument("--max_novel_candidates", type=int, default=50000, help="Max number of novel generated candidates to keep")

    parser.add_argument("--epsilon", type=float, default=None, help="Property tolerance for 'band' rule")
    parser.add_argument("--class_weight", type=float, default=None, help="Weight for soluble-confidence penalty in score")
    parser.add_argument("--coverage_topk", type=int, default=None, help="Top-k used for coverage summary")

    parser.add_argument("--embedding_pooling", type=str, default="mean", choices=["mean", "cls", "max"], help="Pooling for embedding extraction on novel candidates")
    parser.add_argument("--embedding_batch_size", type=int, default=None, help="Batch size for novel embedding extraction")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    args = parser.parse_args()
    main(args)
