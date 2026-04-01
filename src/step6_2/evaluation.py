"""Shared evaluation helpers for Step 6_2 runs."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.chi.inverse_design_common import (
    infer_coefficients_for_novel_candidates,
    prepare_novel_inference_cache,
    resolve_training_smiles,
)
from src.evaluation.polymer_class import PolymerClassifier
from src.utils.chemistry import (
    canonicalize_smiles,
    check_validity,
    compute_sa_score,
    count_stars,
    has_terminal_connection_stars,
)
from .config import ResolvedStep62Config


REQUIRED_SAMPLE_COLUMNS = {
    "sample_id",
    "smiles",
    "target_row_id",
    "target_row_key",
    "round_id",
    "sampling_seed",
    "run_name",
    "canonical_family",
}


@dataclass
class Step62Evaluator:
    """Shared Step 6_2 evaluator."""

    resolved: ResolvedStep62Config
    inference_cache: Dict[str, object]
    novelty_reference: set[str]
    polymer_classifier: PolymerClassifier
    device: str
    target_sa_max: float
    chi_band_epsilon: float = 0.05


def load_step62_evaluator(
    resolved: ResolvedStep62Config,
    *,
    device: str,
    embedding_pooling: str = "mean",
) -> Step62Evaluator:
    """Load the shared Step 4-based evaluator bundle."""

    chi_cfg = {
        "embedding_timestep": int(
            resolved.base_config.get("chi_training", {})
            .get("shared", {})
            .get("embedding", {})
            .get("timestep", 1)
        ),
        "uncertainty_enabled": False,
        "uncertainty_mc_samples": 0,
    }
    args = SimpleNamespace(
        model_size=resolved.model_size,
        step4_checkpoint=None,
        step4_class_checkpoint=None,
        backbone_checkpoint=None,
        embedding_pooling=embedding_pooling,
        uncertainty_enabled=False,
        uncertainty_mc_samples=0,
    )
    try:
        inference_cache = prepare_novel_inference_cache(
            args=args,
            config=resolved.base_config,
            chi_cfg=chi_cfg,
            results_dir=resolved.results_dir,
            step4_reg_metrics_dir=resolved.step4_reg_metrics_dir,
            step4_cls_metrics_dir=resolved.step4_cls_metrics_dir,
            device=device,
            split_mode=resolved.split_mode,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Step 6_2 evaluation requires existing Step 4 regression/classification checkpoints. "
            f"Resolved Step4_1 metrics dir: {resolved.step4_reg_metrics_dir}. "
            f"Resolved Step4_2 metrics dir: {resolved.step4_cls_metrics_dir}. "
            f"Original error: {exc}"
        ) from exc
    novelty_reference = resolve_training_smiles(resolved.results_dir, resolved.base_results_dir)
    classifier = PolymerClassifier(patterns=resolved.polymer_patterns)
    return Step62Evaluator(
        resolved=resolved,
        inference_cache=inference_cache,
        novelty_reference=novelty_reference,
        polymer_classifier=classifier,
        device=device,
        target_sa_max=float(resolved.step6_2["target_sa_max"]),
    )


def build_generated_samples_frame(
    smiles_list: Iterable[str],
    *,
    target_row: pd.Series,
    round_id: int,
    sampling_seed: int,
    run_name: str,
    canonical_family: str,
    sample_id_start: int = 1,
) -> pd.DataFrame:
    """Build the raw generated-sample frame for one target row and one round."""

    rows: List[Dict[str, Any]] = []
    for offset, smiles in enumerate(smiles_list):
        sample_id = int(sample_id_start + offset)
        rows.append(
            {
                "sample_id": sample_id,
                "target_row_id": int(target_row["target_row_id"]),
                "target_row_key": str(target_row["target_row_key"]),
                "round_id": int(round_id),
                "sampling_seed": int(sampling_seed),
                "run_name": str(run_name),
                "canonical_family": str(canonical_family),
                "c_target": str(target_row["c_target"]),
                "temperature": float(target_row["temperature"]),
                "phi": float(target_row["phi"]),
                "chi_target": float(target_row["chi_target"]),
                "property_rule": str(target_row.get("property_rule", "upper_bound")),
                "smiles": str(smiles),
            }
        )
    return pd.DataFrame(rows)


def _compute_basic_sample_annotations(
    sample_df: pd.DataFrame,
    evaluator: Step62Evaluator,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in sample_df.to_dict(orient="records"):
        smiles = str(row["smiles"])
        valid_ok = bool(check_validity(smiles))
        canonical_smiles = canonicalize_smiles(smiles) if valid_ok else None
        if canonical_smiles is None and valid_ok:
            canonical_smiles = smiles
        star_ok = bool(
            valid_ok
            and count_stars(smiles) == 2
            and has_terminal_connection_stars(smiles, expected_stars=2)
        )
        sa_score = float(compute_sa_score(smiles)) if valid_ok else np.nan
        sa_ok = bool(valid_ok and np.isfinite(sa_score) and sa_score <= evaluator.target_sa_max)
        family_matches = evaluator.polymer_classifier.classify(smiles) if valid_ok else {}
        matched_classes = sorted([name for name, matched in family_matches.items() if matched])
        c_target = str(row["c_target"])
        rows.append(
            {
                **row,
                "canonical_smiles": canonical_smiles,
                "valid_ok": int(valid_ok),
                "novel_ok": int(bool(valid_ok and canonical_smiles not in evaluator.novelty_reference)),
                "star_ok": int(star_ok),
                "sa_score": sa_score,
                "sa_ok": int(sa_ok),
                "class_ok": int(bool(family_matches.get(c_target, False))),
                "matched_polymer_classes": ",".join(matched_classes),
            }
        )
    return pd.DataFrame(rows)


def _infer_step4_scores(
    annotated_df: pd.DataFrame,
    evaluator: Step62Evaluator,
) -> pd.DataFrame:
    valid_df = annotated_df.loc[annotated_df["valid_ok"].astype(int) == 1].copy()
    if valid_df.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "target_row_id",
                "class_prob",
                "chi_pred_target",
                "class_logit",
                "class_prob_std",
                "chi_pred_std_target",
            ]
        )

    inference_batch_size = int(evaluator.resolved.step6_2.get("inference_batch_size", 128))
    novel_df = valid_df[["sample_id", "smiles", "canonical_smiles"]].copy()
    novel_df = novel_df.rename(columns={"sample_id": "polymer_id", "smiles": "SMILES"})
    novel_df["Polymer"] = novel_df["polymer_id"].map(lambda x: f"sample_{int(x)}")
    target_df = (
        evaluator.resolved.target_family_df[
            ["target_row_id", "temperature", "phi", "chi_target", "property_rule"]
        ]
        .drop_duplicates("target_row_id")
        .rename(columns={"target_row_id": "target_id", "chi_target": "target_chi"})
        .copy()
    )
    inferred = infer_coefficients_for_novel_candidates(
        novel_df=novel_df[["polymer_id", "Polymer", "SMILES", "canonical_smiles"]],
        target_df=target_df,
        config=evaluator.resolved.base_config,
        model_size=evaluator.resolved.model_size,
        split_mode=evaluator.resolved.split_mode,
        chi_checkpoint_path=PathLikeShim(evaluator.inference_cache["chi_checkpoint_path"]),
        class_checkpoint_path=PathLikeShim(evaluator.inference_cache["class_checkpoint_path"]),
        backbone_checkpoint_path=None,
        device=evaluator.device,
        timestep=int(
            evaluator.resolved.base_config.get("chi_training", {})
            .get("shared", {})
            .get("embedding", {})
            .get("timestep", 1)
        ),
        pooling="mean",
        batch_size=inference_batch_size,
        uncertainty_enabled=False,
        uncertainty_mc_samples=0,
        inference_cache=evaluator.inference_cache,
    )
    inferred = inferred.rename(columns={"target_id": "target_row_id"})
    inferred = inferred.rename(columns={"polymer_id": "sample_id"})
    keep_cols = [
        "sample_id",
        "target_row_id",
        "class_logit",
        "class_prob",
        "class_logit_std",
        "class_prob_std",
        "chi_pred_target",
        "chi_pred_std_target",
    ]
    return inferred[keep_cols].copy()


class PathLikeShim(str):
    """String subclass that behaves as a simple path-like placeholder."""

    def __new__(cls, value: str):
        return str.__new__(cls, value)


def _compute_chi_ok(
    chi_pred: float,
    chi_target: float,
    property_rule: str,
    epsilon: float,
) -> int:
    if not np.isfinite(chi_pred):
        return 0
    rule = str(property_rule).strip().lower()
    if rule == "lower_bound":
        return int(chi_pred >= chi_target)
    if rule == "band":
        return int(abs(chi_pred - chi_target) <= float(epsilon))
    return int(chi_pred <= chi_target)


def evaluate_generated_samples(
    sample_df: pd.DataFrame,
    evaluator: Step62Evaluator,
    *,
    chi_band_epsilon: float = 0.05,
) -> pd.DataFrame:
    """Evaluate one raw generated-sample frame against the shared success gate."""

    missing = sorted(REQUIRED_SAMPLE_COLUMNS - set(sample_df.columns))
    if missing:
        raise ValueError(f"Generated sample dataframe missing required columns: {missing}")

    prompt_cols = ["c_target", "temperature", "phi", "chi_target", "property_rule"]
    has_prompt_fields = all(col in sample_df.columns for col in prompt_cols)
    if has_prompt_fields:
        merged = sample_df.copy()
        if "property_rule" not in merged.columns:
            merged["property_rule"] = "upper_bound"
        missing_prompt_rows = (
            merged["c_target"].isna()
            | merged["temperature"].isna()
            | merged["phi"].isna()
            | merged["chi_target"].isna()
        )
        if bool(missing_prompt_rows.any()):
            target_info = evaluator.resolved.target_family_df[
                ["target_row_id", "target_row_key", "c_target", "temperature", "phi", "chi_target", "property_rule"]
            ].drop_duplicates("target_row_id")
            fill_df = merged.loc[missing_prompt_rows].drop(columns=[col for col in prompt_cols if col in merged.columns])
            fill_df = fill_df.merge(
                target_info,
                on=["target_row_id", "target_row_key"],
                how="left",
                suffixes=("", "_target"),
            )
            if fill_df["c_target"].isna().any():
                raise ValueError("Some generated samples reference unknown target_row_id/target_row_key pairs.")
            merged.loc[missing_prompt_rows, prompt_cols] = fill_df[prompt_cols].to_numpy()
    else:
        target_info = evaluator.resolved.target_family_df[
            ["target_row_id", "target_row_key", "c_target", "temperature", "phi", "chi_target", "property_rule"]
        ].drop_duplicates("target_row_id")
        merged = sample_df.merge(
            target_info,
            on=["target_row_id", "target_row_key"],
            how="left",
            suffixes=("", "_target"),
        )
        if merged["c_target"].isna().any():
            raise ValueError("Some generated samples reference unknown target_row_id/target_row_key pairs.")

    annotated_df = _compute_basic_sample_annotations(merged, evaluator)
    inferred_df = _infer_step4_scores(annotated_df, evaluator)
    out = annotated_df.merge(inferred_df, on=["sample_id", "target_row_id"], how="left")

    out["class_prob"] = pd.to_numeric(out["class_prob"], errors="coerce")
    out["chi_pred_target"] = pd.to_numeric(out["chi_pred_target"], errors="coerce")
    out["soluble_ok"] = (out["class_prob"] >= 0.5).fillna(False).astype(int)
    out["chi_ok"] = [
        _compute_chi_ok(chi_pred, chi_target, property_rule, chi_band_epsilon)
        for chi_pred, chi_target, property_rule in zip(
            out["chi_pred_target"],
            out["chi_target"],
            out["property_rule"],
        )
    ]
    out["success_hit"] = (
        out["valid_ok"].astype(int)
        & out["novel_ok"].astype(int)
        & out["star_ok"].astype(int)
        & out["sa_ok"].astype(int)
        & out["soluble_ok"].astype(int)
        & out["class_ok"].astype(int)
        & pd.Series(out["chi_ok"], index=out.index).astype(int)
    ).astype(int)
    return out


def aggregate_target_row_metrics(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-round metrics for each target row."""

    group_cols = [
        "run_name",
        "canonical_family",
        "round_id",
        "sampling_seed",
        "target_row_id",
        "target_row_key",
        "c_target",
        "temperature",
        "phi",
        "chi_target",
    ]
    rows: List[Dict[str, Any]] = []
    for keys, sub in evaluation_df.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, keys)}
        row["n_samples"] = int(len(sub))
        for gate in ["valid_ok", "novel_ok", "star_ok", "sa_ok", "soluble_ok", "class_ok", "chi_ok", "success_hit"]:
            row[f"{gate}_count"] = int(sub[gate].astype(int).sum())
            row[f"{gate}_rate"] = float(sub[gate].astype(float).mean()) if len(sub) else np.nan
        row["benchmark_soluble_oracle_calls"] = int(sub["valid_ok"].astype(int).sum())
        row["benchmark_chi_oracle_calls"] = int(sub["valid_ok"].astype(int).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_target_rows(target_row_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-target-row metrics across sampling rounds."""

    group_cols = [
        "run_name",
        "canonical_family",
        "target_row_id",
        "target_row_key",
        "c_target",
        "temperature",
        "phi",
        "chi_target",
    ]
    rows: List[Dict[str, Any]] = []
    for keys, sub in target_row_metrics_df.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, keys)}
        row["num_rounds"] = int(sub["round_id"].nunique())
        for gate in ["valid_ok", "novel_ok", "star_ok", "sa_ok", "soluble_ok", "class_ok", "chi_ok", "success_hit"]:
            rate_col = f"{gate}_rate"
            row[f"mean_{gate}_rate"] = float(sub[rate_col].mean()) if len(sub) else np.nan
            row[f"std_{gate}_rate"] = float(sub[rate_col].std(ddof=0)) if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_round_metrics(
    evaluation_df: pd.DataFrame,
    target_row_metrics_df: pd.DataFrame,
    *,
    benchmark_training_oracle_calls: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Aggregate run-level round metrics."""

    benchmark_training_oracle_calls = benchmark_training_oracle_calls or {}
    group_cols = ["run_name", "canonical_family", "round_id", "sampling_seed"]
    rows: List[Dict[str, Any]] = []
    for keys, sub in evaluation_df.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, keys)}
        target_sub = target_row_metrics_df.loc[
            (target_row_metrics_df["run_name"] == row["run_name"])
            & (target_row_metrics_df["round_id"] == row["round_id"])
        ]
        row["n_generated_samples"] = int(len(sub))
        row["success_hit_rate"] = float(sub["success_hit"].astype(float).mean()) if len(sub) else np.nan
        row["macro_target_row_success_hit_rate"] = (
            float(target_sub["success_hit_rate"].mean()) if len(target_sub) else np.nan
        )
        for gate in ["valid_ok", "novel_ok", "star_ok", "sa_ok", "soluble_ok", "class_ok", "chi_ok"]:
            row[f"mean_{gate}_rate"] = float(sub[gate].astype(float).mean()) if len(sub) else np.nan
        row["benchmark_soluble_oracle_calls"] = int(target_sub["benchmark_soluble_oracle_calls"].sum()) if len(target_sub) else 0
        row["benchmark_chi_oracle_calls"] = int(target_sub["benchmark_chi_oracle_calls"].sum()) if len(target_sub) else 0
        row["training_soluble_oracle_calls"] = int(benchmark_training_oracle_calls.get("training_soluble_oracle_calls", 0))
        row["training_chi_oracle_calls"] = int(benchmark_training_oracle_calls.get("training_chi_oracle_calls", 0))
        rows.append(row)
    return pd.DataFrame(rows)


def build_method_metrics(
    round_metrics_df: pd.DataFrame,
    target_row_summary_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Build the run-level summary dictionary."""

    if round_metrics_df.empty:
        return {
            "mean_success_hit_rate": np.nan,
            "std_success_hit_rate": np.nan,
            "macro_average_row_mean_success_hit_rate": np.nan,
        }
    return {
        "mean_success_hit_rate": float(round_metrics_df["success_hit_rate"].mean()),
        "std_success_hit_rate": float(round_metrics_df["success_hit_rate"].std(ddof=0)),
        "macro_average_row_mean_success_hit_rate": (
            float(target_row_summary_df["mean_success_hit_rate"].mean())
            if not target_row_summary_df.empty
            else np.nan
        ),
        "mean_benchmark_soluble_oracle_calls": float(round_metrics_df["benchmark_soluble_oracle_calls"].mean()),
        "mean_benchmark_chi_oracle_calls": float(round_metrics_df["benchmark_chi_oracle_calls"].mean()),
    }
