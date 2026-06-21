"""Shared evaluation helpers for Step 5 runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from src.chi.inverse_design_common import (
    infer_coefficients_for_novel_candidates,
    prepare_novel_inference_cache,
    resolve_training_smiles,
)
from src.evaluation.polymer_class import BACKBONE_CLASS_MATCH_CLASSES, PolymerClassifier
from src.utils.chemistry import (
    canonicalize_smiles,
    check_validity,
    compute_sa_score,
    count_stars,
    has_terminal_connection_stars,
)
from .config import ResolvedStep5Config, resolve_step5_sa_thresholds


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

_ANNOTATION_GATES = [
    "valid_ok",
    "novel_ok",
    "star_ok",
    "sa_ok_reporting",
    "sa_ok_discovery",
    "sa_ok",
    "soluble_ok",
    "property_success_hit_discovery",
    "property_success_hit",
    "class_ok_loose",
    "class_ok_strict",
    "class_ok",
    "chi_ok",
    "chi_band_ok",
    "success_hit_discovery_loose",
    "success_hit_discovery_strict",
    "success_hit_discovery",
    "success_hit_loose",
    "success_hit_strict",
    "success_hit",
]


@dataclass
class Step5Evaluator:
    """Shared Step 5 evaluator."""

    resolved: ResolvedStep5Config
    inference_cache: Dict[str, object]
    novelty_reference: set[str]
    polymer_classifier: PolymerClassifier
    device: str
    target_sa_max: float
    reporting_sa_thresholds: Dict[str, float]
    discovery_sa_thresholds: Dict[str, float]
    chi_band_epsilon: float = 0.05


def resolve_step5_chi_target_bound_mode(step5_cfg: Mapping[str, Any] | None) -> str:
    if not isinstance(step5_cfg, Mapping):
        return "point_estimate"
    mode = str(step5_cfg.get("chi_target_bound_mode", "point_estimate")).strip().lower()
    if mode not in {"point_estimate", "step3_bootstrap_interval"}:
        raise ValueError(
            "step5.chi_target_bound_mode must be one of "
            "{'point_estimate', 'step3_bootstrap_interval'}."
        )
    return mode


def resolve_effective_chi_bounds(
    *,
    row: Mapping[str, Any],
    chi_target: float,
    property_rule: str,
    epsilon: float,
    step5_cfg: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    mode = resolve_step5_chi_target_bound_mode(step5_cfg)
    rule = str(property_rule).strip().lower()
    chi_target = float(chi_target)

    # Step 3 bootstrap quantiles are uncertainty metadata, not acceptance bounds.
    if rule == "lower_bound":
        lower_bound = float(chi_target)
        upper_bound = float("inf")
    elif rule == "band":
        lower_bound = float(chi_target) - float(epsilon)
        upper_bound = float(chi_target) + float(epsilon)
    else:
        lower_bound = float("-inf")
        # The water-miscible Step 5 benchmark uses a strict upper-bound requirement:
        # chi_pred(T, phi) < chi_target(T, phi). We encode that open interval by
        # tightening the effective upper bound to the previous representable float.
        upper_bound = float(np.nextafter(chi_target, float("-inf")))
    return {
        "chi_target_effective_lower": float(lower_bound),
        "chi_target_effective_upper": float(upper_bound),
        "chi_target_bound_mode": mode,
    }


def compute_chi_ok_from_bounds(
    chi_pred: float,
    *,
    lower_bound: float,
    upper_bound: float,
) -> int:
    if not np.isfinite(chi_pred):
        return 0
    if np.isfinite(lower_bound) and chi_pred < float(lower_bound):
        return 0
    if np.isfinite(upper_bound) and chi_pred > float(upper_bound):
        return 0
    return 1


def compute_chi_penalty_from_bounds(
    chi_pred: float,
    *,
    lower_bound: float,
    upper_bound: float,
) -> float:
    if not np.isfinite(chi_pred):
        return -1.0
    if np.isfinite(lower_bound) and chi_pred < float(lower_bound):
        return float(chi_pred) - float(lower_bound)
    if np.isfinite(upper_bound) and chi_pred > float(upper_bound):
        return float(upper_bound) - float(chi_pred)
    return 0.0


def load_step5_evaluator(
    resolved: ResolvedStep5Config,
    *,
    device: str,
    embedding_pooling: str = "mean",
    skip_novelty_reference: bool = False,
) -> Step5Evaluator:
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
            "Step 5 evaluation requires existing Step 4 regression/classification checkpoints. "
            f"Resolved Step4_1 metrics dir: {resolved.step4_reg_metrics_dir}. "
            f"Resolved Step4_2 metrics dir: {resolved.step4_cls_metrics_dir}. "
            f"Original error: {exc}"
        ) from exc
    novelty_reference = (
        set()
        if bool(skip_novelty_reference)
        else resolve_training_smiles(resolved.results_dir, resolved.base_results_dir)
    )
    classifier = PolymerClassifier(patterns=resolved.polymer_patterns)
    reporting_sa_thresholds: Dict[str, float] = {}
    discovery_sa_thresholds: Dict[str, float] = {}
    for c_target in resolved.available_target_classes:
        thresholds = resolve_step5_sa_thresholds(resolved.step5, c_target)
        reporting_sa_thresholds[str(c_target)] = float(thresholds["reporting"])
        discovery_sa_thresholds[str(c_target)] = float(thresholds["discovery"])
    return Step5Evaluator(
        resolved=resolved,
        inference_cache=inference_cache,
        novelty_reference=novelty_reference,
        polymer_classifier=classifier,
        device=device,
        target_sa_max=float(resolved.step5["target_sa_max"]),
        reporting_sa_thresholds=reporting_sa_thresholds,
        discovery_sa_thresholds=discovery_sa_thresholds,
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
        for optional_col in ("chi_target_boot_q025", "chi_target_boot_q975"):
            optional_value = pd.to_numeric(target_row.get(optional_col, np.nan), errors="coerce")
            if np.isfinite(optional_value):
                rows[-1][optional_col] = float(optional_value)
    return pd.DataFrame(rows)


def _compute_basic_sample_annotations(
    sample_df: pd.DataFrame,
    evaluator: Step5Evaluator,
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
        c_target = str(row["c_target"])
        sa_score = float(compute_sa_score(smiles)) if valid_ok else np.nan
        reporting_sa_max = float(evaluator.reporting_sa_thresholds.get(c_target, evaluator.target_sa_max))
        discovery_sa_max = float(evaluator.discovery_sa_thresholds.get(c_target, reporting_sa_max))
        sa_ok_reporting = bool(valid_ok and np.isfinite(sa_score) and sa_score <= reporting_sa_max)
        sa_ok_discovery = bool(valid_ok and np.isfinite(sa_score) and sa_score <= discovery_sa_max)
        target_class_backbone_defined = bool(c_target in BACKBONE_CLASS_MATCH_CLASSES)
        family_matches_loose = evaluator.polymer_classifier.classify(smiles) if valid_ok else {}
        family_matches_strict = evaluator.polymer_classifier.classify_backbone(smiles) if valid_ok else {}
        matched_classes_loose = sorted([name for name, matched in family_matches_loose.items() if matched])
        matched_classes_strict = sorted([name for name, matched in family_matches_strict.items() if matched])
        class_ok_loose = int(bool(family_matches_loose.get(c_target, False)))
        class_ok_strict = int(bool(family_matches_strict.get(c_target, False)))
        class_ok = int(class_ok_strict if target_class_backbone_defined else class_ok_loose)
        rows.append(
            {
                **row,
                "canonical_smiles": canonical_smiles,
                "target_class_backbone_defined": int(target_class_backbone_defined),
                "valid_ok": int(valid_ok),
                "novel_ok": int(bool(valid_ok and canonical_smiles not in evaluator.novelty_reference)),
                "star_ok": int(star_ok),
                "sa_score": sa_score,
                "target_sa_max_reporting": reporting_sa_max,
                "target_sa_max_discovery": discovery_sa_max,
                "sa_ok_reporting": int(sa_ok_reporting),
                "sa_ok_discovery": int(sa_ok_discovery),
                "sa_ok": int(sa_ok_reporting),
                "class_metric_mode": ("strict_backbone" if target_class_backbone_defined else "loose"),
                "class_ok_loose": class_ok_loose,
                "class_ok_strict": class_ok_strict,
                "class_ok": class_ok,
                "matched_polymer_classes_loose": ",".join(matched_classes_loose),
                "matched_polymer_classes_strict": ",".join(matched_classes_strict),
                "matched_polymer_classes": ",".join(
                    matched_classes_strict if target_class_backbone_defined else matched_classes_loose
                ),
            }
        )
    return pd.DataFrame(rows)


def _infer_step4_scores(
    annotated_df: pd.DataFrame,
    evaluator: Step5Evaluator,
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

    inference_batch_size = int(evaluator.resolved.step5.get("inference_batch_size", 128))
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
    class_checkpoint_raw = evaluator.inference_cache.get("class_checkpoint_path")
    inferred = infer_coefficients_for_novel_candidates(
        novel_df=novel_df[["polymer_id", "Polymer", "SMILES", "canonical_smiles"]],
        target_df=target_df,
        config=evaluator.resolved.base_config,
        model_size=evaluator.resolved.model_size,
        split_mode=evaluator.resolved.split_mode,
        chi_checkpoint_path=Path(evaluator.inference_cache["chi_checkpoint_path"]),
        class_checkpoint_path=Path(class_checkpoint_raw) if class_checkpoint_raw else None,
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
    return int(chi_pred < chi_target)


def evaluate_generated_samples(
    sample_df: pd.DataFrame,
    evaluator: Step5Evaluator,
    *,
    chi_band_epsilon: float | None = None,
) -> pd.DataFrame:
    """Evaluate one raw generated-sample frame against the shared success gate."""

    if chi_band_epsilon is None:
        chi_band_epsilon = float(evaluator.resolved.step5.get("chi_band_epsilon", 0.25))

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
    effective_bounds = [
        resolve_effective_chi_bounds(
            row=row,
            chi_target=float(row["chi_target"]),
            property_rule=str(row.get("property_rule", "upper_bound")),
            epsilon=chi_band_epsilon,
            step5_cfg=evaluator.resolved.step5,
        )
        for row in out.to_dict(orient="records")
    ]
    effective_bounds_df = pd.DataFrame(effective_bounds, index=out.index)
    for column in effective_bounds_df.columns:
        out[column] = effective_bounds_df[column]
    out["chi_ok"] = [
        compute_chi_ok_from_bounds(
            chi_pred,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        for chi_pred, lower_bound, upper_bound in zip(
            out["chi_pred_target"],
            out["chi_target_effective_lower"],
            out["chi_target_effective_upper"],
        )
    ]
    out["chi_band_ok"] = [
        _compute_chi_ok(chi_pred, chi_target, "band", chi_band_epsilon)
        for chi_pred, chi_target in zip(
            out["chi_pred_target"],
            out["chi_target"],
        )
    ]
    base_success_mask = (
        out["valid_ok"].astype(int)
        & out["novel_ok"].astype(int)
        & out["star_ok"].astype(int)
        & out["soluble_ok"].astype(int)
        & pd.Series(out["chi_ok"], index=out.index).astype(int)
    )
    reporting_success_mask = base_success_mask & out["sa_ok_reporting"].astype(int)
    discovery_success_mask = base_success_mask & out["sa_ok_discovery"].astype(int)
    out["property_success_hit_discovery"] = discovery_success_mask.astype(int)
    out["property_success_hit"] = reporting_success_mask.astype(int)
    out["success_hit_discovery_loose"] = (discovery_success_mask & out["class_ok_loose"].astype(int)).astype(int)
    out["success_hit_discovery_strict"] = (discovery_success_mask & out["class_ok_strict"].astype(int)).astype(int)
    out["success_hit_discovery"] = np.where(
        out["target_class_backbone_defined"].astype(int) == 1,
        out["success_hit_discovery_strict"].astype(int),
        out["success_hit_discovery_loose"].astype(int),
    ).astype(int)
    out["success_hit_loose"] = (reporting_success_mask & out["class_ok_loose"].astype(int)).astype(int)
    out["success_hit_strict"] = (reporting_success_mask & out["class_ok_strict"].astype(int)).astype(int)
    out["success_hit"] = np.where(
        out["target_class_backbone_defined"].astype(int) == 1,
        out["success_hit_strict"].astype(int),
        out["success_hit_loose"].astype(int),
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
        for gate in _ANNOTATION_GATES:
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
        for gate in _ANNOTATION_GATES:
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
        row["success_hit_rate_discovery_loose"] = (
            float(sub["success_hit_discovery_loose"].astype(float).mean()) if len(sub) else np.nan
        )
        row["success_hit_rate_discovery_strict"] = (
            float(sub["success_hit_discovery_strict"].astype(float).mean()) if len(sub) else np.nan
        )
        row["success_hit_rate_discovery"] = (
            float(sub["success_hit_discovery"].astype(float).mean()) if len(sub) else np.nan
        )
        row["success_hit_rate_loose"] = float(sub["success_hit_loose"].astype(float).mean()) if len(sub) else np.nan
        row["success_hit_rate_strict"] = float(sub["success_hit_strict"].astype(float).mean()) if len(sub) else np.nan
        row["success_hit_rate"] = float(sub["success_hit"].astype(float).mean()) if len(sub) else np.nan
        row["macro_target_row_success_hit_rate_discovery_loose"] = (
            float(target_sub["success_hit_discovery_loose_rate"].mean()) if len(target_sub) else np.nan
        )
        row["macro_target_row_success_hit_rate_discovery_strict"] = (
            float(target_sub["success_hit_discovery_strict_rate"].mean()) if len(target_sub) else np.nan
        )
        row["macro_target_row_success_hit_rate_discovery"] = (
            float(target_sub["success_hit_discovery_rate"].mean()) if len(target_sub) else np.nan
        )
        row["macro_target_row_success_hit_rate_loose"] = (
            float(target_sub["success_hit_loose_rate"].mean()) if len(target_sub) else np.nan
        )
        row["macro_target_row_success_hit_rate_strict"] = (
            float(target_sub["success_hit_strict_rate"].mean()) if len(target_sub) else np.nan
        )
        row["macro_target_row_success_hit_rate"] = (
            float(target_sub["success_hit_rate"].mean()) if len(target_sub) else np.nan
        )
        for gate in [gate for gate in _ANNOTATION_GATES if not gate.startswith("success_hit")]:
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
            "mean_property_success_hit_rate": np.nan,
            "mean_property_success_hit_rate_discovery": np.nan,
            "std_property_success_hit_rate": np.nan,
            "std_property_success_hit_rate_discovery": np.nan,
            "mean_success_hit_rate": np.nan,
            "mean_success_hit_rate_discovery": np.nan,
            "mean_success_hit_rate_discovery_loose": np.nan,
            "mean_success_hit_rate_discovery_strict": np.nan,
            "mean_success_hit_rate_loose": np.nan,
            "mean_success_hit_rate_strict": np.nan,
            "std_success_hit_rate": np.nan,
            "std_success_hit_rate_discovery": np.nan,
            "std_success_hit_rate_discovery_loose": np.nan,
            "std_success_hit_rate_discovery_strict": np.nan,
            "std_success_hit_rate_loose": np.nan,
            "std_success_hit_rate_strict": np.nan,
            "macro_average_row_mean_success_hit_rate": np.nan,
            "macro_average_row_mean_success_hit_rate_discovery": np.nan,
            "macro_average_row_mean_property_success_hit_rate": np.nan,
            "macro_average_row_mean_property_success_hit_rate_discovery": np.nan,
            "macro_average_row_mean_success_hit_rate_discovery_loose": np.nan,
            "macro_average_row_mean_success_hit_rate_discovery_strict": np.nan,
            "macro_average_row_mean_success_hit_rate_loose": np.nan,
            "macro_average_row_mean_success_hit_rate_strict": np.nan,
        }
    return {
        "mean_property_success_hit_rate": float(round_metrics_df["mean_property_success_hit_rate"].mean()),
        "mean_property_success_hit_rate_discovery": float(
            round_metrics_df["mean_property_success_hit_discovery_rate"].mean()
        ),
        "std_property_success_hit_rate": float(round_metrics_df["mean_property_success_hit_rate"].std(ddof=0)),
        "std_property_success_hit_rate_discovery": float(
            round_metrics_df["mean_property_success_hit_discovery_rate"].std(ddof=0)
        ),
        "mean_success_hit_rate": float(round_metrics_df["success_hit_rate"].mean()),
        "mean_success_hit_rate_discovery": float(round_metrics_df["success_hit_rate_discovery"].mean()),
        "mean_success_hit_rate_discovery_loose": float(round_metrics_df["success_hit_rate_discovery_loose"].mean()),
        "mean_success_hit_rate_discovery_strict": float(round_metrics_df["success_hit_rate_discovery_strict"].mean()),
        "mean_success_hit_rate_loose": float(round_metrics_df["success_hit_rate_loose"].mean()),
        "mean_success_hit_rate_strict": float(round_metrics_df["success_hit_rate_strict"].mean()),
        "std_success_hit_rate": float(round_metrics_df["success_hit_rate"].std(ddof=0)),
        "std_success_hit_rate_discovery": float(round_metrics_df["success_hit_rate_discovery"].std(ddof=0)),
        "std_success_hit_rate_discovery_loose": float(round_metrics_df["success_hit_rate_discovery_loose"].std(ddof=0)),
        "std_success_hit_rate_discovery_strict": float(round_metrics_df["success_hit_rate_discovery_strict"].std(ddof=0)),
        "std_success_hit_rate_loose": float(round_metrics_df["success_hit_rate_loose"].std(ddof=0)),
        "std_success_hit_rate_strict": float(round_metrics_df["success_hit_rate_strict"].std(ddof=0)),
        "macro_average_row_mean_success_hit_rate": (
            float(target_row_summary_df["mean_success_hit_rate"].mean())
            if not target_row_summary_df.empty
            else np.nan
        ),
        "macro_average_row_mean_property_success_hit_rate": (
            float(target_row_summary_df["mean_property_success_hit_rate"].mean())
            if not target_row_summary_df.empty
            else np.nan
        ),
        "macro_average_row_mean_property_success_hit_rate_discovery": (
            float(target_row_summary_df["mean_property_success_hit_discovery_rate"].mean())
            if not target_row_summary_df.empty
            else np.nan
        ),
        "macro_average_row_mean_success_hit_rate_discovery": (
            float(target_row_summary_df["mean_success_hit_discovery_rate"].mean())
            if not target_row_summary_df.empty
            else np.nan
        ),
        "macro_average_row_mean_success_hit_rate_discovery_loose": (
            float(target_row_summary_df["mean_success_hit_discovery_loose_rate"].mean())
            if not target_row_summary_df.empty
            else np.nan
        ),
        "macro_average_row_mean_success_hit_rate_discovery_strict": (
            float(target_row_summary_df["mean_success_hit_discovery_strict_rate"].mean())
            if not target_row_summary_df.empty
            else np.nan
        ),
        "macro_average_row_mean_success_hit_rate_loose": (
            float(target_row_summary_df["mean_success_hit_loose_rate"].mean())
            if not target_row_summary_df.empty
            else np.nan
        ),
        "macro_average_row_mean_success_hit_rate_strict": (
            float(target_row_summary_df["mean_success_hit_strict_rate"].mean())
            if not target_row_summary_df.empty
            else np.nan
        ),
        "mean_benchmark_soluble_oracle_calls": float(round_metrics_df["benchmark_soluble_oracle_calls"].mean()),
        "mean_benchmark_chi_oracle_calls": float(round_metrics_df["benchmark_chi_oracle_calls"].mean()),
        "success_metric_mode": "class_aware_reporting",
        "discovery_success_metric_mode": "class_aware_discovery",
        "property_success_metric_mode": "property_only_reporting",
        "property_discovery_success_metric_mode": "property_only_discovery",
    }
