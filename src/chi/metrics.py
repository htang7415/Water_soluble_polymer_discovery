"""Metrics for chi regression and soluble/insoluble classification."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)



def _safe_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)



def _concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if np.isclose(denom, 0.0):
        return np.nan
    return float(2.0 * cov / denom)



def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float], prefix: str = "") -> Dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)

    if len(y_true) == 0:
        return {
            f"{prefix}n": 0,
            f"{prefix}mae": np.nan,
            f"{prefix}rmse": np.nan,
            f"{prefix}nrmse": np.nan,
            f"{prefix}mse": np.nan,
            f"{prefix}r2": np.nan,
            f"{prefix}mape_pct": np.nan,
            f"{prefix}smape_pct": np.nan,
            f"{prefix}bias": np.nan,
            f"{prefix}max_ae": np.nan,
            f"{prefix}pearson_r": np.nan,
            f"{prefix}spearman_r": np.nan,
            f"{prefix}ccc": np.nan,
            f"{prefix}calib_slope": np.nan,
            f"{prefix}calib_intercept": np.nan,
        }

    error = y_pred - y_true
    abs_error = np.abs(error)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    std_true = float(np.std(y_true))
    nrmse = (rmse / std_true) if std_true > 0 else np.nan
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan

    denom = np.maximum(np.abs(y_true), 1e-8)
    mape_pct = np.mean(abs_error / denom) * 100.0
    smape_denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)
    smape_pct = np.mean((2.0 * abs_error) / smape_denom) * 100.0

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = pearsonr(y_true, y_pred)[0]
        spearman = spearmanr(y_true, y_pred)[0]
    else:
        pearson = np.nan
        spearman = np.nan

    ccc = _concordance_correlation_coefficient(y_true, y_pred)

    if len(y_true) > 1:
        slope, intercept = np.polyfit(y_true, y_pred, deg=1)
    else:
        slope, intercept = np.nan, np.nan

    return {
        f"{prefix}n": int(len(y_true)),
        f"{prefix}mae": _safe_float(mae),
        f"{prefix}rmse": _safe_float(rmse),
        f"{prefix}nrmse": _safe_float(nrmse),
        f"{prefix}mse": _safe_float(mse),
        f"{prefix}r2": _safe_float(r2),
        f"{prefix}mape_pct": _safe_float(mape_pct),
        f"{prefix}smape_pct": _safe_float(smape_pct),
        f"{prefix}bias": _safe_float(np.mean(error)),
        f"{prefix}max_ae": _safe_float(np.max(abs_error)),
        f"{prefix}pearson_r": _safe_float(pearson),
        f"{prefix}spearman_r": _safe_float(spearman),
        f"{prefix}ccc": _safe_float(ccc),
        f"{prefix}calib_slope": _safe_float(slope),
        f"{prefix}calib_intercept": _safe_float(intercept),
    }



def polymer_balanced_nrmse(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    polymer_ids: Iterable[object],
    std_floor: float = 0.02,
    nrmse_clip: float = 10.0,
) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    polymer_arr = np.asarray(list(polymer_ids))

    if not (len(y_true_arr) == len(y_pred_arr) == len(polymer_arr)):
        raise ValueError("polymer_balanced_nrmse expects y_true, y_pred, and polymer_ids to have matching lengths")
    if len(y_true_arr) == 0:
        return np.nan

    finite_mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr) & ~pd.isna(polymer_arr)
    if not np.any(finite_mask):
        return np.nan

    df = pd.DataFrame(
        {
            "polymer_id": polymer_arr[finite_mask],
            "y_true": y_true_arr[finite_mask],
            "y_pred": y_pred_arr[finite_mask],
        }
    )

    per_polymer_scores: List[float] = []
    std_floor = float(std_floor)
    nrmse_clip = float(nrmse_clip)
    for _, sub in df.groupby("polymer_id", sort=False):
        y_true_poly = sub["y_true"].to_numpy(dtype=float)
        y_pred_poly = sub["y_pred"].to_numpy(dtype=float)
        rmse_poly = float(np.sqrt(mean_squared_error(y_true_poly, y_pred_poly)))
        std_poly = float(np.std(y_true_poly)) if len(y_true_poly) >= 2 else 0.0
        denom = max(std_poly, std_floor)
        score = rmse_poly / denom if denom > 0 else np.nan
        if np.isfinite(score):
            per_polymer_scores.append(float(np.clip(score, 0.0, nrmse_clip)))

    if not per_polymer_scores:
        return np.nan
    return _safe_float(np.mean(per_polymer_scores))


def polymer_r2_distribution(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    polymer_ids: Iterable[object],
    prefix: str = "",
) -> Dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    polymer_arr = np.asarray(list(polymer_ids))

    if not (len(y_true_arr) == len(y_pred_arr) == len(polymer_arr)):
        raise ValueError("polymer_r2_distribution expects y_true, y_pred, and polymer_ids to have matching lengths")

    empty_result = {
        f"{prefix}n_polymers": 0,
        f"{prefix}n_valid_polymer_r2": 0,
        f"{prefix}poly_r2_mean": np.nan,
        f"{prefix}poly_r2_median": np.nan,
        f"{prefix}poly_r2_p25": np.nan,
        f"{prefix}poly_r2_p75": np.nan,
        f"{prefix}pct_poly_r2_gt_0": np.nan,
    }
    if len(y_true_arr) == 0:
        return empty_result

    finite_mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr) & ~pd.isna(polymer_arr)
    if not np.any(finite_mask):
        return empty_result

    df = pd.DataFrame(
        {
            "polymer_id": polymer_arr[finite_mask],
            "y_true": y_true_arr[finite_mask],
            "y_pred": y_pred_arr[finite_mask],
        }
    )

    poly_r2_values: List[float] = []
    n_polymers = 0
    for _, sub in df.groupby("polymer_id", sort=False):
        n_polymers += 1
        y_true_poly = sub["y_true"].to_numpy(dtype=float)
        y_pred_poly = sub["y_pred"].to_numpy(dtype=float)
        poly_r2 = r2_score(y_true_poly, y_pred_poly) if len(y_true_poly) > 1 else np.nan
        poly_r2_values.append(_safe_float(poly_r2))

    values = np.asarray(poly_r2_values, dtype=float)
    finite_values = values[np.isfinite(values)]
    return {
        f"{prefix}n_polymers": int(n_polymers),
        f"{prefix}n_valid_polymer_r2": int(finite_values.size),
        f"{prefix}poly_r2_mean": _safe_float(np.mean(finite_values)) if finite_values.size > 0 else np.nan,
        f"{prefix}poly_r2_median": _safe_float(np.median(finite_values)) if finite_values.size > 0 else np.nan,
        f"{prefix}poly_r2_p25": _safe_float(np.percentile(finite_values, 25)) if finite_values.size > 0 else np.nan,
        f"{prefix}poly_r2_p75": _safe_float(np.percentile(finite_values, 75)) if finite_values.size > 0 else np.nan,
        f"{prefix}pct_poly_r2_gt_0": _safe_float(np.mean(finite_values > 0) * 100.0) if finite_values.size > 0 else np.nan,
    }


def classification_metrics(
    y_true: Iterable[int],
    probs: Iterable[float],
    threshold: float = 0.5,
    prefix: str = "",
) -> Dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=int)
    probs = np.asarray(list(probs), dtype=float)

    if len(y_true) == 0:
        return {
            f"{prefix}n": 0,
            f"{prefix}accuracy": np.nan,
            f"{prefix}balanced_accuracy": np.nan,
            f"{prefix}precision": np.nan,
            f"{prefix}recall": np.nan,
            f"{prefix}f1": np.nan,
            f"{prefix}mcc": np.nan,
            f"{prefix}brier": np.nan,
            f"{prefix}auroc": np.nan,
            f"{prefix}auprc": np.nan,
            f"{prefix}positive_rate": np.nan,
            f"{prefix}pred_positive_rate": np.nan,
        }

    pred = (probs >= threshold).astype(int)

    out = {
        f"{prefix}n": int(len(y_true)),
        f"{prefix}accuracy": _safe_float(accuracy_score(y_true, pred)),
        f"{prefix}balanced_accuracy": _safe_float(balanced_accuracy_score(y_true, pred)),
        f"{prefix}precision": _safe_float(precision_score(y_true, pred, zero_division=0)),
        f"{prefix}recall": _safe_float(recall_score(y_true, pred, zero_division=0)),
        f"{prefix}f1": _safe_float(f1_score(y_true, pred, zero_division=0)),
        f"{prefix}mcc": _safe_float(matthews_corrcoef(y_true, pred)),
        f"{prefix}brier": _safe_float(brier_score_loss(y_true, probs)),
        f"{prefix}positive_rate": _safe_float(np.mean(y_true)),
        f"{prefix}pred_positive_rate": _safe_float(np.mean(pred)),
    }

    if len(np.unique(y_true)) > 1:
        out[f"{prefix}auroc"] = _safe_float(roc_auc_score(y_true, probs))
        out[f"{prefix}auprc"] = _safe_float(average_precision_score(y_true, probs))
    else:
        out[f"{prefix}auroc"] = np.nan
        out[f"{prefix}auprc"] = np.nan

    return out



def metrics_by_group(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    group_col: str,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for group, sub in df.groupby(group_col):
        row = {"group": group}
        row.update(regression_metrics(sub[y_true_col], sub[y_pred_col]))
        rows.append(row)
    return pd.DataFrame(rows)



def hit_metrics(errors: Iterable[float], epsilons: Iterable[float]) -> Dict[str, float]:
    err = np.abs(np.asarray(list(errors), dtype=float))
    out: Dict[str, float] = {}
    if err.size == 0:
        for eps in epsilons:
            out[f"hit_rate_eps_{eps}"] = np.nan
        return out

    for eps in epsilons:
        out[f"hit_rate_eps_{eps}"] = _safe_float(np.mean(err <= float(eps)))
    return out



def topk_hit_rate(scores: Iterable[float], hit_flags: Iterable[int], ks: Iterable[int]) -> Dict[str, float]:
    score_arr = np.asarray(list(scores), dtype=float)
    hit_arr = np.asarray(list(hit_flags), dtype=int)
    if score_arr.size == 0:
        return {f"top{k}_hit_rate": np.nan for k in ks}

    order = np.argsort(score_arr)
    out = {}
    for k in ks:
        k_eff = min(int(k), len(order))
        topk_hits = hit_arr[order[:k_eff]]
        out[f"top{k}_hit_rate"] = _safe_float(np.mean(topk_hits)) if k_eff > 0 else np.nan
    return out
