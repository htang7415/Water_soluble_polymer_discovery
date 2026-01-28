from typing import Dict, Sequence, Tuple

import numpy as np
import torch


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    try:
        from scipy.stats import spearmanr

        spearman = float(spearmanr(y_true, y_pred).correlation)
    except Exception:
        spearman = float("nan")
    return {"r2": r2, "mae": mae, "rmse": rmse, "spearman": spearman}


def classification_metrics(y_true: Sequence[int], y_prob: Sequence[float]) -> Dict[str, float]:
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    try:
        from sklearn.metrics import (
            average_precision_score,
            roc_auc_score,
            balanced_accuracy_score,
        )
    except Exception:
        return {"auprc": float("nan"), "auroc": float("nan"), "balanced_acc": float("nan")}
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auprc": float(average_precision_score(y_true, y_prob)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def uncertainty_coverage(y_true: Sequence[float], mean: Sequence[float], std: Sequence[float]) -> Dict[str, float]:
    y_true = np.array(y_true, dtype=float)
    mean = np.array(mean, dtype=float)
    std = np.array(std, dtype=float)
    within_1 = np.mean(np.abs(y_true - mean) <= std)
    within_2 = np.mean(np.abs(y_true - mean) <= 2 * std)
    return {"coverage_68": float(within_1), "coverage_95": float(within_2)}


def trend_error_by_polymer(
    polymer_ids: Sequence[str],
    temperatures: Sequence[float],
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float:
    by_poly = {}
    for pid, t, yt, yp in zip(polymer_ids, temperatures, y_true, y_pred):
        by_poly.setdefault(pid, []).append((t, yt, yp))
    deltas = []
    for pid, items in by_poly.items():
        items = sorted(items, key=lambda x: x[0])
        t_vals = [x[0] for x in items]
        if len(t_vals) < 2:
            continue
        true_delta = items[-1][1] - items[0][1]
        pred_delta = items[-1][2] - items[0][2]
        deltas.append(abs(true_delta - pred_delta))
    return float(np.mean(deltas)) if deltas else float("nan")


def mc_dropout_predict(model, inputs, forward_fn, k: int = 50):
    preds = []
    model.train()
    for _ in range(k):
        with torch.no_grad():
            preds.append(forward_fn(model, inputs).detach().cpu().numpy())
    preds = np.stack(preds, axis=0)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std
