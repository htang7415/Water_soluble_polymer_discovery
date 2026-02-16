#!/usr/bin/env python
"""Step 4_3: traditional ML baselines for chi regression and solubility classification."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.chi.metrics import classification_metrics, hit_metrics, metrics_by_group, regression_metrics  # noqa: E402
from common import (  # noqa: E402
    FingerprintConfig,
    build_final_fit_split_df,
    build_or_load_fingerprint_cache,
    build_tuning_cv_folds,
    features_from_table,
    get_traditional_results_dir,
    load_split_dataset,
    load_traditional_config,
    normalize_split_mode,
    resolve_split_ratios,
    summarize_cv_folds,
)
from src.utils.config import save_config  # noqa: E402
from src.utils.numerics import stable_sigmoid  # noqa: E402
from src.utils.reporting import save_artifact_manifest, save_step_summary, write_initial_log  # noqa: E402

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

@dataclass
class StageConfig:
    split_mode: str
    holdout_test_ratio: float
    seed: int
    tune: bool
    n_trials: int
    tuning_cv_folds: int
    tuning_objective: str
    models: List[str]
    optuna_search_space: Dict[str, object]


def _seed_everything_simple(seed: int) -> Dict[str, object]:
    random.seed(int(seed))
    np.random.seed(int(seed))
    return {"seed": int(seed), "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _save_run_metadata_simple(output_dir: Path, config_path: str, seed_info: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_path": str(config_path),
        "seed": int(seed_info.get("seed", 0)),
        "timestamp_utc": str(seed_info.get("timestamp_utc", "")),
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(payload, f, indent=2)


def _safe_float(v, default=np.nan) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _as_float_pair(value, default_lo: float, default_hi: float) -> Tuple[float, float]:
    lo_default = float(default_lo)
    hi_default = float(default_hi)
    if lo_default > hi_default:
        lo_default, hi_default = hi_default, lo_default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        parsed: List[float] = []
        for item in value:
            try:
                parsed.append(float(item))
            except Exception:
                continue
        if len(parsed) == 2 and np.isfinite(parsed).all():
            lo = float(min(parsed))
            hi = float(max(parsed))
            return lo, hi
    return lo_default, hi_default


def _as_int_pair(value, default_lo: int, default_hi: int) -> Tuple[int, int]:
    lo_default = int(default_lo)
    hi_default = int(default_hi)
    if lo_default > hi_default:
        lo_default, hi_default = hi_default, lo_default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        parsed: List[int] = []
        for item in value:
            try:
                parsed.append(int(float(item)))
            except Exception:
                continue
        if len(parsed) == 2:
            lo = int(min(parsed))
            hi = int(max(parsed))
            return lo, hi
    return lo_default, hi_default


def _as_hidden_options(value, default: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    if not isinstance(value, list) or len(value) == 0:
        return default
    out = []
    for item in value:
        if isinstance(item, list) and len(item) > 0:
            out.append(tuple(int(x) for x in item))
        elif isinstance(item, int):
            out.append((int(item),))
    return out if out else default


def _as_int_options(value, default: List[int]) -> List[int]:
    if not isinstance(value, list) or len(value) == 0:
        return [int(x) for x in default]
    out = []
    for item in value:
        try:
            out.append(int(item))
        except Exception:
            continue
    out = sorted(set(out))
    return out if out else [int(x) for x in default]


def _as_choice_options(value, default: List[object]) -> List[object]:
    if not isinstance(value, list) or len(value) == 0:
        return list(default)
    return list(value)


def _resolve_max_features(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"none", "null", ""}:
            return None
        if text in {"sqrt", "log2"}:
            return text
        try:
            return float(text)
        except Exception:
            return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _resolve_class_weight(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"none", "null", ""}:
            return None
        if text in {"balanced", "balanced_subsample"}:
            return text
        return None
    return None


def _resolve_mlp_batch_size(value):
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    try:
        return int(value)
    except Exception:
        return "auto"


def _resolve_bool_like(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)


def _clean_numeric_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _plot_kde_safe_1d(
    ax,
    values,
    *,
    label: str,
    color: str,
    linewidth: float = 2.0,
    fill: bool = False,
    alpha: float = 0.3,
) -> bool:
    arr = _clean_numeric_array(values)
    if arr.size < 2 or np.isclose(np.std(arr), 0.0):
        return False
    try:
        kde_kwargs = {
            "x": np.asarray(arr, dtype=float),
            "ax": ax,
            "label": label,
            "color": color,
            "linewidth": linewidth,
            "fill": fill,
        }
        if fill:
            kde_kwargs["alpha"] = alpha
        sns.kdeplot(**kde_kwargs)
    except Exception as exc:
        if "Multi-dimensional indexing" not in str(exc):
            raise
        bins = int(np.clip(np.sqrt(arr.size), 10, 60))
        sns.histplot(
            x=np.asarray(arr, dtype=float),
            bins=bins,
            stat="density",
            element="step",
            fill=False,
            color=color,
            linewidth=linewidth,
            label=label,
            ax=ax,
        )
    return True


def _plot_class_prob_density_safe(ax, test_df: pd.DataFrame) -> None:
    try:
        sns.kdeplot(
            data=test_df,
            x="class_prob",
            hue="water_soluble",
            common_norm=False,
            fill=True,
            alpha=0.3,
            ax=ax,
        )
        return
    except Exception as exc:
        if "Multi-dimensional indexing" not in str(exc):
            raise

    for class_value, color in [(1, "#1f77b4"), (0, "#d62728")]:
        sub = test_df.loc[test_df["water_soluble"] == class_value, "class_prob"]
        _plot_kde_safe_1d(
            ax=ax,
            values=sub,
            label=str(class_value),
            color=color,
            linewidth=2.0,
            fill=False,
            alpha=0.3,
        )


def _normalize_tuning_objective(value: object, default_tuning_objective: str) -> str:
    raw = str(value).strip().lower()
    default_norm = str(default_tuning_objective).strip().lower()

    if default_norm == "val_r2":
        if raw in {"val_r2", "r2", "maximize_val_r2"}:
            return "val_r2"
        raise ValueError("step4_3_1_regression.tuning_objective must be one of {'val_r2', 'r2', 'maximize_val_r2'}")

    if default_norm == "val_balanced_accuracy":
        if raw in {"val_balanced_accuracy", "balanced_accuracy", "maximize_val_balanced_accuracy"}:
            return "val_balanced_accuracy"
        raise ValueError(
            "step4_3_2_classification.tuning_objective must be one of "
            "{'val_balanced_accuracy', 'balanced_accuracy', 'maximize_val_balanced_accuracy'}"
        )

    raise ValueError(f"Unsupported default tuning objective: {default_tuning_objective}")


def _objective_summary_name(tuning_objective: str) -> str:
    name = str(tuning_objective).strip().lower()
    if name == "val_r2":
        return "maximize_val_r2"
    if name == "val_balanced_accuracy":
        return "maximize_val_balanced_accuracy"
    return name


def _suggest_int(trial, name: str, search_space: Dict[str, object], default_lo: int, default_hi: int) -> int:
    lo, hi = _as_int_pair(search_space.get(name), default_lo, default_hi)
    if lo == hi:
        return int(lo)
    return int(trial.suggest_int(name, lo, hi))


def _suggest_float(
    trial,
    name: str,
    search_space: Dict[str, object],
    default_lo: float,
    default_hi: float,
    log: bool = False,
) -> float:
    lo, hi = _as_float_pair(search_space.get(name), default_lo, default_hi)
    if bool(log):
        lo = float(max(lo, 1.0e-12))
        hi = float(max(hi, lo))
    if np.isclose(lo, hi):
        return float(lo)
    return float(trial.suggest_float(name, lo, hi, log=bool(log)))


def _sample_mlp_hidden_layers(trial, search_space: Dict[str, object]) -> Tuple[int, ...]:
    """Sample MLP hidden-layer layout.

    Priority:
    1) Legacy explicit architecture list via `mlp_hidden_layers`.
    2) Dynamic depth/width search via `mlp_num_layers` + `mlp_hidden_units`.
    """
    legacy_options = _as_hidden_options(search_space.get("mlp_hidden_layers"), default=[])
    if legacy_options:
        return tuple(int(x) for x in trial.suggest_categorical("mlp_hidden_layers", legacy_options))

    num_layers_raw = search_space.get("mlp_num_layers", [1, 2, 3])
    if isinstance(num_layers_raw, list) and len(num_layers_raw) == 2:
        n_lo = int(min(num_layers_raw))
        n_hi = int(max(num_layers_raw))
        n_layers = int(trial.suggest_int("mlp_num_layers", n_lo, n_hi))
    else:
        n_opts = _as_int_options(num_layers_raw, default=[1, 2, 3])
        n_layers = int(trial.suggest_categorical("mlp_num_layers", n_opts))

    n_layers = int(max(1, n_layers))
    unit_options = _as_int_options(
        search_space.get("mlp_hidden_units", [64, 128, 256, 512, 1024]),
        default=[64, 128, 256, 512, 1024],
    )
    hidden: List[int] = []
    for i in range(n_layers):
        hidden_i = int(trial.suggest_categorical(f"mlp_hidden_{i}", unit_options))
        hidden.append(hidden_i)
    return tuple(hidden)


def _sklearn_kernel_from_name(name: str, length_scale: float, constant_value: float = 1.0, noise_level: float = 1.0e-5):
    name = str(name).strip().lower()
    ls = float(max(length_scale, 1.0e-6))
    cv = float(max(constant_value, 1.0e-8))
    nl = float(max(noise_level, 1.0e-10))
    if name == "rbf":
        base = RBF(length_scale=ls)
    elif name == "matern15":
        base = Matern(length_scale=ls, nu=1.5)
    elif name == "matern25":
        base = Matern(length_scale=ls, nu=2.5)
    else:
        base = RBF(length_scale=ls)
    return ConstantKernel(cv, constant_value_bounds="fixed") * base + WhiteKernel(noise_level=nl)


def _check_xgboost_required(model_names: List[str]) -> None:
    needs_xgb = any(str(m).strip().lower() == "xgboost" for m in model_names)
    if needs_xgb and importlib.util.find_spec("xgboost") is None:
        raise ImportError(
            "dmlc/xgboost is required by selected models but is not importable. "
            "Install xgboost or remove 'xgboost' from config_traditional.yaml models."
        )


def _load_xgboost_classes():
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except Exception as exc:
        raise ImportError(
            "dmlc/xgboost import failed during model construction. "
            "Install/fix xgboost to use model_name='xgboost'."
        ) from exc
    return XGBClassifier, XGBRegressor


def _check_gpy_required(model_names: List[str]) -> None:
    needs_gpr = any(str(m).strip().lower() == "gpr" for m in model_names)
    if needs_gpr and importlib.util.find_spec("GPy") is None:
        raise ImportError(
            "GPy is required by selected models but is not importable. "
            "Install GPy or remove 'gpr' from config_traditional.yaml models."
        )


def _load_gpy_module():
    try:
        import GPy
    except Exception as exc:
        raise ImportError(
            "GPy import failed during model construction. "
            "Install/fix GPy to use model_name='gpr'."
        ) from exc
    return GPy


class GPyRegressor(BaseEstimator, RegressorMixin):
    """Minimal sklearn-compatible wrapper around GPy GPRegression."""

    def __init__(
        self,
        kernel: str = "rbf",
        length_scale: float = 1.0,
        kernel_variance: float = 1.0,
        alpha: float = 1.0e-5,
        max_iters: int = 200,
    ):
        self.kernel = kernel
        self.length_scale = length_scale
        self.kernel_variance = kernel_variance
        self.alpha = alpha
        self.max_iters = max_iters

    def _make_kernel(self, input_dim: int):
        GPy = _load_gpy_module()
        kname = str(self.kernel).strip().lower()
        ls = float(max(self.length_scale, 1.0e-6))
        kv = float(max(self.kernel_variance, 1.0e-8))
        if kname == "rbf":
            return GPy.kern.RBF(input_dim=input_dim, variance=kv, lengthscale=ls, ARD=False)
        if kname == "matern15":
            return GPy.kern.Matern32(input_dim=input_dim, variance=kv, lengthscale=ls, ARD=False)
        if kname == "matern25":
            return GPy.kern.Matern52(input_dim=input_dim, variance=kv, lengthscale=ls, ARD=False)
        return GPy.kern.RBF(input_dim=input_dim, variance=kv, lengthscale=ls, ARD=False)

    def fit(self, X, y):
        GPy = _load_gpy_module()
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D X, got shape={X_arr.shape}")
        if len(X_arr) != len(y_arr):
            raise ValueError(f"X/y length mismatch: {len(X_arr)} vs {len(y_arr)}")

        kernel = self._make_kernel(input_dim=int(X_arr.shape[1]))
        self.model_ = GPy.models.GPRegression(X_arr, y_arr, kernel=kernel)
        noise = float(max(self.alpha, 1.0e-12))
        try:
            self.model_.Gaussian_noise.variance = noise
            self.model_.Gaussian_noise.variance.fix()
        except Exception:
            pass
        try:
            self.model_.optimize(messages=False, max_iters=int(max(1, self.max_iters)))
        except Exception:
            # Keep training robust for difficult folds/hyperparameters.
            pass
        return self

    def predict(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("GPyRegressor is not fitted.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        mean, _ = self.model_.predict(X_arr)
        return np.asarray(mean, dtype=np.float64).ravel()


def _build_regression_estimator(model_name: str, params: Dict[str, object], seed: int):
    model_name = str(model_name).strip().lower()
    if model_name == "ridge":
        alpha = float(params.get("alpha", 1.0))
        fit_intercept = _resolve_bool_like(params.get("fit_intercept", True), default=True)
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=alpha, fit_intercept=fit_intercept)),
            ]
        )
    if model_name == "random_forest":
        max_features = _resolve_max_features(params.get("max_features", 1.0), default=1.0)
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 600)),
            max_depth=None if params.get("max_depth", None) in {None, 0} else int(params["max_depth"]),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            bootstrap=_resolve_bool_like(params.get("bootstrap", True), default=True),
            max_features=max_features,
            n_jobs=-1,
            random_state=int(seed),
        )
    if model_name == "xgboost":
        _, XGBRegressor = _load_xgboost_classes()
        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=int(params.get("n_estimators", 800)),
            learning_rate=float(params.get("learning_rate", 0.03)),
            max_depth=int(params.get("max_depth", 6)),
            min_child_weight=float(params.get("min_child_weight", 1.0)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.9)),
            gamma=float(params.get("gamma", 0.0)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            max_bin=int(params.get("max_bin", 256)),
            random_state=int(seed),
            n_jobs=-1,
            tree_method="hist",
        )
    if model_name == "mlp":
        hidden_layers = tuple(int(x) for x in params.get("hidden_layers", (512, 256)))
        alpha = float(params.get("alpha", 1.0e-4))
        lr = float(params.get("learning_rate_init", 1.0e-3))
        activation = str(params.get("activation", "relu"))
        batch_size = _resolve_mlp_batch_size(params.get("batch_size", "auto"))
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=hidden_layers,
                        activation=activation,
                        batch_size=batch_size,
                        alpha=alpha,
                        learning_rate_init=lr,
                        max_iter=1000,
                        early_stopping=True,
                        n_iter_no_change=30,
                        random_state=int(seed),
                    ),
                ),
            ]
        )
    if model_name == "gpr":
        alpha = float(params.get("alpha", 1.0e-5))
        kernel_name = str(params.get("kernel", "rbf"))
        length_scale = float(params.get("length_scale", 1.0))
        kernel_variance = float(params.get("kernel_variance", 1.0))
        max_iters = int(params.get("max_iters", 200))
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    GPyRegressor(
                        kernel=kernel_name,
                        length_scale=length_scale,
                        kernel_variance=kernel_variance,
                        alpha=alpha,
                        max_iters=max_iters,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported regression model: {model_name}")


def _build_classification_estimator(model_name: str, params: Dict[str, object], seed: int):
    model_name = str(model_name).strip().lower()
    if model_name == "logistic":
        c_val = float(params.get("C", 1.0))
        penalty = str(params.get("penalty", "l2")).strip().lower()
        if penalty not in {"l1", "l2"}:
            penalty = "l2"
        class_weight = _resolve_class_weight(params.get("class_weight", None))
        if class_weight not in {None, "balanced"}:
            class_weight = None
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=c_val,
                        penalty=penalty,
                        class_weight=class_weight,
                        solver="liblinear",
                        max_iter=5000,
                        random_state=int(seed),
                    ),
                ),
            ]
        )
    if model_name == "random_forest":
        max_features = _resolve_max_features(params.get("max_features", "sqrt"), default="sqrt")
        class_weight = _resolve_class_weight(params.get("class_weight", None))
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 600)),
            max_depth=None if params.get("max_depth", None) in {None, 0} else int(params["max_depth"]),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            bootstrap=_resolve_bool_like(params.get("bootstrap", True), default=True),
            max_features=max_features,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=int(seed),
        )
    if model_name == "xgboost":
        XGBClassifier, _ = _load_xgboost_classes()
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=int(params.get("n_estimators", 800)),
            learning_rate=float(params.get("learning_rate", 0.03)),
            max_depth=int(params.get("max_depth", 6)),
            min_child_weight=float(params.get("min_child_weight", 1.0)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.9)),
            gamma=float(params.get("gamma", 0.0)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            scale_pos_weight=float(params.get("scale_pos_weight", 1.0)),
            max_bin=int(params.get("max_bin", 256)),
            random_state=int(seed),
            n_jobs=-1,
            tree_method="hist",
        )
    if model_name == "mlp":
        hidden_layers = tuple(int(x) for x in params.get("hidden_layers", (512, 256)))
        alpha = float(params.get("alpha", 1.0e-4))
        lr = float(params.get("learning_rate_init", 1.0e-3))
        activation = str(params.get("activation", "relu"))
        batch_size = _resolve_mlp_batch_size(params.get("batch_size", "auto"))
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=hidden_layers,
                        activation=activation,
                        batch_size=batch_size,
                        alpha=alpha,
                        learning_rate_init=lr,
                        max_iter=1000,
                        early_stopping=True,
                        n_iter_no_change=30,
                        random_state=int(seed),
                    ),
                ),
            ]
        )
    if model_name == "gpc":
        kernel_name = str(params.get("kernel", "rbf"))
        length_scale = float(params.get("length_scale", 1.0))
        constant_value = float(params.get("kernel_constant", 1.0))
        noise_level = float(params.get("noise_level", 1.0e-5))
        n_restarts_optimizer = int(params.get("n_restarts_optimizer", 0))
        max_iter_predict = int(params.get("max_iter_predict", 100))
        kernel = _sklearn_kernel_from_name(kernel_name, length_scale, constant_value=constant_value, noise_level=noise_level)
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    GaussianProcessClassifier(
                        kernel=kernel,
                        random_state=int(seed),
                        n_restarts_optimizer=n_restarts_optimizer,
                        max_iter_predict=max_iter_predict,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported classification model: {model_name}")


def _default_regression_params(model_name: str) -> Dict[str, object]:
    if model_name == "ridge":
        return {"alpha": 1.0, "fit_intercept": True}
    if model_name == "random_forest":
        return {
            "n_estimators": 600,
            "max_depth": 18,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 1.0,
            "bootstrap": True,
        }
    if model_name == "xgboost":
        return {
            "n_estimators": 800,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 1.0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "max_bin": 256,
        }
    if model_name == "mlp":
        return {
            "hidden_layers": (512, 256),
            "activation": "relu",
            "batch_size": 128,
            "alpha": 1.0e-4,
            "learning_rate_init": 1.0e-3,
        }
    if model_name == "gpr":
        return {
            "alpha": 1.0e-5,
            "kernel": "rbf",
            "length_scale": 1.0,
            "kernel_variance": 1.0,
            "max_iters": 200,
        }
    raise ValueError(f"Unsupported regression model: {model_name}")


def _default_classification_params(model_name: str) -> Dict[str, object]:
    if model_name == "logistic":
        return {"C": 1.0, "penalty": "l2", "class_weight": None}
    if model_name == "random_forest":
        return {
            "n_estimators": 600,
            "max_depth": 18,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "class_weight": None,
        }
    if model_name == "xgboost":
        return {
            "n_estimators": 800,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 1.0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "scale_pos_weight": 1.0,
            "max_bin": 256,
        }
    if model_name == "mlp":
        return {
            "hidden_layers": (512, 256),
            "activation": "relu",
            "batch_size": 128,
            "alpha": 1.0e-4,
            "learning_rate_init": 1.0e-3,
        }
    if model_name == "gpc":
        return {
            "kernel": "rbf",
            "length_scale": 1.0,
            "kernel_constant": 1.0,
            "noise_level": 1.0e-5,
            "n_restarts_optimizer": 0,
            "max_iter_predict": 100,
        }
    raise ValueError(f"Unsupported classification model: {model_name}")


def _sample_regression_params(trial, model_name: str, search_space: Dict[str, object]) -> Dict[str, object]:
    if model_name == "ridge":
        fit_intercept_options = _as_choice_options(search_space.get("ridge_fit_intercept"), default=[True, False])
        return {
            "alpha": _suggest_float(trial, "ridge_alpha", search_space, 1.0e-5, 1.0e3, log=True),
            "fit_intercept": _resolve_bool_like(
                trial.suggest_categorical("ridge_fit_intercept", fit_intercept_options),
                default=True,
            ),
        }
    if model_name == "random_forest":
        max_features_options = _as_choice_options(
            search_space.get("random_forest_max_features"),
            default=["sqrt", "log2", 0.5, 0.75, 1.0],
        )
        bootstrap_options = _as_choice_options(search_space.get("random_forest_bootstrap"), default=[True, False])
        return {
            "n_estimators": _suggest_int(trial, "random_forest_n_estimators", search_space, 200, 1200),
            "max_depth": _suggest_int(trial, "random_forest_max_depth", search_space, 3, 30),
            "min_samples_split": _suggest_int(trial, "random_forest_min_samples_split", search_space, 2, 20),
            "min_samples_leaf": _suggest_int(trial, "random_forest_min_samples_leaf", search_space, 1, 10),
            "max_features": trial.suggest_categorical("random_forest_max_features", max_features_options),
            "bootstrap": _resolve_bool_like(
                trial.suggest_categorical("random_forest_bootstrap", bootstrap_options),
                default=True,
            ),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": _suggest_int(trial, "xgboost_n_estimators", search_space, 200, 1500),
            "learning_rate": _suggest_float(trial, "xgboost_learning_rate", search_space, 1.0e-3, 0.2, log=True),
            "max_depth": _suggest_int(trial, "xgboost_max_depth", search_space, 3, 12),
            "min_child_weight": _suggest_float(trial, "xgboost_min_child_weight", search_space, 1.0, 12.0, log=False),
            "subsample": _suggest_float(trial, "xgboost_subsample", search_space, 0.5, 1.0, log=False),
            "colsample_bytree": _suggest_float(trial, "xgboost_colsample_bytree", search_space, 0.5, 1.0, log=False),
            "gamma": _suggest_float(trial, "xgboost_gamma", search_space, 0.0, 8.0, log=False),
            "reg_alpha": _suggest_float(trial, "xgboost_reg_alpha", search_space, 1.0e-8, 10.0, log=True),
            "reg_lambda": _suggest_float(trial, "xgboost_reg_lambda", search_space, 1.0e-8, 30.0, log=True),
            "max_bin": _suggest_int(trial, "xgboost_max_bin", search_space, 128, 512),
        }
    if model_name == "mlp":
        activation_options = _as_choice_options(search_space.get("mlp_activation"), default=["relu", "tanh"])
        batch_size_options = _as_choice_options(search_space.get("mlp_batch_size"), default=[32, 64, 128, 256])
        return {
            "hidden_layers": _sample_mlp_hidden_layers(trial, search_space=search_space),
            "activation": str(trial.suggest_categorical("mlp_activation", activation_options)),
            "batch_size": trial.suggest_categorical("mlp_batch_size", batch_size_options),
            "alpha": _suggest_float(trial, "mlp_alpha", search_space, 1.0e-7, 1.0e-2, log=True),
            "learning_rate_init": _suggest_float(trial, "mlp_learning_rate_init", search_space, 1.0e-5, 5.0e-3, log=True),
        }
    if model_name == "gpr":
        kernel_options = search_space.get("gpr_kernel", ["rbf", "matern15", "matern25"])
        if not isinstance(kernel_options, list) or len(kernel_options) == 0:
            kernel_options = ["rbf", "matern15", "matern25"]
        return {
            "alpha": _suggest_float(trial, "gpr_alpha", search_space, 1.0e-10, 1.0e-1, log=True),
            "kernel": trial.suggest_categorical("gpr_kernel", [str(x) for x in kernel_options]),
            "length_scale": _suggest_float(trial, "gpr_length_scale", search_space, 1.0e-2, 1.0e2, log=True),
            "kernel_variance": _suggest_float(trial, "gpr_kernel_variance", search_space, 1.0e-2, 10.0, log=True),
            "max_iters": _suggest_int(trial, "gpr_max_iters", search_space, 80, 400),
        }
    raise ValueError(f"Unsupported regression model: {model_name}")


def _sample_classification_params(trial, model_name: str, search_space: Dict[str, object]) -> Dict[str, object]:
    if model_name == "logistic":
        penalty_options = _as_choice_options(search_space.get("logistic_penalty"), default=["l2", "l1"])
        class_weight_options = _as_choice_options(search_space.get("logistic_class_weight"), default=["none", "balanced"])
        return {
            "C": _suggest_float(trial, "logistic_c", search_space, 1.0e-5, 1.0e3, log=True),
            "penalty": str(trial.suggest_categorical("logistic_penalty", penalty_options)),
            "class_weight": trial.suggest_categorical("logistic_class_weight", class_weight_options),
        }
    if model_name == "random_forest":
        max_features_options = _as_choice_options(
            search_space.get("random_forest_max_features"),
            default=["sqrt", "log2", 0.5, 0.75, 1.0],
        )
        bootstrap_options = _as_choice_options(search_space.get("random_forest_bootstrap"), default=[True, False])
        class_weight_options = _as_choice_options(
            search_space.get("random_forest_class_weight"),
            default=["none", "balanced", "balanced_subsample"],
        )
        return {
            "n_estimators": _suggest_int(trial, "random_forest_n_estimators", search_space, 200, 1200),
            "max_depth": _suggest_int(trial, "random_forest_max_depth", search_space, 3, 30),
            "min_samples_split": _suggest_int(trial, "random_forest_min_samples_split", search_space, 2, 20),
            "min_samples_leaf": _suggest_int(trial, "random_forest_min_samples_leaf", search_space, 1, 10),
            "max_features": trial.suggest_categorical("random_forest_max_features", max_features_options),
            "bootstrap": _resolve_bool_like(
                trial.suggest_categorical("random_forest_bootstrap", bootstrap_options),
                default=True,
            ),
            "class_weight": trial.suggest_categorical("random_forest_class_weight", class_weight_options),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": _suggest_int(trial, "xgboost_n_estimators", search_space, 200, 1500),
            "learning_rate": _suggest_float(trial, "xgboost_learning_rate", search_space, 1.0e-3, 0.2, log=True),
            "max_depth": _suggest_int(trial, "xgboost_max_depth", search_space, 3, 12),
            "min_child_weight": _suggest_float(trial, "xgboost_min_child_weight", search_space, 1.0, 12.0, log=False),
            "subsample": _suggest_float(trial, "xgboost_subsample", search_space, 0.5, 1.0, log=False),
            "colsample_bytree": _suggest_float(trial, "xgboost_colsample_bytree", search_space, 0.5, 1.0, log=False),
            "gamma": _suggest_float(trial, "xgboost_gamma", search_space, 0.0, 8.0, log=False),
            "reg_alpha": _suggest_float(trial, "xgboost_reg_alpha", search_space, 1.0e-8, 10.0, log=True),
            "reg_lambda": _suggest_float(trial, "xgboost_reg_lambda", search_space, 1.0e-8, 30.0, log=True),
            "scale_pos_weight": _suggest_float(trial, "xgboost_scale_pos_weight", search_space, 0.25, 5.0, log=True),
            "max_bin": _suggest_int(trial, "xgboost_max_bin", search_space, 128, 512),
        }
    if model_name == "mlp":
        activation_options = _as_choice_options(search_space.get("mlp_activation"), default=["relu", "tanh"])
        batch_size_options = _as_choice_options(search_space.get("mlp_batch_size"), default=[32, 64, 128, 256])
        return {
            "hidden_layers": _sample_mlp_hidden_layers(trial, search_space=search_space),
            "activation": str(trial.suggest_categorical("mlp_activation", activation_options)),
            "batch_size": trial.suggest_categorical("mlp_batch_size", batch_size_options),
            "alpha": _suggest_float(trial, "mlp_alpha", search_space, 1.0e-7, 1.0e-2, log=True),
            "learning_rate_init": _suggest_float(trial, "mlp_learning_rate_init", search_space, 1.0e-5, 5.0e-3, log=True),
        }
    if model_name == "gpc":
        kernel_options = search_space.get("gpc_kernel", ["rbf", "matern15", "matern25"])
        if not isinstance(kernel_options, list) or len(kernel_options) == 0:
            kernel_options = ["rbf", "matern15", "matern25"]
        return {
            "kernel": trial.suggest_categorical("gpc_kernel", [str(x) for x in kernel_options]),
            "length_scale": _suggest_float(trial, "gpc_length_scale", search_space, 1.0e-2, 1.0e2, log=True),
            "kernel_constant": _suggest_float(trial, "gpc_kernel_constant", search_space, 1.0e-2, 10.0, log=True),
            "noise_level": _suggest_float(trial, "gpc_noise_level", search_space, 1.0e-8, 1.0e-2, log=True),
            "n_restarts_optimizer": _suggest_int(trial, "gpc_n_restarts_optimizer", search_space, 0, 5),
            "max_iter_predict": _suggest_int(trial, "gpc_max_iter_predict", search_space, 80, 300),
        }
    raise ValueError(f"Unsupported classification model: {model_name}")


def _predict_class_probability(model, X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return np.zeros((0,), dtype=float)
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return np.asarray(p[:, 1], dtype=float)
        return np.asarray(p.ravel(), dtype=float)
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(X), dtype=float)
        return np.asarray(stable_sigmoid(score), dtype=float)
    pred = np.asarray(model.predict(X), dtype=float)
    return np.clip(pred, 0.0, 1.0)


def _evaluate_regression_cv(
    cv_folds: List[pd.DataFrame],
    fingerprint_table: np.ndarray,
    model_name: str,
    model_params: Dict[str, object],
    seed: int,
    collect_val_predictions: bool = False,
) -> Dict[str, object]:
    fold_rows: List[Dict[str, object]] = []
    val_frames: List[pd.DataFrame] = []
    for fold_id, fold_df in enumerate(cv_folds, start=1):
        train_df = fold_df[fold_df["split"] == "train"].copy().reset_index(drop=True)
        val_df = fold_df[fold_df["split"] == "val"].copy().reset_index(drop=True)
        if train_df.empty or val_df.empty:
            raise ValueError(f"Invalid CV fold={fold_id}: empty train or val.")

        X_train = features_from_table(train_df, fingerprint_table)
        y_train = train_df["chi"].to_numpy(dtype=float)
        X_val = features_from_table(val_df, fingerprint_table)
        y_val = val_df["chi"].to_numpy(dtype=float)

        model = _build_regression_estimator(model_name=model_name, params=model_params, seed=seed)
        model.fit(X_train, y_train)
        y_pred = np.asarray(model.predict(X_val), dtype=float)

        reg = regression_metrics(y_val, y_pred)
        fold_rows.append(
            {
                "fold": int(fold_id),
                "model_name": str(model_name),
                "val_n": int(len(y_val)),
                "val_r2": _safe_float(reg["r2"]),
                "val_rmse": _safe_float(reg["rmse"]),
                "val_mae": _safe_float(reg["mae"]),
            }
        )
        if collect_val_predictions:
            out = val_df.copy()
            out["chi_true"] = y_val
            out["chi_pred"] = y_pred
            out["fold"] = int(fold_id)
            val_frames.append(out[["fold", "polymer_id", "Polymer", "SMILES", "water_soluble", "chi_true", "chi_pred"]].copy())

    fold_metrics_df = pd.DataFrame(fold_rows)
    return {
        "cv_val_r2": float(np.nanmean(fold_metrics_df["val_r2"])) if not fold_metrics_df.empty else np.nan,
        "cv_val_rmse": float(np.nanmean(fold_metrics_df["val_rmse"])) if not fold_metrics_df.empty else np.nan,
        "cv_val_mae": float(np.nanmean(fold_metrics_df["val_mae"])) if not fold_metrics_df.empty else np.nan,
        "fold_metrics": fold_metrics_df,
        "cv_val_predictions": pd.concat(val_frames, ignore_index=True) if val_frames else pd.DataFrame(),
    }


def _evaluate_classification_cv(
    cv_folds: List[pd.DataFrame],
    fingerprint_table: np.ndarray,
    model_name: str,
    model_params: Dict[str, object],
    seed: int,
) -> Dict[str, object]:
    fold_rows: List[Dict[str, object]] = []
    for fold_id, fold_df in enumerate(cv_folds, start=1):
        train_df = fold_df[fold_df["split"] == "train"].copy().reset_index(drop=True)
        val_df = fold_df[fold_df["split"] == "val"].copy().reset_index(drop=True)
        if train_df.empty or val_df.empty:
            raise ValueError(f"Invalid CV fold={fold_id}: empty train or val.")
        if train_df["water_soluble"].nunique() < 2:
            raise ValueError(f"CV fold={fold_id} train has single class; cannot fit classifier.")

        X_train = features_from_table(train_df, fingerprint_table)
        y_train = train_df["water_soluble"].to_numpy(dtype=int)
        X_val = features_from_table(val_df, fingerprint_table)
        y_val = val_df["water_soluble"].to_numpy(dtype=int)

        model = _build_classification_estimator(model_name=model_name, params=model_params, seed=seed)
        model.fit(X_train, y_train)
        p_val = _predict_class_probability(model, X_val)
        cls = classification_metrics(y_val, p_val)
        fold_rows.append(
            {
                "fold": int(fold_id),
                "model_name": str(model_name),
                "val_n": int(len(y_val)),
                "val_balanced_accuracy": _safe_float(cls["balanced_accuracy"]),
                "val_auroc": _safe_float(cls["auroc"]),
                "val_auprc": _safe_float(cls["auprc"]),
                "val_brier": _safe_float(cls["brier"]),
            }
        )
    fold_metrics_df = pd.DataFrame(fold_rows)
    return {
        "cv_val_balanced_accuracy": float(np.nanmean(fold_metrics_df["val_balanced_accuracy"])) if not fold_metrics_df.empty else np.nan,
        "cv_val_auroc": float(np.nanmean(fold_metrics_df["val_auroc"])) if not fold_metrics_df.empty else np.nan,
        "cv_val_auprc": float(np.nanmean(fold_metrics_df["val_auprc"])) if not fold_metrics_df.empty else np.nan,
        "cv_val_brier": float(np.nanmean(fold_metrics_df["val_brier"])) if not fold_metrics_df.empty else np.nan,
        "fold_metrics": fold_metrics_df,
    }


def _save_cv_parity_by_fold_figure(cv_val_df: pd.DataFrame, out_png: Path, dpi: int, font_size: int) -> None:
    if cv_val_df.empty:
        return
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": font_size, "axes.titlesize": font_size, "axes.labelsize": font_size, "legend.fontsize": font_size})
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_df = cv_val_df.copy()
    plot_df["fold"] = plot_df["fold"].astype(str)
    n_folds = int(plot_df["fold"].nunique())
    palette = sns.color_palette("tab10", n_colors=max(n_folds, 3))
    sns.scatterplot(data=plot_df, x="chi_true", y="chi_pred", hue="fold", palette=palette, alpha=0.75, s=18, ax=ax)
    lo = float(min(plot_df["chi_true"].min(), plot_df["chi_pred"].min()))
    hi = float(max(plot_df["chi_true"].max(), plot_df["chi_pred"].max()))
    span = max(hi - lo, 1.0e-8)
    pad = 0.04 * span
    lo_plot = lo - pad
    hi_plot = hi + pad
    ax.plot([lo_plot, hi_plot], [lo_plot, hi_plot], "k--", linewidth=1.1)
    ax.set_xlim(lo_plot, hi_plot)
    ax.set_ylim(lo_plot, hi_plot)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("True chi")
    ax.set_ylabel("Predicted chi")
    reg = regression_metrics(plot_df["chi_true"], plot_df["chi_pred"])
    ax.text(
        0.03,
        0.97,
        f"MAE={reg['mae']:.3f}\nRMSE={reg['rmse']:.3f}\nR2={reg['r2']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.92},
    )
    ax.set_title(f"CV parity by fold (val folds, n={len(plot_df)})")
    ax.legend(title="CV fold", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _plot_optuna_objective(trial_df: pd.DataFrame, out_png: Path, objective_name: str, dpi: int, font_size: int, maximize: bool) -> None:
    if trial_df.empty:
        return
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": font_size, "axes.titlesize": font_size, "axes.labelsize": font_size, "legend.fontsize": font_size})
    fig, ax = plt.subplots(figsize=(6, 5))
    x = trial_df["trial"].to_numpy()
    y = pd.to_numeric(trial_df["objective_value"], errors="coerce").to_numpy()
    if maximize:
        best = pd.to_numeric(trial_df["objective_value"], errors="coerce").cummax().to_numpy()
    else:
        best = pd.to_numeric(trial_df["objective_value"], errors="coerce").cummin().to_numpy()
    ax.plot(x, y, "o", color="#2a9d8f", alpha=0.85, label=f"Trial {objective_name}")
    ax.plot(x, best, "-", color="#e76f51", linewidth=2, label="Best so far")
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("Objective value")
    ax.set_title(f"Optuna optimization objective: {objective_name}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _tune_regression(
    split_df: pd.DataFrame,
    fingerprint_table: np.ndarray,
    stage_cfg: StageConfig,
    tuning_dir: Path,
    dpi: int,
    font_size: int,
) -> Dict[str, object]:
    import optuna

    tuning_dir.mkdir(parents=True, exist_ok=True)
    cv_folds, cv_info = build_tuning_cv_folds(
        split_df=split_df,
        split_mode=stage_cfg.split_mode,
        tuning_cv_folds=stage_cfg.tuning_cv_folds,
        seed=stage_cfg.seed,
    )
    summarize_cv_folds(cv_folds).to_csv(tuning_dir / "optuna_tuning_cv_folds.csv", index=False)

    objective_name = str(stage_cfg.tuning_objective).strip().lower()
    trial_rows: List[Dict[str, object]] = []

    def objective(trial: optuna.Trial) -> float:
        model_name = trial.suggest_categorical("model_name", stage_cfg.models)
        params = _sample_regression_params(trial, model_name=model_name, search_space=stage_cfg.optuna_search_space)
        try:
            cv_eval = _evaluate_regression_cv(
                cv_folds=cv_folds,
                fingerprint_table=fingerprint_table,
                model_name=model_name,
                model_params=params,
                seed=stage_cfg.seed,
            )
            if objective_name == "val_r2":
                score = float(cv_eval["cv_val_r2"])
            else:
                raise ValueError(f"Unsupported regression tuning objective: {objective_name}")
            rmse = float(cv_eval["cv_val_rmse"])
            invalid = int((not np.isfinite(score)) or (not np.isfinite(rmse)))
        except Exception as exc:
            score = -1.0e12
            rmse = np.nan
            invalid = 1
            trial.set_user_attr("error", str(exc))

        trial.set_user_attr("model_name", model_name)
        trial.set_user_attr("model_params", _to_serializable(params))
        trial.set_user_attr("cv_val_r2", score)
        trial.set_user_attr("cv_val_rmse", rmse)
        trial.set_user_attr("cv_n_folds", int(len(cv_folds)))
        trial.set_user_attr("tuning_objective", objective_name)
        trial.set_user_attr("invalid_metrics", invalid)
        if invalid:
            return -1.0e12
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(stage_cfg.n_trials), show_progress_bar=True)

    for t in study.trials:
        row = {
            "trial": int(t.number),
            "state": str(t.state),
            "objective_name": objective_name,
            "objective_direction": "maximize",
            "objective_value": _safe_float(t.value),
            "model_name": str(t.user_attrs.get("model_name", t.params.get("model_name", ""))),
            "val_r2": _safe_float(t.user_attrs.get("cv_val_r2", np.nan)),
            "val_rmse": _safe_float(t.user_attrs.get("cv_val_rmse", np.nan)),
            "invalid_metrics": int(t.user_attrs.get("invalid_metrics", 0)),
            "cv_n_folds": int(t.user_attrs.get("cv_n_folds", len(cv_folds))),
        }
        row.update(t.params)
        trial_rows.append(row)
    if trial_rows:
        trial_df = pd.DataFrame(trial_rows).sort_values("trial").reset_index(drop=True)
    else:
        trial_df = pd.DataFrame(
            columns=[
                "trial",
                "state",
                "objective_name",
                "objective_direction",
                "objective_value",
                "model_name",
                "val_r2",
                "val_rmse",
                "invalid_metrics",
                "cv_n_folds",
            ]
        )
    trial_df.to_csv(tuning_dir / "optuna_trials.csv", index=False)
    trial_df["best_objective_so_far"] = pd.to_numeric(trial_df["objective_value"], errors="coerce").cummax()
    trial_df.to_csv(tuning_dir / "optuna_optimization_objective.csv", index=False)
    _plot_optuna_objective(
        trial_df=trial_df,
        out_png=tuning_dir / "optuna_optimization_objective.png",
        objective_name=objective_name,
        dpi=dpi,
        font_size=font_size,
        maximize=True,
    )

    valid_trials = []
    for t in study.trials:
        invalid = int(t.user_attrs.get("invalid_metrics", 0))
        try:
            value = float(t.value)
        except Exception:
            continue
        if invalid == 0 and np.isfinite(value):
            valid_trials.append(t)

    best_trial = max(valid_trials, key=lambda t: float(t.value)) if valid_trials else None
    fallback_reason = ""
    if best_trial is None:
        best_model_name = str(stage_cfg.models[0])
        best_params = _default_regression_params(best_model_name)
        objective_at_best = np.nan
        fallback_reason = "no_valid_optuna_trial_using_default_model"
    else:
        best_model_name = str(best_trial.user_attrs.get("model_name", best_trial.params.get("model_name", stage_cfg.models[0]))).strip().lower()
        params_attr = best_trial.user_attrs.get("model_params")
        if isinstance(params_attr, dict) and len(params_attr) > 0:
            best_params = dict(params_attr)
        else:
            best_params = {k: v for k, v in best_trial.params.items() if k != "model_name"}
        if len(best_params) == 0:
            best_params = _default_regression_params(best_model_name)
        objective_at_best = float(best_trial.value)

    best_params_by_model = _collect_best_params_by_model(study=study, model_names=stage_cfg.models)
    best_cv_error = ""
    try:
        best_cv = _evaluate_regression_cv(
            cv_folds=cv_folds,
            fingerprint_table=fingerprint_table,
            model_name=best_model_name,
            model_params=best_params,
            seed=stage_cfg.seed,
            collect_val_predictions=True,
        )
    except Exception as exc:
        best_cv = {"fold_metrics": pd.DataFrame(), "cv_val_predictions": pd.DataFrame()}
        best_cv_error = str(exc)

    best_cv["fold_metrics"].to_csv(tuning_dir / "best_trial_cv_fold_metrics.csv", index=False)
    best_cv_val = best_cv.get("cv_val_predictions", pd.DataFrame())
    if isinstance(best_cv_val, pd.DataFrame) and not best_cv_val.empty:
        best_cv_val.to_csv(tuning_dir / "best_trial_cv_val_predictions.csv", index=False)
        _save_cv_parity_by_fold_figure(
            cv_val_df=best_cv_val,
            out_png=tuning_dir / "cv_parity_by_fold.png",
            dpi=dpi,
            font_size=font_size,
        )

    with open(tuning_dir / "optuna_best.json", "w") as f:
        json.dump(
            {
                "best_trial": int(best_trial.number) if best_trial is not None else None,
                "objective": _objective_summary_name(objective_name),
                "objective_name": objective_name,
                "objective_direction": "maximize",
                "objective_value_at_best_trial": _safe_float(objective_at_best),
                "best_model_name": best_model_name,
                "best_params": best_params,
                "best_value_r2": _safe_float(best_trial.user_attrs.get("cv_val_r2", np.nan)) if best_trial is not None else np.nan,
                "best_value_rmse": _safe_float(best_trial.user_attrs.get("cv_val_rmse", np.nan)) if best_trial is not None else np.nan,
                "tuning_cv_folds_requested": int(stage_cfg.tuning_cv_folds),
                "tuning_cv_folds_resolved": int(cv_info.get("resolved_folds", len(cv_folds))),
                "tuning_cv_strategy": str(cv_info.get("strategy", "unknown")),
                "invalid_trial_count": int(trial_df["invalid_metrics"].sum()) if "invalid_metrics" in trial_df.columns else 0,
                "best_trial_number": int(best_trial.number) if best_trial is not None else None,
                "fallback_reason": fallback_reason,
                "best_cv_eval_error": best_cv_error,
            },
            f,
            indent=2,
        )
    return {
        "best_model_name": best_model_name,
        "best_params": best_params,
        "best_params_by_model": best_params_by_model,
        "cv_info": cv_info,
    }


def _tune_classification(
    split_df: pd.DataFrame,
    fingerprint_table: np.ndarray,
    stage_cfg: StageConfig,
    tuning_dir: Path,
    dpi: int,
    font_size: int,
) -> Dict[str, object]:
    import optuna

    tuning_dir.mkdir(parents=True, exist_ok=True)
    cv_folds, cv_info = build_tuning_cv_folds(
        split_df=split_df,
        split_mode=stage_cfg.split_mode,
        tuning_cv_folds=stage_cfg.tuning_cv_folds,
        seed=stage_cfg.seed,
    )
    summarize_cv_folds(cv_folds).to_csv(tuning_dir / "optuna_tuning_cv_folds.csv", index=False)

    objective_name = str(stage_cfg.tuning_objective).strip().lower()
    trial_rows: List[Dict[str, object]] = []

    def objective(trial: optuna.Trial) -> float:
        model_name = trial.suggest_categorical("model_name", stage_cfg.models)
        params = _sample_classification_params(trial, model_name=model_name, search_space=stage_cfg.optuna_search_space)
        try:
            cv_eval = _evaluate_classification_cv(
                cv_folds=cv_folds,
                fingerprint_table=fingerprint_table,
                model_name=model_name,
                model_params=params,
                seed=stage_cfg.seed,
            )
            if objective_name == "val_balanced_accuracy":
                score = float(cv_eval["cv_val_balanced_accuracy"])
            else:
                raise ValueError(f"Unsupported classification tuning objective: {objective_name}")
            auroc = float(cv_eval["cv_val_auroc"])
            invalid = int((not np.isfinite(score)) or (not np.isfinite(auroc)))
        except Exception as exc:
            score = -1.0e12
            auroc = np.nan
            invalid = 1
            trial.set_user_attr("error", str(exc))

        trial.set_user_attr("model_name", model_name)
        trial.set_user_attr("model_params", _to_serializable(params))
        trial.set_user_attr("cv_val_balanced_accuracy", score)
        trial.set_user_attr("cv_val_auroc", auroc)
        trial.set_user_attr("cv_n_folds", int(len(cv_folds)))
        trial.set_user_attr("tuning_objective", objective_name)
        trial.set_user_attr("invalid_metrics", invalid)
        if invalid:
            return -1.0e12
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(stage_cfg.n_trials), show_progress_bar=True)

    for t in study.trials:
        row = {
            "trial": int(t.number),
            "state": str(t.state),
            "objective_name": objective_name,
            "objective_direction": "maximize",
            "objective_value": _safe_float(t.value),
            "model_name": str(t.user_attrs.get("model_name", t.params.get("model_name", ""))),
            "val_balanced_accuracy": _safe_float(t.user_attrs.get("cv_val_balanced_accuracy", np.nan)),
            "val_auroc": _safe_float(t.user_attrs.get("cv_val_auroc", np.nan)),
            "invalid_metrics": int(t.user_attrs.get("invalid_metrics", 0)),
            "cv_n_folds": int(t.user_attrs.get("cv_n_folds", len(cv_folds))),
        }
        row.update(t.params)
        trial_rows.append(row)
    if trial_rows:
        trial_df = pd.DataFrame(trial_rows).sort_values("trial").reset_index(drop=True)
    else:
        trial_df = pd.DataFrame(
            columns=[
                "trial",
                "state",
                "objective_name",
                "objective_direction",
                "objective_value",
                "model_name",
                "val_balanced_accuracy",
                "val_auroc",
                "invalid_metrics",
                "cv_n_folds",
            ]
        )
    trial_df.to_csv(tuning_dir / "optuna_trials.csv", index=False)
    trial_df["best_objective_so_far"] = pd.to_numeric(trial_df["objective_value"], errors="coerce").cummax()
    trial_df.to_csv(tuning_dir / "optuna_optimization_objective.csv", index=False)
    _plot_optuna_objective(
        trial_df=trial_df,
        out_png=tuning_dir / "optuna_optimization_objective.png",
        objective_name=objective_name,
        dpi=dpi,
        font_size=font_size,
        maximize=True,
    )

    valid_trials = []
    for t in study.trials:
        invalid = int(t.user_attrs.get("invalid_metrics", 0))
        try:
            value = float(t.value)
        except Exception:
            continue
        if invalid == 0 and np.isfinite(value):
            valid_trials.append(t)

    best_trial = max(valid_trials, key=lambda t: float(t.value)) if valid_trials else None
    fallback_reason = ""
    if best_trial is None:
        best_model_name = str(stage_cfg.models[0])
        best_params = _default_classification_params(best_model_name)
        objective_at_best = np.nan
        fallback_reason = "no_valid_optuna_trial_using_default_model"
    else:
        best_model_name = str(best_trial.user_attrs.get("model_name", best_trial.params.get("model_name", stage_cfg.models[0]))).strip().lower()
        params_attr = best_trial.user_attrs.get("model_params")
        if isinstance(params_attr, dict) and len(params_attr) > 0:
            best_params = dict(params_attr)
        else:
            best_params = {k: v for k, v in best_trial.params.items() if k != "model_name"}
        if len(best_params) == 0:
            best_params = _default_classification_params(best_model_name)
        objective_at_best = float(best_trial.value)

    best_params_by_model = _collect_best_params_by_model(study=study, model_names=stage_cfg.models)
    best_cv_error = ""
    try:
        best_cv = _evaluate_classification_cv(
            cv_folds=cv_folds,
            fingerprint_table=fingerprint_table,
            model_name=best_model_name,
            model_params=best_params,
            seed=stage_cfg.seed,
        )
    except Exception as exc:
        best_cv = {"fold_metrics": pd.DataFrame()}
        best_cv_error = str(exc)

    best_cv["fold_metrics"].to_csv(tuning_dir / "best_trial_cv_fold_metrics.csv", index=False)

    with open(tuning_dir / "optuna_best.json", "w") as f:
        json.dump(
            {
                "best_trial": int(best_trial.number) if best_trial is not None else None,
                "objective": _objective_summary_name(objective_name),
                "objective_name": objective_name,
                "objective_direction": "maximize",
                "objective_value_at_best_trial": _safe_float(objective_at_best),
                "best_model_name": best_model_name,
                "best_params": best_params,
                "best_value_balanced_accuracy": _safe_float(best_trial.user_attrs.get("cv_val_balanced_accuracy", np.nan))
                if best_trial is not None
                else np.nan,
                "best_value_auroc": _safe_float(best_trial.user_attrs.get("cv_val_auroc", np.nan)) if best_trial is not None else np.nan,
                "tuning_cv_folds_requested": int(stage_cfg.tuning_cv_folds),
                "tuning_cv_folds_resolved": int(cv_info.get("resolved_folds", len(cv_folds))),
                "tuning_cv_strategy": str(cv_info.get("strategy", "unknown")),
                "invalid_trial_count": int(trial_df["invalid_metrics"].sum()) if "invalid_metrics" in trial_df.columns else 0,
                "best_trial_number": int(best_trial.number) if best_trial is not None else None,
                "fallback_reason": fallback_reason,
                "best_cv_eval_error": best_cv_error,
            },
            f,
            indent=2,
        )
    return {
        "best_model_name": best_model_name,
        "best_params": best_params,
        "best_params_by_model": best_params_by_model,
        "cv_info": cv_info,
    }


def _fit_regression_on_final_split(
    final_fit_df: pd.DataFrame,
    fingerprint_table: np.ndarray,
    model_name: str,
    model_params: Dict[str, object],
    seed: int,
):
    train_df = final_fit_df[final_fit_df["split"] == "train"].copy().reset_index(drop=True)
    X_train = features_from_table(train_df, fingerprint_table)
    y_train = train_df["chi"].to_numpy(dtype=float)
    model = _build_regression_estimator(model_name=model_name, params=model_params, seed=seed)
    model.fit(X_train, y_train)
    return model


def _fit_classification_on_final_split(
    final_fit_df: pd.DataFrame,
    fingerprint_table: np.ndarray,
    model_name: str,
    model_params: Dict[str, object],
    seed: int,
):
    train_df = final_fit_df[final_fit_df["split"] == "train"].copy().reset_index(drop=True)
    if train_df["water_soluble"].nunique() < 2:
        raise ValueError("Final train split has single class; classification training is not possible.")
    X_train = features_from_table(train_df, fingerprint_table)
    y_train = train_df["water_soluble"].to_numpy(dtype=int)
    model = _build_classification_estimator(model_name=model_name, params=model_params, seed=seed)
    model.fit(X_train, y_train)
    return model


def _save_regression_predictions(split_df: pd.DataFrame, fingerprint_table: np.ndarray, model, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for split in ["train", "val", "test"]:
        sub = split_df[split_df["split"] == split].copy().reset_index(drop=True)
        if len(sub) > 0:
            X = features_from_table(sub, fingerprint_table)
            y_pred = np.asarray(model.predict(X), dtype=float)
        else:
            y_pred = np.zeros((0,), dtype=float)
        sub["chi_pred"] = y_pred
        sub["chi_error"] = sub["chi_pred"] - sub["chi"]
        sub.to_csv(out_dir / f"chi_predictions_{split}.csv", index=False)
        frames.append(sub)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out_dir / "chi_predictions_all.csv", index=False)
    return all_df


def _collect_regression_metrics(pred_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    class_rows = []
    polymer_rows = []
    for split, sub in pred_df.groupby("split"):
        reg = regression_metrics(sub["chi"], sub["chi_pred"])
        hit = hit_metrics(sub["chi_error"], epsilons=[0.02, 0.05, 0.1, 0.2])
        row = {"split": split}
        row.update(reg)
        row.update(hit)
        rows.append(row)

        group_metrics = metrics_by_group(sub, y_true_col="chi", y_pred_col="chi_pred", group_col="water_soluble")
        group_metrics.insert(0, "split", split)
        class_rows.append(group_metrics)

        poly = (
            sub.groupby(["polymer_id", "Polymer"], as_index=False)[["chi", "chi_pred"]]
            .mean()
            .rename(columns={"chi": "chi_true_mean", "chi_pred": "chi_pred_mean"})
        )
        poly_metric = regression_metrics(poly["chi_true_mean"], poly["chi_pred_mean"], prefix="poly_")
        poly_row = {"split": split}
        poly_row.update(poly_metric)
        polymer_rows.append(poly_row)

    pd.DataFrame(rows).to_csv(out_dir / "chi_metrics_overall.csv", index=False)
    if class_rows:
        pd.concat(class_rows, ignore_index=True).to_csv(out_dir / "chi_metrics_by_class.csv", index=False)
    else:
        pd.DataFrame(columns=["split", "group"]).to_csv(out_dir / "chi_metrics_by_class.csv", index=False)
    pd.DataFrame(polymer_rows).to_csv(out_dir / "chi_metrics_polymer_level.csv", index=False)


def _make_regression_figures(pred_df: pd.DataFrame, fig_dir: Path, dpi: int, font_size: int) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": font_size, "axes.titlesize": font_size, "axes.labelsize": font_size, "legend.fontsize": font_size})

    test_df = pred_df[pred_df["split"] == "test"].copy().reset_index(drop=True)
    if test_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=test_df,
        x="chi",
        y="chi_pred",
        hue="water_soluble",
        palette={1: "#1f77b4", 0: "#d62728"},
        alpha=0.75,
        s=18,
        ax=ax,
        legend=True,
    )
    lo = float(min(test_df["chi"].min(), test_df["chi_pred"].min()))
    hi = float(max(test_df["chi"].max(), test_df["chi_pred"].max()))
    span = max(hi - lo, 1.0e-8)
    pad = 0.04 * span
    lo_plot = lo - pad
    hi_plot = hi + pad
    ax.plot([lo_plot, hi_plot], [lo_plot, hi_plot], "k--", linewidth=1.1)
    ax.set_xlim(lo_plot, hi_plot)
    ax.set_ylim(lo_plot, hi_plot)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("True chi")
    ax.set_ylabel("Predicted chi")
    ax.set_title("chi parity (test)")
    reg = regression_metrics(test_df["chi"], test_df["chi_pred"])
    metrics_text = f"MAE={reg['mae']:.3f}\nRMSE={reg['rmse']:.3f}\nR2={reg['r2']:.3f}"
    ax.text(
        0.03,
        0.97,
        metrics_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.92},
    )
    legend = ax.get_legend()
    if legend is not None:
        handles = getattr(legend, "legend_handles", None)
        if handles is None:
            handles = legend.legendHandles
        labels = [t.get_text() for t in legend.get_texts()]
        legend.remove()
        ax.legend(
            handles=handles,
            labels=labels,
            title="water_soluble",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(fig_dir / "chi_parity_test.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    plotted_any = False
    for split, color in [("train", "#4c78a8"), ("val", "#f58518"), ("test", "#54a24b")]:
        sub = pred_df[pred_df["split"] == split]
        plotted_any = _plot_kde_safe_1d(
            ax=ax,
            values=sub["chi_error"],
            label=split,
            color=color,
            linewidth=2.0,
            fill=False,
            alpha=0.3,
        ) or plotted_any
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("chi prediction error")
    ax.set_title("Residual distribution by split")
    if plotted_any:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    else:
        ax.text(0.5, 0.5, "Insufficient residual variance for KDE", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(fig_dir / "chi_residual_distribution.png", dpi=dpi)
    plt.close(fig)


def _save_classification_predictions(split_df: pd.DataFrame, fingerprint_table: np.ndarray, model, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for split in ["train", "val", "test"]:
        sub = split_df[split_df["split"] == split].copy().reset_index(drop=True)
        if len(sub) > 0:
            X = features_from_table(sub, fingerprint_table)
            p = np.asarray(_predict_class_probability(model, X), dtype=float)
        else:
            p = np.zeros((0,), dtype=float)
        p_clip = np.clip(p, 1.0e-7, 1.0 - 1.0e-7)
        sub["class_prob"] = p
        sub["class_logit"] = np.log(p_clip / (1.0 - p_clip))
        sub["class_pred"] = (sub["class_prob"] >= 0.5).astype(int)
        sub.to_csv(out_dir / f"class_predictions_{split}.csv", index=False)
        frames.append(sub)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out_dir / "class_predictions_all.csv", index=False)
    return all_df


def _collect_classification_metrics(pred_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, sub in pred_df.groupby("split"):
        cls = classification_metrics(sub["water_soluble"], sub["class_prob"])
        row = {"split": split}
        row.update(cls)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "class_metrics_overall.csv", index=False)


def _make_classification_figures(pred_df: pd.DataFrame, fig_dir: Path, dpi: int, font_size: int) -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": font_size, "axes.titlesize": font_size, "axes.labelsize": font_size, "legend.fontsize": font_size})

    test_df = pred_df[pred_df["split"] == "test"].copy().reset_index(drop=True)
    if test_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_class_prob_density_safe(ax=ax, test_df=test_df)
    ax.set_xlabel("Predicted soluble probability")
    ax.set_title("Class probability distribution (test)")
    legend = ax.get_legend()
    if legend is not None:
        handles = getattr(legend, "legend_handles", None)
        if handles is None:
            handles = legend.legendHandles
        labels = [t.get_text() for t in legend.get_texts()]
        legend.remove()
        ax.legend(
            handles=handles,
            labels=labels,
            title="water_soluble",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(fig_dir / "class_prob_distribution_test.png", dpi=dpi)
    plt.close(fig)

    y_true = test_df["water_soluble"].to_numpy(dtype=int)
    y_prob = test_df["class_prob"].to_numpy(dtype=float)
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#1f77b4", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("Classifier ROC (test)")
        fig.tight_layout()
        fig.savefig(fig_dir / "classifier_roc_test.png", dpi=dpi)
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, color="#d62728", linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Classifier PR (test)")
        fig.tight_layout()
        fig.savefig(fig_dir / "classifier_pr_test.png", dpi=dpi)
        plt.close(fig)


def _resolve_stage_config(
    stage_section: Dict,
    shared_split_mode: str,
    shared_holdout: Optional[float],
    seed: int,
    default_tuning_objective: str,
    validate_dependencies: bool = True,
    tune_override: Optional[bool] = None,
    n_trials_override: Optional[int] = None,
    tuning_cv_folds_override: Optional[int] = None,
) -> StageConfig:
    tuning_cv_folds = int(
        tuning_cv_folds_override
        if tuning_cv_folds_override is not None
        else stage_section.get("tuning_cv_folds", 6)
    )
    if tuning_cv_folds < 2:
        raise ValueError("tuning_cv_folds must be >= 2")
    if shared_holdout is None:
        holdout_test_ratio = 1.0 / float(tuning_cv_folds)
    else:
        holdout_test_ratio = float(shared_holdout)

    tune_default = bool(stage_section.get("tune", True))
    tune = bool(tune_default if tune_override is None else tune_override)
    n_trials = int(n_trials_override if n_trials_override is not None else stage_section.get("n_trials", 100))
    objective_raw = stage_section.get("tuning_objective", default_tuning_objective)
    tuning_objective = _normalize_tuning_objective(objective_raw, default_tuning_objective=default_tuning_objective)

    models = stage_section.get("models", [])
    if not isinstance(models, list) or len(models) == 0:
        raise ValueError("Stage config requires a non-empty 'models' list.")
    models = [str(x).strip().lower() for x in models]
    if validate_dependencies:
        _check_xgboost_required(models)
        _check_gpy_required(models)

    optuna_search_space = stage_section.get("optuna_search_space", {})
    if not isinstance(optuna_search_space, dict):
        optuna_search_space = {}

    return StageConfig(
        split_mode=normalize_split_mode(shared_split_mode),
        holdout_test_ratio=float(holdout_test_ratio),
        seed=int(seed),
        tune=bool(tune),
        n_trials=int(n_trials),
        tuning_cv_folds=int(tuning_cv_folds),
        tuning_objective=str(tuning_objective),
        models=models,
        optuna_search_space=dict(optuna_search_space),
    )


def _to_serializable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def _collect_best_params_by_model(study, model_names: List[str]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    normalized = [str(m).strip().lower() for m in model_names]
    for model_name in normalized:
        best_trial = None
        best_value = -np.inf
        for t in study.trials:
            trial_model = str(t.user_attrs.get("model_name", "")).strip().lower()
            if trial_model != model_name:
                continue
            invalid = int(t.user_attrs.get("invalid_metrics", 0))
            if invalid != 0:
                continue
            value = t.value
            if value is None:
                continue
            try:
                score = float(value)
            except Exception:
                continue
            if not np.isfinite(score):
                continue
            if (best_trial is None) or (score > best_value):
                best_trial = t
                best_value = score
        if best_trial is None:
            continue
        params = best_trial.user_attrs.get("model_params")
        if not isinstance(params, dict):
            params = {k: v for k, v in best_trial.params.items() if k != "model_name"}
        out[model_name] = dict(params)
    return out


def _run_single_regression_model(
    *,
    split_df: pd.DataFrame,
    split_assign: pd.DataFrame,
    fingerprint_table: np.ndarray,
    model_name: str,
    model_params: Dict[str, object],
    seed: int,
    out_dir: Path,
    checkpoint_basename: str,
    dpi: int,
    font_size: int,
) -> Dict[str, object]:
    metrics_dir = out_dir / "metrics"
    figures_dir = out_dir / "figures"
    checkpoint_dir = out_dir / "checkpoints"
    for d in [metrics_dir, figures_dir, checkpoint_dir]:
        d.mkdir(parents=True, exist_ok=True)

    split_assign.to_csv(metrics_dir / "split_assignments.csv", index=False)
    split_df.to_csv(metrics_dir / "chi_dataset_with_split.csv", index=False)

    final_fit_df = build_final_fit_split_df(split_df)
    model = _fit_regression_on_final_split(
        final_fit_df=final_fit_df,
        fingerprint_table=fingerprint_table,
        model_name=model_name,
        model_params=model_params,
        seed=seed,
    )
    checkpoint_path = checkpoint_dir / checkpoint_basename
    joblib.dump(model, checkpoint_path)
    with open(checkpoint_dir / f"{Path(checkpoint_basename).stem}.meta.json", "w") as f:
        json.dump({"model_name": model_name, "params": _to_serializable(model_params)}, f, indent=2)

    pred_df = _save_regression_predictions(
        split_df=final_fit_df,
        fingerprint_table=fingerprint_table,
        model=model,
        out_dir=metrics_dir,
    )
    _collect_regression_metrics(pred_df, out_dir=metrics_dir)
    _make_regression_figures(pred_df, fig_dir=figures_dir, dpi=dpi, font_size=font_size)
    save_artifact_manifest(step_dir=out_dir, metrics_dir=metrics_dir, figures_dir=figures_dir, dpi=dpi)

    overall_df = pd.read_csv(metrics_dir / "chi_metrics_overall.csv")
    test_row = overall_df[overall_df["split"] == "test"]
    test_r2 = float(test_row["r2"].iloc[0]) if not test_row.empty and "r2" in test_row.columns else np.nan
    test_rmse = float(test_row["rmse"].iloc[0]) if not test_row.empty and "rmse" in test_row.columns else np.nan
    test_mae = float(test_row["mae"].iloc[0]) if not test_row.empty and "mae" in test_row.columns else np.nan
    return {
        "metrics_dir": str(metrics_dir),
        "figures_dir": str(figures_dir),
        "checkpoint": str(checkpoint_path),
        "test_r2": _safe_float(test_r2),
        "test_rmse": _safe_float(test_rmse),
        "test_mae": _safe_float(test_mae),
    }


def _run_single_classification_model(
    *,
    split_df: pd.DataFrame,
    split_assign: pd.DataFrame,
    fingerprint_table: np.ndarray,
    model_name: str,
    model_params: Dict[str, object],
    seed: int,
    out_dir: Path,
    checkpoint_basename: str,
    dpi: int,
    font_size: int,
) -> Dict[str, object]:
    metrics_dir = out_dir / "metrics"
    figures_dir = out_dir / "figures"
    checkpoint_dir = out_dir / "checkpoints"
    for d in [metrics_dir, figures_dir, checkpoint_dir]:
        d.mkdir(parents=True, exist_ok=True)

    split_assign.to_csv(metrics_dir / "split_assignments.csv", index=False)
    split_df.to_csv(metrics_dir / "chi_dataset_with_split.csv", index=False)

    final_fit_df = build_final_fit_split_df(split_df)
    model = _fit_classification_on_final_split(
        final_fit_df=final_fit_df,
        fingerprint_table=fingerprint_table,
        model_name=model_name,
        model_params=model_params,
        seed=seed,
    )
    checkpoint_path = checkpoint_dir / checkpoint_basename
    joblib.dump(model, checkpoint_path)
    with open(checkpoint_dir / f"{Path(checkpoint_basename).stem}.meta.json", "w") as f:
        json.dump({"model_name": model_name, "params": _to_serializable(model_params)}, f, indent=2)

    pred_df = _save_classification_predictions(
        split_df=final_fit_df,
        fingerprint_table=fingerprint_table,
        model=model,
        out_dir=metrics_dir,
    )
    _collect_classification_metrics(pred_df, out_dir=metrics_dir)
    _make_classification_figures(pred_df, fig_dir=figures_dir, dpi=dpi, font_size=font_size)
    save_artifact_manifest(step_dir=out_dir, metrics_dir=metrics_dir, figures_dir=figures_dir, dpi=dpi)

    overall_df = pd.read_csv(metrics_dir / "class_metrics_overall.csv")
    test_row = overall_df[overall_df["split"] == "test"]
    test_bal = (
        float(test_row["balanced_accuracy"].iloc[0])
        if not test_row.empty and "balanced_accuracy" in test_row.columns
        else np.nan
    )
    test_auroc = float(test_row["auroc"].iloc[0]) if not test_row.empty and "auroc" in test_row.columns else np.nan
    test_f1 = float(test_row["f1"].iloc[0]) if not test_row.empty and "f1" in test_row.columns else np.nan
    return {
        "metrics_dir": str(metrics_dir),
        "figures_dir": str(figures_dir),
        "checkpoint": str(checkpoint_path),
        "test_balanced_accuracy": _safe_float(test_bal),
        "test_auroc": _safe_float(test_auroc),
        "test_f1": _safe_float(test_f1),
    }


def _run_regression_stage(
    *,
    split_df: pd.DataFrame,
    split_assign: pd.DataFrame,
    fingerprint_table: np.ndarray,
    stage_cfg: StageConfig,
    reg_dir: Path,
    dpi: int,
    font_size: int,
) -> Dict[str, object]:
    metrics_dir = reg_dir / "metrics"
    figures_dir = reg_dir / "figures"
    tuning_dir = reg_dir / "tuning"
    checkpoint_dir = reg_dir / "checkpoints"
    models_dir = reg_dir / "models"
    for d in [metrics_dir, figures_dir, tuning_dir, checkpoint_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    split_assign.to_csv(metrics_dir / "split_assignments.csv", index=False)
    split_df.to_csv(metrics_dir / "chi_dataset_with_split.csv", index=False)

    if stage_cfg.tune:
        tuned = _tune_regression(
            split_df=split_df,
            fingerprint_table=fingerprint_table,
            stage_cfg=stage_cfg,
            tuning_dir=tuning_dir,
            dpi=dpi,
            font_size=font_size,
        )
        best_model_name = str(tuned["best_model_name"])
        best_params = dict(tuned["best_params"])
        best_params_by_model = dict(tuned.get("best_params_by_model", {}))
        hyper_summary = {
            "selection_mode": "optuna",
            "objective": _objective_summary_name(stage_cfg.tuning_objective),
            "objective_name": stage_cfg.tuning_objective,
            "best_model_name": best_model_name,
            "best_params": _to_serializable(best_params),
            "tuning_cv_folds": int(stage_cfg.tuning_cv_folds),
        }
    else:
        best_model_name = str(stage_cfg.models[0])
        best_params = _default_regression_params(best_model_name)
        best_params_by_model = {str(m): _default_regression_params(str(m)) for m in stage_cfg.models}
        hyper_summary = {
            "selection_mode": "default_no_tuning",
            "objective_name": stage_cfg.tuning_objective,
            "best_model_name": best_model_name,
            "best_params": _to_serializable(best_params),
            "tuning_cv_folds": int(stage_cfg.tuning_cv_folds),
        }

    with open(metrics_dir / "chosen_hyperparameters.json", "w") as f:
        json.dump({"model_name": best_model_name, "params": _to_serializable(best_params)}, f, indent=2)
    with open(metrics_dir / "hyperparameter_selection_summary.json", "w") as f:
        json.dump(hyper_summary, f, indent=2)

    model_outputs: List[Dict[str, object]] = []
    for model_name in stage_cfg.models:
        params = best_params_by_model.get(str(model_name), None)
        if not isinstance(params, dict):
            params = _default_regression_params(str(model_name))
            param_source = "default_fallback"
        else:
            param_source = "optuna_best_per_model" if stage_cfg.tune else "default_no_tuning"
        params = dict(params)
        model_dir = models_dir / str(model_name)
        out = _run_single_regression_model(
            split_df=split_df,
            split_assign=split_assign,
            fingerprint_table=fingerprint_table,
            model_name=str(model_name),
            model_params=params,
            seed=stage_cfg.seed,
            out_dir=model_dir,
            checkpoint_basename=f"chi_regression_traditional_{model_name}.joblib",
            dpi=dpi,
            font_size=font_size,
        )
        model_outputs.append(
            {
                "model_name": str(model_name),
                "param_source": param_source,
                "params": _to_serializable(params),
                "metrics_dir": out["metrics_dir"],
                "figures_dir": out["figures_dir"],
                "checkpoint": out["checkpoint"],
                "test_r2": out["test_r2"],
                "test_rmse": out["test_rmse"],
                "test_mae": out["test_mae"],
            }
        )

    best_out = _run_single_regression_model(
        split_df=split_df,
        split_assign=split_assign,
        fingerprint_table=fingerprint_table,
        model_name=best_model_name,
        model_params=best_params,
        seed=stage_cfg.seed,
        out_dir=reg_dir,
        checkpoint_basename="chi_regression_traditional_best.joblib",
        dpi=dpi,
        font_size=font_size,
    )

    with open(metrics_dir / "model_hyperparameters_by_model.json", "w") as f:
        json.dump({x["model_name"]: x["params"] for x in model_outputs}, f, indent=2)
    pd.DataFrame(
        [
            {
                "model_name": x["model_name"],
                "param_source": x["param_source"],
                "metrics_dir": x["metrics_dir"],
                "figures_dir": x["figures_dir"],
                "checkpoint": x["checkpoint"],
                "test_r2": x["test_r2"],
                "test_rmse": x["test_rmse"],
                "test_mae": x["test_mae"],
            }
            for x in model_outputs
        ]
    ).to_csv(metrics_dir / "model_metrics_summary.csv", index=False)

    return {
        "metrics_dir": best_out["metrics_dir"],
        "figures_dir": best_out["figures_dir"],
        "checkpoint": best_out["checkpoint"],
        "models_dir": str(models_dir),
        "best_model_name": best_model_name,
        "test_r2": _safe_float(best_out["test_r2"]),
        "test_rmse": _safe_float(best_out["test_rmse"]),
    }


def _run_classification_stage(
    *,
    split_df: pd.DataFrame,
    split_assign: pd.DataFrame,
    fingerprint_table: np.ndarray,
    stage_cfg: StageConfig,
    cls_dir: Path,
    dpi: int,
    font_size: int,
) -> Dict[str, object]:
    metrics_dir = cls_dir / "metrics"
    figures_dir = cls_dir / "figures"
    tuning_dir = cls_dir / "tuning"
    checkpoint_dir = cls_dir / "checkpoints"
    models_dir = cls_dir / "models"
    for d in [metrics_dir, figures_dir, tuning_dir, checkpoint_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    split_assign.to_csv(metrics_dir / "split_assignments.csv", index=False)
    split_df.to_csv(metrics_dir / "chi_dataset_with_split.csv", index=False)

    if stage_cfg.tune:
        tuned = _tune_classification(
            split_df=split_df,
            fingerprint_table=fingerprint_table,
            stage_cfg=stage_cfg,
            tuning_dir=tuning_dir,
            dpi=dpi,
            font_size=font_size,
        )
        best_model_name = str(tuned["best_model_name"])
        best_params = dict(tuned["best_params"])
        best_params_by_model = dict(tuned.get("best_params_by_model", {}))
        hyper_summary = {
            "selection_mode": "optuna",
            "objective": _objective_summary_name(stage_cfg.tuning_objective),
            "objective_name": stage_cfg.tuning_objective,
            "best_model_name": best_model_name,
            "best_params": _to_serializable(best_params),
            "tuning_cv_folds": int(stage_cfg.tuning_cv_folds),
        }
    else:
        best_model_name = str(stage_cfg.models[0])
        best_params = _default_classification_params(best_model_name)
        best_params_by_model = {str(m): _default_classification_params(str(m)) for m in stage_cfg.models}
        hyper_summary = {
            "selection_mode": "default_no_tuning",
            "objective_name": stage_cfg.tuning_objective,
            "best_model_name": best_model_name,
            "best_params": _to_serializable(best_params),
            "tuning_cv_folds": int(stage_cfg.tuning_cv_folds),
        }

    with open(metrics_dir / "chosen_hyperparameters.json", "w") as f:
        json.dump({"model_name": best_model_name, "params": _to_serializable(best_params)}, f, indent=2)
    with open(metrics_dir / "hyperparameter_selection_summary.json", "w") as f:
        json.dump(hyper_summary, f, indent=2)

    model_outputs: List[Dict[str, object]] = []
    for model_name in stage_cfg.models:
        params = best_params_by_model.get(str(model_name), None)
        if not isinstance(params, dict):
            params = _default_classification_params(str(model_name))
            param_source = "default_fallback"
        else:
            param_source = "optuna_best_per_model" if stage_cfg.tune else "default_no_tuning"
        params = dict(params)
        model_dir = models_dir / str(model_name)
        out = _run_single_classification_model(
            split_df=split_df,
            split_assign=split_assign,
            fingerprint_table=fingerprint_table,
            model_name=str(model_name),
            model_params=params,
            seed=stage_cfg.seed,
            out_dir=model_dir,
            checkpoint_basename=f"chi_classifier_traditional_{model_name}.joblib",
            dpi=dpi,
            font_size=font_size,
        )
        model_outputs.append(
            {
                "model_name": str(model_name),
                "param_source": param_source,
                "params": _to_serializable(params),
                "metrics_dir": out["metrics_dir"],
                "figures_dir": out["figures_dir"],
                "checkpoint": out["checkpoint"],
                "test_balanced_accuracy": out["test_balanced_accuracy"],
                "test_auroc": out["test_auroc"],
                "test_f1": out["test_f1"],
            }
        )

    best_out = _run_single_classification_model(
        split_df=split_df,
        split_assign=split_assign,
        fingerprint_table=fingerprint_table,
        model_name=best_model_name,
        model_params=best_params,
        seed=stage_cfg.seed,
        out_dir=cls_dir,
        checkpoint_basename="chi_classifier_traditional_best.joblib",
        dpi=dpi,
        font_size=font_size,
    )

    with open(metrics_dir / "model_hyperparameters_by_model.json", "w") as f:
        json.dump({x["model_name"]: x["params"] for x in model_outputs}, f, indent=2)
    pd.DataFrame(
        [
            {
                "model_name": x["model_name"],
                "param_source": x["param_source"],
                "metrics_dir": x["metrics_dir"],
                "figures_dir": x["figures_dir"],
                "checkpoint": x["checkpoint"],
                "test_balanced_accuracy": x["test_balanced_accuracy"],
                "test_auroc": x["test_auroc"],
                "test_f1": x["test_f1"],
            }
            for x in model_outputs
        ]
    ).to_csv(metrics_dir / "model_metrics_summary.csv", index=False)

    return {
        "metrics_dir": best_out["metrics_dir"],
        "figures_dir": best_out["figures_dir"],
        "checkpoint": best_out["checkpoint"],
        "models_dir": str(models_dir),
        "best_model_name": best_model_name,
        "test_balanced_accuracy": _safe_float(best_out["test_balanced_accuracy"]),
        "test_auroc": _safe_float(best_out["test_auroc"]),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 4_3: traditional regression + classification on Morgan fingerprints")
    parser.add_argument("--config", type=str, default="traditional_step4/configs/config_traditional.yaml")
    parser.add_argument(
        "--stage",
        type=str,
        default="both",
        choices=["both", "step4_3_1", "step4_3_2"],
        help="Run both substeps or only one.",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default=None,
        choices=["polymer", "random"],
        help="Optional split mode override (otherwise uses config traditional_step4.shared.split_mode).",
    )
    parser.add_argument("--regression_dataset_path", type=str, default=None)
    parser.add_argument("--classification_dataset_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--tune", action="store_true", help="Force-enable Optuna tuning.")
    parser.add_argument("--no_tune", action="store_true", help="Disable Optuna tuning.")
    parser.add_argument("--n_trials", type=int, default=None, help="Override Optuna n_trials for both substeps.")
    parser.add_argument("--tuning_cv_folds", type=int, default=None, help="Override tuning CV folds for both substeps.")
    parser.add_argument("--fingerprint_radius", type=int, default=None, help="Override Morgan radius.")
    parser.add_argument("--fingerprint_n_bits", type=int, default=None, help="Override Morgan nBits.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.tune and args.no_tune:
        raise ValueError("Cannot set both --tune and --no_tune.")
    run_reg = args.stage in {"both", "step4_3_1"}
    run_cls = args.stage in {"both", "step4_3_2"}

    config = load_traditional_config(args.config)
    trad_cfg = config.get("traditional_step4", {})
    if not isinstance(trad_cfg, dict):
        raise ValueError("Missing 'traditional_step4' section in config_traditional.yaml")

    paths_cfg = config.get("paths", {})
    results_root = str(paths_cfg.get("results_root", "traditional_step4"))
    data_cfg = config.get("data", {})
    seed = int(args.seed if args.seed is not None else data_cfg.get("random_seed", 42))

    shared_cfg = trad_cfg.get("shared", {})
    if not isinstance(shared_cfg, dict):
        shared_cfg = {}
    split_cfg = shared_cfg.get("split", {}) if isinstance(shared_cfg.get("split", {}), dict) else {}
    fp_cfg_raw = shared_cfg.get("fingerprint", {}) if isinstance(shared_cfg.get("fingerprint", {}), dict) else {}

    split_mode = normalize_split_mode(args.split_mode or shared_cfg.get("split_mode", "polymer"))
    holdout_test_ratio = split_cfg.get("holdout_test_ratio", None)
    tune_override = True if args.tune else (False if args.no_tune else None)

    fp_cfg = FingerprintConfig(
        radius=int(args.fingerprint_radius if args.fingerprint_radius is not None else fp_cfg_raw.get("radius", 3)),
        n_bits=int(args.fingerprint_n_bits if args.fingerprint_n_bits is not None else fp_cfg_raw.get("n_bits", 1024)),
        use_chirality=bool(fp_cfg_raw.get("use_chirality", False)),
        use_features=bool(fp_cfg_raw.get("use_features", False)),
    )

    reg_cfg_section = trad_cfg.get("step4_3_1_regression", {})
    cls_cfg_section = trad_cfg.get("step4_3_2_classification", {})
    if not isinstance(reg_cfg_section, dict) or not isinstance(cls_cfg_section, dict):
        raise ValueError("step4_3_1_regression and step4_3_2_classification must be mapping objects.")

    reg_cfg: Optional[StageConfig] = None
    cls_cfg: Optional[StageConfig] = None
    reg_split_ratios: Optional[Dict[str, float]] = None
    cls_split_ratios: Optional[Dict[str, float]] = None

    if run_reg:
        reg_cfg = _resolve_stage_config(
            stage_section=reg_cfg_section,
            shared_split_mode=split_mode,
            shared_holdout=holdout_test_ratio,
            seed=seed,
            default_tuning_objective="val_r2",
            validate_dependencies=True,
            tune_override=tune_override,
            n_trials_override=args.n_trials,
            tuning_cv_folds_override=args.tuning_cv_folds,
        )
        reg_split_ratios = resolve_split_ratios(reg_cfg.holdout_test_ratio, reg_cfg.tuning_cv_folds)

    if run_cls:
        cls_cfg = _resolve_stage_config(
            stage_section=cls_cfg_section,
            shared_split_mode=split_mode,
            shared_holdout=holdout_test_ratio,
            seed=seed,
            default_tuning_objective="val_balanced_accuracy",
            validate_dependencies=True,
            tune_override=tune_override,
            n_trials_override=args.n_trials,
            tuning_cv_folds_override=args.tuning_cv_folds,
        )
        cls_split_ratios = resolve_split_ratios(cls_cfg.holdout_test_ratio, cls_cfg.tuning_cv_folds)

    reg_dataset = args.regression_dataset_path or shared_cfg.get("regression_dataset_path", "Data/chi/_50_polymers_T_phi.csv")
    cls_dataset = args.classification_dataset_path or shared_cfg.get(
        "classification_dataset_path", "Data/water_solvent/water_solvent_polymers.csv"
    )

    results_dir = get_traditional_results_dir(results_root=results_root, split_mode=split_mode)
    step_dir = results_dir / "step4_3_traditional" / split_mode
    shared_dir = step_dir / "shared"
    reg_dir = step_dir / "step4_3_1_regression"
    cls_dir = step_dir / "step4_3_2_classification"
    pipeline_metrics_dir = step_dir / "pipeline_metrics"
    pipeline_metrics_run_dir = pipeline_metrics_dir if args.stage == "both" else (pipeline_metrics_dir / args.stage)
    for d in [results_dir, step_dir, shared_dir, reg_dir, cls_dir, pipeline_metrics_dir, pipeline_metrics_run_dir]:
        d.mkdir(parents=True, exist_ok=True)

    effective_config = copy.deepcopy(config)
    effective_data_cfg = effective_config.setdefault("data", {})
    if isinstance(effective_data_cfg, dict):
        effective_data_cfg["random_seed"] = int(seed)
    effective_trad_cfg = effective_config.setdefault("traditional_step4", {})
    if not isinstance(effective_trad_cfg, dict):
        effective_trad_cfg = {}
        effective_config["traditional_step4"] = effective_trad_cfg
    effective_shared_cfg = effective_trad_cfg.setdefault("shared", {})
    if not isinstance(effective_shared_cfg, dict):
        effective_shared_cfg = {}
        effective_trad_cfg["shared"] = effective_shared_cfg
    effective_shared_cfg["split_mode"] = str(split_mode)
    effective_shared_cfg["regression_dataset_path"] = str(reg_dataset)
    effective_shared_cfg["classification_dataset_path"] = str(cls_dataset)
    effective_shared_split_cfg = effective_shared_cfg.setdefault("split", {})
    if not isinstance(effective_shared_split_cfg, dict):
        effective_shared_split_cfg = {}
        effective_shared_cfg["split"] = effective_shared_split_cfg
    if reg_cfg is not None:
        effective_shared_split_cfg["holdout_test_ratio"] = float(reg_cfg.holdout_test_ratio)
    elif cls_cfg is not None:
        effective_shared_split_cfg["holdout_test_ratio"] = float(cls_cfg.holdout_test_ratio)
    effective_fp_cfg = effective_shared_cfg.setdefault("fingerprint", {})
    if not isinstance(effective_fp_cfg, dict):
        effective_fp_cfg = {}
        effective_shared_cfg["fingerprint"] = effective_fp_cfg
    effective_fp_cfg.update(
        {
            "radius": int(fp_cfg.radius),
            "n_bits": int(fp_cfg.n_bits),
            "use_chirality": bool(fp_cfg.use_chirality),
            "use_features": bool(fp_cfg.use_features),
        }
    )
    if reg_cfg is not None:
        effective_reg_cfg = effective_trad_cfg.setdefault("step4_3_1_regression", {})
        if isinstance(effective_reg_cfg, dict):
            effective_reg_cfg["tune"] = bool(reg_cfg.tune)
            effective_reg_cfg["n_trials"] = int(reg_cfg.n_trials)
            effective_reg_cfg["tuning_cv_folds"] = int(reg_cfg.tuning_cv_folds)
            effective_reg_cfg["tuning_objective"] = str(reg_cfg.tuning_objective)
            effective_reg_cfg["models"] = list(reg_cfg.models)
    if cls_cfg is not None:
        effective_cls_cfg = effective_trad_cfg.setdefault("step4_3_2_classification", {})
        if isinstance(effective_cls_cfg, dict):
            effective_cls_cfg["tune"] = bool(cls_cfg.tune)
            effective_cls_cfg["n_trials"] = int(cls_cfg.n_trials)
            effective_cls_cfg["tuning_cv_folds"] = int(cls_cfg.tuning_cv_folds)
            effective_cls_cfg["tuning_objective"] = str(cls_cfg.tuning_objective)
            effective_cls_cfg["models"] = list(cls_cfg.models)

    seed_info = _seed_everything_simple(seed)
    save_config(effective_config, step_dir / "config_used.yaml")
    _save_run_metadata_simple(step_dir, args.config, seed_info)
    log_context = {
        "config_path": args.config,
        "stage": args.stage,
        "results_dir": str(results_dir),
        "split_mode": split_mode,
        "regression_dataset_path": str(reg_dataset),
        "classification_dataset_path": str(cls_dataset),
        "fingerprint_radius": int(fp_cfg.radius),
        "fingerprint_n_bits": int(fp_cfg.n_bits),
        "random_seed": int(seed),
    }
    if reg_cfg is not None:
        log_context.update(
            {
                "regression_holdout_test_ratio": float(reg_cfg.holdout_test_ratio),
                "regression_tuning_cv_folds": int(reg_cfg.tuning_cv_folds),
                "regression_tuning_objective": str(reg_cfg.tuning_objective),
                "regression_tune": bool(reg_cfg.tune),
                "regression_n_trials": int(reg_cfg.n_trials),
            }
        )
    if cls_cfg is not None:
        log_context.update(
            {
                "classification_holdout_test_ratio": float(cls_cfg.holdout_test_ratio),
                "classification_tuning_cv_folds": int(cls_cfg.tuning_cv_folds),
                "classification_tuning_objective": str(cls_cfg.tuning_objective),
                "classification_tune": bool(cls_cfg.tune),
                "classification_n_trials": int(cls_cfg.n_trials),
            }
        )
    write_initial_log(
        step_dir=step_dir,
        step_name="step4_3_traditional",
        context=log_context,
    )

    dpi = int(config.get("plotting", {}).get("dpi", 600))
    font_size = int(config.get("plotting", {}).get("font_size", 12))

    print("=" * 80)
    print("Step 4_3 traditional pipeline")
    print(f"Stage: {args.stage}")
    print(f"Split mode: {split_mode}")
    print(f"Regression dataset: {reg_dataset}")
    print(f"Classification dataset: {cls_dataset}")
    print(f"Morgan fingerprint: radius={fp_cfg.radius}, nBits={fp_cfg.n_bits}")
    print("=" * 80)

    summary = {
        "step": "step4_3_traditional",
        "stage": args.stage,
        "split_mode": split_mode,
        "results_dir": str(results_dir),
        "step_dir": str(step_dir),
        "regression_dataset_path": str(reg_dataset),
        "classification_dataset_path": str(cls_dataset),
        "fingerprint_radius": int(fp_cfg.radius),
        "fingerprint_n_bits": int(fp_cfg.n_bits),
        "random_seed": int(seed),
        "regression_enabled": bool(run_reg),
        "classification_enabled": bool(run_cls),
    }
    if reg_cfg is not None:
        summary.update(
            {
                "regression_tune": bool(reg_cfg.tune),
                "regression_n_trials": int(reg_cfg.n_trials),
                "regression_tuning_cv_folds": int(reg_cfg.tuning_cv_folds),
                "regression_tuning_objective": str(reg_cfg.tuning_objective),
            }
        )
    if cls_cfg is not None:
        summary.update(
            {
                "classification_tune": bool(cls_cfg.tune),
                "classification_n_trials": int(cls_cfg.n_trials),
                "classification_tuning_cv_folds": int(cls_cfg.tuning_cv_folds),
                "classification_tuning_objective": str(cls_cfg.tuning_objective),
            }
        )

    if run_reg:
        if reg_cfg is None or reg_split_ratios is None:
            raise RuntimeError("Regression stage requested but regression configuration is not initialized.")
        reg_split_df, reg_split_assign = load_split_dataset(
            dataset_path=reg_dataset,
            split_mode=split_mode,
            split_ratios=reg_split_ratios,
            seed=seed,
            is_classification_dataset=False,
        )
        reg_split_assign.to_csv(shared_dir / "split_assignments_step4_3_1.csv", index=False)
        reg_split_df.to_csv(shared_dir / "chi_dataset_with_split_step4_3_1.csv", index=False)

        reg_table, reg_fp_meta = build_or_load_fingerprint_cache(
            df=reg_split_df,
            fp_cfg=fp_cfg,
            cache_npz=shared_dir / "morgan_fingerprint_table_step4_3_1.npz",
        )
        with open(shared_dir / "morgan_fingerprint_metadata_step4_3_1.json", "w") as f:
            json.dump(reg_fp_meta, f, indent=2)

        reg_out = _run_regression_stage(
            split_df=reg_split_df,
            split_assign=reg_split_assign,
            fingerprint_table=reg_table,
            stage_cfg=reg_cfg,
            reg_dir=reg_dir,
            dpi=dpi,
            font_size=font_size,
        )
        summary.update(
            {
                "step4_3_1_metrics_dir": reg_out["metrics_dir"],
                "step4_3_1_checkpoint": reg_out["checkpoint"],
                "step4_3_1_models_dir": reg_out.get("models_dir", ""),
                "step4_3_1_best_model_name": reg_out["best_model_name"],
                "step4_3_1_test_r2": reg_out["test_r2"],
                "step4_3_1_test_rmse": reg_out["test_rmse"],
            }
        )
    else:
        print("Skipping Step4_3_1 regression stage by request.")

    if run_cls:
        if cls_cfg is None or cls_split_ratios is None:
            raise RuntimeError("Classification stage requested but classification configuration is not initialized.")
        cls_split_df, cls_split_assign = load_split_dataset(
            dataset_path=cls_dataset,
            split_mode=split_mode,
            split_ratios=cls_split_ratios,
            seed=seed,
            is_classification_dataset=True,
        )
        cls_split_assign.to_csv(shared_dir / "split_assignments_step4_3_2.csv", index=False)
        cls_split_df.to_csv(shared_dir / "chi_dataset_with_split_step4_3_2.csv", index=False)

        cls_table, cls_fp_meta = build_or_load_fingerprint_cache(
            df=cls_split_df,
            fp_cfg=fp_cfg,
            cache_npz=shared_dir / "morgan_fingerprint_table_step4_3_2.npz",
        )
        with open(shared_dir / "morgan_fingerprint_metadata_step4_3_2.json", "w") as f:
            json.dump(cls_fp_meta, f, indent=2)

        cls_out = _run_classification_stage(
            split_df=cls_split_df,
            split_assign=cls_split_assign,
            fingerprint_table=cls_table,
            stage_cfg=cls_cfg,
            cls_dir=cls_dir,
            dpi=dpi,
            font_size=font_size,
        )
        summary.update(
            {
                "step4_3_2_metrics_dir": cls_out["metrics_dir"],
                "step4_3_2_checkpoint": cls_out["checkpoint"],
                "step4_3_2_models_dir": cls_out.get("models_dir", ""),
                "step4_3_2_best_model_name": cls_out["best_model_name"],
                "step4_3_2_test_balanced_accuracy": cls_out["test_balanced_accuracy"],
                "step4_3_2_test_auroc": cls_out["test_auroc"],
            }
        )
    else:
        print("Skipping Step4_3_2 classification stage by request.")

    save_step_summary(summary, pipeline_metrics_run_dir)
    print(f"Step4_3 outputs: {step_dir}")


if __name__ == "__main__":
    main()
