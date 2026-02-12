#!/usr/bin/env python
"""Step 4_3: traditional ML baselines for chi regression and solubility classification."""

from __future__ import annotations

import argparse
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
    if isinstance(value, list) and len(value) == 2:
        lo = float(min(value))
        hi = float(max(value))
        return lo, hi
    return float(default_lo), float(default_hi)


def _as_int_pair(value, default_lo: int, default_hi: int) -> Tuple[int, int]:
    if isinstance(value, list) and len(value) == 2:
        lo = int(min(value))
        hi = int(max(value))
        return lo, hi
    return int(default_lo), int(default_hi)


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
    sns.scatterplot(data=plot_df, x="chi_true", y="chi_pred", hue="fold", palette=palette, alpha=0.8, s=20, ax=ax)
    lo = float(min(plot_df["chi_true"].min(), plot_df["chi_pred"].min()))
    hi = float(max(plot_df["chi_true"].max(), plot_df["chi_pred"].max()))
    span = max(hi - lo, 1.0e-8)
    pad = 0.04 * span
    lo_plot = lo - pad
    hi_plot = hi + pad
    ax.plot([lo_plot, hi_plot], [lo_plot, hi_plot], "k--", linewidth=1.0)
    ax.set_xlim(lo_plot, hi_plot)
    ax.set_ylim(lo_plot, hi_plot)
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
    ax.set_title(f"CV parity by fold (n={len(plot_df)})")
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

    objective_name = "val_r2"
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
            score = float(cv_eval["cv_val_r2"])
            rmse = float(cv_eval["cv_val_rmse"])
            invalid = int((not np.isfinite(score)) or (not np.isfinite(rmse)))
        except Exception as exc:
            score = -1.0e12
            rmse = np.nan
            invalid = 1
            trial.set_user_attr("error", str(exc))

        trial.set_user_attr("model_name", model_name)
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
    trial_df = pd.DataFrame(trial_rows).sort_values("trial").reset_index(drop=True)
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

    best_model_name = str(study.best_params["model_name"])
    best_params = {k: v for k, v in study.best_params.items() if k != "model_name"}
    best_cv = _evaluate_regression_cv(
        cv_folds=cv_folds,
        fingerprint_table=fingerprint_table,
        model_name=best_model_name,
        model_params=best_params,
        seed=stage_cfg.seed,
        collect_val_predictions=True,
    )
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
                "best_trial": int(study.best_trial.number),
                "objective": "maximize_val_r2",
                "objective_name": objective_name,
                "objective_direction": "maximize",
                "objective_value_at_best_trial": float(study.best_value),
                "best_model_name": best_model_name,
                "best_params": best_params,
                "best_value_r2": _safe_float(study.best_trial.user_attrs.get("cv_val_r2", np.nan)),
                "best_value_rmse": _safe_float(study.best_trial.user_attrs.get("cv_val_rmse", np.nan)),
                "tuning_cv_folds_requested": int(stage_cfg.tuning_cv_folds),
                "tuning_cv_folds_resolved": int(cv_info.get("resolved_folds", len(cv_folds))),
                "tuning_cv_strategy": str(cv_info.get("strategy", "unknown")),
                "invalid_trial_count": int(trial_df["invalid_metrics"].sum()) if "invalid_metrics" in trial_df.columns else 0,
            },
            f,
            indent=2,
        )
    return {"best_model_name": best_model_name, "best_params": best_params, "cv_info": cv_info}


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

    objective_name = "val_balanced_accuracy"
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
            score = float(cv_eval["cv_val_balanced_accuracy"])
            auroc = float(cv_eval["cv_val_auroc"])
            invalid = int((not np.isfinite(score)) or (not np.isfinite(auroc)))
        except Exception as exc:
            score = -1.0e12
            auroc = np.nan
            invalid = 1
            trial.set_user_attr("error", str(exc))

        trial.set_user_attr("model_name", model_name)
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
    trial_df = pd.DataFrame(trial_rows).sort_values("trial").reset_index(drop=True)
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

    best_model_name = str(study.best_params["model_name"])
    best_params = {k: v for k, v in study.best_params.items() if k != "model_name"}
    best_cv = _evaluate_classification_cv(
        cv_folds=cv_folds,
        fingerprint_table=fingerprint_table,
        model_name=best_model_name,
        model_params=best_params,
        seed=stage_cfg.seed,
    )
    best_cv["fold_metrics"].to_csv(tuning_dir / "best_trial_cv_fold_metrics.csv", index=False)

    with open(tuning_dir / "optuna_best.json", "w") as f:
        json.dump(
            {
                "best_trial": int(study.best_trial.number),
                "objective": "maximize_val_balanced_accuracy",
                "objective_name": objective_name,
                "objective_direction": "maximize",
                "objective_value_at_best_trial": float(study.best_value),
                "best_model_name": best_model_name,
                "best_params": best_params,
                "best_value_balanced_accuracy": _safe_float(study.best_trial.user_attrs.get("cv_val_balanced_accuracy", np.nan)),
                "best_value_auroc": _safe_float(study.best_trial.user_attrs.get("cv_val_auroc", np.nan)),
                "tuning_cv_folds_requested": int(stage_cfg.tuning_cv_folds),
                "tuning_cv_folds_resolved": int(cv_info.get("resolved_folds", len(cv_folds))),
                "tuning_cv_strategy": str(cv_info.get("strategy", "unknown")),
                "invalid_trial_count": int(trial_df["invalid_metrics"].sum()) if "invalid_metrics" in trial_df.columns else 0,
            },
            f,
            indent=2,
        )
    return {"best_model_name": best_model_name, "best_params": best_params, "cv_info": cv_info}


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
    plt.rcParams.update({"font.size": font_size, "axes.titlesize": font_size, "axes.labelsize": font_size})

    test_df = pred_df[pred_df["split"] == "test"].copy().reset_index(drop=True)
    if test_df.empty:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(data=test_df, x="chi", y="chi_pred", s=24, alpha=0.8, ax=ax)
    lo = float(min(test_df["chi"].min(), test_df["chi_pred"].min()))
    hi = float(max(test_df["chi"].max(), test_df["chi_pred"].max()))
    span = max(hi - lo, 1.0e-8)
    pad = 0.04 * span
    lo_plot = lo - pad
    hi_plot = hi + pad
    ax.plot([lo_plot, hi_plot], [lo_plot, hi_plot], "k--", linewidth=1.0)
    ax.set_xlim(lo_plot, hi_plot)
    ax.set_ylim(lo_plot, hi_plot)
    ax.set_xlabel("True chi")
    ax.set_ylabel("Predicted chi")
    ax.set_title("chi parity (test)")
    fig.tight_layout()
    fig.savefig(fig_dir / "chi_parity_test.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(test_df["chi_error"], kde=True, bins=30, ax=ax, color="#1f77b4")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Prediction error (chi_pred - chi_true)")
    ax.set_title("chi residual distribution (test)")
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
    plt.rcParams.update({"font.size": font_size, "axes.titlesize": font_size, "axes.labelsize": font_size})

    test_df = pred_df[pred_df["split"] == "test"].copy().reset_index(drop=True)
    if test_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=test_df, x="class_prob", hue="water_soluble", bins=30, stat="density", common_norm=False, ax=ax)
    ax.set_xlabel("Predicted soluble probability")
    ax.set_title("Probability distribution (test)")
    fig.tight_layout()
    fig.savefig(fig_dir / "class_prob_distribution_test.png", dpi=dpi)
    plt.close(fig)

    y_true = test_df["water_soluble"].to_numpy(dtype=int)
    y_prob = test_df["class_prob"].to_numpy(dtype=float)
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, color="#1f77b4", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC curve (test)")
        fig.tight_layout()
        fig.savefig(fig_dir / "classifier_roc_test.png", dpi=dpi)
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(recall, precision, color="#ff7f0e", linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("PR curve (test)")
        fig.tight_layout()
        fig.savefig(fig_dir / "classifier_pr_test.png", dpi=dpi)
        plt.close(fig)


def _resolve_stage_config(
    stage_section: Dict,
    shared_split_mode: str,
    shared_holdout: Optional[float],
    seed: int,
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
    for d in [metrics_dir, figures_dir, tuning_dir, checkpoint_dir]:
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
        hyper_summary = {
            "selection_mode": "optuna",
            "objective": "maximize_val_r2",
            "best_model_name": best_model_name,
            "best_params": _to_serializable(best_params),
            "tuning_cv_folds": int(stage_cfg.tuning_cv_folds),
        }
    else:
        best_model_name = str(stage_cfg.models[0])
        best_params = _default_regression_params(best_model_name)
        hyper_summary = {
            "selection_mode": "default_no_tuning",
            "best_model_name": best_model_name,
            "best_params": _to_serializable(best_params),
            "tuning_cv_folds": int(stage_cfg.tuning_cv_folds),
        }

    with open(metrics_dir / "chosen_hyperparameters.json", "w") as f:
        json.dump({"model_name": best_model_name, "params": _to_serializable(best_params)}, f, indent=2)
    with open(metrics_dir / "hyperparameter_selection_summary.json", "w") as f:
        json.dump(hyper_summary, f, indent=2)

    final_fit_df = build_final_fit_split_df(split_df)
    model = _fit_regression_on_final_split(
        final_fit_df=final_fit_df,
        fingerprint_table=fingerprint_table,
        model_name=best_model_name,
        model_params=best_params,
        seed=stage_cfg.seed,
    )
    checkpoint_path = checkpoint_dir / "chi_regression_traditional_best.joblib"
    joblib.dump(model, checkpoint_path)
    with open(checkpoint_dir / "chi_regression_traditional_best.meta.json", "w") as f:
        json.dump({"model_name": best_model_name, "params": _to_serializable(best_params)}, f, indent=2)

    pred_df = _save_regression_predictions(
        split_df=final_fit_df,
        fingerprint_table=fingerprint_table,
        model=model,
        out_dir=metrics_dir,
    )
    _collect_regression_metrics(pred_df, out_dir=metrics_dir)
    _make_regression_figures(pred_df, fig_dir=figures_dir, dpi=dpi, font_size=font_size)
    save_artifact_manifest(step_dir=reg_dir, metrics_dir=metrics_dir, figures_dir=figures_dir, dpi=dpi)

    overall_df = pd.read_csv(metrics_dir / "chi_metrics_overall.csv")
    test_row = overall_df[overall_df["split"] == "test"]
    test_r2 = float(test_row["r2"].iloc[0]) if not test_row.empty and "r2" in test_row.columns else np.nan
    test_rmse = float(test_row["rmse"].iloc[0]) if not test_row.empty and "rmse" in test_row.columns else np.nan
    return {
        "metrics_dir": str(metrics_dir),
        "figures_dir": str(figures_dir),
        "checkpoint": str(checkpoint_path),
        "best_model_name": best_model_name,
        "test_r2": _safe_float(test_r2),
        "test_rmse": _safe_float(test_rmse),
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
    for d in [metrics_dir, figures_dir, tuning_dir, checkpoint_dir]:
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
        hyper_summary = {
            "selection_mode": "optuna",
            "objective": "maximize_val_balanced_accuracy",
            "best_model_name": best_model_name,
            "best_params": _to_serializable(best_params),
            "tuning_cv_folds": int(stage_cfg.tuning_cv_folds),
        }
    else:
        best_model_name = str(stage_cfg.models[0])
        best_params = _default_classification_params(best_model_name)
        hyper_summary = {
            "selection_mode": "default_no_tuning",
            "best_model_name": best_model_name,
            "best_params": _to_serializable(best_params),
            "tuning_cv_folds": int(stage_cfg.tuning_cv_folds),
        }

    with open(metrics_dir / "chosen_hyperparameters.json", "w") as f:
        json.dump({"model_name": best_model_name, "params": _to_serializable(best_params)}, f, indent=2)
    with open(metrics_dir / "hyperparameter_selection_summary.json", "w") as f:
        json.dump(hyper_summary, f, indent=2)

    final_fit_df = build_final_fit_split_df(split_df)
    model = _fit_classification_on_final_split(
        final_fit_df=final_fit_df,
        fingerprint_table=fingerprint_table,
        model_name=best_model_name,
        model_params=best_params,
        seed=stage_cfg.seed,
    )
    checkpoint_path = checkpoint_dir / "chi_classifier_traditional_best.joblib"
    joblib.dump(model, checkpoint_path)
    with open(checkpoint_dir / "chi_classifier_traditional_best.meta.json", "w") as f:
        json.dump({"model_name": best_model_name, "params": _to_serializable(best_params)}, f, indent=2)

    pred_df = _save_classification_predictions(
        split_df=final_fit_df,
        fingerprint_table=fingerprint_table,
        model=model,
        out_dir=metrics_dir,
    )
    _collect_classification_metrics(pred_df, out_dir=metrics_dir)
    _make_classification_figures(pred_df, fig_dir=figures_dir, dpi=dpi, font_size=font_size)
    save_artifact_manifest(step_dir=cls_dir, metrics_dir=metrics_dir, figures_dir=figures_dir, dpi=dpi)

    overall_df = pd.read_csv(metrics_dir / "class_metrics_overall.csv")
    test_row = overall_df[overall_df["split"] == "test"]
    test_bal = (
        float(test_row["balanced_accuracy"].iloc[0])
        if not test_row.empty and "balanced_accuracy" in test_row.columns
        else np.nan
    )
    test_auroc = float(test_row["auroc"].iloc[0]) if not test_row.empty and "auroc" in test_row.columns else np.nan
    return {
        "metrics_dir": str(metrics_dir),
        "figures_dir": str(figures_dir),
        "checkpoint": str(checkpoint_path),
        "best_model_name": best_model_name,
        "test_balanced_accuracy": _safe_float(test_bal),
        "test_auroc": _safe_float(test_auroc),
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

    reg_cfg = _resolve_stage_config(
        stage_section=reg_cfg_section,
        shared_split_mode=split_mode,
        shared_holdout=holdout_test_ratio,
        seed=seed,
        validate_dependencies=run_reg,
        tune_override=tune_override,
        n_trials_override=args.n_trials,
        tuning_cv_folds_override=args.tuning_cv_folds,
    )
    cls_cfg = _resolve_stage_config(
        stage_section=cls_cfg_section,
        shared_split_mode=split_mode,
        shared_holdout=holdout_test_ratio,
        seed=seed,
        validate_dependencies=run_cls,
        tune_override=tune_override,
        n_trials_override=args.n_trials,
        tuning_cv_folds_override=args.tuning_cv_folds,
    )

    reg_split_ratios = resolve_split_ratios(reg_cfg.holdout_test_ratio, reg_cfg.tuning_cv_folds)
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

    seed_info = _seed_everything_simple(seed)
    save_config(config, step_dir / "config_used.yaml")
    _save_run_metadata_simple(step_dir, args.config, seed_info)
    write_initial_log(
        step_dir=step_dir,
        step_name="step4_3_traditional",
        context={
            "config_path": args.config,
            "stage": args.stage,
            "results_dir": str(results_dir),
            "split_mode": split_mode,
            "regression_dataset_path": str(reg_dataset),
            "classification_dataset_path": str(cls_dataset),
            "regression_holdout_test_ratio": float(reg_cfg.holdout_test_ratio),
            "classification_holdout_test_ratio": float(cls_cfg.holdout_test_ratio),
            "regression_tuning_cv_folds": int(reg_cfg.tuning_cv_folds),
            "classification_tuning_cv_folds": int(cls_cfg.tuning_cv_folds),
            "fingerprint_radius": int(fp_cfg.radius),
            "fingerprint_n_bits": int(fp_cfg.n_bits),
            "random_seed": int(seed),
        },
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
        "regression_tune": bool(reg_cfg.tune),
        "classification_tune": bool(cls_cfg.tune),
        "regression_n_trials": int(reg_cfg.n_trials),
        "classification_n_trials": int(cls_cfg.n_trials),
        "regression_tuning_cv_folds": int(reg_cfg.tuning_cv_folds),
        "classification_tuning_cv_folds": int(cls_cfg.tuning_cv_folds),
    }

    if run_reg:
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
                "step4_3_1_best_model_name": reg_out["best_model_name"],
                "step4_3_1_test_r2": reg_out["test_r2"],
                "step4_3_1_test_rmse": reg_out["test_rmse"],
            }
        )
    else:
        print("Skipping Step4_3_1 regression stage by request.")

    if run_cls:
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
