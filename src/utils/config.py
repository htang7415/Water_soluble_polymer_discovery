"""Configuration loading utilities."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dictionary containing configuration parameters.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def deep_merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration dictionaries."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_config(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def as_yamlable(value: Any) -> Any:
    """Convert common path containers into YAML-safe primitives."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): as_yamlable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [as_yamlable(item) for item in value]
    return value


def load_step4_config(
    config_path: str = "configs/config4.yaml",
    *,
    base_config_path: str = "configs/config.yaml",
) -> Dict[str, Any]:
    """Load Step 4 config, optionally overlaying a Step 4-only file on the shared base config.

    `configs/config4.yaml` stores only Step 4-specific overrides. The rest of the
    project still relies on shared settings from `configs/config.yaml`, so this
    helper merges the two when the supplied config file exposes top-level
    `step4_1_regression` / `step4_2_classification` blocks.
    """

    raw_config = load_config(config_path)
    if not isinstance(raw_config, dict):
        return raw_config

    is_step4_overlay = "chi_training" not in raw_config and (
        "step4_1_regression" in raw_config or "step4_2_classification" in raw_config
    )
    if not is_step4_overlay:
        return raw_config

    merged = load_config(base_config_path)
    shared_override = {
        key: value
        for key, value in raw_config.items()
        if key not in {"step4_1_regression", "step4_2_classification"}
    }
    if shared_override:
        merged = deep_merge_config(merged, shared_override)

    merged.setdefault("chi_training", {})
    for key in ("step4_1_regression", "step4_2_classification"):
        if isinstance(raw_config.get(key), dict):
            existing = merged["chi_training"].get(key, {})
            if not isinstance(existing, dict):
                existing = {}
            merged["chi_training"][key] = deep_merge_config(existing, raw_config[key])

    return merged


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary.
        config_path: Path to save the configuration file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
