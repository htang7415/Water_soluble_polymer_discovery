"""
Configuration loading and validation utilities.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class Config:
    """
    Configuration container with dot-notation access to nested dictionaries.

    Allows accessing config values via attribute notation:
        config.model.encoder_latent_dim instead of config['model']['encoder_latent_dim']

    Args:
        config_dict: Dictionary containing configuration parameters
    """

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        self._convert_nested(config_dict)

    def _convert_nested(self, data: Any) -> None:
        """Recursively convert nested dictionaries to Config objects."""
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting."""
        self._config[key] = value
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default fallback."""
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to nested dictionary."""
        return self._config.copy()

    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load YAML configuration file and return as Config object.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object with hierarchical access to parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed

    Example:
        >>> config = load_config("configs/config.yaml")
        >>> print(config.model.encoder_latent_dim)
        128
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Validate required sections
    required_sections = ["paths", "chem", "training", "model", "loss_weights"]
    for section in required_sections:
        if section not in config_dict:
            raise ValueError(f"Config missing required section: {section}")

    return Config(config_dict)


def save_config(config: Union[Config, Dict], output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Config object or dictionary to save
        output_path: Path to output YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(config, Config):
        config_dict = config.to_dict()
    else:
        config_dict = config

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def update_config(base_config: Config, updates: Dict[str, Any]) -> Config:
    """
    Update configuration with new values (useful for hyperparameter search).

    Args:
        base_config: Base configuration
        updates: Dictionary of updates (can use dot notation keys like "model.encoder_dropout")

    Returns:
        New Config object with updates applied

    Example:
        >>> updated = update_config(base_config, {"model.encoder_dropout": 0.3})
    """
    config_dict = base_config.to_dict()

    for key, value in updates.items():
        # Handle dot notation for nested updates
        keys = key.split(".")
        current = config_dict

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    return Config(config_dict)


def validate_paths(config: Config, create_dirs: bool = True) -> None:
    """
    Validate that required paths exist and optionally create output directories.

    Args:
        config: Configuration object
        create_dirs: If True, create output directories that don't exist

    Raises:
        FileNotFoundError: If required input files don't exist
    """
    # Check input files exist
    input_files = [
        config.paths.dft_chi_csv,
        config.paths.exp_chi_csv,
        config.paths.solubility_csv,
    ]

    for file_path in input_files:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Required input file not found: {file_path}")

    # Create output directories if needed
    if create_dirs:
        output_dirs = [
            config.paths.processed_dir,
            config.paths.results_dir,
        ]

        for dir_path in output_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
