"""
Logging utilities for experiment tracking and monitoring.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Optional[Path] = None,
    log_file: str = "train.log",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> logging.Logger:
    """
    Setup logging configuration with console and file handlers.

    Args:
        log_dir: Directory for log file. If None, only console logging is enabled
        log_file: Name of log file
        console_level: Logging level for console (DEBUG, INFO, WARNING, ERROR)
        file_level: Logging level for file

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logging(Path("results/run_001"))
        >>> logger.info("Training started")
    """
    # Create logger
    logger = logging.getLogger("polymer_chi_ml")
    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_dir provided)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setLevel(getattr(logging, file_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance. If name is None, returns root polymer_chi_ml logger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    if name is None:
        return logging.getLogger("polymer_chi_ml")
    return logging.getLogger(f"polymer_chi_ml.{name}")


def create_run_directory(
    base_dir: Path,
    experiment_name: str,
    timestamp: Optional[str] = None,
) -> Path:
    """
    Create a unique run directory with timestamp.

    Args:
        base_dir: Base results directory
        experiment_name: Name of experiment (e.g., "dft_pretrain", "multitask")
        timestamp: Optional timestamp string. If None, uses current time

    Returns:
        Path to created run directory

    Example:
        >>> run_dir = create_run_directory(Path("results"), "dft_pretrain")
        >>> # Returns: results/dft_pretrain_20250118_143022/
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = base_dir / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    return run_dir


def save_git_info(run_dir: Path) -> None:
    """
    Save git commit information to file (if in a git repo).

    Args:
        run_dir: Run directory to save git info
    """
    import subprocess

    git_info_path = run_dir / "git_info.txt"

    try:
        # Get git commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        # Get git status
        git_status = subprocess.check_output(
            ["git", "status", "--short"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        # Get git diff
        git_diff = subprocess.check_output(
            ["git", "diff", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        with open(git_info_path, "w") as f:
            f.write(f"Commit: {commit_hash}\n\n")
            f.write("Status:\n")
            f.write(git_status if git_status else "  (no changes)\n")
            f.write("\n")
            if git_diff:
                f.write("Uncommitted diff:\n")
                f.write(git_diff)

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not in a git repo or git not available
        pass


class MetricsLogger:
    """
    Simple metrics logger for tracking training progress.

    Logs metrics to file in CSV format for easy analysis.
    """

    def __init__(self, log_path: Path):
        """
        Args:
            log_path: Path to metrics CSV file
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.header_written = False

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        """
        Log metrics dictionary to file.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step/epoch number
        """
        if step is not None:
            metrics = {"step": step, **metrics}

        # Write header if first time
        if not self.header_written:
            with open(self.log_path, "w") as f:
                f.write(",".join(metrics.keys()) + "\n")
            self.header_written = True

        # Write metrics
        with open(self.log_path, "a") as f:
            values = [str(v) for v in metrics.values()]
            f.write(",".join(values) + "\n")
