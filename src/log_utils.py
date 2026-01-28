from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def start_log(log_path: Path, step_name: str, config_path: str, device: Optional[str] = None, extra: Optional[Dict] = None) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = datetime.utcnow().isoformat()
    with open(log_path, "a") as f:
        f.write(f"step: {step_name}\n")
        f.write(f"start_time_utc: {start_time}\n")
        f.write(f"config: {config_path}\n")
        if device:
            f.write(f"device: {device}\n")
        if extra:
            for k, v in extra.items():
                f.write(f"{k}: {v}\n")
        f.write("---\n")
    return start_time


def end_log(log_path: Path, start_time: str, status: str = "completed", extra: Optional[Dict] = None) -> None:
    end_time = datetime.utcnow().isoformat()
    try:
        duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)).total_seconds()
    except Exception:
        duration = None
    with open(log_path, "a") as f:
        f.write(f"end_time_utc: {end_time}\n")
        if duration is not None:
            f.write(f"duration_seconds: {duration}\n")
        f.write(f"status: {status}\n")
        if extra:
            for k, v in extra.items():
                f.write(f"{k}: {v}\n")
        f.write("===\n")
