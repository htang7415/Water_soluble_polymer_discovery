#!/usr/bin/env python
"""Build the Step 6_2 novelty reference cache ahead of success-hit audits."""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chi.inverse_design_common import resolve_training_smiles
from src.step6_2.config import load_step6_2_config


def _build_temp_step62_config(
    *,
    config_path: str,
    c_target: str,
) -> str:
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload = deepcopy(payload)
    payload.setdefault("step6_2", {})["c_target"] = str(c_target)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        return handle.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Step 6_2 novelty cache.")
    parser.add_argument("--config", default="configs/config6_2.yaml")
    parser.add_argument("--base_config", default="configs/config.yaml")
    parser.add_argument("--model_size", default="small")
    parser.add_argument("--c_target", default="polyamide")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if args.progress:
        os.environ["STEP62_EVAL_LOAD_PROGRESS"] = "1"

    temp_config_path = _build_temp_step62_config(config_path=args.config, c_target=args.c_target)
    start = time.perf_counter()
    resolved = load_step6_2_config(
        config_path=temp_config_path,
        base_config_path=args.base_config,
        model_size=args.model_size,
    )
    train_path = resolved.results_dir / "train_unlabeled.csv"
    if not train_path.exists():
        train_path = resolved.base_results_dir / "train_unlabeled.csv"
    cache_txt = train_path.with_name(f"{train_path.stem}_canonical_smiles_cache.txt.gz")
    cache_meta = cache_txt.with_suffix(".json")

    print(f"train_path={train_path}", flush=True)
    print(f"cache_txt={cache_txt}", flush=True)
    print(f"cache_meta={cache_meta}", flush=True)
    print(f"source_size_bytes={train_path.stat().st_size}", flush=True)

    canonical_smiles = resolve_training_smiles(resolved.results_dir, resolved.base_results_dir)
    elapsed = time.perf_counter() - start

    print(f"unique_canonical_smiles={len(canonical_smiles)}", flush=True)
    print(f"elapsed_s={elapsed:.2f}", flush=True)
    print(f"cache_txt_exists={int(cache_txt.exists())}", flush=True)
    print(f"cache_meta_exists={int(cache_meta.exists())}", flush=True)


if __name__ == "__main__":
    main()
