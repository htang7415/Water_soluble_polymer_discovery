"""Lightweight Step 6_2 study-family metadata."""

from __future__ import annotations

STUDY_BASE_RUNS = {
    "S1": "S1_guided_frozen",
    "S2": "S2_conditional",
    "S3": "S3_conditional_guided",
    "S4_rl": "S4_rl_finetuned",
    "S4_dpo": "S4_dpo",
}

__all__ = ["STUDY_BASE_RUNS"]
