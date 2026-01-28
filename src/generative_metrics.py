"""Generative metrics for polymer sampling."""

from typing import Dict, List, Optional, Set
from collections import Counter
import numpy as np

from src.chem_utils import (
    check_validity,
    count_stars,
    compute_sa_score,
    canonicalize_smiles,
    batch_compute_fingerprints,
    compute_pairwise_diversity,
)


class GenerativeEvaluator:
    def __init__(self, training_smiles: Set[str]):
        self.training_smiles = {canonicalize_smiles(s) or s for s in training_smiles}

    def evaluate(
        self,
        generated_smiles: List[str],
        sample_id: str,
        sampling_time_sec: Optional[float] = None,
        method: Optional[str] = None,
        representation: Optional[str] = None,
        model_size: Optional[str] = None,
    ) -> Dict:
        n_total = len(generated_smiles)
        valid_smiles = []
        star_counts = []
        for s in generated_smiles:
            if check_validity(s):
                valid_smiles.append(s)
                star_counts.append(count_stars(s))
        n_valid = len(valid_smiles)
        validity = n_valid / n_total if n_total else 0.0

        canonical_valid = [canonicalize_smiles(s) or s for s in valid_smiles]
        unique_smiles = list(set(canonical_valid))
        n_unique = len(unique_smiles)
        uniqueness = n_unique / n_valid if n_valid else 0.0
        novel_smiles = [s for s in unique_smiles if s not in self.training_smiles]
        n_novel = len(novel_smiles)
        novelty = n_novel / n_unique if n_unique else 0.0

        star_counter = Counter(star_counts)
        n_valid_star_eq_2 = star_counter.get(2, 0)
        frac_star_eq_2 = n_valid_star_eq_2 / n_valid if n_valid else 0.0
        validity_two_stars = n_valid_star_eq_2 / n_total if n_total else 0.0

        sa_scores = [compute_sa_score(s) for s in valid_smiles]
        sa_scores = [s for s in sa_scores if s is not None]

        lengths = [len(s) for s in valid_smiles]

        fps, _ = batch_compute_fingerprints(unique_smiles[: min(5000, len(unique_smiles))])
        avg_diversity = 0.0
        if len(fps) >= 2:
            if len(fps) > 1000:
                idxs = np.random.choice(len(fps), 1000, replace=False)
                fps = [fps[i] for i in idxs]
            avg_diversity = compute_pairwise_diversity(fps)

        samples_per_sec = 0.0
        valid_per_sec = 0.0
        if sampling_time_sec and sampling_time_sec > 0:
            samples_per_sec = n_total / sampling_time_sec
            valid_per_sec = n_valid / sampling_time_sec

        metrics = {
            "method": method or "",
            "representation": representation or "",
            "model_size": model_size or "",
            "sample_id": sample_id,
            "n_total": n_total,
            "n_valid": n_valid,
            "validity": round(validity, 4),
            "validity_two_stars": round(validity_two_stars, 4),
            "uniqueness": round(uniqueness, 4),
            "novelty": round(novelty, 4),
            "avg_diversity": round(avg_diversity, 4),
            "frac_star_eq_2": round(frac_star_eq_2, 4),
            **_stats(sa_scores, "sa"),
            **_stats(lengths, "length"),
            "samples_per_sec": round(samples_per_sec, 4),
            "valid_per_sec": round(valid_per_sec, 4),
            "star_count_distribution": dict(star_counter),
        }
        return metrics

    def get_valid_samples(self, generated_smiles: List[str], require_two_stars: bool = True) -> List[str]:
        out = []
        for s in generated_smiles:
            if not check_validity(s):
                continue
            if require_two_stars and count_stars(s) != 2:
                continue
            out.append(s)
        return out

    @staticmethod
    def format_metrics_csv(metrics: Dict):
        import pandas as pd

        scalar = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
        return pd.DataFrame([scalar])


def _stats(values: List[float], prefix: str) -> Dict[str, float]:
    if not values:
        return {f"mean_{prefix}": 0.0, f"std_{prefix}": 0.0, f"min_{prefix}": 0.0, f"max_{prefix}": 0.0}
    return {
        f"mean_{prefix}": round(float(np.mean(values)), 4),
        f"std_{prefix}": round(float(np.std(values)), 4),
        f"min_{prefix}": round(float(np.min(values)), 4),
        f"max_{prefix}": round(float(np.max(values)), 4),
    }
