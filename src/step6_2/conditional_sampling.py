"""Conditional sampling helpers for Step 6_2 S2/S3/S4."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from src.sampling.sampler import ConstrainedSampler
from src.step6_2.conditional_diffusion import ConditionalDiscreteMaskingDiffusion
from src.step6_2.config import ResolvedStep62Config
from src.step6_2.frozen_sampling import ResolvedClassSamplingPrior, sample_with_class_prior
from src.data.tokenizer import PSmilesTokenizer


class ConditionalConstrainedSampler(ConstrainedSampler):
    """Constrained sampler that uses Step 6_2 conditional CFG logits."""

    diffusion_model: ConditionalDiscreteMaskingDiffusion

    def __init__(
        self,
        *,
        condition_bundle: torch.Tensor,
        cfg_scale: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if condition_bundle.dim() == 1:
            condition_bundle = condition_bundle.unsqueeze(0)
        if condition_bundle.dim() != 2 or condition_bundle.shape[-1] != 7:
            raise ValueError(
                f"condition_bundle must have shape [batch, 7] or [7], got {tuple(condition_bundle.shape)}"
            )
        self.condition_bundle = condition_bundle.detach().to(
            device=device_from_kwargs(kwargs),
            dtype=torch.float32,
        )
        self.cfg_scale = float(cfg_scale)

    def _condition_for_batch(self, batch_size: int) -> torch.Tensor:
        if self.condition_bundle.shape[0] == batch_size:
            return self.condition_bundle
        if self.condition_bundle.shape[0] == 1:
            return self.condition_bundle.expand(batch_size, -1)
        raise ValueError(
            f"Conditional sampler condition bundle has batch dimension {self.condition_bundle.shape[0]}, "
            f"but current batch_size is {batch_size}"
        )

    def _sample_from_ids(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        show_progress: bool = True,
    ) -> Tuple[torch.Tensor, List[str]]:
        self.diffusion_model.eval()
        batch_size = ids.shape[0]
        cond = self._condition_for_batch(batch_size)

        final_logits = None
        steps = range(self.num_steps, 0, -1)
        if show_progress:
            from tqdm import tqdm

            steps = tqdm(steps, desc="Conditional sampling")

        for t in steps:
            timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            step_progress = self._step_progress_frac(int(t))

            with torch.no_grad():
                logits = self.diffusion_model.classifier_free_guidance_logits(
                    ids,
                    timesteps,
                    attention_mask,
                    condition_bundle=cond,
                    cfg_scale=self.cfg_scale,
                )

            logits = logits / self.temperature
            if self.use_constraints:
                logits = self._apply_star_constraint(logits, ids, max_stars=self.target_stars)
                logits = self._apply_exact_star_budget_constraint(logits, ids, target_stars=self.target_stars)
                logits = self._apply_position_aware_paren_constraints(logits, ids)
                logits = self._apply_ring_constraints(logits, ids)
                logits = self._apply_bond_placement_constraints(logits, ids)
            logits = self._apply_class_token_bias(logits, fixed_mask=fixed_mask, step_progress=step_progress)
            logits = self._apply_sampling_filters(logits)
            logits = self._apply_special_token_constraints(logits, ids)
            logits = self._ensure_valid_logits(logits)

            probs = self._logits_to_probs(logits)
            is_masked = (ids == self.mask_id) & (~fixed_mask)
            unmask_prob = 1.0 / t

            for i in range(batch_size):
                masked_pos = torch.where(is_masked[i])[0]
                if len(masked_pos) == 0:
                    continue

                num_unmask = max(1, int(len(masked_pos) * unmask_prob))
                unmask_indices = torch.randperm(len(masked_pos), device=self.device)[:num_unmask]
                unmask_positions = masked_pos[unmask_indices]

                for pos in unmask_positions:
                    sampled = torch.multinomial(probs[i, pos], 1)
                    ids[i, pos] = sampled
                    sampled_token = int(sampled.item())

                    if self.use_constraints:
                        if sampled_token == self.star_id:
                            non_mask = ids[i] != self.mask_id
                            current_stars = ((ids[i] == self.star_id) & non_mask).sum().item()
                            if current_stars >= self.target_stars:
                                remaining_mask = (ids[i] == self.mask_id) & (~fixed_mask[i])
                                logits[i, remaining_mask, self.star_id] = float("-inf")
                                probs[i] = self._logits_to_probs(logits[i])
                        elif sampled_token in self.bond_ids:
                            next_pos = pos + 1
                            if next_pos < len(ids[i]) and ids[i, next_pos] == self.mask_id:
                                for bond_id in self.bond_ids:
                                    logits[i, next_pos, bond_id] = float("-inf")
                                probs[i] = self._logits_to_probs(logits[i])
                        elif sampled_token == self.open_paren_id:
                            next_pos = pos + 1
                            if next_pos < len(ids[i]) and ids[i, next_pos] == self.mask_id:
                                logits[i, next_pos, self.close_paren_id] = float("-inf")
                                probs[i] = self._logits_to_probs(logits[i])

            if t == 1:
                final_logits = logits

        if self.use_constraints:
            ids = self._fix_ring_closures(ids, final_logits, fixed_mask=fixed_mask)
            ids = self._fix_bond_placement(ids, final_logits, fixed_mask=fixed_mask)
            ids = self._fix_paren_balance(ids, final_logits, fixed_mask=fixed_mask)
            ids = self._fix_star_count(ids, final_logits, target_stars=self.target_stars, fixed_mask=fixed_mask)
            ids = self._fix_ring_closures(ids, final_logits, fixed_mask=fixed_mask)
            ids = self._fix_paren_balance(ids, final_logits, fixed_mask=fixed_mask)

        smiles_list = self.tokenizer.batch_decode(ids.cpu().tolist(), skip_special_tokens=True)
        return ids, smiles_list


def device_from_kwargs(kwargs: Dict[str, object]) -> str:
    device = kwargs.get("device", "cpu")
    return str(device)


def create_conditional_sampler(
    *,
    diffusion_model: ConditionalDiscreteMaskingDiffusion,
    tokenizer: PSmilesTokenizer,
    resolved: ResolvedStep62Config,
    prior: ResolvedClassSamplingPrior,
    condition_bundle: torch.Tensor,
    cfg_scale: float,
    device: str,
    num_steps: int | None = None,
) -> ConditionalConstrainedSampler:
    sampling_cfg = resolved.base_config.get("sampling", {})
    sampler = ConditionalConstrainedSampler(
        diffusion_model=diffusion_model,
        tokenizer=tokenizer,
        num_steps=int(
            resolved.base_config["diffusion"]["num_steps"] if num_steps is None else num_steps
        ),
        temperature=float(resolved.step6_2["sampling_temperature"]),
        top_k=sampling_cfg.get("top_k"),
        top_p=sampling_cfg.get("top_p"),
        target_stars=int(sampling_cfg.get("target_stars", 2)),
        use_constraints=bool(sampling_cfg.get("use_constraints", True)),
        device=device,
        condition_bundle=condition_bundle,
        cfg_scale=float(cfg_scale),
    )
    sampler.set_class_token_bias_start_frac(float(resolved.step6_2.get("class_token_bias_start_frac", 0.0)))
    if prior.class_token_logit_bias is not None:
        sampler.set_class_token_logit_bias(prior.class_token_logit_bias)
    return sampler


def sample_conditional_with_class_prior(
    *,
    sampler: ConditionalConstrainedSampler,
    tokenizer: PSmilesTokenizer,
    prior: ResolvedClassSamplingPrior,
    resolved: ResolvedStep62Config,
    num_samples: int,
    show_progress: bool = True,
) -> Tuple[List[str], Dict[str, object]]:
    """Conditional Step 6_2 sampling through the shared family-aware decode path."""
    return sample_with_class_prior(
        sampler=sampler,
        tokenizer=tokenizer,
        prior=prior,
        resolved=resolved,
        num_samples=num_samples,
        show_progress=show_progress,
    )
