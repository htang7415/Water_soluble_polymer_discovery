"""Trajectory recording and log-prob utilities for Step 6_2 S4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from src.step6_2.conditional_sampling import ConditionalConstrainedSampler
from src.step6_2.config import ResolvedStep62Config
from src.step6_2.frozen_sampling import (
    ResolvedClassSamplingPrior,
    _build_decode_constraint_multi_spans,
    _sample_lengths,
)
from src.data.tokenizer import PSmilesTokenizer


@dataclass
class TrajectoryStepRecord:
    """One reverse-diffusion step worth of replayable token decisions."""

    timestep: int
    ids_before: torch.Tensor
    attention_mask: torch.Tensor
    fixed_mask: torch.Tensor
    unmask_positions: torch.Tensor
    sampled_token_ids: torch.Tensor
    unmask_counts: torch.Tensor


@dataclass
class SamplingTrajectoryRecord:
    """Full conditional sampling trajectory for one batch."""

    condition_bundle: torch.Tensor
    cfg_scale: float
    final_ids: torch.Tensor
    final_attention_mask: torch.Tensor
    steps: List[TrajectoryStepRecord]


def _pad_event_matrix(values: List[torch.Tensor], *, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if not values:
        return (
            torch.empty((0, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )
    counts = torch.tensor([int(v.numel()) for v in values], dtype=torch.long, device=device)
    max_count = int(counts.max().item()) if len(counts) else 0
    padded = torch.full((len(values), max_count), -1, dtype=torch.long, device=device)
    for idx, tensor in enumerate(values):
        if tensor.numel() > 0:
            padded[idx, : tensor.numel()] = tensor.to(device=device, dtype=torch.long)
    return padded, counts


class TrajectoryConditionalSampler(ConditionalConstrainedSampler):
    """Conditional sampler that can emit replayable trajectories for S4."""

    def _apply_within_step_constraint_updates(
        self,
        logits_row: torch.Tensor,
        probs_row: torch.Tensor,
        ids_row: torch.Tensor,
        fixed_mask_row: torch.Tensor,
        sampled_token: int,
        pos: int,
    ) -> torch.Tensor:
        if not self.use_constraints:
            return probs_row

        if sampled_token == self.star_id:
            non_mask = ids_row != self.mask_id
            current_stars = ((ids_row == self.star_id) & non_mask).sum().item()
            if current_stars >= self.target_stars:
                remaining_mask = (ids_row == self.mask_id) & (~fixed_mask_row)
                logits_row[remaining_mask, self.star_id] = float("-inf")
                probs_row = self._logits_to_probs(logits_row)
        elif sampled_token in self.bond_ids:
            next_pos = pos + 1
            if next_pos < len(ids_row) and ids_row[next_pos] == self.mask_id:
                for bond_id in self.bond_ids:
                    logits_row[next_pos, bond_id] = float("-inf")
                probs_row = self._logits_to_probs(logits_row)
        elif sampled_token == self.open_paren_id:
            next_pos = pos + 1
            if next_pos < len(ids_row) and ids_row[next_pos] == self.mask_id:
                logits_row[next_pos, self.close_paren_id] = float("-inf")
                probs_row = self._logits_to_probs(logits_row)
        return probs_row

    def _compute_conditioned_logits(
        self,
        ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        *,
        grad_enabled: bool = False,
        diffusion_model_override=None,
    ) -> torch.Tensor:
        diffusion_model = diffusion_model_override or self.diffusion_model
        batch_size = ids.shape[0]
        cond = self._condition_for_batch(batch_size)
        step_progress = self._step_progress_frac(int(timesteps[0].item()))
        if grad_enabled:
            logits = diffusion_model.classifier_free_guidance_logits_impl(
                ids,
                timesteps,
                attention_mask,
                condition_bundle=cond,
                cfg_scale=self.cfg_scale,
            )
        else:
            logits = diffusion_model.classifier_free_guidance_logits(
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
        if (
            self.class_token_logit_bias is not None
            and step_progress >= float(self.class_token_bias_start_frac)
        ):
            bias_mask = (~fixed_mask).unsqueeze(-1).float()
            logits = logits + self.class_token_logit_bias.unsqueeze(0).unsqueeze(0) * bias_mask
        logits = self._apply_sampling_filters(logits)
        logits = self._apply_special_token_constraints(logits, ids)
        return self._ensure_valid_logits(logits)

    def sample_with_lengths_trajectory(
        self,
        lengths: Sequence[int],
        *,
        max_length: int | None = None,
        show_progress: bool = True,
    ) -> Tuple[torch.Tensor, List[str], SamplingTrajectoryRecord]:
        if not lengths:
            empty_ids = torch.empty((0, 0), dtype=torch.long, device=self.device)
            empty_mask = torch.empty((0, 0), dtype=torch.long, device=self.device)
            return empty_ids, [], SamplingTrajectoryRecord(
                condition_bundle=self._condition_for_batch(0),
                cfg_scale=float(self.cfg_scale),
                final_ids=empty_ids,
                final_attention_mask=empty_mask,
                steps=[],
            )

        lengths = [max(2, int(length)) for length in lengths]
        seq_length = max(lengths)
        if max_length is not None and seq_length > int(max_length):
            raise ValueError(f"Max length {max_length} is smaller than required {seq_length}")

        batch_size = len(lengths)
        ids = torch.full((batch_size, seq_length), self.mask_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros_like(ids)
        fixed_mask = torch.zeros_like(ids, dtype=torch.bool)

        ids[:, 0] = self.bos_id
        fixed_mask[:, 0] = True

        for row_idx, length in enumerate(lengths):
            eos_pos = int(length) - 1
            ids[row_idx, eos_pos] = self.eos_id
            fixed_mask[row_idx, eos_pos] = True
            attention_mask[row_idx, : int(length)] = 1
            if int(length) < seq_length:
                ids[row_idx, int(length) :] = self.pad_id
                fixed_mask[row_idx, int(length) :] = True

        return self._sample_from_ids_with_trajectory(ids, attention_mask, fixed_mask, show_progress=show_progress)

    def sample_with_fixed_spans_trajectory(
        self,
        *,
        span_token_ids: Sequence[Sequence[int]],
        span_start_positions: Sequence[int],
        seq_length: int,
        lengths: Sequence[int] | None = None,
        show_progress: bool = True,
    ) -> Tuple[torch.Tensor, List[str], SamplingTrajectoryRecord]:
        batch_size = len(span_token_ids)
        if len(span_start_positions) != batch_size:
            raise ValueError("span_start_positions must match span_token_ids")
        if lengths is not None and len(lengths) != batch_size:
            raise ValueError("lengths must match span_token_ids")

        ids = torch.full((batch_size, seq_length), self.mask_id, dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(ids)
        fixed_mask = torch.zeros_like(ids, dtype=torch.bool)
        ids[:, 0] = self.bos_id
        fixed_mask[:, 0] = True

        effective_lengths: List[int] = []
        if lengths is None:
            ids[:, -1] = self.eos_id
            fixed_mask[:, -1] = True
            effective_lengths = [int(seq_length)] * batch_size
        else:
            attention_mask = torch.zeros_like(ids)
            for row_idx, raw_length in enumerate(lengths):
                length = max(2, int(raw_length))
                if length > int(seq_length):
                    raise ValueError(f"length {length} exceeds seq_length={seq_length}")
                effective_lengths.append(length)
                eos_pos = length - 1
                ids[row_idx, eos_pos] = self.eos_id
                fixed_mask[row_idx, eos_pos] = True
                attention_mask[row_idx, :length] = 1
                if length < int(seq_length):
                    ids[row_idx, length:] = self.pad_id
                    fixed_mask[row_idx, length:] = True

        for row_idx, (span_ids, start_pos) in enumerate(zip(span_token_ids, span_start_positions)):
            span = [int(token_id) for token_id in span_ids]
            if not span:
                continue
            effective_length = effective_lengths[row_idx]
            end_pos = int(start_pos) + len(span)
            if int(start_pos) < 1 or end_pos > (effective_length - 1):
                raise ValueError(
                    f"Fixed span [{start_pos}, {end_pos}) does not fit sequence length {effective_length}"
                )
            ids[row_idx, int(start_pos) : end_pos] = torch.tensor(span, dtype=torch.long, device=self.device)
            fixed_mask[row_idx, int(start_pos) : end_pos] = True

        return self._sample_from_ids_with_trajectory(ids, attention_mask, fixed_mask, show_progress=show_progress)

    def sample_with_multiple_fixed_spans_trajectory(
        self,
        *,
        span_token_ids: Sequence[Sequence[Sequence[int]]],
        span_start_positions: Sequence[Sequence[int]],
        seq_length: int,
        lengths: Sequence[int] | None = None,
        show_progress: bool = True,
    ) -> Tuple[torch.Tensor, List[str], SamplingTrajectoryRecord]:
        batch_size = len(span_token_ids)
        if len(span_start_positions) != batch_size:
            raise ValueError("span_start_positions must match span_token_ids")
        if lengths is not None and len(lengths) != batch_size:
            raise ValueError("lengths must match span_token_ids")

        ids = torch.full((batch_size, seq_length), self.mask_id, dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(ids)
        fixed_mask = torch.zeros_like(ids, dtype=torch.bool)
        ids[:, 0] = self.bos_id
        fixed_mask[:, 0] = True

        effective_lengths: List[int] = []
        if lengths is None:
            ids[:, -1] = self.eos_id
            fixed_mask[:, -1] = True
            effective_lengths = [int(seq_length)] * batch_size
        else:
            attention_mask = torch.zeros_like(ids)
            for row_idx, raw_length in enumerate(lengths):
                length = max(2, int(raw_length))
                if length > int(seq_length):
                    raise ValueError(f"length {length} exceeds seq_length={seq_length}")
                effective_lengths.append(length)
                eos_pos = length - 1
                ids[row_idx, eos_pos] = self.eos_id
                fixed_mask[row_idx, eos_pos] = True
                attention_mask[row_idx, :length] = 1
                if length < int(seq_length):
                    ids[row_idx, length:] = self.pad_id
                    fixed_mask[row_idx, length:] = True

        for row_idx, (sample_spans, sample_starts) in enumerate(zip(span_token_ids, span_start_positions)):
            if len(sample_spans) != len(sample_starts):
                raise ValueError("Each sample's span list must match its start-position list")
            effective_length = effective_lengths[row_idx]
            prev_end = 1
            for span_ids, start_pos in zip(sample_spans, sample_starts):
                span = [int(token_id) for token_id in span_ids]
                if not span:
                    continue
                end_pos = int(start_pos) + len(span)
                if int(start_pos) < 1 or end_pos > (effective_length - 1):
                    raise ValueError(
                        f"Fixed span [{start_pos}, {end_pos}) does not fit sequence length {effective_length}"
                    )
                if int(start_pos) < prev_end:
                    raise ValueError(
                        f"Overlapping fixed spans detected for sample {row_idx}: start={start_pos}, previous_end={prev_end}"
                    )
                ids[row_idx, int(start_pos) : end_pos] = torch.tensor(span, dtype=torch.long, device=self.device)
                fixed_mask[row_idx, int(start_pos) : end_pos] = True
                prev_end = end_pos

        return self._sample_from_ids_with_trajectory(ids, attention_mask, fixed_mask, show_progress=show_progress)

    def sample_batch_trajectory(
        self,
        *,
        num_samples: int,
        seq_length: int,
        batch_size: int = 256,
        show_progress: bool = True,
        lengths: Sequence[int] | None = None,
    ) -> Tuple[List[torch.Tensor], List[str], List[SamplingTrajectoryRecord]]:
        all_ids: List[torch.Tensor] = []
        all_smiles: List[str] = []
        all_trajectories: List[SamplingTrajectoryRecord] = []
        if lengths is not None and len(lengths) != int(num_samples):
            raise ValueError("lengths must match num_samples")

        sample_idx = 0
        num_batches = (int(num_samples) + int(batch_size) - 1) // int(batch_size)
        iterator = range(num_batches)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Batch trajectory sampling")

        for _ in iterator:
            current_batch_size = min(int(batch_size), int(num_samples) - sample_idx)
            if lengths is None:
                ids, smiles, trajectory = self.sample_with_trajectory(
                    batch_size=current_batch_size,
                    seq_length=int(seq_length),
                    show_progress=False,
                )
            else:
                batch_lengths = lengths[sample_idx : sample_idx + current_batch_size]
                ids, smiles, trajectory = self.sample_with_lengths_trajectory(
                    batch_lengths,
                    max_length=int(seq_length),
                    show_progress=False,
                )
            all_ids.append(ids)
            all_smiles.extend(smiles)
            all_trajectories.append(trajectory)
            sample_idx += current_batch_size
        return all_ids, all_smiles, all_trajectories

    def sample_batch_with_fixed_spans_trajectory(
        self,
        *,
        num_samples: int,
        seq_length: int,
        span_token_ids: Sequence[Sequence[int]],
        span_start_positions: Sequence[int],
        batch_size: int = 256,
        show_progress: bool = True,
        lengths: Sequence[int] | None = None,
    ) -> Tuple[List[torch.Tensor], List[str], List[SamplingTrajectoryRecord]]:
        if len(span_token_ids) != int(num_samples):
            raise ValueError("span_token_ids must match num_samples")
        if len(span_start_positions) != int(num_samples):
            raise ValueError("span_start_positions must match num_samples")
        if lengths is not None and len(lengths) != int(num_samples):
            raise ValueError("lengths must match num_samples")

        all_ids: List[torch.Tensor] = []
        all_smiles: List[str] = []
        all_trajectories: List[SamplingTrajectoryRecord] = []
        sample_idx = 0
        num_batches = (int(num_samples) + int(batch_size) - 1) // int(batch_size)
        iterator = range(num_batches)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Batch trajectory sampling")

        for _ in iterator:
            current_batch_size = min(int(batch_size), int(num_samples) - sample_idx)
            batch_span_ids = span_token_ids[sample_idx : sample_idx + current_batch_size]
            batch_span_starts = span_start_positions[sample_idx : sample_idx + current_batch_size]
            batch_lengths = None if lengths is None else lengths[sample_idx : sample_idx + current_batch_size]
            ids, smiles, trajectory = self.sample_with_fixed_spans_trajectory(
                span_token_ids=batch_span_ids,
                span_start_positions=batch_span_starts,
                seq_length=int(seq_length),
                lengths=batch_lengths,
                show_progress=False,
            )
            all_ids.append(ids)
            all_smiles.extend(smiles)
            all_trajectories.append(trajectory)
            sample_idx += current_batch_size
        return all_ids, all_smiles, all_trajectories

    def sample_batch_with_multiple_fixed_spans_trajectory(
        self,
        *,
        num_samples: int,
        seq_length: int,
        span_token_ids: Sequence[Sequence[Sequence[int]]],
        span_start_positions: Sequence[Sequence[int]],
        batch_size: int = 256,
        show_progress: bool = True,
        lengths: Sequence[int] | None = None,
    ) -> Tuple[List[torch.Tensor], List[str], List[SamplingTrajectoryRecord]]:
        if len(span_token_ids) != int(num_samples):
            raise ValueError("span_token_ids must match num_samples")
        if len(span_start_positions) != int(num_samples):
            raise ValueError("span_start_positions must match num_samples")
        if lengths is not None and len(lengths) != int(num_samples):
            raise ValueError("lengths must match num_samples")

        all_ids: List[torch.Tensor] = []
        all_smiles: List[str] = []
        all_trajectories: List[SamplingTrajectoryRecord] = []
        sample_idx = 0
        num_batches = (int(num_samples) + int(batch_size) - 1) // int(batch_size)
        iterator = range(num_batches)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Batch trajectory sampling")

        for _ in iterator:
            current_batch_size = min(int(batch_size), int(num_samples) - sample_idx)
            batch_span_ids = span_token_ids[sample_idx : sample_idx + current_batch_size]
            batch_span_starts = span_start_positions[sample_idx : sample_idx + current_batch_size]
            batch_lengths = None if lengths is None else lengths[sample_idx : sample_idx + current_batch_size]
            ids, smiles, trajectory = self.sample_with_multiple_fixed_spans_trajectory(
                span_token_ids=batch_span_ids,
                span_start_positions=batch_span_starts,
                seq_length=int(seq_length),
                lengths=batch_lengths,
                show_progress=False,
            )
            all_ids.append(ids)
            all_smiles.extend(smiles)
            all_trajectories.append(trajectory)
            sample_idx += current_batch_size
        return all_ids, all_smiles, all_trajectories

    def sample_with_trajectory(
        self,
        batch_size: int,
        seq_length: int,
        show_progress: bool = True,
    ) -> Tuple[torch.Tensor, List[str], SamplingTrajectoryRecord]:
        ids = torch.full(
            (batch_size, seq_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device,
        )
        ids[:, 0] = self.bos_id
        ids[:, -1] = self.eos_id
        attention_mask = torch.ones_like(ids)
        fixed_mask = ids != self.mask_id
        return self._sample_from_ids_with_trajectory(ids, attention_mask, fixed_mask, show_progress=show_progress)

    def _sample_from_ids_with_trajectory(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        *,
        show_progress: bool = True,
    ) -> Tuple[torch.Tensor, List[str], SamplingTrajectoryRecord]:
        self.diffusion_model.eval()
        batch_size = ids.shape[0]
        steps = range(self.num_steps, 0, -1)
        if show_progress:
            from tqdm import tqdm

            steps = tqdm(steps, desc="Conditional trajectory sampling")

        records: List[TrajectoryStepRecord] = []
        final_logits = None
        for t in steps:
            ids_before = ids.clone()
            timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            logits = self._compute_conditioned_logits(ids, timesteps, attention_mask, fixed_mask)
            probs = self._logits_to_probs(logits)
            is_masked = (ids == self.mask_id) & (~fixed_mask)
            unmask_prob = 1.0 / t

            step_positions: List[torch.Tensor] = []
            step_tokens: List[torch.Tensor] = []
            for i in range(batch_size):
                masked_pos = torch.where(is_masked[i])[0]
                if len(masked_pos) == 0:
                    step_positions.append(torch.empty((0,), dtype=torch.long, device=self.device))
                    step_tokens.append(torch.empty((0,), dtype=torch.long, device=self.device))
                    continue

                num_unmask = max(1, int(len(masked_pos) * unmask_prob))
                unmask_indices = torch.randperm(len(masked_pos), device=self.device)[:num_unmask]
                unmask_positions = masked_pos[unmask_indices]
                chosen_tokens: List[int] = []
                for pos in unmask_positions:
                    sampled = torch.multinomial(probs[i, pos], 1)
                    token_id = int(sampled.item())
                    ids[i, pos] = sampled
                    probs[i] = self._apply_within_step_constraint_updates(
                        logits[i],
                        probs[i],
                        ids[i],
                        fixed_mask[i],
                        token_id,
                        int(pos.item()),
                    )
                    chosen_tokens.append(token_id)
                step_positions.append(unmask_positions.to(dtype=torch.long))
                step_tokens.append(torch.tensor(chosen_tokens, dtype=torch.long, device=self.device))

            padded_positions, unmask_counts = _pad_event_matrix(step_positions, device=self.device)
            padded_tokens, _ = _pad_event_matrix(step_tokens, device=self.device)
            records.append(
                TrajectoryStepRecord(
                    timestep=int(t),
                    ids_before=ids_before.detach().clone(),
                    attention_mask=attention_mask.detach().clone(),
                    fixed_mask=fixed_mask.detach().clone(),
                    unmask_positions=padded_positions.detach().clone(),
                    sampled_token_ids=padded_tokens.detach().clone(),
                    unmask_counts=unmask_counts.detach().clone(),
                )
            )
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
        trajectory = SamplingTrajectoryRecord(
            condition_bundle=self._condition_for_batch(batch_size).detach().clone(),
            cfg_scale=float(self.cfg_scale),
            final_ids=ids.detach().clone(),
            final_attention_mask=attention_mask.detach().clone(),
            steps=records,
        )
        return ids, smiles_list, trajectory

    def replay_trajectory_logprob(
        self,
        trajectory: SamplingTrajectoryRecord,
        *,
        grad_enabled: bool = True,
        diffusion_model_override=None,
    ) -> Dict[str, torch.Tensor]:
        """Replay a recorded trajectory under the current conditional policy."""

        batch_size = int(trajectory.final_ids.shape[0])
        total_logprob = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        per_step_logprob: List[torch.Tensor] = []

        original_condition = self.condition_bundle
        self.condition_bundle = trajectory.condition_bundle.to(self.device)
        try:
            for step in trajectory.steps:
                ids = step.ids_before.to(self.device).clone()
                attention_mask = step.attention_mask.to(self.device)
                fixed_mask = step.fixed_mask.to(self.device)
                timesteps = torch.full((batch_size,), int(step.timestep), device=self.device, dtype=torch.long)
                logits = self._compute_conditioned_logits(
                    ids,
                    timesteps,
                    attention_mask,
                    fixed_mask,
                    grad_enabled=grad_enabled,
                    diffusion_model_override=diffusion_model_override,
                )
                step_logprob_rows: List[torch.Tensor] = []

                for i in range(batch_size):
                    logits_row = logits[i].clone()
                    probs_row = self._logits_to_probs(logits_row)
                    ids_row = ids[i].clone()
                    count = int(step.unmask_counts[i].item())
                    event_terms: List[torch.Tensor] = []
                    for event_idx in range(count):
                        pos = int(step.unmask_positions[i, event_idx].item())
                        token_id = int(step.sampled_token_ids[i, event_idx].item())
                        event_prob = probs_row[pos, token_id].clamp(min=1.0e-12)
                        event_terms.append(torch.log(event_prob))
                        ids_row[pos] = token_id
                        probs_row = self._apply_within_step_constraint_updates(
                            logits_row,
                            probs_row,
                            ids_row,
                            fixed_mask[i],
                            token_id,
                            pos,
                        )
                    if event_terms:
                        step_logprob_rows.append(torch.stack(event_terms).sum())
                    else:
                        step_logprob_rows.append(torch.zeros((), dtype=torch.float32, device=self.device))

                step_logprob = torch.stack(step_logprob_rows, dim=0)

                total_logprob = total_logprob + step_logprob
                per_step_logprob.append(step_logprob)
        finally:
            self.condition_bundle = original_condition

        stacked = (
            torch.stack(per_step_logprob, dim=1)
            if per_step_logprob
            else torch.empty((batch_size, 0), dtype=torch.float32, device=self.device)
        )
        return {
            "trajectory_logprob": total_logprob,
            "per_step_logprob": stacked,
        }

    def compute_trajectory_kl(
        self,
        trajectory: SamplingTrajectoryRecord,
        *,
        reference_diffusion_model,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-trajectory KL against a frozen reference policy."""

        batch_size = int(trajectory.final_ids.shape[0])
        total_kl = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        per_step_kl: List[torch.Tensor] = []

        original_condition = self.condition_bundle
        self.condition_bundle = trajectory.condition_bundle.to(self.device)
        try:
            for step in trajectory.steps:
                ids = step.ids_before.to(self.device).clone()
                attention_mask = step.attention_mask.to(self.device)
                fixed_mask = step.fixed_mask.to(self.device)
                active_mask = ((ids == self.mask_id) & (~fixed_mask) & attention_mask.bool()).float()
                timesteps = torch.full((batch_size,), int(step.timestep), device=self.device, dtype=torch.long)
                policy_logits = self._compute_conditioned_logits(
                    ids,
                    timesteps,
                    attention_mask,
                    fixed_mask,
                    grad_enabled=True,
                )
                with torch.no_grad():
                    reference_logits = self._compute_conditioned_logits(
                        ids,
                        timesteps,
                        attention_mask,
                        fixed_mask,
                        grad_enabled=False,
                        diffusion_model_override=reference_diffusion_model,
                    )
                policy_probs = self._logits_to_probs(policy_logits)
                reference_probs = self._logits_to_probs(reference_logits)
                policy_log_probs = torch.log(policy_probs.clamp(min=1.0e-12))
                reference_log_probs = torch.log(reference_probs.clamp(min=1.0e-12))
                per_position_kl = (policy_probs * (policy_log_probs - reference_log_probs)).sum(dim=-1)
                denom = active_mask.sum(dim=1).clamp(min=1.0)
                step_kl = (per_position_kl * active_mask).sum(dim=1) / denom
                total_kl = total_kl + step_kl
                per_step_kl.append(step_kl)
        finally:
            self.condition_bundle = original_condition

        stacked = (
            torch.stack(per_step_kl, dim=1)
            if per_step_kl
            else torch.empty((batch_size, 0), dtype=torch.float32, device=self.device)
        )
        return {
            "trajectory_kl": total_kl,
            "per_step_kl": stacked,
        }

    def single_step_logprob_t1(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate clean-sequence log-prob at t=1 for DPO-style objectives."""

        batch_size = int(input_ids.shape[0])
        cond = self._condition_for_batch(batch_size)
        timesteps = torch.ones(batch_size, dtype=torch.long, device=self.device)
        logits = self.diffusion_model.classifier_free_guidance_logits(
            input_ids,
            timesteps,
            attention_mask,
            condition_bundle=cond,
            cfg_scale=self.cfg_scale,
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        active_mask = attention_mask.bool()
        if hasattr(self, "pad_id"):
            active_mask = active_mask & (input_ids != self.pad_id)
        return (token_log_probs * active_mask.float()).sum(dim=1)


def sample_trajectories_with_class_prior(
    *,
    sampler: TrajectoryConditionalSampler,
    tokenizer: PSmilesTokenizer,
    prior: ResolvedClassSamplingPrior,
    resolved: ResolvedStep62Config,
    num_samples: int,
    show_progress: bool = True,
) -> Tuple[List[str], List[SamplingTrajectoryRecord], Dict[str, object]]:
    """Trajectory-enabled variant of Step 6 class-prior sampling."""

    sampling_cfg = resolved.base_config.get("sampling", {})
    lengths = _sample_lengths(
        prior=prior,
        tokenizer=tokenizer,
        num_samples=int(num_samples),
        sampling_cfg=sampling_cfg,
    )
    batch_size = int(sampling_cfg.get("batch_size", 128))
    if prior.motif_token_ids:
        multi_spans, multi_span_starts, lengths = _build_decode_constraint_multi_spans(
            motif_token_ids=prior.motif_token_ids,
            lengths=lengths,
            center_min_frac=prior.center_min_frac,
            center_max_frac=prior.center_max_frac,
            seq_length=int(tokenizer.max_length),
            spans_per_sample=prior.spans_per_sample,
        )
        all_ids, smiles, trajectories = sampler.sample_batch_with_multiple_fixed_spans_trajectory(
            num_samples=int(num_samples),
            seq_length=int(tokenizer.max_length),
            span_token_ids=multi_spans,
            span_start_positions=multi_span_starts,
            batch_size=batch_size,
            show_progress=show_progress,
            lengths=lengths,
        )
    else:
        all_ids, smiles, trajectories = sampler.sample_batch_trajectory(
            num_samples=int(num_samples),
            seq_length=int(tokenizer.max_length),
            batch_size=batch_size,
            show_progress=show_progress,
            lengths=lengths,
        )

    metadata = {
        "num_samples": int(num_samples),
        "num_trajectory_batches": int(len(trajectories)),
        "spans_per_sample": int(prior.spans_per_sample),
        "motif_count": int(len(prior.motifs)),
        "motif_source": prior.motif_source,
        "length_prior_count": int(len(prior.length_prior_lengths)),
        "length_prior_source": prior.length_prior_source,
        "class_token_bias_enabled": bool(prior.class_token_logit_bias is not None),
        "class_token_bias_strength": float(prior.class_token_bias_strength),
    }
    return smiles, trajectories, metadata
