"""Guided sampler for Step 6_2 S1."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from src.sampling.sampler import ConstrainedSampler
from src.step6_2.conditional_sampling import ConditionalConstrainedSampler
from src.step6_2.evaluation import Step62Evaluator
from src.step6_2.rewards import score_guidance_batch


class GuidedSampler(ConstrainedSampler):
    """Frozen-model guided sampler with late oracle guidance."""

    def __init__(
        self,
        *,
        evaluator: Step62Evaluator,
        target_row: Dict[str, object],
        best_of_k: int,
        guidance_start_frac: float,
        class_guidance_start_frac: float,
        class_guidance_min_valid_frac: float,
        class_surrogate_mode: str,
        sol_log_prob_floor: float,
        w_sol: float,
        w_chi: float,
        w_class: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.evaluator = evaluator
        self.target_row = target_row
        self.best_of_k = int(best_of_k)
        self.guidance_start_frac = float(guidance_start_frac)
        self.class_guidance_start_frac = float(class_guidance_start_frac)
        self.class_guidance_min_valid_frac = float(class_guidance_min_valid_frac)
        self.class_surrogate_mode = str(class_surrogate_mode)
        self.sol_log_prob_floor = float(sol_log_prob_floor)
        self.w_sol = float(w_sol)
        self.w_chi = float(w_chi)
        self.w_class = float(w_class)
        self.training_oracle_calls_soluble = 0
        self.training_oracle_calls_chi = 0
        self.class_guidance_suppressed_steps = 0

    def reset_guidance_stats(self) -> None:
        self.training_oracle_calls_soluble = 0
        self.training_oracle_calls_chi = 0
        self.class_guidance_suppressed_steps = 0

    def get_guidance_stats(self) -> Dict[str, int]:
        return {
            "training_soluble_oracle_calls": int(self.training_oracle_calls_soluble),
            "training_chi_oracle_calls": int(self.training_oracle_calls_chi),
            "class_guidance_suppressed_steps": int(self.class_guidance_suppressed_steps),
        }

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

    def _sample_from_ids(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        show_progress: bool = True,
    ) -> Tuple[torch.Tensor, List[str]]:
        self.diffusion_model.eval()
        backbone = self.diffusion_model.backbone
        batch_size = ids.shape[0]
        final_logits = None
        steps = range(self.num_steps, 0, -1)

        if show_progress:
            from tqdm import tqdm

            steps = tqdm(steps, desc="Guided sampling")

        for t in steps:
            timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            step_progress = self._step_progress_frac(int(t))
            with torch.no_grad():
                logits = backbone(ids, timesteps, attention_mask)

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
            probs = self._logits_to_probs(logits)

            is_masked = (ids == self.mask_id) & (~fixed_mask)
            unmask_prob = 1.0 / t
            guided = step_progress >= self.guidance_start_frac and self.best_of_k >= 2

            if not guided:
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
                        probs[i] = self._apply_within_step_constraint_updates(
                            logits[i],
                            probs[i],
                            ids[i],
                            fixed_mask[i],
                            int(sampled.item()),
                            int(pos.item()),
                        )
            else:
                candidate_blocks: List[torch.Tensor] = []
                attention_blocks: List[torch.Tensor] = []
                unmask_positions_by_sample: Dict[int, torch.Tensor] = {}

                for i in range(batch_size):
                    masked_pos = torch.where(is_masked[i])[0]
                    if len(masked_pos) == 0:
                        continue
                    num_unmask = max(1, int(len(masked_pos) * unmask_prob))
                    unmask_indices = torch.randperm(len(masked_pos), device=self.device)[:num_unmask]
                    unmask_positions = masked_pos[unmask_indices]
                    unmask_positions_by_sample[i] = unmask_positions

                    candidate_ids = ids[i].unsqueeze(0).repeat(self.best_of_k, 1)
                    for pos in masked_pos:
                        sampled = torch.multinomial(probs[i, pos], self.best_of_k, replacement=True)
                        candidate_ids[:, int(pos.item())] = sampled
                    candidate_blocks.append(candidate_ids)
                    attention_blocks.append(attention_mask[i].unsqueeze(0).repeat(self.best_of_k, 1))

                if candidate_blocks:
                    provisional_ids = torch.cat(candidate_blocks, dim=0)
                    provisional_attention = torch.cat(attention_blocks, dim=0)
                    class_term_enabled = step_progress >= self.class_guidance_start_frac
                    scored = score_guidance_batch(
                        provisional_ids,
                        provisional_attention,
                        target_row=self.target_row,
                        evaluator=self.evaluator,
                        tokenizer=self.tokenizer,
                        class_surrogate_mode=self.class_surrogate_mode,
                        class_term_enabled=class_term_enabled,
                        sol_log_prob_floor=self.sol_log_prob_floor,
                        w_sol=self.w_sol,
                        w_chi=self.w_chi,
                        w_class=self.w_class,
                    )
                    self.training_oracle_calls_soluble += int(scored["oracle_calls_soluble"])
                    self.training_oracle_calls_chi += int(scored["oracle_calls_chi"])

                    valid_frac = float(scored["valid_frac"])
                    class_term_final = bool(class_term_enabled and valid_frac >= self.class_guidance_min_valid_frac)
                    if class_term_enabled and not class_term_final:
                        self.class_guidance_suppressed_steps += 1
                        scored["reward"] = scored["reward"] - float(self.w_class) * scored["class_surrogate"]

                    rewards = scored["reward"]
                    reward_offset = 0
                    for i, unmask_positions in unmask_positions_by_sample.items():
                        sample_rewards = rewards[reward_offset : reward_offset + self.best_of_k]
                        best_idx = int(torch.argmax(sample_rewards).item())
                        chosen_ids = provisional_ids[reward_offset + best_idx]
                        reward_offset += self.best_of_k
                        for pos in unmask_positions:
                            chosen_token = int(chosen_ids[int(pos.item())].item())
                            ids[i, int(pos.item())] = chosen_token
                            probs[i] = self._apply_within_step_constraint_updates(
                                logits[i],
                                probs[i],
                                ids[i],
                                fixed_mask[i],
                                chosen_token,
                                int(pos.item()),
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
        return ids, smiles_list


class GuidedConditionalSampler(ConditionalConstrainedSampler):
    """Conditional guided sampler for Step 6_2 S3."""

    def __init__(
        self,
        *,
        evaluator: Step62Evaluator,
        target_row: Dict[str, object],
        best_of_k: int,
        guidance_start_frac: float,
        class_guidance_start_frac: float,
        class_guidance_min_valid_frac: float,
        class_surrogate_mode: str,
        sol_log_prob_floor: float,
        w_sol: float,
        w_chi: float,
        w_class: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.evaluator = evaluator
        self.target_row = target_row
        self.best_of_k = int(best_of_k)
        self.guidance_start_frac = float(guidance_start_frac)
        self.class_guidance_start_frac = float(class_guidance_start_frac)
        self.class_guidance_min_valid_frac = float(class_guidance_min_valid_frac)
        self.class_surrogate_mode = str(class_surrogate_mode)
        self.sol_log_prob_floor = float(sol_log_prob_floor)
        self.w_sol = float(w_sol)
        self.w_chi = float(w_chi)
        self.w_class = float(w_class)
        self.training_oracle_calls_soluble = 0
        self.training_oracle_calls_chi = 0
        self.class_guidance_suppressed_steps = 0

    def reset_guidance_stats(self) -> None:
        self.training_oracle_calls_soluble = 0
        self.training_oracle_calls_chi = 0
        self.class_guidance_suppressed_steps = 0

    def get_guidance_stats(self) -> Dict[str, int]:
        return {
            "training_soluble_oracle_calls": int(self.training_oracle_calls_soluble),
            "training_chi_oracle_calls": int(self.training_oracle_calls_chi),
            "class_guidance_suppressed_steps": int(self.class_guidance_suppressed_steps),
        }

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

            steps = tqdm(steps, desc="Conditional guided sampling")

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
            if (
                self.class_token_logit_bias is not None
                and step_progress >= float(self.class_token_bias_start_frac)
            ):
                bias_mask = (~fixed_mask).unsqueeze(-1).float()
                logits = logits + self.class_token_logit_bias.unsqueeze(0).unsqueeze(0) * bias_mask
            logits = self._apply_sampling_filters(logits)
            logits = self._apply_special_token_constraints(logits, ids)
            probs = self._logits_to_probs(logits)

            is_masked = (ids == self.mask_id) & (~fixed_mask)
            unmask_prob = 1.0 / t
            guided = step_progress >= self.guidance_start_frac and self.best_of_k >= 2

            if not guided:
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
                        probs[i] = self._apply_within_step_constraint_updates(
                            logits[i],
                            probs[i],
                            ids[i],
                            fixed_mask[i],
                            int(sampled.item()),
                            int(pos.item()),
                        )
            else:
                candidate_blocks: List[torch.Tensor] = []
                attention_blocks: List[torch.Tensor] = []
                unmask_positions_by_sample: Dict[int, torch.Tensor] = {}

                for i in range(batch_size):
                    masked_pos = torch.where(is_masked[i])[0]
                    if len(masked_pos) == 0:
                        continue
                    num_unmask = max(1, int(len(masked_pos) * unmask_prob))
                    unmask_indices = torch.randperm(len(masked_pos), device=self.device)[:num_unmask]
                    unmask_positions = masked_pos[unmask_indices]
                    unmask_positions_by_sample[i] = unmask_positions

                    candidate_ids = ids[i].unsqueeze(0).repeat(self.best_of_k, 1)
                    for pos in masked_pos:
                        sampled = torch.multinomial(probs[i, pos], self.best_of_k, replacement=True)
                        candidate_ids[:, int(pos.item())] = sampled
                    candidate_blocks.append(candidate_ids)
                    attention_blocks.append(attention_mask[i].unsqueeze(0).repeat(self.best_of_k, 1))

                if candidate_blocks:
                    provisional_ids = torch.cat(candidate_blocks, dim=0)
                    provisional_attention = torch.cat(attention_blocks, dim=0)
                    class_term_enabled = step_progress >= self.class_guidance_start_frac
                    scored = score_guidance_batch(
                        provisional_ids,
                        provisional_attention,
                        target_row=self.target_row,
                        evaluator=self.evaluator,
                        tokenizer=self.tokenizer,
                        class_surrogate_mode=self.class_surrogate_mode,
                        class_term_enabled=class_term_enabled,
                        sol_log_prob_floor=self.sol_log_prob_floor,
                        w_sol=self.w_sol,
                        w_chi=self.w_chi,
                        w_class=self.w_class,
                    )
                    self.training_oracle_calls_soluble += int(scored["oracle_calls_soluble"])
                    self.training_oracle_calls_chi += int(scored["oracle_calls_chi"])

                    valid_frac = float(scored["valid_frac"])
                    class_term_final = bool(class_term_enabled and valid_frac >= self.class_guidance_min_valid_frac)
                    if class_term_enabled and not class_term_final:
                        self.class_guidance_suppressed_steps += 1
                        scored["reward"] = scored["reward"] - float(self.w_class) * scored["class_surrogate"]

                    rewards = scored["reward"]
                    reward_offset = 0
                    for i, unmask_positions in unmask_positions_by_sample.items():
                        sample_rewards = rewards[reward_offset : reward_offset + self.best_of_k]
                        best_idx = int(torch.argmax(sample_rewards).item())
                        chosen_ids = provisional_ids[reward_offset + best_idx]
                        reward_offset += self.best_of_k
                        for pos in unmask_positions:
                            chosen_token = int(chosen_ids[int(pos.item())].item())
                            ids[i, int(pos.item())] = chosen_token
                            probs[i] = self._apply_within_step_constraint_updates(
                                logits[i],
                                probs[i],
                                ids[i],
                                fixed_mask[i],
                                chosen_token,
                                int(pos.item()),
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
        return ids, smiles_list
