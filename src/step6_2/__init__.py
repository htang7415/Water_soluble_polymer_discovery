"""Shared utilities for Step 6_2 inverse design."""

from .condition_encoder import ConditionEncoder
from .conditional_diffusion import ConditionalDiscreteMaskingDiffusion
from .conditional_dit import ConditionalDiffusionBackbone
from .config import (
    ExactChiTargetLookup,
    ResolvedStep62Config,
    build_run_config,
    load_step6_2_config,
)
from .dataset import (
    ConditionScaler,
    Step62ConditionalDataset,
    build_condition_bundle,
    build_condition_scaler,
    build_source_batch_counts,
    build_step62_supervised_frames,
    load_step62_water_dataset,
    step62_collate_fn,
)
from .dpo import (
    DpoTrainingArtifacts,
    Step62DpoDataset,
    build_dpo_pair_splits,
    step62_dpo_collate_fn,
    train_s4_dpo_alignment,
)
from .hpo import STUDY_BASE_RUNS, run_optuna_study, suggest_trial_params
from .rl_trainer import RlTrainingArtifacts, train_s4_rl_alignment
from .run_core import build_s4_warm_start_run_cfg, create_run_dirs, execute_step62_run, run_single_target_sampling
from .supervised import (
    Step62AuxHeads,
    build_optimizer_and_scheduler,
    build_s2_components_from_step1,
    compute_s2_mt_losses,
    load_step62_checkpoint_into_modules,
)
from .train_s2 import S2TrainingArtifacts, train_s2_supervised_run
from .trajectory import (
    SamplingTrajectoryRecord,
    TrajectoryConditionalSampler,
    TrajectoryStepRecord,
)

__all__ = [
    "ConditionEncoder",
    "ConditionalDiffusionBackbone",
    "ConditionalDiscreteMaskingDiffusion",
    "ConditionScaler",
    "DpoTrainingArtifacts",
    "ExactChiTargetLookup",
    "RlTrainingArtifacts",
    "ResolvedStep62Config",
    "Step62ConditionalDataset",
    "Step62AuxHeads",
    "S2TrainingArtifacts",
    "Step62DpoDataset",
    "SamplingTrajectoryRecord",
    "build_condition_bundle",
    "build_condition_scaler",
    "build_dpo_pair_splits",
    "build_optimizer_and_scheduler",
    "build_run_config",
    "build_s2_components_from_step1",
    "build_s4_warm_start_run_cfg",
    "build_source_batch_counts",
    "build_step62_supervised_frames",
    "compute_s2_mt_losses",
    "create_run_dirs",
    "execute_step62_run",
    "load_step6_2_config",
    "load_step62_checkpoint_into_modules",
    "load_step62_water_dataset",
    "run_optuna_study",
    "run_single_target_sampling",
    "STUDY_BASE_RUNS",
    "step62_collate_fn",
    "step62_dpo_collate_fn",
    "suggest_trial_params",
    "train_s4_dpo_alignment",
    "train_s4_rl_alignment",
    "train_s2_supervised_run",
    "TrajectoryConditionalSampler",
    "TrajectoryStepRecord",
]
