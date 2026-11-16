"""Typed access to config.yaml `validation`; defaults match the main experiments.

Every field is a plain typed value, so sweeps only need a config edit. Extra keys are kept.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="allow")


class EncoderConfig(Settings):
    model_path: Path
    embedding_dim: int = 1024
    max_length: int = 512
    batch_size: int = Field(default=32, gt=0)


class ModelConfig(Settings):
    deterministic: bool = True
    max_sequence_length: int = Field(default=10, gt=0)
    embedding_dim: int = Field(default=512, gt=0)
    num_blocks: int = Field(default=2, gt=0)
    num_heads: int = Field(default=2, gt=0)
    dropout: float = 0.1
    batch_size: int = Field(default=128, gt=0)
    max_epochs: int = Field(default=50, gt=0)
    patience: int = Field(default=5, gt=0)
    min_delta: float = 0.0
    learning_rate: float = 1e-4
    popularity_power: float = 1.0
    seeds: list[int] = Field(default_factory=lambda: [42, 43, 44], min_length=1)

    @model_validator(mode="after")
    def validate_model(self) -> "ModelConfig":
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        return self


class EvaluationConfig(Settings):
    checkpoint_metric: str = "HR@10"
    cutoffs: list[int] = Field(default_factory=lambda: [4, 8, 10, 20, 30])
    primary_cutoff: int = 10
    bootstrap_samples: int = Field(default=10000, gt=0)
    familywise_alpha: float = 0.05
    multiple_comparison_correction: str = "bonferroni"
    min_scene_coverage: float = 0.95
    max_arm_coverage_gap: float = 0.05


class CohortConfig(Settings):
    mode: str = "leave_one_out"
    metadata_missing_policy: str = "zero_vector"
    timezone: str = "UTC"
    min_user_interactions: int = Field(default=5, gt=0)
    retained_sequence_length: int = Field(default=13, gt=2)


class GraphExecutionConfig(Settings):
    # Execution choices only: no change to model weights or loss.
    chunk_items: int = Field(default=256, gt=0)
    chunk_nodes: int = Field(default=32768, gt=0)
    # Bounds the sum of squared per-video scene counts in a chunk; used only by scene_attention.
    chunk_scene_pairs: int = Field(default=1048576, gt=0)
    checkpoint: Literal["auto", "always", "never"] = "auto"
    feature_cache_mb: int = Field(default=2048, ge=0)


class ValidationConfig(Settings):
    run_id: str
    dataset: dict[str, Any]
    cohort: CohortConfig = Field(default_factory=CohortConfig)
    encoder: EncoderConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    graph_execution: GraphExecutionConfig = Field(default_factory=GraphExecutionConfig)
    output_dir: Path


def build_validation_config(*, run_id, dataset, settings, model_path, output_dir) -> ValidationConfig:
    """Assemble the typed settings after the caller resolves its own paths."""
    return ValidationConfig.model_validate(
        {
            "run_id": run_id,
            "dataset": dataset,
            "cohort": settings.get("cohort") or {},
            "encoder": {**(settings.get("encoder") or {}), "model_path": model_path},
            "model": settings.get("model") or {},
            "evaluation": settings.get("evaluation") or {},
            "graph_execution": settings.get("graph_execution") or {},
            "output_dir": output_dir,
        }
    )


def validation_config(context) -> ValidationConfig:
    from rec_experiments import load_experiments

    load_experiments(context)
    return build_validation_config(
        run_id=context.run_id,
        dataset={key: context.path("data", key) for key in context.data},
        settings=context.config["validation"],
        model_path=context.path("models", "bge"),
        output_dir=context.run_root,
    )
