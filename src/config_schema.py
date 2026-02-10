"""Pydantic schema for experiment.yaml validation.

Hard-codes allowed values for Phase 1. In Phase 2 (Core), these will be loaded
from configs/project_standards.yaml instead.

Validation errors produce clear messages so agents can self-correct.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Hard-coded allowed values (Phase 1)
# Phase 2 will load these from configs/project_standards.yaml
# ---------------------------------------------------------------------------
ALLOWED_MODEL_TYPES: list[str] = [
    "xgboost",
    "lightgbm",
    "logistic_regression",
    "random_forest",
]

ALLOWED_METRICS: list[str] = [
    "roc_auc",
    "f1",
    "precision",
    "recall",
    "log_loss",
]

ALLOWED_SPLIT_STRATEGIES: list[str] = [
    "stratified_kfold",
    "temporal_split",
]

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------
class EvaluationConfig(BaseModel):
    """Evaluation settings: which metric to optimise and how to split data."""

    metric: str
    split_strategy: str

    @field_validator("metric")
    @classmethod
    def metric_must_be_allowed(cls, v: str) -> str:
        if v not in ALLOWED_METRICS:
            raise ValueError(
                f"metric '{v}' is not allowed. "
                f"Choose one of: {ALLOWED_METRICS}"
            )
        return v

    @field_validator("split_strategy")
    @classmethod
    def split_strategy_must_be_allowed(cls, v: str) -> str:
        if v not in ALLOWED_SPLIT_STRATEGIES:
            raise ValueError(
                f"split_strategy '{v}' is not allowed. "
                f"Choose one of: {ALLOWED_SPLIT_STRATEGIES}"
            )
        return v


class ProtectedConfig(BaseModel):
    """Protected fields that must not change without explicit human approval."""

    metric_definition: str
    split_strategy: str

    @field_validator("metric_definition")
    @classmethod
    def metric_definition_must_be_allowed(cls, v: str) -> str:
        if v not in ALLOWED_METRICS:
            raise ValueError(
                f"protected.metric_definition '{v}' is not allowed. "
                f"Choose one of: {ALLOWED_METRICS}"
            )
        return v

    @field_validator("split_strategy")
    @classmethod
    def split_strategy_must_be_allowed(cls, v: str) -> str:
        if v not in ALLOWED_SPLIT_STRATEGIES:
            raise ValueError(
                f"protected.split_strategy '{v}' is not allowed. "
                f"Choose one of: {ALLOWED_SPLIT_STRATEGIES}"
            )
        return v


# ---------------------------------------------------------------------------
# Main experiment config
# ---------------------------------------------------------------------------
class ExperimentConfig(BaseModel):
    """Schema for configs/experiment.yaml.

    Enforces required fields, types, and allowed values.  Validation errors
    include the field name and the set of allowed values so that an agent can
    self-correct without extra look-ups.
    """

    project_name: str = Field(
        ..., min_length=1, description="Unique project identifier"
    )
    client_name: str = Field(
        ..., min_length=1, description="Client identifier for MLflow tagging"
    )
    dataset_version: str = Field(
        ..., description="SHA256 content hash of the dataset in data/shadow/"
    )
    feature_pipeline_version: str = Field(
        default="auto",
        description="Overwritten at runtime with computed hash",
    )
    model_type: str = Field(
        ..., description="Model type to train"
    )
    random_seed: int = Field(
        ..., description="Random seed for reproducibility"
    )
    hyperparameters: dict[str, Any] = Field(
        ..., description="Model-specific hyperparameters"
    )
    evaluation: EvaluationConfig
    protected: ProtectedConfig

    # -- field validators ------------------------------------------------

    @field_validator("dataset_version")
    @classmethod
    def dataset_version_must_be_sha256(cls, v: str) -> str:
        if not _SHA256_PATTERN.match(v):
            raise ValueError(
                f"dataset_version must be a 64-character lowercase hex string "
                f"(SHA256 hash), got '{v}'"
            )
        return v

    @field_validator("model_type")
    @classmethod
    def model_type_must_be_allowed(cls, v: str) -> str:
        if v not in ALLOWED_MODEL_TYPES:
            raise ValueError(
                f"model_type '{v}' is not allowed. "
                f"Choose one of: {ALLOWED_MODEL_TYPES}"
            )
        return v

    @field_validator("random_seed")
    @classmethod
    def random_seed_must_be_nonnegative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                f"random_seed must be a non-negative integer, got {v}"
            )
        return v

    @field_validator("hyperparameters")
    @classmethod
    def hyperparameters_must_not_be_empty(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("hyperparameters must not be empty")
        return v

    # -- cross-field validators ------------------------------------------

    @model_validator(mode="after")
    def protected_must_match_evaluation(self) -> "ExperimentConfig":
        """The protected block's values must be consistent with evaluation."""
        errors: list[str] = []
        if self.protected.metric_definition != self.evaluation.metric:
            errors.append(
                f"protected.metric_definition ('{self.protected.metric_definition}') "
                f"does not match evaluation.metric ('{self.evaluation.metric}'). "
                f"These must be the same value."
            )
        if self.protected.split_strategy != self.evaluation.split_strategy:
            errors.append(
                f"protected.split_strategy ('{self.protected.split_strategy}') "
                f"does not match evaluation.split_strategy "
                f"('{self.evaluation.split_strategy}'). "
                f"These must be the same value."
            )
        if errors:
            raise ValueError(
                "Protected field mismatch:\n  " + "\n  ".join(errors)
            )
        return self


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file.

    Returns a validated ExperimentConfig or raises a clear ValidationError.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if not path.suffix in (".yaml", ".yml"):
        raise ValueError(f"Config file must be .yaml or .yml, got: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping, got {type(raw).__name__}"
        )

    return ExperimentConfig(**raw)
