"""Tests for config schema validation (Section 5.6).

Covers:
  - Valid config loads without errors.
  - Config with a missing required field raises a clear validation error.
  - Config with an invalid value (e.g., model_type: "invalid") is rejected.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config_schema import ExperimentConfig, load_experiment_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _valid_config_dict() -> dict:
    """Return a minimal valid experiment config as a plain dict."""
    return {
        "project_name": "test_project",
        "client_name": "test_client",
        "dataset_version": "a" * 64,
        "model_type": "xgboost",
        "random_seed": 42,
        "hyperparameters": {"max_depth": 6, "learning_rate": 0.05},
        "evaluation": {
            "metric": "roc_auc",
            "split_strategy": "stratified_kfold",
        },
        "protected": {
            "metric_definition": "roc_auc",
            "split_strategy": "stratified_kfold",
        },
    }


# ---------------------------------------------------------------------------
# Valid config
# ---------------------------------------------------------------------------
class TestValidConfig:
    def test_valid_config_loads(self):
        """A complete, correct config dict is accepted without errors."""
        cfg = ExperimentConfig(**_valid_config_dict())
        assert cfg.project_name == "test_project"
        assert cfg.model_type == "xgboost"
        assert cfg.random_seed == 42
        assert cfg.evaluation.metric == "roc_auc"
        assert cfg.protected.metric_definition == "roc_auc"

    def test_valid_config_from_yaml(self, tmp_path: Path):
        """load_experiment_config reads and validates a YAML file."""
        yaml_content = textwrap.dedent("""\
            project_name: test_project
            client_name: test_client
            dataset_version: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
            model_type: xgboost
            random_seed: 42
            hyperparameters:
              max_depth: 6
              learning_rate: 0.05
            evaluation:
              metric: roc_auc
              split_strategy: stratified_kfold
            protected:
              metric_definition: roc_auc
              split_strategy: stratified_kfold
        """)
        config_file = tmp_path / "experiment.yaml"
        config_file.write_text(yaml_content)
        cfg = load_experiment_config(config_file)
        assert cfg.project_name == "test_project"
        assert cfg.evaluation.split_strategy == "stratified_kfold"

    def test_feature_pipeline_version_defaults_to_auto(self):
        """feature_pipeline_version defaults to 'auto' when omitted."""
        cfg = ExperimentConfig(**_valid_config_dict())
        assert cfg.feature_pipeline_version == "auto"

    @pytest.mark.parametrize("model_type", [
        "xgboost", "lightgbm", "logistic_regression", "random_forest",
    ])
    def test_all_allowed_model_types_accepted(self, model_type: str):
        d = _valid_config_dict()
        d["model_type"] = model_type
        cfg = ExperimentConfig(**d)
        assert cfg.model_type == model_type

    @pytest.mark.parametrize("metric", [
        "roc_auc", "f1", "precision", "recall", "log_loss",
    ])
    def test_all_allowed_metrics_accepted(self, metric: str):
        d = _valid_config_dict()
        d["evaluation"]["metric"] = metric
        d["protected"]["metric_definition"] = metric
        cfg = ExperimentConfig(**d)
        assert cfg.evaluation.metric == metric


# ---------------------------------------------------------------------------
# Missing required fields
# ---------------------------------------------------------------------------
class TestMissingFields:
    @pytest.mark.parametrize("field", [
        "project_name",
        "client_name",
        "dataset_version",
        "model_type",
        "random_seed",
        "hyperparameters",
        "evaluation",
        "protected",
    ])
    def test_missing_required_field_rejected(self, field: str):
        """Each required top-level field must be present."""
        d = _valid_config_dict()
        del d[field]
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(**d)
        # Pydantic error should mention the missing field
        assert field in str(exc_info.value)

    def test_missing_evaluation_metric_rejected(self):
        d = _valid_config_dict()
        del d["evaluation"]["metric"]
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(**d)
        assert "metric" in str(exc_info.value)

    def test_missing_protected_block_fields_rejected(self):
        d = _valid_config_dict()
        del d["protected"]["metric_definition"]
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(**d)
        assert "metric_definition" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Invalid values
# ---------------------------------------------------------------------------
class TestInvalidValues:
    def test_invalid_model_type_rejected(self):
        """model_type not in allowed list is rejected with a clear message."""
        d = _valid_config_dict()
        d["model_type"] = "invalid"
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(**d)
        err = str(exc_info.value)
        assert "invalid" in err
        assert "model_type" in err

    def test_invalid_metric_rejected(self):
        d = _valid_config_dict()
        d["evaluation"]["metric"] = "mape"
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(**d)
        assert "mape" in str(exc_info.value)

    def test_invalid_split_strategy_rejected(self):
        d = _valid_config_dict()
        d["evaluation"]["split_strategy"] = "leave_one_out"
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(**d)
        assert "leave_one_out" in str(exc_info.value)

    def test_invalid_dataset_version_rejected(self):
        """dataset_version must be a 64-char hex string (SHA256)."""
        d = _valid_config_dict()
        d["dataset_version"] = "not-a-hash"
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(**d)
        assert "not-a-hash" in str(exc_info.value)

    def test_negative_random_seed_rejected(self):
        d = _valid_config_dict()
        d["random_seed"] = -1
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(**d)
        assert "-1" in str(exc_info.value)

    def test_empty_hyperparameters_rejected(self):
        d = _valid_config_dict()
        d["hyperparameters"] = {}
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(**d)
        assert "hyperparameters" in str(exc_info.value)

    def test_empty_project_name_rejected(self):
        d = _valid_config_dict()
        d["project_name"] = ""
        with pytest.raises(ValidationError):
            ExperimentConfig(**d)

    def test_protected_evaluation_mismatch_rejected(self):
        """protected fields must match their evaluation counterparts."""
        d = _valid_config_dict()
        d["protected"]["metric_definition"] = "f1"
        # evaluation.metric is still roc_auc -> mismatch
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(**d)
        assert "mismatch" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# File loader edge cases
# ---------------------------------------------------------------------------
class TestLoaderEdgeCases:
    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_experiment_config("nonexistent.yaml")

    def test_non_yaml_extension_rejected(self, tmp_path: Path):
        f = tmp_path / "config.json"
        f.write_text("{}")
        with pytest.raises(ValueError, match="must be .yaml or .yml"):
            load_experiment_config(f)

    def test_non_mapping_yaml_rejected(self, tmp_path: Path):
        f = tmp_path / "bad.yaml"
        f.write_text("- a list\n- not a mapping\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_experiment_config(f)
