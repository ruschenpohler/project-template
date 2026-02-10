"""Smoke test for run_experiment.py (Section 5.6).

Covers:
  - run_experiment.py completes end-to-end with a toy config and creates an
    MLflow run.
  - All Tier 1 fields are present in the MLflow run, including dependency
    lock hash.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import mlflow
import pytest

# ---------------------------------------------------------------------------
# Tier 1 parameter keys that must be present on every run (Section 5.3 / 5.4)
# ---------------------------------------------------------------------------
TIER_1_PARAMS = [
    "git_commit_hash",
    "git_dirty",
    "python_version",
    "dataset_version",
    "dataset_schema_hash",
    "dependency_lock_hash",
    "feature_pipeline_hash",
    "model_type",
    "random_seed",
    "evaluation_metric",
    "evaluation_split_strategy",
]

TIER_1_TAGS = [
    "client_name",
    "project_name",
]

EXPECTED_METRICS = [
    "roc_auc",
    "f1",
    "precision",
    "recall",
    "log_loss",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def experiment_run(tmp_path: Path):
    """Run a full experiment in an isolated temp directory and return the
    MLflow run ID and tracking URI for verification.

    Uses tmp_path as the working directory so that mlflow.db, mlruns/, and
    config files don't pollute the real project.
    """
    # -- Write a valid experiment config -----------------------------------
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    experiment_yaml = config_dir / "experiment.yaml"
    experiment_yaml.write_text(textwrap.dedent("""\
        project_name: smoke_test
        client_name: test_client
        dataset_version: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        model_type: xgboost
        random_seed: 42
        hyperparameters:
          max_depth: 3
          learning_rate: 0.1
        evaluation:
          metric: roc_auc
          split_strategy: stratified_kfold
        protected:
          metric_definition: roc_auc
          split_strategy: stratified_kfold
    """))

    reasoning_yaml = config_dir / "reasoning.yaml"
    reasoning_yaml.write_text(textwrap.dedent("""\
        hypothesis_category: hyperparameter_tuning
        change_description: "Smoke test run"
        expected_effect: explore
        outcome: null
        metric_delta: null
    """))

    # -- Write a dummy uv.lock so the dependency hash is computed ----------
    uv_lock = tmp_path / "uv.lock"
    uv_lock.write_text("fake-lock-content-for-testing\n")

    # -- Copy run_experiment.py into the temp dir so it can be invoked -----
    # We import from the real src/ package, so PYTHONPATH must include the
    # project root.
    project_root = Path(__file__).resolve().parent.parent
    run_script = project_root / "run_experiment.py"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    result = subprocess.run(
        [sys.executable, str(run_script),
         "--config", str(experiment_yaml)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
        timeout=120,
    )

    # -- Parse run ID from stdout ------------------------------------------
    assert result.returncode == 0, (
        f"run_experiment.py failed.\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )

    match = re.search(r"Run ID:\s+(\S+)", result.stdout)
    assert match, (
        f"Could not find 'Run ID: ...' in stdout.\n"
        f"--- stdout ---\n{result.stdout}"
    )
    run_id = match.group(1)

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    return run_id, tracking_uri


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestEntrypointSmoke:
    """Smoke test: the entrypoint completes and creates an MLflow run."""

    def test_run_completes_and_creates_mlflow_run(self, experiment_run):
        run_id, tracking_uri = experiment_run
        mlflow.set_tracking_uri(tracking_uri)
        run = mlflow.get_run(run_id)
        assert run is not None
        assert run.info.status == "FINISHED"

    def test_tier_1_params_present(self, experiment_run):
        """Every Tier 1 parameter key must exist on the run."""
        run_id, tracking_uri = experiment_run
        mlflow.set_tracking_uri(tracking_uri)
        run = mlflow.get_run(run_id)
        params = run.data.params

        missing = [k for k in TIER_1_PARAMS if k not in params]
        assert not missing, (
            f"Tier 1 parameters missing from MLflow run: {missing}\n"
            f"Logged params: {sorted(params.keys())}"
        )

    def test_tier_1_tags_present(self, experiment_run):
        """client_name and project_name must be set as tags."""
        run_id, tracking_uri = experiment_run
        mlflow.set_tracking_uri(tracking_uri)
        run = mlflow.get_run(run_id)
        tags = run.data.tags

        missing = [k for k in TIER_1_TAGS if k not in tags]
        assert not missing, (
            f"Tier 1 tags missing from MLflow run: {missing}\n"
            f"Logged tags: {sorted(k for k in tags if not k.startswith('mlflow.'))}"
        )
        assert tags["client_name"] == "test_client"
        assert tags["project_name"] == "smoke_test"

    def test_metrics_logged(self, experiment_run):
        """All evaluation metrics must be logged."""
        run_id, tracking_uri = experiment_run
        mlflow.set_tracking_uri(tracking_uri)
        run = mlflow.get_run(run_id)
        metrics = run.data.metrics

        missing = [k for k in EXPECTED_METRICS if k not in metrics]
        assert not missing, (
            f"Expected metrics missing from MLflow run: {missing}\n"
            f"Logged metrics: {sorted(metrics.keys())}"
        )

    def test_dependency_lock_hash_is_real_sha256(self, experiment_run):
        """dependency_lock_hash must be a 64-char hex string (SHA256)."""
        run_id, tracking_uri = experiment_run
        mlflow.set_tracking_uri(tracking_uri)
        run = mlflow.get_run(run_id)
        dep_hash = run.data.params["dependency_lock_hash"]
        assert re.match(r"^[0-9a-f]{64}$", dep_hash), (
            f"dependency_lock_hash is not a valid SHA256: '{dep_hash}'"
        )

    def test_hyperparameters_logged(self, experiment_run):
        """Hyperparameters from config must appear with hp__ prefix."""
        run_id, tracking_uri = experiment_run
        mlflow.set_tracking_uri(tracking_uri)
        run = mlflow.get_run(run_id)
        params = run.data.params

        assert "hp__max_depth" in params
        assert "hp__learning_rate" in params
        assert params["hp__max_depth"] == "3"
        assert params["hp__learning_rate"] == "0.1"

    def test_reasoning_metadata_logged(self, experiment_run):
        """Reasoning fields must appear with reasoning__ prefix."""
        run_id, tracking_uri = experiment_run
        mlflow.set_tracking_uri(tracking_uri)
        run = mlflow.get_run(run_id)
        params = run.data.params

        assert params.get("reasoning__hypothesis_category") == "hyperparameter_tuning"
        assert params.get("reasoning__change_description") == "Smoke test run"
        assert params.get("reasoning__expected_effect") == "explore"

    def test_artifacts_uploaded(self, experiment_run):
        """Config files must be uploaded as artifacts."""
        run_id, tracking_uri = experiment_run
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        artifacts = [a.path for a in client.list_artifacts(run_id)]
        assert "experiment.yaml" in artifacts
        assert "reasoning.yaml" in artifacts
