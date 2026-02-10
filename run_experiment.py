#!/usr/bin/env python
"""Single experiment entrypoint.

All experiments MUST be run through this script. No ad hoc training scripts.
See AGENTS.md for the full rule set.

Usage:
    python run_experiment.py --config configs/experiment.yaml

Steps (Section 5.3 of the implementation plan):
 1. Load and validate experiment config against Pydantic schema.
 2. Load reasoning metadata from configs/reasoning.yaml.
 3. Capture environment metadata (git commit hash, dirty flag, Python version).
 4. Compute dependency lock hash (SHA256 of uv.lock).
 5. Create an MLflow run tagged with client_name and project_name.
 6. Log all Tier 1 parameters.
 7. Log reasoning metadata as MLflow parameters.
 8. Execute model training (src/model.py).
 9. Log metrics to MLflow.
10. Log artifacts to MLflow.
11. Close the MLflow run cleanly, printing the run ID to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import mlflow
import yaml

from src.config_schema import ExperimentConfig, load_experiment_config
from src.data_access import load_shadow_dataset
from src.model import train_and_evaluate

# ---------------------------------------------------------------------------
# MLflow backend (Section 5.4: local SQLite)
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
REASONING_CONFIG_PATH = Path("configs/reasoning.yaml")
UV_LOCK_PATH = Path("uv.lock")


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
def _get_git_info() -> dict[str, str]:
    """Return git commit hash and dirty flag."""
    info: dict[str, str] = {}
    try:
        commit_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        info["git_commit_hash"] = commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["git_commit_hash"] = "unknown"

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        info["git_dirty"] = str(bool(status))
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["git_dirty"] = "unknown"

    return info


def _compute_file_hash(path: Path) -> str:
    """SHA256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_reasoning_metadata(path: Path) -> dict[str, str | None]:
    """Load reasoning.yaml and return its fields as a flat dict."""
    if not path.exists():
        print(f"WARNING: Reasoning config not found at {path}. "
              "Logging empty reasoning metadata.", file=sys.stderr)
        return {}
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        print(f"WARNING: Reasoning config at {path} is not a YAML mapping. "
              "Logging empty reasoning metadata.", file=sys.stderr)
        return {}
    return raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an experiment. All experiments go through this script."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the experiment YAML config (e.g. configs/experiment.yaml)",
    )
    args = parser.parse_args()

    # ---- 1. Load and validate experiment config --------------------------
    print(f"Loading config: {args.config}")
    try:
        config: ExperimentConfig = load_experiment_config(args.config)
    except Exception as e:
        print(f"ERROR: Config validation failed:\n{e}", file=sys.stderr)
        sys.exit(1)
    print(f"Config validated: project={config.project_name}, "
          f"model={config.model_type}")

    # ---- 2. Load reasoning metadata --------------------------------------
    reasoning = _load_reasoning_metadata(REASONING_CONFIG_PATH)

    # ---- 3. Capture environment metadata ---------------------------------
    git_info = _get_git_info()
    python_version = sys.version

    # ---- 4. Compute dependency lock hash ---------------------------------
    if UV_LOCK_PATH.exists():
        dependency_lock_hash = _compute_file_hash(UV_LOCK_PATH)
    else:
        print("WARNING: uv.lock not found. Logging dependency_lock_hash "
              "as 'missing'.", file=sys.stderr)
        dependency_lock_hash = "missing"

    # ---- 5. Feature pipeline hash (placeholder for Phase 1) --------------
    feature_pipeline_hash = "placeholder__phase2_will_compute"

    # ---- 6. Load data via governed access layer --------------------------
    dataset = load_shadow_dataset(config.dataset_version)

    # ---- 7. Set up MLflow and create run ---------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.project_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # -- Tags ----------------------------------------------------------
        mlflow.set_tag("client_name", config.client_name)
        mlflow.set_tag("project_name", config.project_name)

        # -- Tier 1 parameters (mandatory per run) -------------------------
        # Git
        mlflow.log_param("git_commit_hash", git_info["git_commit_hash"])
        mlflow.log_param("git_dirty", git_info["git_dirty"])
        mlflow.log_param("python_version", python_version)

        # Data
        mlflow.log_param("dataset_version", dataset.content_hash)
        mlflow.log_param("dataset_schema_hash", dataset.schema_hash)

        # Dependencies
        mlflow.log_param("dependency_lock_hash", dependency_lock_hash)

        # Feature pipeline
        mlflow.log_param("feature_pipeline_hash", feature_pipeline_hash)

        # Model config
        mlflow.log_param("model_type", config.model_type)
        mlflow.log_param("random_seed", config.random_seed)
        for hp_name, hp_value in config.hyperparameters.items():
            mlflow.log_param(f"hp__{hp_name}", hp_value)

        # Evaluation config
        mlflow.log_param("evaluation_metric", config.evaluation.metric)
        mlflow.log_param("evaluation_split_strategy",
                         config.evaluation.split_strategy)

        # -- Reasoning metadata (Section 5.3 step 7) ----------------------
        for key, value in reasoning.items():
            mlflow.log_param(f"reasoning__{key}", value)

        # -- 8. Execute model training -------------------------------------
        print("Training model...")
        metrics = train_and_evaluate(config, dataset)

        # -- 9. Log metrics ------------------------------------------------
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        print(f"Metrics: { {k: round(v, 4) for k, v in metrics.items()} }")

        # -- 10. Log artifacts ---------------------------------------------
        # Log the experiment config itself as an artifact for reproducibility
        mlflow.log_artifact(args.config)
        if REASONING_CONFIG_PATH.exists():
            mlflow.log_artifact(str(REASONING_CONFIG_PATH))

    # ---- 11. Print run ID to stdout --------------------------------------
    print(f"\nMLflow run completed successfully.")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()
