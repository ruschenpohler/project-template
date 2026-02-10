#!/usr/bin/env python
"""Verify that the MLflow local SQLite backend is configured correctly.

Run from the project root:
    uv run python scripts/verify_mlflow_setup.py

This script:
  1. Confirms MLFLOW_TRACKING_URI points to the local SQLite database.
  2. Connects to the backend and lists experiments.
  3. Checks that mlflow.db exists (created on first experiment run).
  4. Prints instructions for launching the MLflow UI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow

TRACKING_URI = "sqlite:///mlflow.db"
DB_PATH = Path("mlflow.db")


def main() -> None:
    print("=" * 60)
    print("  MLflow Setup Verification")
    print("=" * 60)

    # ---- 1. Set tracking URI ---------------------------------------------
    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"\n[1] Tracking URI: {TRACKING_URI}")
    print("    (relative path — resolves to mlflow.db in the current directory)")

    # ---- 2. Check database file ------------------------------------------
    if DB_PATH.exists():
        size_kb = DB_PATH.stat().st_size / 1024
        print(f"\n[2] Database file: {DB_PATH.resolve()}")
        print(f"    Size: {size_kb:.1f} KB")
    else:
        print(f"\n[2] Database file: NOT FOUND")
        print("    This is normal if you haven't run an experiment yet.")
        print("    Run your first experiment to create it:")
        print("      uv run python run_experiment.py --config configs/experiment.yaml")

    # ---- 3. List experiments ---------------------------------------------
    print(f"\n[3] Experiments in MLflow:")
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        if not experiments:
            print("    (none — run your first experiment)")
        for exp in experiments:
            runs = client.search_runs(exp.experiment_id)
            print(f"    - {exp.name} (id={exp.experiment_id}, "
                  f"runs={len(runs)})")
            for run in runs:
                status = run.info.status
                run_id = run.info.run_id[:12]
                model = run.data.params.get("model_type", "?")
                metric = run.data.metrics.get("roc_auc", None)
                metric_str = f"roc_auc={metric:.4f}" if metric is not None else "no metrics"
                print(f"      {run_id}...  {status}  {model}  {metric_str}")
    except Exception as e:
        print(f"    ERROR connecting to MLflow: {e}")
        sys.exit(1)

    # ---- 4. .gitignore check ---------------------------------------------
    print(f"\n[4] .gitignore check:")
    gitignore = Path(".gitignore")
    if gitignore.exists():
        content = gitignore.read_text()
        for entry in ["mlflow.db", "mlruns/", "mlartifacts/"]:
            if entry in content:
                print(f"    {entry:15s}  OK (gitignored)")
            else:
                print(f"    {entry:15s}  MISSING — add to .gitignore!")
    else:
        print("    WARNING: .gitignore not found")

    # ---- 5. Instructions -------------------------------------------------
    print(f"\n[5] To launch the MLflow UI:")
    print(f"    uv run mlflow ui --backend-store-uri {TRACKING_URI}")
    print(f"    Then open http://127.0.0.1:5000 in your browser.")
    print()
    print("=" * 60)
    print("  Setup OK")
    print("=" * 60)


if __name__ == "__main__":
    main()
